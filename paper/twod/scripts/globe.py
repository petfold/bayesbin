"""The globe: binning each dataset's counts over the H3 tree, on all its data.

Writes data/globe_<name>.json (the numbers the paper quotes) and build/globe_<name>.npz
(per-cell arrays for the maps)."""

import sys
import time

import h3
import numpy as np

from common import BUILD, DATASETS, PLACES, RES, date, load_events, span, write_json
from h3tree import ALL, EARTH, Tree, finest, logL, merge, region_area

GAMMA_DISPLAY = 0.1  # the per-bin factor of the merged maps (a bonus of -log 0.1 = 2.3 nats per merge)


def run(name):
    ev = load_events(name)
    t0, t1, days = span(ev)
    counts = {}
    for _, c, n in ev:
        counts[c] = counts.get(c, 0.0) + n
    total = sum(counts.values())
    rate0 = total / (EARTH * days)

    tic = time.perf_counter()
    tree = Tree(set(counts))
    t_tree = time.perf_counter() - tic
    Y, E, eE = tree.totals(counts, days)

    tic = time.perf_counter()
    a, b, rho, logz = tree.search_prior(Y, E, eE, rate0)
    t_search = time.perf_counter() - tic
    reps = 20
    tic = time.perf_counter()
    for _ in range(reps):
        tree.logZ(Y, E, eE, a, b, rho)
    t_pass = (time.perf_counter() - tic) / reps

    # the best partition
    _, _, cut, _ = tree.logZ(Y, E, eE, a, b, rho, maximise=True)
    best = tree.best_bins(cut)
    bins = finest(best)
    bY = np.array([sum(counts.get(c, 0.0) for c in bn) for bn in bins])
    bE = np.array([region_area(c) for c in best]) * days
    bres = np.array([-1 if c == "world" else h3.get_resolution(c) for c in best])
    by_res = {str(r): {"bins": int((bres == r).sum()), "area_share": float(bE[bres == r].sum() / (EARTH * days))}
              for r in range(-1, RES + 1) if (bres == r).any()}

    # the posterior: expected bins per resolution, and every finest cell's posterior mean rate
    pb, pe = tree.posterior(Y, E, eE, a, b, rho)
    eres = tree.res[tree.ep] + 1
    expected = {str(r): float(pb[tree.res == r].sum() + pe[eres == r].sum()) for r in range(-1, RES + 1)}
    rate_node, rate_empty = (Y + a) / (E + b), a / (eE + b)
    empty_index = {c: k for k, c in enumerate(tree.ecell)}
    pm, cover = np.zeros(len(ALL)), np.zeros(len(ALL))
    for i, c in enumerate(ALL):
        s, w = pb[0] * rate_node[0], pb[0]
        for r in range(0, RES + 1):
            anc = h3.cell_to_parent(c, r) if r < RES else c
            j = tree.index.get(anc)
            if j is not None:
                s += pb[j] * rate_node[j]
                w += pb[j]
            else:
                k = empty_index[anc]
                s += pe[k] * rate_empty[k]
                w += pe[k]
                break
        pm[i], cover[i] = s / w, w

    # greedy merging of the best partition's bins (for display)
    tic = time.perf_counter()
    regions = merge(bins, bY, bE, a, b, GAMMA_DISPLAY)
    t_merge = time.perf_counter() - tic
    rY = np.array([sum(counts.get(c, 0.0) for c in rg) for rg in regions])
    rE = np.array([sum(region_area(c) for c in rg) for rg in regions])
    empty = rY == 0

    # places: the best partition's bin around each
    at = {c: k for k, c in enumerate(best)}
    places = {}
    for p, (lat, lng) in PLACES.items():
        for r in range(0, RES + 1):
            c = h3.latlng_to_cell(lat, lng, r)
            if c in at:
                k = at[c]
                places[p] = {"res": r, "area_km2": float(bE[k] / days), "events": float(bY[k])}
                break

    # per-cell arrays for the maps
    index = {c: i for i, c in enumerate(ALL)}
    tb, tarea = np.zeros(len(ALL), int), np.zeros(len(ALL))
    for k, bn in enumerate(bins):
        for c in bn:
            tb[index[c]], tarea[index[c]] = k, bE[k] / days
    mb, marea = np.zeros(len(ALL), int), np.zeros(len(ALL))
    for k, rg in enumerate(regions):
        for c in rg:
            mb[index[c]], marea[index[c]] = k, rE[k]
    BUILD.mkdir(exist_ok=True)
    np.savez_compressed(BUILD / f"globe_{name}.npz", tree_bin=tb, tree_area=tarea, merged_bin=mb,
                        merged_area=marea, pm_rate=pm * 1e4)  # rates per 10,000 km² per day

    write_json(f"globe_{name}.json", {
        "dataset": DATASETS[name]["label"], "events": total, "cells_with_events": len(counts),
        "cells": len(ALL), "res": RES, "days": days, "first": date(t0), "last": date(t1),
        "nodes": len(tree.cells), "empty_children": len(tree.ecell),
        "prior": {"alpha": a, "beta": b, "rho": rho, "mean_vs_global": (a / b) / rate0, "log_evidence": logz},
        "best_partition": {"bins": len(best), "empty_bins": int((bY == 0).sum()), "by_res": by_res,
                           "rate_min": float(((bY + a) / (bE + b)).min() * 1e4),
                           "rate_max": float(((bY + a) / (bE + b)).max() * 1e4)},
        "posterior_expected_bins": expected, "coverage_error": float(np.abs(cover - 1).max()),
        "merged": {"gamma": GAMMA_DISPLAY, "regions": len(regions), "empty_regions": int(empty.sum()),
                   "empty_share": float(rE[empty].sum() / EARTH), "largest_km2": float(rE.max()),
                   "largest_events": float(rY[np.argmax(rE)])},
        "places": places,
        "seconds": {"tree": t_tree, "pass": t_pass, "prior_search": t_search,
                    "prior_settings": 7 * 6 * 8, "merge": t_merge},
    })


if __name__ == "__main__":
    for name in sys.argv[1:] or list(DATASETS):
        run(name)
