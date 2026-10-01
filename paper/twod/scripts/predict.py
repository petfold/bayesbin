"""Held-out prediction: fit on the first half of each dataset's time span, score the second.

The score is log P(test counts | training counts), the joint predictive of every finest cell's
test count; the data-only factors, the same for every method, are left out. For a fixed
partition it is Σ_bins [log L(train + test) − log L(train)]; for the tree, averaged over its
partitions, log Z(train + test) − log Z(train). Each method's prior is chosen by its own
training evidence. Writes data/heldout.csv and data/heldout_<name>.json."""

import csv
import sys

import numpy as np

from common import DATA, DATASETS, RES, load_events, write_json
from h3tree import ALL, ALPHAS, EARTH, SCALES, Tree, bin_totals, finest, grid_bins, logL, merge

MERGE_GAMMAS = [0.1, 1.0, 10.0]


def fixed_score(bins, tr, al, days_tr, days_all, rate0, prior=None):
    Ytr, Etr = bin_totals(bins, tr, days_tr)
    Yal, Eal = bin_totals(bins, al, days_all)
    if prior is None:  # the partition's own training evidence picks alpha and the prior mean
        best = max((logL(Ytr, Etr, a, a / rate0 * s).sum(), a, a / rate0 * s) for a in ALPHAS for s in SCALES)
        prior = best[1:]
    a, b = prior
    return float((logL(Yal, Eal, a, b) - logL(Ytr, Etr, a, b)).sum())


def run(name):
    ev = load_events(name)
    t = np.array([e[0] for e in ev])
    t0, t1 = t.min(), t.max()
    mid = (t0 + t1) / 2
    days_tr, days_all = (mid - t0) / 86400, (t1 - t0) / 86400
    tr, al = {}, {}
    for ts, c, n in ev:
        al[c] = al.get(c, 0.0) + n
        if ts < mid:
            tr[c] = tr.get(c, 0.0) + n
    n_train = sum(tr.values())
    n_test = sum(al.values()) - n_train
    rate0 = n_train / (EARTH * days_tr)
    rows = [("world", "one rate for the whole Earth", 1, fixed_score([ALL], tr, al, days_tr, days_all, rate0))]
    for r in range(0, RES + 1):
        bins = grid_bins(r)
        rows.append((f"grid{r}", f"fixed H3 grid, resolution {r}", len(bins),
                     fixed_score(bins, tr, al, days_tr, days_all, rate0)))
    tree = Tree(set(al))  # the union's cells: the same tree for both fits
    Ytr, Etr, eEtr = tree.totals(tr, days_tr)
    Yal, Eal, eEal = tree.totals(al, days_all)
    a, b, rho, _ = tree.search_prior(Ytr, Etr, eEtr, rate0)
    zt = tree.logZ(Ytr, Etr, eEtr, a, b, rho)[0]
    za = tree.logZ(Yal, Eal, eEal, a, b, rho)[0]
    rows.append(("tree", "tree, averaged over its partitions", None, float(za - zt)))
    best = finest(tree.best_bins(tree.logZ(Ytr, Etr, eEtr, a, b, rho, maximise=True)[2]))
    rows.append(("best", "tree, best partition only", len(best),
                 fixed_score(best, tr, al, days_tr, days_all, rate0, (a, b))))
    bY, bE = bin_totals(best, tr, days_tr)
    for g in MERGE_GAMMAS:
        merged = merge(best, bY, bE, a, b, g)
        rows.append((f"merged{g:g}", f"best partition, then greedy merging (bonus {-np.log(g):+.1f} nats per merge)",
                     len(merged), fixed_score(merged, tr, al, days_tr, days_all, rate0, (a, b))))
    top = max(r[3] for r in rows)
    print(f"{name}: train {days_tr:.1f} days, {n_train:.0f} events; test {days_all - days_tr:.1f} days, {n_test:.0f}")
    for key, label, nb, v in rows:
        print(f"  {label:<62} {'' if nb is None else nb:>7} {v:>14,.0f} {(v - top) / n_test:+8.3f} per event")
    write_json(f"heldout_{name}.json", {"days_train": days_tr, "days_test": days_all - days_tr,
                                         "events_train": n_train, "events_test": n_test,
                                         "prior": {"alpha": a, "beta": b, "rho": rho}})
    return [(name, key, label, "" if nb is None else nb, v, v - top, (v - top) / n_test) for key, label, nb, v in rows]


if __name__ == "__main__":
    out = []
    for name in sys.argv[1:] or list(DATASETS):
        out += run(name)
    with open(DATA / "heldout.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "key", "method", "bins", "logp", "vs_best", "per_event"])
        for row in out:
            w.writerow(row[:4] + tuple(f"{x:.6g}" for x in row[4:]))
    print("wrote data/heldout.csv")
