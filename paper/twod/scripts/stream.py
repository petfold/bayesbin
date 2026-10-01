"""Streaming over the tree: Worldwatch's replay of its Layer-0 count model pooled over the H3 tree,
summarised for the paper -> data/stream.json.

The replay is Worldwatch's research/replay_changepoint/tree_layer0.py: the USGS catalogue in
5-minute windows, every node of the H3 tree above the resolution-3 cells with events running
Worldwatch's count model (a discounted Gamma-Poisson with a grid of burst factors) on its region's
counts, the nodes' log predictive scores summed with a forgetting time, the tree recursion after
every window, every cell with events scored. It saves its per-cell tallies to tree_layer0.npz in
WW_RESEARCH (default ~/.cache/worldwatch-research); this script only reads them (seconds).
"""

import json
import os
from pathlib import Path

import numpy as np

from common import DATA

RESEARCH = Path(os.environ.get("WW_RESEARCH", Path.home() / ".cache" / "worldwatch-research"))
WARMUP = 2 * 288  # windows before scoring starts (tree_layer0.py's)
RHO = 0.1  # a node's prior probability of being one bin (tree_layer0.py's)
ALONE, RES2, TREE = "cell alone (Layer 0 today)", "the resolution-2 cell", "tree, memory 3 days"
CLASSES = [("busy", "300 or more", 300, None), ("medium", "31--299", 31, 299),
           ("sparse", "3--30", 3, 30), ("rare", "1--2", 1, 2)]


def main():
    path = RESEARCH / "tree_layer0.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found: run Worldwatch's research/replay_changepoint/tree_layer0.py first")
    R = np.load(path, allow_pickle=True)
    variants = [str(v) for v in R["variants"]]
    names, events = [str(x) for x in R["names"]], R["events"]
    T, width = int(R["T"]), int(R["width"])
    n = T - WARMUP
    days = n * width / 86400
    out = {"cells": len(names), "events": int(events.sum()), "windows": n, "days": days,
           "window_seconds": width, "rho": RHO, "variants": variants, "classes": {}}
    for key, label, lo, hi in CLASSES:
        sel = (events >= lo) & (events <= (hi if hi is not None else np.inf))
        rows = {}
        for v, name in enumerate(variants):
            h = R["hist"][v][sel].sum(axis=0)
            cdf = np.cumsum(h) / h.sum()
            tot = sel.sum() * n
            rows[name] = {
                "ks": float(np.max(np.abs(cdf - np.arange(1, 201) / 200))),
                "p999": float(R["q999"][v][sel].sum() / tot),
                "p99": float(R["q99"][v][sel].sum() / tot),
                "alarms_per_cell_day": float(R["alarms"][v][sel].sum() / days / sel.sum()),
                "log_score_per_cell_day": float((R["logp"][v][sel] - R["logp"][0][sel]).sum() / days / sel.sum()),
            }
        w = R["wbar"][sel].mean(axis=0)  # mean weight of the cell's bin at resolutions 0..3 (30-day memory)
        out["classes"][key] = {"label": label, "cells": int(sel.sum()), "events": int(events[sel].sum()),
                               "variants": rows, "weight_by_res": [float(x) for x in w]}
    out["log_score_gain"] = {name: float((R["logp"][v] - R["logp"][0]).sum()) for v, name in enumerate(variants)}
    out["alarms_per_day"] = {name: float(R["alarms"][v].sum() / days) for v, name in enumerate(variants)}
    keep = dict(zip([str(x) for x in R["keep_names"]], R["keep"], strict=True))
    big = [(int(w), str(c), float(mg)) for w, c, mg in R["big"] if int(w) >= WARMUP]
    out["big"] = {"quakes": len(big), "min_mag": 5.0,
                  "in_window": {name: int(sum(keep[c][v, w] >= 0.999 for w, c, _ in big if c in keep))
                                for v, name in enumerate(variants)},
                  "within_hour": {name: int(sum(np.nanmax(keep[c][v, w:w + 3600 // width]) >= 0.999
                                                for w, c, _ in big if c in keep))
                                  for v, name in enumerate(variants)}}
    for k in (ALONE, RES2, TREE):
        assert k in variants, k
    with open(DATA / "stream.json", "w") as f:
        json.dump(out, f, indent=1)
    c = out["classes"]
    print(f"{out['cells']} cells, {days:.0f} days: P(q > 0.999) in sparse cells {c['sparse']['variants'][ALONE]['p999']:.5f} "
          f"alone, {c['sparse']['variants'][TREE]['p999']:.5f} pooled; log score {out['log_score_gain'][TREE]:+.0f} nats")


if __name__ == "__main__":
    main()
