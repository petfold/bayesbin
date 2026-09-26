#!/usr/bin/env python3
"""Time bayesbin against the original binsdfc on the same data and outputs.

    tools/bench_vs_binsdfc.py BINSDFC [BINSDFC ...] [--long LONG_DATA]

Each case is timed 3 times; the median is shown. binsdfc is timed as a whole
process (reading its input included); bayesbin in-process (numpy/scipy import
excluded, reading the input included); with the fused kernels if numba is
installed (BAYESBIN_NUMBA=0 for the NumPy path; the first call compiles or
loads them, so a warm-up fit runs first).
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from bayesbin import BernoulliModel, fit, spike_counts
from bayesbin.core import _forward, _log_binom

ROOT = Path(__file__).resolve().parent.parent


def _median_time(fn, n=3):
    ts = []
    for _ in range(n):
        if ts and ts[0] > 30:  # a slow case: one run is enough
            break
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _load(path, t0, t1):
    trials = [list(map(int, line.split()[1:])) for line in open(path)]
    return spike_counts(trials, t0, t1)


def _evidence_only(path, t0, t1, m):
    s, g = _load(path, t0, t1)
    model = BernoulliModel(s, g)
    fwd = _forward(model, m)
    return [fwd[k, -1] - _log_binom(len(s) - 1, k) for k in range(m + 1)]


def _sdf(path, t0, t1, m, mass):
    s, g = _load(path, t0, t1)
    return fit(BernoulliModel(s, g), m, m_mass=mass)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("binsdfc", nargs="+")
    ap.add_argument("--long")
    ap.add_argument("--cap", type=float, default=600, help="seconds per binsdfc run")
    a = ap.parse_args()
    seed1 = ROOT / "tests" / "data" / "testdata_seed1.txt"
    cases = [  # label, data, t_start, t_end (inclusive), max M, what, credible mass
        ("T=600  M<=10 evidence", seed1, -100, 499, 10, "M", None),
        ("T=600  M<=10 rate+sd, best M", seed1, -100, 499, 10, "v", 0.0),
        ("T=600  M<=10 rate+sd, 90% M", seed1, -100, 499, 10, "v", 0.9),
    ]
    if a.long:
        cases += [
            ("T=2016 M<=10 evidence", a.long, 0, 2015, 10, "M", None),
            ("T=2016 M<=30 evidence", a.long, 0, 2015, 30, "M", None),
            ("T=2016 M<=30 rate+sd, best M", a.long, 0, 2015, 30, "v", 0.0),
            ("T=2016 M<=30 rate+sd, 90% M", a.long, 0, 2015, 30, "v", 0.9),
        ]
    _sdf(seed1, -100, -37, 2, None)  # warm-up: the fused kernels' compile or cache load
    names = [Path(b).name for b in a.binsdfc]
    print(f"{'case':32s}" + "".join(f"{n:>16s}" for n in names) + f"{'bayesbin':>12s}", flush=True)
    for label, data, t0, t1, m, what, mass in cases:
        row = []
        for b in a.binsdfc:
            args = [b, "-s", str(t0), "-e", str(t1 + 1), "-m", str(m)]
            args += ["-M", "-n"] if what == "M" else ["-v", "-l", str(mass)]
            try:
                row.append(_median_time(lambda: subprocess.run(
                    args + [str(data)], check=True, capture_output=True, timeout=a.cap)))
            except subprocess.TimeoutExpired:
                row.append(float("inf"))
        if what == "M":
            py = _median_time(lambda: _evidence_only(data, t0, t1, m))
        else:
            py = _median_time(lambda: _sdf(data, t0, t1, m, mass))
        cells = "".join(f"{'>' + str(int(a.cap)):>15s}s" if t == float("inf") else f"{t:15.2f}s" for t in row)
        print(f"{label:32s}" + cells + f"{py:11.2f}s", flush=True)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
