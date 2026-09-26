#!/usr/bin/env python3
"""Seeded Python 3 port of binsdfc's testdata.py: 30 trials of a cell with
background 0.01/ms, a transient of 0.08/ms from 80 ms for 50 ms, then 0.05/ms
for 250 ms. Writes binsdfc's input format to stdout:
    <stimulus name> <spike time 1> ... <spike time N>

Derived from testdata.py in binsdfc 0.1, Copyright (C) 2006-2007 Dominik Endres;
like it, this file is under the GNU General Public License, version 2 or (at
your option) any later version (see cpp/COPYING). The rest of tools/ and the
Python package are BSD-3-Clause.
"""

import sys

import numpy as np

NUMTRIALS, BACKGROUND, TRANSIENT, SUSTAINED = 30, 0.01, 0.08, 0.05
LATENCY, TRANSIENT_DURATION, SUSTAINED_DURATION = 80, 50, 250


def rate(t: int) -> float:
    if t < LATENCY:
        return BACKGROUND
    if t < LATENCY + TRANSIENT_DURATION:
        return TRANSIENT
    if t < LATENCY + TRANSIENT_DURATION + SUSTAINED_DURATION:
        return SUSTAINED
    return BACKGROUND


def main(seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    times = np.arange(-100, 500)
    p = np.array([rate(t) for t in times])
    for _ in range(NUMTRIALS):
        spikes = times[rng.random(len(times)) < p]
        print("Stimulus0", *spikes.tolist())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
