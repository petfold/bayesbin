#!/usr/bin/env python3
"""Synthetic long series for timing: 30 trials of a daily profile in 5-minute
slots (288 a day: night, morning ramp, day, evening peak, late evening) with a
2-hour burst once a week, as spike trains in binsdfc's input format.

    tools/make_longdata.py T [SEED] > big$T.txt     # e.g. T = 2016 (a week)
"""

import sys

import numpy as np


def main(T: int, seed: int = 4) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    d = t % 288
    p = np.where(d < 72, 0.01, np.where(d < 96, 0.025, np.where(d < 228, 0.04,
                 np.where(d < 252, 0.07, 0.02))))
    week = (t // 2016) * 2016
    p = p + 0.1 * ((week + 1500 <= t) & (t < week + 1524))
    for _ in range(30):
        print("S", *t[rng.random(T) < p].tolist())


if __name__ == "__main__":
    main(int(sys.argv[1]), *(int(a) for a in sys.argv[2:3]))
