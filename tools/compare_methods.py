#!/usr/bin/env python3
"""The numbers of the User Guide's comparison with other methods (section 3):

- the guide's example data: the chance excess of spikes after the burst;
- how often the true rate lies within bayesbin's ±1 sd and ±2 sd (100 datasets);
- RMS error against the true rate, 30 datasets per number of trials, for bayesbin,
  Shimazaki & Shinomoto's histogram (2007) and Gaussian-kernel (2010) choices of
  width from the data, and the best fixed histogram and kernel chosen knowing the
  truth (edges averaged over every offset: no aligning them with the steps).

    tools/compare_methods.py          # a few minutes
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson

from bayesbin import BernoulliModel, fit, spike_counts

t = np.arange(-100, 500)
T = t.size
p_true = np.select([t < 80, t < 130, t < 380], [0.01, 0.08, 0.05], 0.01)


def data(seed: int, n: int):
    rng = np.random.default_rng(seed)
    return spike_counts([t[rng.random(T) < p_true] for _ in range(n)], -100, 499)


def rms(est):
    return float(np.sqrt(np.mean((est - p_true) ** 2)))


def histogram(s, n, w, offset=0):
    edges = np.unique(np.clip(np.concatenate([[0], np.arange(offset, T, w), [T]]), 0, T))
    est = np.empty(T)
    for a, b in zip(edges[:-1], edges[1:]):
        est[a:b] = s[a:b].sum() / (n * (b - a))
    return est


def ss_histogram(s, n):
    """Shimazaki & Shinomoto (2007): the width minimising (2 mean - var) / (n w)² of the bin
    counts, averaged over bin-edge offsets."""
    best = None
    for w in range(1, 151):
        costs = []
        for o in range(0, w, max(1, w // 10)):
            m = (T - o) // w
            if m < 2:
                continue
            k = s[o:o + m * w].reshape(m, w).sum(axis=1)
            costs.append((2 * k.mean() - k.var()) / (n * w) ** 2)
        if costs and (best is None or np.mean(costs) < best[0]):
            best = (np.mean(costs), w)
    return histogram(s, n, best[1])


def ss_kernel(s, n, bandwidths=(1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 80)):
    """Shimazaki & Shinomoto (2010), Gaussian kernel: the bandwidth minimising
    ∫ rate² dt - 2 Σ_{i≠j} k(t_i - t_j) / n² over the pooled spike times."""
    times = np.repeat(np.arange(T), s.astype(int)).astype(float)
    d = times[:, None] - times[None, :]
    best = None
    for b in bandwidths:
        k2 = np.exp(-d**2 / (4 * b * b)) / np.sqrt(4 * np.pi * b * b)
        k1 = np.exp(-d**2 / (2 * b * b)) / np.sqrt(2 * np.pi * b * b)
        c = (k2.sum() - 2 * (k1.sum() - np.trace(k1))) / n**2
        if best is None or c < best[0]:
            best = (c, b)
    return gaussian_filter1d(s / n, best[1], mode="nearest")


def main() -> None:
    s, g = data(1, 30)
    w = (t >= 130) & (t < 160)
    obs, exp = s[w].sum(), 30 * p_true[w].sum()
    print(f"guide data, 130-159 ms: {obs:.0f} spikes, {exp:.0f} expected; P(>= {obs:.0f}) = "
          f"{poisson.sf(obs - 1, exp):.4f} ({(obs - exp) / np.sqrt(exp):.1f} sd)")
    in1, in2 = [], []
    for seed in range(100, 200):
        s, g = data(seed, 30)
        r = fit(BernoulliModel(s, g), 20)
        z = np.abs(r.rate - p_true) / r.rate_std
        in1.append(np.mean(z < 1))
        in2.append(np.mean(z < 2))
    print(f"true rate within ±1 sd: {np.mean(in1):.0%} of time points, ±2 sd: {np.mean(in2):.0%}")
    widths = [2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    bws = [2, 3, 5, 7, 10, 15, 20, 30, 50]
    print("RMS error against the true rate (and as a multiple of bayesbin's):")
    print(f"{'trials':>6} {'bayesbin':>9} {'S&S hist':>17} {'S&S kernel':>17} {'best hist':>17} {'best kernel':>17}")
    for n in (2, 5, 10, 30, 100):
        res = {k: [] for k in ("bb", "ssh", "ssk")}
        hist = {w: [] for w in widths}
        kern = {b: [] for b in bws}
        for seed in range(30):
            s, g = data(1000 + seed, n)
            res["bb"].append(rms(fit(BernoulliModel(s, g), 20).rate))
            res["ssh"].append(rms(ss_histogram(s, n)))
            if n <= 30:  # the pairwise kernel cost is O(spikes²)
                res["ssk"].append(rms(ss_kernel(s, n)))
            for w in widths:
                hist[w].append(np.mean([rms(histogram(s, n, w, o)) for o in range(w)]))
            for b in bws:
                kern[b].append(rms(gaussian_filter1d(s / n, b, mode="nearest")))
        B = np.mean(res["bb"])
        both = lambda e: f"{e:.4f} ({e / B:.2f}x)"
        cols = [both(np.mean(res["ssh"])), both(np.mean(res["ssk"])) if res["ssk"] else "-",
                both(min(np.mean(v) for v in hist.values())), both(min(np.mean(v) for v in kern.values()))]
        print(f"{n:>6} {B:9.4f} " + " ".join(f"{c:>17}" for c in cols))


if __name__ == "__main__":
    main()
