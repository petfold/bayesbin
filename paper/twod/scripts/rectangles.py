"""Rectangle partitions of an m × n grid: prototypes, checks, counts and timings.

- rows, then columns per band: bayesbin along each candidate band's column sums, then bayesbin's
  own programme over the rows on the table of band evidences; checked against enumeration;
- recursive (guillotine) cuts: an inside pass over all sub-rectangles (numba); checked against an
  independent memoised recursion;
- counts of the partition families on small grids, and of the cut trees that build them.

Writes data/rectangles.json and data/timings.csv. Poisson counts, Gamma(1, 1) per bin."""

import csv
import functools
import itertools
import math
import time
from collections import Counter

import numba
import numpy as np
from bayesbin import PoissonModel, fit
from bayesbin.core import _backward, _forward, _log_binom, bin_posterior
from scipy.special import gammaln, logsumexp

from common import DATA, QUICK, write_json

ALPHA, BETA, RHO = 1.0, 1.0, 0.6


# --- rows, then columns per band --------------------------------------------------------------

def table_fit(L, max_m):
    """bayesbin's programme on a T×T table of log bin evidences: log P(D | M) per M and
    W[a, b] = P(a..b is a bin | D)."""
    T = L.shape[0]
    max_m = min(max_m, T - 1)
    fwd, bwd = _forward(L, max_m), _backward(L, max_m)
    log_ev = np.array([fwd[m, T - 1] - _log_binom(T - 1, m) for m in range(max_m + 1)])
    post = np.exp(log_ev - logsumexp(log_ev))
    with np.errstate(divide="ignore"):
        log_c = np.log(post) - fwd[:, T - 1]
    return log_ev, bin_posterior(L, fwd, bwd, log_c)


def band_model(y, a, b):
    return PoissonModel(y[a:b + 1].sum(0), ALPHA, BETA, np.full(y.shape[1], float(b - a + 1)))


def band_log_marginal(model, max_m):
    """The band's marginal likelihood over its column partitions and M, without the band
    series' data-only factor (which differs between bands; the grid's is one constant)."""
    T = model.T
    max_m = min(max_m, T - 1)
    fwd = _forward(model, max_m)
    log_ev = np.array([fwd[m, T - 1] - _log_binom(T - 1, m) for m in range(max_m + 1)])
    return logsumexp(log_ev) - np.log(len(log_ev))


def two_level(y, max_r, max_c, w_min=1e-12):
    m, n = y.shape
    tic = time.perf_counter()
    Z = np.full((m, m), -np.inf)
    for a in range(m):
        for b in range(a, m):
            Z[a, b] = band_log_marginal(band_model(y, a, b), max_c)
    t_bands = time.perf_counter() - tic
    log_ev, W = table_fit(Z, max_r)
    tic = time.perf_counter()
    rate, ef2 = np.zeros((m, n)), np.zeros((m, n))
    used = 0
    for a, b in zip(*np.nonzero(W > w_min)):
        r = fit(band_model(y, a, b), max_boundaries=max_c)
        rate[a:b + 1] += W[a, b] * r.rate
        ef2[a:b + 1] += W[a, b] * (r.rate_std ** 2 + r.rate ** 2)
        used += 1
    cover = np.array([W[:k + 1, k:].sum() for k in range(m)])[:, None]
    return {"log_marginal": logsumexp(log_ev) - np.log(len(log_ev)), "m_map": int(np.argmax(log_ev)),
            "rate": rate / cover, "t_bands": t_bands, "t_rates": time.perf_counter() - tic,
            "bands": m * (m + 1) // 2, "refitted": used}


def log_rect(y, r0, r1, c0, c1):
    Y, E = y[r0:r1 + 1, c0:c1 + 1].sum(), (r1 - r0 + 1) * (c1 - c0 + 1)
    return (ALPHA * np.log(BETA) - gammaln(ALPHA) + gammaln(ALPHA + Y) - (ALPHA + Y) * np.log(BETA + E),
            (ALPHA + Y) / (BETA + E))


def segments(T, cuts):
    edges = [0, *[c + 1 for c in cuts], T]
    return [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]


def log_prior_1d(T, k, max_m):
    max_m = min(max_m, T - 1)
    return -np.inf if k > max_m else -np.log(max_m + 1) - _log_binom(T - 1, k)


def enumerate_two_level(y, max_r, max_c):
    m, n = y.shape
    subsets = lambda T: [s for k in range(T) for s in itertools.combinations(range(T - 1), k)]
    terms, rates = [], []
    for R in subsets(m):
        lpR = log_prior_1d(m, len(R), max_r)
        if lpR == -np.inf:
            continue
        bands = segments(m, R)
        for Cs in itertools.product(subsets(n), repeat=len(bands)):
            lp, ll, rate = lpR, 0.0, np.zeros((m, n))
            for (a, b), C in zip(bands, Cs):
                lp += log_prior_1d(n, len(C), max_c)
                for c0, c1 in segments(n, C):
                    le, mu = log_rect(y, a, b, c0, c1)
                    ll += le
                    rate[a:b + 1, c0:c1 + 1] = mu
            if lp > -np.inf:
                terms.append(lp + ll)
                rates.append(rate)
    terms = np.array(terms)
    w = np.exp(terms - logsumexp(terms))
    return float(logsumexp(terms)), np.tensordot(w, np.array(rates), axes=1), len(terms)


# --- recursive (guillotine) cuts ---------------------------------------------------------------

@numba.njit(cache=True)
def tri(a, b, T):  # packed index of the interval a..b of 0..T-1
    return a * T - a * (a - 1) // 2 + (b - a)


@numba.njit(parallel=True, cache=True)
def inside(S, m, n, alpha, beta, rho):
    k0 = alpha * math.log(beta) - math.lgamma(alpha)
    lr, l1r = math.log(rho), math.log(1.0 - rho)
    Z = np.full((m * (m + 1) // 2, n * (n + 1) // 2), -np.inf)
    for h in range(1, m + 1):
        for w in range(1, n + 1):
            for r0 in numba.prange(m - h + 1):
                r1 = r0 + h - 1
                p = tri(r0, r1, m)
                for c0 in range(n - w + 1):
                    c1 = c0 + w - 1
                    q = tri(c0, c1, n)
                    Y = S[r1 + 1, c1 + 1] - S[r0, c1 + 1] - S[r1 + 1, c0] + S[r0, c0]
                    leaf = k0 + math.lgamma(alpha + Y) - (alpha + Y) * math.log(beta + h * w)
                    if h == 1 and w == 1:
                        Z[p, q] = leaf
                        continue
                    nd = (h > 1) + (w > 1)
                    mx = -np.inf
                    for k in range(r0, r1):
                        mx = max(mx, Z[tri(r0, k, m), q] + Z[tri(k + 1, r1, m), q] - math.log(h - 1))
                    for k in range(c0, c1):
                        mx = max(mx, Z[p, tri(c0, k, n)] + Z[p, tri(k + 1, c1, n)] - math.log(w - 1))
                    s = 0.0
                    for k in range(r0, r1):
                        s += math.exp(Z[tri(r0, k, m), q] + Z[tri(k + 1, r1, m), q] - math.log(h - 1) - mx)
                    for k in range(c0, c1):
                        s += math.exp(Z[p, tri(c0, k, n)] + Z[p, tri(k + 1, c1, n)] - math.log(w - 1) - mx)
                    split = l1r - math.log(nd) + mx + math.log(s)
                    hi = max(lr + leaf, split)
                    Z[p, q] = hi + math.log(math.exp(lr + leaf - hi) + math.exp(split - hi))
    return Z[tri(0, m - 1, m), tri(0, n - 1, n)]


def guillotine(y):
    S = np.zeros((y.shape[0] + 1, y.shape[1] + 1))
    S[1:, 1:] = y.cumsum(0).cumsum(1)
    return inside(S, y.shape[0], y.shape[1], ALPHA, BETA, RHO)


def guillotine_reference(y):
    @functools.lru_cache(None)
    def Z(r0, r1, c0, c1):
        leaf = log_rect(y, r0, r1, c0, c1)[0]
        if r0 == r1 and c0 == c1:
            return leaf
        dirs = []
        if r1 > r0:
            dirs.append(logsumexp([Z(r0, k, c0, c1) + Z(k + 1, r1, c0, c1) for k in range(r0, r1)]) - np.log(r1 - r0))
        if c1 > c0:
            dirs.append(logsumexp([Z(r0, r1, c0, k) + Z(r0, r1, k + 1, c1) for k in range(c0, c1)]) - np.log(c1 - c0))
        return np.logaddexp(np.log(RHO) + leaf, np.log(1 - RHO) + logsumexp(dirs) - np.log(len(dirs)))
    return float(Z(0, y.shape[0] - 1, 0, y.shape[1] - 1))


# --- counting partitions and the trees that build them ------------------------------------------

def all_partitions(h, w):
    filled = np.zeros((h, w), bool)
    out, cur = [], []

    def rec():
        empty = np.argwhere(~filled)
        if len(empty) == 0:
            out.append(tuple(cur))
            return
        r, c = empty[0]
        cmax = c
        while cmax + 1 < w and not filled[r, cmax + 1]:
            cmax += 1
        for c1 in range(c, cmax + 1):
            for r1 in range(r, h):
                if filled[r1, c:c1 + 1].any():
                    break
                filled[r:r1 + 1, c:c1 + 1] = True
                cur.append((r, r1, c, c1))
                rec()
                cur.pop()
                filled[r:r1 + 1, c:c1 + 1] = False
    rec()
    return out


def is_guillotine(rects):
    if len(rects) == 1:
        return True
    r0, r1 = min(r[0] for r in rects), max(r[1] for r in rects)
    c0, c1 = min(r[2] for r in rects), max(r[3] for r in rects)
    for k in range(r0, r1):
        if all(r[1] <= k or r[0] > k for r in rects):
            return is_guillotine([r for r in rects if r[1] <= k]) and is_guillotine([r for r in rects if r[0] > k])
    for k in range(c0, c1):
        if all(r[3] <= k or r[2] > k for r in rects):
            return is_guillotine([r for r in rects if r[3] <= k]) and is_guillotine([r for r in rects if r[2] > k])
    return False


def binary_trees(h, w):
    @functools.lru_cache(None)
    def T(h, w):
        if h == w == 1:
            return 1
        return 1 + sum(T(k, w) * T(h - k, w) for k in range(1, h)) + sum(T(h, k) * T(h, w - k) for k in range(1, w))
    return T(h, w)


def multiway_trees(h, w):
    """Trees whose nodes take all their parallel cuts at once (a set of >= 2 strips), each strip
    one bin or cut the other way: (number of trees, multiplicity of each partition)."""
    @functools.lru_cache(None)
    def parts(r0, r1, c0, c1, orient):
        out = []
        if orient in (None, "leaf"):
            out.append(frozenset([(r0, r1, c0, c1)]))
        for horiz in ([True] if orient == "H" else [False] if orient == "V" else [] if orient == "leaf" else [True, False]):
            lo, hi = (r0, r1) if horiz else (c0, c1)
            res = []

            def rec(start, acc, k):
                if start > hi:
                    if k >= 2:
                        res.extend(acc)
                    return
                for end in range(start, hi + 1):
                    if end == hi and k == 0:
                        continue
                    sub = (start, end, c0, c1) if horiz else (r0, r1, start, end)
                    child = parts(*sub, "leaf") + parts(*sub, "V" if horiz else "H")
                    rec(end + 1, [x | y for x in acc for y in child], k + 1)
            rec(lo, [frozenset()], 0)
            out += res
        return out
    trees = parts(0, h - 1, 0, w - 1, None)
    return len(trees), Counter(trees)


def planted(m, n, rng):
    yy, xx = np.mgrid[0:m, 0:n] / max(m, n)
    lam = np.full((m, n), 1.0)
    lam[(yy - 0.35) ** 2 + (xx - 0.6) ** 2 < 0.18 ** 2] = 4.0  # a disc
    lam[(yy > 0.7) & (xx < 0.3)] = 0.2  # a corner rectangle
    return rng.poisson(lam)


def main():
    rng = np.random.default_rng(1)
    out = {"two_level_checks": [], "guillotine_checks": [], "counts": []}
    for shape, max_r, max_c in [((4, 4), 3, 3), ((4, 5), 2, 2), ((5, 4), 4, 1)]:
        y = rng.poisson(2.0, shape)
        y[: shape[0] // 2, : shape[1] // 2] += 4
        res = two_level(y, max_r, max_c)
        lm, rate, count = enumerate_two_level(y, max_r, max_c)
        out["two_level_checks"].append({"shape": list(shape), "max_r": max_r, "max_c": max_c, "partitions": count,
                                        "dp": res["log_marginal"], "enumeration": lm,
                                        "rate_error": float(np.abs(res["rate"] - rate).max())})
    for shape in [(3, 3), (4, 5), (6, 6)]:
        y = rng.poisson(2.0, shape)
        y[: shape[0] // 2, : shape[1] // 2] += 4
        out["guillotine_checks"].append({"shape": list(shape), "dp": float(guillotine(y)),
                                         "reference": guillotine_reference(y)})
    for h, w in ([(2, 2), (3, 3), (3, 4)] if QUICK else [(2, 2), (3, 3), (3, 4), (4, 4)]):
        ps = all_partitions(h, w)
        n_multi, mult = multiway_trees(h, w)
        singles = frozenset((i, i, j, j) for i in range(h) for j in range(w))
        out["counts"].append({"grid": f"{h}x{w}", "tilings": len(ps), "guillotine": sum(is_guillotine(list(p)) for p in ps),
                              "two_level": sum(math.comb(h - 1, k - 1) * (2 ** (w - 1)) ** k for k in range(1, h + 1)),
                              "product": 2 ** (h - 1) * 2 ** (w - 1), "binary_trees": binary_trees(h, w),
                              "multiway_trees": n_multi, "multiway_partitions": len(mult),
                              "singles_multiplicity": mult[singles]})
    rows = []
    for side in ([32, 64] if QUICK else [32, 64, 100, 128, 200]):
        y = planted(side, side, rng)
        res = two_level(y, 10, 10)
        rows.append({"side": side, "method": "two_level_evidence", "seconds": res["t_bands"], "bands": res["bands"],
                     "refitted": res["refitted"], "m_map": res["m_map"], "table_mb": ""})
        rows.append({"side": side, "method": "two_level_rates", "seconds": res["t_rates"], "bands": res["bands"],
                     "refitted": res["refitted"], "m_map": res["m_map"], "table_mb": ""})
    guillotine(planted(8, 8, rng))  # compile
    for side in ([32] if QUICK else [32, 64, 100]):
        y = planted(side, side, rng)
        tic = time.perf_counter()
        guillotine(y)
        rows.append({"side": side, "method": "recursive_evidence", "seconds": time.perf_counter() - tic, "bands": "",
                     "refitted": "", "m_map": "", "table_mb": (side * (side + 1) // 2) ** 2 * 8 / 1e6})
    tic = time.perf_counter()
    model = PoissonModel(rng.poisson(3.0, 4).astype(float), 1.0, 1.0, np.full(4, 3.0))
    for _ in range(200):
        band_log_marginal(model, 10)
    out["call_overhead_ms"] = (time.perf_counter() - tic) / 200 * 1e3
    with open(DATA / "timings.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote data/timings.csv")
    write_json("rectangles.json", out)


if __name__ == "__main__":
    main()
