"""1-D: does a hierarchical prior predict better than bayesbin's own?

The evidence (the sequential predictive score) of hourly world earthquake counts under
bayesbin's prior (uniform M, uniform boundary positions), a free binary-cut tree (the same
segmentations, a prior over cut trees, O(T³)) and a dyadic tree (cuts only at midpoints, O(T)).
All three use bayesbin's weak Gamma prior per bin. Writes data/tree1d.json."""

import math
import time

import numba
import numpy as np
from bayesbin import PoissonModel, fit

from common import QUICK, load_events, write_json

RHOS = [0.1, 0.3, 0.5, 0.7, 0.9]


@numba.njit(parallel=True, cache=True)
def binary_tree(S, T, alpha, beta, lr, l1r):
    """log Z of 0..T-1 under the binary-cut tree prior: every interval is one bin with
    probability rho, or is cut at a uniform position into two, each binned the same way."""
    k0 = alpha * math.log(beta) - math.lgamma(alpha)
    Z = np.full((T, T), -np.inf)
    for a in range(T):
        Y = S[a + 1] - S[a]
        Z[a, a] = k0 + math.lgamma(alpha + Y) - (alpha + Y) * math.log(beta + 1)
    for n in range(2, T + 1):
        for a in numba.prange(T - n + 1):
            b = a + n - 1
            mx = -np.inf
            for k in range(a, b):
                mx = max(mx, Z[a, k] + Z[k + 1, b])
            s = 0.0
            for k in range(a, b):
                s += math.exp(Z[a, k] + Z[k + 1, b] - mx)
            Y = S[b + 1] - S[a]
            leaf = k0 + math.lgamma(alpha + Y) - (alpha + Y) * math.log(beta + n)
            split = l1r - math.log(n - 1) + mx + math.log(s)
            hi = max(lr + leaf, split)
            Z[a, b] = hi + math.log(math.exp(lr + leaf - hi) + math.exp(split - hi))
    return Z[0, T - 1]


def main():
    ev = load_events("usgs")
    ts = np.array([e[0] for e in ev])
    T = 512 if QUICK else 2048
    y = np.bincount(((ts - ts.min() // 3600 * 3600) // 3600).astype(int), minlength=T)[:T].astype(float)
    alpha, beta = 1.0, 1.0 / y.mean()  # bayesbin's weak prior: worth one event, centred on the mean
    S = np.concatenate([[0.0], np.cumsum(y)])
    k0 = alpha * math.log(beta) - math.lgamma(alpha)

    def L(a, n):
        Y = S[a + n] - S[a]
        return k0 + math.lgamma(alpha + Y) - (alpha + Y) * math.log(beta + n)

    def dyadic(a, n, lr, l1r):
        if n == 1:
            return L(a, 1)
        h = n // 2
        return np.logaddexp(lr + L(a, n), l1r + dyadic(a, h, lr, l1r) + dyadic(a + h, h, lr, l1r))

    out = {"T": T, "events": float(y.sum()), "mean_per_hour": float(y.mean()), "max_per_hour": float(y.max())}
    m = PoissonModel(y, alpha, beta)
    fit(PoissonModel(y[:64], alpha, beta), max_boundaries=10)  # load the fused kernels before timing
    tic = time.perf_counter()
    r = fit(m, max_boundaries=100)
    out["bayesbin"] = {"log_evidence": r.log_marginal - m.log_data_constant(), "m_map": r.m_map,
                       "max_boundaries": 100, "seconds": time.perf_counter() - tic}
    binary_tree(S, 8, alpha, beta, math.log(0.5), math.log(0.5))  # compile
    for key, f in [("dyadic", lambda lr, l1r: dyadic(0, T, lr, l1r)),
                   ("binary", lambda lr, l1r: binary_tree(S, T, alpha, beta, lr, l1r))]:
        res = []
        for rho in RHOS:
            tic = time.perf_counter()
            res.append((f(math.log(rho), math.log1p(-rho)), rho, time.perf_counter() - tic))
        z, rho, sec = max(res)
        out[key] = {"log_evidence": z, "rho": rho, "seconds_per_rho": float(np.mean([x[2] for x in res])),
                    "rho_at_edge": rho in (RHOS[0], RHOS[-1])}
    for k in ("bayesbin", "dyadic", "binary"):
        print(f"{k:>9}: log evidence {out[k]['log_evidence']:.1f}")
    write_json("tree1d.json", out)


if __name__ == "__main__":
    main()
