#!/usr/bin/env python3
"""Rate and sd of a fit in long double (80-bit on x86: 64-bit mantissa), from the
model's double-precision bin evidences, with the variance in its stable form
(within bins + between bins about the rate, no E[f²] - E[f]²). O(T³) in Python
loops: a reference for small T (a few hundred), for testing the fast paths'
precision, not for use.

    from longdouble_reference import reference
    rate, sd = reference(model, max_boundaries)
"""

from __future__ import annotations

import numpy as np

LD = np.longdouble


def _lse(x: np.ndarray) -> np.longdouble:
    m = np.max(x)
    if not np.isfinite(m):
        return LD(-np.inf)
    return m + np.log(np.sum(np.exp(x - m)))


def _forward(L: np.ndarray, M: int) -> np.ndarray:
    T = L.shape[0]
    f = np.full((M + 1, T), -np.inf, dtype=LD)
    f[0] = L[0]
    for m in range(1, M + 1):
        for k in range(m, T):
            f[m, k] = _lse(f[m - 1, :k] + L[1:k + 1, k])
    return f


def reference(model, max_boundaries: int, min_m_posterior: float = 1e-12):
    """(rate, sd) as fit(model, max_boundaries) computes them (average over every M
    with posterior >= min_m_posterior), in long double."""
    from math import lgamma

    L = model.log_bin_evidence().astype(LD)
    T, M = L.shape[0], min(max_boundaries, L.shape[0] - 1)
    m1, m2 = (np.nan_to_num(x).astype(LD) for x in model.bin_moments())
    fwd = _forward(L, M)
    rev = _forward(L[::-1, ::-1].T.copy(), M)
    bwd = np.full((M + 1, T), -np.inf, dtype=LD)
    bwd[:, :-1] = rev[:, T - 2::-1]
    log_binom = [LD(lgamma(T) - lgamma(m + 1) - lgamma(T - m)) for m in range(M + 1)]
    log_ev = np.array([fwd[m, T - 1] - log_binom[m] for m in range(M + 1)])
    post = np.exp(log_ev - _lse(log_ev))
    use = post >= min_m_posterior
    with np.errstate(divide="ignore"):
        log_c = np.where(use, np.log(post / post[use].sum()), -np.inf) - fwd[:, T - 1]
    K = M + 1
    left = np.full((K, T), -np.inf, dtype=LD)
    left[0, 0] = 0
    left[1:, 1:] = fwd[:K - 1, :T - 1]
    right = np.full((K, T), -np.inf, dtype=LD)
    right[0, T - 1] = 0
    right[1:] = bwd[:K - 1]
    W = np.zeros((T, T), dtype=LD)
    for a in range(T):  # W[a, b] = Σ_{i+j used} exp(left_i[a] + L[a, b] + right_j[b] + log_c[i+j])
        acc = np.full(T, -np.inf, dtype=LD)
        for i in range(K):
            if np.isfinite(left[i, a]):
                for j in range(K - i):
                    if np.isfinite(log_c[i + j]):
                        acc = np.logaddexp(acc, left[i, a] + right[j] + log_c[i + j])
        W[a, a:] = np.exp(acc[a:] + L[a, a:])
    rate, within = np.zeros(T, dtype=LD), np.zeros(T, dtype=LD)
    for a in range(T):
        for b in range(a, T):
            rate[a:b + 1] += W[a, b] * m1[a, b]
            within[a:b + 1] += W[a, b] * (m2[a, b] - m1[a, b] ** 2)
    between = np.zeros(T, dtype=LD)
    for a in range(T):
        for b in range(a, T):
            between[a:b + 1] += W[a, b] * (m1[a, b] - rate[a:b + 1]) ** 2
    return rate, np.sqrt(within + between)
