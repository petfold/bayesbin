"""Fused kernels (optional: needs numba; `pip install bayesbin[fast]`).

The NumPy path does each element-wise step as its own pass over memory; these
kernels do a whole step per element in one loop, reading the bin evidences
straight from prefix sums and lgamma tables. The matrix products stay in BLAS.
Results agree with the NumPy path to rounding (the tests run both).

A model is passed as `params = (code, P1, P2, T1, T2, T3, k0, k1, k2, k3)`:

    code 0, Bernoulli: P1, P2 = prefix sums of s and g (int64); T1, T2, T3 =
        gammaln(n + σ), gammaln(n + γ), gammaln(n + σ + γ); k0, k1, k2 =
        gammaln(σ + γ), gammaln(σ), gammaln(γ); k3 unused (σ, γ in `shape`)
    code 1, Poisson, constant exposure e0: P1 = prefix sums of y (int64);
        T1 = gammaln(n + α); T2 = log(e0·n + β) per bin length; k0 = α log β -
        gammaln(α); k1 = e0; k2 = α; k3 = β; P2, T3 unused
    code 2, Poisson, varying exposure: as code 1, but P2 = prefix sums of e
        (float64) and T2, k1 unused

and `shape = (σ, γ)` or `(α, β)` for the posterior moments.

Each kernel dispatches once on `code` to an inlined body in which `code` is a
constant, so that every loop is compiled for one model: a generic loop that
branches on the model per element ran 8× slower.

The kernels run on numba's threads (`NUMBA_NUM_THREADS`, `numba.set_num_threads`),
the matrix products included: each is split into fixed parts, one BLAS call per
part, with BLAS itself held to one thread (threadpoolctl) while they run. Letting
OpenBLAS thread the products instead made its idle workers spin against numba's.
The parts are fixed, not one per thread, and combined in a fixed order, so the
results do not depend on the number of threads.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from threadpoolctl import ThreadpoolController

_BLAS = ThreadpoolController()


def one_blas_thread():
    """Context: BLAS on one thread (the kernels split the products over numba's threads)."""
    return _BLAS.limit(limits=1, user_api="blas")

_NEG = -np.inf
_TINY = np.finfo(np.float64).tiny
# the parallel loops split work into a fixed number of parts (not one per thread), and
# combine them in a fixed order: results do not depend on the number of threads
_NCH = 16  # row groups in the gain slice (interleaved)
_MP = 4  # parts of the M steps in the rows before a block
_RB = 32  # rows per part of a bin-posterior tile
_BW = 64  # bin ends per part in the model folding


@njit(inline="always", cache=True)
def _iec(code, P1, P2, T1, T2, T3, k0, k1, k2, k3, a, b):
    """Log evidence of the bin a..b, prior normalisation included (as bin_block)."""
    if code == 0:
        s = int(P1[b + 1] - P1[a])  # int(): the branch also compiles for Poisson's float P2
        g = int(P2[b + 1] - P2[a])
        L = T1[s] + T2[g] - T3[s + g]
        L = L + k0  # the same order as the NumPy path: identical doubles
        L = L - k1
        L = L - k2
        return L
    y = int(P1[b + 1] - P1[a])
    if code == 1:
        lE = T2[b - a + 1]
    else:
        lE = math.log(P2[b + 1] - P2[a] + k3)
    L = k0 + T1[y]
    L = L - (y + k2) * lE
    return L


@njit(inline="always", cache=True)
def _moments(code, P1, P2, k1, sh0, sh1, a, b):
    """Posterior E[x] and E[x²] of the bin a..b."""
    if code == 0:
        s = P1[b + 1] - P1[a]
        n = s + (P2[b + 1] - P2[a])
        aa = s + sh0
        nn = n + sh0
        nn = nn + sh1
        m1 = aa / nn
        return m1, (aa + 1.0) * aa / ((nn + 1.0) * nn)
    y = P1[b + 1] - P1[a]
    if code == 1:
        E = k1 * (b - a + 1) if k1 != 1.0 else float(b - a + 1)
    else:
        E = P2[b + 1] - P2[a]
    aa = y + sh0
    bb = E + sh1
    return aa / bb, aa * (aa + 1.0) / (bb * bb)


@njit(inline="always", cache=True)
def _at(T, reverse, a, b):
    """Map bin (a, b) of the (possibly reversed) sequence to the original one."""
    if reverse:
        return T - 1 - b, T - 1 - a
    return a, b


@njit(inline="always", cache=True)
def _base(code, P1, P2, T1, T2, T3, k0, k1, k2, k3, T, reverse):
    out = np.empty(T)
    for k in range(T):
        a, bb = _at(T, reverse, 0, k)
        out[k] = _iec(code, P1, P2, T1, T2, T3, k0, k1, k2, k3, a, bb)
    return out


@njit(cache=True)
def base(code, P1, P2, T1, T2, T3, k0, k1, k2, k3, T, reverse):
    """b[k] = log evidence of 0..k as one bin."""
    if code == 0:
        return _base(0, P1, P2, T1, T2, T3, k0, k1, k2, k3, T, reverse)
    if code == 1:
        return _base(1, P1, P2, T1, T2, T3, k0, k1, k2, k3, T, reverse)
    return _base(2, P1, P2, T1, T2, T3, k0, k1, k2, k3, T, reverse)


@njit(inline="always", cache=True)
def _gain_slice(code, P1, P2, T1, T2, T3, k0, k1, k2, k3, T, reverse, b, c0, c1, lo):
    nr, w = c1 - 1, c1 - c0
    E = np.empty((nr, w))
    cg = np.full((_NCH, w), _NEG)  # column maxima per row group (a max: any order is exact)
    for g in prange(_NCH):  # rows in interleaved groups: whole rows per thread
        for r in range(g, nr, _NCH):
            br = b[r]
            for u in range(w):
                k = c0 + u
                if r < k:
                    a, bb = _at(T, reverse, r + 1, k)
                    x = _iec(code, P1, P2, T1, T2, T3, k0, k1, k2, k3, a, bb) - b[k] + br
                    E[r, u] = x
                    if x > cg[g, u]:
                        cg[g, u] = x
                else:
                    E[r, u] = _NEG
    c = np.empty(w)
    reach = np.empty(w, np.bool_)
    for u in range(w):
        mx = _NEG
        for g in range(_NCH):
            if cg[g, u] > mx:
                mx = cg[g, u]
        reach[u] = mx != _NEG
        c[u] = mx if reach[u] else 0.0
    for r in prange(nr):
        for u in range(w):
            x = E[r, u] - c[u]
            E[r, u] = math.exp(x) if x >= lo else 0.0
    return E, c, reach


@njit(cache=True, parallel=True)
def gain_slice(code, P1, P2, T1, T2, T3, k0, k1, k2, k3, T, reverse, b, c0, c1, lo):
    """E[r, u] = exp(G[r, u] - c[u]) for rows r < c1-1, where G[r, u] = L[r+1, c0+u] - b[c0+u]
    + b[r] (0 where r >= c0+u) and c[u] is the column maximum (0 for a column with no finite
    gain), flushed to 0 below exp(lo); and c, and which columns have a finite gain."""
    if code == 0:
        return _gain_slice(0, P1, P2, T1, T2, T3, k0, k1, k2, k3, T, reverse, b, c0, c1, lo)
    if code == 1:
        return _gain_slice(1, P1, P2, T1, T2, T3, k0, k1, k2, k3, T, reverse, b, c0, c1, lo)
    return _gain_slice(2, P1, P2, T1, T2, T3, k0, k1, k2, k3, T, reverse, b, c0, c1, lo)


@njit(cache=True, parallel=True)
def rows_before(phi, M, c0, lo, E):
    """The rows before a block, for all M steps at once: X[m-1, r] = exp(phi[m-1, r] - q[m-1])
    for r < c0 (q the row maximum, 0 for a row with no finite entry; flushed to 0 below
    exp(lo)), and before = X @ E[:c0], in _MP parts of the steps, one BLAS call each (BLAS
    held to one thread by the caller). Returns before, q, and which rows are finite."""
    q = np.empty(M)
    qfin = np.empty(M, np.bool_)
    before = np.empty((M, E.shape[1]))
    Eb = E[:c0]
    step = (M + _MP - 1) // _MP
    for part in prange(_MP):
        i0 = part * step
        i1 = min(M, i0 + step)
        if i0 >= i1:
            continue
        X = np.empty((i1 - i0, c0))
        for i in range(i0, i1):
            mx = _NEG
            for r in range(c0):
                if phi[i, r] > mx:
                    mx = phi[i, r]
            qfin[i] = mx != _NEG
            q[i] = mx if qfin[i] else 0.0
            for r in range(c0):
                x = phi[i, r] - q[i]
                X[i - i0, r] = math.exp(x) if x >= lo else 0.0
        before[i0:i1] = np.dot(X, Eb)
    return before, q, qfin


@njit(inline="always", cache=True)
def _block_steps(code, P1, P2, T1, T2, T3, k0, k1, k2, k3, T, reverse, b,
                 phi, E, before, q, qfin, cmax, reach, c0, c1, floor, lo):
    M = phi.shape[0] - 1
    nr, w = c1 - 1, c1 - c0
    sums = np.empty(w)
    n_exact = 0
    for m in range(1, M + 1):
        p_in = _NEG
        for r in range(c0, nr):
            if phi[m - 1, r] > p_in:
                p_in = phi[m - 1, r]
        p_b = q[m - 1] if (c0 > 0 and qfin[m - 1]) else _NEG
        p = max(p_in, p_b)
        if p == _NEG:
            continue
        if p_b != _NEG:
            f = math.exp(p_b - p)
            for u in range(w):
                sums[u] = before[m - 1, u] * f
        else:
            for u in range(w):
                sums[u] = 0.0
        for r in range(c0, nr):
            x = phi[m - 1, r] - p
            if x < lo:
                continue
            vr = math.exp(x)
            for u in range(w):
                sums[u] += vr * E[r, u]
        for u in range(w):
            k = c0 + u
            if k < m:
                phi[m, k] = _NEG
            elif sums[u] >= floor or not reach[u]:
                phi[m, k] = (math.log(sums[u]) if sums[u] > 0.0 else _NEG) + p + cmax[u]
            else:  # underflow could matter: exact log-sum-exp over the column
                n_exact += 1
                mx = _NEG
                for r in range(k):
                    if phi[m - 1, r] == _NEG:
                        continue
                    a, bb = _at(T, reverse, r + 1, k)
                    t = phi[m - 1, r] + _iec(code, P1, P2, T1, T2, T3, k0, k1, k2, k3, a, bb) - b[k] + b[r]
                    if t > mx:
                        mx = t
                if mx == _NEG:
                    phi[m, k] = _NEG
                else:
                    s = 0.0
                    for r in range(k):
                        if phi[m - 1, r] == _NEG:
                            continue
                        a, bb = _at(T, reverse, r + 1, k)
                        s += math.exp(phi[m - 1, r] + _iec(code, P1, P2, T1, T2, T3, k0, k1, k2, k3, a, bb)
                                      - b[k] + b[r] - mx)
                    phi[m, k] = mx + math.log(s)
    return n_exact


@njit(cache=True)
def block_steps(code, P1, P2, T1, T2, T3, k0, k1, k2, k3, T, reverse, b,
                phi, E, before, q, qfin, cmax, reach, c0, c1, floor, lo):
    """The steps m = 1..M for the columns c0..c1-1: the rows before the block come in as
    `before` (their matrix product, scaled by q[m-1]); the rows inside it are added here,
    step by step (terms below exp(lo) dropped, as in the product), with the log, the
    cut-off below `floor` and the exact fallback. Returns the number of columns redone
    exactly."""
    if code == 0:
        return _block_steps(0, P1, P2, T1, T2, T3, k0, k1, k2, k3, T, reverse, b,
                            phi, E, before, q, qfin, cmax, reach, c0, c1, floor, lo)
    if code == 1:
        return _block_steps(1, P1, P2, T1, T2, T3, k0, k1, k2, k3, T, reverse, b,
                            phi, E, before, q, qfin, cmax, reach, c0, c1, floor, lo)
    return _block_steps(2, P1, P2, T1, T2, T3, k0, k1, k2, k3, T, reverse, b,
                        phi, E, before, q, qfin, cmax, reach, c0, c1, floor, lo)


@njit(inline="always", cache=True)
def _tile_accumulate(code, P1, P2, T1, T2, T3, k0, k1, k2, k3, sh0, sh1,
                     A, Bt, cscale, a0, a1, b0, ma, mb, left, R, log_risk, d1, d2, ends):
    na, nb = a1 - a0, Bt.shape[1]
    cmin = _TINY / cscale
    K = left.shape[0]
    # rows in parts of _RB, each with its own product C = A[rows] @ Bt (one BLAS call, BLAS
    # held to one thread by the caller) and its own column sums; the row sums are per row
    ng = (na + _RB - 1) // _RB
    col1 = np.zeros((ng, nb))
    col2 = np.zeros((ng, nb))
    colw = np.zeros((ng, nb))
    rows1 = np.zeros(na)
    rows2 = np.zeros(na)
    for g in prange(ng):
        C = np.dot(A[a0 + g * _RB:min(a1, a0 + (g + 1) * _RB)], Bt)
        for i in range(g * _RB, min(na, (g + 1) * _RB)):
            a = a0 + i
            mai = ma[a]
            row1 = 0.0
            row2 = 0.0
            for j in range(max(0, a - b0), nb):
                bb = b0 + j
                L = _iec(code, P1, P2, T1, T2, T3, k0, k1, k2, k3, a, bb)
                shift = mai + mb[bb] + L
                if log_risk + shift > 0.0:
                    mx = _NEG
                    for t in range(K):
                        v = left[t, a] + R[t, bb]
                        if v > mx:
                            mx = v
                    if mx == _NEG:
                        continue
                    s = 0.0
                    for t in range(K):
                        s += math.exp(left[t, a] + R[t, bb] - mx)
                    W = math.exp(mx + math.log(s) + L)
                else:
                    cv = C[i - g * _RB, j]
                    if cv < cmin:  # C < TINY: W < TINY exp(-log_risk), negligible
                        continue
                    W = (cv * cscale) * math.exp(shift)
                m1, m2 = _moments(code, P1, P2, k1, sh0, sh1, a, bb)
                q1 = W * m1
                q2 = W * m2
                row1 += q1
                row2 += q2
                col1[g, j] += q1
                col2[g, j] += q2
                colw[g, j] += W
            rows1[i] = row1
            rows2[i] = row2
    for i in range(na):  # the ranges a..b through the difference arrays: +q at a
        d1[a0 + i] += rows1[i]
        d2[a0 + i] += rows2[i]
    for j in prange(nb):  # -q at b + 1, the groups in a fixed order
        s1 = 0.0
        s2 = 0.0
        sw = 0.0
        for g in range(ng):
            s1 += col1[g, j]
            s2 += col2[g, j]
            sw += colw[g, j]
        d1[b0 + j + 1] -= s1
        d2[b0 + j + 1] -= s2
        ends[b0 + j] += sw


@njit(cache=True, parallel=True)
def tile_accumulate(code, P1, P2, T1, T2, T3, k0, k1, k2, k3, sh0, sh1,
                    A, Bt, cscale, a0, a1, b0, ma, mb, left, R, log_risk, d1, d2, ends):
    """For the bins of one tile (starts a0..a1-1, ends b0..b0+nb-1): W = cscale C[a, b]
    exp(ma[a] + mb[b] + L[a, b]) with C = A[a0:a1] @ Bt (or, where underflow could matter,
    exactly in log space), and its contributions to the rates, second moments and boundary
    posterior, in one pass."""
    if code == 0:
        _tile_accumulate(0, P1, P2, T1, T2, T3, k0, k1, k2, k3, sh0, sh1,
                         A, Bt, cscale, a0, a1, b0, ma, mb, left, R, log_risk, d1, d2, ends)
    elif code == 1:
        _tile_accumulate(1, P1, P2, T1, T2, T3, k0, k1, k2, k3, sh0, sh1,
                         A, Bt, cscale, a0, a1, b0, ma, mb, left, R, log_risk, d1, d2, ends)
    else:
        _tile_accumulate(2, P1, P2, T1, T2, T3, k0, k1, k2, k3, sh0, sh1,
                         A, Bt, cscale, a0, a1, b0, ma, mb, left, R, log_risk, d1, d2, ends)


@njit(cache=True, parallel=True)
def fold_models(log_c, right):
    """R[i, b] = log Σ_j exp(log_c[i+j] + right[j, b]) over the models with a weight, as
    core._fold_models: per b, one exp per term and one log per entry."""
    K, T = right.shape
    R = np.empty((K, T))
    used = np.flatnonzero(np.isfinite(log_c))
    for part in prange((T + _BW - 1) // _BW):
        col = np.empty(K)
        for b in range(part * _BW, min(T, (part + 1) * _BW)):
            for j in range(K):
                col[j] = right[j, b]
            for i in range(K):
                mx = _NEG
                for M in used:
                    if M >= i:
                        v = log_c[M] + col[M - i]
                        if v > mx:
                            mx = v
                if mx == _NEG:
                    R[i, b] = _NEG
                    continue
                s = 0.0
                for M in used:
                    if M >= i:
                        v = log_c[M] + col[M - i]
                        if v != _NEG:
                            s += math.exp(v - mx)
                R[i, b] = mx + math.log(s)
    return R
