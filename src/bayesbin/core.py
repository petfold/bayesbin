"""Exact Bayesian binning (Endres, Oram, Schindelin & Földiák, NIPS 20, 2008).

A rate on T ordered intervals is modelled as piecewise constant with M bin
boundaries. Boundary configurations have a uniform prior given M, each bin's
rate a conjugate prior, and M a uniform prior on 0..max_boundaries. The
evidence of every M comes from one forward dynamic programme, O(M·T²):

    fwd[m][k] = log Σ  P(intervals 0..k as m+1 bins, the last ending at k)

A backward programme over the tail, combined with the forward one, gives the
posterior probability of every candidate bin [a, b] at once. The predictive
rate, its variance and the boundary posterior then follow from sums over
those bins, in O(M·T²) overall -- no per-time-point re-run of the evidence computation (the
paper's "add a virtual spike" device gives the same numbers; the tests check
it).

Two likelihoods share the machinery:

    BernoulliModel  per interval, s events and g non-events over trials,
                    firing probability f ~ Beta(σ, γ)  (the paper's PSTH)
    PoissonModel    per interval, a count y over exposure e, rate λ ~ Gamma(α, β)
                    (counts per window, several events per window allowed)

All arithmetic is in log space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp

_NEG_INF = -np.inf
_TINY = np.finfo(float).tiny  # smallest normal double: below it a term is lost or imprecise


def _log_binom(n: int, k: int) -> float:
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


# --- likelihoods ------------------------------------------------------------------


@dataclass(frozen=True)
class BernoulliModel:
    """Per interval k: s[k] trials with an event, g[k] without; within a bin
    the event probability is constant, f ~ Beta(sigma, gamma)."""

    s: np.ndarray
    g: np.ndarray
    sigma: float = 1.0
    gamma: float = 32.0  # binsdfc's default: σ = 1, γ = 32

    def __post_init__(self) -> None:
        s, g = np.asarray(self.s, float), np.asarray(self.g, float)
        if s.shape != g.shape or s.ndim != 1 or len(s) == 0:
            raise ValueError("s and g must be equal-length 1-D arrays")
        if (s < 0).any() or (g < 0).any():
            raise ValueError("counts must be non-negative")
        if self.sigma <= 0 or self.gamma <= 0:
            raise ValueError("sigma and gamma must be positive")
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "g", g)

    @property
    def T(self) -> int:
        return len(self.s)

    def _sums(self) -> tuple[np.ndarray, np.ndarray]:
        """S[a, b], G[a, b]: totals over intervals a..b (inclusive), a <= b."""
        cs = np.concatenate([[0.0], np.cumsum(self.s)])
        cg = np.concatenate([[0.0], np.cumsum(self.g)])
        return cs[None, 1:] - cs[:-1, None], cg[None, 1:] - cg[:-1, None]

    def log_bin_evidence(self) -> np.ndarray:
        """L[a, b] = log ∫ Π_{k=a..b} f^s (1-f)^g Beta(f; σ, γ) df; -inf below the diagonal."""
        S, G = self._sums()
        S, G = np.maximum(S, 0.0), np.maximum(G, 0.0)  # below the diagonal: masked next
        a, c = self.sigma, self.gamma
        L = (gammaln(S + a) + gammaln(G + c) - gammaln(S + G + a + c)
             + gammaln(a + c) - gammaln(a) - gammaln(c))
        return np.where(np.triu(np.ones_like(L, bool)), L, _NEG_INF)

    def bin_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Posterior E[f] and E[f²] of every bin [a, b]."""
        S, G = self._sums()
        S, G = np.maximum(S, 0.0), np.maximum(G, 0.0)
        a, n = S + self.sigma, S + G + self.sigma + self.gamma
        return a / n, a * (a + 1) / (n * (n + 1))

    def log_data_constant(self) -> float:
        return 0.0  # the Bernoulli sequence likelihood has no data-only factor


@dataclass(frozen=True)
class PoissonModel:
    """Per interval k: count y[k] over exposure e[k] (default 1); within a bin
    the rate is constant, λ ~ Gamma(alpha, beta) (shape, rate)."""

    y: np.ndarray
    alpha: float
    beta: float
    e: np.ndarray | None = None

    def __post_init__(self) -> None:
        y = np.asarray(self.y, float)
        e = np.ones_like(y) if self.e is None else np.asarray(self.e, float)
        if y.ndim != 1 or len(y) == 0 or e.shape != y.shape:
            raise ValueError("y (and e) must be equal-length 1-D arrays")
        if (y < 0).any() or (e < 0).any():
            raise ValueError("counts and exposures must be non-negative")
        if (e[y > 0] == 0).any():
            raise ValueError("an interval with zero exposure cannot have events")
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha and beta must be positive")
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "e", e)

    @classmethod
    def weak_prior(cls, y, e=None, weight: float = 1.0) -> PoissonModel:
        """Gamma prior centred on the overall rate, worth `weight` events."""
        y = np.asarray(y, float)
        e = np.ones_like(y) if e is None else np.asarray(e, float)
        rate = max(y.sum(), 0.5) / e.sum()
        return cls(y, alpha=weight, beta=weight / rate, e=e)

    @property
    def T(self) -> int:
        return len(self.y)

    def _sums(self) -> tuple[np.ndarray, np.ndarray]:
        cy = np.concatenate([[0.0], np.cumsum(self.y)])
        ce = np.concatenate([[0.0], np.cumsum(self.e)])
        return cy[None, 1:] - cy[:-1, None], ce[None, 1:] - ce[:-1, None]

    def log_bin_evidence(self) -> np.ndarray:
        """L[a, b] = log ∫ Π λ^y e^{-λe} Gamma(λ; α, β) dλ, without the data-only
        factor Π e^y / y! (see log_data_constant)."""
        Y, E = self._sums()
        Y, E = np.maximum(Y, 0.0), np.maximum(E, 0.0)
        a, b = self.alpha, self.beta
        with np.errstate(invalid="ignore"):
            L = a * np.log(b) - gammaln(a) + gammaln(Y + a) - (Y + a) * np.log(E + b)
        return np.where(np.triu(np.ones_like(L, bool)), L, _NEG_INF)

    def bin_moments(self) -> tuple[np.ndarray, np.ndarray]:
        Y, E = self._sums()
        a, b = np.maximum(Y, 0.0) + self.alpha, np.maximum(E, 0.0) + self.beta
        return a / b, a * (a + 1) / b**2

    def log_data_constant(self) -> float:
        pos = self.y > 0
        return float(np.sum(self.y[pos] * np.log(self.e[pos])) - np.sum(gammaln(self.y + 1)))


# --- the dynamic programmes ---------------------------------------------------------


def _forward_exact(L: np.ndarray, max_m: int) -> np.ndarray:
    """fwd[m, k]: log evidence of intervals 0..k as m+1 bins, the last ending at k.
    Plain log-sum-exp over a T×T array per step: the reference for tests."""
    T = L.shape[0]
    fwd = np.full((max_m + 1, T), _NEG_INF)
    fwd[0] = L[0]
    # shifted[r, k] = L[r+1, k]: a bin starting just after a boundary at r
    shifted = np.full((T, T), _NEG_INF)
    shifted[:-1] = L[1:]
    for m in range(1, max_m + 1):
        fwd[m] = logsumexp(fwd[m - 1][:, None] + shifted, axis=0)
    return fwd


def _forward(L: np.ndarray, max_m: int, *, tol: float = 1e-13) -> np.ndarray:
    """fwd[m, k] as in _forward_exact, one matrix-vector product per step.

    Relative to the one-bin evidence b[k] = L[0, k], a step is
        φ_m[k] = log Σ_{r<k} exp(φ_{m-1}[r] + G[r, k]),
        G[r, k] = L[r+1, k] - b[k] + b[r]   (the gain of a boundary at r),
    and exp(G - column max) is computed once. Each step then scales exp(φ) by
    its maximum and multiplies. A term dropped to underflow is below the
    smallest normal double, so a column whose sum is not far above T·TINY could
    be off by more than `tol` (relative); those columns are recomputed exactly."""
    T = L.shape[0]
    fwd = np.full((max_m + 1, T), _NEG_INF)
    b = L[0].copy()
    fwd[0] = b
    if max_m == 0 or T == 1:
        return fwd
    G = np.full((T, T), _NEG_INF)
    G[:-1] = L[1:] - b[None, :] + b[:-1, None]  # G[r, k] = L[r+1, k] - b[k] + b[r]
    G[np.tril_indices(T)] = _NEG_INF  # only r < k
    c = G.max(axis=0)
    c[~np.isfinite(c)] = 0.0
    E = np.exp(G - c)
    floor = 2 * T * _TINY / tol  # below this a column's sum may carry > tol relative error
    phi = np.zeros(T)
    for m in range(1, max_m + 1):
        p = np.max(phi[np.isfinite(phi)]) if np.isfinite(phi).any() else 0.0
        sums = np.exp(phi - p) @ E
        with np.errstate(divide="ignore"):
            new = np.log(sums) + p + c
        bad = (sums < floor) & np.isfinite(G).any(axis=0)
        if bad.any():
            new[bad] = logsumexp(phi[:, None] + G[:, bad], axis=0)
        phi = new
        fwd[m] = b + phi
    return fwd


def _backward(L: np.ndarray, max_m: int, *, exact: bool = False) -> np.ndarray:
    """bwd[j, k]: log evidence of intervals k+1..T-1 as j+1 bins (k < T-1): the
    forward programme on the reversed sequence."""
    T = L.shape[0]
    L_rev = L[::-1, ::-1].T  # L_rev[a, b] = L[T-1-b, T-1-a]
    f = (_forward_exact if exact else _forward)(L_rev, max_m)
    bwd = np.full((max_m + 1, T), _NEG_INF)
    bwd[:, :-1] = f[:, T - 2::-1]  # bwd[j, k] = f[j, T-2-k]
    return bwd


@dataclass
class BinningResult:
    """Posterior summaries, averaged over the number of bin boundaries M."""

    log_evidence: np.ndarray  # log P(D | M), M = 0..max_boundaries
    m_posterior: np.ndarray  # P(M | D), uniform prior on M
    rate: np.ndarray  # predictive rate per interval, E[f_k | D]
    rate_std: np.ndarray  # posterior std of f_k
    boundary_posterior: np.ndarray  # P(a bin ends at k | D), k = 0..T-2
    bin_posterior: np.ndarray | None  # P([a, b] is a bin | D), if kept

    @property
    def log_marginal(self) -> float:
        """log P(D), the prior over M included."""
        return float(logsumexp(self.log_evidence) - np.log(len(self.log_evidence)))

    @property
    def m_map(self) -> int:
        return int(np.argmax(self.m_posterior))


def _messages(fwd: np.ndarray, bwd: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """left[i, a]: log evidence of intervals 0..a-1 as i bins;
    right[j, b]: log evidence of intervals b+1..T-1 as j bins."""
    T = fwd.shape[1]
    left = np.full((K, T), _NEG_INF)
    left[0, 0] = 0.0
    left[1:, 1:] = fwd[:K - 1, :-1]
    right = np.full((K, T), _NEG_INF)
    right[0, T - 1] = 0.0
    right[1:] = bwd[:K - 1]
    return left, right


def bin_posterior(L: np.ndarray, fwd: np.ndarray, bwd: np.ndarray, log_c: np.ndarray,
                  *, exact: bool = False, tol: float = 1e-14) -> np.ndarray:
    """W[a, b] = Σ_M c_M Σ_i exp(left_i[a] + L[a, b] + right_{M-i}[b]): the posterior
    probability that [a, b] is a bin, for weights log_c[M] = log(P(M | D) / Z_M)
    (-inf for an M left out).

    The average over M folds into the right-hand messages first,
    R_i[b] = log Σ_j c_{i+j} right_j[b] (O(K²·T)), which leaves one sum over the
    bin's index i: a scaled matrix product, O(K·T²). A term it drops to
    underflow is below the smallest normal double; wherever that could cost more
    than `tol` in W, the entry is recomputed exactly in log space. `exact=True`
    does every entry that way (the reference for tests)."""
    T, K = L.shape[0], len(log_c)
    left, right = _messages(fwd, bwd, K)
    R = np.full((K, T), _NEG_INF)
    for i in range(K):
        j = np.arange(K - i)
        R[i] = logsumexp(log_c[i + j][:, None] + right[j], axis=0)

    if exact:
        C = np.full((T, T), _NEG_INF)
        for i in range(K):
            C = np.logaddexp(C, left[i][:, None] + R[i][None, :])
    else:
        ma, mb = left.max(axis=0), R.max(axis=0)
        ma[~np.isfinite(ma)] = 0.0
        mb[~np.isfinite(mb)] = 0.0
        S = np.exp(left - ma).T @ np.exp(R - mb)  # (T, K) @ (K, T)
        with np.errstate(divide="ignore"):
            C = np.log(S) + ma[:, None] + mb[None, :]
        # each dropped term is < TINY in the scaled product, so W is off by at most
        # K·TINY·exp(ma + mb + L); recompute exactly wherever that could matter
        with np.errstate(invalid="ignore"):
            risk = np.log(K * _TINY) + ma[:, None] + mb[None, :] + L > np.log(max(tol, _TINY))
        if risk.any():
            aa, bb = np.nonzero(risk)
            C[aa, bb] = logsumexp(left[:, aa] + R[:, bb], axis=0)
    with np.errstate(invalid="ignore", over="ignore"):
        W = np.exp(C + L)
    return np.nan_to_num(W, nan=0.0, posinf=0.0)


def bin_posterior_for_m(L: np.ndarray, fwd: np.ndarray, bwd: np.ndarray, M: int,
                        *, exact: bool = False) -> np.ndarray:
    """P([a, b] is one of the M+1 bins | D, M), for all a <= b."""
    log_c = np.full(M + 1, _NEG_INF)
    log_c[M] = -fwd[M, L.shape[0] - 1]
    return bin_posterior(L, fwd, bwd, log_c, exact=exact)


def _cover_sum(Q: np.ndarray) -> np.ndarray:
    """out[k] = Σ_{a<=k<=b} Q[a, b]."""
    P = np.cumsum(Q, axis=0)  # P[k, b] = Σ_{a<=k} Q[a, b]
    R = np.cumsum(P[:, ::-1], axis=1)[:, ::-1]  # R[k, b] = Σ_{b'>=b} P[k, b']
    return np.diagonal(R).copy()


def credible_m_range(log_evidence: np.ndarray, mass: float) -> tuple[int, int]:
    """The smallest interval of M around the most probable one holding at least
    `mass` posterior probability, grown toward the more probable neighbour
    (binsdfc's --p-post-lbound rule; mass=0 gives the most probable M alone)."""
    post = np.exp(log_evidence - logsumexp(log_evidence))
    lo = hi = int(np.argmax(log_evidence))
    top = len(log_evidence) - 1
    while post[lo:hi + 1].sum() < mass:
        if lo > 0 and (hi == top or log_evidence[lo - 1] > log_evidence[hi + 1]):
            lo -= 1
        elif hi < top:
            hi += 1
        else:
            break
    return lo, hi


def fit(model, max_boundaries: int = 10, *, m_mass: float | None = None,
        keep_bins: bool = False, min_m_posterior: float = 1e-12,
        exact: bool = False) -> BinningResult:
    """Exact posterior over piecewise-constant rates for `model`
    (BernoulliModel or PoissonModel).

    Predictions average over every M = 0..max_boundaries by default, as the
    paper recommends. With `m_mass`, only over the credible range of M holding
    that much posterior mass (see credible_m_range), as binsdfc does.
    `exact=True` computes everything in plain log space (slower; the fast
    paths carry proven bounds: 1e-13 relative in the evidences, 1e-14 absolute
    in the bin posterior -- see _forward and bin_posterior)."""
    T = model.T
    max_m = min(max_boundaries, T - 1)
    L = model.log_bin_evidence()
    if exact:
        fwd, bwd = _forward_exact(L, max_m), _backward(L, max_m, exact=True)
    else:
        fwd, bwd = _forward(L, max_m), _backward(L, max_m)
    log_ev = np.array([fwd[m, T - 1] - _log_binom(T - 1, m) for m in range(max_m + 1)])
    log_ev += model.log_data_constant()
    post = np.exp(log_ev - logsumexp(log_ev))

    use = post >= min_m_posterior
    if m_mass is not None:
        lo, hi = credible_m_range(log_ev, m_mass)
        use = np.zeros_like(use)
        use[lo:hi + 1] = True
    mean, second = model.bin_moments()
    with np.errstate(divide="ignore"):
        log_c = np.where(use, np.log(post / post[use].sum()), _NEG_INF) - fwd[:, T - 1]
    W = bin_posterior(L, fwd, bwd, log_c, exact=exact)

    rate = _cover_sum(W * np.nan_to_num(mean))
    var = np.clip(_cover_sum(W * np.nan_to_num(second)) - rate**2, 0.0, None)
    boundary = W[:, :-1].sum(axis=0)  # bins ending at b < T-1
    return BinningResult(log_ev, post, rate, np.sqrt(var), boundary, W if keep_bins else None)


def spike_counts(trials, t_start: int, t_end: int) -> tuple[np.ndarray, np.ndarray]:
    """(s, g) per interval t_start..t_end (inclusive) from spike trains given as
    integer spike times, one iterable per trial -- the paper's representation,
    at most one spike per interval per trial."""
    T = t_end - t_start + 1
    s = np.zeros(T)
    for trial in trials:
        idx = np.asarray([t for t in trial if t_start <= t <= t_end], int) - t_start
        if len(np.unique(idx)) != len(idx):
            raise ValueError("more than one spike in an interval: use a finer discretization")
        s[idx] += 1
    return s, len(trials) - s
