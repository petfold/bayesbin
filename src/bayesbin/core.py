"""Exact Bayesian binning (Endres, Oram, Schindelin & Földiák, NIPS 20, 2008).

A rate on T ordered intervals is modelled as piecewise constant with M bin
boundaries. Boundary configurations have a uniform prior given M, each bin's
rate a conjugate prior, and M a uniform prior on 0..max_boundaries. The
evidence of every M comes from one forward dynamic programme, O(M·T²):

    fwd[m][k] = log Σ  P(intervals 0..k as m+1 bins, the last ending at k)

A backward programme over the tail, combined with the forward one, gives the
posterior probability of every candidate bin [a, b] at once. The predictive
rate, its variance and the boundary posterior then follow from sums over
those bins -- no per-time-point re-run of the evidence computation (the
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


def _forward(L: np.ndarray, max_m: int) -> np.ndarray:
    """fwd[m, k]: log evidence of intervals 0..k as m+1 bins, the last ending at k."""
    T = L.shape[0]
    fwd = np.full((max_m + 1, T), _NEG_INF)
    fwd[0] = L[0]
    # shifted[r, k] = L[r+1, k]: a bin starting just after a boundary at r
    shifted = np.full((T, T), _NEG_INF)
    shifted[:-1] = L[1:]
    for m in range(1, max_m + 1):
        fwd[m] = logsumexp(fwd[m - 1][:, None] + shifted, axis=0)
    return fwd


def _backward(L: np.ndarray, max_m: int) -> np.ndarray:
    """bwd[j, k]: log evidence of intervals k+1..T-1 as j+1 bins (k < T-1)."""
    T = L.shape[0]
    bwd = np.full((max_m + 1, T), _NEG_INF)
    bwd[0, :-1] = L[1:, T - 1]
    shifted = np.full((T, T), _NEG_INF)  # shifted[k, r] = L[k+1, r]
    shifted[:-1] = L[1:]
    # the next boundary r must leave room: r <= T-2
    shifted[:, T - 1] = _NEG_INF
    for j in range(1, max_m + 1):
        bwd[j] = logsumexp(shifted + bwd[j - 1][None, :], axis=1)
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


def bin_posterior_for_m(L: np.ndarray, fwd: np.ndarray, bwd: np.ndarray, M: int) -> np.ndarray:
    """P([a, b] is one of the M+1 bins | D, M), for all a <= b."""
    T = L.shape[0]
    Z = fwd[M, T - 1]
    acc = np.full((T, T), _NEG_INF)
    for i in range(M + 1):  # [a, b] is bin number i
        left = np.full(T, _NEG_INF)  # log evidence of 0..a-1 as i bins
        if i == 0:
            left[0] = 0.0
        else:
            left[1:] = fwd[i - 1, :-1]
        right = np.full(T, _NEG_INF)  # log evidence of b+1..T-1 as M-i bins
        if i == M:
            right[T - 1] = 0.0
        else:
            right = bwd[M - i - 1]
        acc = np.logaddexp(acc, left[:, None] + right[None, :])
    with np.errstate(invalid="ignore"):
        W = np.exp(acc + L - Z)
    return np.nan_to_num(W, nan=0.0)


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
        keep_bins: bool = False, min_m_posterior: float = 1e-12) -> BinningResult:
    """Exact posterior over piecewise-constant rates for `model`
    (BernoulliModel or PoissonModel).

    Predictions average over every M = 0..max_boundaries by default, as the
    paper recommends. With `m_mass`, only over the credible range of M holding
    that much posterior mass (see credible_m_range), as binsdfc does."""
    T = model.T
    max_m = min(max_boundaries, T - 1)
    L = model.log_bin_evidence()
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
    W = np.zeros((T, T))
    for m in np.flatnonzero(use):
        W += post[m] * bin_posterior_for_m(L, fwd, bwd, m)
    W /= post[use].sum()

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
