"""Bayesian binning as data arrive: the forward programme one interval at a time.

The forward programme's column k depends only on the data up to k, so a new
interval adds one column and changes nothing before it: O(M·T) per interval,
the same total as one batch fit, with nothing recomputed. From the forward
state alone come, exactly as a batch fit on the data so far gives them:

- the evidence of every number of boundaries M, and P(M | data);
- the rate *now* (in the latest interval) and its standard deviation;
- where the current bin started (the last change point);
- the predictive distribution of the next interval's count.

Estimates for earlier intervals given all the data (smoothing) need the
backward programme, which starts from the end and changes with every new
interval: `fit()` runs the batch fit on the data so far.

    ob = OnlineBinning.poisson(alpha=1.0, beta=0.5, max_boundaries=10)
    for y in stream:
        p = ob.next_cdf(y)          # before the update: how surprising is y?
        ob.update(y)
        rate, sd = ob.rate_now()

The priors are those of BernoulliModel and PoissonModel; a prior cannot be fitted
to the data as PoissonModel.weak_prior does, since the data are not all there yet.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import betabinom, nbinom

from bayesbin.core import BernoulliModel, BinningResult, PoissonModel, _log_binom, fit

_NEG_INF = -np.inf


class OnlineBinning:
    """The forward state of Bayesian binning, updated one interval (or a few) at a time.
    Make one with OnlineBinning.bernoulli(...) or OnlineBinning.poisson(...)."""

    def __init__(self, kind: str, prior: tuple[float, float], max_boundaries: int,
                 min_m_posterior: float = 1e-12):
        if max_boundaries < 0:
            raise ValueError("max_boundaries must be >= 0")
        if prior[0] <= 0 or prior[1] <= 0:
            raise ValueError("the prior's parameters must be positive")
        self.kind, self.prior, self.max_boundaries = kind, prior, max_boundaries
        self.min_m_posterior = min_m_posterior
        a, b = prior
        # per-bin prior normaliser, as the batch models add it (same order of operations)
        if kind == "bernoulli":
            self._k = (gammaln(a + b), gammaln(a), gammaln(b))
        else:
            self._k = (a * np.log(b) - gammaln(a),)
        self.T = 0
        cap = 64
        self._x1, self._x2 = np.zeros(cap), np.zeros(cap)  # s, g  or  y, e
        self._c1, self._c2 = np.zeros(cap + 1), np.zeros(cap + 1)  # their prefix sums
        self._fwd = np.full((max_boundaries + 1, cap), _NEG_INF)
        self._data_const = 0.0
        self._last_col = None  # L(a, T-1) for a = 0..T-1

    @classmethod
    def bernoulli(cls, max_boundaries: int = 10, sigma: float = 1.0, gamma: float = 32.0,
                  **kw) -> OnlineBinning:
        """Per interval, s trials with an event and g without; f ~ Beta(sigma, gamma) per bin
        (as BernoulliModel)."""
        return cls("bernoulli", (sigma, gamma), max_boundaries, **kw)

    @classmethod
    def poisson(cls, alpha: float, beta: float, max_boundaries: int = 10, **kw) -> OnlineBinning:
        """Per interval, a count y over exposure e (default 1); λ ~ Gamma(alpha, beta) per bin
        (shape, rate; as PoissonModel)."""
        return cls("poisson", (alpha, beta), max_boundaries, **kw)

    # --- adding data ----------------------------------------------------------------

    def update(self, x1, x2=None) -> None:
        """Add one interval or several (arrays), in order.

        Bernoulli: update(s, g), events and non-events. Poisson: update(y) or
        update(y, e), counts and exposures (default 1)."""
        x1 = np.atleast_1d(np.asarray(x1, float))
        if self.kind == "bernoulli":
            if x2 is None:
                raise ValueError("Bernoulli: update(s, g)")
            x2 = np.atleast_1d(np.asarray(x2, float))
        else:
            x2 = np.ones_like(x1) if x2 is None else np.atleast_1d(np.asarray(x2, float))
        if x1.shape != x2.shape or x1.ndim != 1:
            raise ValueError("the two arguments must have the same length")
        if (x1 < 0).any() or (x2 < 0).any():
            raise ValueError("counts (and exposures) must be non-negative")
        if self.kind == "poisson" and ((x2 == 0) & (x1 > 0)).any():
            raise ValueError("an interval with zero exposure cannot have events")
        for v1, v2 in zip(x1, x2):
            self._add(v1, v2)

    def _grow(self) -> None:
        cap = 2 * self._x1.size
        for name in ("_x1", "_x2"):
            a = np.zeros(cap)
            a[:self.T] = getattr(self, name)[:self.T]
            setattr(self, name, a)
        for name in ("_c1", "_c2"):
            a = np.zeros(cap + 1)
            a[:self.T + 1] = getattr(self, name)[:self.T + 1]
            setattr(self, name, a)
        f = np.full((self.max_boundaries + 1, cap), _NEG_INF)
        f[:, :self.T] = self._fwd[:, :self.T]
        self._fwd = f

    def _add(self, v1: float, v2: float) -> None:
        if self.T == self._x1.size:
            self._grow()
        k = self.T
        self._x1[k], self._x2[k] = v1, v2
        self._c1[k + 1], self._c2[k + 1] = self._c1[k] + v1, self._c2[k] + v2
        if self.kind == "poisson" and v1 > 0:
            self._data_const += v1 * np.log(v2)
        if self.kind == "poisson":
            self._data_const -= gammaln(v1 + 1)
        self.T = k + 1
        col = self._column(k)  # L(a, k), a = 0..k
        f = self._fwd
        f[0, k] = col[0]
        mm = min(self.max_boundaries, k)
        if mm > 0:
            # fwd[m, k] = log Σ_{r<k} exp(fwd[m-1, r] + L(r+1, k)), for all m at once
            with np.errstate(invalid="ignore"):
                f[1:mm + 1, k] = logsumexp(f[:mm, :k] + col[None, 1:k + 1], axis=1)
        self._last_col = col

    def _stats(self, a: np.ndarray, b: int):
        """The totals of the bins a..b (a an array): (S, G) or (Y, E)."""
        return self._c1[b + 1] - self._c1[a], self._c2[b + 1] - self._c2[a]

    def _column(self, b: int) -> np.ndarray:
        """L(a, b), the log evidence of the bin a..b with the prior's normaliser, a = 0..b."""
        a = np.arange(b + 1)
        u, v = self._stats(a, b)
        p, q = self.prior
        if self.kind == "bernoulli":
            L = gammaln(u + p)
            L += gammaln(v + q)
            L -= gammaln(u + v + p + q)
            L += self._k[0]
            L -= self._k[1]
            L -= self._k[2]
            return L
        L = self._k[0] + gammaln(u + p)
        L -= (u + p) * np.log(v + q)
        return L

    def _moments(self, a: np.ndarray, b: int):
        u, v = self._stats(a, b)
        p, q = self.prior
        if self.kind == "bernoulli":
            A, n = u + p, u + v + p + q
            return A / n, A * (A + 1.0) / (n * (n + 1.0))
        A, B = u + p, v + q
        return A / B, A * (A + 1.0) / B**2

    # --- what the forward state says about the data so far ------------------------------

    @property
    def log_evidence(self) -> np.ndarray:
        """log P(data | M), M = 0..min(max_boundaries, T-1): as fit(...).log_evidence."""
        if self.T == 0:
            return np.zeros(1)
        max_m = min(self.max_boundaries, self.T - 1)
        lev = np.array([self._fwd[m, self.T - 1] - _log_binom(self.T - 1, m) for m in range(max_m + 1)])
        return lev + self._data_const

    @property
    def m_posterior(self) -> np.ndarray:
        """P(M | data), a uniform prior on M."""
        lev = self.log_evidence
        return np.exp(lev - logsumexp(lev))

    @property
    def m_map(self) -> int:
        return int(np.argmax(self.m_posterior))

    @property
    def log_marginal(self) -> float:
        """log P(data), the prior over M included: as fit(...).log_marginal."""
        lev = self.log_evidence
        return float(logsumexp(lev) - np.log(len(lev)))

    def current_bin_start(self) -> np.ndarray:
        """P(the current bin, the one holding the latest interval, starts at a | data),
        a = 0..T-1; a = 0 means no change so far. Averaged over M as fit() does (the
        M with P(M | data) >= min_m_posterior)."""
        if self.T == 0:
            raise ValueError("no data yet")
        T, f, col = self.T, self._fwd, self._last_col
        post = self.m_posterior
        use = post >= self.min_m_posterior
        w = np.where(use, post, 0.0) / post[use].sum()
        p = np.zeros(T)
        p[0] += w[0]
        for m in range(1, len(w)):
            if w[m] == 0.0:
                continue
            with np.errstate(invalid="ignore"):  # -inf - (-inf) where no such bin: 0 below
                lp = f[m - 1, :T - 1] + col[1:T] - f[m, T - 1]
            p[1:] += w[m] * np.exp(np.where(np.isfinite(lp), lp, _NEG_INF))
        return p

    def rate_now(self) -> tuple[float, float]:
        """The rate in the latest interval given the data so far, and its standard
        deviation (the variance within bins plus between them: no cancellation)."""
        p = self.current_bin_start()
        a = np.arange(self.T)
        m1, m2 = self._moments(a, self.T - 1)
        p = p / p.sum()
        mean = float(np.dot(p, m1))
        var = float(np.dot(p, m2 - m1**2) + np.dot(p, (m1 - mean) ** 2))
        return mean, float(np.sqrt(max(var, 0.0)))

    # --- the next interval ----------------------------------------------------------

    def _next_mixture(self):
        """The predictive distribution of the next interval as a mixture: its bin either
        continues the current one (started at a) or is a new bin (the prior). Under the
        model for T+1 intervals, M = 0..min(max_boundaries, T) equally likely, so that
        next_pmf(x) = P(data, x) / P(data), both under that model. Returns (weights for
        a = 0..T-1, weight of a new bin)."""
        T, f, col = self.T, self._fwd, self._last_col
        if T == 0:
            return np.zeros(0), 1.0
        Mn = min(self.max_boundaries, T)
        cont = np.full((Mn + 1, T), _NEG_INF)
        new = np.full(Mn + 1, _NEG_INF)
        for m in range(Mn + 1):
            lb = -_log_binom(T, m)
            if m == 0:
                cont[0, 0] = lb + f[0, T - 1]
                continue
            cont[m, 1:] = lb + f[m - 1, :T - 1] + col[1:T]
            new[m] = lb + f[m - 1, T - 1]
        with np.errstate(divide="ignore"):
            wa = logsumexp(cont, axis=0)
            wn = logsumexp(new)
            z = logsumexp(np.append(wa, wn))
        return np.exp(wa - z), float(np.exp(wn - z))

    def _next_components(self, x, size):
        """pmf and cdf of x under each component (rows: a = 0..T-1, then the new bin)."""
        x = np.atleast_1d(np.asarray(x, float))
        p, q = self.prior
        a = np.arange(self.T)
        u, v = self._stats(a, self.T - 1) if self.T else (np.zeros(0), np.zeros(0))
        A, B = np.append(u + p, p)[:, None], np.append(v + q, q)[:, None]
        if self.kind == "bernoulli":
            d = betabinom(size, A, B)
        else:  # the Gamma-Poisson predictive: negative binomial
            d = nbinom(A, B / (B + size))
        return d.pmf(x[None, :]), d.cdf(x[None, :])

    def next_pmf(self, x, size: float = 1.0) -> np.ndarray:
        """P(the next interval's count = x | data so far): Bernoulli, x events among `size`
        trials (the probability of the count, so with the binomial coefficient C(size, x);
        the batch evidences are of one sequence of trials, without it); Poisson, x events
        over exposure `size`."""
        wa, wn = self._next_mixture()
        pmf, _ = self._next_components(x, size)
        return np.append(wa, wn) @ pmf

    def next_cdf(self, x, size: float = 1.0) -> np.ndarray:
        """P(the next interval's count <= x | data so far) (see next_pmf)."""
        wa, wn = self._next_mixture()
        _, cdf = self._next_components(x, size)
        return np.append(wa, wn) @ cdf

    # --- everything, from the data so far --------------------------------------------

    def model(self):
        """The batch model of the data so far."""
        n = self.T
        if self.kind == "bernoulli":
            return BernoulliModel(self._x1[:n].copy(), self._x2[:n].copy(), *self.prior)
        return PoissonModel(self._x1[:n].copy(), *self.prior, e=self._x2[:n].copy())

    def fit(self, **kw) -> BinningResult:
        """The batch fit of the data so far (every interval's rate given all of it:
        the backward programme included), with this max_boundaries."""
        kw.setdefault("max_boundaries", self.max_boundaries)
        kw.setdefault("min_m_posterior", self.min_m_posterior)
        return fit(self.model(), **kw)
