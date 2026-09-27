"""Change points in a stream without end: Bayesian online change-point detection.

The batch model's prior (every number of boundaries up to a maximum equally
likely, every placement equally likely) suits a fixed record. A stream needs a
prior that does not depend on how long it will run: here, as in Adams & MacKay
(2007) and Fearnhead & Liu (2007), each interval starts a new segment with a
constant probability h (the "hazard"; 1/h is the expected segment length), and
each segment's rate is drawn afresh from the same Beta or Gamma prior as in
BernoulliModel and PoissonModel.

The state is the posterior over the current run length (how many intervals
since the last change) with each run's sufficient statistics. A new interval
either extends a run, weighted by that run's predictive probability of the new
count, or starts a new one, with the prior predictive. Runs whose probability
falls below `prune` are dropped: the cost per interval is the number of
plausible run lengths, not the length of the stream. (Dropping them is safe in
practice, not guaranteed: a dropped run cannot come back. prune=0 keeps every run
and is exact.)

    cp = ChangePointStream.poisson(alpha=1.0, beta=0.25, expected_run_length=200)
    for y in stream:
        q = cp.pit(y)               # before the update: the randomized PIT of y, uniform if calibrated
        cp.update(y)
        rate, sd = cp.rate_now()
        recent = cp.p_change_within(5)

Probabilities of counts are of the counts themselves: Bernoulli ones include the
binomial coefficient (the batch evidences of bayesbin.core are of one sequence of
trials, without it); Poisson ones include 1/y!.
"""

from __future__ import annotations

import numpy as np
from scipy.special import betainc, betaln, gammaln, xlogy

_NEG_INF = -np.inf
_NEGLIGIBLE = 1e-16  # predictive components lighter than this (of the total) are left out of cdfs


def _lse(a: np.ndarray, axis: int | None = None):
    """log Σ exp(a), plain NumPy (scipy's version costs ~0.3 ms a call on small arrays)."""
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return out.item() if axis is None else np.squeeze(out, axis)


class ChangePointStream:
    """Bayesian online change-point detection with conjugate segment rates. Make one
    with ChangePointStream.poisson(...) or ChangePointStream.bernoulli(...)."""

    def __init__(self, kind: str, prior: tuple[float, float], expected_run_length: float,
                 prune: float = 1e-12, max_runs: int | None = None):
        if prior[0] <= 0 or prior[1] <= 0:
            raise ValueError("the prior's parameters must be positive")
        if not expected_run_length >= 1:
            raise ValueError("expected_run_length must be >= 1")
        if not 0 <= prune < 1:
            raise ValueError("prune must be in [0, 1)")
        self.kind, self.prior, self.prune, self.max_runs = kind, prior, prune, max_runs
        self.hazard = 1.0 / expected_run_length
        self._log_h = np.log(self.hazard)
        self._log_1mh = np.log1p(-self.hazard) if self.hazard < 1 else _NEG_INF
        self.T = 0
        self.log_marginal = 0.0  # log P(all counts so far): the sum of the log predictives
        # the runs: log posterior, length, and totals (s, g) or (y, e); empty before any data
        self._logp = np.zeros(0)
        self._len = np.zeros(0, np.int64)
        self._u = np.zeros(0)
        self._v = np.zeros(0)
        self._cache = None  # (T, x, size, per-component log pmf of x): pit() then update() on x

    @classmethod
    def bernoulli(cls, expected_run_length: float, sigma: float = 1.0, gamma: float = 1.0,
                  **kw) -> ChangePointStream:
        """Per interval, s events among n trials; each segment's f ~ Beta(sigma, gamma). (The
        default prior is flat, as the rates of a stream are rarely known to be small.)"""
        return cls("bernoulli", (sigma, gamma), expected_run_length, **kw)

    @classmethod
    def poisson(cls, alpha: float, beta: float, expected_run_length: float,
                **kw) -> ChangePointStream:
        """Per interval, a count y over exposure e (default 1); each segment's λ ~ Gamma(alpha,
        beta) (shape, rate)."""
        return cls("poisson", (alpha, beta), expected_run_length, **kw)

    # --- the predictive of one interval, for every run and for a new segment ------------

    def _components(self):
        """(log weights, u, v): the next interval's bin continues each run (weight
        P(run) (1 - h)) or starts a new segment (weight h, the prior: u = v = 0)."""
        if self.T == 0:
            return np.zeros(1), np.zeros(1), np.zeros(1)
        return (np.append(self._logp + self._log_1mh, self._log_h),
                np.append(self._u, 0.0), np.append(self._v, 0.0))

    def _logpmf(self, x, size, u, v):
        """log P(x | a segment with totals u, v so far), for arrays u, v (rows) and x (columns)."""
        x = np.asarray(x, float)[None, :]
        p, q = self.prior
        A, B = (u + p)[:, None], (v + q)[:, None]
        if self.kind == "bernoulli":  # Beta-binomial: x events among `size` trials
            n = float(size)
            with np.errstate(invalid="ignore"):
                lp = (gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1)
                      + betaln(x + A, n - x + B) - betaln(A, B))
            return np.where((x >= 0) & (x <= n) & (x == np.floor(x)), lp, _NEG_INF)
        e = float(size)  # Gamma-Poisson: negative binomial over exposure e
        lp = (gammaln(x + A) - gammaln(A) - gammaln(x + 1)
              + A * np.log(B / (B + e)) + xlogy(x, e / (B + e)))
        return np.where((x >= 0) & (x == np.floor(x)), lp, _NEG_INF)

    def _component_logpmf(self, x: float, size: float, lw, u, v) -> np.ndarray:
        """Each component's log pmf of the single count x, kept for update(x) after pit(x)."""
        c = self._cache
        if c is not None and c[0] == self.T and c[1] == x and c[2] == size:
            return c[3]
        lp = self._logpmf([x], size, u, v)[:, 0]
        self._cache = (self.T, x, size, lp)
        return lp

    def next_logpmf(self, x, size: float | None = None) -> np.ndarray:
        """log P(the next interval's count = x | data so far). size: the number of trials
        (Bernoulli; required) or the exposure (Poisson; default 1)."""
        size = self._size(size)
        lw, u, v = self._components()
        x = np.atleast_1d(np.asarray(x, float))
        if len(x) == 1:
            return np.array([_lse(lw + self._component_logpmf(float(x[0]), size, lw, u, v))])
        return _lse(lw[:, None] + self._logpmf(x, size, u, v), axis=0)

    def next_pmf(self, x, size: float | None = None) -> np.ndarray:
        return np.exp(self.next_logpmf(x, size))

    def next_cdf(self, x, size: float | None = None) -> np.ndarray:
        """P(the next interval's count <= x | data so far)."""
        size = self._size(size)
        x = np.atleast_1d(np.asarray(x, float))
        if self.kind == "poisson":  # negative binomial cdf in closed form: I_p(A, k + 1)
            lw, u, v = self._components()
            w = np.exp(lw - _lse(lw))
            big = w > _NEGLIGIBLE  # the rest change the cdf by < 1e-16 each
            w, A, B = w[big], u[big] + self.prior[0], v[big] + self.prior[1]
            k = np.floor(x)
            c = betainc(A[:, None], np.maximum(k, 0)[None, :] + 1, (B / (B + size))[:, None])
            return np.where(k < 0, 0.0, w @ c)
        top = int(max(np.floor(x.max()), -1))
        if top < 0:
            return np.zeros(len(x))
        c = np.cumsum(self.next_pmf(np.arange(top + 1), size))
        idx = np.floor(x).astype(int)
        return np.where(idx < 0, 0.0, c[np.clip(idx, 0, top)])

    def pit(self, x: float, size: float | None = None, u: float | None = None,
            rng: np.random.Generator | None = None) -> float:
        """The randomized probability integral transform of a count x before it is added:
        P(count < x) + u P(count = x), u uniform in [0, 1). Uniform on [0, 1) when the model
        is calibrated, which a test can check; near 1: surprisingly high, near 0: low."""
        if u is None:
            u = (rng or np.random.default_rng()).random()
        below = float(self.next_cdf(x - 1, size)[0]) if x >= 1 else 0.0
        return below + u * float(self.next_pmf(x, size)[0])

    def _size(self, size):
        if self.kind == "bernoulli":
            if size is None:
                raise ValueError("Bernoulli: give the number of trials (size)")
            return float(size)
        return 1.0 if size is None else float(size)

    # --- adding data ---------------------------------------------------------------

    def update(self, x, size=None) -> None:
        """Add one interval or several (arrays), in order. Bernoulli: update(s, n), s events
        among n trials; Poisson: update(y) or update(y, e)."""
        x = np.atleast_1d(np.asarray(x, float))
        if self.kind == "bernoulli" and size is None:
            raise ValueError("Bernoulli: update(s, n)")
        size = np.broadcast_to(np.asarray(1.0 if size is None else size, float), x.shape)
        if (x < 0).any() or (size < 0).any() or (x != np.floor(x)).any():
            raise ValueError("counts must be non-negative integers (and sizes non-negative)")
        if self.kind == "bernoulli" and (x > size).any():
            raise ValueError("more events than trials")
        if self.kind == "poisson" and ((size == 0) & (x > 0)).any():
            raise ValueError("an interval with zero exposure cannot have events")
        for xi, si in zip(x, size):
            self._add(float(xi), float(si))

    def _add(self, x: float, size: float) -> None:
        lw, u, v = self._components()
        lp = lw + self._component_logpmf(x, size, lw, u, v)
        step = _lse(lp)  # log P(x | data so far)
        self.log_marginal += float(step)
        lp -= step
        du, dv = (x, size - x) if self.kind == "bernoulli" else (x, size)
        if self.T == 0:
            logp, ln, uu, vv = lp[-1:], np.ones(1, np.int64), np.array([du]), np.array([dv])
        else:  # the runs extended by one, then the new segment (length 1)
            logp = lp
            ln = np.append(self._len + 1, 1)
            uu = np.append(self._u + du, du)
            vv = np.append(self._v + dv, dv)
        keep = logp >= np.log(self.prune) if self.prune > 0 else np.isfinite(logp)
        keep[np.argmax(logp)] = True
        if self.max_runs is not None and keep.sum() > self.max_runs:
            keep = np.zeros_like(keep)
            keep[np.argsort(logp)[-self.max_runs:]] = True
        logp = logp[keep]
        self._logp = logp - _lse(logp)  # renormalised after pruning
        self._len, self._u, self._v = ln[keep], uu[keep], vv[keep]
        self.T += 1

    # --- the state -----------------------------------------------------------------

    def run_length_posterior(self) -> tuple[np.ndarray, np.ndarray]:
        """(lengths, probabilities): P(the current segment is ℓ intervals long | data), for the
        run lengths kept, in increasing length."""
        o = np.argsort(self._len)
        return self._len[o].copy(), np.exp(self._logp[o])

    def p_change_within(self, k: int) -> float:
        """P(a new segment started within the last k intervals | data) = P(run length <= k)
        (the first interval counts as a start: for k >= T this is 1)."""
        return float(np.exp(self._logp[self._len <= k]).sum())

    def rate_now(self) -> tuple[float, float]:
        """The rate in the latest interval given the data so far, and its standard deviation
        (within segments plus between run lengths: no cancellation)."""
        if self.T == 0:
            raise ValueError("no data yet")
        p = np.exp(self._logp)
        a, b = self.prior
        A, B = self._u + a, self._v + b
        if self.kind == "bernoulli":
            m1 = A / (A + B)
            within = A * B / ((A + B) ** 2 * (A + B + 1))
        else:
            m1 = A / B
            within = A / B**2
        mean = float(p @ m1)
        return mean, float(np.sqrt(p @ within + p @ (m1 - mean) ** 2))

    @property
    def n_runs(self) -> int:
        """How many run lengths the state keeps (the cost per interval)."""
        return len(self._logp)
