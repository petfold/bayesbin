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
falls below `prune` are dropped. (Dropping them is safe in practice, not
guaranteed: a dropped run cannot come back. prune=0 keeps every run and is exact.)

In a long quiet stretch every run length since the last change stays plausible,
so the state would grow with the segment. It is bounded instead by merging old
run lengths: those longer than `exact_recent` fall into buckets of equal width in
log length (`merge_bins` per octave, as the geometric cascade does with time), and
the runs in one bucket become one component, the Gamma (or Beta) posterior with
the same rate mean and variance as their mixture. Neighbouring old runs differ by
about 1/√ℓ posterior sd, so the error is small, and the state grows with the log
of the segment's length: exact_recent + merge_bins·log2(ℓ / exact_recent)
components at most. Against keeping every run, with the default 32 per octave:
a 30,000-interval quiet stream kept 313 components instead of 30,000 (17x
faster), and the rate, its sd and the PIT moved by at most 2e-6, 3e-5 and 1e-6;
on counts of ~500 with a change every few hundred intervals, 3e-6, 3e-4, 1e-5
(16 per octave: 5e-4, 4e-2, 2e-3). merge_bins=None keeps every run.

    cp = ChangePointStream.poisson(alpha=1.0, beta=0.25, expected_run_length=200)
    for y in stream:
        q = cp.pit(y)               # before the update: the randomized PIT of y, uniform if calibrated
        cp.update(y)
        rate, sd = cp.rate_now()
        recent = cp.p_change_within(5)

Overdispersed counts (bursty: each event brings others, one story many reports) break
the Poisson segments' calibration. ChangePointStream.negbinomial gives each segment a
negative binomial likelihood of a known dispersion instead, still conjugate (a Beta
prior on its p, the same Gamma prior on the rate in the Poisson limit), and
ChangePointStream.overdispersed averages a bank of them over a grid of dispersions,
weighted by their marginal likelihoods (ChangePointMixture).

The hazard itself can be learnt: hazard_strength=a puts a Beta(a, a(L - 1)) prior on h
(mean 1/L, L = expected_run_length; a: the prior's weight in change points). The
recursion is then exact over the joint of the run length and the number of change
points so far, c (Wilson, Nassar & Gold 2010): the next interval starts a segment with
probability E[h | c] = (a + c)/(a + b + T - 1). Only a few dozen values of c stay
above `prune`, so it costs about twice the fixed hazard's per interval;
hazard_posterior() gives h's posterior mean and sd. On 5,000 intervals of segments of
mean length 40 a prior centred on 1/1000 (a = 1) found 0.0239 ± 0.0025.

When did a change happen? With lag=L, p_change_at(k) is P(a segment started at the k-th
latest interval | all data so far), k <= L: fixed-lag smoothing. The recursion is linear
in the state, so the probability of the paths with a start at s follows the same update;
it is kept as each entry's share of the state's probability, which a continuing run keeps
and a new run takes as the posterior mean of the shares before it (exact against
enumeration, merging and pruning included). About twice the cost per update.

Probabilities of counts are of the counts themselves: Bernoulli ones include the
binomial coefficient (the batch evidences of bayesbin.core are of one sequence of
trials, without it); Poisson ones include 1/y!.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import betainc, betaln, gammaln, xlogy

_NEG_INF = -np.inf
_ZERO, _ZERO_INT = np.zeros(1), np.zeros(1, np.int64)
_NEGLIGIBLE = 1e-16  # predictive components lighter than this (of the total) are left out of cdfs
_TERMS = 64  # cdfs up to counts below this sum the pmf's terms (by their ratios)
# the dispersions an overdispersed stream averages over (the negative binomial size per unit
# exposure; inf: Poisson), as Worldwatch's Layer-0 count model's grid
DISPERSIONS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, np.inf)


def _lse(a: np.ndarray, axis: int | None = None):
    """log Σ exp(a), plain NumPy (scipy's version costs ~0.3 ms a call on small arrays)."""
    if axis is None:  # the common case, in as few NumPy calls as possible
        top = float(a.max())
        return top + math.log(float(np.exp(a - top).sum())) if math.isfinite(top) else top
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis)


class ChangePointStream:
    """Bayesian online change-point detection with conjugate segment rates. Make one
    with ChangePointStream.poisson(...) or ChangePointStream.bernoulli(...). Options (as
    keywords of either): prune, max_runs, merge_bins, exact_recent (the state, above),
    hazard_strength (a learnt hazard: None keeps it fixed at 1/expected_run_length) and lag
    (fixed-lag smoothing: p_change_at(k) for the last `lag` intervals)."""

    def __init__(self, kind: str, prior: tuple[float, float], expected_run_length: float,
                 prune: float = 1e-12, max_runs: int | None = None, merge_bins: int | None = 32,
                 exact_recent: int = 128, dispersion: float | None = None,
                 hazard_strength: float | None = None, lag: int | None = None):
        if prior[0] <= 0 or prior[1] <= 0:
            raise ValueError("the prior's parameters must be positive")
        if not expected_run_length >= 1:
            raise ValueError("expected_run_length must be >= 1")
        if not 0 <= prune < 1:
            raise ValueError("prune must be in [0, 1)")
        if merge_bins is not None and merge_bins < 1:
            raise ValueError("merge_bins must be >= 1 (or None: no merging)")
        if kind == "negbin" and not (dispersion is not None and 0 < dispersion < np.inf):
            raise ValueError("negbin: a finite, positive dispersion")
        if hazard_strength is not None and not (hazard_strength > 0 and expected_run_length > 1):
            raise ValueError("hazard_strength must be positive (and expected_run_length > 1)")
        if lag is not None and not (isinstance(lag, int) and lag >= 1):
            raise ValueError("lag must be a positive integer (or None)")
        self.kind, self.prior, self.prune, self.max_runs = kind, prior, prune, max_runs
        self.dispersion = dispersion
        self.merge_bins, self.exact_recent = merge_bins, exact_recent
        self.hazard = 1.0 / expected_run_length
        self._log_h = np.log(self.hazard)
        self._log_1mh = np.log1p(-self.hazard) if self.hazard < 1 else _NEG_INF
        self._log_h_arr = np.array([self._log_h])
        self._log_prune = np.log(prune) if prune > 0 else _NEG_INF
        # a hazard learnt from the data: h ~ Beta(a, b) with mean 1/expected_run_length and the
        # weight of `hazard_strength` change points; the state then also holds the number of
        # change points so far (c, from _c0 on: the columns of _p2)
        self.hazard_strength = hazard_strength
        self._learn = hazard_strength is not None
        if self._learn:
            self._ha = float(hazard_strength)
            self._hb = self._ha * (expected_run_length - 1.0)
        # P(run, c | data), as probabilities (pruning keeps them far from underflow; no exp
        # per entry): rows as _logp, which is the log of its row sums
        self._p2 = np.zeros((0, 1))
        self._c0 = 0
        self._joint = None  # (T, P(change | c), P(none | c), P(c)): _components for _add_joint
        # fixed-lag smoothing: for each of the last `lag` intervals s, the share of each state
        # entry's probability that comes from paths in which a segment starts at s (a share is
        # kept by a continuing run; a new run's is the posterior mean of the shares before it)
        self.lag = lag
        self._q = None  # (lags, runs) or, with a learnt hazard, (lags, runs, c); the oldest first
        self.T = 0
        self.log_marginal = 0.0  # log P(all counts so far): the sum of the log predictives
        # the runs: log posterior, the range of lengths each stands for (lo == hi unless merged),
        # and totals (s, g) or (y, e); empty before any data
        self._logp = np.zeros(0)
        self._lo = np.zeros(0, np.int64)
        self._hi = np.zeros(0, np.int64)
        self._u = np.zeros(0)
        self._v = np.zeros(0)
        self._cache = None  # (T, x, size, per-component log pmf of x): pit() then update() on x
        self._comp = None  # (T, the components): pit() then update()

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

    @classmethod
    def negbinomial(cls, alpha: float, beta: float, dispersion: float, expected_run_length: float,
                    **kw) -> ChangePointStream:
        """Per interval, a count y over exposure e (default 1), negative binomial with mean λe
        and size dispersion·e (variance λe (1 + λ/dispersion)): Poisson counts whose rate
        also varies from interval to interval. Conjugate: y ~ NB(r e, p), p ~ Beta(a, b) per
        segment, with a = βr + α + 2 and b = α + α(1 + α)/(βr), which give λ = r(1 - p)/p the
        mean α/β and variance α/β² of the Gamma(alpha, beta) prior of poisson() for every r,
        and that prior itself as r grows. (Matching the mean alone leaves λ's prior variance
        infinite for small r: the prior predictive's tail gets far too heavy.)"""
        if not (0 < dispersion < np.inf):
            raise ValueError("dispersion must be finite and positive (np.inf: use poisson())")
        r = float(dispersion)
        a_ = beta * r + alpha + 2.0
        b_ = alpha + alpha * (1.0 + alpha) / (beta * r)
        return cls("negbin", (a_, b_), expected_run_length, dispersion=r, **kw)

    @classmethod
    def overdispersed(cls, alpha: float, beta: float, expected_run_length: float,
                      dispersions=DISPERSIONS, **kw) -> ChangePointMixture:
        """Counts of unknown overdispersion: a ChangePointMixture of negbinomial streams (and
        poisson for np.inf) over `dispersions`, weighted by their marginal likelihoods."""
        return ChangePointMixture(alpha, beta, expected_run_length, dispersions, **kw)

    # --- the predictive of one interval, for every run and for a new segment ------------

    def _components(self):
        """(log weights, u, v): the next interval's bin continues each run (weight
        P(run) (1 - h)) or starts a new segment (weight h, the prior: u = v = 0). With a learnt
        hazard, h given each number of change points so far, summed over them."""
        if self._comp is not None and self._comp[0] == self.T:
            return self._comp[1]
        if self.T == 0:
            comp = np.zeros(1), np.zeros(1), np.zeros(1)
        else:
            if self._learn:
                h, h1 = self._hazards()
                col = self._p2.sum(axis=0)  # P(c | data)
                self._joint = (self.T, h, h1, col)
                with np.errstate(divide="ignore"):
                    lw = np.log(np.append(self._p2 @ h1, col @ h))
            else:
                lw = np.concatenate((self._logp + self._log_1mh, self._log_h_arr))
            comp = lw, np.concatenate((self._u, _ZERO)), np.concatenate((self._v, _ZERO))
        self._comp = (self.T, comp)
        return comp

    def _hazards(self):
        """P(a change next | c change points so far) and P(none), for the columns c of the
        state: h | c ~ Beta(a + c, b + T - 1 - c) after T intervals (T - 1 transitions)."""
        c = self._c0 + np.arange(self._p2.shape[1])
        n = self._ha + self._hb + self.T - 1
        return (self._ha + c) / n, (self._hb + (self.T - 1) - c) / n

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
        e = float(size)
        if self.kind == "negbin":  # beta-negative-binomial: NB(r e, p), p ~ Beta(A, B)
            n = self.dispersion * e
            if n == 0:  # no exposure: no events
                return np.where(x == 0, 0.0, _NEG_INF) + 0.0 * A
            lp = (gammaln(n + x) - gammaln(n) - gammaln(x + 1)
                  + betaln(A + n, B + x) - betaln(A, B))
            return np.where((x >= 0) & (x == np.floor(x)), lp, _NEG_INF)
        # Gamma-Poisson: negative binomial over exposure e (log1p: log(B / (B + e)) loses
        # ~1e-16 B/e of its value, which A multiplies)
        lp = (gammaln(x + A) - gammaln(A) - gammaln(x + 1)
              - A * np.log1p(e / B) + xlogy(x, e / (B + e)))
        return np.where((x >= 0) & (x == np.floor(x)), lp, _NEG_INF)

    def _logpmf1(self, x: float, size: float, u, v) -> np.ndarray:
        """_logpmf for one count: a 1-D array over the segments, by the same arithmetic in
        fewer NumPy calls (one interval at a time is the common case)."""
        p, q = self.prior
        A, B = u + p, v + q
        if not (x >= 0 and float(x).is_integer() and (self.kind != "bernoulli" or x <= size)):
            return np.full(len(A), _NEG_INF)
        if self.kind == "bernoulli":
            n = size
            return (gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1)
                    + betaln(x + A, n - x + B) - betaln(A, B))
        e = size
        if self.kind == "negbin":
            n = self.dispersion * e
            if n == 0:
                return np.full(len(A), 0.0 if x == 0 else _NEG_INF)
            return (gammaln(n + x) - gammaln(n) - gammaln(x + 1)
                    + betaln(A + n, B + x) - betaln(A, B))
        if x == 0:  # the general formula's other terms are exactly 0 here
            return -A * np.log1p(e / B)
        return (gammaln(x + A) - gammaln(A) - gammaln(x + 1)
                - A * np.log1p(e / B) + xlogy(x, e / (B + e)))

    def _component_logpmf(self, x: float, size: float, lw, u, v) -> np.ndarray:
        """Each component's log pmf of the single count x, kept for update(x) after pit(x)."""
        c = self._cache
        if c is not None and c[0] == self.T and c[1] == x and c[2] == size:
            return c[3]
        lp = self._logpmf1(x, size, u, v)
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
        k = np.floor(np.atleast_1d(np.asarray(x, float)))
        top = int(max(k.max(), -1))
        if top < 0:
            return np.zeros(len(k))
        lw, u, v = self._components()
        w = np.exp(lw - _lse(lw))
        big = w > _NEGLIGIBLE  # the rest change the cdf by < 1e-16 each
        w, A, B = w[big], u[big] + self.prior[0], v[big] + self.prior[1]
        if self.kind != "poisson" and top < _TERMS:  # (for Poisson betainc is as fast)
            t = self._pmf_terms(top, size, A, B)
            if t is not None:
                c = np.cumsum(t, axis=1)
                return np.where(k < 0, 0.0, w @ c[:, np.clip(k, 0, c.shape[1] - 1).astype(int)])
        if self.kind == "poisson":  # negative binomial cdf in closed form: I_p(A, k + 1)
            c = betainc(A[:, None], np.maximum(k, 0)[None, :] + 1, (B / (B + size))[:, None])
            return np.where(k < 0, 0.0, w @ c)
        c = np.cumsum(self.next_pmf(np.arange(top + 1), size))
        return np.where(k < 0, 0.0, c[np.clip(k, 0, top).astype(int)])

    def _pmf_terms(self, top: int, size: float, A, B):
        """P(count = 0..top | each segment) of a Beta posterior (negbin, Bernoulli): the first
        term, then the ratios of successive ones (no betaln per count). None if a first term
        is below ~1e-304: the cdf then sums the predictive pmf."""
        j = np.arange(top, dtype=float)
        if self.kind == "negbin":  # beta-negative-binomial: NB(n, p), p ~ Beta(A, B)
            n = self.dispersion * size
            log0 = betaln(A + n, B) - betaln(A, B)
            ratio = (n + j) * (B[:, None] + j) / ((j + 1) * ((A + B + n)[:, None] + j))
        else:  # beta-binomial over n trials: none beyond n
            if not float(size).is_integer():
                return None
            n = int(size)
            top = min(top, n)
            j = j[:top]
            log0 = betaln(A, B + n) - betaln(A, B)
            ratio = (n - j) * (A[:, None] + j) / ((j + 1) * ((B + n - 1)[:, None] - j))
        if not log0.min() > -700:
            return None
        t = np.empty((len(A), top + 1))
        t[:, 0] = np.exp(log0)
        np.cumprod(ratio, axis=1, out=t[:, 1:])
        t[:, 1:] *= t[:, :1]
        return t

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
        if self.kind == "bernoulli" and size is None:
            raise ValueError("Bernoulli: update(s, n)")
        if np.isscalar(x) and (size is None or np.isscalar(size)):  # one interval
            xs, ss = [float(x)], [1.0 if size is None else float(size)]
        else:
            xs = np.atleast_1d(np.asarray(x, float))
            ss = np.broadcast_to(np.asarray(1.0 if size is None else size, float), xs.shape)
        for xi, si in zip(xs, ss):  # every interval checked before any is added
            if not (xi >= 0 and si >= 0 and float(xi).is_integer()):
                raise ValueError("counts must be non-negative integers (and sizes non-negative)")
            if self.kind == "bernoulli" and xi > si:
                raise ValueError("more events than trials")
            if self.kind != "bernoulli" and si == 0 and xi > 0:
                raise ValueError("an interval with zero exposure cannot have events")
        for xi, si in zip(xs, ss):
            self._add(float(xi), float(si))

    def _add(self, x: float, size: float) -> None:
        lw, u, v = self._components()
        lpmf = self._component_logpmf(x, size, lw, u, v)
        if self._learn:
            lp = self._add_joint(lpmf)
        else:
            if self._q is not None:  # a new run's share: the mean of the shares before it
                self._q = np.concatenate((self._q, (self._q @ np.exp(self._logp))[:, None]), axis=1)
            lp = lw + lpmf
            step = _lse(lp)  # log P(x | data so far)
            self.log_marginal += step
            lp -= step
        if self.kind == "bernoulli":
            du, dv = x, size - x
        elif self.kind == "negbin":  # the Beta posterior of p: A += r e, B += y
            du, dv = self.dispersion * size, x
        else:
            du, dv = x, size
        # the runs extended by one, then the new segment (length 1): the state is kept in
        # decreasing run length
        lo = np.concatenate((self._lo, _ZERO_INT)) + 1
        hi = np.concatenate((self._hi, _ZERO_INT)) + 1
        u, v = u + du, v + dv
        keep = lp >= self._log_prune if self.prune > 0 else np.isfinite(lp)
        new_kept = True  # the new run (the last)
        if keep.all() and (self.max_runs is None or len(lp) <= self.max_runs):
            self._logp, self._lo, self._hi, self._u, self._v = lp, lo, hi, u, v
        else:
            keep[np.argmax(lp)] = True
            if self.max_runs is not None and keep.sum() > self.max_runs:
                keep = np.zeros_like(keep)
                keep[np.argsort(lp)[-self.max_runs:]] = True
            lp = lp[keep]
            z = _lse(lp)
            self._logp = lp - z  # renormalised after pruning
            self._lo, self._hi, self._u, self._v = lo[keep], hi[keep], u[keep], v[keep]
            if self._learn:
                self._p2 = self._p2[keep] * math.exp(-z)
            if self._q is not None:
                self._q = self._q[:, keep]
            new_kept = bool(keep[-1])
        if self.merge_bins is not None:
            self._merge()
        if self.lag is not None:
            self._start_share(new_kept)
        self.T += 1

    def _start_share(self, new_kept: bool) -> None:
        """The shares of the interval just added: all of the new run's paths start a segment
        there, no other's. The oldest are dropped beyond `lag`."""
        shape = self._p2.shape if self._learn else self._logp.shape
        q = np.zeros((1, *shape))
        if new_kept:
            q[0, -1] = 1.0
        self._q = q if self._q is None else np.concatenate((self._q, q))[-self.lag:]

    def _add_joint(self, lpmf: np.ndarray) -> np.ndarray:
        """The joint posterior of (run, number of change points) after one more interval, each
        run's log pmf of it (and the prior's, last) given: continuing a run keeps c, a change
        adds one. Entries below `prune` are dropped. Returns the runs' log posteriors."""
        if self.T == 0:
            self.log_marginal += float(lpmf[0])
            self._p2, self._c0 = np.ones((1, 1)), 0
            return np.zeros(1)
        n, m = self._p2.shape
        _, h, h1, col = self._joint  # from _components, at this T
        if self._q is not None:  # shares: kept by continuing runs; a new run's is the mean
            Q = np.zeros((len(self._q), n + 1, m + 1))
            Q[:, :n, :m] = self._q
            with np.errstate(invalid="ignore", divide="ignore"):
                Q[:, n, 1:] = np.nan_to_num((self._q * self._p2).sum(axis=1) / col)
            self._q = Q
        top = float(lpmf.max())
        f = np.exp(lpmf - top)  # each run's pmf of the count (and the prior's), scaled
        P = np.zeros((n + 1, m + 1))
        np.multiply(self._p2 * h1, f[:n, None], out=P[:n, :m])
        P[n, 1:] = col * h * f[n]
        z = float(P.sum())
        self.log_marginal += top + math.log(z)  # log P(x | data so far)
        P /= z
        small = P < self.prune
        if small.any():
            small.flat[np.argmax(P)] = False
            P[small] = 0.0
            P /= P.sum()  # renormalised after pruning
        if not (P[:, 0].any() and P[:, -1].any()):  # no mass left at an end of c's range
            cols = np.flatnonzero(P.any(axis=0))
            P = P[:, cols[0]:cols[-1] + 1]
            if self._q is not None:
                self._q = self._q[:, :, cols[0]:cols[-1] + 1]
            self._c0 += int(cols[0])
        self._p2 = P
        with np.errstate(divide="ignore"):
            return np.log(P.sum(axis=1))

    def _merge(self) -> None:
        """Merge the runs longer than exact_recent that share a bucket of log length. The state
        is in decreasing length, so these runs come first and a bucket's are neighbours."""
        n = int(np.count_nonzero(self._lo > self.exact_recent))
        if n < 2:
            return
        # a component's bucket is that of its longest run, which ages by one each interval
        # (by its shortest, a merged component would stay put and absorb every run after it)
        bucket = np.floor(np.log2(self._hi[:n]) * self.merge_bins)
        same = np.flatnonzero(bucket[1:] == bucket[:-1])  # i and i + 1 share a bucket
        if not len(same):
            return
        groups: list[list[int]] = []  # [first, end) of each bucket with more than one
        for i in same.tolist():
            if groups and groups[-1][1] == i + 1:
                groups[-1][1] = i + 2
            else:
                groups.append([i, i + 2])
        keep = None
        for a, b in groups:
            merged = self._moment_match(a, b)
            if merged is None:
                continue
            # the bucket's first (longest) component becomes the merged one
            if self._q is not None:  # its shares: the members', weighted by their probabilities
                if self._learn:
                    pa = self._p2[a:b]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        self._q[:, a] = np.nan_to_num((self._q[:, a:b] * pa).sum(axis=1) / pa.sum(axis=0))
                else:
                    w = np.exp(self._logp[a:b] - self._logp[a:b].max())
                    self._q[:, a] = self._q[:, a:b] @ (w / w.sum())
            self._logp[a], self._u[a], self._v[a] = merged
            self._lo[a] = self._lo[b - 1]
            if self._learn:
                self._p2[a] = self._p2[a:b].sum(axis=0)
            if keep is None:
                keep = np.ones(len(self._logp), bool)
            keep[a + 1:b] = False
        if keep is not None:
            self._logp, self._lo, self._hi = self._logp[keep], self._lo[keep], self._hi[keep]
            self._u, self._v = self._u[keep], self._v[keep]
            if self._learn:
                self._p2 = self._p2[keep]
            if self._q is not None:
                self._q = self._q[:, keep]

    def _moment_match(self, a: int, b: int):
        """(log p, u, v) of one component for the runs a..b-1: their total probability and the
        conjugate posterior with the mixture's rate mean and variance (variance in the two-pass
        form); None if no valid one exists (a Beta mixture too spread out). In Python floats,
        as a bucket holds a few runs."""
        lw = self._logp[a:b].tolist()
        top = max(lw)
        w = [math.exp(x - top) for x in lw]
        W = math.fsum(w)
        p, q = self.prior
        A = [x + p for x in self._u[a:b].tolist()]
        B = [x + q for x in self._v[a:b].tolist()]
        beta_family = self.kind in ("bernoulli", "negbin")  # a Beta posterior (of f, or of p)
        if beta_family:
            m = [x / (x + y) for x, y in zip(A, B)]
            s = [x * y / ((x + y) ** 2 * (x + y + 1)) for x, y in zip(A, B)]
        else:
            m = [x / y for x, y in zip(A, B)]
            s = [x / y**2 for x, y in zip(A, B)]
        M = math.fsum(wi * mi for wi, mi in zip(w, m)) / W
        V = math.fsum(wi * (si + (mi - M) ** 2) for wi, si, mi in zip(w, s, m)) / W
        if not V > 0:
            return None
        if beta_family:
            n_ = M * (1 - M) / V - 1
            if not n_ > 0:
                return None
            A_, B_ = M * n_, (1 - M) * n_
        else:
            A_, B_ = M * M / V, M / V
        return top + math.log(W), A_ - p, B_ - q

    # --- the state -----------------------------------------------------------------

    def run_length_posterior(self) -> tuple[np.ndarray, np.ndarray]:
        """(lengths, probabilities): P(the current segment is ℓ intervals long | data), for the
        run lengths kept, in increasing length. A merged component's probability is spread
        evenly over the lengths it stands for (exact up to exact_recent)."""
        lo, hi, p = self.run_length_ranges()
        n = hi - lo + 1
        lengths = np.concatenate([np.arange(a, b + 1) for a, b in zip(lo, hi)]) if len(lo) else lo
        return lengths, np.repeat(p / n, n)

    def run_length_ranges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(lo, hi, probabilities): the state as kept, one row per component, which stands for
        the run lengths lo..hi (lo == hi unless merged), in increasing length."""
        o = np.argsort(self._lo)
        return self._lo[o].copy(), self._hi[o].copy(), np.exp(self._logp[o])

    def p_change_within(self, k: int) -> float:
        """P(a new segment started within the last k intervals | data) = P(run length <= k)
        (the first interval counts as a start: for k >= T this is 1). Exact for k up to
        exact_recent; a merged component counts in proportion to its lengths up to k."""
        p = np.exp(self._logp)
        frac = np.clip((k - self._lo + 1) / (self._hi - self._lo + 1), 0.0, 1.0)
        return float(p @ frac)

    def p_change_at(self, k: int) -> float:
        """P(a segment started at the k-th latest interval | data so far), k = 1..lag (k = 1:
        the latest, P(run length = 1)): fixed-lag smoothing, which needs lag >= k. Later
        intervals' data count too, so it sharpens as they arrive. The first interval always
        starts one."""
        if self.lag is None or not 1 <= k <= self.lag:
            raise ValueError(f"k must be in 1..lag (lag={self.lag})")
        if k > self.T:
            raise ValueError(f"only {self.T} intervals so far")
        q = self._q[len(self._q) - k]
        if self._learn:
            return float((q * self._p2).sum() / self._p2.sum())
        return float(q @ np.exp(self._logp))

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
        elif self.kind == "negbin":  # λ = r (1 - p) / p, p ~ Beta(A, B)
            r = self.dispersion
            m1 = r * B / (A - 1)
            within = r * r * B * (A + B - 1) / ((A - 1) ** 2 * (A - 2))
        else:
            m1 = A / B
            within = A / B**2
        mean = float(p @ m1)
        return mean, float(np.sqrt(p @ within + p @ (m1 - mean) ** 2))

    def hazard_posterior(self) -> tuple[float, float]:
        """The hazard's posterior mean and sd given the data so far: with hazard_strength set,
        a mixture of Beta(a + c, b + T - 1 - c) over the number of change points c; else the
        fixed hazard and 0. (1 / the mean is the expected segment length.)"""
        if not self._learn:
            return self.hazard, 0.0
        a, b = self._ha, self._hb
        if self.T == 0:
            return a / (a + b), math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        w = self._p2.sum(axis=0)
        w = w / w.sum()
        c = self._c0 + np.arange(len(w))
        A, B = a + c, b + (self.T - 1) - c
        mean = float(w @ (A / (A + B)))
        second = float(w @ (A * (A + 1) / ((A + B) * (A + B + 1))))
        return mean, math.sqrt(max(second - mean * mean, 0.0))

    @property
    def n_runs(self) -> int:
        """How many components the state keeps (the cost per interval)."""
        return len(self._logp)


class ChangePointMixture:
    """Change points in counts of unknown overdispersion: one ChangePointStream per dispersion
    (negbinomial, and poisson for np.inf), all fed the same counts, averaged with weights
    proportional to their marginal likelihoods (a uniform prior over the grid). Same interface
    as ChangePointStream for Poisson-type counts; dispersion_posterior() gives the weights."""

    def __init__(self, alpha: float, beta: float, expected_run_length: float,
                 dispersions=DISPERSIONS, **kw):
        self.dispersions = tuple(float(r) for r in dispersions)
        self.members = [ChangePointStream.poisson(alpha, beta, expected_run_length, **kw) if r == np.inf
                        else ChangePointStream.negbinomial(alpha, beta, r, expected_run_length, **kw)
                        for r in self.dispersions]
        self.kind, self.prior = "mixture", (alpha, beta)

    @property
    def T(self) -> int:
        return self.members[0].T

    @property
    def hazard(self) -> float:
        return self.members[0].hazard

    def _weights(self) -> np.ndarray:
        lm = np.array([m.log_marginal for m in self.members])
        return np.exp(lm - _lse(lm))

    def dispersion_posterior(self) -> tuple[np.ndarray, np.ndarray]:
        """(dispersions, P(dispersion | data so far))."""
        return np.array(self.dispersions), self._weights()

    @property
    def log_marginal(self) -> float:
        """log P(all counts so far), the prior over the grid included."""
        lm = np.array([m.log_marginal for m in self.members])
        return float(_lse(lm) - np.log(len(lm)))

    def update(self, x, size=None) -> None:
        for m in self.members:
            m.update(x, size)

    def next_pmf(self, x, size: float | None = None) -> np.ndarray:
        return self._weights() @ np.array([m.next_pmf(x, size) for m in self.members])

    def next_logpmf(self, x, size: float | None = None) -> np.ndarray:
        return np.log(self.next_pmf(x, size))

    def next_cdf(self, x, size: float | None = None) -> np.ndarray:
        return self._weights() @ np.array([m.next_cdf(x, size) for m in self.members])

    def pit(self, x: float, size: float | None = None, u: float | None = None,
            rng: np.random.Generator | None = None) -> float:
        """The randomized PIT of a count x before it is added (see ChangePointStream.pit)."""
        if u is None:
            u = (rng or np.random.default_rng()).random()
        below = float(self.next_cdf(x - 1, size)[0]) if x >= 1 else 0.0
        return below + u * float(self.next_pmf(x, size)[0])

    def rate_now(self) -> tuple[float, float]:
        """The rate now and its standard deviation, averaged over the dispersions."""
        w = self._weights()
        ms = np.array([m.rate_now() for m in self.members])
        mean = float(w @ ms[:, 0])
        return mean, float(np.sqrt(w @ (ms[:, 1] ** 2 + (ms[:, 0] - mean) ** 2)))

    def p_change_within(self, k: int) -> float:
        return float(self._weights() @ np.array([m.p_change_within(k) for m in self.members]))

    def p_change_at(self, k: int) -> float:
        """P(a segment started at the k-th latest interval | data so far), averaged over the
        dispersions (needs lag; see ChangePointStream.p_change_at)."""
        return float(self._weights() @ np.array([m.p_change_at(k) for m in self.members]))

    def hazard_posterior(self) -> tuple[float, float]:
        """The hazard's posterior mean and sd, averaged over the dispersions."""
        w = self._weights()
        hs = np.array([m.hazard_posterior() for m in self.members])
        mean = float(w @ hs[:, 0])
        return mean, float(np.sqrt(max(w @ (hs[:, 1] ** 2 + hs[:, 0] ** 2) - mean**2, 0.0)))

    @property
    def n_runs(self) -> int:
        return sum(m.n_runs for m in self.members)
