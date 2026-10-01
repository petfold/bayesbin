"""best_prior: the prior on each bin's rate chosen by the evidence."""

import warnings
from dataclasses import replace

import numpy as np
import pytest

from bayesbin import BernoulliModel, PoissonModel, best_prior, fit, fit_cyclic


def _segments(seed, n_seg, length):
    rng = np.random.default_rng(seed)
    y = rng.poisson(np.repeat(rng.gamma(3.0, 2.0, n_seg), length)).astype(float)  # Gamma(3, rate 0.5)
    n = rng.integers(10, 30, n_seg * length)
    s = rng.binomial(n, np.repeat(rng.beta(2.0, 8.0, n_seg), length)).astype(float)
    return y, s, n


@pytest.mark.parametrize("kind", ["poisson", "bernoulli"])
def test_best_prior_maximises_the_evidence(kind):
    y, s, n = _segments(3, 8, 60)
    model = PoissonModel.weak_prior(y) if kind == "poisson" else BernoulliModel(s, n - s, 1.0, 32.0)
    best, lm = best_prior(model, 12)
    assert lm == pytest.approx(fit(best, 12).log_marginal, rel=1e-12)
    assert lm > fit(model, 12).log_marginal
    names = ("alpha", "beta") if kind == "poisson" else ("sigma", "gamma")
    for name in names:  # a local maximum: a 2% step either way is worse
        for f in (0.98, 1.02):
            other = replace(best, **{name: getattr(best, name) * f})
            assert fit(other, 12).log_marginal < lm


def test_best_prior_recovers_the_prior_the_rates_came_from():
    y, _, _ = _segments(4, 60, 40)
    best, _ = best_prior(PoissonModel.weak_prior(y), 80)
    assert best.alpha / best.beta == pytest.approx(6.0, rel=0.2)  # Gamma(3, 0.5): mean 6
    assert 1.5 < best.alpha < 6.0


def test_best_prior_for_a_cycle_and_at_a_bound():
    rng = np.random.default_rng(7)
    hour = np.arange(24)
    counts = rng.poisson(np.where((hour >= 21) | (hour < 3), 4.0, 1.0) * 30).astype(float)
    model = PoissonModel.weak_prior(counts, np.full(24, 30))
    best, lm = best_prior(model, 6, cyclic=True)
    assert lm == pytest.approx(fit_cyclic(best, 6).log_marginal, rel=1e-12)
    flat = PoissonModel.weak_prior(np.full(200, 3.0))  # no change at all: the best prior is a point
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        best_prior(flat, 5, bound=1e4)
    assert any("bound" in str(x.message) for x in w)
