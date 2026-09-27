"""OnlineBinning against the batch fit of the same data."""

import numpy as np
import pytest
from scipy.special import gammaln

from bayesbin import BernoulliModel, OnlineBinning, PoissonModel, fit

M = 8


def _poisson_data():
    rng = np.random.default_rng(7)
    e = rng.uniform(0.5, 2.0, 150)
    y = rng.poisson(np.repeat([2.0, 7.0, 3.0], 50) * e).astype(float)
    return y, e


def _bernoulli_data():
    rng = np.random.default_rng(8)
    n = 20
    s = rng.binomial(n, np.repeat([0.05, 0.3, 0.1], 50)).astype(float)
    return s, n - s


def _online_and_batch(kind):
    """(OnlineBinning fed interval by interval, a function k -> the batch model of the first
    k intervals, the data)"""
    if kind == "poisson":
        y, e = _poisson_data()
        return (OnlineBinning.poisson(1.0, 0.3, max_boundaries=M),
                lambda k: PoissonModel(y[:k], 1.0, 0.3, e=e[:k]), (y, e))
    s, g = _bernoulli_data()
    return (OnlineBinning.bernoulli(max_boundaries=M, sigma=1.0, gamma=1.0),
            lambda k: BernoulliModel(s[:k], g[:k], 1.0, 1.0), (s, g))


@pytest.mark.parametrize("kind", ["poisson", "bernoulli"])
def test_online_state_equals_the_batch_fit_of_the_data_so_far(kind):
    ob, batch, (x1, x2) = _online_and_batch(kind)
    for k in range(len(x1)):
        ob.update(x1[k], x2[k])
        if ob.T in (1, 2, 3, 40, 51, 100, 150):
            r = fit(batch(ob.T), M, keep_bins=True)
            np.testing.assert_allclose(ob.log_evidence, r.log_evidence, rtol=1e-13, atol=1e-12)
            np.testing.assert_allclose(ob.m_posterior, r.m_posterior, rtol=1e-11, atol=1e-14)
            assert ob.log_marginal == pytest.approx(r.log_marginal, rel=1e-13)
            mean, sd = ob.rate_now()
            assert mean == pytest.approx(r.rate[-1], rel=1e-11)
            assert sd == pytest.approx(r.rate_std[-1], rel=1e-9)
            # P(the current bin starts at a) is the bin posterior of [a, T-1]
            np.testing.assert_allclose(ob.current_bin_start(), r.bin_posterior[:, -1], atol=1e-12)


@pytest.mark.parametrize("kind", ["poisson", "bernoulli"])
def test_the_predictive_is_the_ratio_of_batch_marginals(kind):
    """next_pmf(x) = P(data, x) / P(data) under the model for T+1 intervals: its ratios are
    those of the batch marginal likelihoods of the data with x appended."""
    ob, batch, (x1, x2) = _online_and_batch(kind)
    for T in (0, 1, 60, 120):
        ob2, _, _ = _online_and_batch(kind)
        ob2.update(x1[:T], x2[:T])
        if kind == "poisson":
            size, xs = x2[T], np.arange(0, 12)
            models = [PoissonModel(np.append(x1[:T], x), 1.0, 0.3, e=np.append(x2[:T], size)) for x in xs]
            total = ob2.next_pmf(np.arange(400), size).sum()
            logc = np.zeros(len(xs))
        else:
            size = 20
            xs = np.arange(0, size + 1)
            models = [BernoulliModel(np.append(x1[:T], x), np.append(x2[:T], size - x), 1.0, 1.0) for x in xs]
            total = ob2.next_pmf(xs, size).sum()
            # the predictive is of the count: the binomial coefficient the sequence evidence lacks
            logc = gammaln(size + 1) - gammaln(xs + 1) - gammaln(size - xs + 1)
        assert total == pytest.approx(1.0, abs=1e-12)
        pmf = ob2.next_pmf(xs, size)
        lm = np.array([fit(m, M).log_marginal for m in models]) + logc
        np.testing.assert_allclose(np.log(pmf) - np.log(pmf[0]), lm - lm[0], atol=1e-10)
        cdf = ob2.next_cdf(xs, size)
        np.testing.assert_allclose(cdf, np.cumsum(pmf), atol=1e-12)


def test_updates_in_chunks_equal_updates_one_at_a_time():
    y, e = _poisson_data()
    one = OnlineBinning.poisson(1.0, 0.3, max_boundaries=M)
    for k in range(len(y)):
        one.update(y[k], e[k])
    chunks = OnlineBinning.poisson(1.0, 0.3, max_boundaries=M)
    for k0 in range(0, len(y), 37):
        chunks.update(y[k0:k0 + 37], e[k0:k0 + 37])
    np.testing.assert_array_equal(one.log_evidence, chunks.log_evidence)
    assert one.rate_now() == chunks.rate_now()


def test_fit_is_the_batch_fit_of_the_data_so_far():
    s, g = _bernoulli_data()
    ob = OnlineBinning.bernoulli(max_boundaries=M, sigma=1.0, gamma=1.0)
    ob.update(s, g)
    a, b = ob.fit(), fit(BernoulliModel(s, g, 1.0, 1.0), M)
    np.testing.assert_array_equal(a.rate, b.rate)
    np.testing.assert_array_equal(a.boundary_posterior, b.boundary_posterior)


def test_online_inputs_are_checked():
    with pytest.raises(ValueError):
        OnlineBinning.bernoulli().update([1.0])  # g missing
    with pytest.raises(ValueError):
        OnlineBinning.poisson(1.0, 1.0).update([1.0], [0.0])  # events without exposure
    with pytest.raises(ValueError):
        OnlineBinning.poisson(1.0, 1.0).update([-1.0])
    with pytest.raises(ValueError):
        OnlineBinning.poisson(0.0, 1.0)
    with pytest.raises(ValueError):
        OnlineBinning.poisson(1.0, 1.0).rate_now()  # no data yet
