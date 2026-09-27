"""ChangePointStream: exact against enumeration of every segmentation, calibrated on data
from the model, and not on overdispersed data."""

import itertools

import numpy as np
import pytest
from scipy.special import betaln, gammaln, logsumexp
from scipy.stats import kstest

from bayesbin import ChangePointStream


def _segment_logev(kind, x, n, prior):
    """log P(the counts of one segment), its rate integrated over the prior."""
    p, q = prior
    if kind == "poisson":
        Y, E = x.sum(), n.sum()
        return (p * np.log(q) - gammaln(p) + gammaln(Y + p) - (Y + p) * np.log(E + q)
                + np.sum(x * np.log(np.where(n > 0, n, 1.0))) - np.sum(gammaln(x + 1)))
    S, F = x.sum(), (n - x).sum()
    return (np.sum(gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1))
            + betaln(S + p, F + q) - betaln(p, q))


def _enumerate(kind, x, n, prior, h):
    """(log P(x), {run length: P}, E[rate now]) over all 2^(T-1) segmentations."""
    T = len(x)
    p, q = prior
    terms, runs, rates = [], {}, []
    for bits in itertools.product([0, 1], repeat=T - 1):
        starts = [0] + [i + 1 for i, b in enumerate(bits) if b]
        ends = starts[1:] + [T]
        lp = sum(bits) * np.log(h) + (T - 1 - sum(bits)) * np.log1p(-h)
        lp += sum(_segment_logev(kind, x[a:b], n[a:b], prior) for a, b in zip(starts, ends))
        terms.append(lp)
        runs.setdefault(T - starts[-1], []).append(lp)
        last = slice(starts[-1], T)
        A = x[last].sum() + p
        B = (n[last] - x[last]).sum() + q if kind == "bernoulli" else n[last].sum() + q
        rates.append(A / (A + B) if kind == "bernoulli" else A / B)
    z = logsumexp(terms)
    w = np.exp(np.array(terms) - z)
    return z, {k: float(np.exp(logsumexp(v) - z)) for k, v in runs.items()}, float(w @ np.array(rates))


def _small(kind, seed=0, T=9):
    rng = np.random.default_rng(seed)
    if kind == "poisson":
        n = rng.uniform(0.5, 2.0, T)
        x = rng.poisson(np.repeat([1.0, 6.0, 2.0], 3) * n).astype(float)
        return x, n, ChangePointStream.poisson(1.5, 0.5, expected_run_length=4, prune=0)
    n = rng.integers(5, 15, T).astype(float)
    x = rng.binomial(n.astype(int), np.repeat([0.1, 0.7, 0.3], 3)).astype(float)
    return x, n, ChangePointStream.bernoulli(4, sigma=1.0, gamma=2.0, prune=0)


@pytest.mark.parametrize("kind", ["poisson", "bernoulli"])
def test_stream_is_exact_against_enumerating_every_segmentation(kind):
    x, n, cp = _small(kind)
    for T in range(1, len(x) + 1):
        cp.update(x[T - 1], n[T - 1])
        z, runs, rate = _enumerate(kind, x[:T], n[:T], cp.prior, cp.hazard)
        assert cp.log_marginal == pytest.approx(z, rel=1e-13, abs=1e-12)
        lengths, probs = cp.run_length_posterior()
        assert {int(k) for k in lengths} == set(runs)
        for k, p in zip(lengths, probs):
            assert p == pytest.approx(runs[int(k)], abs=1e-13)
        assert cp.rate_now()[0] == pytest.approx(rate, rel=1e-12)


def _generate(rng, T, h, alpha, beta, overdispersed=False):
    lam, y = rng.gamma(alpha, 1 / beta), np.empty(T)
    for t in range(T):
        if t > 0 and rng.random() < h:
            lam = rng.gamma(alpha, 1 / beta)
        y[t] = rng.poisson(rng.gamma(0.5, 2 * lam) if overdispersed else lam)
    return y


def test_stream_pit_is_uniform_on_data_from_the_model_and_not_on_overdispersed_data():
    alpha, beta, run = 2.0, 0.4, 100
    for overdispersed, ok in ((False, True), (True, False)):
        rng = np.random.default_rng(11)
        y = _generate(rng, 3000, 1 / run, alpha, beta, overdispersed)
        cp = ChangePointStream.poisson(alpha, beta, expected_run_length=run)
        q = np.empty(len(y))
        for t, yt in enumerate(y):
            q[t] = cp.pit(yt, u=rng.random())
            cp.update(yt)
        p = kstest(q, "uniform").pvalue
        assert (p > 0.01) if ok else (p < 1e-10), (overdispersed, p)


def test_stream_bernoulli_pit_is_uniform_on_data_from_the_model():
    rng = np.random.default_rng(12)
    n, h = 30, 1 / 80
    f, q = rng.beta(1.0, 1.0), []
    cp = ChangePointStream.bernoulli(80, sigma=1.0, gamma=1.0)
    for t in range(3000):
        if t > 0 and rng.random() < h:
            f = rng.beta(1.0, 1.0)
        s = rng.binomial(n, f)
        q.append(cp.pit(s, n, u=rng.random()))
        cp.update(s, n)
    assert kstest(q, "uniform").pvalue > 0.01


def test_pruning_costs_almost_nothing():
    rng = np.random.default_rng(3)
    y = np.concatenate([rng.poisson(lam, 300) for lam in (3, 12, 5, 30, 8)]).astype(float)
    a = ChangePointStream.poisson(2.0, 0.4, expected_run_length=300)  # prune=1e-12
    b = ChangePointStream.poisson(2.0, 0.4, expected_run_length=300, prune=0)
    for yt in y:
        assert a.next_logpmf(yt)[0] == pytest.approx(b.next_logpmf(yt)[0], abs=1e-8)
        a.update(yt)
        b.update(yt)
        ra, rb = a.rate_now(), b.rate_now()
        assert ra[0] == pytest.approx(rb[0], rel=1e-7) and ra[1] == pytest.approx(rb[1], rel=1e-7)
    assert a.n_runs < b.n_runs / 3


def test_stream_predictive_pmf_and_cdf_agree():
    rng = np.random.default_rng(5)
    cp = ChangePointStream.poisson(2.0, 0.4, expected_run_length=50)
    cp.update(rng.poisson(5, 200), rng.uniform(0.5, 2.0, 200))
    pmf = cp.next_pmf(np.arange(400), 1.7)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(cp.next_cdf(np.arange(60), 1.7), np.cumsum(pmf[:60]), atol=1e-13)
    b = ChangePointStream.bernoulli(50)
    b.update(rng.binomial(20, 0.3, 100), 20)
    # betaln differences of values near -1300 after 100 intervals: ~1e-13 in each term
    assert b.next_pmf(np.arange(21), 20).sum() == pytest.approx(1.0, abs=1e-10)
    q = cp.pit(7, 1.7, u=0.0)
    assert q == pytest.approx(float(cp.next_cdf(6, 1.7)[0]))


def test_stream_updates_in_chunks_equal_updates_one_at_a_time():
    rng = np.random.default_rng(6)
    y, e = rng.poisson(4, 300).astype(float), rng.uniform(0.5, 2, 300)
    one = ChangePointStream.poisson(1.0, 0.25, expected_run_length=60)
    for yt, et in zip(y, e):
        one.update(yt, et)
    chunks = ChangePointStream.poisson(1.0, 0.25, expected_run_length=60)
    for k in range(0, 300, 41):
        chunks.update(y[k:k + 41], e[k:k + 41])
    assert one.log_marginal == chunks.log_marginal
    assert one.rate_now() == chunks.rate_now()


def test_stream_inputs_are_checked():
    with pytest.raises(ValueError):
        ChangePointStream.poisson(1.0, 1.0, expected_run_length=0.5)
    with pytest.raises(ValueError):
        ChangePointStream.poisson(1.0, 1.0, 10).update([1.5])  # not a count
    with pytest.raises(ValueError):
        ChangePointStream.bernoulli(10).update([3], [2])  # more events than trials
    with pytest.raises(ValueError):
        ChangePointStream.bernoulli(10).update([3])  # trials missing
    with pytest.raises(ValueError):
        ChangePointStream.poisson(1.0, 1.0, 10).rate_now()  # no data yet
