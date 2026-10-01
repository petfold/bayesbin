"""latency_posterior and separation_level (binsdfc's -L and -y), exact against enumeration."""

import itertools
import math

import numpy as np
import pytest
from scipy.special import betainc, betaincc, gammainc, gammaincc, logsumexp

from bayesbin import BernoulliModel, PoissonModel, fit, latency_posterior, separation_level


def _tails(model, idx, level):
    if isinstance(model, PoissonModel):
        A, B = model.y[idx].sum() + model.alpha, model.e[idx].sum() + model.beta
        return gammainc(A, B * level), gammaincc(A, B * level)
    A, B = model.s[idx].sum() + model.sigma, model.g[idx].sum() + model.gamma
    return betainc(A, B, level), betaincc(A, B, level)


def _enumerate(model, level, max_m, inhibitory=False):
    T = model.T
    L = model.log_bin_evidence()
    by_m = {}
    for M in range(max_m + 1):
        for ends in itertools.combinations(range(T - 1), M):
            edges = [-1, *ends, T - 1]
            bins = [np.arange(edges[i] + 1, edges[i + 1] + 1) for i in range(M + 1)]
            lp = sum(L[b[0], b[-1]] for b in bins)
            lat = np.zeros(T)
            run = 1.0  # P(every bin so far below the level)
            for b in bins:
                below, above = _tails(model, b, level)
                if inhibitory:
                    below, above = above, below
                lat[b[0]] = run * above
                run *= below
            by_m.setdefault(M, []).append((lp, lat))
    log_ev = np.array([logsumexp([r[0] for r in by_m[M]]) - math.log(math.comb(T - 1, M)) for M in range(max_m + 1)])
    post = np.exp(log_ev - logsumexp(log_ev))
    out = np.zeros(T)
    for M, rows in by_m.items():
        z = logsumexp([r[0] for r in rows])
        for lp, lat in rows:
            out += post[M] * math.exp(lp - z) * lat
    return out


def _models(T=7):
    rng = np.random.default_rng(8)
    n = rng.integers(8, 20, T)
    s = rng.binomial(n, np.array([0.05, 0.05, 0.1, 0.5, 0.6, 0.4, 0.5][:T])).astype(float)
    e = rng.uniform(0.5, 2.0, T)
    y = rng.poisson(np.array([1, 1, 2, 8, 9, 7, 8][:T]) * e).astype(float)
    return [BernoulliModel(s, n - s, 1.0, 4.0), PoissonModel(y, 1.0, 0.25, e)]


@pytest.mark.parametrize("which,level", [(0, 0.2), (1, 4.0)])
@pytest.mark.parametrize("inhibitory", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_latency_posterior_is_exact_against_enumeration(which, level, inhibitory, exact):
    model = _models()[which]
    got = latency_posterior(model, level, 6, inhibitory=inhibitory, exact=exact)
    want = _enumerate(model, level, 6, inhibitory)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-14)
    assert 0 < got.sum() <= 1


def test_a_response_has_its_latency_where_the_rate_rises():
    """Trials with a low rate up to 80 and a high one after: the latency posterior peaks at 80,
    at the level where a latency is most probable."""
    rng = np.random.default_rng(4)
    T, trials = 160, 40
    f = np.where(np.arange(T) < 80, 0.02, 0.15)
    s = rng.binomial(trials, f).astype(float)
    model = BernoulliModel(s, trials - s, 1.0, 32.0)
    level, p = separation_level(model, 6)
    assert 0.02 < level < 0.15 and p > 0.95
    lat = latency_posterior(model, level, 6)
    assert abs(int(np.argmax(lat)) - 80) <= 2 and lat.sum() == pytest.approx(p)
    assert fit(model, 6).rate[:70].max() < level < fit(model, 6).rate[90:].min()
