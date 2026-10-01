"""fit_cyclic: binning on a circle (a bin may wrap round from the last interval to the first),
exact against enumerating every circular partition."""

import itertools
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from bayesbin import BernoulliModel, PoissonModel, fit, fit_cyclic


def _arc(model, idx):
    """(log evidence, posterior mean rate) of one bin made of the intervals idx."""
    if isinstance(model, PoissonModel):
        Y, E = model.y[idx].sum(), model.e[idx].sum()
        one = PoissonModel(np.array([Y]), model.alpha, model.beta, np.array([E]))
        return float(one.log_bin_evidence()[0, 0]), (Y + model.alpha) / (E + model.beta)
    S, G = model.s[idx].sum(), model.g[idx].sum()
    one = BernoulliModel(np.array([S]), np.array([G]), model.sigma, model.gamma)
    return float(one.log_bin_evidence()[0, 0]), (S + model.sigma) / (S + G + model.sigma + model.gamma)


def _enumerate(model, max_m):
    """log P(D | M), E[rate_k | D] and P(a bin ends at k | D) over every circular partition."""
    T = model.T
    by_m = {}  # M -> [(log evidence, rates, ends)]
    for M in [0, *range(2, max_m + 1)]:
        for ends in itertools.combinations(range(T), M):
            arcs = ([list(range(T))] if M == 0 else
                    [[k % T for k in range(ends[i - 1] + 1, ends[i] + 1 + (T if i == 0 else 0))]
                     for i in range(M)])
            lp, rates = 0.0, np.zeros(T)
            for idx in arcs:
                ev, r = _arc(model, np.array(idx))
                lp += ev
                rates[idx] = r
            by_m.setdefault(M, []).append((lp, rates, ends))
    log_ev = np.full(max_m + 1, -np.inf)
    for M, rows in by_m.items():
        log_ev[M] = logsumexp([r[0] for r in rows]) - math.log(math.comb(T, M)) + model.log_data_constant()
    post = np.exp(log_ev - logsumexp(log_ev))
    rate, boundary = np.zeros(T), np.zeros(T)
    for M, rows in by_m.items():
        z = logsumexp([r[0] for r in rows])
        for lp, rates, ends in rows:
            w = post[M] * math.exp(lp - z)
            rate += w * rates
            boundary[list(ends)] += w
    return log_ev, rate, boundary


def _models(T=7):
    rng = np.random.default_rng(2)
    e = rng.uniform(0.5, 2.0, T)
    y = rng.poisson(np.array([6, 1, 1, 2, 1, 7, 8][:T]) * e).astype(float)
    n = rng.integers(5, 15, T)
    s = rng.binomial(n, np.array([0.6, 0.1, 0.1, 0.2, 0.1, 0.7, 0.6][:T])).astype(float)
    return [PoissonModel(y, 1.5, 0.5, e), BernoulliModel(s, n - s, 1.0, 2.0)]


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("exact", [True, False])
def test_fit_cyclic_is_exact_against_enumerating_every_circular_partition(which, exact):
    model = _models()[which]
    res = fit_cyclic(model, 7, exact=exact)
    log_ev, rate, boundary = _enumerate(model, 7)
    assert res.log_evidence[1] == -np.inf and res.m_posterior[1] == 0.0
    np.testing.assert_allclose(res.log_evidence[[0, *range(2, 8)]], log_ev[[0, *range(2, 8)]], rtol=1e-12)
    np.testing.assert_allclose(res.rate, rate, rtol=1e-10)
    np.testing.assert_allclose(res.boundary_posterior, boundary, atol=1e-12)
    assert res.m_posterior.sum() == pytest.approx(1.0)


def test_fit_cyclic_does_not_depend_on_where_the_cycle_starts():
    rng = np.random.default_rng(5)
    T = 24
    lam = np.where((np.arange(T) >= 21) | (np.arange(T) < 3), 9.0, 2.0)  # high round midnight
    y = rng.poisson(lam * 4).astype(float)
    a = fit_cyclic(PoissonModel(y, 1.0, 0.25), 6)
    for r in (5, 13):
        b = fit_cyclic(PoissonModel(np.roll(y, r), 1.0, 0.25), 6)
        np.testing.assert_allclose(b.log_evidence, a.log_evidence, rtol=1e-12)
        np.testing.assert_allclose(b.rate, np.roll(a.rate, r), rtol=1e-9)
        np.testing.assert_allclose(b.boundary_posterior, np.roll(a.boundary_posterior, r), atol=1e-10)


def test_a_bin_round_midnight_is_one_bin():
    """Counts high from 21:00 to 03:00 every day: on the circle that is one bin, with no
    boundary at midnight; a linear fit has to split it there."""
    rng = np.random.default_rng(6)
    T = 24
    lam = np.where((np.arange(T) >= 21) | (np.arange(T) < 3), 9.0, 2.0)
    y = rng.poisson(lam * 10).astype(float)
    model = PoissonModel(y, 1.0, 0.1)
    cyc, lin = fit_cyclic(model, 6), fit(model, 6)
    assert cyc.boundary_posterior[T - 1] < 0.05  # between 23:00 and 00:00
    assert cyc.boundary_posterior[2] > 0.9 and cyc.boundary_posterior[20] > 0.9
    assert cyc.m_map == 2 and lin.m_map == 2  # two bins on the circle, three on the line
    assert cyc.log_marginal > lin.log_marginal
    assert abs(cyc.rate[0] - cyc.rate[T - 1]) < 1e-4 * cyc.rate[0]  # 23:00 and 00:00 share a rate
