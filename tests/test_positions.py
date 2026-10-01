"""boundary_positions: where each of M bin boundaries lies, exact against enumeration."""

import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from bayesbin import BernoulliModel, PoissonModel, boundary_positions


def _models(T=7):
    rng = np.random.default_rng(2)
    e = rng.uniform(0.5, 2.0, T)
    y = rng.poisson(np.array([6, 1, 1, 2, 1, 7, 8][:T]) * e).astype(float)
    n = rng.integers(5, 15, T)
    s = rng.binomial(n, np.array([0.6, 0.1, 0.1, 0.2, 0.1, 0.7, 0.6][:T])).astype(float)
    return [PoissonModel(y, 1.5, 0.5, e), BernoulliModel(s, n - s, 1.0, 2.0)]


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("exact", [True, False])
def test_boundary_positions_are_exact_against_enumeration(which, exact):
    """P(the j-th of M boundaries is after interval k | D, M), over every placement of M."""
    model = _models()[which]
    T, M = model.T, 3
    L = model.log_bin_evidence()
    terms, places = [], []
    for ends in itertools.combinations(range(T - 1), M):
        edges = [-1, *ends, T - 1]
        terms.append(sum(L[edges[i] + 1, edges[i + 1]] for i in range(M + 1)))
        places.append(ends)
    w = np.exp(np.array(terms) - logsumexp(terms))
    want = np.zeros((M, T - 1))
    for wi, ends in zip(w, places):
        for j, k in enumerate(ends):
            want[j, k] += wi
    got = boundary_positions(model, M, exact=exact)
    np.testing.assert_allclose(got, want, atol=1e-12)
    np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=1e-12)
    with pytest.raises(ValueError):
        boundary_positions(model, T)


def test_boundary_positions_on_a_step():
    """Four steps, 120 intervals each: each boundary's posterior sits at its own step."""
    rng = np.random.default_rng(3)
    y = rng.poisson(np.repeat([2.0, 8.0, 3.0, 12.0], 120)).astype(float)
    pos = boundary_positions(PoissonModel.weak_prior(y), 3)
    assert list(np.argmax(pos, axis=1)) == pytest.approx([119, 239, 359], abs=2)
    assert (pos.max(axis=1) > 0.3).all()
