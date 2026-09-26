"""bayesbin against the original binsdfc, against brute-force enumeration, and
against the paper's own device for predictions (adding a virtual spike)."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate
from scipy.special import betaln, gammaln, logsumexp

from bayesbin import BernoulliModel, PoissonModel, credible_m_range, fit, spike_counts

DATA = Path(__file__).parent / "data"


def _seed1():
    trials = [list(map(int, line.split()[1:])) for line in open(DATA / "testdata_seed1.txt")]
    return spike_counts(trials, -100, 499)  # binsdfc -s -100 -e 500: the end is exclusive


# --- the original program as the reference -----------------------------------------


def test_evidences_match_binsdfc():
    # reference: binsdfc -s -100 -e 500 -m 10 -M -n tests/data/testdata_seed1.txt
    ref = np.loadtxt(DATA / "binsdfc_seed1_marginal.txt")
    s, g = _seed1()
    r = fit(BernoulliModel(s, g), 10)
    np.testing.assert_allclose(r.log_evidence, ref[:, 1], atol=6e-3)  # printed to 6 digits
    header = open(DATA / "binsdfc_seed1_marginal.txt").read().splitlines()[1]
    assert r.log_marginal == pytest.approx(float(header.split(":")[1]), abs=6e-3)


@pytest.mark.parametrize("mass", ["0", "0.9"])
def test_rate_and_error_bars_match_binsdfc(mass):
    # reference: binsdfc -s -100 -e 500 -m 10 -v -l <mass> tests/data/testdata_seed1.txt
    ref = np.loadtxt(DATA / f"binsdfc_seed1_sdf_l{mass}.txt")
    s, g = _seed1()
    r = fit(BernoulliModel(s, g), 10, m_mass=float(mass))
    np.testing.assert_allclose(r.rate, ref[:, 1], rtol=2e-5)
    # binsdfc adds in log space through a lookup table (logAdd<400, 40, ...>), an
    # approximation that shows in the variance at the 1e-5 level
    np.testing.assert_allclose(r.rate_std, ref[:, 2], rtol=5e-5)


def test_credible_m_range_is_binsdfcs_rule():
    ref = np.loadtxt(DATA / "binsdfc_seed1_marginal.txt")[:, 1]
    assert credible_m_range(ref, 0.0) == (7, 7)  # "Including models with 7<=M<=7"
    assert credible_m_range(ref, 0.9) == (5, 10)  # "... 5<=M<=10 at probability 0.96"


# --- brute force: every boundary configuration ----------------------------------------


def _configs(T, M):
    """All ways to place M boundaries: bins as (a, b) inclusive."""
    for cuts in itertools.combinations(range(T - 1), M):
        edges = [-1, *cuts, T - 1]
        yield [(edges[i] + 1, edges[i + 1]) for i in range(M + 1)]


def _brute(model, max_m):
    L = model.log_bin_evidence()
    mean, _ = model.bin_moments()
    T = model.T
    log_ev, rates = [], []
    for M in range(max_m + 1):
        terms, rate_terms = [], []
        for bins in _configs(T, M):
            lw = sum(L[a, b] for a, b in bins)
            terms.append(lw)
            r = np.zeros(T)
            for a, b in bins:
                r[a:b + 1] = mean[a, b]
            rate_terms.append((lw, r))
        z = logsumexp(terms)
        log_ev.append(z - math.log(math.comb(T - 1, M)) + model.log_data_constant())
        rates.append(sum(np.exp(lw - z) * r for lw, r in rate_terms))
    log_ev = np.array(log_ev)
    post = np.exp(log_ev - logsumexp(log_ev))
    return log_ev, sum(p * r for p, r in zip(post, rates))


def test_bernoulli_dynamic_programme_equals_enumeration():
    rng = np.random.default_rng(3)
    s = rng.integers(0, 6, 9).astype(float)
    model = BernoulliModel(s, 6 - s, sigma=0.7, gamma=2.5)
    log_ev, rate = _brute(model, 4)
    r = fit(model, 4)
    np.testing.assert_allclose(r.log_evidence, log_ev, rtol=1e-12)
    np.testing.assert_allclose(r.rate, rate, rtol=1e-10)


def test_poisson_dynamic_programme_equals_enumeration():
    rng = np.random.default_rng(4)
    y = rng.poisson([1, 1, 1, 6, 6, 1, 1, 1, 3]).astype(float)
    e = rng.uniform(0.5, 2.0, len(y))
    model = PoissonModel(y, alpha=1.3, beta=0.8, e=e)
    log_ev, rate = _brute(model, 4)
    r = fit(model, 4)
    np.testing.assert_allclose(r.log_evidence, log_ev, rtol=1e-12)
    np.testing.assert_allclose(r.rate, rate, rtol=1e-10)


def test_poisson_one_bin_evidence_is_the_integral():
    y, e, a, b = np.array([2.0, 0.0, 3.0]), np.array([1.0, 0.5, 2.0]), 1.5, 0.7

    def integrand(lam):
        prior = math.exp(a * math.log(b) - gammaln(a) + (a - 1) * math.log(lam) - b * lam)
        return prior * np.prod([math.exp(-lam * ek) * (lam * ek) ** yk / math.factorial(int(yk))
                                for yk, ek in zip(y, e)])

    direct = math.log(integrate.quad(integrand, 0, np.inf)[0])
    assert fit(PoissonModel(y, a, b, e=e), 0).log_evidence[0] == pytest.approx(direct, rel=1e-9)


def test_bernoulli_one_bin_evidence_is_the_beta_function():
    s, g = np.array([1.0, 2.0, 0.0]), np.array([3.0, 2.0, 4.0])
    r = fit(BernoulliModel(s, g, sigma=2.0, gamma=5.0), 0)
    assert r.log_evidence[0] == pytest.approx(betaln(3 + 2, 9 + 5) - betaln(2, 5))


# --- the paper's device: predictions as evidence ratios with a virtual spike -----------


def test_forward_backward_equals_virtual_spike_ratios():
    """§4 of the paper: P(spike | k, D, M) = E_k[M] / E[M], where E_k is the evidence
    with one more spike at k. The backward pass gives all k at once; check it."""
    from bayesbin.core import _backward, _cover_sum, _forward, bin_posterior_for_m

    rng = np.random.default_rng(5)
    s = rng.integers(0, 4, 14).astype(float)
    g = 4 - s
    model = BernoulliModel(s, g)
    L = model.log_bin_evidence()
    for M in (0, 2, 5):
        z = fit(model, M).log_evidence[M]
        via_ratio = []
        for k in range(len(s)):
            s1 = s.copy()
            s1[k] += 1
            via_ratio.append(math.exp(fit(BernoulliModel(s1, g), M).log_evidence[M] - z))
        W = bin_posterior_for_m(L, _forward(L, M), _backward(L, M), M)
        np.testing.assert_allclose(_cover_sum(W * model.bin_moments()[0]), via_ratio, rtol=1e-10)


# --- invariants and behaviour ----------------------------------------------------------


def test_every_interval_lies_in_exactly_one_bin():
    s, g = _seed1()
    r = fit(BernoulliModel(s[:120], g[:120]), 8, keep_bins=True)
    from bayesbin.core import _cover_sum

    np.testing.assert_allclose(_cover_sum(r.bin_posterior), 1.0, atol=1e-10)
    assert r.m_posterior.sum() == pytest.approx(1.0)


def test_recovers_the_response_onset():
    # the generator switches from 0.01 to 0.08 per ms at t = 80 ms (index 180)
    s, g = _seed1()
    r = fit(BernoulliModel(s, g), 10)
    onset = int(np.argmax(r.boundary_posterior[150:210])) + 150
    assert abs(onset - 179) <= 3  # the last background interval is t = 79
    assert r.rate[200] > 3 * r.rate[50]


def test_poisson_weak_prior_centres_on_the_overall_rate():
    y = np.array([0, 0, 5, 0, 1.0])
    m = PoissonModel.weak_prior(y)
    assert m.alpha / m.beta == pytest.approx(y.sum() / len(y))


def test_more_than_one_spike_per_interval_is_refused():
    with pytest.raises(ValueError):
        spike_counts([[3, 3]], 0, 5)


# --- the fast bin posterior against the exact log-space one ---------------------------


def _strong_poisson(T=400, seed=6):
    """Lots of data and sharp steps: evidences span thousands of nats."""
    rng = np.random.default_rng(seed)
    lam = np.where((np.arange(T) // 50) % 2 == 0, 20.0, 200.0)
    return PoissonModel.weak_prior(rng.poisson(lam).astype(float))


@pytest.mark.parametrize("mass", [None, 0.0, 0.9])
def test_fast_bin_posterior_equals_exact(mass):
    for model in (_strong_poisson(), BernoulliModel(*_seed1())):
        fast = fit(model, 12, m_mass=mass, keep_bins=True)
        slow = fit(model, 12, m_mass=mass, keep_bins=True, exact=True)
        np.testing.assert_allclose(fast.bin_posterior, slow.bin_posterior, atol=1e-12)
        np.testing.assert_allclose(fast.rate, slow.rate, rtol=1e-10)
        # var = E[f²] - E[f]² cancels: rates ~200 with sd ~2 magnify rounding ~10⁴×
        np.testing.assert_allclose(fast.rate_std, slow.rate_std, rtol=1e-6)


def test_underflow_fallback_recomputes_exactly():
    from bayesbin import bin_posterior
    from bayesbin.core import _backward, _forward

    model = _strong_poisson(T=200)
    L = model.log_bin_evidence()
    fwd, bwd = _forward(L, 8), _backward(L, 8)
    log_c = -fwd[:, -1] - np.log(9)
    everywhere = bin_posterior(L, fwd, bwd, log_c, tol=0.0)  # every entry takes the fallback
    np.testing.assert_allclose(bin_posterior(L, fwd, bwd, log_c), everywhere, atol=1e-12)
    np.testing.assert_allclose(bin_posterior(L, fwd, bwd, log_c, exact=True), everywhere, atol=1e-13)


def test_fast_dynamic_programmes_equal_the_exact_ones():
    from bayesbin.core import _backward, _forward, _forward_exact

    for model in (_strong_poisson(T=300), BernoulliModel(*_seed1()),
                  PoissonModel.weak_prior(np.zeros(50))):  # all-zero data: flat gains
        L = model.log_bin_evidence()
        np.testing.assert_allclose(_forward(L, 15), _forward_exact(L, 15), rtol=1e-12)
        np.testing.assert_allclose(_backward(L, 15), _backward(L, 15, exact=True), rtol=1e-12)


# --- the forward-backward C++ (cpp/binsdfc-fb), if it has been built -------------------

_FB = Path(__file__).parent.parent / "cpp" / "binsdfc-fb" / "binsdfc"


@pytest.mark.skipif(not _FB.exists(), reason="cpp/binsdfc-fb not built (see its README.fb.md)")
@pytest.mark.parametrize("mass", ["0", "0.9"])
def test_forward_backward_cpp_matches_bayesbin_and_the_original(mass):
    import subprocess

    args = [str(_FB), "-s", "-100", "-e", "500", "-m", "10", "-v", "-l", mass, str(DATA / "testdata_seed1.txt")]
    fb = np.loadtxt(subprocess.run(args, capture_output=True, text=True, check=True).stdout.splitlines())
    vs = np.loadtxt(subprocess.run([args[0], "-V", *args[1:]], capture_output=True, text=True,
                                   check=True).stdout.splitlines())
    orig = np.loadtxt(DATA / f"binsdfc_seed1_sdf_l{mass}.txt")
    np.testing.assert_allclose(vs[:, 1:3], orig[:, 1:3], rtol=0, atol=0)  # -V is the original path
    s, g = _seed1()
    r = fit(BernoulliModel(s, g), 10, m_mass=float(mass))
    np.testing.assert_allclose(fb[:, 1], r.rate, rtol=6e-6)  # 6 printed digits
    np.testing.assert_allclose(fb[:, 2], r.rate_std, rtol=6e-6)


@pytest.mark.skipif(not _FB.exists(), reason="cpp/binsdfc-fb not built (see its README.fb.md)")
def test_forward_backward_cpp_on_data_that_exercises_the_underflow_bounds():
    # 400 trials, rate switching 0.02 / 0.3 every 50 intervals: evidences span ~10^4 nats
    import subprocess

    path = DATA / "strong_400trials.txt"
    out = subprocess.run([str(_FB), "-s", "0", "-e", "400", "-m", "12", "-v", "-l", "0.9", str(path)],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    fb = np.loadtxt(out)
    trials = [list(map(int, line.split()[1:])) for line in open(path)]
    s, g = spike_counts(trials, 0, 399)
    r = fit(BernoulliModel(s, g), 12, m_mass=0.9, exact=True)
    np.testing.assert_allclose(fb[:, 1], r.rate, rtol=6e-6)
    np.testing.assert_allclose(fb[:, 2], r.rate_std, rtol=6e-6, atol=1e-6)
