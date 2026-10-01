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


def _enumerate(kind, x, n, prior, h, hazard_prior=None):
    """(log P(x), {run length: P}, E[rate now]) over all 2^(T-1) segmentations; with
    hazard_prior = (a, b), h ~ Beta(a, b) instead of fixed, and E[h | x] as well."""
    T = len(x)
    p, q = prior
    terms, runs, rates, hazards = [], {}, [], []
    for bits in itertools.product([0, 1], repeat=T - 1):
        starts = [0] + [i + 1 for i, b in enumerate(bits) if b]
        ends = starts[1:] + [T]
        c = sum(bits)
        if hazard_prior is None:
            lp = c * np.log(h) + (T - 1 - c) * np.log1p(-h)
        else:
            ha, hb = hazard_prior
            lp = betaln(ha + c, hb + T - 1 - c) - betaln(ha, hb)
            hazards.append((ha + c) / (ha + hb + T - 1))
        lp += sum(_segment_logev(kind, x[a:b], n[a:b], prior) for a, b in zip(starts, ends))
        terms.append(lp)
        runs.setdefault(T - starts[-1], []).append(lp)
        last = slice(starts[-1], T)
        A = x[last].sum() + p
        B = (n[last] - x[last]).sum() + q if kind == "bernoulli" else n[last].sum() + q
        rates.append(A / (A + B) if kind == "bernoulli" else A / B)
    z = logsumexp(terms)
    w = np.exp(np.array(terms) - z)
    out = z, {k: float(np.exp(logsumexp(v) - z)) for k, v in runs.items()}, float(w @ np.array(rates))
    return out if hazard_prior is None else (*out, float(w @ np.array(hazards)))


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


@pytest.mark.parametrize("kind", ["poisson", "bernoulli"])
def test_a_learnt_hazard_is_exact_against_enumerating_every_segmentation(kind):
    """h ~ Beta(a, b): each segmentation weighs B(a + c, b + T - 1 - c) / B(a, b) for its c
    change points; the stream tracks (run length, c) jointly."""
    x, n, _ = _small(kind)
    kw = dict(expected_run_length=4, prune=0, hazard_strength=0.7)
    cp = (ChangePointStream.poisson(1.5, 0.5, **kw) if kind == "poisson"
          else ChangePointStream.bernoulli(sigma=1.0, gamma=2.0, **kw))
    assert cp.hazard_posterior()[0] == pytest.approx(0.25)
    for T in range(1, len(x) + 1):
        cp.update(x[T - 1], n[T - 1])
        z, runs, rate, hazard = _enumerate(kind, x[:T], n[:T], cp.prior, None, hazard_prior=(0.7, 2.1))
        assert cp.log_marginal == pytest.approx(z, rel=1e-13, abs=1e-12)
        lengths, probs = cp.run_length_posterior()
        assert {int(k) for k in lengths} == set(runs)
        for k, p in zip(lengths, probs):
            assert p == pytest.approx(runs[int(k)], abs=1e-13)
        assert cp.rate_now()[0] == pytest.approx(rate, rel=1e-12)
        assert cp.hazard_posterior()[0] == pytest.approx(hazard, rel=1e-12)


def _starts(kind, x, n, prior, h=None, hazard_prior=None):
    """P(a segment starts at interval s | x), s = 0..T-1, over every segmentation."""
    T = len(x)
    terms, starts_of = [], []
    for bits in itertools.product([0, 1], repeat=T - 1):
        starts = [0] + [i + 1 for i, b in enumerate(bits) if b]
        ends = starts[1:] + [T]
        c = sum(bits)
        if hazard_prior is None:
            lp = c * np.log(h) + (T - 1 - c) * np.log1p(-h)
        else:
            lp = betaln(hazard_prior[0] + c, hazard_prior[1] + T - 1 - c) - betaln(*hazard_prior)
        terms.append(lp + sum(_segment_logev(kind, x[a:b], n[a:b], prior) for a, b in zip(starts, ends)))
        starts_of.append(starts)
    w = np.exp(np.array(terms) - logsumexp(terms))
    out = np.zeros(T)
    for wi, st in zip(w, starts_of):
        out[st] += wi
    return out


@pytest.mark.parametrize("kind", ["poisson", "bernoulli"])
@pytest.mark.parametrize("learn", [False, True])
def test_fixed_lag_smoothing_is_exact_against_enumeration(kind, learn):
    """p_change_at(k): P(a segment started k - 1 intervals before the latest | all data so far),
    the later intervals' data included, for every k up to the lag."""
    x, n, _ = _small(kind)
    kw = dict(expected_run_length=4, prune=0, lag=6, hazard_strength=0.7 if learn else None)
    cp = (ChangePointStream.poisson(1.5, 0.5, **kw) if kind == "poisson"
          else ChangePointStream.bernoulli(sigma=1.0, gamma=2.0, **kw))
    for T in range(1, len(x) + 1):
        cp.update(x[T - 1], n[T - 1])
        want = _starts(kind, x[:T], n[:T], cp.prior, cp.hazard, (0.7, 2.1) if learn else None)
        for k in range(1, min(T, 6) + 1):
            assert cp.p_change_at(k) == pytest.approx(want[T - k], abs=1e-12), (T, k)
        assert cp.p_change_at(1) == pytest.approx(cp.p_change_within(1), abs=1e-12)
    with pytest.raises(ValueError):
        cp.p_change_at(7)


def test_smoothing_sharpens_a_change_as_data_arrive():
    """A step in the rate, 2 to 6 at interval 400: when that interval is the latest, a start
    there is unlikely (one high count); twelve intervals later it is the likeliest start, and
    within one interval of it nearly certain. The same with merging and pruning off."""
    y = np.r_[np.full(400, 2.0), np.full(12, 6.0)]
    for kw in ({}, {"merge_bins": None, "prune": 0}):
        cp = ChangePointStream.poisson(1.0, 0.25, expected_run_length=300, lag=14, **kw)
        seen = []
        for t, yt in enumerate(y):
            cp.update(yt)
            if t >= 400:
                seen.append(cp.p_change_at(t - 400 + 1))
        at = [cp.p_change_at(k) for k in (11, 12, 13)]  # starts at 401, 400, 399
        assert seen[0] < 0.05 and seen[-1] > 0.7 and np.all(np.diff(seen) > 0), seen
        assert max(at) == at[1] and sum(at) > 0.95, at
    assert cp.p_change_at(12) == pytest.approx(seen[-1])


def test_a_strong_hazard_prior_is_the_fixed_hazard():
    rng = np.random.default_rng(9)
    y = np.concatenate([rng.poisson(lam, 150) for lam in (2, 7, 3, 12)]).astype(float)
    fixed = ChangePointStream.poisson(1.0, 0.25, expected_run_length=50)
    strong = ChangePointStream.poisson(1.0, 0.25, expected_run_length=50, hazard_strength=1e9)
    for yt in y:
        assert strong.pit(yt, u=0.3) == pytest.approx(fixed.pit(yt, u=0.3), abs=1e-6)
        fixed.update(yt)
        strong.update(yt)
    assert strong.log_marginal == pytest.approx(fixed.log_marginal, rel=1e-7)
    assert strong.hazard_posterior()[0] == pytest.approx(1 / 50, rel=1e-6)
    assert strong.rate_now()[0] == pytest.approx(fixed.rate_now()[0], rel=1e-6)


def test_the_hazard_is_learnt_from_the_data():
    """Segments of geometric length (mean 40, a hazard of 1/40) and rates from the prior: a weak
    prior centred on 1/1000 finds 1/40, predicts better than the fixed 1/1000, and keeps its
    state bounded (the number of change points is known to within a few dozen)."""
    rng = np.random.default_rng(8)
    y = []
    while len(y) < 5000:
        y.extend(rng.poisson(rng.gamma(2.0, 2.0), rng.geometric(1 / 40)))
    y = np.array(y[:5000], float)
    wrong = ChangePointStream.poisson(2.0, 0.5, expected_run_length=1000)
    learnt = ChangePointStream.poisson(2.0, 0.5, expected_run_length=1000, hazard_strength=1.0)
    widest = 0
    for yt in y:
        wrong.update(yt)
        learnt.update(yt)
        widest = max(widest, learnt._p2.shape[1])
    mean, sd = learnt.hazard_posterior()
    assert abs(mean - 1 / 40) < 3 * sd and sd < 0.3 * mean, (mean, sd)
    assert learnt.log_marginal > wrong.log_marginal + 20
    assert widest < 120, widest


def test_negbinomial_stream_is_exact_against_enumeration():
    """Negative binomial segments, NB(r e, p) with p ~ Beta: the marginal likelihood and run
    lengths against enumerating every segmentation (segment evidence in closed form)."""
    rng = np.random.default_rng(3)
    T, r, (alpha, beta) = 9, 2.0, (1.5, 0.5)
    e = rng.uniform(0.5, 2.0, T)
    x = rng.poisson(rng.gamma(r * e, np.repeat([1.0, 6.0, 2.0], 3) / r)).astype(float)
    cp = ChangePointStream.negbinomial(alpha, beta, r, expected_run_length=4, prune=0)
    a, b = cp.prior

    def seg(xs, es):
        n = r * es
        return (np.sum(gammaln(n + xs) - gammaln(n) - gammaln(xs + 1))
                + betaln(a + n.sum(), b + xs.sum()) - betaln(a, b))

    for t in range(1, T + 1):
        cp.update(x[t - 1], e[t - 1])
        terms = []
        for bits in itertools.product([0, 1], repeat=t - 1):
            starts = [0] + [i + 1 for i, bit in enumerate(bits) if bit]
            ends = starts[1:] + [t]
            lp = sum(bits) * np.log(cp.hazard) + (t - 1 - sum(bits)) * np.log1p(-cp.hazard)
            terms.append(lp + sum(seg(x[i:j], e[i:j]) for i, j in zip(starts, ends)))
        assert cp.log_marginal == pytest.approx(logsumexp(terms), rel=1e-13, abs=1e-12)


def test_negbinomial_tends_to_poisson_and_the_mixture_finds_the_dispersion():
    rng = np.random.default_rng(4)
    y = rng.poisson(4.0, 300)
    p = ChangePointStream.poisson(1.0, 0.25, expected_run_length=200)
    nb = ChangePointStream.negbinomial(1.0, 0.25, 1e6, expected_run_length=200)
    p.update(y)
    nb.update(y)
    assert nb.log_marginal == pytest.approx(p.log_marginal, abs=1e-3)
    assert nb.rate_now()[0] == pytest.approx(p.rate_now()[0], rel=1e-5)
    for true_r in (1.0, 8.0):
        yy = rng.poisson(rng.gamma(true_r, 5.0 / true_r, 1500))
        mix = ChangePointStream.overdispersed(1.0, 0.2, expected_run_length=1000)
        mix.update(yy)
        r, w = mix.dispersion_posterior()
        assert r[np.argmax(w)] == true_r and w.sum() == pytest.approx(1.0)
    # the beta-negative-binomial has a power-law tail: sum far enough out
    assert mix.next_pmf(np.arange(20000)).sum() == pytest.approx(1.0, abs=1e-7)
    np.testing.assert_allclose(mix.next_cdf(np.arange(40)), np.cumsum(mix.next_pmf(np.arange(40))), atol=1e-12)


def test_overdispersed_counts_are_calibrated_by_the_mixture_and_not_by_poisson():
    rng = np.random.default_rng(14)
    lam, y = rng.gamma(2.0, 2.5), []
    for t in range(2500):
        if t > 0 and rng.random() < 1 / 300:
            lam = rng.gamma(2.0, 2.5)
        y.append(rng.poisson(rng.gamma(1.5, lam / 1.5)))  # dispersion 1.5: variance lam (1 + lam/1.5)
    q = {"mixture": [], "poisson": []}
    mix = ChangePointStream.overdispersed(2.0, 0.4, expected_run_length=300)
    poi = ChangePointStream.poisson(2.0, 0.4, expected_run_length=300)
    for yt in y:
        q["mixture"].append(mix.pit(yt, u=rng.random()))
        q["poisson"].append(poi.pit(yt, u=rng.random()))
        mix.update(yt)
        poi.update(yt)
    assert kstest(q["mixture"], "uniform").pvalue > 0.01
    assert kstest(q["poisson"], "uniform").pvalue < 1e-10


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
    # pruning alone (merging, which also bounds the state, off in both)
    a = ChangePointStream.poisson(2.0, 0.4, expected_run_length=300, merge_bins=None)  # prune=1e-12
    b = ChangePointStream.poisson(2.0, 0.4, expected_run_length=300, prune=0, merge_bins=None)
    for yt in y:
        assert a.next_logpmf(yt)[0] == pytest.approx(b.next_logpmf(yt)[0], abs=1e-8)
        a.update(yt)
        b.update(yt)
        ra, rb = a.rate_now(), b.rate_now()
        assert ra[0] == pytest.approx(rb[0], rel=1e-7) and ra[1] == pytest.approx(rb[1], rel=1e-7)
    assert a.n_runs < b.n_runs / 3


@pytest.mark.parametrize("kind", ["poisson", "bernoulli"])
def test_merging_old_run_lengths_bounds_the_state_and_costs_little(kind):
    """A long quiet stretch, then a change: every run length since the start stays
    plausible, so without merging the state grows with the stretch; with it, with the
    log of it, and the answers barely move."""
    rng = np.random.default_rng(13)
    if kind == "poisson":
        x = np.concatenate([rng.poisson(3.0, 1800), rng.poisson(5.0, 300)]).astype(float)
        n = np.ones_like(x)
        make = lambda **kw: ChangePointStream.poisson(1.0, 0.25, expected_run_length=1000, **kw)
    else:
        n = np.full(2100, 25.0)
        x = np.concatenate([rng.binomial(25, 0.2, 1800), rng.binomial(25, 0.3, 300)]).astype(float)
        make = lambda **kw: ChangePointStream.bernoulli(1000, **kw)
    merged, exact = make(), make(merge_bins=None)
    u = rng.random(len(x))
    worst = dict(rate=0.0, sd=0.0, pit=0.0, recent=0.0)
    most = [0, 0]  # the largest states, merged and not (the change prunes the runs spanning it)
    for t in range(len(x)):
        q1, q2 = merged.pit(x[t], n[t], u=u[t]), exact.pit(x[t], n[t], u=u[t])
        worst["pit"] = max(worst["pit"], abs(q1 - q2))
        merged.update(x[t], n[t])
        exact.update(x[t], n[t])
        (r1, s1), (r2, s2) = merged.rate_now(), exact.rate_now()
        worst["rate"] = max(worst["rate"], abs(r1 / r2 - 1))
        worst["sd"] = max(worst["sd"], abs(s1 / s2 - 1))
        worst["recent"] = max(worst["recent"], abs(merged.p_change_within(20) - exact.p_change_within(20)))
        most = [max(most[0], merged.n_runs), max(most[1], exact.n_runs)]
    bound = merged.exact_recent + merged.merge_bins * np.log2(len(x) / merged.exact_recent) + 2
    assert most[0] <= bound and most[1] > 1500, most
    assert worst["rate"] < 1e-4 and worst["sd"] < 1e-3 and worst["pit"] < 1e-4 and worst["recent"] < 1e-4, worst
    # the run-length posterior still sums to 1, and is the same where exact (up to exact_recent)
    lengths, p = merged.run_length_posterior()
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    l2, p2 = exact.run_length_posterior()
    short = lengths <= merged.exact_recent
    np.testing.assert_allclose(p[short], p2[np.isin(l2, lengths[short])], atol=1e-6)


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
    # the cdfs of Beta posteriors by the ratios of successive pmf terms (counts below 64),
    # beyond the number of trials, and with no exposure
    np.testing.assert_allclose(b.next_cdf(np.arange(-1, 25), 20),
                               np.concatenate([[0.0], np.cumsum(b.next_pmf(np.arange(25), 20))]), atol=1e-10)
    nb = ChangePointStream.negbinomial(1.0, 0.4, 1.5, expected_run_length=50)
    nb.update(rng.negative_binomial(1.5, 0.3, 200))
    for e in (1.0, 2.5, 0.0):
        np.testing.assert_allclose(nb.next_cdf(np.arange(-1, 60), e),
                                   np.concatenate([[0.0], np.cumsum(nb.next_pmf(np.arange(60), e))]), atol=1e-12)


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
    for bad in (1.5, -1, np.inf, np.nan):  # one interval at a time has its own checks
        with pytest.raises(ValueError):
            ChangePointStream.poisson(1.0, 1.0, 10).update(bad)
    with pytest.raises(ValueError):
        ChangePointStream.poisson(1.0, 1.0, 10).update(2, 0.0)  # events without exposure
    with pytest.raises(ValueError):
        ChangePointStream.bernoulli(10).update([3], [2])  # more events than trials
    with pytest.raises(ValueError):
        ChangePointStream.bernoulli(10).update([3])  # trials missing
    with pytest.raises(ValueError):
        ChangePointStream.poisson(1.0, 1.0, 10).rate_now()  # no data yet
