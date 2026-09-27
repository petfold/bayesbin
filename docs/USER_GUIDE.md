# bayesbin user guide

**bayesbin estimates a rate that changes along an ordered axis — time, position,
dose, age — from counts of events, with error bars, without asking you to choose
bin widths.** It is exact: no fitting by optimisation, no sampling, no tuning of
smoothing parameters.

This guide is for people who want to use it, not derive it. The mathematics is
in the paper (Endres, Oram, Schindelin & Földiák, NIPS 2007); here there is
none beyond the words "average" and "probability".

Contents:

1. [Why not a histogram?](#1-why-not-a-histogram)
2. [The idea in plain words](#2-the-idea-in-plain-words)
3. [Is it right for my data?](#3-is-it-right-for-my-data)
4. [Install](#4-install)
5. [Tutorial 1: spike trains (a PSTH)](#5-tutorial-1-spike-trains-a-psth)
6. [Tutorial 2: event counts with exposure](#6-tutorial-2-event-counts-with-exposure)
7. [Tutorial 3: success rates](#7-tutorial-3-success-rates)
8. [Tutorial 4: a daily profile](#8-tutorial-4-a-daily-profile)
9. [Choosing the settings](#9-choosing-the-settings)
10. [Reading the results](#10-reading-the-results)
11. [Size and speed](#11-size-and-speed)
12. [Pitfalls and questions](#12-pitfalls-and-questions)

## 1. Why not a histogram?

The usual way to see how a rate changes is to count events in bins of a fixed
width and plot the counts. Everything then depends on the width:

- **Bins too small**: each bin holds a handful of events, and the plot is mostly
  noise. Real features are there, but you cannot tell them from chance.
- **Bins too large**: the noise averages out, but so do the features. A sharp
  onset becomes a ramp, and a short burst is diluted into its neighbours or
  disappears.
- **Bins of one fixed size**: even the best compromise is a compromise
  everywhere. A rate that jumps suddenly and then stays flat for a long time
  needs small bins at the jump and large bins on the flat stretch — no single
  width does both. And where exactly the bin edges fall (at 0, or at 5?) changes
  the picture too.

The figure shows 30 repeated trials of a simulated neuron: background firing, a
sudden 50-ms burst after a stimulus, a long plateau, back to background. The
dashed line is the true rate.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/bins-dark.png">
  <img alt="Four panels over the same data. 1-ms bins: a jagged trace of spikes, RMS error 0.033. 50-ms bins: smooth steps, but the sharp onset at 80 ms becomes a ramp and the burst is flattened, RMS error 0.013. 10-ms bins: still jagged on the plateau and still smearing, RMS error 0.012. bayesbin: a curve that jumps at 80 ms, stays flat on the plateau and drops at 380 ms, with a shaded error band, RMS error 0.007." src="img/bins-light.png">
</picture>

bayesbin does not pick one binning. It considers every way of cutting the axis
into bins, of every size, and lets the data say how much each one is worth. The
result (bottom panel) has the sharp onset *and* the smooth plateau, an error
band that is narrow where the data are clear and wide where they are not, and
about half the error of the best fixed histogram.

## 2. The idea in plain words

The model behind bayesbin is simple: **the rate is constant within each bin, and
may jump between bins.** The bins can have any widths, and there can be any
number of them, from one bin for the whole axis up to a limit you set.

For one particular set of bins, it is easy to say how well it explains the
data: a bin that lumps a burst together with a quiet stretch explains the
counts badly; a bin that fits them both well is good. Too many bins are
penalised automatically: each extra bin must earn its place by explaining the
data better, so it only appears where the data demand it (this is the "Occam
factor" of Bayesian model comparison, and it is why no smoothing parameter is
needed).

bayesbin then **averages over all possible sets of bins**, each weighted by how
well it explains the data. There are astronomically many of them, but a
dynamic-programming trick sums over all of them exactly, in a time that grows
only with the square of the axis length. The average of many step functions,
weighted like that, is the estimate you get: sharp where every good set of
bins agrees on a step, smooth where they disagree about where a step is, and
with error bars that come from the same average.

As by-products you get, for free:

- the probability that the rate changes at each point (change points);
- the probability of each number of bins, including the probability that
  there is no change at all;
- the probability of each possible bin.

## 3. Is it right for my data?

bayesbin fits when all of these hold:

- **One ordered axis.** Time, position along a genome, dose, age, distance:
  anything where "neighbouring" means something. Not for unordered categories,
  and not (yet) for two-dimensional data.
- **Events counted per interval.** Either counts (0, 1, 2, … events in each
  interval, possibly with an *exposure* such as observation time or population
  size) or successes out of trials (a spike or none in each trial; a purchase
  or none per visitor).
- **A rate that is roughly piecewise constant**, or at least can be described
  by steps. Smoothly drifting rates work too, as a staircase; they just need
  more bins.

And these are the assumptions built in — know them, because data that break
them will fool it:

- **Given the rate, events are independent.** For counts this means Poisson
  variation: the spread of counts around their mean is as large as the mean.
  Bursty data, where events come in clumps (retweets, aftershocks, a server
  logging the same error ten times), are *overdispersed*: bayesbin will read
  each clump as a real change of rate and add bins for them. For trials it
  means that trials are independent repeats of the same underlying rate.
- **No preference for smoothness.** Neighbouring bins' rates are unrelated a
  priori; every number of bins up to the limit is equally likely a priori, and
  so is every placement of the boundaries.
- **Every trial follows the same rate profile.** If your trials differ
  systematically (the neuron adapts over the session, the website changed on
  day 3), the estimate is the average profile, and its error bars understate
  the spread between trials.

Good uses: peri-stimulus time histograms, arrival or incident rates, photon or
particle counts, cases per week with population as exposure, conversion or
failure rates along time of day or dose, change-point detection with
uncertainty, daily or weekly profiles.

## 4. Install

```sh
pip install "bayesbin @ git+https://github.com/petfold/bayesbin"          # NumPy/SciPy only
pip install "bayesbin[fast] @ git+https://github.com/petfold/bayesbin"    # + numba: ~2x faster, all cores
```

The `fast` version compiles its kernels the first time it is used in a new
environment (about a minute, once); after that they load from a cache.

## 5. Tutorial 1: spike trains (a PSTH)

Thirty trials, spike times in milliseconds, recorded from 100 ms before a
stimulus to 500 ms after. We simulate them here; with your own data, `trials`
is a list with one array of integer spike times per trial.

```python
import numpy as np
from bayesbin import BernoulliModel, fit, spike_counts

rng = np.random.default_rng(1)
t = np.arange(-100, 500)                       # 600 intervals of 1 ms
p_true = np.select([t < 80, t < 130, t < 380], [0.01, 0.08, 0.05], 0.01)
trials = [t[rng.random(t.size) < p_true] for _ in range(30)]   # 30 trials of spike times

s, g = spike_counts(trials, t_start=-100, t_end=499)
result = fit(BernoulliModel(s, g), max_boundaries=20)
```

`spike_counts` turns the trials into two numbers per millisecond: `s`, how many
trials had a spike there, and `g`, how many did not. `BernoulliModel` says "in
each interval, each trial either has an event or not", and `fit` does the rest.

The estimated firing probability and its error bar at t = 100 ms (index 200,
since the axis starts at −100):

```python
print(f"{result.rate[200]:.4f} ± {result.rate_std[200]:.4f}")   # 0.0739 ± 0.0069 (true: 0.08)
```

`result.rate` is the estimate for every millisecond, and `result.rate_std` its
standard deviation: plot `rate ± rate_std` as in the figure above.

**Did we allow enough bins?** `max_boundaries` is the most bin boundaries
considered. Check that the probability of the largest number is negligible;
if not, raise the limit and fit again:

```python
print(result.m_map, f"{result.m_posterior[-1]:.4f}")   # 7 0.0026: most probable 7 boundaries; 20 is plenty
```

**Did anything happen at all?** The probability of zero boundaries — one flat
rate throughout — is the probability that the stimulus did nothing:

```python
print(f"{result.m_posterior[0]:.1e}")   # 1.7e-76
```

**Where does the rate change?** `result.boundary_posterior[k]` is the
probability that a bin ends right after interval k, averaged over everything
else. Summed over a window, it is the expected number of changes in that
window:

```python
b = result.boundary_posterior
for lo, hi in [(75, 85), (125, 135), (375, 385)]:
    print(lo, hi, round(float(b[lo + 100:hi + 101].sum()), 2))
# 75 85 1.08     the onset at 80 ms: found, and placed to within a few ms
# 125 135 0.05   the burst's end at 130 ms (0.08 -> 0.05): too small a step to be sure of
# 375 385 0.93   the offset at 380 ms: found
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/boundaries-dark.png">
  <img alt="The probability that a bin ends at each time: a tall narrow peak of about 0.85 at the 80-ms onset, a small spread-out bump near 380 ms, low values elsewhere, and almost nothing at 130 ms." src="img/boundaries-light.png">
</picture>

Note that the most probable number of boundaries, 7, is more than the 3 real
changes: some fluctuations are ambiguous, and the model keeps them in play with
some probability. Don't read the number of boundaries as the number of real
changes; read the boundary probabilities, which are high only where a change is
well supported.

## 6. Tutorial 2: event counts with exposure

A counter records events every day, but it runs for a different number of
hours each day. The rate we want is events per hour; the hours are the
*exposure*.

```python
from bayesbin import PoissonModel

rng = np.random.default_rng(2)
hours_on = rng.uniform(5, 24, 120)             # exposure: hours the counter ran each day
true_rate = np.where(np.arange(120) < 70, 2.0, 5.0)   # events per hour
counts = rng.poisson(true_rate * hours_on)

result = fit(PoissonModel.weak_prior(counts, hours_on), max_boundaries=6)
print(result.rate[[10, 100]].round(2), result.rate_std[[10, 100]].round(2))   # [2.08 5.1 ] [0.1  0.09]
```

The rate comes out per unit of exposure (per hour), whatever the exposure of
each day. Without an exposure, every interval counts as one unit.

`PoissonModel.weak_prior` picks a mild prior centred on the overall rate, worth
about one event; it lets the data speak. When did the rate change?

```python
day = int(np.argmax(result.boundary_posterior))
print(day, round(float(result.boundary_posterior[day]), 3))   # 69 1.0: after day 69, certainly
print({m: round(float(p), 2) for m, p in enumerate(result.m_posterior) if p > 0.01})
# {1: 0.16, 2: 0.6, 3: 0.19, 4: 0.05}
```

A boundary after interval k means the new rate starts at k + 1 (indexes count
from 0): here the change comes between day 69 and day 70, and it is certain. The model also gives some probability to
a second, weaker boundary elsewhere, but its location is so spread out that no
single day gets more than a few per cent.

## 7. Tutorial 3: success rates

The same Bernoulli model works for any "successes out of trials" data: here,
visitors and buyers per hour of the day.

```python
rng = np.random.default_rng(3)
visitors = rng.integers(200, 400, 24)          # per hour of the day
p = np.where((np.arange(24) >= 9) & (np.arange(24) < 18), 0.06, 0.03)
buyers = rng.binomial(visitors, p)

result = fit(BernoulliModel(buyers, visitors - buyers, sigma=1, gamma=1), max_boundaries=5)
print(result.rate[[3, 12, 21]].round(3), result.rate_std[[3, 12, 21]].round(3))   # [0.031 0.062 0.031] [0.004 0.005 0.005]
print([(h, round(float(x), 2)) for h, x in enumerate(result.boundary_posterior) if x > 0.2])
# [(7, 0.34), (8, 1.0), (17, 0.81)]: a change at 9:00 (right after hour 8; perhaps at 8:00), and at 18:00
```

**Mind the prior here.** `BernoulliModel`'s default prior (`sigma=1,
gamma=32`) expects small probabilities, around 1/33: right for spikes per
millisecond, wrong for conversion rates or failure rates of 10–90%. For those,
use the flat prior `sigma=1, gamma=1`. The prior is worth `sigma + gamma`
imaginary trials per bin, so with hundreds of real trials it barely matters;
with few trials it does.

## 8. Tutorial 4: a daily profile

To learn the shape of a typical day — traffic, calls, arrivals by time of day
— don't feed bayesbin weeks of data as one long series: the same shape would
have to be re-learnt every day, with many boundaries, at great cost. Fold it
instead: add the days up, slot by slot, and use the number of days as the
exposure. Missing data fit in naturally: a slot seen on fewer days has less
exposure.

```python
rng = np.random.default_rng(4)
slot = np.arange(288)                          # 5-minute slots of a day
profile = np.select([slot < 84, slot < 108, slot < 216, slot < 252], [0.5, 2.0, 3.0, 5.0], 1.0)
observed = rng.random((30, 288)) > 0.1         # 30 days; about 10% of slots missing
per_day = np.where(observed, rng.poisson(profile, (30, 288)), 0)

counts = per_day.sum(axis=0)                   # fold: add the days up, slot by slot
days_seen = observed.sum(axis=0)               # exposure: how many days each slot was seen
result = fit(PoissonModel.weak_prior(counts, days_seen), max_boundaries=15)
print(result.m_map, result.rate[[40, 100, 150, 230, 270]].round(2))   # 4 [0.49 1.98 2.99 4.98 1.08]
print([(int(k), round(float(x), 2)) for k, x in enumerate(result.boundary_posterior) if x > 0.5])
# [(83, 1.0), (107, 0.94), (215, 1.0), (251, 1.0)]: all four changes, to the slot
```

The rate is per slot per day. This treats the days as independent repeats of
one profile. If some days are unusual (holidays) or the level drifts over the
weeks, fit the shape on a recent window and model the level separately.

## 9. Choosing the settings

There are few, and the defaults are sensible.

- **`max_boundaries`** (default 10): the most bin boundaries considered. Too
  low, and a real change is forced out; the sign is that `m_posterior[-1]` is
  not small. Raise it until `m_posterior[-1]` is below about 0.001. The time
  grows in proportion to it.
- **The prior of each bin's rate.**
  - `BernoulliModel(s, g, sigma=1, gamma=32)`: a Beta(σ, γ) prior on each bin's
    probability, mean σ / (σ + γ), worth σ + γ imaginary trials. The default
    suits spikes per millisecond; use `sigma=1, gamma=1` for proportions that
    could be anything.
  - `PoissonModel.weak_prior(y, e, weight=1)`: a Gamma prior centred on the
    overall rate, worth `weight` events. Or give it yourself:
    `PoissonModel(y, alpha, beta, e)` with mean α / β, worth α events.
- **`m_mass`** (default: average over every number of boundaries, as the paper
  recommends): `m_mass=0.9` averages over the smallest range of numbers of
  boundaries holding 90% probability, and `m_mass=0.0` uses the most probable
  number alone (as the original program does). The default is best for the
  rate; the others are mainly for comparing with other tools.
- **`keep_bins=True`**: also return `bin_posterior`, the probability of every
  possible bin [a, b]. It needs memory for a T × T table (T intervals), so keep
  it to T of a few thousand.
- **`exact=True`**: do everything in the slow, plain way (the reference the
  tests use). You will not need it.

## 10. Reading the results

`fit` returns a `BinningResult`:

| field | what it is |
|---|---|
| `rate` | the estimated rate in each interval: its posterior mean, averaged over all binnings |
| `rate_std` | its posterior standard deviation: the error bar |
| `boundary_posterior` | for each interval k (but the last), the probability that a bin ends right after it; summed over a window, the expected number of changes there |
| `m_posterior` | the probability of each number of boundaries, 0 to `max_boundaries` |
| `m_map` | the most probable number of boundaries |
| `log_evidence` | log P(data given each number of boundaries), for comparing models |
| `log_marginal` | log P(data), the prior over the number of boundaries included |
| `bin_posterior` | the probability of each bin [a, b], if `keep_bins=True` |

The units of `rate`: for `BernoulliModel`, a probability per trial per
interval; for `PoissonModel`, events per unit of exposure (per interval, if you
gave none).

## 11. Size and speed

The cost grows with the square of the number of intervals T and in proportion
to `max_boundaries`. Memory grows only in proportion to both. On a 2012 laptop
(4 cores):

| T (intervals) | max_boundaries | NumPy only, 1 core | with `fast`, 1 core | with `fast`, 4 cores |
|---|---|---|---|---|
| 600 | 10 | 0.05 s | 0.02 s | 0.01 s |
| 2016 (a week of 5-minute slots) | 30 | 0.5 s | 0.3 s | 0.1 s |
| 8064 (four weeks) | 120 | 8.7 s | 5.8 s | 2.0 s |
| 12096 (six weeks) | 120 | 19 s | 13 s | 4.3 s |

- With `fast`, it uses all cores. Set `NUMBA_NUM_THREADS` to the number of
  *physical* cores for the best time (hyperthreads don't help).
- Many short series: fit them in parallel from a thread pool, one thread each.
- **Very strong evidence is slower.** Counts in the hundreds or thousands per
  interval, or hundreds of trials with sharp changes, make the numbers span an
  enormous range, and much of the work then falls back to a slower exact path:
  up to about 10 times slower (a week of 5-minute slots at 1000 events per slot:
  about 1 s on 4 cores). Coarser intervals help, as does folding (Tutorial 4).
- Long series of a repeating pattern: fold them (Tutorial 4). It is faster and
  it is the better model.

## 12. Pitfalls and questions

**"more than one spike in an interval: use a finer discretization".**
`spike_counts` assumes at most one spike per trial per interval (the paper's
setting). Use finer intervals, or count spikes per interval yourself and use
`PoissonModel` on the totals.

**My counts are bursty.** See overdispersion in [section 3](#3-is-it-right-for-my-data):
clumps look like changes of rate. Coarser intervals, or de-duplicating the
events, help; the error bars will otherwise be too narrow and the boundaries
too many.

**The estimate curves near the ends.** There are fewer data on the far side of
a boundary near the ends of the axis, so the estimate there is less certain;
the error band shows it.

**The estimate looks smooth, although the model is steps.** That is the
average over binnings: where the data don't say exactly where a step is, the
average of many step positions is a ramp. Where they do, it is a sharp step.

**Why not a kernel smoother?** It has a bandwidth — the same fixed-width
problem as a histogram — and no error bars or change points of its own.

**Why not Bayesian Blocks (astropy)?** It finds the *single* best set of bins,
with a penalty per bin that you choose. bayesbin averages over all of them,
which gives error bars, change-point probabilities, and no penalty to tune.

**Non-integer counts** (e.g. weights) work, on a slower path. **Intervals of
unequal width**: use `PoissonModel` with the widths as exposure.

**More.** The method: [the paper](https://papers.nips.cc/paper_files/paper/2007/hash/b73ce398c39f506af761d2277d853a92-Abstract.html)
and [docs/NOTES.md](NOTES.md) (related papers, the 2-D question, performance
notes). The code: [README](../README.md).
