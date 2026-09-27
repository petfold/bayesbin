# bayesbin

[![tests](https://github.com/petfold/bayesbin/actions/workflows/tests.yml/badge.svg)](https://github.com/petfold/bayesbin/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-BSD--3--Clause-blue)](https://github.com/petfold/bayesbin/blob/main/LICENSE)
[![status](https://img.shields.io/badge/status-beta-yellow)](#status-and-limits)

Exact Bayesian binning of rates in NumPy/SciPy, after

> D. Endres, M. Oram, J. Schindelin, P. Földiák (2008). *Bayesian binning beats
> approximate alternatives: estimating peri-stimulus time histograms.*
> Advances in Neural Information Processing Systems 20, 393–400. MIT Press.
> ([NeurIPS page](https://papers.nips.cc/paper_files/paper/2007/hash/b73ce398c39f506af761d2277d853a92-Abstract.html); a copy in [paper/](https://github.com/petfold/bayesbin/tree/main/paper))

A rate on T ordered intervals is modelled as piecewise constant with M bin
boundaries. Boundary positions, per-bin rates (conjugate priors) and M itself
are all integrated out exactly: one forward dynamic programme gives the
evidence of every M in O(M·T²). A matching backward programme gives the
posterior of every candidate bin at once, from which the predictive rate, its
error bars and the posterior over boundary positions follow.

**New to this? Start with the [User Guide](https://github.com/petfold/bayesbin/blob/main/docs/USER_GUIDE.md)**: why fixed-width
bins mislead, the assumptions in plain words, four worked examples (spike
trains, counts with exposure, success rates, a daily profile) and the pitfalls.
For AI coding assistants there is a compact [llms.txt](https://github.com/petfold/bayesbin/blob/main/llms.txt).

```sh
pip install bayesbin              # NumPy/SciPy only
pip install "bayesbin[fast]"      # + numba kernels: ~2x faster, all cores
```

```python
from bayesbin import BernoulliModel, PoissonModel, fit, spike_counts

# spike trains, as in the paper: one list of integer spike times per trial
s, g = spike_counts(trials, t_start=-100, t_end=499)
r = fit(BernoulliModel(s, g, sigma=1.0, gamma=32.0), max_boundaries=10)
r.rate, r.rate_std          # predictive firing probability per interval, ± 1 sd
r.m_posterior               # P(M | data)
r.boundary_posterior        # P(a bin ends at interval k | data)

# counts per window, several events per window allowed, with exposure
r = fit(PoissonModel.weak_prior(counts, exposure), max_boundaries=20)
```

By default predictions average over every M, as the paper recommends;
`m_mass=0.9` restricts them to the credible range of M, which is what the
original program does.

NumPy and SciPy are all it needs. With the `fast` extra (`pip install
bayesbin[fast]`: numba and threadpoolctl) the work runs in fused kernels on all
cores: about 2× faster on one core, and scaling to about 3× more on four. The
first call in a new environment compiles them (about a minute, once; cached
afterwards), and `BAYESBIN_NUMBA=0` switches them off. Both paths give the same
results to rounding, and the fused ones the same bits on any number of threads.

Threads: numba's, `NUMBA_NUM_THREADS` or `numba.set_num_threads(n)`; the
default is every logical CPU, and hyperthreads gain nothing here, so set it to
the number of physical cores for the best time. While a fit runs, the fused
path holds BLAS to one thread and splits the matrix products over numba's
threads itself. The kernels release the GIL, so many separate fits (many
short series) can also run in parallel from a thread pool, each on one numba
thread (under numba's TBB or OpenMP threading layer).

## What it is for

Any rate that varies along an ordered axis and is observed as events per
interval, where you want the rate *and* its uncertainty without choosing bin
widths by hand:

- **Peri-stimulus time histograms** (the paper's case): spike trains over
  repeated trials, the firing probability per millisecond, with error bars,
  from a few dozen trials.
- **Event counts per window**: arrivals, requests, incidents, photon or
  particle counts, cases per week — with an exposure per window (observation
  time, population, detector area) when windows differ.
- **Proportions along an axis**: successes out of trials per interval
  (conversion or failure rates by time of day, by age, by dose), with the
  Bernoulli model.
- **Change points**: `boundary_posterior` is the posterior probability that the
  rate changes after each interval, averaged over every segmentation.
- **Periodic profiles**: daily or weekly shapes, with the days (or weeks) as
  trials and the time of day as the axis.

Nothing is fitted by optimisation and nothing is sampled: every segmentation
into up to M + 1 bins, and every M, is summed exactly. Compared with
Bayesian Blocks (Scargle et al. 2013, ApJ 764:167; `astropy.stats.bayesian_blocks`),
which finds the single best segmentation under a penalty per block, this
averages over all segmentations, so the rate is smooth where the data do not
decide where a step is, and comes with error bars.

The axis is 1-D; see [docs/NOTES.md](https://github.com/petfold/bayesbin/blob/main/docs/NOTES.md) for how far this extends
to two dimensions.

## Models

| model | per interval | per-bin prior | use |
|---|---|---|---|
| `BernoulliModel(s, g, sigma, gamma)` | s trials with an event, g without | f ~ Beta(σ, γ) | the paper's PSTH |
| `PoissonModel(y, alpha, beta, e)` | count y over exposure e | λ ~ Gamma(α, β) | event counts per window |

`BernoulliModel` defaults to σ = 1, γ = 32, the original program's default.

## Verification

`pytest` (41 tests; 5 need `cpp/binsdfc-fb` built, 7 need the `fast` extra):

- **Against the original C++ program** (`binsdfc` 0.1, in [reference/](https://github.com/petfold/bayesbin/tree/main/reference)),
  on a seeded dataset in its own input format (`tools/make_testdata.py`):
  - log P(D | M) for M = 0..10 and the marginal likelihood agree to every
    printed digit;
  - the predictive rate agrees within 2 × 10⁻⁵ relative;
  - its standard deviation agrees within 5 × 10⁻⁵. The original adds in log
    space through an interpolated lookup table, which shows at that level.
- **Against brute-force enumeration** of every boundary configuration, for both
  models: evidence and predictive rate to 10⁻¹⁰.
- **Against the paper's own device** (§4): P(spike | k) as the ratio of
  evidences with and without a virtual spike at k equals the forward–backward
  result for every k.
- The fast paths against the exact log-space ones, on data whose evidences span
  thousands of nats, with the underflow fallback forced; the fused kernels
  against the NumPy path and the exact one, for both models (constant and
  varying exposure).
- **Against a long-double reference** (`tools/longdouble_reference.py`: the
  whole computation in 80-bit arithmetic, the variance in its stable form),
  on steps strong enough that the sd, √(E[f²] − E[f]²), cancels: every path's
  rate to 10⁻¹², its sd to 10⁻⁹. (Both moments are divided by the computed
  coverage, the posterior of the bins covering each interval, which is 1 in
  exact arithmetic; its rounding error would otherwise reach the sd amplified
  by rate²/var.)
- The User Guide's examples run and print what the guide says they print.
- One-bin evidences against direct numerical integration; every interval
  covered by exactly one bin; the simulated response onset recovered.

## Status and limits

- Cost: O(M·T²) time, O(T·M) memory. Nothing of size T×T is formed on the
  default path: the models give bin evidences and posterior moments block by
  block (`bin_block`, from prefix sums and `lgamma` tables); the forward and
  backward programmes run in blocks of 256 columns, each block's slice of
  exponentiated gains made once from the upper triangle and serving all M
  steps (the rows before a block, final for every step, as one matrix
  product; only the rows inside it step by step); the bin posterior is
  accumulated in tiles of 256 × 1024 bins, as scaled matrix products, into the
  rates and the boundary posterior. The factors of every matrix product are
  flushed to 0 below a threshold chosen so that no product is subnormal
  (subnormal arithmetic is ~100× slower on x86, and BLAS runs without
  flush-to-zero); the error bounds count the flushed terms as lost.
  `keep_bins=True` (the whole bin posterior) and `exact=True` use T×T arrays. Wherever underflow could cost
  more than 10⁻¹³ (relative, evidences) or 10⁻¹⁴ (absolute, bin posterior),
  that entry is recomputed exactly in log space, and `exact=True` does
  everything that way. See the timings below.
- Not yet ported from the original: latency posteriors, signal separation
  levels, hyperparameter optimisation (`-P`), bin-boundary position posteriors
  for a fixed M (`-p`).
- Planned: a release on PyPI, so that `pip install bayesbin` works (steps in
  [docs/NOTES.md](https://github.com/petfold/bayesbin/blob/main/docs/NOTES.md#plan)); cyclic profiles (a bin may wrap round
  the end of a day or week); 2-D via recursive partitions (see
  [docs/NOTES.md](https://github.com/petfold/bayesbin/blob/main/docs/NOTES.md)).

## Speed against the original

binsdfc 0.1 unmodified (`g++ -O2 -fopenmp`; its own flags `-march=native
-ffast-math` made no real difference), against bayesbin with NumPy 2.5,
OpenBLAS and numba 0.67. Intel i7-3612QM (4 cores, 2 hyperthreads each; AVX,
no AVX2 or FMA); "1 core" means one physical core, and "4 threads" four
separate physical cores (`taskset`; numba, OpenMP and OpenBLAS thread counts
set to match). binsdfc is timed as a process, bayesbin in-process without the
import and after a warm-up call (the fused kernels' cache load, once per
process).

| case | binsdfc, 1 core | binsdfc, 4 threads | binsdfc-fb, 1 core | binsdfc-fb, 4 threads | bayesbin (NumPy), 1 core | bayesbin + numba, 1 core | bayesbin + numba, 4 threads |
|---|---|---|---|---|---|---|---|
| T=300, M≤10, rate ± sd | 0.98 s | 0.27 s | | | 0.018 s | 0.008 s | 0.006 s |
| T=600, M≤10, rate ± sd | 50.0 s | 12.4 s | 0.033 s | 0.021 s | 0.050 s | 0.023 s | 0.012 s |
| T=600, M≤10, evidence only | 0.065 s | — | 0.020 s | 0.017 s | 0.014 s | 0.008 s | 0.005 s |
| T=2016, M≤30, evidence only | 2.0 s | — | 0.14 s | 0.088 s | 0.14 s | 0.083 s | 0.035 s |
| T=2016, M≤30, rate ± sd | stopped after 26 min | | 0.33 s | 0.17 s | 0.51 s | 0.26 s | 0.099 s |

`binsdfc-fb` is the original with the forward–backward SDF, the matrix-vector
central iteration and table-driven bin evidences added ([cpp/binsdfc-fb/](https://github.com/petfold/bayesbin/blob/main/cpp/binsdfc-fb/README.fb.md)); best
of 5 runs. (binsdfc itself always runs 4 threads.)

- The evidences use the same dynamic programme in both. binsdfc's triple loop
  is the slowest; binsdfc-fb and bayesbin both run its central iteration in
  column blocks, bayesbin with the rows before each block as one matrix
  product for all M steps (BLAS-3), which binsdfc-fb does not have.
- For the rate and its error bars binsdfc uses the paper's virtual-spike
  device: for every time point it reruns the whole programme twice (rate and
  second moment), O(M·T³). bayesbin gets every time point from one backward
  pass, O(M·T²). The gap is the algorithm, not the language.
- binsdfc fixes 4 OpenMP threads over time points (`omp_set_num_threads(4)`)
  and scales almost 4×. bayesbin's NumPy path runs on one thread outside BLAS;
  its fused path runs every step on numba's threads (the matrix products split
  into fixed parts) except the steps inside each 256-column block of the
  dynamic programme, which are sequential in M, and scales about 2.9× on 4
  cores.

### Larger problems

30 trials of a synthetic daily profile (5-minute slots: night, morning ramp,
day, evening peak, plus a 2-hour burst each week; `tools/make_longdata.py T`),
rate ± sd, most probable M only (`-l 0`, `m_mass=0.0`), peak memory from
`/usr/bin/time` and `getrusage`. All but the 12-week row run back to back, one
run each; the laptop was thermally throttled.

| T | M ≤ | binsdfc-fb, 1 core | binsdfc-fb, 4 threads | binsdfc-fb memory | bayesbin (NumPy), 1 core | bayesbin + numba, 1 core | bayesbin + numba, 4 threads |
|---|---|---|---|---|---|---|---|
| 2016 (1 week) | 30 | 0.25 s | 0.10 s | 12 MB | 0.54 s, 88 MB | 0.27 s, 175 MB | 0.11 s, 177 MB |
| 4032 (2 weeks) | 60 | 1.44 s | 0.45 s | 19 MB | 1.99 s, 110 MB | 1.18 s, 189 MB | 0.53 s, 192 MB |
| 8064 (4 weeks) | 120 | 9.2 s | 2.9 s | 44 MB | 8.7 s, 179 MB | 5.8 s, 235 MB | 2.0 s, 242 MB |
| 12096 (6 weeks) | 120 | 20.3 s | 6.3 s | 61 MB | 18.6 s, 250 MB | 13.0 s, 271 MB | 4.3 s, 280 MB |
| 24192 (12 weeks) | 120 | | 86 s | 116 MB | | | |

- binsdfc-fb memory is O(T·M): no T×T array is kept. (Before: ≈14·T²
  bytes, 2.1 GB at 6 weeks; 12 weeks would have needed ≈8 GB.) Output is
  byte-identical to the T² version at 2 and 6 weeks. bayesbin is O(T·M) too
  now (its figures include ≈60 MB of Python and NumPy, and ≈90 MB more for
  numba and LLVM); before, ≈70·T² bytes (1.1 GB at 2 weeks).
- Time grows about as T² and linearly in M; the original needs 8.5 s for the
  evidences alone at T=4032.
- binsdfc-fb and bayesbin agree to the 6 printed digits at T=4032.
- A periodic series needs M to grow with its length: the most probable M was
  106 at 2 weeks (M ≤ 120) and 297 at 6 weeks (M ≤ 400), the same daily shape
  re-learnt every day. For daily or weekly profiles, fold the series instead
  (days as trials, time of day as the axis): T = 288, a few milliseconds.

## Licence

Two licences, by directory:

- **BSD-3-Clause** ([LICENSE](https://github.com/petfold/bayesbin/blob/main/LICENSE)): the Python package `src/bayesbin` (all
  that `pip` installs), its tests, `tools/bench_vs_binsdfc.py` and the
  documentation. The package was written from the paper; the C++ program was
  used only as a reference to test against, and none of its code is in it.
- **GPL-2.0-or-later**: [reference/binsdfc-0.1/](https://github.com/petfold/bayesbin/tree/main/reference) (Dominik Endres's
  original, unmodified, with its provenance in [reference/README.md](https://github.com/petfold/bayesbin/blob/main/reference/README.md)),
  [cpp/binsdfc-fb/](https://github.com/petfold/bayesbin/blob/main/cpp/binsdfc-fb/README.fb.md) (that program with a
  forward–backward SDF and the speed-ups described there, each changed file
  marked) and `tools/make_testdata.py` (a port of its test-data script). Their
  licence text is in [cpp/COPYING](https://github.com/petfold/bayesbin/blob/main/cpp/COPYING) and
  [reference/COPYING](https://github.com/petfold/bayesbin/blob/main/reference/COPYING).
- The paper in [paper/](https://github.com/petfold/bayesbin/tree/main/paper) is copyright its authors, not under either
  licence.

## Citing

If you use this, please cite the paper:

```bibtex
@inproceedings{endres2008bayesian,
  title     = {Bayesian binning beats approximate alternatives: estimating peri-stimulus time histograms},
  author    = {Endres, Dominik and Oram, Mike and Schindelin, Johannes and F{\"o}ldi{\'a}k, Peter},
  booktitle = {Advances in Neural Information Processing Systems 20},
  pages     = {393--400},
  publisher = {MIT Press},
  year      = {2008}
}
```

The method is Endres, Oram, Schindelin and Földiák's; binsdfc, the original
C++ implementation, is Dominik Endres's; bayesbin and the binsdfc-fb changes
are by Peter Foldiak.
