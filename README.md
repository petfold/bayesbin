# bayesbin

Exact Bayesian binning of rates in NumPy/SciPy, after

> D. Endres, M. Oram, J. Schindelin, P. Földiák (2008). *Bayesian binning beats
> approximate alternatives: estimating peri-stimulus time histograms.*
> Advances in Neural Information Processing Systems 20, 393–400. MIT Press.
> ([paper/](paper/))

A rate on T ordered intervals is modelled as piecewise constant with M bin
boundaries. Boundary positions, per-bin rates (conjugate priors) and M itself
are all integrated out exactly: one forward dynamic programme gives the
evidence of every M in O(M·T²). A matching backward programme gives the
posterior of every candidate bin at once, from which the predictive rate, its
error bars and the posterior over boundary positions follow.

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

## Models

| model | per interval | per-bin prior | use |
|---|---|---|---|
| `BernoulliModel(s, g, sigma, gamma)` | s trials with an event, g without | f ~ Beta(σ, γ) | the paper's PSTH |
| `PoissonModel(y, alpha, beta, e)` | count y over exposure e | λ ~ Gamma(α, β) | event counts per window |

`BernoulliModel` defaults to σ = 1, γ = 32, the original program's default.

## Verification

`pytest` (18 tests):

- **Against the original C++ program** (`binsdfc` 0.1, in [reference/](reference/)),
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
  thousands of nats, with the underflow fallback forced.
- One-bin evidences against direct numerical integration; every interval
  covered by exactly one bin; the simulated response onset recovered.

## Status and limits

- Cost: O(M·T²) time and O(T²) memory. Each step of the forward and
  backward programmes is one matrix–vector product, and the bin posterior one
  matrix product, on precomputed exponentials. Wherever underflow could cost
  more than 10⁻¹³ (relative, evidences) or 10⁻¹⁴ (absolute, bin posterior),
  that entry is recomputed exactly in log space, and `exact=True` does
  everything that way. See the timings below.
- Not yet ported from the original: latency posteriors, signal separation
  levels, hyperparameter optimisation (`-P`), bin-boundary position posteriors
  for a fixed M (`-p`).
- Planned: cyclic profiles (a bin may wrap round the end of a day or week);
  2-D via recursive partitions (see [docs/NOTES.md](docs/NOTES.md)).

## Speed against the original

binsdfc 0.1 unmodified (`g++ -O2 -fopenmp`; its own flags `-march=native
-ffast-math` made no real difference), against bayesbin with NumPy 2.5 and
OpenBLAS. Intel i7-3612QM (4 cores, 2 hyperthreads each); "1 core" means one
physical core, and "4 threads" four separate physical cores. binsdfc is timed
as a process, bayesbin in-process without the import.

| case | binsdfc, 1 core | binsdfc, 4 threads | bayesbin, 1 core | bayesbin, 4 threads |
|---|---|---|---|---|
| T=300, M≤10, rate ± sd | 0.98 s | 0.27 s | 0.04 s | 0.04 s |
| T=600, M≤10, rate ± sd | 50.0 s | 12.5 s | 0.15 s | 0.15 s |
| T=600, M≤10, evidence only | 0.07 s | — | 0.08 s | — |
| T=2016, M≤30, evidence only | 2.0 s | — | 0.69 s | — |
| T=2016, M≤30, rate ± sd | stopped after 26 min | | 1.5 s | 1.5 s |

- The evidences use the same dynamic programme in both, so they take about
  the same time. At T=2016 bayesbin's matrix-vector steps are about 3× faster
  than binsdfc's triple loop.
- For the rate and its error bars binsdfc uses the paper's virtual-spike
  device: for every time point it reruns the whole programme twice (rate and
  second moment), O(M·T³). bayesbin gets every time point from one backward
  pass, O(M·T²). The gap is the algorithm, not the language.
- binsdfc fixes 4 OpenMP threads over time points (`omp_set_num_threads(4)`)
  and scales almost 4×. bayesbin gains nothing from threads: its remaining time
  is single-threaded element-wise work on T×T arrays (the bin evidences, the
  gain matrix, the moments), not the matrix products.

## Licence

Not chosen yet. The code in `src/` was written from the paper; the original
program was used only as a reference to test against. `reference/binsdfc-0.1/`
is the original, GPL-2.0-or-later, kept unmodified with its provenance in
[reference/README.md](reference/README.md).
