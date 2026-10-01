# Changelog

## Unreleased

- Overdispersed (bursty) counts in streams: `ChangePointStream.negbinomial`
  gives each segment a negative binomial likelihood of a known dispersion,
  still conjugate (NB(r e, p) with a Beta prior on p, matched to the Gamma
  prior's mean and variance for every r, and tending to `.poisson` as r grows);
  `ChangePointStream.overdispersed` returns a `ChangePointMixture`, a bank of
  them over a grid of dispersions (as Worldwatch's Layer-0 count model's),
  weighted by their marginal likelihoods, with the same interface and
  `dispersion_posterior()`. On overdispersed counts its PIT is uniform where
  the Poisson segments' is not (tested); exact against enumeration.
- A hazard learnt from the data: `hazard_strength=a` (any `ChangePointStream`,
  and `.overdispersed`) puts a Beta(a, a(L - 1)) prior on the hazard (mean
  1/L, L = `expected_run_length`), and the recursion runs exactly over the
  joint of the run length and the number of change points so far.
  `hazard_posterior()` gives its posterior mean and sd. Exact against
  enumerating every segmentation; a strong prior equals the fixed hazard;
  segments of mean length 40 under a prior centred on 1/1000: 0.0239 ± 0.0025.
  About twice the fixed hazard's cost per interval. The fixed hazard is
  unchanged (bit for bit).
- `ChangePointStream` is 3-4x faster per `update()` and 2-4x per `pit()` then
  `update()` (i7-3612QM: 170 µs an update with ~250 components, was 540). The
  state is kept in decreasing run length, so the runs of a bucket are
  neighbours and merging needs no sort; one count at a time has its own paths
  (log-sum-exp, the pmf, the checks); the components are kept between `pit()`
  and `update()`; Beta-posterior segments (negative binomial, Bernoulli) get
  their cdf from the ratios of successive pmf terms. The Poisson pmf uses
  `-A log1p(e/B)` for `A log(B/(B+e))`, which lost ~1e-16 B/e of its value
  (2.5e-12 at B ~ 1e4; now 3e-15). Results agree with 0.3.0's to rounding:
  the same components, log marginals within 7e-15 relative, PITs within 4e-10.

## 0.3.0 (2026-09-27)

- `ChangePointStream`: its state is bounded. Run lengths up to `exact_recent`
  (128) are kept exactly; older ones are merged within buckets of log length
  (`merge_bins`, 32 per octave) into moment-matched components, so the state
  grows with the log of the current segment's length. 30,000 quiet intervals:
  313 components instead of 30,000, 17x faster. Against every run kept: at
  most 3e-4 relative in the sd and 1e-5 in the rate and the PIT (counts of ~500
  with frequent changes), typically 1e-5 or less.
  `merge_bins=None` keeps every run. `run_length_ranges()` gives the state as
  kept; `run_length_posterior()` spreads a merged component over its lengths.
  Merging is on by default, so results differ slightly from 0.2.0's;
  `merge_bins=None` gives 0.2.0's exactly.
- User Guide: "How it compares with other methods" (what kind of learning it
  is, the functions it suits, the alternatives, measured against data-chosen
  and oracle histograms and kernels: `tools/compare_methods.py`); section 1's
  figure explained (a chance excess of spikes, and what a ±1 sd band means);
  spike trains and PSTHs introduced for readers outside neuroscience.

## 0.2.0 (2026-09-27)

Data that arrive over time:

- `OnlineBinning`: the batch model's forward programme one interval at a time
  (or a few). Exactly what `fit` gives for the latest interval: the evidence of
  every number of boundaries, the rate now and its sd, where the current bin
  started, and the predictive distribution of the next count; `fit()` for the
  smoothed past.
- `ChangePointStream`: Bayesian online change-point detection (a constant
  hazard; the same Beta and Gamma segment models) for streams without end: the
  rate now, the probability of a recent change, the run-length posterior, and
  the next count's pmf, cdf and randomized PIT (calibrated surprise).

Accuracy and speed of the batch fit:

- Data with strong steps (counts in the hundreds, many trials, sharp changes):
  the forward pass scales against every interval as its own bin, and its exact
  fallback is one pass per column, skips negligible rows and runs in parallel;
  up to 10x faster (T=4800 with 16 strong steps: 31 s -> 3.2 s).
- The rate and its sd are divided by the computed coverage: the sd no longer
  loses digits where a rate is large and tightly known (1e-6 -> 1e-10, against
  a long-double reference, `tools/longdouble_reference.py`).
- The fused kernels' results are bit-identical on any number of threads; they
  release the GIL (fits in parallel from a thread pool).

Documentation: the User Guide (six tutorials), `llms.txt`, this changelog.
The C++ binsdfc-fb in the repository has the same strong-data scaling, exact
fallback and coverage fix (not part of the package).

## 0.1.0 (2026-09-27)

First release: exact Bayesian binning (Endres, Oram, Schindelin & Földiák,
NIPS 2007) with Bernoulli and Poisson likelihoods, the forward-backward bin
posterior in O(M·T²) time and O(T·M) memory, optional numba kernels (the
`fast` extra), tested against the original binsdfc program, brute-force
enumeration and the exact log-space computation.
