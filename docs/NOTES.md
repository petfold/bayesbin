# Notes

## Plan

1. **Released on PyPI** as `bayesbin`: 0.1.0 and 0.2.0 (2026-09-27; see CHANGELOG.md), trusted publishing from
   `.github/workflows/publish.yml` (environment `pypi`, deployable from `v*` tags
   only). Next release: bump the version in `pyproject.toml`, then `git tag vX.Y.Z
   && git push origin vX.Y.Z`; a version can never be re-uploaded, only yanked.
2. **Streams without end**: `OnlineBinning` (2026-09-27) is the exact incremental
   forward pass of the batch model; `ChangePointStream` (same day) is Bayesian
   online change-point detection (Adams & MacKay 2007; Fearnhead & Liu 2007): a
   constant hazard, the same Beta and Gamma segment evidences, run lengths below
   1e-12 dropped. Exact against enumeration; calibrated PIT on data from the
   model. Its state is bounded by merging old run lengths (longer than
   `exact_recent`) within buckets of log length, `merge_bins` per octave, into
   one moment-matched component: in a long quiet stretch every run length stays
   plausible (a Bayes factor near 1 per position), and neighbouring ones differ
   by ~1/√ℓ posterior sd. At 32 per octave, against every run: 30,000 quiet
   intervals 313 components not 30,000 (17x faster); rate, sd, PIT within
   2e-6, 3e-5, 1e-6; with frequent changes in counts of ~500, 3e-6, 3e-4, 1e-5
   (16 per octave was 100x worse there). A trap on the way: bucketing a merged
   component by its *shortest* run keeps it in the first bucket, absorbing every
   run after it (errors of 60% in the sd); by its longest run it ages. Left: a
   hazard learnt from the data (a Beta prior on it), fixed-lag smoothing, the
   fused kernels for the updates.
3. **Cyclic profiles**: a bin may wrap round the end of a day or week.
4. **Ports from binsdfc**: latency posteriors, signal separation levels,
   hyperparameter optimisation, boundary position posteriors for a fixed M.
5. **2-D** via recursive partitions (below: where the method stops being 1-D).
6. **The last sequential step**: the steps inside each 256-column block of the
   dynamic programme (about 15% of the time on 4 threads). A `prange` over
   columns per step cost more in launches than it saved. Running the forward
   and backward passes at once (two threads, half of numba's each) gained only
   5-10% on 4 cores, for 14% more memory and threading-layer-specific code, and
   was reverted. A narrower block for the inside rows is the option left.

## The papers

- D. Endres, M. Oram, J. Schindelin, P. Földiák (2008). *Bayesian binning beats
  approximate alternatives: estimating peri-stimulus time histograms.* NIPS 20,
  393–400. — the method implemented here ([NeurIPS](https://papers.nips.cc/paper_files/paper/2007/hash/b73ce398c39f506af761d2277d853a92-Abstract.html)).
- D. Endres, P. Földiák (2005). *Bayesian bin distribution inference and mutual
  information.* IEEE Trans. Information Theory 51(11). — the density-estimation
  formalism the dynamic programme comes from; also mutual information.
- D. Endres, J. Schindelin, P. Földiák, M. Oram (2010). *Modelling spike trains
  and extracting response latency with Bayesian binning.* J. Physiology (Paris),
  doi:10.1016/j.jphysparis.2009.11.015. — latencies (binsdfc `-y`, `-L`).
- D. Endres, M. Oram (2010). *Feature extraction from spike trains with Bayesian
  binning: "latency is where the signal starts".* J. Computational Neuroscience,
  doi:10.1007/s10827-009-0157-3.

## Implementations

- **binsdfc 0.1** (C++, GPL-2+), D. Endres — found, in [reference/](../reference/),
  with provenance. Builds with a current g++; reproduces its documented tutorial.
- **Matlab** — reported to exist; not located yet.
- **bayesbin** (this repository) — NumPy/SciPy, written from the paper;
  optional numba kernels (fused, multi-threaded); `OnlineBinning` and
  `ChangePointStream` for data that arrive over time.
- **binsdfc-fb** (this repository, [cpp/binsdfc-fb/](../cpp/binsdfc-fb/README.fb.md))
  — binsdfc 0.1 with bayesbin's algorithms: the forward–backward bin
  posterior, O(T·M) memory, OpenMP, the saturated reference and fast exact
  fallback, coverage normalisation. GPL-2.0-or-later.

## Performance lessons (bayesbin)

- Subnormals were the largest hidden cost. The scaled matrix products (forward
  steps, bin posterior) had a few percent subnormal factors, and products of
  two small normal factors underflow too; with no flush-to-zero in OpenBLAS the
  products ran at a third to a quarter of their speed. Flushing factors below
  √TINY (forward) or below 2^-1011 after an exact 2^500 scaling (bin posterior,
  whose error bound needed the tighter cut) removes them; the bounds count the
  flushed terms as lost. That alone took the NumPy path from 30 s to 19 s at
  T=12096 (one core, an Ivy Bridge laptop).
- In numba, a loop that branches on the model per element (one kernel for
  Bernoulli and Poisson) ran 8× slower than the same loop for one model. Each
  kernel now dispatches once to an inlined body with the model code as a
  constant.
- Without SVML (numba) or AVX-512 (NumPy), exp is scalar, about 12 ns on this
  laptop: it and the lgamma table lookups set the floor of the element-wise work.
- Threads: numba's kernels and OpenBLAS's threaded products, alternating, got
  in each other's way: after each product OpenBLAS's workers spin for a while
  (`OPENBLAS_THREAD_TIMEOUT`, read only when the library loads), on the cores
  numba's threads need, so 4 threads gave 1.5×. Holding BLAS to one thread
  (threadpoolctl) and splitting each product into fixed parts on numba's own
  threads, one BLAS call per part, gave 2.9×. Parts that are too small cost
  more in OpenBLAS's per-call packing than they gain.
- Fixed parts (not one per thread), combined in a fixed order, keep the results
  bit-identical on any number of threads; a test checks it.
- Data with strong steps: the forward pass scales each column's sum by the
  largest row factor times the largest gain, which fails where the two sit on
  different rows. Relative to the one-bin evidence the row factors grow
  steeply (one bin over a step fits badly) while the gains peak at the steps,
  so on counts stepping 2 → 9 → 4 the scale overshot by a median 530 nats and
  64% of columns took the exact fallback (0.86 s for T=1200, any number of
  threads). Relative to every interval as its own bin both are nearly level:
  28% fall back. The fallback is now one pass per column (terms more than 60
  nats below the running maximum not exponentiated) and runs a step's columns
  in parallel: 0.09 s on 4 threads. With stronger steps (rates 20 → 400) most
  columns still fall back, and fast: T=4800 with 16 steps, 31 s → 3.2 s.
  Counts in the hundreds or thousands per interval are such data (a week of
  5-minute slots at 1000 per slot: 79% of columns fall back). Each row's term
  is bounded by its step value plus the column's largest gain, so rows already
  more than 60 nats below the running maximum skip the bin-evidence lookup:
  25-35% faster (that week: 1.43 s → 0.97 s on 4 threads). The bound is loose,
  and the limit is structural: even knowing the final maximum it would leave
  77% of rows to evaluate, while only 2% of terms matter. Picking those out
  safely needs pruning in the style of PELT (Killick et al. 2012), whose
  guarantee is for the penalised best segmentation, not for exact sums over
  segmentations with a given number of bins. Computing gammaln by Stirling's
  series instead of the table lookups was slower (18 ns against 9 ns a term).
- Ported to binsdfc-fb (the reference, the fallback, the coverage): strong
  spike data 0.80 s → 0.14 s on 4 threads. A trap in the port: its bin
  evidences leave out the prior's normaliser (a constant per bin), harmless in
  the gains, where it cancels, but summed over single intervals it put a linear
  trend of 3.5 nats per index into the reference and made the benchmark 11×
  slower until the reference counted it once per index.
- The sd, √(E[f²] − E[f]²), cancels where a rate is large and tightly known:
  at rate 400 ± 1.6 an error in the moments grows 6·10⁴ times. The error that
  mattered was one factor common to both moments: the computed coverage (the
  posterior of the bins covering an interval, 1 in exact arithmetic) was off
  by 2.5·10⁻¹¹ even from the exact path, as the evidences span 10⁵ nats. So
  the sd was 1e-7 to 1e-6 off in every path, the exact one included. Dividing
  both moments by the computed coverage (one more difference array) brought
  the rate from 2.5·10⁻¹¹ to 7.5·10⁻¹⁴ and the sd to 10⁻¹⁰, against a
  long-double reference; the stable two-pass variance would reach 10⁻¹¹ but
  needs a second pass over all bins.

## Where the method stops being 1-D

The exact sum needs the bins to be contiguous intervals on an ordered axis: the
evidence is a product of per-bin factors, each depending on two neighbouring
boundaries — a chain, hence the dynamic programme. Arbitrary partitions of a
plane have no such chain. Three routes:

1. **Axis-aligned grids, one axis at a time.** Given the column partition, a row
   band's factor is a product over its cells and depends only on the band's two
   boundaries, so the 1-D programme runs exactly on each axis given the other.
   Alternating gives a sampler (or optimiser) with exact steps.
2. **Unrolling** a small second axis into one ordered axis (a week of 5-minute
   slots is T = 2016) — only where the ordering means something.
3. **Recursive partitions** (each region either keeps one rate or splits into
   children): exact in any dimension, by recursion from the leaves. Cf. optional
   Pólya trees (W. H. Wong, L. Ma, Annals of Statistics, 2010). H3 hexagonal
   cells form such a tree (7 children per cell), which fits spatial rates.

## Periodic profiles

For a daily or weekly rate profile, fold the series: days (or weeks) as
trials, time of day as the axis, the finest time slots as the candidate
boundaries. `PoissonModel` then gives the profile with its uncertainty,
which can feed a forecast's predictive distribution. Caveats:

- the days are treated as independent repeats of one profile; if the level
  drifts or some days are exceptional, fit the shape on a recent window and
  model the level separately;
- unfolded, a periodic series needs M to grow with its length (the same shape
  re-learnt every period; see the README's larger problems), so folding is
  also far cheaper;
- a bin cannot yet wrap round the end of the period (planned: cyclic
  profiles).

Other directions: change points along long series, spatial rates over a
hierarchical grid (route 3 above), mutual information between streams (the
2005 paper).
