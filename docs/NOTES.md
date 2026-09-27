# Notes

## Plan

1. **Publish on PyPI** as `bayesbin` (the name was free on 2026-09-27), with
   the `fast` extra (numba). The workflow is in place
   (`.github/workflows/publish.yml`, trusted publishing, no stored token):
   - on pypi.org, add a pending trusted publisher: project `bayesbin`, owner
     `petfold`, repository `bayesbin`, workflow `publish.yml`, environment
     `pypi` (the operator does this, with their own PyPI account);
   - check the version in `pyproject.toml`, then `git tag v0.1.0 && git push
     origin v0.1.0`: the workflow runs the tests, builds, checks and uploads.
     A version can never be re-uploaded, only yanked;
   - afterwards: the README's install lines become `pip install bayesbin` /
     `pip install "bayesbin[fast]"`, and a PyPI badge goes next to the others.
2. **Cyclic profiles**: a bin may wrap round the end of a day or week.
3. **Ports from binsdfc**: latency posteriors, signal separation levels,
   hyperparameter optimisation, boundary position posteriors for a fixed M.
4. **2-D** via recursive partitions (below: where the method stops being 1-D).
5. **The last sequential step**: the steps inside each 256-column block of the
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
  optional numba kernels for the element-wise work.

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
