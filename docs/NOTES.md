# Notes

## The papers

- D. Endres, M. Oram, J. Schindelin, P. Földiák (2008). *Bayesian binning beats
  approximate alternatives: estimating peri-stimulus time histograms.* NIPS 20,
  393–400. — the method implemented here ([paper/](../paper/)).
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
  T=12096.
- In numba, a loop that branches on the model per element (one kernel for
  Bernoulli and Poisson) ran 8× slower than the same loop for one model. Each
  kernel now dispatches once to an inlined body with the model code as a
  constant.
- Without SVML (numba) or AVX-512 (NumPy), exp is scalar, about 12 ns on this
  laptop: it and the lgamma table lookups set the floor of the element-wise work.

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

## First application: Worldwatch

Worldwatch (global anomaly detection over open data streams) models event
counts per window with hour-of-day and day-of-week factors fixed at 24 and 7
slots. Planned use, offline first:

- daily/weekly rate profiles per stream and cell with `PoissonModel` — days as
  trials, time of day as the peri-stimulus axis, the stream's fixed archive bins
  as the candidate boundaries; the profile's uncertainty then feeds the
  predictive distribution;
- caveat: days are not independent repeats (rates drift, news days differ), so
  the profile gives the shape, fitted on a recent window, while Worldwatch's own
  forgetting rate and burstiness posterior keep the level;
- later: change points in the archive, spatial resolution over the H3 tree,
  mutual information between streams.

Success criterion before anything goes live: better calibration (PIT
uniformity) of the count models on the archive than the fixed hourly factors.
