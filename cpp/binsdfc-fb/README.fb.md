# binsdfc-fb: binsdfc 0.1 with a forward–backward SDF

A modified copy of Dominik Endres's binsdfc 0.1 (GPL-2.0-or-later, as the
original; unmodified original in `../../reference/binsdfc-0.1/`).

## What changed (2026-09-26)

- `forwardbackward.h/.cpp` (new):
  - `fb::fastForward`: the central iteration (the paper's subE[]) for all m as
    matrix-vector products. Relative to the one-bin evidence, a step is a sum
    of exp(φ[r]) · exp(G[r][k]) with G the gain of a boundary at r, and
    exp(G − column max) is computed once: T²/2 multiply-adds per step, no exp
    or log in the inner loop. A term lost to underflow is below DBL_MIN, so any
    column whose sum could be off by more than 1e-13 (relative) is recomputed
    exactly with a log-sum-exp. The gain matrix is stored as a packed upper
    triangle; now stored block-major instead: for each block of 64
    columns its rows are contiguous and padded (zeros where k ≤ r), built in
    64×64 tiles (cache-friendly whether the bin evidences are read along rows
    or, for the reversed pass, along columns). The steps loop over column
    blocks outside and over m inside: block [k0,k1) at step m needs φ_{m-1}
    only for r < k1, which is final, so a block stays in cache for all m
    instead of the whole matrix streaming from memory once per m; the scale
    is a per-block running maximum (consistent within each column).
  - `forwardBackward`: the SDF and its second moment at every time index in one
    pass, O(M·T²). The original obtains each time index as a ratio of evidences
    with a virtual spike added there (section 4 of the paper), one run of the
    central iteration per index and moment, O(M·T³). The backward messages are
    the forward iteration on the reversed time axis; the average over M is
    folded into them; the sum over a bin's index is a scaled product, with the
    same bound (1e-14 absolute per bin probability) and exact recomputation.
    Reuses the evidences' forward iteration when the data are unchanged.
    The bin posterior runs row by row over the bin start a, in simple
    vectorised loops: the sum over the bin index as contiguous multiply-adds
    along b, the posterior moments from per-length reciprocal tables (every
    interval holds one spike or gap per trial, checked; else divided), `exp`
    vectorised (`vmath.cpp`). 0.134 s → 0.073 s at T=2016, M≤30, one thread.
    OpenMP throughout.
- `spikecounter.h/.cpp`: the bin evidences log B(s+σ, g+γ) from `lgamma`
  tables indexed by the integer counts, and the counts from prefix sums (see
  Memory below); `mDataVersion`, so the central iteration of the evidences can
  be reused by the SDF.
- Memory (lean mode, the default with the plain Beta prior): no T×T array.
  - Per-bin counts and log evidences come from prefix sums and the `lgamma`
    tables (`spikeCounter::countsOf`, `iecOf`); `mIntervalCounts` and
    `mIntervalEvidences` are built only when one of the original functor paths
    needs them (`ensureTables()`, e.g. for `-b`, `-p`, `-y`, `-L`).
  - The (never read) spike co-occurrence counts are not kept.
  - The gain matrix exists one column block at a time (T × 64 doubles), built
    just before that block's m-loop. Completed groups of 256 rows get a fixed
    scale and are exponentiated once per step, so a block combines them with
    one multiply per row instead of an `exp`.
  - The bin posterior runs in tiles (64 bin starts × 512 bin ends), so a chunk
    of the right-hand factors stays in cache for a whole group of rows.
  - `forwardBackward` reads the cached forward iteration and the reversed one
    in place instead of copying them.
  - Peak memory at 6 weeks (T = 12096, M ≤ 120): 2.1 GB → 61 MB; output
    byte-identical.
- The functor paths (`-b`, `-p`, `-y`, `-L`, …) use exact log-additions
  (`logAddR`) instead of the table lookup when the new paths are on, as do the
  evidence sums over M, so their averages are as exact as the evidences they
  are divided by. `-b`'s boundary standard deviations now match those computed
  from `-p`'s posterior to the printed digits; the original's differ by up to
  1.4e-4 (the table lookup, through E[k²] − E[k]²).
- `evidencecomputer.cpp`: `evidenceComputer<evifunc>::computeEvidences`
  (the plain evidences) uses `fb::fastForward` with the plain Beta prior; the
  original table-lookup iteration otherwise.
- `spikedensityfunction.h/.cpp`: `getSDFForwardBackward()`.
- `binsdfc.cpp`: the SDF uses it by default, on `OMP_NUM_THREADS` threads (else all);
  the original's fixed `omp_set_num_threads(4)` now applies to the `-V` path only. `--virtual-spike` / `-V` switches
  every new path off, so the output is the original's, bit for bit. Output
  format unchanged.
- `vmath.cpp` (new): the vectorised `exp`, alone compiled with `-ffast-math`.
  Below −700 it returns exactly 0 (libmvec leaves its fast path for large
  negative arguments), so the underflow bounds use LOST = 1e-304 per dropped
  term instead of DBL_MIN. The hot loops run with flush-to-zero /
  denormals-are-zero per thread (`fb::FlushDenormals`): subnormal products are
  below that bound, and slow on x86.
- `CMakeLists.txt`: the new source files; `-ffast-math` off globally (see Build).

With the upper bound on the firing probability (the incomplete-Beta prior) the
original paths are used, as before.

## Build

`vmath.cpp` alone is compiled with `-ffast-math`, which is when glibc declares
its vector `exp`; everything else, and the link, without it:

    g++ -O2 -fopenmp -ffast-math -c vmath.cpp
    g++ -O2 -fopenmp -fno-math-errno -std=c++14 -o binsdfc binsdfc.cpp evidencecomputer.cpp \
        logadd.cpp simplexminimiser.cpp spikecounter.cpp spikedensityfunction.cpp \
        forwardbackward.cpp vmath.o

or `cmake . && make`. The original's CMake flags put `-ffast-math` on every
file; that is removed, because the new code relies on −∞ arithmetic, which
`-ffinite-math-only` lets the compiler assume away.

## Agreement

- `-V` reproduces the original output exactly (tests/data/testdata_seed1.txt).
- The evidences print the same digits as the original's.
- The SDF and its standard deviation agree with bayesbin to the 6 printed
  digits, on that dataset and on one that exercises the underflow bounds (400
  trials, rates switching 0.02/0.3: evidences spanning ~10⁴ nats), and with
  the original within 2.5e-5 (the original's table-lookup log-add).
