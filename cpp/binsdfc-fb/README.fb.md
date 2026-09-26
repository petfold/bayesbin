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
    triangle, filled row by row.
  - `forwardBackward`: the SDF and its second moment at every time index in one
    pass, O(M·T²). The original obtains each time index as a ratio of evidences
    with a virtual spike added there (section 4 of the paper), one run of the
    central iteration per index and moment, O(M·T³). The backward messages are
    the forward iteration on the reversed time axis; the average over M is
    folded into them; the sum over a bin's index is a scaled product, with the
    same bound (1e-14 absolute per bin probability) and exact recomputation.
    Reuses the evidences' forward iteration when the data are unchanged.
    OpenMP throughout.
- `spikecounter.h/.cpp`: the bin evidences log B(s+σ, g+γ) from `lgamma`
  tables indexed by the integer counts, and the counts from prefix sums, so no
  `lgamma` per bin and every row in parallel (0.19 s → 0.014 s at T=2016);
  the (unused) spike-pair co-occurrence counts from the spiking positions,
  O(spikes²) per trial instead of O(T²); `mDataVersion`, so the central
  iteration of the evidences can be reused by the SDF.
- `evidencecomputer.cpp`: `evidenceComputer<evifunc>::computeEvidences`
  (the plain evidences) uses `fb::fastForward` with the plain Beta prior; the
  original table-lookup iteration otherwise.
- `spikedensityfunction.h/.cpp`: `getSDFForwardBackward()`.
- `binsdfc.cpp`: the SDF uses it by default. `--virtual-spike` / `-V` switches
  every new path off, so the output is the original's, bit for bit. Output
  format unchanged.
- `CMakeLists.txt`: the new source file.

With the upper bound on the firing probability (the incomplete-Beta prior) the
original paths are used, as before.

## Build

    g++ -O2 -fopenmp -std=c++14 -o binsdfc binsdfc.cpp evidencecomputer.cpp logadd.cpp \
        simplexminimiser.cpp spikecounter.cpp spikedensityfunction.cpp forwardbackward.cpp

## Agreement

- `-V` reproduces the original output exactly (tests/data/testdata_seed1.txt).
- The evidences print the same digits as the original's.
- The SDF and its standard deviation agree with bayesbin to the 6 printed
  digits, on that dataset and on one that exercises the underflow bounds (400
  trials, rates switching 0.02/0.3: evidences spanning ~10⁴ nats), and with
  the original within 2.5e-5 (the original's table-lookup log-add).
