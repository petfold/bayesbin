# binsdfc-fb: binsdfc 0.1 with a forward–backward SDF

A modified copy of Dominik Endres's binsdfc 0.1 (GPL-2.0-or-later, as the
original; unmodified original in `../../reference/binsdfc-0.1/`).

## What changed (2026-09-26)

- `forwardbackward.h/.cpp` (new): the SDF and its second moment at every time
  index in one pass, O(M·T²). The original obtains each time index as a ratio
  of evidences with a virtual spike added there (section 4 of the paper), one
  run of the central iteration per index and moment, O(M·T³). Exact
  max-shifted log-sums, OpenMP over bin start.
- `spikedensityfunction.h/.cpp`: `getSDFForwardBackward()`.
- `binsdfc.cpp`: the SDF uses it by default; `--virtual-spike` / `-V` computes
  the SDF the original way, for comparison. Output format unchanged.
- `CMakeLists.txt`: the new source file.

With the upper bound on the firing probability (the incomplete-Beta prior) the
original path is used, as before.

## Build

    g++ -O2 -fopenmp -std=c++11 -o binsdfc binsdfc.cpp evidencecomputer.cpp logadd.cpp \
        simplexminimiser.cpp spikecounter.cpp spikedensityfunction.cpp forwardbackward.cpp

## Agreement

On `tests/data/testdata_seed1.txt`, `-V` reproduces the original output exactly;
the forward–backward SDF agrees with bayesbin to the 6 printed digits and with
the original within 2.5e-5 (the original's table-lookup log-add).
