# binsdfc 0.1 — the original implementation

Dominik Endres's command-line implementation of the NIPS 2008 paper, C++ with
a Python 2 test-data script. GPL version 2 or later (see the file headers;
the licence text, which the original archive did not include, is in [COPYING](COPYING)).
Kept here unmodified, as the reference `bayesbin` is tested against.

## Provenance

- mloss.org project 67, "Spike train feature extraction by Bayesian binning",
  version 0.1, posted 22 Feb 2008, updated 24 Sep 2008:
  <https://mloss.org/software/view/67/> (unreachable on 2026-09-26),
  static mirror <https://mloss-static.ml.tu-berlin.de/software/view/67/index.html>.
- Its download pointed to `www.compsens.uni-tuebingen.de/pub/pages/personals/3/binsdfc.0.1.tar`,
  which is gone; the archive here is the Internet Archive's copy,
  <http://web.archive.org/web/20150928190519/http://www.compsens.uni-tuebingen.de/pub/pages/personals/3/binsdfc.0.1.tar>
  (614,400 bytes).
- A Matlab implementation is said to exist; not found yet.

## Building

The CMake file asks for `-ffast-math -march=native`; for reference numbers
build without them:

    g++ -O2 -fopenmp -o binsdfc binsdfc.cpp evidencecomputer.cpp logadd.cpp \
        simplexminimiser.cpp spikecounter.cpp spikedensityfunction.cpp

(warnings about `getIncompleteBeta` being "used but never defined" are
harmless for the options used here).

## The reference outputs in tests/data

    ../../tools/make_testdata.py 1 > testdata_seed1.txt
    ./binsdfc -s -100 -e 500 -m 10 -M -n testdata_seed1.txt   > binsdfc_seed1_marginal.txt
    ./binsdfc -s -100 -e 500 -m 10 -v -l 0   testdata_seed1.txt > binsdfc_seed1_sdf_l0.txt
    ./binsdfc -s -100 -e 500 -m 10 -v -l 0.9 testdata_seed1.txt > binsdfc_seed1_sdf_l0.9.txt

Notes on its conventions, found while matching it:

- `-e` is exclusive: `-s -100 -e 500` covers t = −100..499 (600 intervals).
- `-f`/`-g` are the parameters of a standard Beta(σ, γ) prior
  (the paper's eq. 5 writes the exponents as σ, γ; the evidence formula and
  the program use Beta(σ, γ), i.e. exponents σ−1, γ−1). Defaults σ = 1, γ = 32.
- The prior over M is uniform on 0..`-m` (default 10); predictions use only the
  most probable M unless `-l` asks for a credible range.
