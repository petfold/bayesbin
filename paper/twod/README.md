# Exact Bayesian binning beyond one dimension (draft)

A LaTeX draft (`main.tex`, TikZ and pgfplots, BibTeX in `refs.bib`) on which families of 2-D
partitions keep Bayesian binning's exact sum, with experiments on Worldwatch's news-event and
earthquake counts over the H3 grid of the globe. The analysis behind it is in
[docs/NOTES.md](../../docs/NOTES.md#where-the-method-stops-being-1-d).

Every number in the text is a macro, every table body and plotted series a generated file, and
every map a generated figure, so the paper follows the data:

```sh
make venv        # once: the pipeline's environment in .venv (numpy, scipy, numba, h3, duckdb, matplotlib, bayesbin)
make data        # rerun the analysis on the current data (about 7 minutes on 4 cores; most of it timings)
make             # rebuild main.pdf (pdflatex + bibtex)
```

## Updating with more data

The pipeline reads the events where Worldwatch keeps them; set these to point elsewhere:

| variable | default | what |
|---|---|---|
| `WW_ARCHIVE` | `~/worldwatch-archive` | Worldwatch's Parquet archive (the daily backup, `worldwatch-pull`); news events are the `gdelt_events` stream of its `bins` table |
| `WW_USGS` | `~/.cache/worldwatch-research/usgs` | the USGS catalogue as CSV (Worldwatch's `research/replay_changepoint/fetch_usgs.py START END`) |
| `WW_RESEARCH` | `~/.cache/worldwatch-research` | Worldwatch's research results: the streaming replay's `tree_layer0.npz` (its `research/replay_changepoint/tree_layer0.py`, ~20 min on 4 cores) and, if there, `tree_layer0_emsc.npz` (the same on the EMSC catalogue: `TREE_CATALOGUE=emsc`) |
| `TWOD_RES` | `3` | the finest H3 resolution (news cells cannot be finer than the archive's, resolution 3) |
| `TWOD_QUICK` | unset | `1`: smaller prototype sizes, for a fast check |

More data means: refresh the archive (or fetch a longer USGS period, and rerun Worldwatch's
`prepare.py` and `tree_layer0.py` for the streaming section), then `make data && make`.
The held-out test always splits each record at the middle of its time span.

The text's qualitative claims (the tree predicts best, merging predicts worse than the finest grid,
the 1-D tree ties, the North Pacific is one resolution-0 bin, the checks pass, the chosen prior
lies inside its grid, ...) are checked by `scripts/macros.py` against the new results. Any that
no longer hold are printed and listed in a box under the abstract, so the sentences to rewrite
are visible at once. The numbers themselves update by themselves.

## Layout

| path | what |
|---|---|
| `main.tex`, `refs.bib` | the draft |
| `scripts/common.py` | paths, data loading, the places looked up |
| `scripts/h3tree.py` | the recursion over the H3 tree, its posterior, the prior search, greedy merging |
| `scripts/globe.py` | the binning of each dataset on all its data → `data/globe_*.json`, `build/globe_*.npz` |
| `scripts/predict.py` | held-out prediction → `data/heldout.csv`, `data/heldout_*.json` |
| `scripts/tree1d.py` | 1-D: bayesbin's prior against binary-cut and dyadic trees → `data/tree1d.json` |
| `scripts/rectangles.py` | rows then columns, recursive cuts: checks, counts, timings → `data/rectangles.json`, `data/timings.csv` |
| `scripts/stream.py` | the tree streamed over the earthquake counts (Worldwatch's replay), summarised → `data/stream.json` |
| `scripts/maps.py` | the world maps → `figures/*.pdf` (and `build/*.png` for the web) |
| `scripts/macros.py` | `data/numbers.tex`, `data/*_table.tex`, `data/plot_*.csv`, the claim checks |
| `data/` | results the paper reads (committed, so the PDF builds without the event data) |
| `figures/` | generated maps (committed) |
| `build/` | intermediate arrays and web images (not committed) |

Coastlines: Natural Earth 1:110m (`data/ne_110m_coastline.geojson`), public domain.
Timings are machine-dependent; the draft's are from a 4-core Intel i7-3612QM laptop.
