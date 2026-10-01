# The 2-D draft: `make` builds main.pdf from the results in data/ and the figures in figures/;
# `make data` reruns the whole analysis on the current data first (see README.md).
#
#   make venv          the pipeline's Python environment (.venv), once
#   make data          globe, held-out prediction, 1-D comparison, rectangles, maps, then numbers
#   make numbers       only the LaTeX macros and generated tables, from the results already in data/
#   make quick         a fast check of the pipeline (smaller prototype sizes)
#   make               the PDF

PY ?= .venv/bin/python
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
export WW_ARCHIVE WW_USGS TWOD_RES TWOD_QUICK

FIGS = figures/globe_bins.pdf figures/globe_zoom.pdf figures/globe_rates.pdf

all: main.pdf

venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e ../..

data: globe heldout tree1d rectangles maps numbers

globe:
	$(PY) scripts/globe.py
heldout:
	$(PY) scripts/predict.py
tree1d:
	$(PY) scripts/tree1d.py
rectangles:
	$(PY) scripts/rectangles.py
maps:
	$(PY) scripts/maps.py
numbers:
	$(PY) scripts/macros.py

quick:
	$(MAKE) data TWOD_QUICK=1

main.pdf: main.tex refs.bib data/numbers.tex $(FIGS) $(wildcard data/*_table.tex data/plot_*.csv)
	$(LATEX) main
	bibtex main
	$(LATEX) main
	$(LATEX) main
	for i in 1 2 3; do grep -q "Rerun to get" main.log && $(LATEX) main; done; true

clean:
	rm -f main.aux main.bbl main.blg main.log main.out

.PHONY: all venv data globe heldout tree1d rectangles maps numbers quick clean
