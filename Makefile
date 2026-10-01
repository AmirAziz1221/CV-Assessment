# Targets are added phase by phase. Run `make help`.
PY ?= python

help:           ## list targets
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/ -/'

setup:          ## install dependencies
	$(PY) -m pip install -r requirements.txt

data:           ## download + arrange the public dataset (skip if you have your own data/raw)
	$(PY) scripts/prepare_data.py

analyze:        ## Phase 1: dataset analysis -> docs/figures + stats.json
	$(PY) scripts/analyze_dataset.py

splits:         ## Phase 2: group-aware split manifest + leakage report
	$(PY) scripts/make_splits.py

resolution:     ## Phase 2: resolution / resize-mode study
	$(PY) scripts/resolution_study.py

augment-check:  ## Phase 2: augmentation samples + defect-retention safety test
	$(PY) scripts/augmentation_check.py

prep: data analyze splits resolution augment-check   ## everything up to the end of Phase 2

test:           ## run the unit tests
	$(PY) -m pytest -q

lint:           ## style check
	ruff check src scripts tests

.PHONY: help setup data analyze splits resolution augment-check prep test lint
