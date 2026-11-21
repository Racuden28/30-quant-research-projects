#!/usr/bin/env bash
# Reproduce every number and figure in README.md (about 25 minutes on 4 cores)
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests
cd src
# python data.py   # rebuilds data/*.csv from data/raw/ (Ken French files) and skfolio (already committed)
python backtest.py
python simulate.py
python sweep.py
python evaluate.py
python figures.py
