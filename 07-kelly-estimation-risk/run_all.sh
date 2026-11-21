#!/usr/bin/env bash
# Reproduce every number and figure in README.md (about 2 minutes on a laptop)
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests -q
python src/backtest.py
python src/simulate.py
python src/evaluate.py
python src/figures.py
