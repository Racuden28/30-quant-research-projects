#!/usr/bin/env bash
# Reproduce every number and figure in README.md
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python src/backtest.py
python src/evaluate.py
python src/merton_diagnostic.py
python src/figures.py
