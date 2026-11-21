#!/usr/bin/env bash
# Reproduce every number and figure in README.md
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${PYTHON:-python3}" src/backtest.py
"${PYTHON:-python3}" src/evaluate.py
"${PYTHON:-python3}" src/figures.py
