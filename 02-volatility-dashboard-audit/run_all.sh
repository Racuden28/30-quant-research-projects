#!/usr/bin/env bash
# Reproduce every number, figure and the dashboard in README.md (10–25 minutes on 4 cores)
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests
python src/forecast.py
python src/evaluate.py
python src/figures.py
python src/dashboard.py
