#!/usr/bin/env bash
# Reproduce every number and figure in README.md (about 6 minutes on one core)
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests
python src/run_experiments.py
python src/figures.py
