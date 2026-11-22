#!/usr/bin/env bash
# Reproduce every number and figure in README.md (about 8 minutes on 4 cores)
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
rm -rf results/cache
python -m pytest -q tests
cd src
python run_experiments.py
python figures.py
