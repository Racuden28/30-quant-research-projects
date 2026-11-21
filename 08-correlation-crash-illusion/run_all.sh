#!/usr/bin/env bash
# Reproduce every number and figure in README.md (about 20 minutes on 4 cores)
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python src/data.py
python -m pytest -q tests
python src/conditioning.py
python src/forecast.py
python src/stress.py
python src/figures.py
