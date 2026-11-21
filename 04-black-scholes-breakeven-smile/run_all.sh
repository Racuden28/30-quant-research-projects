#!/usr/bin/env bash
# Reproduce every number and figure in README.md
set -euo pipefail
cd "$(dirname "$0")/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python accuracy.py
python analysis.py
python validation.py
python ablation.py
python figures.py
