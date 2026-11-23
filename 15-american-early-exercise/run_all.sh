#!/usr/bin/env bash
# Reproduce every number and figure in README.md (about 20 minutes on 4 cores, 18 min measured on a
# clean copy; accuracy.py, with its 1,300 high-precision reference prices, takes most of it).
set -euo pipefail
cd "$(dirname "$0")/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python build_data.py     # data/sp500_option_inputs.csv from data/raw
python accuracy.py       # Part I: 13 lattice methods vs high-precision references
python history.py        # Part II: 1928-2026 early-exercise premia
python exercise.py       # Part III: exercise-rule misspecification, GARCH world, LSM
python figures.py
