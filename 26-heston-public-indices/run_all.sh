#!/usr/bin/env bash
# Reproduce every number and figure in README.md (about 15 minutes on 4 cores)
set -euo pipefail
cd "$(dirname "$0")/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python calibrate.py     # experiment 1: daily triangular + joint calibration, 2008-2026 (~4 min)
python evaluate.py      # experiments 2-5: feasibility, state dependence, out-of-sample forecasts, P vs Q
python ablations.py     # bounds, VIX9D, fixed kappa, naive VVIX (weekly subsample)
python chain.py         # experiment 6: the 2026-11-23 S&P 500 option chain
python figures.py
cd .. && python -m pytest -q tests
