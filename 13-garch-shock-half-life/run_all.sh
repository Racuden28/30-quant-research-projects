#!/usr/bin/env bash
# Reproduce every number and figure in README.md (about 15 minutes on 4 cores)
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests
python src/backtest.py      # walk-forward forecasts, 6 models x 5 horizons
python src/irf.py           # out-of-sample local projections (model's own forecast as the yardstick)
python src/sim_check.py     # placebo: does the measurement recover a known half-life?
python src/null_worlds.py   # main test: data vs the same measurement in simulated GARCH / CGARCH worlds
python src/stability.py     # decade-by-decade parameter stability, parametric bootstrap LR
python src/episodes.py      # event study of 130 detected volatility shocks
python src/evaluate.py      # QLIKE, Diebold-Mariano + Holm, stress windows, calibration
python src/figures.py
