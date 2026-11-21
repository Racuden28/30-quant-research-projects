#!/usr/bin/env bash
# Reproduce every number and figure in README.md (about 5 minutes on 4 cores)
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests -q
cd src
python backtest.py      # 6 models x 8 books, rolling out-of-sample VaR/ES forecasts
python theory.py        # closed form + GARCH Monte Carlo
python evaluate.py      # backtests, scores, capital, shading, model choice, stress
python figures.py
