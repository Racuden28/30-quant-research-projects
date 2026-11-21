#!/usr/bin/env bash
# Reproduce every number and figure in README.md from the committed panel (about one minute).
# To start from the raw Ken French files instead, run: python src/fetch_data.py
set -euo pipefail
cd "$(dirname "$0")/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -d ../data/raw ] && python data.py          # market-proxy validation (needs the raw zips)
python backtest.py
python evaluate.py
python horizon_gap.py
python robustness.py
python figures.py
cd .. && python -m pytest -q tests
