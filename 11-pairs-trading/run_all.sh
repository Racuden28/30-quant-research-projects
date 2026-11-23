#!/usr/bin/env bash
# Reproduce every number and figure in README.md from the committed data (about 4 minutes on 4 cores).
# The raw downloads are in data/raw; src/data.py rebuilds the return panels from them.
set -euo pipefail
cd "$(dirname "$0")/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python data.py
python backtest.py countries
python backtest.py --nofilter
python backtest.py industries
python evaluate.py
python diagnostics.py
python figures.py
cd .. && python -m pytest -q tests
