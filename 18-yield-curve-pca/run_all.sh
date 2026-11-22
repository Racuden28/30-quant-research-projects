#!/usr/bin/env bash
# Reproduce every number and figure in README.md (about 2 minutes on 4 cores)
set -euo pipefail
cd "$(dirname "$0")"
python src/data.py
python -m pytest -q tests
python src/backtest.py --robust
python src/analysis.py
python src/figures.py
