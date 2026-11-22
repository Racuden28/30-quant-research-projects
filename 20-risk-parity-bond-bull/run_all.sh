#!/usr/bin/env bash
# Reproduce every number and figure in README.md (about 1 minute on one core)
set -euo pipefail
cd "$(dirname "$0")"
(cd src && python data.py > /dev/null)   # data/monthly_panel.csv + data checks, built from data/raw/
python -m pytest -q tests
cd src
python backtest.py             # all strategies and variants -> results/bt_*.csv.gz
python evaluate.py > /dev/null # every table -> results/*.csv (results/summary.csv has the headlines)
python figures.py              # figures/*.png
