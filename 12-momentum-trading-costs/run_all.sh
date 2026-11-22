#!/usr/bin/env bash
# Reproduce every number and figure in README.md from the committed data (under 2 minutes on 4 cores).
# To re-download the ETF data first: python src/fetch_data.py
set -euo pipefail
cd "$(dirname "$0")/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python data.py            # data checks (stale-quote shares, extreme returns)
python backtest.py 4      # 126 backtests: 2 universes x 3 signals x 21 designs
python evaluate.py        # summary, marginal break-evens, hypotheses (Holm), stress windows
python decay.py           # event-time decay and the frontier it predicts
python select_speed.py    # choosing the trading speed in real time
python costs_check.py     # spread estimators vs the tick floor
python figures.py
cd .. && python -m pytest -q tests
