#!/usr/bin/env bash
# Reproduce every number and figure in README.md (about 2 minutes on 4 cores)
set -euo pipefail
cd "$(dirname "$0")/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python frontier.py      # experiment 1: frontier, floor, stress windows, robustness
python simulation.py    # experiment 2: simulated worlds + validation of the textbook law
python which_delta.py   # experiment 3: which volatility goes inside the delta
python costs.py         # experiment 4: transaction costs, calendar vs band rules
python figures.py
cd .. && python -m pytest -q tests
