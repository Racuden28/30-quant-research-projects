#!/usr/bin/env bash
# Reproduce every number and figure in README.md from the committed data.
# About 45 minutes on 4 cores (the two placebo runs take most of it).
set -euo pipefail
cd "$(dirname "$0")/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python synthetic.py              # Part 0: Gaussian trials we control
python run_placebo.py 4          # Part 1: 300 sign-flip worlds (true Sharpe = 0 for every rule)
python run_real.py               # Part 2: the same searches on real data, with out-of-sample results
python run_reality_check.py 4    # Part 2: 200 drift-preserving sign-flip worlds per split (exact max test)
python analyze.py                # size, power, implied N, real-data decisions, stress years
python audit.py                  # Part 3: this portfolio's own claims (needs the sibling project folders)
python figures.py
cd .. && python -m pytest -q tests
