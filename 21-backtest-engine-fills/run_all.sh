#!/usr/bin/env bash
# Reproduce every number and figure in README.md (about 10 minutes on 4 cores; the HMM refits are most of it)
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
rm -rf results/cache
python -m pytest -q tests
cd src
python build_hmm.py
python run_experiments.py
python figures.py
