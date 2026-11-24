#!/usr/bin/env bash
# Reproduce every number and figure in README.md from the committed data in data/.
# About 25 minutes on 4 cores: 95 walk-forward runs, 80 of them placebo markets
# (40 sign-flip placebos used in the paper + 2 x 40 rejected placebo designs, section 7).
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python src/theory.py
python src/experiments.py                       # results/runs/*.pkl (skips runs that exist)
python src/experiments.py --placebo permute     # rejected placebo v1 -> results/runs_permute/
python src/experiments.py --placebo bootstrap   # rejected placebo v2 -> results/runs_bootstrap/
python src/evaluate.py
python src/diagnostics.py
python src/figures.py
python -m pytest -q tests
