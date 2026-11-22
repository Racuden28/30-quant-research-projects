#!/usr/bin/env bash
# Reproduce every number and figure in README.md from the committed Ken French files (about 2 minutes).
# To refresh the raw files, see src/fetch_data.js (runs in a browser on the Ken French site).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results figures
cd src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python data.py
python analysis.py > ../results/analysis_log.txt
python robustness.py > ../results/robustness_log.txt
python figures.py
cd .. && python -m pytest -q tests
