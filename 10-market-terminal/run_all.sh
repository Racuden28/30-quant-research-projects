#!/usr/bin/env bash
# Reproduce every number, figure and the terminal page in README.md
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python src/audit.py            # results/*.csv (~2 min)
python src/figures.py          # figures/*.png
python src/terminal_data.py    # terminal/index.html + terminal_data.json
python -m pytest tests -q
