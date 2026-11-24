#!/usr/bin/env bash
# Reproduce every number and figure in README.md from the committed data (about 6 minutes on 4 cores).
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# numba's on-disk cache breaks on network drives when several processes share it: keep it local
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/numba_cache_p25}"
cd src
python hedging.py
python pairs.py
python mechanism.py
python evaluate.py
python figures.py
cd .. && python -m pytest -q tests -p no:cacheprovider
