"""Part 1: the sign-flip placebo. Every rule's true Sharpe ratio is exactly zero in these worlds.

Each world multiplies every day's (month's) excess returns by one common random sign, which keeps the
volatility path, fat tails and cross-asset correlation of the real data but removes any predictability.
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import pandas as pd

from data import RES
from search import World, run_world

N_WORLDS = int(os.environ.get("N_WORLDS", 300))
_W = None


def _init():
    global _W
    _W = World()


def _one(seed):
    return run_world(_W, seed, do_cluster=True)


if __name__ == "__main__":
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with Pool(procs, initializer=_init) as p:
        rows = [r for rs in p.imap_unordered(_one, range(1, N_WORLDS + 1), chunksize=2) for r in rs]
    df = pd.DataFrame(rows).sort_values(["seed", "split", "search"])
    df.to_csv(RES / "placebo_searches.csv.gz", index=False, float_format="%.6g")
    print(df.shape)
