"""Part 2 benchmark: an exact max-Sharpe test by drift-preserving sign flips (a 'reality check').

For split Y and world w, each index's (industry's) excess return becomes s_t (x_t - mu) + mu, where
mu is that series' mean over the split's in-sample window and s_t is one common random sign per day
(month). Buy & hold keeps its drift, so a timing rule's exposure to the equity premium is not
mistaken for skill, but no rule can predict the sign of the next return. The p-value of the real
in-sample winner is the share of worlds whose best in-sample Sharpe ratio is at least as high.
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import dsr as D
from data import INDICES, RES
from search import SPLIT_YEARS, World, split_rows
from zoo import zoo_a, zoo_b

N_WORLDS = int(os.environ.get("N_RC", 200))
_W = None


def _init():
    global _W
    _W = World()


def max_sr(R):
    m, sd, _, _ = D.moments(R)
    return float(np.max(np.where(sd > 0, m / np.where(sd > 0, sd, 1), -np.inf)))


def _one(seed):
    W = _W
    rng = np.random.default_rng(10_000 + seed)
    sa = rng.choice([-1.0, 1.0], size=len(W.XA))
    sb = rng.choice([-1.0, 1.0], size=len(W.XB))
    rows = []
    for Y in SPLIT_YEARS:
        ia, _ = split_rows(W.dA, Y)
        ib, _ = split_rows(W.dB, Y)
        muA = W.XA.iloc[ia].mean().fillna(0.0)  # indices with no data yet: unchanged (all NaN)
        XA = (W.XA - muA).mul(sa, axis=0) + muA
        gA, nA, eA, oA = zoo_a(XA)
        okA = eA <= ia[0]
        for basis, R in (("gross", gA), ("net", nA)):
            Ris = R[ia].astype(np.float64)
            rows.append(dict(seed=seed, split=Y, search="A-pooled", basis=basis, max_sr=max_sr(Ris[:, okA])))
            for j, c in enumerate(INDICES):
                ids = np.flatnonzero(okA & (oA == j))
                if ids.size:
                    rows.append(dict(seed=seed, split=Y, search=f"A-{c}", basis=basis, max_sr=max_sr(Ris[:, ids])))
        if ib.size and Y >= 1945:
            muB = W.XB.iloc[ib].mean().fillna(0.0)
            XB = (W.XB - muB).mul(sb, axis=0) + muB
            gB, nB, eB = zoo_b(XB.to_numpy())
            for basis, R in (("gross", gB), ("net", nB)):
                rows.append(dict(seed=seed, split=Y, search="B", basis=basis, max_sr=max_sr(R[ib].astype(np.float64))))
    return rows


if __name__ == "__main__":
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with Pool(procs, initializer=_init) as p:
        rows = [r for rs in p.imap_unordered(_one, range(1, N_WORLDS + 1)) for r in rs]
    pd.DataFrame(rows).sort_values(["seed", "split", "search", "basis"]).to_csv(
        RES / "reality_check_null.csv.gz", index=False, float_format="%.6g")
