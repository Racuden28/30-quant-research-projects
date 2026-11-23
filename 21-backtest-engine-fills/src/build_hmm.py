"""Compute (and cache) the real-time HMM positions for the ten indices and their ETFs (about 5-10 min)."""
from __future__ import annotations

import os
from multiprocessing import Pool

import data
from strategies import hmm_realtime


def one(sym: str) -> str:
    g = data.ohlc(sym).dropna(subset=["close"])
    px = g.adjclose if sym in data.ETF_OF.values() else g.close
    hmm_realtime(px.pct_change(), key=sym)
    return sym


if __name__ == "__main__":
    syms = data.INDICES + list(data.ETF_OF.values())
    with Pool(int(os.environ.get("NPROC", "4"))) as p:
        for s in p.imap_unordered(one, syms):
            print("hmm", s, flush=True)
