"""Builds every daily series used in project 19 for one index (cached in results/cache)."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import hmm
from core import (COST_BP, RES, ewma_vol, load_closes, load_index, ma_signal, matched_vol_placebo,
                  realtime, strategy_returns)

CACHE = RES / "cache"
LADDER = ["R0 Hindsight", "R1 Smoothed", "R2 Filtered", "R3 Real-time"]


def build(name: str, K: int = 2, refit: int = 21, window: int | None = None, total_return: bool = False,
          closes: pd.DataFrame | None = None, use_cache: bool = True, tag: str = "") -> dict:
    key = f"{name}_K{K}_refit{refit}_win{window}_tr{int(total_return)}{tag}.pkl"
    CACHE.mkdir(parents=True, exist_ok=True)
    if use_cache and (CACHE / key).exists():
        return pickle.loads((CACHE / key).read_bytes())
    closes = load_closes() if closes is None else closes
    df = load_index(name, closes, total_return)
    x = df.x.values
    full = hmm.fit(x, K, n_starts=20, rng=np.random.default_rng(1))
    turb = K - 1
    gam = hmm.smoothed(x, full)
    ffull, _ = hmm.filtered(x, full)
    rt = realtime(x, K, refit=refit, window=window)
    ix = df.index
    P = pd.DataFrame({
        "p_smooth": gam[:, turb],                                   # hindsight: whole sample
        "p_smooth_pred": (gam @ full.A)[:, turb],
        "p_full_pred": (ffull @ full.A)[:, turb],                   # filtered, but full-sample parameters
        "p_full_filt": ffull[:, turb],
        "p_rt_pred": rt.pred[:, turb],                              # real time
        "p_rt_filt": rt.filt[:, turb],
        "er_rt": rt.exp_ret,
    }, index=ix)
    pos = pd.DataFrame(index=ix)
    # R0: what many tutorials backtest: today's smoothed state applied to today's return
    pos["R0 Hindsight"] = (P.p_smooth.shift(-1) < 0.5).astype(float).where(P.p_smooth.shift(-1).notna())
    pos["R1 Smoothed"] = (P.p_smooth_pred < 0.5).astype(float)
    pos["R2 Filtered"] = (P.p_full_pred < 0.5).astype(float)
    pos["R3 Real-time"] = (P.p_rt_pred < 0.5).astype(float).where(P.p_rt_pred.notna())
    pos["Buy & hold"] = 1.0
    vol = ewma_vol(df.r)
    pos["Vol placebo"] = matched_vol_placebo(pos["R3 Real-time"], vol)
    pos["MA200"] = ma_signal(df.p)
    pos["HMM weight"] = (1 - P.p_rt_pred).where(P.p_rt_pred.notna())
    rf_pct = 100 * np.log1p(df.rf)
    pos["HMM mean rule"] = (P.er_rt > rf_pct).astype(float).where(P.er_rt.notna())
    # evaluation sample: every strategy defined (held positions start one day later)
    ok = pos.notna().all(axis=1)
    first = ok.idxmax()
    pos = pos.loc[first:].iloc[:-1]           # last day: R0 needs tomorrow
    ret = {}
    for c in pos.columns:
        st = strategy_returns(pos[c], df.r.loc[pos.index], df.rf.loc[pos.index], COST_BP)
        ret[c] = st
    out = dict(name=name, K=K, df=df.loc[pos.index[0]:], P=P, pos=pos, ret=ret, full=full, fits=rt.fits,
               vol=vol, start=pos.index[1], end=pos.index[-1])
    (CACHE / key).write_bytes(pickle.dumps(out))
    return out
