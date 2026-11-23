"""
The three strategies of projects 03, 19 and 12, rewritten as pure *decision* functions.

Each returns target weights decided at the close of day t from data up to that close. None of them
knows how it will be filled; that is the engine's job. The signal definitions follow the original
projects exactly (same parameters, same pre-registered choices), so that the only thing that changes
in project 21 is the fill.

* ma200(p)               project 03: long if close > 200-day SMA (including today), else cash.
* hmm_realtime(r)        project 19, rung R3: long if the real-time predicted probability of the
                         turbulent state of a 2-state Gaussian HMM is below 0.5. Expanding refits every
                         21 days after 5 years. `hmm.py` and `_realtime` are copied from project 19.
* momentum_targets(...)  project 12, textbook 12-1: at each month end rank by the m-11..m-1 compounded
                         excess return, long the top 30%, short the bottom 30%, equal weights, $1/$1.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import hmm

CACHE = Path(__file__).resolve().parents[1] / "results" / "cache"


# ------------------------------------------------------------------------------------- 03: MA200
def ma200(p: pd.Series, n: int = 200) -> pd.Series:
    ma = p.rolling(n, min_periods=n).mean()
    return (p > ma).astype(float).where(ma.notna())


# ------------------------------------------------------------------------------------- 19: HMM
def _realtime_pred(x: np.ndarray, K: int = 2, min_train: int = 1260, refit: int = 21,
                   restart_every: int = 12, seed: int = 0) -> np.ndarray:
    """Copy of project 19's core.realtime, returning only P(turbulent at t+1 | r_1..r_t)."""
    T = len(x)
    rng = np.random.default_rng(seed)
    pred = np.full(T, np.nan)
    prev = None
    for j, tau in enumerate(range(min_train - 1, T, refit)):
        train = x[:tau + 1]
        cands = [hmm.fit(train, K)]
        if prev is not None:
            cands.append(hmm.fit(train, K, init=prev))
        if j % restart_every == 0:
            cands.append(hmm.fit(train, K, n_starts=5, rng=rng))
        best = max(cands, key=lambda c: c.loglik)
        prev = best
        end = min(T, tau + refit)
        f, _ = hmm.filtered(x[:end], best)
        pred[tau:end] = (f[tau:] @ best.A)[:, K - 1]
    return pred


def hmm_realtime(r: pd.Series, key: str | None = None, use_cache: bool = True) -> pd.Series:
    """Long (1) when the real-time turbulent probability for tomorrow is below 0.5, else 0."""
    if key is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        f = CACHE / f"hmm_{key}.pkl"
        if use_cache and f.exists():
            return pickle.loads(f.read_bytes())
    r = r.dropna()
    x = 100 * np.log1p(r.values)
    pr = pd.Series(_realtime_pred(x), index=r.index)
    pos = (pr < 0.5).astype(float).where(pr.notna())
    if key is not None:
        f.write_bytes(pickle.dumps(pos))
    return pos


def hmm_smoothed_bug(r: pd.Series) -> pd.Series:
    """Project 19's rung R1: full-sample smoothed probabilities. Look-ahead by construction; it is
    here only so the causality audit has something it must catch."""
    r = r.dropna()
    x = 100 * np.log1p(r.values)
    p = hmm.fit(x, 2, n_starts=3, rng=np.random.default_rng(1))
    g = hmm.smoothed(x, p) @ p.A
    return pd.Series((g[:, 1] < 0.5).astype(float), index=r.index)


def centred_ma_bug(p: pd.Series, n: int = 200) -> pd.Series:
    """A classic engine bug: a centred moving average (pandas `center=True`)."""
    ma = p.rolling(n, min_periods=n, center=True).mean()
    return (p > ma).astype(float).where(ma.notna())


# ------------------------------------------------------------------------------------- 12: momentum
def month_end_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month in `index`."""
    s = pd.Series(np.arange(len(index)), index=index)
    return index[s.groupby(index.to_period("M")).max().values]


def momentum_targets(ex: pd.DataFrame, lookback: int = 12, skip: int = 1, q: float = 0.3,
                     min_assets: int = 10) -> pd.DataFrame:
    """Daily (T, N) target frame: NaN except on the last trading day of each month.

    ex: daily excess returns (NaN where an asset does not exist). A month counts as present only if
    the asset has a return on (almost) every trading day of it, and an asset is ranked only if all
    `lookback` months are present, as in project 12. Assets not ranked at a rebalance get weight 0."""
    per = ex.index.to_period("M")
    g = (1.0 + ex).groupby(per)
    monthly = g.prod(min_count=1) - 1.0
    counts = ex.notna().groupby(per).sum()
    size = ex.index.to_series().groupby(per).size()
    present = counts.ge(size.values[:, None] * 0.999)
    monthly = monthly.where(present)
    lr = np.log1p(monthly)
    sig = lr.shift(skip).rolling(lookback - skip, min_periods=lookback - skip).sum()
    full = present.astype(float).rolling(lookback, min_periods=lookback).sum().eq(lookback)
    sig = sig.where(full & present)
    days = month_end_days(ex.index)
    tgt = pd.DataFrame(np.nan, index=ex.index, columns=ex.columns)
    started = False
    for m, d in zip(sig.index, days):
        s = sig.loc[m].dropna()
        if len(s) < min_assets:
            if started:
                tgt.loc[d] = 0.0
            continue
        started = True
        pct = s.rank(pct=True)
        w = pd.Series(0.0, index=ex.columns)
        lo, hi = pct.index[pct <= q], pct.index[pct > 1 - q]
        w[hi] = 1.0 / len(hi)
        w[lo] = -1.0 / len(lo)
        tgt.loc[d] = w
    return tgt
