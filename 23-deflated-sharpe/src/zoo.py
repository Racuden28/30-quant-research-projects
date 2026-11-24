"""The strategy zoo: every rule a reasonable researcher might try on these two panels.

Zoo A (time-series timing, 10 indices, daily): 58 signals x {long/flat, long/short} = 116 rules per index.
Zoo B (cross-sectional, 49 industries, monthly): 11 lookbacks x 2 skips x 4 breadths x {momentum, reversal}.
A signal decided at the close of t is traded at t+1; costs are charged on position changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COST_A = 0.0005   # 5 bp per unit change in the index position (as in projects 03 and 19)
COST_B = 0.0010   # 10 bp per unit of one-way turnover in the industry long-short book
WARMUP_A = 600    # trading days of history before a rule of an index is eligible
WARMUP_B = 61     # months


def signal_specs_a():
    s = []
    for p in [5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300]:
        s.append(("SMA", p, 0))
    for f in [5, 10, 20, 50]:
        for sl in [50, 100, 150, 200, 250]:
            if f < sl:
                s.append(("XMA", f, sl))
    for L in [5, 10, 21, 42, 63, 126, 189, 252, 378, 504]:
        s.append(("TSMOM", L, 0))
    for n in [20, 50, 100, 200, 252]:
        s.append(("CHAN", n, 0))
    for L in [1, 2, 3, 5, 10]:
        s.append(("REV", L, 0))
    for w in [10, 21, 63]:
        for lb in [252, 504]:
            s.append(("VOL", w, lb))
    return s


SIGNALS_A = signal_specs_a()
MODES = ["LF", "LS"]


def _sma(P, n):
    c = np.cumsum(np.insert(P, 0, 0.0))
    out = np.full(P.size, np.nan)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def _lag_ratio(P, L):
    out = np.full(P.size, np.nan)
    out[L:] = P[L:] / P[:-L]
    return out


def signals_a(x: np.ndarray):
    """Boolean 'risk-on' signals (n_signals x T) from one index's excess-return series x."""
    P = np.cumprod(1 + x)
    s = pd.Series(P)
    out = np.zeros((len(SIGNALS_A), x.size), dtype=np.int8)
    rv_cache = {}
    for i, (kind, a, b) in enumerate(SIGNALS_A):
        with np.errstate(invalid="ignore"):
            if kind == "SMA":
                sig = P > _sma(P, a)
            elif kind == "XMA":
                sig = _sma(P, a) > _sma(P, b)
            elif kind == "TSMOM":
                sig = _lag_ratio(P, a) > 1
            elif kind == "CHAN":
                hi = s.rolling(a).max().to_numpy()
                lo = s.rolling(a).min().to_numpy()
                sig = (P - lo) > 0.5 * (hi - lo)
            elif kind == "REV":
                sig = _lag_ratio(P, a) < 1
            elif kind == "VOL":
                if a not in rv_cache:
                    rv_cache[a] = pd.Series(x ** 2).rolling(a).mean().to_numpy()
                rv = rv_cache[a]
                sig = rv < pd.Series(rv).rolling(b).mean().to_numpy()
        out[i] = sig
    return out


def trial_names_a(indices):
    return [f"{ix}|{k}{a}{'/' + str(b) if k in ('XMA', 'VOL') else ''}|{m}"
            for ix in indices for (k, a, b) in SIGNALS_A for m in MODES]


def zoo_a(X: pd.DataFrame):
    """Gross and net daily returns of every zoo-A trial on the union calendar.

    Returns gross (T x N float32), net, start index of eligibility per trial, trial index -> index name.
    Days on which an index does not trade carry a zero return for its trials.
    """
    T = len(X)
    ns = len(SIGNALS_A)
    N = X.shape[1] * ns * 2
    gross = np.zeros((T, N), dtype=np.float32)
    net = np.zeros((T, N), dtype=np.float32)
    elig = np.zeros(N, dtype=np.int64)
    owner = np.zeros(N, dtype=np.int64)
    for j, c in enumerate(X.columns):
        col = X[c].to_numpy()
        rows = np.flatnonzero(~np.isnan(col))
        x = col[rows]
        sig = signals_a(x).astype(np.float64)
        for m, mode in enumerate(MODES):
            pos = sig if mode == "LF" else 2 * sig - 1
            held = np.zeros_like(pos)
            held[:, 1:] = pos[:, :-1]          # decided at t-1, earns x_t
            g = held * x
            dpos = np.abs(np.diff(held, axis=1, prepend=0.0))
            dpos[:, 0] = 0.0
            nt = g - COST_A * dpos
            base = j * ns * 2
            idx = base + np.arange(ns) * 2 + m
            gross[np.ix_(rows, idx)] = g.T
            net[np.ix_(rows, idx)] = nt.T
            elig[idx] = rows[min(WARMUP_A, rows.size - 1)]
            owner[idx] = j
    return gross, net, elig, owner


# ---------------------------------------------------------------- zoo B
SPECS_B = [(L, S, q, d) for L in [1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 60] for S in [0, 1]
           for q in [3, 5, 7, 10] for d in [1, -1]]


def trial_names_b():
    return [f"IND|{'MOM' if d == 1 else 'REV'}{L}-{S}|q{q}" for (L, S, q, d) in SPECS_B]


def zoo_b(R: np.ndarray):
    """Monthly long-short industry rules. R: months x industries excess returns (NaN = missing).

    Returns gross, net (M x N), eligibility start month.
    """
    M, K = R.shape
    # formation score = sum of simple monthly returns (robust to the sign-flipped placebo, where a
    # +120% month becomes -120% and a log return is undefined)
    clog = np.vstack([np.zeros(K), np.cumsum(np.nan_to_num(R, nan=0.0), 0)])   # sum of months < t
    avail = ~np.isnan(R)
    cav = np.vstack([np.zeros(K), np.cumsum(avail, 0)])
    N = len(SPECS_B)
    gross = np.zeros((M, N))
    net = np.zeros((M, N))
    cache = {}
    for n, (L, S, q, d) in enumerate(SPECS_B):
        key = (L, S, q)
        if key not in cache:
            W = np.zeros((M, K))
            for t in range(L + S - 1, M - 1):
                # formation months t-S-L+1 .. t-S (inclusive), decided at end of t, held over t+1
                a, b = t - S - L + 1, t - S + 1
                score = clog[b] - clog[a]
                ok = (cav[b] - cav[a] == L) & avail[t] & avail[t + 1]
                ids = np.flatnonzero(ok)
                if ids.size < 2 * q:
                    continue
                o = ids[np.argsort(score[ids], kind="stable")]
                W[t, o[-q:]] = 1.0 / q
                W[t, o[:q]] = -1.0 / q
            g = np.zeros(M)
            g[1:] = np.nansum(W[:-1] * np.nan_to_num(R[1:], nan=0.0), 1)
            turn = np.zeros(M)
            turn[1:] = np.abs(np.diff(W, axis=0)).sum(1)  # traded at end of t, cost booked in t+1
            turn = np.roll(turn, 1)
            turn[0] = 0
            cache[key] = (g, turn)
        g, turn = cache[key]
        gross[:, n] = d * g
        net[:, n] = d * g - COST_B * turn
    return gross, net, np.full(N, WARMUP_B)

