"""
Monthly cross-sectional momentum engine with exact turnover accounting.

Timing convention (no look-ahead):
  * At the close of month m we know every daily return up to and including the last trading day of m.
  * The signal, the eligibility screen, the cost state and the target weights are computed from that
    information only.
  * The portfolio held during month m+1 is bought at the close of m. Turnover at m is measured
    against the *drifted* weights of the portfolio that was held during m, so price moves alone never
    count as trading.
  * Cost of the trade at m is charged to the return of month m+1.

Portfolios are self-financing in excess-return space: returns are excess of the T-bill, a long-short
book is a $1 long / $1 short position per $1 of capital, and a long-only book is $1 long per $1 of
capital funded at the bill rate. Within a month the book is buy-and-hold (weights drift); daily book
returns are computed from the drifted positions, so the volatility-scaling rule sees the same daily
P&L an investor would.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------- inputs
@dataclass
class Universe:
    name: str
    daily: pd.DataFrame                 # daily excess returns, NaN where the asset does not exist
    monthly: pd.DataFrame = field(init=False)      # compounded monthly excess return, row = month m
    month_end: pd.DatetimeIndex = field(init=False)
    vol21: pd.DataFrame = field(init=False)        # annualised realised vol over the 21 days to m
    present: pd.DataFrame = field(init=False)      # asset has a full month of returns in month m
    extra_eligible: pd.DataFrame | None = None     # optional extra screen known at m (e.g. liquidity)
    halftick: pd.DataFrame | None = None           # optional half-tick / price floor at m (decimal)

    def __post_init__(self):
        d = self.daily
        per = d.index.to_period("M")
        g = (1.0 + d).groupby(per)
        self.monthly = g.prod(min_count=1) - 1.0
        counts = d.notna().groupby(per).sum()
        size = d.index.to_series().groupby(per).size()
        self.present = counts.ge(size.values[:, None] * 0.999)
        self.monthly = self.monthly.where(self.present)
        self.monthly.index = self.monthly.index.to_timestamp("M")
        self.present.index = self.monthly.index
        ends = d.index.to_series().groupby(per).max()
        self.month_end = pd.DatetimeIndex(ends.values)
        roll = d.rolling(21, min_periods=15).std() * np.sqrt(252)
        self.vol21 = roll.loc[self.month_end].set_axis(self.monthly.index)


# --------------------------------------------------------------------------------------- signals
def momentum_signal(U: Universe, lookback: int, skip: int) -> pd.DataFrame:
    """Cumulative log excess return from the end of m-lookback to the end of m-skip.

    Requires every month in the window to be present (a fund listed mid-window is not ranked)."""
    lr = np.log1p(U.monthly)
    s = lr.shift(skip).rolling(lookback - skip, min_periods=lookback - skip).sum()
    full = U.present.astype(float).rolling(lookback, min_periods=lookback).sum().eq(lookback)
    return s.where(full)


# --------------------------------------------------------------------------------------- designs
@dataclass(frozen=True)
class Design:
    key: str
    label: str
    q: float = 0.3                 # fraction of the eligible universe in each leg
    hold: int = 1                  # Jegadeesh-Titman overlapping holding period (months)
    q_exit: float | None = None    # banding: keep a name until it leaves the top/bottom q_exit
    smooth: int = 1                # average the percentile rank over the last `smooth` months
    theta: float = 1.0             # partial rebalancing: trade this fraction of the gap to target
    vol_target: float | None = None   # annualised target vol (Barroso & Santa-Clara scaling)
    vol_window: int = 126
    max_lev: float = 2.0
    long_only: bool = False


def _legs(pct: pd.Series, q: float, prev_long: set, prev_short: set, q_exit: float | None,
          long_only: bool):
    long = set(pct.index[pct > 1 - q])
    short = set() if long_only else set(pct.index[pct <= q])
    if q_exit is not None:
        long |= {a for a in prev_long if a in pct.index and pct[a] > 1 - q_exit}
        if not long_only:
            short |= {a for a in prev_short if a in pct.index and pct[a] <= q_exit}
    return long, short


def _weights(long: set, short: set, cols: pd.Index) -> pd.Series:
    w = pd.Series(0.0, index=cols)
    if long:
        w[list(long)] = 1.0 / len(long)
    if short:
        w[list(short)] = -1.0 / len(short)
    return w


def _month_daily_path(w: np.ndarray, rd: np.ndarray):
    """Buy-and-hold within a month. rd: (days x N) excess returns (NaN -> 0 for held assets).

    Returns the daily book returns and the end-of-month drifted weights."""
    rd = np.nan_to_num(rd, nan=0.0)
    growth = np.cumprod(1.0 + rd, axis=0)                     # value of $1 in each asset
    x = w[None, :] * growth                                    # positions after each day
    x_prev = np.vstack([w[None, :], x[:-1]])
    pnl = (x_prev * rd).sum(axis=1)
    v = 1.0 + np.concatenate([[0.0], np.cumsum(pnl)])          # book value (excess frame)
    r_book = pnl / v[:-1]
    drift = x[-1] / v[-1]
    return r_book, drift, v[-1] - 1.0


def run(U: Universe, signal: pd.DataFrame, D: Design, start: str | None = None,
        end: str | None = None, min_assets: int = 10) -> dict:
    """Run one design. Returns dict with monthly frame and daily book returns."""
    months = U.monthly.index
    cols = U.monthly.columns
    elig_all = signal.notna() & U.present
    if U.extra_eligible is not None:
        elig_all &= U.extra_eligible.reindex_like(elig_all).fillna(False)
    sig_rank = signal.where(elig_all).rank(axis=1, pct=True)
    daily = U.daily
    per = daily.index.to_period("M")
    day_groups = {p: np.where(per == p)[0] for p in per.unique()}
    vol = U.vol21
    ht = U.halftick

    w_drift = pd.Series(0.0, index=cols)
    prev_long: set = set()
    prev_short: set = set()
    past_targets: list[pd.Series] = []
    rank_hist: list[pd.Series] = []
    book_daily: list[pd.Series] = []
    rows = []
    started = False
    for i in range(len(months) - 1):
        m, m1 = months[i], months[i + 1]
        if start is not None and m < pd.Timestamp(start) - pd.offsets.MonthEnd(1):
            continue
        if end is not None and m1 > pd.Timestamp(end) + pd.offsets.MonthEnd(0):
            break
        pct = sig_rank.loc[m].dropna()
        if D.smooth > 1:
            rank_hist.append(pct)
            rank_hist = rank_hist[-D.smooth:]
            pct = pd.concat(rank_hist, axis=1).mean(axis=1, skipna=True).reindex(pct.index)
            pct = pct.rank(pct=True)
        if len(pct) < min_assets and not started:
            continue
        started = True
        if len(pct) >= min_assets:
            long, short = _legs(pct, D.q, prev_long, prev_short, D.q_exit, D.long_only)
            tgt = _weights(long, short, cols)
        else:                                   # universe temporarily too thin: hold cash
            long, short, tgt = set(), set(), pd.Series(0.0, index=cols)
        prev_long, prev_short = long, short
        if D.hold > 1:
            past_targets.append(tgt)
            past_targets = past_targets[-D.hold:]
            tgt = sum(past_targets) / len(past_targets)
        # volatility scaling uses the realised daily returns of the *unscaled* book so far
        scale = 1.0
        if D.vol_target is not None:
            hist = pd.concat(book_daily[-8:]) if book_daily else pd.Series(dtype=float)
            hist = hist.iloc[-D.vol_window:]
            if len(hist) >= 60 and hist.std() > 0:
                scale = min(D.vol_target / (hist.std() * np.sqrt(252)), D.max_lev)
            else:
                scale = 1.0
        target = tgt * scale
        new_w = w_drift + D.theta * (target - w_drift) if D.theta < 1.0 else target
        # an asset that cannot be traded at m (not present) keeps its drifted weight
        frozen = ~U.present.loc[m].reindex(cols).fillna(False).values
        new_w = new_w.where(~frozen, w_drift)
        dw = (new_w - w_drift).abs()
        tau = float(dw.sum())
        sig_m = vol.loc[m].reindex(cols)
        vtau_raw = float((dw * sig_m.fillna(sig_m.mean())).sum())
        tick_cost = float((dw * ht.loc[m].reindex(cols).fillna(0.0)).sum()) if ht is not None else 0.0
        # next month: buy and hold
        idx = day_groups[m1.to_period("M")]
        rd = daily.iloc[idx].reindex(columns=cols).values
        r_book, drift, r_m = _month_daily_path(new_w.values, rd)
        # unscaled book daily returns (for the vol-scaling estimate): same path at unit scale
        if D.vol_target is not None and scale > 0:
            book_daily.append(pd.Series(r_book / scale, index=daily.index[idx]))
        else:
            book_daily.append(pd.Series(r_book, index=daily.index[idx]))
        book_daily = book_daily[-8:]
        w_drift = pd.Series(drift, index=cols)
        rows.append({"month": m1, "gross": r_m, "turnover": tau, "vturn_raw": vtau_raw,
                     "sigma_bar": float(vol.loc[m][elig_all.loc[m]].mean()),
                     "tick_cost": tick_cost, "n_long": len(long), "n_short": len(short),
                     "n_elig": int(elig_all.loc[m].sum()), "scale": scale,
                     "gross_exposure": float(new_w.abs().sum())})
    out = pd.DataFrame(rows).set_index("month")
    return {"monthly": out}


# --------------------------------------------------------------------------------------- costs
def add_costs(df: pd.DataFrame, sigma_ref: float, c_bps: float, model: str = "vol") -> pd.Series:
    """Net monthly return for a one-way cost of c_bps per $ traded.

    model 'const': cost = c x turnover.  model 'vol': cost = c x sum |dw_i| sigma_i / sigma_ref, so an
    asset at the reference volatility costs exactly c. Tick floors (ETFs) are added on top."""
    c = c_bps / 1e4
    if model == "const":
        cost = c * df["turnover"]
    else:
        cost = c * df["vturn_raw"] / sigma_ref
    return df["gross"] - cost - df["tick_cost"]


def breakeven(df: pd.DataFrame, sigma_ref: float, model: str = "vol") -> float:
    """One-way cost (bps) at which the mean net return is zero (tick floors included)."""
    g = df["gross"].mean() - df["tick_cost"].mean()
    t = df["turnover"].mean() if model == "const" else df["vturn_raw"].mean() / sigma_ref
    return 1e4 * g / t if t > 0 else np.nan


def kappa(df: pd.DataFrame) -> dict:
    """Cost-timing multiplier relative to a trader who trades the average asset in the average month.

    kappa = sum_m sum_i |dw| sigma_i / (sum_m tau_m * mean_m sigma_bar_m)  (unit-free; 1 = no timing)
    kappa_time  = sum_m tau_m sigma_bar_m / (sum_m tau_m * mean_m sigma_bar_m)   (when it trades)
    kappa_cross = kappa / kappa_time                                             (what it trades)"""
    ref = df["sigma_bar"].mean()
    k = df["vturn_raw"].sum() / (df["turnover"].sum() * ref)
    kt = (df["turnover"] * df["sigma_bar"]).sum() / (df["turnover"].sum() * ref)
    return {"kappa": k, "kappa_time": kt, "kappa_cross": k / kt}
