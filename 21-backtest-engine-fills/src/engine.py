"""
One backtest engine for every strategy in project 21.

The engine separates three things that bespoke backtests usually fuse into one line of code
(`pos.shift(1) * r`):

1. **Decision.** A target weight vector is decided at the close of day t, from data up to and
   including that close. That part belongs to the strategy and the engine never sees how it was made.
2. **Fill.** When, and at which price, the trade towards that target actually happens. This is the
   engine's job and it is an explicit, named argument:

   * ``"close"``    trade at the close of day t, the same close the signal was computed from
                    (what projects 03, 12 and 19 assume; a market-on-close order placed on a
                    signal computed from a price a few minutes earlier);
   * ``"open"``     trade at the open of day t+1 (an end-of-day system that sends orders overnight);
   * ``"close+k"``  trade at the close of day t+k, k >= 1 (``"close+1"`` = next close).

3. **Accounting.** Every day is split into an overnight piece (close t-1 -> open t, dividends go
   ex here) and an intraday piece (open t -> close t). Positions held before an open fill earn the
   overnight piece, positions held after it earn the intraday piece. Weights drift between trades;
   turnover is measured against the drifted weights, so price moves alone never count as trading.
   Costs are charged at the fill, per unit of one-way turnover.

Returns are reported in an excess-of-cash frame: the book return on day t is
    rf_t + sum_i w_i (R_i,t - rf_t),
so a long/flat timing rule holds cash when out, and a $1 long / $1 short book earns the cash rate on its
capital plus the long-short spread. The cash rate accrues in the intraday piece.

Delay accounting. For a single asset held at weight 0 or 1, the identity
    sum_t gross_close,t - sum_t gross_close+1,t = sum_t (pos_t - pos_{t-1}) (R_{t+1} - rf_{t+1})
holds exactly (tests/test_engine.py): moving the fill one close later costs precisely the excess
return of the first day after each trade. ``first_day_pnl`` returns that series for any book.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TDAYS = 252


@dataclass
class Market:
    """Aligned daily panel. All arrays are (T, N) with a shared DatetimeIndex.

    close_ret:  total close-to-close return (dividends included where the data has them)
    on_ret:     overnight return close t-1 -> open t (dividend included); NaN where no real open
    rf:         (T,) daily cash return
    tradeable:  (T, N) bool, asset has a price that day
    """
    index: pd.DatetimeIndex
    assets: list
    close_ret: np.ndarray
    on_ret: np.ndarray
    rf: np.ndarray
    tradeable: np.ndarray

    @property
    def in_ret(self) -> np.ndarray:
        """Intraday return open t -> close t implied by the two pieces."""
        return (1.0 + self.close_ret) / (1.0 + self.on_ret) - 1.0

    def slice(self, start=None, end=None) -> "Market":
        m = np.ones(len(self.index), bool)
        if start is not None:
            m &= self.index >= pd.Timestamp(start)
        if end is not None:
            m &= self.index <= pd.Timestamp(end)
        return Market(self.index[m], self.assets, self.close_ret[m], self.on_ret[m], self.rf[m],
                      self.tradeable[m])


def market_from_ohlc(panels: dict[str, pd.DataFrame], rf: pd.Series, index: pd.DatetimeIndex | None = None) -> Market:
    """Build a Market from per-asset frames with columns open, close and optionally adjclose.

    Total return uses adjclose when present (Yahoo's dividend-adjusted close), else close.
    The dividend factor of day t is (adj_t/adj_{t-1}) / (close_t/close_{t-1}); it is applied to the
    overnight piece, because a stock goes ex-dividend at the open."""
    assets = list(panels)
    if index is None:
        index = pd.DatetimeIndex(sorted(set().union(*[p.index for p in panels.values()])))
    T, N = len(index), len(assets)
    cr = np.full((T, N), np.nan)
    on = np.full((T, N), np.nan)
    tr = np.zeros((T, N), bool)
    for j, a in enumerate(assets):
        p = panels[a].reindex(index)
        c = p["close"]
        adj = p["adjclose"] if "adjclose" in p and p["adjclose"].notna().any() else c
        c_prev = c.ffill().shift(1)
        adj_prev = adj.ffill().shift(1)
        r_tot = adj / adj_prev - 1.0
        div = (adj / adj_prev) / (c / c_prev)
        o = p["open"] if "open" in p else pd.Series(np.nan, index=index)
        r_on = (o / c_prev) * div - 1.0
        cr[:, j] = r_tot.values
        on[:, j] = r_on.values
        tr[:, j] = c.notna().values
    rfv = rf.reindex(index).ffill().fillna(0.0).values
    return Market(index, assets, cr, on, rfv, tr)


@dataclass
class Result:
    index: pd.DatetimeIndex
    gross: pd.Series           # daily book return before costs (total, i.e. includes rf)
    net: pd.Series             # after costs
    excess_gross: pd.Series    # gross - rf
    excess_net: pd.Series
    turnover: pd.Series        # one-way turnover traded on each day
    weights: np.ndarray        # (T, N) weights held at each close, after any close trade

    def sharpe(self, net: bool = True) -> float:
        x = (self.excess_net if net else self.excess_gross).dropna()
        return float(x.mean() / x.std(ddof=1) * np.sqrt(TDAYS))

    def ann_turnover(self) -> float:
        return float(self.turnover.mean() * TDAYS)


def _parse_fill(fill: str) -> tuple[str, int]:
    if fill == "close":
        return "close", 0
    if fill == "open":
        return "open", 1
    if fill.startswith("close+"):
        k = int(fill.split("+")[1])
        if k < 1:
            raise ValueError(fill)
        return "close", k
    raise ValueError(f"unknown fill {fill!r}")


def run(mkt: Market, target: np.ndarray, fill: str = "close", cost_bps: float = 0.0,
        start_flat: bool = True) -> Result:
    """Simulate a book.

    target: (T, N) target weights decided at the close of each day; a row of NaN means "no decision
            today, let the weights drift". Weight rows may sum to anything (cash = 1 - sum).
    fill:   see module docstring.
    cost_bps: cost per unit of one-way turnover, in basis points, charged at the fill.
    """
    target = np.asarray(target, float)
    if target.ndim == 1:
        target = target[:, None]
    T, N = mkt.close_ret.shape
    assert target.shape == (T, N), (target.shape, (T, N))
    kind, lag = _parse_fill(fill)
    # the decision of day t is executed on day t+lag (at its open if kind == "open", else its close)
    exe = np.full_like(target, np.nan)
    if lag == 0:
        exe[:] = target
    else:
        exe[lag:] = target[:-lag]
    dec_row = ~np.all(np.isnan(exe), axis=1)

    R = np.nan_to_num(mkt.close_ret, nan=0.0)
    if kind == "open":
        ON = mkt.on_ret
        if np.isnan(ON[dec_row & np.isfinite(R).all(1)]).any():
            bad = mkt.index[dec_row & np.isnan(ON).any(axis=1)]
            if len(bad):
                raise ValueError(f"open fill requested on days without an open price, e.g. {bad[0].date()}")
        ON = np.nan_to_num(ON, nan=0.0)
        IN = (1.0 + R) / (1.0 + ON) - 1.0
    rf = mkt.rf
    c = cost_bps / 1e4

    w = np.zeros(N)
    gross = np.zeros(T)
    net = np.zeros(T)
    turn = np.zeros(T)
    W = np.zeros((T, N))
    for t in range(T):
        tw = exe[t]
        has = dec_row[t]
        if kind == "open" and has:
            # overnight on the old weights
            g_on = float(w @ ON[t])
            w_d = w * (1.0 + ON[t]) / (1.0 + g_on) if (1.0 + g_on) != 0 else w
            new = np.where(np.isnan(tw), w_d, tw)
            tau = float(np.abs(new - w_d).sum())
            g_in = rf[t] + float(new @ (IN[t] - rf[t]))
            gt = (1.0 + g_on) * (1.0 + g_in) - 1.0
            nt = (1.0 + g_on) * (1.0 - c * tau) * (1.0 + g_in) - 1.0
            w = new * (1.0 + IN[t]) / (1.0 + g_in) if (1.0 + g_in) != 0 else new
            turn[t] = tau
        else:
            gt = rf[t] + float(w @ (R[t] - rf[t]))
            nt = gt
            w = w * (1.0 + R[t]) / (1.0 + gt) if (1.0 + gt) != 0 else w
            if kind == "close" and has:
                new = np.where(np.isnan(tw), w, tw)
                tau = float(np.abs(new - w).sum())
                nt = (1.0 + gt) * (1.0 - c * tau) - 1.0
                w = new
                turn[t] = tau
        gross[t], net[t] = gt, nt
        W[t] = w
    ix = mkt.index
    g, n = pd.Series(gross, ix), pd.Series(net, ix)
    rfs = pd.Series(rf, ix)
    return Result(ix, g, n, g - rfs, n - rfs, pd.Series(turn, ix), W)


def run_timing(mkt: Market, pos: pd.Series, fill: str = "close", cost_bps: float = 0.0) -> Result:
    """Single-asset long/flat (or any weight) rule. pos[t] is the target weight decided at close t;
    NaN during warm-up means 'flat, no decision' (the book starts in cash)."""
    p = pos.reindex(mkt.index).values.astype(float)
    tgt = np.where(np.isnan(p), 0.0, p)
    return run(mkt, tgt[:, None], fill, cost_bps)


def first_day_pnl(target: np.ndarray, mkt: Market) -> pd.Series:
    """Excess P&L of the first day after each change in a 0/1 (or constant-weight) target:
    d_t = sum_i (tgt_i,t - tgt_i,t-1) (R_i,t+1 - rf_t+1), dated t+1. Its sum is exactly the gross
    cost of filling one close later for a single long/flat asset."""
    tgt = np.nan_to_num(np.asarray(target, float).reshape(len(mkt.index), -1), nan=0.0)
    d = np.vstack([tgt[:1], np.diff(tgt, axis=0)])
    ex = np.nan_to_num(mkt.close_ret, nan=0.0) - mkt.rf[:, None]
    out = np.zeros(len(mkt.index))
    out[1:] = (d[:-1] * ex[1:]).sum(axis=1)
    return pd.Series(out, mkt.index)
