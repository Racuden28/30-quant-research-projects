"""Core building blocks for project 03: data, signals, placebos and statistics.

Timing convention used everywhere (and tested in tests/test_core.py):
    signal[t] uses closes up to and including day t;
    position[t] = signal[t] is held over the return from close t to close t+1, r[t+1].
So a strategy's return on day t+1 is position[t] * r[t+1]. Nothing on day t+1 is known at t.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "index_closes.csv"

TDAYS = 252
VOL_WIN = 21          # trailing realised-vol window used to define the volatility state (pre-registered)
PLACEBO_WIN = 1260    # rolling 5y window for the matched placebo's exposure target and vol quantile
COST_BP = 5.0         # one-way cost per unit of turnover, in basis points (index futures / ETF level)

INDICES = ["GSPC", "IXIC", "GSPTSE", "FTSE", "GDAXI", "FCHI", "N225", "HSI", "AXJO", "BVSP"]
LABEL = {"GSPC": "S&P 500", "IXIC": "Nasdaq Composite", "GSPTSE": "TSX (Canada)", "FTSE": "FTSE 100",
         "GDAXI": "DAX", "FCHI": "CAC 40", "N225": "Nikkei 225", "HSI": "Hang Seng",
         "AXJO": "ASX 200", "BVSP": "Ibovespa"}
# Ibovespa before July 1994 is a nominal index under hyperinflation; start it after the Real Plan settled.
START = {"BVSP": "1995-01-01"}
US = {"GSPC", "IXIC"}


def load_closes() -> pd.DataFrame:
    return pd.read_csv(DATA, index_col=0, parse_dates=True)


def load_index(name: str, closes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Price, simple return and daily cash return for one index, on that market's own trading days."""
    closes = load_closes() if closes is None else closes
    p = closes[name].dropna()
    if name in START:
        p = p[p.index >= START[name]]
    df = pd.DataFrame({"p": p})
    df["r"] = df.p.pct_change()
    if name in US:
        # 13-week T-bill discount yield (%, annualised), available from 1960; zero before.
        irx = closes["IRX"].dropna().reindex(df.index, method="ffill")
        df["rf"] = (irx.fillna(0.0) / 100.0) / TDAYS
        df.loc[df.index < "1960-01-04", "rf"] = 0.0
    else:
        # Local price index without local cash rate: excess return = price return (see Limitations).
        df["rf"] = 0.0
    return df.dropna(subset=["r"])


# --------------------------------------------------------------------------- signals
def ma_signal(p: pd.Series, n: int) -> pd.Series:
    """1 if close > n-day simple moving average of closes (including today), else 0. NaN during warm-up."""
    ma = p.rolling(n, min_periods=n).mean()
    s = (p > ma).astype(float)
    return s.where(ma.notna())


def crossover_signal(p: pd.Series, fast: int, slow: int) -> pd.Series:
    f = p.rolling(fast, min_periods=fast).mean()
    s_ = p.rolling(slow, min_periods=slow).mean()
    return (f > s_).astype(float).where(s_.notna())


def trailing_vol(r: pd.Series, win: int = VOL_WIN) -> pd.Series:
    """Annualised std of the last `win` daily returns, including today's."""
    return r.rolling(win, min_periods=win).std() * np.sqrt(TDAYS)


def matched_vol_placebo(signal: pd.Series, vol: pd.Series, win: int = PLACEBO_WIN) -> pd.Series:
    """A signal that knows nothing about price trend, only about current volatility.

    At each day t it is invested iff vol[t] is at or below the q-th quantile of vol over the
    trailing `win` days, where q is the fraction of those same `win` days the trend rule was
    invested. Both inputs use data up to t only, so the placebo is tradeable, has (on average)
    the same exposure as the trend rule, and is out of the market in the highest-vol states.
    """
    s = signal.values
    v = vol.values
    out = np.full(len(s), np.nan)
    for t in range(win, len(s)):
        ws, wv = s[t - win + 1:t + 1], v[t - win + 1:t + 1]
        ok = ~np.isnan(ws) & ~np.isnan(wv)
        if ok.sum() < win // 2 or np.isnan(v[t]):
            continue
        q = ws[ok].mean()
        thr = np.quantile(wv[ok], q) if q > 0 else -np.inf
        out[t] = float(v[t] <= thr)
    return pd.Series(out, index=signal.index)


def spell_shuffle(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random signal with exactly the same in/out spell lengths (same exposure and turnover),
    in random order, starting state random. The null of 'no timing information at all'."""
    s = signal.astype(int)
    change = np.flatnonzero(np.diff(s)) + 1
    bounds = np.concatenate([[0], change, [len(s)]])
    lens = np.diff(bounds)
    vals = s[bounds[:-1]]
    on, off = rng.permutation(lens[vals == 1]), rng.permutation(lens[vals == 0])
    state = int(rng.integers(2))
    out, i, j = [], 0, 0
    while i < len(on) or j < len(off):
        if (state == 1 and i < len(on)) or j >= len(off):
            out.append(np.ones(on[i], int)); i += 1; state = 0
        else:
            out.append(np.zeros(off[j], int)); j += 1; state = 1
    return np.concatenate(out)[:len(s)].astype(float)


@dataclass
class StratReturns:
    gross: pd.Series      # excess return over cash, before costs
    net: pd.Series        # excess return after costs
    turnover: float       # one-way trades per year
    exposure: float       # average fraction of days invested


def strategy_returns(pos: pd.Series, r: pd.Series, rf: pd.Series, cost_bp: float = COST_BP) -> StratReturns:
    """Excess return of holding `pos[t]` over r[t+1]; cash earns rf. Costs on |Δpos|."""
    pos_l = pos.shift(1)                      # position held during day t (decided at close t-1)
    ex = pos_l * (r - rf)
    trades = pos_l.diff().abs()
    net = ex - trades * cost_bp / 1e4
    valid = pos_l.notna()
    return StratReturns(ex[valid], net[valid].fillna(ex[valid]),
                        float(trades[valid].sum() / valid.sum() * TDAYS), float(pos_l[valid].mean()))


# --------------------------------------------------------------------------- statistics
def sharpe(x: np.ndarray, axis=0) -> np.ndarray:
    return np.nanmean(x, axis=axis) / np.nanstd(x, axis=axis, ddof=1) * np.sqrt(TDAYS)


def stationary_bootstrap_idx(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices (circular)."""
    new = rng.random(n) < 1.0 / mean_block
    new[0] = True
    block_id = np.cumsum(new) - 1
    starts = rng.integers(n, size=block_id[-1] + 1)
    first = np.flatnonzero(new)
    offset = np.arange(n) - first[block_id]
    return (starts[block_id] + offset) % n


def nw_se_mean(x: np.ndarray, lags: int) -> float:
    """Newey-West standard error of a sample mean."""
    x = x - x.mean()
    n = len(x)
    g = np.dot(x, x) / n
    for L in range(1, lags + 1):
        g += 2 * (1 - L / (lags + 1)) * np.dot(x[L:], x[:-L]) / n
    return float(np.sqrt(g / n))


def sharpe_diff_hac(a: np.ndarray, b: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Ledoit-Wolf (2008) HAC test of SR(a) - SR(b) via the delta method. Returns (diff, p-value).
    Sharpe ratios are annualised; the test is invariant to that scaling."""
    from scipy.stats import norm
    m1, m2 = a.mean(), b.mean()
    g1, g2 = (a ** 2).mean(), (b ** 2).mean()
    d = m1 / np.sqrt(g1 - m1 ** 2) - m2 / np.sqrt(g2 - m2 ** 2)
    y = np.column_stack([a - m1, b - m2, a ** 2 - g1, b ** 2 - g2])
    n = len(a)
    S = y.T @ y / n
    for L in range(1, lags + 1):
        G = y[L:].T @ y[:-L] / n
        S += (1 - L / (lags + 1)) * (G + G.T)
    grad = np.array([g1 / (g1 - m1 ** 2) ** 1.5, -g2 / (g2 - m2 ** 2) ** 1.5,
                     -0.5 * m1 / (g1 - m1 ** 2) ** 1.5, 0.5 * m2 / (g2 - m2 ** 2) ** 1.5])
    se = np.sqrt(grad @ S @ grad / n)
    return float(d * np.sqrt(TDAYS)), float(2 * (1 - norm.cdf(abs(d / se))))


def romano_wolf(stat: np.ndarray, boot: np.ndarray) -> np.ndarray:
    """Romano-Wolf step-down adjusted p-values for H0_k: theta_k <= 0 (one-sided), non-studentised.
    stat: (K,) observed statistics; boot: (B, K) bootstrap statistics already centred at the
    observed value (boot - stat), i.e. draws from the null distribution."""
    K = len(stat)
    order = np.argsort(-stat)
    p_adj = np.empty(K)
    prev = 0.0
    for step, k in enumerate(order):
        remaining = order[step:]
        mx = boot[:, remaining].max(axis=1)
        p = (mx >= stat[k]).mean()
        prev = max(prev, p)
        p_adj[k] = prev
    return p_adj


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, (m - i) * p[k])
        adj[k] = min(1.0, running)
    return adj


def max_drawdown(x: pd.Series) -> float:
    w = (1 + x).cumprod()
    return float((w / w.cummax() - 1).min())
