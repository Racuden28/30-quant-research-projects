"""Data, real-time regime engine, strategies and statistics for project 19.

Timing convention (tested in tests/test_core.py):
    a probability or signal dated t uses returns up to and including day t;
    position[t] is decided at close t and held over the next day's return r[t+1].
The one deliberate exception is the "hindsight" ladder rungs, which are built to show how much
look-ahead is worth. They are labelled as such everywhere.

The statistics helpers at the bottom (Sharpe, stationary bootstrap, Newey-West, Ledoit-Wolf HAC
Sharpe-difference test, Holm) are copied from project 03 so this folder runs on its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import hmm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "index_closes.csv"
DIVS = ROOT / "data" / "shiller_dividends.csv"
RES = ROOT / "results"

TDAYS = 252
MIN_TRAIN = 1260      # 5 years of daily data before the first real-time fit (pre-registered)
REFIT = 21            # refit the HMM every 21 trading days on an expanding window (pre-registered)
PLACEBO_WIN = 1260    # trailing window for the matched volatility placebo
PLACEBO_MIN = 252     # ... with at least one year of real-time HMM signal
EWMA_LAMBDA = 0.94    # RiskMetrics
COST_BP = 5.0         # one-way cost per unit of turnover, bp

INDICES = ["GSPC", "IXIC", "GSPTSE", "FTSE", "GDAXI", "FCHI", "N225", "HSI", "AXJO", "BVSP"]
LABEL = {"GSPC": "S&P 500", "IXIC": "Nasdaq Composite", "GSPTSE": "TSX (Canada)", "FTSE": "FTSE 100",
         "GDAXI": "DAX", "FCHI": "CAC 40", "N225": "Nikkei 225", "HSI": "Hang Seng",
         "AXJO": "ASX 200", "BVSP": "Ibovespa"}
START = {"BVSP": "1995-01-01"}   # skip Brazilian hyperinflation, as in project 03
US = {"GSPC", "IXIC"}


# --------------------------------------------------------------------------- data
def load_closes() -> pd.DataFrame:
    return pd.read_csv(DATA, index_col=0, parse_dates=True)


def sp500_dividend_yield(index: pd.DatetimeIndex) -> pd.Series:
    """Daily dividend yield of the S&P 500 from Shiller's monthly data (12m dividend / monthly price),
    carried forward daily; held at its last value (2023-06) afterwards. Used only for a robustness run."""
    d = pd.read_csv(DIVS, parse_dates=["date"]).set_index("date")
    y = (d.dividend_12m / d.sp500_monthly_avg) / TDAYS
    # a monthly figure is known at the end of its month at the earliest: shift one month to stay real-time
    y.index = y.index + pd.offsets.MonthEnd(1)
    return y.reindex(index.union(y.index)).ffill().reindex(index).fillna(0.0)


def load_index(name: str, closes: pd.DataFrame | None = None, total_return: bool = False) -> pd.DataFrame:
    """Price, simple return, log return (%) and daily cash return for one index on its own trading days."""
    closes = load_closes() if closes is None else closes
    p = closes[name].dropna()
    if name in START:
        p = p[p.index >= START[name]]
    df = pd.DataFrame({"p": p})
    df["r"] = df.p.pct_change()
    if total_return:
        if name != "GSPC":
            raise ValueError("dividends are only available for the S&P 500")
        df["r"] = df.r + sp500_dividend_yield(df.index)
    df["x"] = 100 * np.log1p(df.r)
    if name in US:
        irx = closes["IRX"].dropna().reindex(df.index, method="ffill")
        df["rf"] = (irx.fillna(0.0) / 100.0) / TDAYS
        df.loc[df.index < "1960-01-04", "rf"] = 0.0
    else:
        df["rf"] = 0.0
    return df.dropna(subset=["r"])


# --------------------------------------------------------------------------- real-time HMM engine
@dataclass
class RealTime:
    filt: np.ndarray          # (T, K) P(s_t | r_1..t, params fitted on r_1..t_refit<=t)
    pred: np.ndarray          # (T, K) P(s_{t+1} | same information)
    exp_ret: np.ndarray       # (T,)  predicted next-day mean return, %: pred_t @ mu
    fits: list                # [(t_refit, HMMParams)]


def realtime(x: np.ndarray, K: int = 2, min_train: int = MIN_TRAIN, refit: int = REFIT,
             window: int | None = None, restart_every: int = 12, seed: int = 0) -> RealTime:
    """Expanding-window (or rolling `window`) refits every `refit` days. Between refits the filter keeps
    running forward with the last parameters. Each refit starts from the previous fit (warm start) and
    from the default start; every `restart_every`-th refit also tries 4 random starts. Highest likelihood wins.
    Nothing dated t uses any return after t (tests/test_core.py::test_realtime_has_no_look_ahead)."""
    T = len(x)
    rng = np.random.default_rng(seed)
    filt = np.full((T, K), np.nan)
    pred = np.full((T, K), np.nan)
    er = np.full(T, np.nan)
    fits, prev = [], None
    for j, tau in enumerate(range(min_train - 1, T, refit)):
        lo = 0 if window is None else max(0, tau + 1 - window)
        train = x[lo:tau + 1]
        cands = [hmm.fit(train, K)]
        if prev is not None:
            cands.append(hmm.fit(train, K, init=prev))
        if j % restart_every == 0:
            cands.append(hmm.fit(train, K, n_starts=5, rng=rng))
        best = max(cands, key=lambda c: c.loglik)
        prev = best
        end = min(T, tau + refit)
        f, _ = hmm.filtered(x[lo:end], best)
        filt[tau:end] = f[tau - lo:]
        pred[tau:end] = f[tau - lo:] @ best.A
        er[tau:end] = pred[tau:end] @ best.mu
        fits.append((tau, best))
    return RealTime(filt, pred, er, fits)


def ewma_vol(r: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """Annualised RiskMetrics volatility using returns up to and including day t."""
    v = (r ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(v * TDAYS)


def matched_vol_placebo(signal: pd.Series, vol: pd.Series, win: int = PLACEBO_WIN, min_obs: int = PLACEBO_MIN) -> pd.Series:
    """Invested iff vol[t] is at or below the q-th quantile of vol over the trailing window, where q is
    the fraction of the same window that `signal` was invested. Knows volatility only, never the HMM's
    view of returns; same average exposure; out in exactly the highest-volatility states."""
    s, v = signal.values, vol.values
    out = np.full(len(s), np.nan)
    for t in range(len(s)):
        a = max(0, t - win + 1)
        ws, wv = s[a:t + 1], v[a:t + 1]
        ok = ~np.isnan(ws) & ~np.isnan(wv)
        if ok.sum() < min_obs or np.isnan(v[t]):
            continue
        q = ws[ok].mean()
        thr = np.quantile(wv[ok], q) if q > 0 else -np.inf
        out[t] = float(v[t] <= thr)
    return pd.Series(out, index=signal.index)


def ma_signal(p: pd.Series, n: int = 200) -> pd.Series:
    ma = p.rolling(n, min_periods=n).mean()
    return (p > ma).astype(float).where(ma.notna())


@dataclass
class StratReturns:
    gross: pd.Series
    net: pd.Series
    turnover: float
    exposure: float


def strategy_returns(pos: pd.Series, r: pd.Series, rf: pd.Series, cost_bp: float = COST_BP) -> StratReturns:
    """Excess return of holding pos[t] over r[t+1]; cash earns rf; costs on |Δpos|."""
    pos_l = pos.shift(1)
    ex = pos_l * (r - rf)
    trades = pos_l.diff().abs()
    net = ex - trades * cost_bp / 1e4
    valid = pos_l.notna()
    return StratReturns(ex[valid], net[valid].fillna(ex[valid]),
                        float(trades[valid].sum() / valid.sum() * TDAYS), float(pos_l[valid].mean()))


def max_drawdown(x: pd.Series) -> float:
    w = (1 + x).cumprod()
    return float((w / w.cummax() - 1).min())


def spells(flag: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """(start, end_exclusive) of runs of True of length >= min_len."""
    f = np.asarray(flag, bool).astype(int)
    d = np.diff(np.concatenate([[0], f, [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(a, b) for a, b in zip(st, en) if b - a >= min_len]


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
