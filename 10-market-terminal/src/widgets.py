"""Terminal widgets and forward targets. Every widget at row t uses data up to and including the close of t only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ASSETS = ["SPY", "EEM", "TLT", "HYG", "GLD", "USO", "URA"]

WIDGETS = ["chg1d", "chg5d", "chg21d", "mom12_1", "ma200", "cross", "hi52", "dd",
           "rsi14", "boll", "z1d", "streak", "rv21", "volratio"]
WIDGET_LABEL = {
    "chg1d": "1-day change", "chg5d": "5-day change", "chg21d": "1-month change", "mom12_1": "12-1 momentum",
    "ma200": "Gap to 200-day MA", "cross": "Golden / death cross", "hi52": "Distance from 52-week high",
    "dd": "Drawdown from all-time high", "rsi14": "RSI (14)", "boll": "Bollinger %b", "z1d": "Today's move in σ",
    "streak": "Up/down streak", "rv21": "21-day realised vol", "volratio": "Vol ratio (5d / 63d)",
}
# Pre-registered grouping (PREREGISTRATION.md, H2/H3)
FAMILY = {"chg1d": "return", "chg5d": "return", "chg21d": "return", "rsi14": "return", "boll": "return",
          "streak": "return", "mom12_1": "trend", "ma200": "trend", "cross": "trend", "hi52": "trend",
          "dd": "trend", "z1d": "risk", "rv21": "risk", "volratio": "risk"}
TARGETS = ["ret", "vol", "mdd"]
TARGET_LABEL = {"ret": "next-month return", "vol": "next-month volatility", "mdd": "next-month drawdown"}


def load_returns() -> pd.DataFrame:
    return pd.read_csv(DATA / "etf_daily_log_returns.csv", index_col=0, parse_dates=True)[ASSETS]


def _rsi(r: pd.Series, n: int = 14) -> pd.Series:
    up, dn = r.clip(lower=0), (-r).clip(lower=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad.replace(0, np.nan))


def _streak(r: pd.Series) -> pd.Series:
    s = np.sign(r.values)
    out = np.zeros(len(s))
    for i in range(len(s)):
        if s[i] == 0 or np.isnan(s[i]):
            out[i] = 0
        elif i > 0 and np.sign(out[i - 1]) == s[i]:
            out[i] = out[i - 1] + s[i]
        else:
            out[i] = s[i]
    return pd.Series(out, index=r.index)


def compute_widgets(r: pd.Series) -> pd.DataFrame:
    """r: daily log returns of one asset with no interior gaps."""
    r = r.dropna()
    lp = r.cumsum()  # log price index
    p = np.exp(lp)
    w = pd.DataFrame(index=r.index)
    w["chg1d"] = r
    w["chg5d"] = r.rolling(5).sum()
    w["chg21d"] = r.rolling(21).sum()
    w["mom12_1"] = lp.shift(21) - lp.shift(252)
    sma200, sma50 = p.rolling(200).mean(), p.rolling(50).mean()
    w["ma200"] = lp - np.log(sma200)
    w["cross"] = np.where(sma200.isna(), np.nan, np.where(sma50 > sma200, 1.0, -1.0))
    w["hi52"] = lp - lp.rolling(252).max()
    w["dd"] = lp - lp.cummax()
    w["rsi14"] = _rsi(r)
    m20, s20 = p.rolling(20).mean(), p.rolling(20).std()
    w["boll"] = (p - (m20 - 2 * s20)) / (4 * s20)
    w["z1d"] = r / r.rolling(63).std()
    w["streak"] = _streak(r)
    w["rv21"] = np.sqrt(252 * (r ** 2).rolling(21).mean())
    rv5 = np.sqrt(252 * (r ** 2).rolling(5).mean())
    rv63 = np.sqrt(252 * (r ** 2).rolling(63).mean())
    w["volratio"] = np.log(rv5.replace(0, np.nan) / rv63)
    return w[WIDGETS]


def compute_targets(r: pd.Series, h: int = 21) -> pd.DataFrame:
    """Targets at row t use returns t+1..t+h only (NaN where the window is incomplete)."""
    r = r.dropna()
    a = r.values
    n = len(a)
    ret = np.full(n, np.nan)
    vol = np.full(n, np.nan)
    mdd = np.full(n, np.nan)
    cs = np.concatenate([[0.0], np.cumsum(a)])
    cs2 = np.concatenate([[0.0], np.cumsum(a ** 2)])
    for t in range(n - h):
        ret[t] = cs[t + 1 + h] - cs[t + 1]
        vol[t] = 0.5 * np.log(max(252 * (cs2[t + 1 + h] - cs2[t + 1]) / h, 1e-10))
        path = cs[t + 1:t + 2 + h] - cs[t + 1]  # includes 0 at start
        mdd[t] = np.min(path - np.maximum.accumulate(path))
    return pd.DataFrame({"ret": ret, "vol": vol, "mdd": mdd}, index=r.index)


# ---------------- monthly (Shiller) ----------------
M_WIDGETS = ["chg1m", "mom12_1", "ma10m", "hi12m", "dd", "rv12m", "z1m"]
M_LABEL = {"chg1m": "1-month change", "mom12_1": "12-1 momentum", "ma10m": "Gap to 10-month MA",
           "hi12m": "Distance from 12-month high", "dd": "Drawdown from all-time high",
           "rv12m": "12-month realised vol", "z1m": "Last month in σ"}
M_TARGETS = ["ret1", "ret12", "vol3", "mdd12"]


def load_shiller() -> pd.Series:
    d = pd.read_csv(DATA / "sp500_monthly_shiller.csv", parse_dates=["Date"], index_col="Date")
    p = d["SP500"].replace(0, np.nan).dropna()
    return np.log(p).diff().dropna()


def monthly_widgets(r: pd.Series) -> pd.DataFrame:
    lp = r.cumsum()
    w = pd.DataFrame(index=r.index)
    w["chg1m"] = r
    w["mom12_1"] = lp.shift(1) - lp.shift(12)
    w["ma10m"] = lp - np.log(np.exp(lp).rolling(10).mean())
    w["hi12m"] = lp - lp.rolling(12).max()
    w["dd"] = lp - lp.cummax()
    w["rv12m"] = np.sqrt(12 * (r ** 2).rolling(12).mean())
    w["z1m"] = r / r.rolling(36).std()
    return w[M_WIDGETS]


def monthly_targets(r: pd.Series, skip: int = 1) -> pd.DataFrame:
    """Targets at month t start after month t+skip (Working-effect guard for averaged prices)."""
    a = r.values
    n = len(a)
    cs = np.concatenate([[0.0], np.cumsum(a)])
    cs2 = np.concatenate([[0.0], np.cumsum(a ** 2)])
    out = {k: np.full(n, np.nan) for k in M_TARGETS}
    s0 = 1 + skip  # first return included: index t+1+skip
    for t in range(n):
        st = t + s0
        if st + 1 <= n:
            out["ret1"][t] = cs[st + 1] - cs[st]
        if st + 3 <= n:
            out["vol3"][t] = 0.5 * np.log(max(12 * (cs2[st + 3] - cs2[st]) / 3, 1e-10))
        if st + 12 <= n:
            out["ret12"][t] = cs[st + 12] - cs[st]
            path = cs[st:st + 13] - cs[st]
            out["mdd12"][t] = np.min(path - np.maximum.accumulate(path))
    return pd.DataFrame(out, index=r.index)
