"""Load the three data sets used in project 16. Nothing is downloaded here; see data/README.md."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

ETFS = ["SPY", "EEM", "HYG", "USO", "TLT", "GLD", "URA"]
EWMA_LAMBDA = 0.94  # RiskMetrics daily decay
T_DAYS = 21          # one-month options
H_GRID = [1, 2, 3, 4, 5, 10, 21]   # rebalancing intervals in trading days (21 = hedge once, never touch)
H_FIT = [1, 2, 3, 4, 5]            # intervals used to fit the linear frequency law (daily to weekly)


def ewma_vol(r: pd.Series, lam: float = EWMA_LAMBDA, warmup: int = 60) -> pd.Series:
    """Annualised EWMA volatility known at the close of each day (uses that day's return).

    Seeded with the sample variance of the first `warmup` returns; those days are set to NaN.
    """
    x = r.fillna(0.0).to_numpy()
    v = np.empty_like(x)
    v0 = np.var(x[:warmup])
    for i, xi in enumerate(x):
        v0 = lam * v0 + (1 - lam) * xi * xi
        v[i] = v0
    out = pd.Series(np.sqrt(252 * v), index=r.index)
    out.iloc[:warmup] = np.nan
    return out


def spx() -> pd.DataFrame:
    """S&P 500 price index (^GSPC) daily close 1927-12-30 to 2026-11-17, with VIX where available."""
    d = pd.read_csv(DATA / "gspc_daily.csv", parse_dates=["date"]).set_index("date")
    d["r"] = np.log(d.close).diff()
    d["ewma"] = ewma_vol(d.r)
    vix = pd.read_csv(DATA / "vix_daily.csv", parse_dates=["DATE"]).set_index("DATE").CLOSE / 100.0
    d["vix"] = vix.reindex(d.index)
    return d.iloc[1:]


def etf_prices() -> dict[str, pd.DataFrame]:
    """Total-return price paths for the 7 ETFs of project 01 (rebuilt from adjusted log returns)."""
    lr = pd.read_csv(DATA / "etf_daily_log_returns.csv", parse_dates=["date"]).set_index("date")
    out = {}
    for a in ETFS:
        r = lr[a].dropna()
        d = pd.DataFrame({"r": r})
        d["close"] = 100.0 * np.exp(r.cumsum())
        d["ewma"] = ewma_vol(r)
        out[a] = d
    return out


def universe() -> dict[str, pd.DataFrame]:
    """Every underlying in the study, keyed by label. SPX uses its EWMA quote here (VIX is separate)."""
    u = {"SPX": spx()}
    u.update(etf_prices())
    return u
