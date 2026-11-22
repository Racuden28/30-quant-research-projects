"""
Load the two universes used in project 12.

Universe A - 49 US industries (Ken French Data Library, CRSP vintage 202609), daily value-weighted
*excess* returns 1926-07 to 2026-09. The file is the panel built by project 09 (same vintage,
same parser), copied here so the project is self-contained. Columns: 49 industries, MKT, RF.

Universe B - 42 single-country equity ETFs listed in the US (iShares MSCI country funds plus
VanEck Vietnam and Global X Greece/Argentina), daily OHLCV and dividend-adjusted close from Yahoo
Finance, 1996-03 to 2026-11-17. Built by src/fetch_data.py (or, from a cloud machine where Yahoo is
blocked, by src/fetch_yahoo_browser.js run in a browser). The 3-month T-bill (^IRX, from project 03)
is the cash rate for this universe.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

# One country per fund. EWUS (UK small caps) duplicates EWU and is excluded; SPY is kept only as a
# reference for the spread-estimator check in costs_check.py.
COUNTRY = {
    "EWA": "Australia", "EWC": "Canada", "EWD": "Sweden", "EWG": "Germany", "EWH": "Hong Kong",
    "EWI": "Italy", "EWJ": "Japan", "EWK": "Belgium", "EWL": "Switzerland", "EWM": "Malaysia",
    "EWN": "Netherlands", "EWO": "Austria", "EWP": "Spain", "EWQ": "France", "EWS": "Singapore",
    "EWU": "United Kingdom", "EWW": "Mexico", "EWZ": "Brazil", "EWT": "Taiwan", "EWY": "South Korea",
    "EZA": "South Africa", "FXI": "China", "ECH": "Chile", "EIS": "Israel", "TUR": "Turkey",
    "THD": "Thailand", "EPU": "Peru", "EIDO": "Indonesia", "EPHE": "Philippines", "ENZL": "New Zealand",
    "EPOL": "Poland", "EIRL": "Ireland", "ENOR": "Norway", "EDEN": "Denmark", "EFNL": "Finland",
    "INDA": "India", "GREK": "Greece", "ARGT": "Argentina", "KSA": "Saudi Arabia", "QAT": "Qatar",
    "UAE": "UAE", "VNM": "Vietnam",
}

LAST_MONTH_A = "2026-09"     # last complete month in the French vintage
LAST_MONTH_B = "2026-10"     # Yahoo data end 2026-11-17; November is incomplete


def industries() -> pd.DataFrame:
    """Daily excess returns of the 49 industries (decimal), NaN where an industry does not exist."""
    p = pd.read_csv(DATA / "industries_daily.csv.gz", index_col=0, parse_dates=True)
    return p.drop(columns=["MKT", "RF"])


def industries_rf() -> pd.Series:
    p = pd.read_csv(DATA / "industries_daily.csv.gz", index_col=0, parse_dates=True)
    return p["RF"]


def etf_raw() -> pd.DataFrame:
    d = pd.read_csv(DATA / "country_etf_daily.csv.gz", parse_dates=["date"])
    return d.dropna(subset=["close", "high", "low", "adjclose"])


def etf_clean(min_dollar_volume: float = 0.0) -> dict[str, pd.DataFrame]:
    """Wide daily frames for universe B.

    Returns dict with 'ret' (daily excess total returns), 'close', 'dv' (dollar volume) and
    'stale' (True on days with zero volume or high == low, i.e. no informative trade).
    A day only counts as a return observation if the fund traded that day; stale days carry the
    return forward to the next traded day, which is what an investor holding the fund earns.
    """
    d = etf_raw()
    d = d[d.ticker.isin(COUNTRY)]
    adj = d.pivot(index="date", columns="ticker", values="adjclose").sort_index()
    close = d.pivot(index="date", columns="ticker", values="close").reindex(adj.index)
    vol = d.pivot(index="date", columns="ticker", values="volume").reindex(adj.index).fillna(0.0)
    hi = d.pivot(index="date", columns="ticker", values="high").reindex(adj.index)
    lo = d.pivot(index="date", columns="ticker", values="low").reindex(adj.index)
    ret = adj.pct_change(fill_method=None)
    stale = (vol <= 0) | (hi <= lo)
    irx = pd.read_csv(DATA / "irx_daily.csv", index_col=0, parse_dates=True)["IRX"]
    rf = (irx.reindex(adj.index).ffill() / 100.0) / 252.0
    ret = ret.sub(rf, axis=0)
    return {"ret": ret, "close": close, "dv": close * vol, "stale": stale, "rf": rf}


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().values)


if __name__ == "__main__":
    a = industries()
    print("A:", a.shape, a.index[0].date(), a.index[-1].date())
    b = etf_clean()
    r = b["ret"]
    print("B:", r.shape, r.index[0].date(), r.index[-1].date())
    big = (r.abs() > 0.25).sum()
    print("daily |r|>25%:", big[big > 0].to_dict())
    print("stale share by fund (first 3 years of each):")
    for t in r.columns:
        s = b["stale"][t][r[t].notna()]
        print(f"  {t:5s} {COUNTRY[t]:15s} first {s.index[0].date()}  stale {s.iloc[:756].mean():.0%}  "
              f"after-2010 {s[s.index >= '2010'].mean():.0%}")
