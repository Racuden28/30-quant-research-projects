"""
Data for project 21. Everything is read from ../data, nothing is downloaded at run time.

* yahoo_ohlc.csv.gz   Daily open/high/low/close/adjclose/volume from Yahoo Finance for the ten equity
                      indices of projects 03 and 19, QQQ
                      and the 13-week T-bill yield (^IRX). Pulled 2026-11-23 through a browser
                      (src/fetch_yahoo_browser.js); the sample is cut at 2026-11-20, the last full day.
                      New relative to 03/19: the *open*, which is what an end-of-day system trades at.
* country_etf_daily.csv.gz  42 US-listed single-country ETFs + SPY with OHLC and adjusted close,
                      1996-2026, copied from project 12 (its universe B), deliberately.
* industries_daily.csv.gz   49 Fama-French industries, daily excess returns 1926-2026, copied from
                      project 12 (its universe A), deliberately.
* index_closes_p03.csv      the exact close file of projects 03 and 19 (to 2026-11-17), used only to
                      check that this engine reproduces their published numbers.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from engine import Market, market_from_ohlc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
FIG = ROOT / "figures"
END = "2026-11-20"
TDAYS = 252

INDICES = ["GSPC", "IXIC", "GSPTSE", "FTSE", "GDAXI", "FCHI", "N225", "HSI", "AXJO", "BVSP"]
LABEL = {"GSPC": "S&P 500", "IXIC": "Nasdaq Comp.", "GSPTSE": "TSX", "FTSE": "FTSE 100",
         "GDAXI": "DAX", "FCHI": "CAC 40", "N225": "Nikkei 225", "HSI": "Hang Seng",
         "AXJO": "ASX 200", "BVSP": "Ibovespa", "SP500TR": "S&P 500 TR", "QQQ": "QQQ"}
START = {"BVSP": "1995-01-01"}        # skip the Brazilian hyperinflation, as in projects 03 and 19
US = {"GSPC", "IXIC", "SP500TR", "QQQ"}
# the US-listed ETF that tracks (or, for the Nasdaq, nearly tracks) each index
ETF_OF = {"GSPC": "SPY", "IXIC": "QQQ", "GSPTSE": "EWC", "FTSE": "EWU", "GDAXI": "EWG", "FCHI": "EWQ",
          "N225": "EWJ", "HSI": "EWH", "AXJO": "EWA", "BVSP": "EWZ"}

STRESS = {"GFC 2008-09": ("2008-09-01", "2009-03-31"),
          "COVID 2020": ("2020-02-19", "2020-04-30"),
          "Inflation 2022": ("2022-01-03", "2022-10-31")}


@lru_cache(maxsize=None)
def _yahoo() -> pd.DataFrame:
    d = pd.read_csv(DATA / "yahoo_ohlc.csv.gz", parse_dates=["date"])
    d["ticker"] = d.ticker.str.lstrip("^")
    return d[d.date <= END]


@lru_cache(maxsize=None)
def _etfs() -> pd.DataFrame:
    return pd.read_csv(DATA / "country_etf_daily.csv.gz", parse_dates=["date"])


def ohlc(sym: str) -> pd.DataFrame:
    d = _yahoo()
    if sym in set(d.ticker):
        g = d[d.ticker == sym]
    else:
        g = _etfs()
        g = g[g.ticker == sym]
    g = g.set_index("date")[["open", "high", "low", "close", "adjclose", "volume"]].sort_index()
    g = g[~g.index.duplicated(keep="last")]
    if sym in START:
        g = g[g.index >= START[sym]]
    return g


def real_open_mask(g: pd.DataFrame, win: int = 250, thresh: float = 0.8) -> pd.Series:
    """True from the first date after which, in a trailing year, at least 80% of opens differ from both
    the same day's close and the previous close. Before that, Yahoo's 'open' is a copy of a close."""
    real = (g.open != g.close) & (g.open != g.close.shift()) & g.open.notna()
    share = real.rolling(win, min_periods=win).mean()
    ok = share >= thresh
    if not ok.any():
        return pd.Series(False, index=g.index)
    first = ok.idxmax() - pd.Timedelta(days=400)     # the window that first passed started a year earlier
    return pd.Series(g.index >= max(first, g.index[0]), index=g.index)


@lru_cache(maxsize=None)
def irx() -> pd.Series:
    """Daily cash return from the 13-week T-bill discount yield (%), zero before 1960."""
    g = _yahoo()
    y = g[g.ticker == "IRX"].set_index("date").close
    return (y / 100.0 / TDAYS).rename("rf")


def cash_for(sym: str, index: pd.DatetimeIndex, use_us_cash: bool) -> pd.Series:
    if not use_us_cash:
        return pd.Series(0.0, index=index)
    r = irx().reindex(index.union(irx().index)).ffill().reindex(index).fillna(0.0)
    return r


def single_market(sym: str, opens: bool = False, total_return: bool = True) -> tuple[Market, pd.DataFrame]:
    """Market for one instrument on its own trading days. US instruments (and the US-listed ETFs) use
    the T-bill as cash; foreign indices use zero cash, as in projects 03 and 19.
    If `opens`, the sample is restricted to the period with real opens."""
    g = ohlc(sym).dropna(subset=["close"])
    if opens:
        g = g[real_open_mask(g)]
    panel = g.copy()
    if not total_return:
        panel["adjclose"] = panel["close"]
    us_cash = sym in US or sym in ETF_OF.values()
    rf = cash_for(sym, g.index, us_cash)
    m = market_from_ohlc({sym: panel}, rf, g.index)
    if not opens:
        m.on_ret[:] = np.nan
    return m, g


def closes_p03() -> pd.DataFrame:
    return pd.read_csv(DATA / "index_closes_p03.csv", index_col=0, parse_dates=True)


def industries() -> tuple[pd.DataFrame, pd.Series]:
    p = pd.read_csv(DATA / "industries_daily.csv.gz", index_col=0, parse_dates=True)
    rf = p["RF"]
    ex = p.drop(columns=["MKT", "RF"])
    return ex, rf


def country_etfs() -> pd.DataFrame:
    return _etfs()
