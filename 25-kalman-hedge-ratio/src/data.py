"""
Data used by project 25. Everything is reused on purpose from earlier projects and copied unchanged into data/:

  industries_excess_daily.csv.gz  49 Fama-French industries, daily excess returns, plus the value-weighted
                                  market MKT and RF, Jul 1926 - Sep 2026 (built by project 09 from the Ken French
                                  Data Library, vintage 202609)
  p09_forecast_panel.csv.gz       project 09's month-end beta forecasts of nine estimators and the next-month
                                  sums needed to score them (results/forecast_panel.csv.gz of project 09)
  returns_countries.csv.gz        SPY and 21 iShares single-country ETFs, daily total returns from Yahoo adjusted
                                  closes, Mar 1996 - 20 Nov 2026 (project 11)
  returns_industries.csv.gz       the 49 industries as daily total returns (project 11)
  industries_nfirms_monthly.csv   firms per industry and month (project 11's eligibility rule)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

COUNTRY_NAME = {"SPY": "United States", "EWA": "Australia", "EWC": "Canada", "EWD": "Sweden", "EWG": "Germany",
                "EWH": "Hong Kong", "EWI": "Italy", "EWJ": "Japan", "EWK": "Belgium", "EWL": "Switzerland",
                "EWM": "Malaysia", "EWN": "Netherlands", "EWO": "Austria", "EWP": "Spain", "EWQ": "France",
                "EWS": "Singapore", "EWU": "United Kingdom", "EWW": "Mexico", "EWY": "South Korea",
                "EWT": "Taiwan", "EWZ": "Brazil", "EZA": "South Africa"}

# Country ETFs: each series starts after the last calendar year in which 10 % or more of its daily returns were
# exactly zero (stale prices; see project 11), and evaluation starts in January 2006 for all.
COUNTRY_ZERO_MAX = 0.10
COUNTRY_EVAL_START = "2006-01-01"
INDUSTRY_EVAL = ("1929-08", "2026-09")       # project 09's evaluation months


def hedging_panel(universe: str):
    """Return (Y DataFrame of hedged assets, x Series of the hedge instrument, evaluation start)."""
    if universe == "industries":
        p = pd.read_csv(DATA / "industries_excess_daily.csv.gz", index_col=0, parse_dates=True)
        x = p.pop("MKT")
        p.pop("RF")
        return p, x, pd.Timestamp(INDUSTRY_EVAL[0] + "-01")
    if universe == "countries":
        r = pd.read_csv(DATA / "returns_countries.csv.gz", index_col=0, parse_dates=True)
        x = r.pop("SPY")
        zero = (r == 0).groupby(r.index.year).mean()
        for c in r.columns:
            bad = zero.index[zero[c] >= COUNTRY_ZERO_MAX]
            if len(bad):
                r.loc[r.index.year <= bad.max(), c] = np.nan
        keep = x.notna()
        return r[keep], x[keep], pd.Timestamp(COUNTRY_EVAL_START)
    raise ValueError(universe)


def pairs_panel(universe: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"returns_{universe}.csv.gz", index_col=0, parse_dates=True)


def p09_panel() -> pd.DataFrame:
    return pd.read_csv(DATA / "p09_forecast_panel.csv.gz")
