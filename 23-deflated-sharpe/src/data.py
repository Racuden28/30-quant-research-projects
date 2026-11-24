"""Load the two real panels used by the strategy zoo.

Zoo A: 10 equity price indices, daily, 1928-2026 (the file from project 03).
Zoo B: 49 Fama-French industry portfolios, monthly excess returns, 1926-2026 (the daily file from project 12).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True)

INDICES = ["GSPC", "IXIC", "GSPTSE", "FTSE", "GDAXI", "N225", "HSI", "AXJO", "BVSP", "FCHI"]
LABEL = {"GSPC": "S&P 500", "IXIC": "Nasdaq Comp.", "GSPTSE": "TSX", "FTSE": "FTSE 100", "GDAXI": "DAX",
         "N225": "Nikkei 225", "HSI": "Hang Seng", "AXJO": "ASX 200", "BVSP": "Ibovespa", "FCHI": "CAC 40"}
US = {"GSPC", "IXIC"}
BVSP_START = "1995-01-01"  # skip the hyperinflation years, as in project 03


def load_indices():
    """Union-calendar panel of daily excess simple returns (NaN where the index does not trade)."""
    px = pd.read_csv(DATA / "index_closes.csv", parse_dates=["date"], index_col="date")
    ind = pd.read_csv(DATA / "industries_daily.csv.gz", parse_dates=[0], index_col=0)
    rf_ff = ind["RF"]  # daily simple, through 2026-09
    irx = px["IRX"].ffill() / 100 / 252
    out = {}
    for c in INDICES:
        p = px[c].dropna()
        if c == "BVSP":
            p = p[p.index >= BVSP_START]
        r = p.pct_change().dropna()
        if c in US:
            rf = rf_ff.reindex(r.index)
            rf = rf.fillna(irx.reindex(r.index)).fillna(0.0)
        else:
            rf = 0.0  # local cash not available; same convention as project 03
        out[c] = r - rf
    X = pd.DataFrame(out).sort_index()
    return X


def load_industries():
    """Monthly excess returns of the 49 industries (NaN where missing) and the EW market."""
    d = pd.read_csv(DATA / "industries_daily.csv.gz", parse_dates=[0], index_col=0)
    rf = d.pop("RF")
    d = d.drop(columns=["MKT"])
    raw = d.add(rf, axis=0)  # file stores excess returns; rebuild raw to compound correctly
    g = raw.index.to_period("M")
    valid = raw.notna().groupby(g).sum()
    mret = (1 + raw.fillna(0)).groupby(g).prod() - 1
    mret = mret.where(valid >= 15)
    mrf = (1 + rf).groupby(g).prod() - 1
    X = mret.sub(mrf, axis=0)
    X.index = X.index.to_timestamp("M")
    return X


if __name__ == "__main__":
    X = load_indices()
    print(X.apply(lambda s: s.first_valid_index()).to_string())
    print(X.shape, X.index[-1])
    B = load_industries()
    print(B.shape, B.index[0], B.index[-1], B.notna().sum(1).describe())
    # sanity: annualised excess mean / vol
    print((X.mean() * 252).round(3).to_dict())
    print((B.mean().mean() * 12).round(4))
