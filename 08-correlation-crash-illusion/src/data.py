"""
Build the two research universes as daily log returns.

* STOCKS: 19 US large caps, 2 Jan 1990 to 28 Dec 2022, adjusted closes shipped with the open-source
  `skfolio` package (skfolio.datasets.load_sp500_dataset). The package ships 20; Range Resources (RRC)
  is removed because its early quotes are stale: 70-92% of days have an unchanged price in 1990-92
  (vs under 15% for every other stock) and a spurious +110% log return on 10 Apr 1990. Stale prices
  bias correlations towards zero, which is exactly the quantity under study.
* ETFS: six cross-asset ETFs (SPY, EEM, TLT, HYG, GLD, USO), 12 Apr 2007 to 20 Nov 2026, from the
  Yahoo Finance adjusted closes already committed in project 01. URA is dropped because it starts in
  Nov 2010 and would cut the Global Financial Crisis out of the sample.

Output: data/stocks_log_returns.csv, data/etfs_log_returns.csv (decimal units).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ETF_SRC = ROOT.parent / "01-monte-carlo-calibration" / "data" / "etf_daily_log_returns.csv"
DROPPED = {"RRC": "stale quotes 1990-1996"}
ETFS = ["SPY", "EEM", "TLT", "HYG", "GLD", "USO"]

SECTOR = {"AAPL": "Tech", "AMD": "Tech", "MSFT": "Tech", "BAC": "Financials", "JPM": "Financials",
          "BBY": "Cons. disc.", "HD": "Cons. disc.", "CVX": "Energy", "XOM": "Energy",
          "GE": "Industrials", "JNJ": "Health", "LLY": "Health", "MRK": "Health", "PFE": "Health",
          "UNH": "Health", "KO": "Staples", "PEP": "Staples", "PG": "Staples", "WMT": "Staples"}


def build():
    from skfolio.datasets import load_sp500_dataset

    DATA.mkdir(exist_ok=True)
    px = load_sp500_dataset().drop(columns=["RRC"])
    r = np.log(px).diff().iloc[1:]
    r.index.name = "date"
    r.to_csv(DATA / "stocks_log_returns.csv", float_format="%.6f")

    e = pd.read_csv(ETF_SRC, index_col=0, parse_dates=True)[ETFS].dropna()
    e.to_csv(DATA / "etfs_log_returns.csv", float_format="%.4f")
    print("stocks", r.shape, r.index[0].date(), r.index[-1].date())
    print("etfs  ", e.shape, e.index[0].date(), e.index[-1].date())
    print("share of exact-zero daily returns, stocks:", float((r == 0).mean().mean()).__round__(4))


def load(universe: str) -> pd.DataFrame:
    f = DATA / ("stocks_log_returns.csv" if universe == "stocks" else "etfs_log_returns.csv")
    return pd.read_csv(f, index_col=0, parse_dates=True)


if __name__ == "__main__":
    build()
