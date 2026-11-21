"""Download daily OHLC for SPY and the nine original SPDR sector ETFs and write
data/sector_ohlc_features.csv (the only input of the pipeline).

    python src/fetch_data.py          # needs yfinance and internet access to Yahoo Finance

File format (one row per day on which all ten ETFs traded, 1998-12-23 onwards), integers in
units of 1e-5 log points, for each ticker T:
    T_r  ln(AdjClose_t / AdjClose_{t-1})     close-to-close, dividend- and split-adjusted
    T_h  ln(High_t / Open_t)
    T_l  ln(Low_t / Open_t)
    T_c  ln(Close_t / Open_t)
The overnight return is T_r - T_c. Intraday ratios are scale-free, so they need no adjustment.

The committed file was produced on 2026-11-21 from Yahoo's chart API
(https://query1.finance.yahoo.com/v8/finance/chart/<T>?period1=0&period2=1800000000&interval=1d)
with exactly this transformation; this script reproduces it through yfinance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
OUT = Path(__file__).resolve().parents[1] / "data" / "sector_ohlc_features.csv"


def main():
    import yfinance as yf
    raw = yf.download(TICKERS, start="1998-12-01", auto_adjust=False, actions=False, progress=False)
    cols = {}
    for t in TICKERS:
        o, h, l, c, a = (raw[k][t] for k in ["Open", "High", "Low", "Close", "Adj Close"])
        cols[f"{t}_r"] = np.log(a / a.shift())
        cols[f"{t}_h"] = np.log(h / o)
        cols[f"{t}_l"] = np.log(l / o)
        cols[f"{t}_c"] = np.log(c / o)
    df = pd.DataFrame(cols).dropna()
    df = (df * 1e5).round().astype(int)
    df.index = df.index.strftime("%Y%m%d")
    df.index.name = "date"
    df.to_csv(OUT)
    print(f"wrote {OUT} with {len(df)} rows")


if __name__ == "__main__":
    main()
