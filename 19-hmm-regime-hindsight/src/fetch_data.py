"""Download the daily closes used in project 19 (identical to project 03) into data/index_closes.csv.

The committed CSV was pulled from Yahoo Finance's chart endpoint on 2026-11-18. Price indices
(not total return): long total-return histories are not freely available for these markets.
Results can differ marginally if Yahoo revises history.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["^GSPC", "^IXIC", "^GSPTSE", "^FTSE", "^GDAXI", "^N225", "^HSI", "^AXJO", "^BVSP", "^FCHI", "^IRX"]
OUT = Path(__file__).resolve().parents[1] / "data" / "index_closes.csv"

if __name__ == "__main__":
    frames = {}
    for t in TICKERS:
        h = yf.Ticker(t).history(period="max", interval="1d", auto_adjust=False)
        frames[t.lstrip("^")] = h["Close"].tz_localize(None).rename(None)
    df = pd.concat(frames, axis=1).sort_index()
    df.index.name = "date"
    df.to_csv(OUT, float_format="%.7g")
    print(df.shape, "->", OUT)
