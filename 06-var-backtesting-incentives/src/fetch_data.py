"""
Rebuild data/etf_daily_log_returns.csv from Yahoo Finance.

Daily adjusted closes (dividends and splits included) -> log returns on the NYSE calendar,
10 Apr 2006 to the end date below. Returns are stored in decimal units rounded to 1 basis point.
Values may differ in the last digit from the committed file if Yahoo revises its adjustments.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

TICKERS = ["SPY", "EEM", "TLT", "GLD", "USO", "URA", "HYG"]
START, END = "2006-04-10", "2026-11-20"
OUT = Path(__file__).resolve().parents[1] / "data" / "etf_daily_log_returns.csv"

px = yf.download(TICKERS, start=START, end=pd.Timestamp(END) + pd.Timedelta(days=1),
                 auto_adjust=True, progress=False)["Close"][TICKERS]
days = mcal.get_calendar("NYSE").valid_days(START, END).tz_localize(None)
px = px.reindex(days)
ret = np.log(px).diff().iloc[1:].round(4)
ret.index.name = "date"
ret.to_csv(OUT, float_format="%.4f")
print("saved", OUT, ret.shape)
