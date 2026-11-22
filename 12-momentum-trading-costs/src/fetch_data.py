"""
Rebuild the two data files from their public sources (needs internet; ~2 minutes).

Universe A: copied from project 09 (Ken French 49 industries, daily, CRSP vintage 202609). To rebuild
from French's site, run ../09-beta-horse-race/src/fetch_data.py and copy its data/panel_daily.csv.gz
to data/industries_daily.csv.gz.

Universe B: daily OHLCV + adjusted close for the 42 country ETFs and SPY from Yahoo Finance's chart
API. The committed file was downloaded on 2026-11-22 with src/fetch_yahoo_browser.js (same endpoint,
same fields, run from a browser because Yahoo blocks the cloud machine this project was built on).
Yahoo revises adjusted closes when dividends are restated, so a fresh download can differ in the
last digits.
"""
from __future__ import annotations

import gzip
import io
import json
import time
import urllib.request
from datetime import datetime, timezone

import data

TICKERS = list(data.COUNTRY) + ["SPY", "EWUS"]
URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1=0&period2=1795000000"
       "&interval=1d&events=div,split")


def fmt(x):
    return "" if x is None else f"{float(x):.7g}"


def main():
    buf = io.StringIO()
    buf.write("ticker,date,open,high,low,close,adjclose,volume\n")
    for t in TICKERS:
        req = urllib.request.Request(URL.format(t=t), headers={"User-Agent": "Mozilla/5.0"})
        j = json.loads(urllib.request.urlopen(req, timeout=60).read())
        r = j["chart"]["result"][0]
        q, adj = r["indicators"]["quote"][0], r["indicators"]["adjclose"][0]["adjclose"]
        for i, ts in enumerate(r["timestamp"]):
            d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            v = q["volume"][i]
            buf.write(f"{t},{d},{fmt(q['open'][i])},{fmt(q['high'][i])},{fmt(q['low'][i])},"
                      f"{fmt(q['close'][i])},{fmt(adj[i])},{'' if v is None else v}\n")
        print(t, len(r["timestamp"]))
        time.sleep(0.5)
    with gzip.open(data.DATA / "country_etf_daily.csv.gz", "wt") as f:
        f.write(buf.getvalue())


if __name__ == "__main__":
    main()
