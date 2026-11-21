"""Where the data comes from.

* data/etf_daily_log_returns.csv is the same file as Project 01 (see ../01-monte-carlo-calibration/src/fetch_data.py:
  Yahoo adjusted closes -> NYSE-calendar log returns, 11 Apr 2006 to 20 Nov 2026).
* data/sp500_monthly_shiller.csv is fetched below from the datasets/s-and-p-500 repository (Shiller's series).
"""
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "sp500_monthly_shiller.csv"

if __name__ == "__main__":
    OUT.write_bytes(urlopen(URL, timeout=60).read())
    print("saved", OUT)
