"""Load the CBOE index panel (Yahoo daily closes) and the S&P 500, and define paths and windows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"
FIG = ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

INDICES = ["VIX9D", "VIX", "VIX3M", "VIX6M", "SKEW", "VVIX"]
START = "2008-01-02"  # first day on which VIX6M exists (VIX9D starts 2011-01-03)

STRESS = {
    "GFC 2008-09 to 2009-03": ("2008-09-01", "2009-03-31"),
    "Covid 2020-02 to 2020-05": ("2020-02-15", "2020-05-31"),
    "Bear 2022": ("2022-01-01", "2022-10-31"),
}


def _close(name: str) -> pd.Series:
    df = pd.read_csv(RAW / f"{name}.csv", parse_dates=["date"])
    s = pd.to_numeric(df["close"], errors="coerce")
    s.index = df["date"]
    s = s[~s.index.duplicated(keep="last")].dropna()
    return s.rename(name)


def load_panel() -> pd.DataFrame:
    """Daily panel 2008-01-02 .. last date, one row per day with all of VIX, VIX3M, VIX6M, SKEW, VVIX
    (VIX9D may be missing before 2011). Adds the S&P 500 close for the physical-measure checks."""
    cols = {n: _close(n) for n in INDICES + ["GSPC"]}
    df = pd.concat(cols.values(), axis=1)
    df = df.loc[START:]
    need = ["VIX", "VIX3M", "VIX6M", "SKEW", "VVIX", "GSPC"]
    df = df.dropna(subset=need)
    # basic sanity filters on obviously broken prints
    bad = (df["SKEW"] < 95) | (df["SKEW"] > 200) | (df["VVIX"] < 50) | (df["VVIX"] > 250)
    df = df[~bad]
    return df


def in_window(idx: pd.DatetimeIndex, win: tuple[str, str]) -> np.ndarray:
    return np.asarray((idx >= win[0]) & (idx <= win[1]))
