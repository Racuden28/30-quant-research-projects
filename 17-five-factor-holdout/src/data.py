"""Load the Ken French monthly files (CRSP vintage 202609) into one tidy panel of decimal returns.

Raw files live in data/raw/ as the first table of each official zip ("Average Value Weighted Returns --
Monthly" for portfolios), extracted verbatim by src/fetch_data.js. Missing values (-99.99, -999) are NaN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

FACTORS = ["Mkt", "SMB", "HML", "RMW", "CMA", "UMD"]
MODELS = {
    "CAPM": ["Mkt"],
    "FF3": ["Mkt", "SMB", "HML"],
    "C4": ["Mkt", "SMB", "HML", "UMD"],
    "FF5": ["Mkt", "SMB", "HML", "RMW", "CMA"],
    "FF5-HML": ["Mkt", "SMB", "RMW", "CMA"],
    "FF6": ["Mkt", "SMB", "HML", "RMW", "CMA", "UMD"],
    "FF6-HML": ["Mkt", "SMB", "RMW", "CMA", "UMD"],
}
IS_START, IS_END = "1963-07", "2013-12"     # Fama & French (2015) sample
HO_START, HO_END = "2014-01", "2026-09"     # data that did not exist when the model was chosen
STRESS = {"GFC 2007-09": ("2007-10", "2009-03"), "COVID 2020": ("2020-02", "2020-04"),
          "Inflation 2022": ("2022-01", "2022-10")}
TEST_ASSETS = {"25 Size-B/M": "25_Portfolios_5x5.csv", "25 Size-OP": "25_Portfolios_ME_OP_5x5.csv",
               "25 Size-Inv": "25_Portfolios_ME_INV_5x5.csv", "10 Momentum": "10_Portfolios_Prior_12_2.csv",
               "49 Industries": "49_Industry_Portfolios.csv"}
PRIMARY_ASSETS = ["25 Size-B/M", "25 Size-OP", "25 Size-Inv"]


def read_raw(name: str) -> pd.DataFrame:
    lines = [ln for ln in (RAW / name).read_text().splitlines() if ln and not ln.startswith("#")]
    rows = [ln.split(",") for ln in lines]
    df = pd.DataFrame(rows[1:], columns=rows[0]).set_index("date")
    df.index = pd.PeriodIndex(pd.to_datetime(df.index, format="%Y%m"), freq="M")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.mask(df <= -99.99)
    return df / 100.0


def vintage(name: str) -> str:
    first = (RAW / name).read_text().splitlines()[0]
    return first[first.find("using the ") + 10:][:6] if "using the " in first else "unknown"


def factors() -> pd.DataFrame:
    f5 = read_raw("F-F_Research_Data_5_Factors_2x3.csv").rename(columns={"Mkt-RF": "Mkt"})
    mom = read_raw("F-F_Momentum_Factor.csv")
    mom.columns = ["UMD"]
    out = f5.join(mom, how="inner")
    return out[FACTORS + ["RF"]].dropna()


def test_assets(family: str) -> pd.DataFrame:
    """Excess returns of one test-asset family, aligned with the factor sample."""
    f = factors()
    p = read_raw(TEST_ASSETS[family]).loc[f.index[0]:f.index[-1]]
    return p.sub(f.loc[p.index, "RF"], axis=0)


def window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[pd.Period(start, "M"):pd.Period(end, "M")]


def vintage_2017() -> pd.DataFrame | None:
    """Mkt, SMB, HML, UMD as distributed in linearmodels (an early-2017 vintage), or None if absent."""
    try:
        from linearmodels.datasets import french
    except ImportError:
        return None
    d = french.load()
    d.index = pd.PeriodIndex(pd.to_datetime(d["dates"]), freq="M")
    return d.rename(columns={"MktRF": "Mkt", "Mom": "UMD"})[["Mkt", "SMB", "HML", "UMD", "RF"]]


if __name__ == "__main__":
    f = factors()
    print("factors", f.index[0], f.index[-1], f.shape, "vintage", vintage("F-F_Research_Data_5_Factors_2x3.csv"))
    for k in TEST_ASSETS:
        t = test_assets(k)
        print(k, t.shape, int(t.isna().sum().sum()), "missing")
