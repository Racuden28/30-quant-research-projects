"""Build the three research universes and save them as CSV (the CSVs are committed).

Universe A  12 Fama-French industry portfolios (value-weighted), monthly excess returns, 1949-01 to 2026-09.
Universe C  49 Fama-French industry portfolios (value-weighted), monthly excess returns, 1969-07 to 2026-09
            (1969-07 is the first month in which all 49 industries have returns).
            Source for A and C: Ken French data library, files 12_Industry_Portfolios_CSV.zip,
            49_Industry_Portfolios_CSV.zip and F-F_Research_Data_Factors_CSV.zip (CRSP vintage 202609),
            first section ("Average Value Weighted Returns -- Monthly"). They were downloaded on 2026-11-21
            with src/fetch_kf.js and are stored unmodified (as decimals) in data/raw/.
Universe B  19 S&P 500 constituents, daily simple returns, 1990-01-03 to 2022-12-28.
            Source: `skfolio.datasets.load_sp500_dataset` (adjusted closes, rounded to 3 decimals).
            RRC (Range Resources) is dropped by a rule fixed before any backtest: an asset is
            excluded if more than 30% of its daily returns are exactly zero in any calendar year
            (stale prices). RRC has 92% zero-return days in 1990 and 70% in its worst year after 1991.
            The next worst name is AMD at 27% in 1990, which is normal for the 1/8-dollar tick era.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"

IND_FILE = DATA / "industries12_monthly_excess.csv"
STK_FILE = DATA / "sp500_19_daily_returns.csv"
IND49_FILE = DATA / "industries49_monthly_excess.csv"


def _kf(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / name, index_col=0, parse_dates=True)


def build() -> None:
    DATA.mkdir(exist_ok=True)
    rf = _kf("kf_rf_monthly.csv")["RF"]
    for raw, out, start in [("kf_12_industries_vw_monthly.csv", IND_FILE, "1949-01"),
                            ("kf_49_industries_vw_monthly.csv", IND49_FILE, "1969-07")]:
        r = _kf(raw).loc[start:]
        assert (r > -0.99).all().all(), "missing-value codes (-99.99) present"
        ex = r.sub(rf.reindex(r.index), axis=0)
        ex.index.name = "date"
        ex.round(6).to_csv(out)
        print(f"{out.name}: {ex.shape} {ex.index[0]:%Y-%m}..{ex.index[-1]:%Y-%m}")

    from skfolio.datasets import load_sp500_dataset
    px = load_sp500_dataset()
    r = px.pct_change().iloc[1:]
    zero_share = (r == 0).groupby(r.index.year).mean()
    stale = [c for c in r.columns if (zero_share[c] > 0.30).any()]
    assert stale == ["RRC"], stale
    r = r.drop(columns=stale)
    r.index.name = "date"
    r.round(8).to_csv(STK_FILE)
    print(f"stocks {r.shape} {r.index[0]:%Y-%m-%d}..{r.index[-1]:%Y-%m-%d}; dropped {stale}")


def load(universe: str) -> pd.DataFrame:
    f = {"A": IND_FILE, "B": STK_FILE, "C": IND49_FILE}[universe]
    return pd.read_csv(f, index_col=0, parse_dates=True)


if __name__ == "__main__":
    build()
