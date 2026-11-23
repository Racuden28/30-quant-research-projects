"""
Build the two daily total-return panels used by every other script.

Inputs (data/raw/):
  ff49_panel_daily.csv.gz   49 Ken French industries, daily *excess* returns + RF, 1926-07 to 2026-09
                            (built by project 09 from the vintage-202609 zips; copied unchanged)
  yahoo_adjclose.csv.gz     Yahoo adjusted closes (dividends and splits applied), downloaded 2026-11-22
  yahoo_volume.csv.gz       Yahoo share volume (used only for the data-quality table)
  49_Industry_Portfolios_CSV.zip  monthly French file, for the number of firms per industry

Outputs (data/):
  returns_industries.csv.gz  daily simple total returns, 49 columns
  returns_countries.csv.gz   daily simple total returns, 22 columns (SPY + 21 iShares single-country ETFs)
  industries_nfirms_monthly.csv  number of firms per industry and month
and results/data_quality.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, DATA, RES = ROOT / "data" / "raw", ROOT / "data", ROOT / "results"

COUNTRIES = ["SPY", "EWA", "EWC", "EWD", "EWG", "EWH", "EWI", "EWJ", "EWK", "EWL", "EWM", "EWN", "EWO",
             "EWP", "EWQ", "EWS", "EWU", "EWW", "EWY", "EWT", "EWZ", "EZA"]
COUNTRY_NAME = {"SPY": "United States", "EWA": "Australia", "EWC": "Canada", "EWD": "Sweden", "EWG": "Germany",
                "EWH": "Hong Kong", "EWI": "Italy", "EWJ": "Japan", "EWK": "Belgium", "EWL": "Switzerland",
                "EWM": "Malaysia", "EWN": "Netherlands", "EWO": "Austria", "EWP": "Spain", "EWQ": "France",
                "EWS": "Singapore", "EWU": "United Kingdom", "EWW": "Mexico", "EWY": "South Korea",
                "EWT": "Taiwan", "EWZ": "Brazil", "EZA": "South Africa"}


def industry_nfirms() -> pd.DataFrame:
    """Monthly number of firms per industry (French's 'Number of Firms in Portfolios' table)."""
    import zipfile
    with zipfile.ZipFile(RAW / "49_Industry_Portfolios_CSV.zip") as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    start = next(i for i, l in enumerate(lines) if "Number of Firms in Portfolios" in l)
    head = next(i for i in range(start, len(lines)) if lines[i].startswith(","))
    cols = [c.strip() for c in lines[head].split(",")[1:]]
    idx, rows = [], []
    for l in lines[head + 1:]:
        parts = [x.strip() for x in l.split(",")]
        if len(parts[0]) != 6 or not parts[0].isdigit():
            break
        idx.append(pd.Timestamp(parts[0][:4] + "-" + parts[0][4:] + "-01"))
        rows.append([float(x) for x in parts[1:]])
    return pd.DataFrame(rows, index=idx, columns=cols)


def build_industries() -> pd.DataFrame:
    p = pd.read_csv(RAW / "ff49_panel_daily.csv.gz", index_col=0, parse_dates=True)
    rf = p.pop("RF")
    p = p.drop(columns="MKT")
    return p.add(rf, axis=0)  # total return = excess + RF (NaN where the industry did not exist)


def build_countries() -> tuple[pd.DataFrame, pd.DataFrame]:
    px = pd.read_csv(RAW / "yahoo_adjclose.csv.gz", index_col=0, parse_dates=True)[COUNTRIES]
    vol = pd.read_csv(RAW / "yahoo_volume.csv.gz", index_col=0, parse_dates=True)[COUNTRIES]
    px = px.loc["1996-03-18":]
    vol = vol.loc[px.index]
    r = px.pct_change(fill_method=None)
    r.iloc[0] = np.nan
    return r, vol


def quality(r: pd.DataFrame, vol: pd.DataFrame | None, name: str) -> pd.DataFrame:
    rows = []
    for c in r.columns:
        x = r[c].dropna()
        row = {"universe": name, "asset": c, "first": x.index.min().date(), "last": x.index.max().date(),
               "days": len(x), "zero_return_share": float((x == 0).mean()),
               "zero_return_share_pre2005": float((x.loc[:"2004"] == 0).mean()) if len(x.loc[:"2004"]) else np.nan,
               "max_abs_return": float(x.abs().max()), "date_max_abs": x.abs().idxmax().date(),
               "ann_vol": float(x.std() * np.sqrt(252))}
        if vol is not None:
            v = vol[c].reindex(x.index)
            row["zero_volume_share"] = float((v == 0).mean())
            row["median_volume_2025"] = float(v.loc["2025"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def load(universe: str) -> pd.DataFrame:
    f = {"industries": "returns_industries.csv.gz", "countries": "returns_countries.csv.gz"}[universe]
    return pd.read_csv(DATA / f, index_col=0, parse_dates=True)


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    ind = build_industries()
    cty, vol = build_countries()
    ind.to_csv(DATA / "returns_industries.csv.gz", float_format="%.7g")
    industry_nfirms().to_csv(DATA / "industries_nfirms_monthly.csv")
    cty.to_csv(DATA / "returns_countries.csv.gz", float_format="%.7g")
    q = pd.concat([quality(ind, None, "industries"), quality(cty, vol, "countries")])
    q.to_csv(RES / "data_quality.csv", index=False, float_format="%.5g")
    print(ind.shape, ind.index.min().date(), ind.index.max().date())
    print(cty.shape, cty.index.min().date(), cty.index.max().date())
    print(q[q.universe == "countries"].to_string())
    print(q[q.universe == "industries"].sort_values("max_abs_return").tail(5).to_string())
