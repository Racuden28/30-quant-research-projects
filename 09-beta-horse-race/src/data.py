"""
Parse the Ken French Data Library files and build the daily panel used by every other script.

Inputs (data/raw/, written by fetch_data.py; CRSP vintage 202609):
  49_Industry_Portfolios_daily_CSV.zip   daily value-weighted returns of 49 industries, 1926-07 ->
  49_Industry_Portfolios_CSV.zip          monthly file: number of firms and average firm size
  F-F_Research_Data_Factors_CSV.zip       monthly Mkt-RF and RF

Output: data/panel_daily.csv.gz with, per NYSE trading day,
  * the 49 industry *excess* returns (decimal),
  * MKT, the excess return of a value-weighted market built from the 49 industries
    (weights = number of firms x average firm size at the previous month-end), and
  * RF, the daily T-bill rate that compounds to French's monthly RF within each month
    (the same convention French uses in his daily factor file).

The daily Fama-French factor file is not used directly; section "Data" of the README explains why
the industry-built market is used and how closely it tracks the official monthly Mkt (corr 0.9999).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PANEL = ROOT / "data" / "panel_daily.csv.gz"
SIZE = ROOT / "data" / "industry_avg_firm_size_monthly.csv"   # $ millions, French "Average Firm Size"


def _read_zip_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def parse_section(text: str, title: str, date_len: int) -> pd.DataFrame:
    """Return the table that follows the line containing `title` (or the first table if title='')."""
    lines = text.splitlines()
    start = 0
    if title:
        start = next(i for i, l in enumerate(lines) if title in l)
    head = next(i for i in range(start, len(lines)) if lines[i].startswith(","))
    cols = [c.strip() for c in lines[head].split(",")[1:]]
    rows, idx = [], []
    for l in lines[head + 1:]:
        s = l.strip()
        if not s or not s[:date_len].isdigit() or (len(s.split(",")[0].strip()) != date_len):
            break
        parts = s.split(",")
        idx.append(parts[0].strip())
        rows.append([float(p) for p in parts[1:]])
    df = pd.DataFrame(rows, index=idx, columns=cols)
    df.index = pd.to_datetime(df.index, format="%Y%m%d" if date_len == 8 else "%Y%m")
    return df.replace([-99.99, -999.0], np.nan)


def build_panel(raw: Path = RAW) -> pd.DataFrame:
    daily = parse_section(_read_zip_text(raw / "49_Industry_Portfolios_daily_CSV.zip"),
                          "Average Value Weighted Returns -- Daily", 8) / 100.0
    mtxt = _read_zip_text(raw / "49_Industry_Portfolios_CSV.zip")
    nfirms = parse_section(mtxt, "Number of Firms in Portfolios", 6)
    size = parse_section(mtxt, "Average Firm Size", 6)
    ff = parse_section(_read_zip_text(raw / "F-F_Research_Data_Factors_CSV.zip"), "", 6) / 100.0

    # market weights for month m = capitalisation at the end of month m-1 (French reports firm size
    # "at the end of the previous month"? -> to be safe we lag by one month, which can only remove
    # look-ahead, never add it)
    cap = (nfirms.fillna(0) * size.fillna(0)).shift(1)
    month = daily.index.to_period("M")
    w = cap.reindex(month.to_timestamp()).set_axis(daily.index)
    w = w.where(daily.notna(), 0.0)
    total_mkt = (daily.fillna(0) * w).sum(axis=1) / w.sum(axis=1)

    # daily RF: simple daily rate compounding to the monthly RF over that month's trading days
    rf_m = ff["RF"].reindex(month.to_timestamp()).set_axis(daily.index)
    ndays = pd.Series(1, index=daily.index).groupby(month).transform("size")
    rf = (1 + rf_m) ** (1 / ndays) - 1

    panel = daily.sub(rf, axis=0)
    panel["MKT"] = total_mkt - rf
    panel["RF"] = rf
    first_valid = w.sum(axis=1) > 0
    return panel[first_valid & rf.notna()]


def validate_market(panel: pd.DataFrame, raw: Path = RAW) -> pd.Series:
    """Compare the monthly-compounded industry market with French's official monthly Mkt."""
    ff = parse_section(_read_zip_text(raw / "F-F_Research_Data_Factors_CSV.zip"), "", 6) / 100.0
    tot = (panel["MKT"] + panel["RF"])
    m = (1 + tot).groupby(panel.index.to_period("M")).prod() - 1
    off = (ff["Mkt-RF"] + ff["RF"])
    off.index = off.index.to_period("M")
    j = pd.concat([m.rename("ours"), off.rename("official")], axis=1).dropna()
    d = j.ours - j.official
    return pd.Series({"months": len(j), "corr": j.ours.corr(j.official),
                      "mean_diff_bp": 1e4 * d.mean(), "mad_bp": 1e4 * d.abs().mean(),
                      "te_ann_pct": 100 * d.std() * np.sqrt(12),
                      "beta_ours_on_official": np.cov(j.ours, j.official)[0, 1] / j.official.var()})


def load_panel() -> pd.DataFrame:
    return pd.read_csv(PANEL, index_col=0, parse_dates=True)


if __name__ == "__main__":
    p = build_panel()
    p.to_csv(PANEL, float_format="%.7g")
    parse_section(_read_zip_text(RAW / "49_Industry_Portfolios_CSV.zip"), "Average Firm Size", 6).to_csv(
        SIZE, float_format="%.2f")
    v = validate_market(p)
    (ROOT / "results").mkdir(exist_ok=True)
    v.to_csv(ROOT / "results" / "market_proxy_validation.csv", header=["value"])
    print(p.shape, p.index.min().date(), p.index.max().date())
    print(v.round(5).to_string())
