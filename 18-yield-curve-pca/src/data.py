"""Load and clean the three data sets used in project 18.

* FRED constant-maturity Treasury yields (DGS1MO ... DGS30), daily 1962-2026, percent.
* Federal Reserve Guerkaynak-Sack-Wright (GSW) Nelson-Siegel-Svensson parameters, daily 1961-2026.
  Zero-coupon yields at any maturity are rebuilt from the six parameters.
* Six iShares Treasury ETFs (SHY, IEI, IEF, TLH, TLT, GOVT), daily adjusted closes from Yahoo.

All three files were downloaded on 2026-11-22 in a browser (the cloud container cannot reach FRED,
the Fed or Yahoo) and are committed in data/. Running this module prints a short audit and writes
results/data_audit.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"

FRED_COLS = {"DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1, "DGS2": 2, "DGS3": 3,
             "DGS5": 5, "DGS7": 7, "DGS10": 10, "DGS20": 20, "DGS30": 30}
LABEL = {1 / 12: "1M", 0.25: "3M", 0.5: "6M", 1: "1Y", 2: "2Y", 3: "3Y", 5: "5Y", 7: "7Y",
         10: "10Y", 20: "20Y", 30: "30Y"}

# Two panels of the CMT coupon curve (bills are excluded: see README, Limitations).
# Panel A is long (50 years) and narrow, panel B is wide and short.
PANELS = {
    "A": {"mats": ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y"], "start": "1976-06-01",
          "desc": "CMT 1Y-10Y, 1976-2026"},
    "B": {"mats": ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"],
          "start": "2006-02-09", "desc": "CMT 1Y-30Y, 2006-2026"},
}


def cmt() -> pd.DataFrame:
    """All CMT yields in percent, columns labelled 1M ... 30Y, index = date."""
    df = pd.read_csv(DATA / "fred_cmt_daily.csv", parse_dates=["observation_date"], index_col=0)
    df = df.rename(columns={c: LABEL[m] for c, m in FRED_COLS.items()})
    return df.astype(float)


def panel(name: str) -> pd.DataFrame:
    """Yield levels (percent) for one panel, keeping only days where every maturity is quoted."""
    spec = PANELS[name]
    df = cmt().loc[spec["start"]:, spec["mats"]].dropna(how="any")
    return df


# Extra experiment panels: the same hedging study on the GSW fitted zero curve (G) and on the
# total returns of traded Treasury ETFs (E). See experiment_data().
GSW_MATS = [1, 2, 3, 5, 7, 10]
ETF_MAT = {"SHY": 2, "IEI": 5, "IEF": 8, "TLH": 15, "TLT": 25}  # nominal maturity bucket, for ordering
ALL_PANELS = ["A", "B", "G", "E"]
PANEL_DESC = {"A": "CMT 1Y-10Y, 1976-2026", "B": "CMT 1Y-30Y, 2006-2026",
              "G": "GSW fitted zero curve 1Y-10Y, 1976-2026", "E": "Treasury ETF returns, 2007-2026"}


def experiment_data(name: str):
    """(changes DataFrame, maturities in years, 2Y CMT level at the previous close).

    For A, B and G the changes are yield changes in bp. For E they are *minus* the ETF log returns
    in bp, so that a long position has the same sign convention as a long bond (P&L = -exposure x dy).
    """
    if name in PANELS:
        lv = panel(name)
        return changes(lv), years(lv.columns), lv["2Y"].shift(1).iloc[1:]
    two = cmt()["2Y"].dropna()
    if name == "G":
        lv = gsw(GSW_MATS).loc["1976-06-01":].dropna()
        lv.columns = [f"{m}Y" for m in GSW_MATS]
        dy = changes(lv)
    elif name == "E":
        dy = -etf_returns()
    else:
        raise ValueError(name)
    lvl = two.reindex(dy.index, method="ffill")
    m = np.array(GSW_MATS, float) if name == "G" else np.array([ETF_MAT[c] for c in dy.columns], float)
    return dy, m, lvl


def changes(levels: pd.DataFrame) -> pd.DataFrame:
    """Daily yield changes in basis points between consecutive quoted days."""
    return (levels.diff() * 100).iloc[1:]


def years(labels) -> np.ndarray:
    inv = {v: k for k, v in LABEL.items()}
    return np.array([inv[x] for x in labels], dtype=float)


# ---------------------------------------------------------------- GSW (fitted) curve
def nss_yield(n, b0, b1, b2, b3, t1, t2):
    """Nelson-Siegel-Svensson continuously compounded zero yield (percent) at maturity n (years).

    GSW report TAU2 = -999.99 and BETA3 = 0 on days where only the four-parameter Nelson-Siegel
    curve was fitted; the Svensson term is then dropped.
    """
    n = np.asarray(n, dtype=float)
    x1 = n / t1
    f1 = (1 - np.exp(-x1)) / x1
    y = b0 + b1 * f1 + b2 * (f1 - np.exp(-x1))
    if np.isfinite(t2) and t2 > 0 and b3 != 0:
        x2 = n / t2
        y = y + b3 * ((1 - np.exp(-x2)) / x2 - np.exp(-x2))
    return y


def gsw(mats_years) -> pd.DataFrame:
    """GSW zero yields (percent) at the given maturities, rebuilt from the daily NSS parameters."""
    p = pd.read_csv(DATA / "gsw_nss_params_daily.csv", parse_dates=["Date"], index_col=0)
    p = p.dropna(subset=["BETA0", "BETA1", "BETA2", "TAU1"])
    p["BETA3"] = p["BETA3"].fillna(0.0)
    out = np.empty((len(p), len(mats_years)))
    for i, r in enumerate(p[["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]].to_numpy()):
        out[i] = nss_yield(mats_years, *r)
    return pd.DataFrame(out, index=p.index, columns=[f"{m:g}Y" for m in mats_years])


def gsw_check() -> float:
    """Max abs difference (percent) between rebuilt yields and the published SVENY columns."""
    p = pd.read_csv(DATA / "gsw_nss_params_daily.csv", parse_dates=["Date"], index_col=0)
    g = gsw([2, 5, 10, 30])
    pub = p.loc[g.index, ["SVENY02", "SVENY05", "SVENY10", "SVENY30"]].to_numpy()
    return float(np.nanmax(np.abs(g.to_numpy() - pub)))


# ---------------------------------------------------------------- ETFs
ETFS = ["SHY", "IEI", "IEF", "TLH", "TLT"]


def etf_returns() -> pd.DataFrame:
    """Daily total log returns (dividends reinvested), in basis points, common sample."""
    df = pd.read_csv(DATA / "treasury_etfs_daily.csv.gz", parse_dates=["date"])
    px = df.pivot(index="date", columns="sym", values="adjclose")[ETFS].dropna()
    return (np.log(px).diff() * 1e4).iloc[1:]


def audit() -> pd.DataFrame:
    rows = []
    for name in PANELS:
        lv = panel(name)
        dy = changes(lv)
        rows.append({"dataset": f"panel {name}: {PANELS[name]['desc']}", "first": lv.index[0].date(),
                     "last": lv.index[-1].date(), "days": len(dy),
                     "max_abs_change_bp": float(dy.abs().max().max()),
                     "zero_change_share": float((dy == 0).mean().mean())})
    g = gsw([1, 2, 5, 10])
    rows.append({"dataset": "GSW zero curve (rebuilt from NSS parameters)", "first": g.index[0].date(),
                 "last": g.index[-1].date(), "days": len(g) - 1,
                 "max_abs_change_bp": float((g.diff() * 100).abs().max().max()),
                 "zero_change_share": float(((g.diff() * 100) == 0).mean().mean())})
    e = etf_returns()
    rows.append({"dataset": "Treasury ETFs " + "/".join(ETFS), "first": e.index[0].date(),
                 "last": e.index[-1].date(), "days": len(e),
                 "max_abs_change_bp": float(e.abs().max().max()),
                 "zero_change_share": float((e == 0).mean().mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    a = audit()
    a.to_csv(RES / "data_audit.csv", index=False)
    print(a.to_string(index=False))
    print(f"GSW rebuild vs published SVENY, max abs diff: {gsw_check():.2e} percent")
