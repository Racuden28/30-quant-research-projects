"""
Sensitivity and ablation: every pre-registered parameter moved one notch either way, and the two
ingredients of Welch's estimator (winsorizing and age-weighting) switched off one at a time.
Nothing here was used to choose the main specification; it reports how much the conclusions move.

Output: results/robustness.csv  (excess-loss ratio vs the textbook beta, both horizons)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import estimators as est
from data import load_panel
from evaluate import common_sample, load

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def excess_ratio(df, col, horizon):
    cs = df.dropna(subset=[col, "beta_OLS-5Y-M", f"{horizon}_n"])
    n, sxx, sxy, syy = (cs[f"{horizon}_{k}"].to_numpy() for k in ("n", "sxx", "sxy", "syy"))
    bstar = sxy / sxx
    def ex(b):
        return pd.Series((b - bstar) ** 2 * sxx / n, index=cs.index).groupby(cs["eval_month"].to_numpy()).mean().sum()
    return ex(cs[col].to_numpy()) / ex(cs["beta_OLS-5Y-M"].to_numpy())


def main():
    p = load_panel()
    assets = [c for c in p.columns if c not in ("MKT", "RF")]
    Y, x, rf = p[assets].to_numpy(float), p["MKT"].to_numpy(float), p["RF"].to_numpy(float)
    per = p.index.to_period("M")
    months = per.unique()
    mid = np.searchsorted(months.asi8, per.asi8)
    E = np.array([np.flatnonzero(mid == k)[-1] for k in range(len(months))])
    Ym, xm = est.monthly_excess(Y, x, rf, mid)

    variants = {
        "OLS-1Y-D (main, 252d)": est.ols_daily(Y, x, E),
        "OLS daily 126d": est.ols_daily(Y, x, E, 126, 100),
        "OLS daily 504d": est.ols_daily(Y, x, E, 504, 400),
        "OLS daily 1260d": est.ols_daily(Y, x, E, 1260, 1000),
        "OLS monthly 36m": est.ols_monthly(Ym, xm, 36, 30),
        "OLS monthly 120m": est.ols_monthly(Ym, xm, 120, 72),
        "EWMA hl 63d": est.ewma(Y, x, E, 63),
        "EWMA (main, hl 126d)": est.ewma(Y, x, E, 126),
        "EWMA hl 252d": est.ewma(Y, x, E, 252),
        "Dimson 1 lag": est.dimson(Y, x, E, 1),
        "Dimson (main, 2 lags)": est.dimson(Y, x, E, 2),
        "Dimson 5 lags": est.dimson(Y, x, E, 5),
        "Welch-SWA (main)": est.welch_swa(Y, x, E),
        "Welch: age-weighting only": est.welch_swa(Y, x, E, bounds=(-1e9, 1e9)),
        "Welch: winsorizing only": est.welch_swa(Y, x, E, rho=0.0),
        "Welch: neither (OLS 756d)": est.welch_swa(Y, x, E, bounds=(-1e9, 1e9), rho=0.0),
        "Welch bounds (-1, 3)": est.welch_swa(Y, x, E, bounds=(-1.0, 3.0)),
        "Welch bounds (-3, 5)": est.welch_swa(Y, x, E, bounds=(-3.0, 5.0)),
    }
    b1y, se2 = est.ols_daily(Y, x, E, return_se=True)
    variants["Vasicek (main)"] = est.vasicek(b1y, se2)
    variants["Blume on monthly 5y beta"] = est.blume(est.ols_monthly(Ym, xm))
    variants["Vasicek on monthly 5y beta"] = est.vasicek(*est.ols_daily(Ym, xm, np.arange(len(xm)), 60, 36, return_se=True))
    variants["Dimson-adjusted 1y + Vasicek"] = est.vasicek(est.dimson(Y, x, E), se2)

    df = load()
    cs = common_sample(df, "d")
    keep = cs.index
    rows = []
    for name, B in variants.items():
        col = "beta_var"
        df[col] = B.ravel()
        sub = df.loc[keep]
        rows.append(dict(variant=name, ratio_next_month_daily=excess_ratio(sub, col, "d"),
                         ratio_next_year_monthly=excess_ratio(sub, col, "m")))
    out = pd.DataFrame(rows).set_index("variant")
    out.to_csv(RES / "robustness.csv", float_format="%.4f")
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
