"""
Walk-forward backtest: at every month-end, every estimator forecasts every industry's beta using only
data up to that day; the forecast is then scored on the *following* month (daily hedge) and the
following twelve months (monthly hedge).

Output: results/forecast_panel.csv.gz, one row per (forecast month, industry) with
  beta_<estimator>             ex-ante beta
  d_n, d_sxx, d_sxy, d_syy     next month's daily sums of x^2, x*y, y^2 (excess returns, decimals)
  m_sxx, m_sxy, m_syy, m_n     the same over the next 12 monthly excess returns
  r_next, mkt_next             next month's excess return of the industry and of the market

The out-of-sample hedge loss of any beta b is then exact and cheap:
    L(b) = (syy - 2 b sxy + b^2 sxx) / n
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data import load_panel
from estimators import ESTIMATORS, BASELINES, all_betas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "forecast_panel.csv.gz"
MIN_DAYS_EVAL = 15


def run(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    p = load_panel() if panel is None else panel
    assets = [c for c in p.columns if c not in ("MKT", "RF")]
    Y = p[assets].to_numpy(float)
    x = p["MKT"].to_numpy(float)
    rf = p["RF"].to_numpy(float)
    per = p.index.to_period("M")
    months = per.unique()
    month_id = np.searchsorted(months.asi8, per.asi8)
    E = np.array([np.flatnonzero(month_id == k)[-1] for k in range(len(months))])

    betas, Ym, xm = all_betas(Y, x, rf, month_id, E)
    K, N = len(months), len(assets)

    # next-month daily sums
    d = {k: np.full((K, N), np.nan) for k in ("n", "sxx", "sxy", "syy")}
    for k in range(K - 1):
        sl = month_id == k + 1
        y, xx = Y[sl], x[sl]
        v = ~np.isnan(y)
        n = v.sum(0)
        yz = np.where(v, y, 0.0)
        X = np.where(v, xx[:, None], 0.0)
        d["n"][k] = n
        d["sxx"][k] = (X * X).sum(0)
        d["sxy"][k] = (X * yz).sum(0)
        d["syy"][k] = (yz * yz).sum(0)
        d["n"][k][n < MIN_DAYS_EVAL] = np.nan

    # next-12-month monthly sums (all 12 months required)
    m = {k: np.full((K, N), np.nan) for k in ("n", "sxx", "sxy", "syy")}
    for k in range(K - 12):
        y, xx = Ym[k + 1:k + 13], xm[k + 1:k + 13]
        ok = ~np.isnan(y).any(0)
        m["n"][k] = np.where(ok, 12.0, np.nan)
        m["sxx"][k] = (xx ** 2).sum() * np.ones(N)
        m["sxy"][k] = np.nansum(xx[:, None] * y, 0)
        m["syy"][k] = np.nansum(y * y, 0)

    r_next = np.vstack([Ym[1:], np.full((1, N), np.nan)])
    mkt_next = np.append(xm[1:], np.nan)

    idx = pd.MultiIndex.from_product([months.astype(str), assets], names=["month", "asset"])
    cols = {f"beta_{k}": betas[k].ravel() for k in ESTIMATORS + BASELINES}
    cols.update({f"d_{k}": v.ravel() for k, v in d.items()})
    cols.update({f"m_{k}": v.ravel() for k, v in m.items()})
    cols["r_next"] = r_next.ravel()
    cols["mkt_next"] = np.repeat(mkt_next, N)
    df = pd.DataFrame(cols, index=idx)
    return df


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = run()
    df.to_csv(OUT, float_format="%.6g")
    print(df.shape, f"{time.time() - t0:.0f}s")
    print(df.filter(like="beta_").describe().T.round(3).to_string())
