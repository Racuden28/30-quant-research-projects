"""
Out-of-sample hedging horse race: Kalman hedge ratios against static, rolling and exponentially weighted ones.

For every hedged asset y and the hedge instrument x (the market for industries, SPY for country ETFs):
  * the Kalman filter is run for every signal-to-noise ratio r on the grid; every January the r with the highest
    profile likelihood on all earlier data is chosen, and that filter's predicted beta is used for the year;
  * the matched EWMA forgets at the Kalman steady-state rate implied by the same r_hat and the same past data,
    on three clocks: calendar days, cumulative market variance, cumulative absolute market return;
  * benchmarks: EWMA with project 09's 126-day half-life, rolling 252-day OLS (project 09's OLS-1Y-D), rolling
    63-day OLS, expanding OLS through the origin (the static hedge ratio), and a heteroskedastic Kalman filter.

Every beta is the one known before the day it is used (protocol D). Protocol M holds the beta known at a
month-end for the whole next month (project 09's protocol).

Output, per universe:
  results/hedge_<u>.npz        asset-month sums: n, sxx, sxy, syy, the sum of squared hedge errors per estimator
                               and protocol, beta turnover, and the tuning-curve losses per grid value
  results/hedge_rhat_<u>.csv   the chosen r_hat, its implied half-life and s2 at every re-estimation date
"""
from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from data import RES, hedging_panel
from kalman import (R_GRID, ew_beta, kf_scalar_grid, kf_scalar_hr_grid, matched_halflife, matched_lambda,
                    rolling_ols, steady_state_pm)

MIN_FIRST = 504            # valid days before the first estimate
EWMA_HL = 126              # project 09's EWMA half-life
EST = ["KF", "EW-match", "EW-var", "EW-vol", "EWMA-126", "OLS-252", "OLS-63", "Static", "KF-HR", "KF-SSE", "EW-SSE"]


def reestimation_index(dates: pd.DatetimeIndex) -> np.ndarray:
    """Position of the first trading day of every calendar year (the day r is re-chosen)."""
    y = dates.year
    return np.nonzero(np.r_[True, y[1:] != y[:-1]])[0]


def choose(LL: np.ndarray, starts: np.ndarray, nvalid: np.ndarray, T: int) -> np.ndarray:
    """For each day, the grid index chosen at the latest re-estimation date (or -1 before enough data)."""
    g = np.full(T, -1)
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else T
        if nvalid[s] < MIN_FIRST or np.all(np.isnan(LL[s])):
            continue
        g[s:e] = int(np.nanargmax(LL[s]))
    return g


def choose_sse(Bgrid: np.ndarray, y, x, starts, nvalid, T) -> np.ndarray:
    """Like choose(), but the grid value with the smallest past sum of squared hedge errors (the hedging loss
    itself, unweighted) instead of the highest likelihood. Uses only days before each re-estimation date."""
    e2 = (y[:, None] - Bgrid * x[:, None]) ** 2
    e2 = np.where(np.isnan(e2), 0.0, e2)
    c = np.vstack([np.zeros((1, Bgrid.shape[1])), np.cumsum(e2, 0)])[:T]   # sums over days < t
    g = np.full(T, -1)
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else T
        if nvalid[s] < MIN_FIRST:
            continue
        g[s:e] = int(np.argmin(c[s]))
    return g


def one_asset(args):
    y, x, dates, month_id, K, first = args
    T = len(y)
    valid = ~np.isnan(y) & ~np.isnan(x)
    nvalid = np.r_[0, np.cumsum(valid)][:T]                    # valid days strictly before t
    starts = reestimation_index(dates)
    x2 = np.where(valid, x * x, 0.0)
    ax = np.where(valid, np.abs(x), 0.0)
    cx2 = np.r_[0, np.cumsum(x2)][:T]
    cax = np.r_[0, np.cumsum(ax)][:T]

    B, LL, PM = kf_scalar_grid(y, x, R_GRID)
    gsel = choose(LL, starts, nvalid, T)
    ok = gsel >= 0
    out = {k: np.full(T, np.nan) for k in EST}
    out["KF"][ok] = B[np.nonzero(ok)[0], gsel[ok]]

    # matched EWMAs: per-day log decay that equals the Kalman steady state on average
    s2 = np.where(nvalid > 0, cx2 / np.maximum(nvalid, 1), np.nan)
    mabs = np.where(nvalid > 0, cax / np.maximum(nvalid, 1), np.nan)
    rhat = np.where(ok, R_GRID[np.maximum(gsel, 0)], np.nan)
    # s2 and mean |x| are frozen at the re-estimation date, like r_hat
    s2y, may = np.full(T, np.nan), np.full(T, np.nan)
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else T
        s2y[s:e], may[s:e] = s2[s], mabs[s]
    with np.errstate(invalid="ignore", divide="ignore"):
        loglam = np.log(matched_lambda(np.nan_to_num(rhat), np.nan_to_num(s2y, nan=1.0)))   # = -log(1 + a)
        a = -loglam
        dec_cal = np.exp(loglam)
        dec_var = np.exp(-a * x2 / s2y)
        dec_vol = np.exp(-a * ax / may)
    # before the first estimate, the EWMAs simply accumulate (decay 1); their output is masked anyway
    for name, d in (("EW-match", dec_cal), ("EW-var", dec_var), ("EW-vol", dec_vol)):
        d = np.where(ok, d, 1.0)
        bb = ew_beta(y, x, np.ascontiguousarray(d))
        out[name][ok] = bb[ok]
    lam126 = 0.5 ** (1 / EWMA_HL)
    out["EWMA-126"] = ew_beta(y, x, np.full(T, lam126))
    out["OLS-252"] = rolling_ols(y, x, 252, 200)
    out["OLS-63"] = rolling_ols(y, x, 63, 50)
    out["Static"] = ew_beta(y, x, np.ones(T))
    Bh, LLh = kf_scalar_hr_grid(y, x, R_GRID)
    gh = choose(LLh, starts, nvalid, T)
    okh = gh >= 0
    out["KF-HR"][okh] = Bh[np.nonzero(okh)[0], gh[okh]]
    # tuning curve: every fixed r for the Kalman filter and for its calendar-matched EWMA (s2 frozen yearly)
    curve_kf = B
    curve_ew = np.full((T, len(R_GRID)), np.nan)
    for g, r in enumerate(R_GRID):
        lam = matched_lambda(r, np.nan_to_num(s2y, nan=1.0))
        curve_ew[:, g] = ew_beta(y, x, np.ascontiguousarray(np.where(np.isnan(s2y), 1.0, lam)))
    # the same two families with the memory chosen by past hedging loss instead of likelihood
    gk = choose_sse(curve_kf, y, x, starts, nvalid, T)
    gk_ok = gk >= 0
    out["KF-SSE"][gk_ok] = curve_kf[np.nonzero(gk_ok)[0], gk[gk_ok]]
    ge = choose_sse(curve_ew, y, x, starts, nvalid, T)
    ge_ok = ge >= 0
    out["EW-SSE"][ge_ok] = curve_ew[np.nonzero(ge_ok)[0], ge[ge_ok]]
    rows = []
    for s in starts:
        if ok[s] and gk_ok[s]:
            rows.append((dates[s], R_GRID[gsel[s]], float(matched_halflife(R_GRID[gsel[s]], s2[s])), s2[s],
                         R_GRID[gk[s]], float(matched_halflife(R_GRID[gk[s]], s2[s])),
                         R_GRID[ge[s]], float(matched_halflife(R_GRID[ge[s]], s2[s])),
                         R_GRID[gh[s]] if okh[s] else np.nan))
    for k in EST:                                            # common availability rule
        out[k][nvalid < MIN_FIRST] = np.nan

    sums = month_sums(y, x, out, curve_kf, curve_ew, month_id, K, first)
    return sums, rows, out


def month_sums(y, x, betas: dict, curve_kf, curve_ew, month_id, K, first_day_of_month):
    """Aggregate squared hedge errors to (asset-)month level for both protocols."""
    valid = ~np.isnan(y) & ~np.isnan(x)
    res = {}
    mid = month_id[valid]
    yy, xx = y[valid], x[valid]

    def agg(v):
        return np.bincount(mid, weights=v, minlength=K)

    res["n"] = agg(np.ones_like(yy))
    res["sxx"], res["sxy"], res["syy"] = agg(xx * xx), agg(xx * yy), agg(yy * yy)
    for k, b in betas.items():
        bd = b[valid]
        e = yy - bd * xx
        res[f"D|{k}"] = agg(np.where(np.isnan(e), np.nan, e * e))        # NaN propagates: month incomplete
        bm = b[first_day_of_month][month_id][valid]                      # protocol M: beta of the month's first day
        em = yy - bm * xx
        res[f"M|{k}"] = agg(em * em)
        db = np.abs(np.diff(np.r_[b[0], b]))[valid]
        res[f"TO|{k}"] = agg(np.where(np.isnan(db), 0.0, db))
    res["curve_kf"] = np.stack([agg((yy - curve_kf[valid, g] * xx) ** 2) for g in range(curve_kf.shape[1])], 1)
    res["curve_ew"] = np.stack([agg((yy - curve_ew[valid, g] * xx) ** 2) for g in range(curve_ew.shape[1])], 1)
    return res


def run(universe: str, procs: int = 4):
    Y, x, eval_start = hedging_panel(universe)
    dates = Y.index
    per = dates.to_period("M")
    months = per.unique()
    month_id = np.searchsorted(months.asi8, per.asi8)
    K = len(months)
    first = np.array([np.flatnonzero(month_id == k)[0] for k in range(K)])
    xv = x.to_numpy(float)
    assets = list(Y.columns)
    jobs = [(np.ascontiguousarray(Y[a].to_numpy(float)), xv, dates, month_id, K, first) for a in assets]
    with Pool(procs) as pool:
        outs = pool.map(one_asset, jobs, chunksize=1)
    arrays = {"months": months.astype(str).to_numpy(), "assets": np.array(assets), "rgrid": R_GRID,
              "eval_start": str(eval_start.to_period("M"))}
    rrows = []
    per_asset = []
    for a, (sums, rows, _) in zip(assets, outs):
        per_asset.append(sums)
        rrows += [(a, *r) for r in rows]
    for key in per_asset[0]:
        arrays[key] = np.stack([pa[key] for pa in per_asset], 1)   # (K, N) or (K, N, G)
    np.savez_compressed(RES / f"hedge_{universe}.npz", **arrays)
    pd.DataFrame(rrows, columns=["asset", "date", "r_hat", "halflife_days", "s2", "r_sse", "halflife_sse",
                                    "r_ew_sse", "halflife_ew_sse", "r_hr"]).to_csv(
        RES / f"hedge_rhat_{universe}.csv", index=False, float_format="%.6g")
    # daily betas of a few assets for the illustration figure
    keep = {"industries": ["Banks", "Steel", "Utils", "Softw"], "countries": ["EWJ", "EWZ", "EWG", "EWH"]}[universe]
    keep = [k for k in keep if k in assets]
    ill = {}
    for a in keep:
        b = outs[assets.index(a)][2]
        for k in ("KF", "EW-match", "EW-var", "EWMA-126", "OLS-252", "KF-HR"):
            ill[f"{a}|{k}"] = b[k]
    pd.DataFrame(ill, index=dates).iloc[::1].to_csv(RES / f"hedge_betas_{universe}.csv.gz", float_format="%.5g")


if __name__ == "__main__":
    for u in sys.argv[1:] or ["industries", "countries"]:
        run(u)
        print("done", u, flush=True)
