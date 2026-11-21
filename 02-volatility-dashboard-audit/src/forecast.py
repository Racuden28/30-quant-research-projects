"""Build every dashboard forecast and the three benchmark models, then store the realized targets.

    python src/forecast.py

For each asset, horizon h in {5, 22, 66} and forecast origin t (every trading day), a forecast of the
h-day-ahead close-to-close variance  RV_{t,h} = sum_{k=1..h} r_{t+k}^2  is formed from information
available at the close of day t:

* dashboard grid: estimator (5) x weighting (12) x horizon scaling (2) = 120 forecasts
      F = h * sigma2_t                 ("sqrt" scaling, the sqrt-of-time rule in variance form)
      F = h * sigma2_t * VR_hat_t(h)    ("vr" scaling, trailing variance ratio, shrunk to 1)
* ablations: PK without the overnight term, GK rescaled to the C2C level (both x 12 weightings, sqrt)
* benchmarks: HAR-RV on C2C, HAR-RV on the GK+overnight proxy, GJR-GARCH(1,1)-t

Output: results/forecasts_h{h}.npz with arrays F (T x M), RV (T), dates, models; one file per asset/h
packed into a single npz per horizon.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimators import (ABLATIONS, ESTIMATORS, WEIGHTINGS, daily_proxies,  # noqa: E402
                        variance_ratio_hat, weighted_variance)

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
RES.mkdir(exist_ok=True)
ASSETS = ["SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
HORIZONS = [5, 22, 66]
REFIT = 22                     # benchmark re-estimation frequency (trading days)


def load() -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame], pd.DataFrame]:
    raw = pd.read_csv(DATA / "sector_ohlc_features.csv")
    dates = pd.to_datetime(raw["date"].astype(str), format="%Y%m%d")
    out, fixes = {}, {}
    for a in ASSETS:
        r, u, d, c = (raw[f"{a}_{k}"].to_numpy(float) / 1e5 for k in "rhlc")
        bad = (u < np.maximum(c, 0)) | (d > np.minimum(c, 0))
        fixes[a] = bad.mean()
        u, d = np.maximum.reduce([u, c, np.zeros_like(u)]), np.minimum.reduce([d, c, np.zeros_like(d)])
        out[a] = pd.DataFrame({"r": r, "o": r - c, "u": u, "d": d, "c": c}, index=dates)
    return pd.DatetimeIndex(dates), out, pd.Series(fixes, name="share_rows_fixed")


def realized(r: np.ndarray, h: int) -> np.ndarray:
    """RV_{t,h} = sum r_{t+1..t+h}^2, NaN where the window runs past the sample."""
    s = np.concatenate([[0.0], np.cumsum(r ** 2)])
    rv = np.full(len(r), np.nan)
    rv[: len(r) - h] = s[1 + h: len(r) + 1] - s[1: len(r) + 1 - h]
    return rv


def har_forecasts(proxy: np.ndarray, r2: np.ndarray, h: int, min_obs: int = 1000) -> np.ndarray:
    """HAR-RV (Corsi 2009) with daily/weekly/monthly components of ``proxy``; target mean r^2 over h days.

    Expanding-window OLS, refit every REFIT days, using only (x_s, y_s) pairs with s + h <= t.
    """
    p = pd.Series(proxy)
    X = np.column_stack([np.ones(len(p)), p, p.rolling(5).mean(), p.rolling(22).mean()])
    y = realized(np.sqrt(r2), h) / h
    f = np.full(len(p), np.nan)
    beta = None
    for t in range(len(p)):
        if (t % REFIT == 0) or beta is None:
            last = t - h                     # last origin whose target is fully observed at t
            if last < min_obs:               # (also guards against negative-index slicing)
                continue
            ok = ~np.isnan(X[: last + 1]).any(1) & ~np.isnan(y[: last + 1])
            if ok.sum() >= min_obs:
                beta = np.linalg.lstsq(X[: last + 1][ok], y[: last + 1][ok], rcond=None)[0]
        if beta is not None and not np.isnan(X[t]).any():
            f[t] = h * max(X[t] @ beta, 0.1 * X[t, 3])    # floor: 10% of the monthly component
    return f


_GJR_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def gjr_path(r: np.ndarray, key: str, window: int = 2000, min_obs: int = 1000):
    """GJR-GARCH(1,1) with Student-t shocks, refit every REFIT days on a trailing window; between
    refits the variance recursion is filtered forward with fixed parameters.

    Returns, for every origin t, the one-step-ahead variance, the persistence alpha + gamma/2 + beta
    and the long-run variance (all in daily log-return units)."""
    if key in _GJR_CACHE:
        return _GJR_CACHE[key]
    from arch import arch_model
    x = 100 * r
    nxt_a, pers_a, lr_a = (np.full(len(r), np.nan) for _ in range(3))
    params = None
    for t in range(len(r)):
        if t + 1 >= min_obs and (t % REFIT == 0 or params is None):
            lo = max(0, t + 1 - window)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = arch_model(x[lo: t + 1], mean="Constant", vol="GARCH", p=1, o=1, q=1,
                                 dist="t").fit(disp="off", show_warning=False)
            params = res.params
            mu, om, al, ga, be = (params[k] for k in ["mu", "omega", "alpha[1]", "gamma[1]", "beta[1]"])
            sig2 = float(res.conditional_volatility[-1] ** 2)
        elif params is not None:
            sig2 = nxt
        if params is None:
            continue
        eps = x[t] - mu
        nxt = om + (al + ga * (eps < 0)) * eps ** 2 + be * sig2
        pers = min(al + ga / 2 + be, 0.9999)
        nxt_a[t], pers_a[t], lr_a[t] = nxt / 1e4, pers, om / (1 - pers) / 1e4
    _GJR_CACHE[key] = (nxt_a, pers_a, lr_a)
    return _GJR_CACHE[key]


def gjr_forecasts(r: np.ndarray, h: int, key: str) -> np.ndarray:
    """h-step cumulative variance: sum_k lr + pers^k (sigma2_{t+1} - lr)."""
    nxt, pers, lr = gjr_path(r, key)
    k = np.arange(h)[None, :]
    return np.sum(lr[:, None] + pers[:, None] ** k * (nxt - lr)[:, None], axis=1)


def build_asset(df: pd.DataFrame, h: int, name: str) -> tuple[np.ndarray, list[str]]:
    px = daily_proxies(df.r, df.o, df.u, df.d, df.c)
    vr = variance_ratio_hat(df.r, h).to_numpy()
    cols, names = [], []
    for e in ESTIMATORS:
        for w in WEIGHTINGS:
            s2 = weighted_variance(px, df.o, df.c, e, w).to_numpy()
            cols += [h * s2, h * s2 * vr]
            names += [f"{e}|{w}|sqrt", f"{e}|{w}|vr"]
    for e in ABLATIONS:
        for w in WEIGHTINGS:
            cols.append(h * weighted_variance(px, df.o, df.c, e, w).to_numpy())
            names.append(f"{e}|{w}|sqrt")
    r2 = (df.r ** 2).to_numpy()
    cols.append(har_forecasts(px["C2C"].to_numpy(), r2, h)); names.append("HAR-C2C")
    cols.append(har_forecasts(px["GK"].to_numpy(), r2, h)); names.append("HAR-GK")
    cols.append(gjr_forecasts(df.r.to_numpy(), h, name)); names.append("GJR-GARCH-t")
    return np.column_stack(cols), names


def main():
    dates, data, fixes = load()
    fixes.to_csv(RES / "data_quality_fixes.csv")
    for h in HORIZONS:
        pack = {}
        for a in ASSETS:
            F, names = build_asset(data[a], h, a)
            pack[f"F_{a}"] = F.astype(np.float32)
            pack[f"RV_{a}"] = realized(data[a].r.to_numpy(), h)
            print(f"h={h} {a}: {F.shape[1]} forecasts", flush=True)
        np.savez_compressed(RES / f"forecasts_h{h}.npz", dates=dates.strftime("%Y-%m-%d").to_numpy(),
                            models=np.array(names), assets=np.array(ASSETS), **pack)


if __name__ == "__main__":
    main()
