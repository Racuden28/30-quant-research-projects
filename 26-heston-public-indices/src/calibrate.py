"""Daily Heston calibration from public CBOE indices, 2008-01 to 2026-11 (experiment 1).

Identification is triangular under Heston, which is the point of the design:

  step A  term structure  VIX9D, VIX, VIX3M, VIX6M  ->  v0, kappa, theta   (4 numbers, 3 unknowns)
  step B  VVIX                                       ->  sigma               (VIX^2 is affine in v, so VVIX
                                                                             does not depend on rho)
  step C  SKEW                                       ->  rho                 (given sigma from step B)

Step A is solved exactly: for fixed kappa, VIX_tau^2 = theta (1 - b_tau) + v0 b_tau is linear in
(theta, v0), so we profile kappa on a grid and refine. Step B takes the low-sigma root (VVIX is hump
shaped in sigma; the second root has v stuck at zero and is not a usable model). Step C is a 1-d root in
rho; when the observed SKEW is above what rho = -1 can deliver, the day is "SKEW-infeasible" and we
record how far off it is and which sigma SKEW alone would need.

Outputs results/daily_calibration.csv.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares, minimize_scalar, nnls

import heston as H
from data import RES, load_panel

K_LO, K_HI = 0.1, 30.0      # mean-reversion half-life between 8 days and 7 years
THETA_HI = 1.0              # long-run vol at most 100%
KGRID = np.exp(np.linspace(np.log(K_LO), np.log(K_HI), 200))
RHO_MIN = -0.999
SIG_LO, SIG_HI = 1e-3, 40.0
SIG_UPPER_CAP = 60.0  # beyond this the cumulant contour cannot be trusted (tested)


def fit_term(vols: np.ndarray, taus: np.ndarray, kgrid=None, theta_hi=THETA_HI):
    """Least squares in vol points over (v0, kappa, theta). Returns v0, kappa, theta, rmse (vol pts)."""
    kgrid = KGRID if kgrid is None else kgrid

    def solve(k):
        b = H.term_b(k, taus)
        w = 1.0 / (2 * vols)  # d vol = d var / (2 vol)
        A = np.column_stack([(1 - b) * w, b * w])
        y = vols**2 * w
        (th, v0), _ = nnls(A, y)
        err = H.vix_model(v0, k, th, taus) - vols
        pen = 0.0 if th <= theta_hi else 1.0 + th  # infeasible long-run level
        return np.sqrt(np.mean(err**2)) + pen, v0, th

    errs = np.array([solve(k)[0] for k in kgrid])
    i = int(np.argmin(errs))
    lo, hi = kgrid[max(i - 1, 0)], kgrid[min(i + 1, len(kgrid) - 1)]
    r = minimize_scalar(lambda lk: solve(np.exp(lk))[0], bounds=(np.log(lo), np.log(hi)), method="bounded",
                        options={"xatol": 1e-6})
    k = float(np.exp(r.x))
    rmse, v0, th = solve(k)
    return v0, k, th, 100 * rmse


def vvix_peak(v0, k, th):
    r = minimize_scalar(lambda ls: -H.vvix_model(v0, k, th, np.exp(ls)), bounds=(np.log(0.05), np.log(SIG_HI)),
                        method="bounded")
    return float(np.exp(r.x)), -float(r.fun)


def sigma_from_vvix(vvix, v0, k, th):
    """Low-sigma root of VVIX_model(sigma) = vvix. Returns (sigma, status)."""
    s_pk, vv_pk = vvix_peak(v0, k, th)
    if vvix > vv_pk:
        return s_pk, "above_peak"
    f = lambda s: H.vvix_model(v0, k, th, s) - vvix
    if f(SIG_LO) > 0:
        return SIG_LO, "below_min"
    return brentq(f, SIG_LO, s_pk, xtol=1e-8), "ok"


def rho_from_skew(skew, v0, k, th, s):
    f = lambda r: H.skew_index(v0, k, th, s, r) - skew
    lo, hi = f(RHO_MIN), f(0.999)
    if lo < 0:  # even rho = -1 is not skewed enough
        return RHO_MIN, "infeasible", lo
    if hi > 0:
        return 0.999, "too_little_skew", hi
    return brentq(f, RHO_MIN, 0.999, xtol=1e-8), "ok", 0.0


def sigma_for_skew(skew, v0, k, th, rho=RHO_MIN):
    """Smallest sigma that reproduces SKEW at a given rho (the 'SKEW-implied vol of vol')."""
    f = lambda s: H.skew_index(v0, k, th, s, rho) - skew
    hi = 20.0  # cumulants stay exact (contour test) up to here
    if not f(hi) > 0:
        return np.nan
    if f(SIG_LO) > 0:
        return SIG_LO
    return brentq(f, SIG_LO, hi, xtol=1e-7)


def upper_branch(row, v0, k, th, s_cap=SIG_UPPER_CAP):
    """The second VVIX root (sigma above the VVIX peak, where v is absorbed near zero most of the time).
    Returns its sigma, the rho that SKEW then needs, and how degenerate the variance process is there."""
    s_pk, vv_pk = vvix_peak(v0, k, th)
    vvix = row["VVIX"] / 100
    f = lambda s: H.vvix_model(v0, k, th, s) - vvix
    if vvix > vv_pk or f(s_cap) > 0:
        return dict(sigma_hi=np.nan, rho_hi=np.nan, rho_hi_status="no_root", feller_hi=np.nan)
    s_hi = brentq(f, s_pk, s_cap, xtol=1e-6)
    g = lambda r: H.skew_index(v0, k, th, s_hi, r) - row["SKEW"]
    lo, hi = g(RHO_MIN), g(0.999)
    if lo < 0:
        r_hi, st = np.nan, "infeasible"
    elif hi > 0:
        r_hi, st = np.nan, "too_much_skew"
    else:
        r_hi, st = brentq(g, RHO_MIN, 0.999, xtol=1e-7), "ok"
    return dict(sigma_hi=s_hi, rho_hi=r_hi, rho_hi_status=st, feller_hi=2 * k * th / s_hi**2)


def calibrate_day(row: pd.Series) -> dict:
    names = [n for n in H.TENORS if np.isfinite(row.get(n, np.nan))]
    vols = np.array([row[n] for n in names]) / 100
    taus = np.array([H.TENORS[n] for n in names])
    v0, k, th, rmse = fit_term(vols, taus)
    out = dict(v0=v0, kappa=k, theta=th, term_rmse=rmse, n_tenors=len(names))
    for n, tau in H.TENORS.items():
        out[f"fit_{n}"] = 100 * float(H.vix_model(v0, k, th, tau))
    out["kappa_at_bound"] = bool(k <= K_LO * 1.0001 or k >= K_HI * 0.9999)
    vvix = row["VVIX"] / 100
    s, st = sigma_from_vvix(vvix, v0, k, th)
    out.update(sigma_status=st, vvix_max=100 * vvix_peak(v0, k, th)[1])
    if st != "ok":  # Heston cannot reach the observed VVIX at any sigma
        out.update(sigma=np.nan, rho=np.nan, rho_status="no_sigma", skew_fit=np.nan, skew_gap=np.nan)
    else:
        rho, rst, gap = rho_from_skew(row["SKEW"], v0, k, th, s)
        out.update(sigma=s, rho=rho, rho_status=rst, skew_fit=H.skew_index(v0, k, th, s, rho), skew_gap=gap)
    # SKEW-implied vol of vol at rho = -1 (lower bound on what SKEW needs)
    out["sigma_skew_min"] = sigma_for_skew(row["SKEW"], v0, k, th)
    out["vvix_at_sigma_skew"] = 100 * H.vvix_model(v0, k, th, out["sigma_skew_min"]) if np.isfinite(
        out["sigma_skew_min"]) else np.nan
    # standardized-skewness bookkeeping: best Heston can do with sigma_VVIX vs market
    S_mkt = (100 - row["SKEW"]) / 10
    # with sigma from VVIX (or the VVIX-maximising sigma if VVIX is out of reach, which is generous)
    S_max = H.skewness(H.T30, v0, k, th, s, RHO_MIN)
    out.update(S_mkt=S_mkt, S_heston_max=S_max, nonheston_share=1 - S_max / S_mkt if S_mkt < 0 else np.nan)
    out["feller"] = 2 * k * th / out["sigma"] ** 2 if np.isfinite(out["sigma"]) else np.nan
    out.update(upper_branch(row, v0, k, th))
    return out


def joint_fit(row: pd.Series, scale: dict, x0) -> dict:
    """Compromise fit: all six indices at once, each residual measured in units of that index's median
    absolute daily change (so 1 = an ordinary day's move). Bounds as in the triangular fit."""
    names = [n for n in H.TENORS if np.isfinite(row.get(n, np.nan))]
    taus = np.array([H.TENORS[n] for n in names])
    obs = np.array([row[n] for n in names])

    def resid(x):
        v0, k, th, s, r = x
        e = list((100 * H.vix_model(v0, k, th, taus) - obs) / np.array([scale[n] for n in names]))
        e.append((H.skew_index(v0, k, th, s, r) - row["SKEW"]) / scale["SKEW"])
        e.append((100 * H.vvix_model(v0, k, th, s) - row["VVIX"]) / scale["VVIX"])
        return np.array(e)

    lb = [1e-5, K_LO, 1e-4, 0.01, RHO_MIN]
    ub = [4.0, K_HI, THETA_HI, 10.0, 0.999]
    x0 = np.clip(x0, np.array(lb) + 1e-9, np.array(ub) - 1e-9)
    r = least_squares(resid, x0, bounds=(lb, ub), x_scale=[0.01, 1, 0.01, 0.3, 0.3], max_nfev=400)
    e = resid(r.x)
    out = {f"J_{p}": v for p, v in zip(["v0", "kappa", "theta", "sigma", "rho"], r.x)}
    out.update(J_err_term=float(np.sqrt(np.mean(e[: len(names)] ** 2))), J_err_skew=float(e[-2]),
               J_err_vvix=float(e[-1]))
    return out


def main():
    t0 = time.time()
    df = load_panel()
    rows = [calibrate_day(r) for _, r in df.iterrows()]
    cal = pd.DataFrame(rows, index=df.index)
    cal = pd.concat([df, cal], axis=1)
    scale = {n: float(df[n].diff().abs().median()) for n in ["VIX9D", "VIX", "VIX3M", "VIX6M", "SKEW", "VVIX"]}
    pd.Series(scale).to_csv(RES / "noise_scale.csv", header=["median_abs_daily_change"])
    J = []
    for d, r in cal.iterrows():
        s0 = r["sigma"] if np.isfinite(r["sigma"]) else r["sigma_skew_min"] if np.isfinite(r["sigma_skew_min"]) else 1.0
        x0 = [r["v0"], r["kappa"], r["theta"], min(s0, 5.0), -0.9]
        J.append(joint_fit(r, scale, x0))
    cal = pd.concat([cal, pd.DataFrame(J, index=cal.index)], axis=1)
    cal.to_csv(RES / "daily_calibration.csv", float_format="%.8g")
    print(f"calibrated {len(cal)} days in {time.time() - t0:.0f}s")
    print(cal[["v0", "kappa", "theta", "term_rmse", "sigma", "rho", "sigma_skew_min", "nonheston_share",
               "feller"]].describe().round(4).T)
    print(cal.sigma_status.value_counts(), cal.rho_status.value_counts())


if __name__ == "__main__":
    main()
