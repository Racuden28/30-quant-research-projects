"""Ablations of the triangular calibration on a weekly subsample (every 5th trading day).

A0  main specification (kappa in [0.1, 30], theta <= 1, all available tenors)
A1  wide bounds: kappa in [0.02, 80], theta <= 10
A2  no VIX9D anywhere (same information set in 2008-2010 and after)
A3  only VIX and VIX3M + VIX6M with kappa fixed at the full-sample median (no daily kappa freedom)
A4  naive VVIX: instantaneous vol of VIX at t=0, b sigma sqrt(v0) / (2 VIX^2), instead of the exact
    30-day expected quadratic variation of the VIX futures
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import calibrate as C
import heston as H
from data import RES, load_panel


def run(df, variant):
    rows = []
    kmed = 4.4
    for d, r in df.iterrows():
        r = r.copy()
        kgrid, th_hi = None, C.THETA_HI
        if variant == "A1":
            kgrid, th_hi = np.exp(np.linspace(np.log(0.02), np.log(80), 240)), 10.0
        if variant in ("A2", "A3"):
            r["VIX9D"] = np.nan
        names = [n for n in H.TENORS if np.isfinite(r.get(n, np.nan))]
        vols = np.array([r[n] for n in names]) / 100
        taus = np.array([H.TENORS[n] for n in names])
        if variant == "A3":
            kgrid = np.array([kmed * 0.9999, kmed, kmed * 1.0001])
        v0, k, th, rmse = C.fit_term(vols, taus, kgrid=kgrid, theta_hi=th_hi)
        if variant == "A4":
            b = H.term_b(k, H.T30)
            vix2 = th * (1 - b) + b * v0
            s = (r.VVIX / 100) * 2 * vix2 / (b * np.sqrt(v0))
            st = "ok"
        else:
            s, st = C.sigma_from_vvix(r.VVIX / 100, v0, k, th)
        smin = C.sigma_for_skew(r.SKEW, v0, k, th)
        if st == "ok":
            rho, rst, _ = C.rho_from_skew(r.SKEW, v0, k, th, s)
        else:
            rho, rst = np.nan, "no_sigma"
        rows.append(dict(date=d, kappa=k, theta=th, term_rmse=rmse, sigma=s if st == "ok" else np.nan,
                         sigma_skew_min=smin, feasible=rst == "ok", kappa_at_bound=(k <= 0.1001 or k >= 29.99)))
    return pd.DataFrame(rows).set_index("date")


def main():
    df = load_panel().iloc[::5]
    out = []
    for v in ["A0", "A1", "A2", "A3", "A4"]:
        r = run(df, v)
        ratio = (r.sigma_skew_min / r.sigma).dropna()
        out.append(dict(variant=v, days=len(r), share_feasible=r.feasible.mean(), sigma_vvix=r.sigma.median(),
                        sigma_skew_min=r.sigma_skew_min.median(), ratio_median=ratio.median(),
                        share_ratio_gt1=(ratio > 1).mean(), kappa_median=r.kappa.median(),
                        theta_vol_median=np.sqrt(r.theta).median(), term_rmse_median=r.term_rmse.median(),
                        kappa_at_bound=r.kappa_at_bound.mean()))
        print(out[-1])
    pd.DataFrame(out).to_csv(RES / "ablations.csv", index=False, float_format="%.4g")


if __name__ == "__main__":
    main()
