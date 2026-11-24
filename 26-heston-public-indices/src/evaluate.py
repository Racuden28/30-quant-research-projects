"""Experiments 2-5 on the daily calibration (run after calibrate.py).

E2  The two vol-of-vols: how often SKEW and VVIX are jointly feasible, the ratio sigma_SKEW / sigma_VVIX,
    and the share of 30-day skewness Heston cannot produce. Year-by-year, stress windows, block-bootstrap CIs.
E3  State dependence: Heston's own comparative statics (d VVIX / d VIX, d SKEW / d VIX from bumping v0)
    against the daily regression slopes in the data.
E4  Out-of-sample next-day forecasts. Structure (kappa, theta, sigma, rho) is frozen at day t; only v is
    moved to match VIX on day t+1. Targets: VIX9D, VIX3M, VIX6M (vs parallel and proportional shifts),
    and changes in SKEW and VVIX (vs no change). Diebold-Mariano with Newey-West, Holm across the family.
E5  Physical vs risk-neutral: kappa, sigma and rho estimated from the time series vs the calibrated ones.
    Under Heston with the standard lambda*v variance risk premium, sigma and rho must be equal under P and Q.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import heston as H
from data import RES, STRESS, in_window

rng = np.random.default_rng(26)
BLOCK = 63  # quarter-long blocks for the stationary/moving block bootstrap
NBOOT = 2000


# ------------------------------------------------------------------------------------------ helpers
def block_boot(x: np.ndarray, stat, block=BLOCK, nboot=NBOOT, seed=26):
    """Moving-block bootstrap percentile CI of stat(x) for a daily series x (1-d or 2-d rows)."""
    rng = np.random.default_rng(seed)  # own stream per call: results do not depend on run order
    x = np.asarray(x)
    n = len(x)
    nb = int(np.ceil(n / block))
    out = np.empty(nboot)
    for b in range(nboot):
        starts = rng.integers(0, n - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[b] = stat(x[idx])
    return np.nanpercentile(out, [2.5, 97.5])


def nw_se_mean(d: np.ndarray, lags=10):
    d = d - d.mean()
    n = len(d)
    g = np.array([np.sum(d[k:] * d[: n - k]) / n for k in range(lags + 1)])
    v = g[0] + 2 * np.sum((1 - np.arange(1, lags + 1) / (lags + 1)) * g[1:])
    return np.sqrt(v / n)


def dm_test(e_model, e_bench, lags=10):
    """DM on squared-error differential (positive mean = model better). Returns gain %, t, two-sided p."""
    from scipy.stats import norm
    d = e_bench**2 - e_model**2
    t = d.mean() / nw_se_mean(d, lags)
    return 100 * d.mean() / np.mean(e_bench**2), t, 2 * norm.sf(abs(t))


def holm(p: np.ndarray):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def ols_slope_nw(y, x, lags=5):
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n = len(y)
    S = np.zeros((2, 2))
    for k in range(lags + 1):
        w = 1.0 if k == 0 else 1 - k / (lags + 1)
        G = (X[k:] * e[k:, None]).T @ (X[: n - k] * e[: n - k, None]) / n
        S += w * (G if k == 0 else G + G.T)
    Q = np.linalg.inv(X.T @ X / n)
    V = Q @ S @ Q / n
    return b[1], np.sqrt(V[1, 1])


# ------------------------------------------------------------------------------------------ E2
def e2_feasibility(c: pd.DataFrame):
    ok = c.sigma_status == "ok"
    # the non-Heston share is defined only where VVIX pins sigma (low root exists)
    c = c.assign(nonheston_share=c.nonheston_share.where(ok))
    c = c.assign(ratio=c.sigma_skew_min / c.sigma, infeasible=(c.rho_status != "ok").astype(float),
                 vvix_unreachable=(~ok).astype(float))
    rows = []
    groups = {"Full sample 2008-2026": np.ones(len(c), bool)}
    groups.update({k: in_window(c.index, w) for k, w in STRESS.items()})
    groups["Nov 2026 shock (11-16 to 11-23)"] = in_window(c.index, ("2026-11-16", "2026-11-23"))
    groups["Calm: VIX below its median"] = (c.VIX < c.VIX.median()).values
    groups["Turbulent: VIX top decile"] = (c.VIX > c.VIX.quantile(0.9)).values
    for name, m in groups.items():
        s = c[m]
        r = s.ratio[s.sigma_status == "ok"].values
        row = dict(window=name, days=len(s), share_infeasible=s.infeasible.mean(),
                   share_vvix_unreachable=s.vvix_unreachable.mean(),
                   sigma_vvix_median=s.sigma.median(), sigma_skew_min_median=s.sigma_skew_min.median(),
                   ratio_median=np.median(r) if len(r) else np.nan,
                   nonheston_share_median=s.nonheston_share.median(),
                   term_rmse_median=s.term_rmse.median(), kappa_median=s.kappa.median(),
                   feller_violated=(s.feller < 1).mean())
        if name.startswith("Full"):
            row["share_infeasible_ci"] = block_boot(s.infeasible.values, np.mean).round(3).tolist()
            row["ratio_median_ci"] = block_boot(r, np.median).round(3).tolist()
            row["nonheston_share_ci"] = block_boot(s.nonheston_share.dropna().values, np.median).round(3).tolist()
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "e2_feasibility_by_window.csv", index=False, float_format="%.4g")
    # the degenerate upper VVIX branch (sigma beyond the VVIX peak)
    up = c.rho_hi_status == "ok"
    ub = pd.DataFrame([dict(share_days_upper_branch_exact=up.mean(),
                            share_days_any_branch_exact=((c.rho_status == "ok") | up).mean(),
                            sigma_hi_median=c.sigma_hi[up].median(), rho_hi_median=c.rho_hi[up].median(),
                            rho_hi_share_positive=(c.rho_hi[up] > 0).mean(), feller_hi_median=c.feller_hi[up].median(),
                            sigma_hi_over_sigma_lo_median=(c.sigma_hi / c.sigma)[up].median())])
    ub.to_csv(RES / "e2_upper_branch.csv", index=False, float_format="%.4g")
    print(ub.T)
    yr = c.groupby(c.index.year).agg(days=("VIX", "size"), share_infeasible=("infeasible", "mean"),
                                     share_vvix_unreachable=("vvix_unreachable", "mean"),
                                     sigma_vvix=("sigma", "median"), sigma_skew_min=("sigma_skew_min", "median"),
                                     nonheston_share=("nonheston_share", "median"),
                                     kappa=("kappa", "median"), theta_vol=("theta", lambda t: np.sqrt(t).median()),
                                     term_rmse=("term_rmse", "median"))
    yr.to_csv(RES / "e2_by_year.csv", float_format="%.4g")
    # what drives infeasibility: logit-free view, shares by SKEW and VVIX terciles
    c["SKEW_t"] = pd.qcut(c.SKEW, 3, labels=["low SKEW", "mid SKEW", "high SKEW"])
    c["VVIX_t"] = pd.qcut(c.VVIX, 3, labels=["low VVIX", "mid VVIX", "high VVIX"])
    piv = c.pivot_table(index="SKEW_t", columns="VVIX_t", values="infeasible", aggfunc="mean", observed=True)
    piv.to_csv(RES / "e2_infeasible_by_terciles.csv", float_format="%.3f")
    print(tab.round(3).to_string())
    print(yr.round(3).to_string())
    print(piv.round(2))
    return tab


# ------------------------------------------------------------------------------------------ E3
def model_elasticities(r: pd.Series, h=0.02):
    """Heston comparative statics at day-t parameters: bump v0 by +/- h (relative) and read off
    dVVIX/dVIX and dSKEW/dVIX (index points per VIX point)."""
    v0, k, th = r.v0, r.kappa, r.theta
    s = r.sigma if np.isfinite(r.sigma) else r.J_sigma
    rho = r.rho if np.isfinite(r.rho) else r.J_rho
    vals = []
    for f in (1 - h, 1 + h):
        v = v0 * f
        vals.append((100 * H.vix_model(v, k, th, H.T30), 100 * H.vvix_model(v, k, th, s),
                     H.skew_index(v, k, th, s, rho)))
    (x0, a0, s0), (x1, a1, s1) = vals
    return (a1 - a0) / (x1 - x0), (s1 - s0) / (x1 - x0)


def e3_state_dependence(c: pd.DataFrame):
    sub = c.iloc[::5]  # weekly subsample is plenty for the model side
    el = np.array([model_elasticities(r) for _, r in sub.iterrows()])
    d = c[["VIX", "VVIX", "SKEW"]].diff().dropna()
    b_vv, se_vv = ols_slope_nw(d.VVIX.values, d.VIX.values)
    b_sk, se_sk = ols_slope_nw(d.SKEW.values, d.VIX.values)
    out = pd.DataFrame([
        dict(quantity="dVVIX/dVIX", heston_median=np.median(el[:, 0]), heston_p10=np.percentile(el[:, 0], 10),
             heston_p90=np.percentile(el[:, 0], 90), heston_share_positive=(el[:, 0] > 0).mean(),
             data_slope=b_vv, data_se=se_vv, data_ci_lo=b_vv - 1.96 * se_vv, data_ci_hi=b_vv + 1.96 * se_vv),
        dict(quantity="dSKEW/dVIX", heston_median=np.median(el[:, 1]), heston_p10=np.percentile(el[:, 1], 10),
             heston_p90=np.percentile(el[:, 1], 90), heston_share_positive=(el[:, 1] > 0).mean(),
             data_slope=b_sk, data_se=se_sk, data_ci_lo=b_sk - 1.96 * se_sk, data_ci_hi=b_sk + 1.96 * se_sk),
    ])
    out.to_csv(RES / "e3_state_dependence.csv", index=False, float_format="%.4g")
    pd.DataFrame(el, index=sub.index, columns=["dVVIX_dVIX", "dSKEW_dVIX"]).to_csv(
        RES / "e3_model_elasticities_weekly.csv", float_format="%.5g")
    print(out.round(3).to_string())


# ------------------------------------------------------------------------------------------ E4
def e4_forecasts(c: pd.DataFrame):
    rows = []
    recs = []
    idx = c.index
    for i in range(len(c) - 1):
        r, n = c.iloc[i], c.iloc[i + 1]
        k, th = r.kappa, r.theta
        b30 = H.term_b(k, H.T30)
        # v on day t+1 implied by VIX_{t+1} with day-t structure
        v1 = max((n.VIX / 100) ** 2 - th * (1 - b30), 1e-6) / b30
        rec = dict(date=idx[i + 1])
        for nm in ["VIX9D", "VIX3M", "VIX6M"]:
            if not (np.isfinite(r[nm]) and np.isfinite(n[nm])):
                continue
            pred = 100 * H.vix_model(v1, k, th, H.TENORS[nm])
            rec[f"{nm}_obs"] = n[nm]
            rec[f"{nm}_heston"] = pred
            rec[f"{nm}_shift"] = r[nm] + (n.VIX - r.VIX)
            rec[f"{nm}_prop"] = r[nm] * n.VIX / r.VIX
            rec[f"{nm}_rw"] = r[nm]
        recs.append(rec)
    f = pd.DataFrame(recs).set_index("date")
    # SKEW / VVIX change forecasts (weekly subsample of origins is not needed: closed forms are cheap)
    sk, vv = [], []
    for i in range(len(c) - 1):
        r, n = c.iloc[i], c.iloc[i + 1]
        k, th = r.kappa, r.theta
        s = r.sigma if np.isfinite(r.sigma) else r.J_sigma
        rho = r.rho if np.isfinite(r.rho) else r.J_rho
        b30 = H.term_b(k, H.T30)
        v1 = max((n.VIX / 100) ** 2 - th * (1 - b30), 1e-6) / b30
        v0 = max((r.VIX / 100) ** 2 - th * (1 - b30), 1e-6) / b30
        sk.append(H.skew_index(v1, k, th, s, rho) - H.skew_index(v0, k, th, s, rho))
        vv.append(100 * (H.vvix_model(v1, k, th, s) - H.vvix_model(v0, k, th, s)))
    f["dSKEW_obs"] = c.SKEW.diff().values[1:]
    f["dSKEW_heston"] = sk
    f["dVVIX_obs"] = c.VVIX.diff().values[1:]
    f["dVVIX_heston"] = vv
    # scaled-down version: best linear use of the Heston signal would be beta*signal; we report the raw
    # model and also the sign hit-rate, which needs no scaling
    f.to_csv(RES / "e4_forecasts.csv", float_format="%.5g")
    for nm in ["VIX9D", "VIX3M", "VIX6M"]:
        g = f.dropna(subset=[f"{nm}_obs"])
        e_h = (g[f"{nm}_heston"] - g[f"{nm}_obs"]).values
        for bench in ["shift", "prop", "rw"]:
            e_b = (g[f"{nm}_{bench}"] - g[f"{nm}_obs"]).values
            gain, t, p = dm_test(e_h, e_b)
            rows.append(dict(target=nm, benchmark=bench, n=len(g), rmse_heston=np.sqrt(np.mean(e_h**2)),
                             rmse_bench=np.sqrt(np.mean(e_b**2)), mse_gain_pct=gain, dm_t=t, p=p))
    for nm in ["dSKEW", "dVVIX"]:
        e_h = (f[f"{nm}_heston"] - f[f"{nm}_obs"]).values
        e_b = (0.0 - f[f"{nm}_obs"]).values
        gain, t, p = dm_test(e_h, e_b)
        hit = np.mean(np.sign(f[f"{nm}_heston"]) == np.sign(f[f"{nm}_obs"]))
        rows.append(dict(target=nm, benchmark="no change", n=len(f), rmse_heston=np.sqrt(np.mean(e_h**2)),
                         rmse_bench=np.sqrt(np.mean(e_b**2)), mse_gain_pct=gain, dm_t=t, p=p, sign_hit=hit))
    tab = pd.DataFrame(rows)
    tab["p_holm"] = holm(tab.p.values)
    # stress-window breakdown for the term-structure forecasts vs the best simple benchmark
    srows = []
    for wname, w in list(STRESS.items()) + [("Nov 2026 shock", ("2026-11-16", "2026-11-23"))]:
        m = in_window(f.index, w)
        for nm in ["VIX9D", "VIX3M", "VIX6M"]:
            g = f[m].dropna(subset=[f"{nm}_obs"])
            if len(g) < 3:
                continue
            row = dict(window=wname, target=nm, n=len(g))
            for mdl in ["heston", "shift", "prop", "rw"]:
                row[f"rmse_{mdl}"] = np.sqrt(np.mean((g[f"{nm}_{mdl}"] - g[f"{nm}_obs"]) ** 2))
            srows.append(row)
    pd.DataFrame(srows).to_csv(RES / "e4_forecasts_stress.csv", index=False, float_format="%.4g")
    tab.to_csv(RES / "e4_forecast_tests.csv", index=False, float_format="%.4g")
    print(tab.round(4).to_string())
    print(pd.DataFrame(srows).round(3).to_string())


# ------------------------------------------------------------------------------------------ E5
def e5_p_vs_q(c: pd.DataFrame):
    """Physical-measure counterparts, all full-sample estimators written on day pairs (t, t+1) so that a
    moving-block bootstrap of pairs (one-year blocks) gives honest CIs.

    v_t is backed out of VIX_t with the day's term-structure fit (VIX^2 = a + b v).
    kappa^P: AR(1) of VIX^2 on 5-day steps;  sigma^P: sqrt( sum dv^2 / sum v dt );  rho^P: corr(d lnS, dv).
    """
    b30 = H.term_b(c.kappa.values, H.T30)
    v = np.maximum(((c.VIX / 100) ** 2 - c.theta * (1 - b30)) / b30, 1e-6).values
    vix2 = ((c.VIX / 100) ** 2).values
    lnS = np.log(c.GSPC.values)
    days = c.index.values.astype("datetime64[D]").astype(float)
    n = len(c)
    i1 = np.arange(n - 1)
    dv, vlag, dl = v[i1 + 1] - v[i1], v[i1], lnS[i1 + 1] - lnS[i1]
    dts = (days[i1 + 1] - days[i1]) / 365
    i5 = np.arange(n - 5)
    x5, y5, g5 = vix2[i5], vix2[i5 + 5], (days[i5 + 5] - days[i5]) / 365

    def kappa_p(ix):
        ix = ix[ix < len(i5)]
        x, y = x5[ix], y5[ix]
        phi = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
        T = len(ix) / 5
        phi_bc = phi + (1 + 3 * phi) / T  # Kendall bias correction on the effective sample
        return -np.log(np.clip(phi_bc, 1e-6, 0.999999)) / np.mean(g5[ix])

    def sig_p(ix):
        return np.sqrt(np.sum(dv[ix] ** 2) / np.sum(vlag[ix] * dts[ix]))

    def rho_p(ix):
        return np.corrcoef(dl[ix], dv[ix])[0, 1]

    def sig_p_robust(ix):  # median-based: robust to the few days with huge dv
        z = dv[ix] / np.sqrt(vlag[ix] * dts[ix])
        return 1.4826 * np.median(np.abs(z - np.median(z)))

    est = {f.__name__: f(i1) for f in (kappa_p, sig_p, sig_p_robust, rho_p)}
    nb = int(np.ceil((n - 1) / 252))
    rng = np.random.default_rng(2605)
    B = []
    for _ in range(1000):
        st = rng.integers(0, n - 1 - 252, nb)
        ix = (st[:, None] + np.arange(252)[None, :]).ravel()[: n - 1]
        B.append([f(ix) for f in (kappa_p, sig_p, sig_p_robust, rho_p)])
    B = np.array(B)
    ci = np.nanpercentile(B, [2.5, 97.5], axis=0)
    yrs = []
    for y, g in c.groupby(c.index.year):
        ix = np.where(c.index.year[:-1] == y)[0]
        if len(ix) < 100:
            continue
        yrs.append(dict(year=y, kappa_P=kappa_p(ix), sigma_P=sig_p(ix), sigma_P_robust=sig_p_robust(ix),
                        rho_P=rho_p(ix), kappa_Q=g.kappa.median(), sigma_Q_vvix=g.sigma.median(),
                        sigma_Q_skew_min=g.sigma_skew_min.median(), sigma_Q_joint=g.J_sigma.median(),
                        rho_Q_joint=g.J_rho.median()))
    yrs = pd.DataFrame(yrs)
    yrs.to_csv(RES / "e5_p_vs_q_by_year.csv", index=False, float_format="%.4g")
    tab = pd.DataFrame([
        dict(param="kappa", P_estimator="AR(1) of VIX^2, 5-day steps", P=est["kappa_p"], P_ci_lo=ci[0, 0],
             P_ci_hi=ci[1, 0], Q_median=c.kappa.median(), Q_source="term-structure fit"),
        dict(param="sigma", P_estimator="sqrt(sum dv^2 / sum v dt)", P=est["sig_p"], P_ci_lo=ci[0, 1],
             P_ci_hi=ci[1, 1], Q_median=c.sigma.median(), Q_source="VVIX"),
        dict(param="sigma", P_estimator="robust (MAD of standardised dv)", P=est["sig_p_robust"],
             P_ci_lo=ci[0, 2], P_ci_hi=ci[1, 2], Q_median=c.sigma.median(), Q_source="VVIX"),
        dict(param="sigma", P_estimator="robust (MAD of standardised dv)", P=est["sig_p_robust"],
             P_ci_lo=ci[0, 2], P_ci_hi=ci[1, 2], Q_median=c.sigma_skew_min.median(),
             Q_source="minimum needed by SKEW (rho=-1)"),
        dict(param="rho", P_estimator="corr(d lnS, dv), daily", P=est["rho_p"], P_ci_lo=ci[0, 3],
             P_ci_hi=ci[1, 3], Q_median=c.J_rho.median(), Q_source="joint six-index fit"),
    ])
    tab.to_csv(RES / "e5_p_vs_q.csv", index=False, float_format="%.4g")
    print(tab.round(3).to_string())
    print(yrs.round(3).to_string())
    rP = est["rho_p"]
    sub = c[c.sigma_status == "ok"].iloc[::5]
    miss = [1 - H.skewness(H.T30, r.v0, r.kappa, r.theta, r.sigma, rP) / r.S_mkt for _, r in sub.iterrows()]
    pd.Series(miss, index=sub.index, name="nonheston_share_rhoP").to_csv(RES / "e5_nonheston_share_rhoP.csv")
    print("non-Heston share of skewness with rho = rho^P:", np.median(miss))


def main():
    c = pd.read_csv(RES / "daily_calibration.csv", index_col=0, parse_dates=True)
    import sys
    which = sys.argv[1:] or ["e2", "e3", "e4", "e5"]
    for name, f in [("e2", e2_feasibility), ("e3", e3_state_dependence), ("e4", e4_forecasts),
                    ("e5", e5_p_vs_q)]:
        if name in which:
            f(c)


if __name__ == "__main__":
    main()
