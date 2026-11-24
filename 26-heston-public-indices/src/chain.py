"""Experiment 6: out-of-route validation on the full S&P 500 option chain of 2026-11-23 (one snapshot).

1. Forwards and discount factors per expiry from put-call parity (no rate or dividend data needed).
2. Replicate VIX9D / VIX / VIX3M / VIX6M and SKEW from the chain with the CBOE formulas, and compare with
   the published indices (checks that the data and our reading of the definitions are right).
3. Calibrate Heston directly to ~all OTM options from 7 days to 1 year (implied-vol errors, each expiry
   weighted equally), and compare with the parameters the index route gives on the same day.
4. Per-expiry (sigma, rho) with (v0, kappa, theta) held at the chain fit: the term structure of vol-of-vol
   Heston would need.
5. Price the chain with the index-route parameters: how wrong is a Heston built from public indices only?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import calibrate as C
import heston as H
from bs import black_price, implied_vol
from data import RAW, RES

SNAP = "2026-11-23"
SETTLE_UTC_HOUR = 21  # SPX PM settlement 16:00 New York = 21:00 UTC in November


def load_chain():
    df = pd.read_csv(RAW / "chain_SPX_20261123.csv.gz")
    t0 = df.spot_time.iloc[0]
    df["T"] = (df.expiry + SETTLE_UTC_HOUR * 3600 - t0) / (365 * 86400)
    df = df[(df.bid > 0) & (df.ask > df.bid)].copy()
    df["mid"] = 0.5 * (df.bid + df.ask)
    return df, float(df.spot.iloc[0])


def forwards(df, spot):
    """Per expiry: regress (C - P) on K over strikes within 5% of spot: C - P = D F - D K."""
    out = {}
    for e, g in df.groupby("expiry"):
        c = g[g.type == "C"].set_index("strike").mid
        p = g[g.type == "P"].set_index("strike").mid
        ks = c.index.intersection(p.index)
        ks = ks[(ks > 0.95 * spot) & (ks < 1.05 * spot)]
        if len(ks) < 5:
            continue
        y = (c[ks] - p[ks]).values
        X = np.column_stack([np.ones(len(ks)), -ks.values])
        (DF, D), *_ = np.linalg.lstsq(X, y, rcond=None)
        out[e] = dict(T=float(g["T"].iloc[0]), D=D, F=DF / D)
    return pd.DataFrame(out).T


def otm_table(df, fw):
    rows = []
    for e, g in df.groupby("expiry"):
        if e not in fw.index:
            continue
        F, D, T = fw.loc[e, ["F", "D", "T"]]
        g = g[((g.type == "P") & (g.strike < F)) | ((g.type == "C") & (g.strike >= F))].copy()
        g["F"], g["D"], g["T"] = F, D, T
        g["k"] = np.log(g.strike / F)
        g["iv"] = implied_vol(g.mid.values, F, g.strike.values, T, D, (g.type == "C").values)
        g["iv_bid"] = implied_vol(g.bid.values, F, g.strike.values, T, D, (g.type == "C").values)
        g["iv_ask"] = implied_vol(g.ask.values, F, g.strike.values, T, D, (g.type == "C").values)
        rows.append(g)
    return pd.concat(rows)


# --------------------------------------------------------------------------------------------- CBOE
def cboe_strip(g):
    """CBOE strike selection for one expiry: OTM puts below K0 and calls above, stop after two consecutive
    zero bids; at K0 the average of put and call. Returns K, Q (undiscounted / D), K0, F."""
    F, D = g.F.iloc[0], g.D.iloc[0]
    calls = g[g.type == "C"].set_index("strike").sort_index()
    puts = g[g.type == "P"].set_index("strike").sort_index()
    K0 = max([k for k in puts.index.union(calls.index) if k <= F], default=np.nan)
    sel = []
    for side, frame, order in (("P", puts, -1), ("C", calls, 1)):
        ks = frame.index.values[::order]
        zeros = 0
        for k in ks:
            if (order == -1 and k > K0) or (order == 1 and k <= K0):
                continue
            q = frame.loc[k]
            if q.bid <= 0:
                zeros += 1
                if zeros >= 2:
                    break
                continue
            zeros = 0
            sel.append((k, q.mid))
    if K0 in puts.index and K0 in calls.index:
        sel.append((K0, 0.5 * (puts.loc[K0].mid + calls.loc[K0].mid)))
    s = pd.Series(dict(sel)).sort_index()
    return s.index.values.astype(float), s.values / D, K0, F


def cboe_moments(K, Q, K0, F, T):
    dK = np.gradient(K)
    var = 2 / T * np.sum(dK / K**2 * Q) - (F / K0 - 1) ** 2 / T
    l0 = np.log(K0 / F)
    e1 = -(1 + np.log(F / K0) - F / K0)
    e2 = 2 * l0 * (F / K0 - 1) + 0.5 * l0**2
    e3 = 3 * l0**2 * (l0 / 3 - 1 + F / K0)
    lk = np.log(K / F)
    P1 = np.sum(-1 / K**2 * Q * dK) + e1
    P2 = np.sum(2 / K**2 * (1 - lk) * Q * dK) + e2
    P3 = np.sum(3 / K**2 * (2 * lk - lk**2) * Q * dK) + e3
    S = (P3 - 3 * P1 * P2 + 2 * P1**3) / (P2 - P1**2) ** 1.5
    return var, S


def replicate_indices(otm_full, fw):
    """Replicates the CBOE indices from standard (third-Friday) and weekly expiries, interpolating to the
    target horizon in total variance (VIX family) or linearly in T (SKEW), as CBOE does."""
    res = {}
    per = []
    for e in fw.index:
        g = otm_full[otm_full.expiry == e]
        T = fw.loc[e, "T"]
        if T < 1 / 365 or len(g) < 20:
            continue
        K, Q, K0, F = cboe_strip(g)
        var, S = cboe_moments(K, Q, K0, F, T)
        per.append(dict(expiry=e, T=T, var=var, S=S, n_strikes=len(K)))
    per = pd.DataFrame(per).sort_values("T")
    for name, days in [("VIX9D", 9), ("VIX", 30), ("VIX3M", 93), ("VIX6M", 183)]:
        tgt = days / 365
        lo = per[per["T"] <= tgt].iloc[-1]
        hi = per[per["T"] > tgt].iloc[0]
        w = (hi["T"] - tgt) / (hi["T"] - lo["T"])
        tv = w * lo["var"] * lo["T"] + (1 - w) * hi["var"] * hi["T"]
        res[name] = 100 * np.sqrt(tv / tgt)
    tgt = 30 / 365
    lo = per[per["T"] <= tgt].iloc[-1]
    hi = per[per["T"] > tgt].iloc[0]
    w = (hi["T"] - tgt) / (hi["T"] - lo["T"])
    res["SKEW"] = 100 - 10 * (w * lo.S + (1 - w) * hi.S)
    return res, per


# --------------------------------------------------------------------------------------------- fit
def model_iv(params, g):
    T, F = g["T"].iloc[0], g.F.iloc[0]
    k = g.k.values
    pr = H.otm_price_over_F(k, T, params)
    return implied_vol(pr * F, F, g.strike.values, T, 1.0, k >= 0)


def fit_chain(otm, x0):
    groups = [g for _, g in otm.groupby("expiry")]

    def resid(x):
        out = []
        for g in groups:
            iv = model_iv(x, g)
            e = np.nan_to_num(iv - g.iv.values, nan=0.5)
            out.append(e / np.sqrt(len(g)))  # each expiry gets equal total weight
        return np.concatenate(out)

    lb = [1e-4, 0.05, 1e-3, 0.05, -0.999]
    ub = [1.0, 50.0, 1.0, 10.0, 0.5]
    r = least_squares(resid, np.clip(x0, lb, ub), bounds=(lb, ub), x_scale=[0.01, 1, 0.01, 0.3, 0.3],
                      max_nfev=300)
    return r.x


def per_expiry_sigma_rho(otm, base):
    v0, k, th = base[:3]
    rows = []
    for e, g in otm.groupby("expiry"):
        def resid(x):
            return np.nan_to_num(model_iv((v0, k, th, x[0], x[1]), g) - g.iv.values, nan=0.5)
        r = least_squares(resid, [base[3], base[4]], bounds=([0.05, -0.999], [15.0, 0.5]))
        rows.append(dict(expiry=e, T=g["T"].iloc[0], sigma=r.x[0], rho=r.x[1],
                         rmse_volpts=100 * np.sqrt(np.mean(r.fun**2)), n=len(g)))
    return pd.DataFrame(rows)


def iv_errors(params, otm, label):
    rows = []
    for e, g in otm.groupby("expiry"):
        iv = model_iv(params, g)
        rows.append(pd.DataFrame(dict(expiry=e, T=g["T"].values, k=g.k.values, strike=g.strike.values,
                                      iv=g.iv.values, iv_model=iv, half_spread=0.5 * (g.iv_ask - g.iv_bid).values,
                                      model=label)))
    return pd.concat(rows)


def main():
    df, spot = load_chain()
    fw = forwards(df, spot)
    otm_full = otm_table(df, fw)
    rep, per = replicate_indices(otm_full, fw)
    per.to_csv(RES / "e6_chain_per_expiry_moments.csv", index=False, float_format="%.6g")
    idx = {n: float(pd.read_csv(RAW / f"{n}.csv").set_index("date").loc[SNAP, "close"]) for n in
           ["VIX9D", "VIX", "VIX3M", "VIX6M", "VVIX"]}
    skew_prev = float(pd.read_csv(RAW / "SKEW.csv").set_index("date").loc["2026-11-20", "close"])
    rep_tab = pd.DataFrame([dict(index=n, replicated=rep[n], published=idx.get(n, np.nan)) for n in
                            ["VIX9D", "VIX", "VIX3M", "VIX6M", "SKEW"]])
    rep_tab.loc[rep_tab["index"] == "SKEW", "published"] = np.nan
    rep_tab["note"] = ["published value at ~20:00 UTC"] * 4 + [f"not yet published; 2026-11-20 close {skew_prev:.2f}"]
    rep_tab.to_csv(RES / "e6_replication.csv", index=False, float_format="%.4f")
    print(rep_tab)

    # fitting set: 7 days to 1 year, OTM, IV defined, half-spread below 5 vol points, |k| within 4 sd
    otm = otm_full[(otm_full["T"] >= 7 / 365) & (otm_full["T"] <= 1.0)].dropna(subset=["iv", "iv_bid", "iv_ask"])
    otm = otm[(otm.iv_ask - otm.iv_bid) < 0.10]
    sd = otm.iv * np.sqrt(otm["T"])
    otm = otm[np.abs(otm.k) < 4 * sd]
    print("fitting set:", len(otm), "options,", otm.expiry.nunique(), "expiries")

    # index route on the same day (SKEW replicated from the chain, since the published value is not out)
    row = pd.Series({**idx, "SKEW": rep["SKEW"]})
    tri = C.calibrate_day(row)
    p_tri = (tri["v0"], tri["kappa"], tri["theta"], tri["sigma"], tri["rho"]) if np.isfinite(tri["sigma"]) else None
    scale = pd.read_csv(RES / "noise_scale.csv", index_col=0).iloc[:, 0].to_dict()
    jf = C.joint_fit(row, scale, [tri["v0"], tri["kappa"], tri["theta"], 1.5, -0.9])
    p_joint = tuple(jf[f"J_{p}"] for p in ["v0", "kappa", "theta", "sigma", "rho"])

    p_chain = fit_chain(otm, list(p_joint))
    per_exp = per_expiry_sigma_rho(otm, p_chain)
    per_exp.to_csv(RES / "e6_per_expiry_sigma_rho.csv", index=False, float_format="%.5g")

    errs = [iv_errors(p_chain, otm, "Heston fitted to the chain")]
    if p_tri is not None:
        errs.append(iv_errors(p_tri, otm, "Index route: triangular (rho at bound)"))
    errs.append(iv_errors(p_joint, otm, "Index route: joint six-index fit"))
    E = pd.concat(errs)
    E.to_csv(RES / "e6_iv_errors.csv.gz", index=False, float_format="%.6g")

    def implied_indices(p):
        v0, k, th, s, r = p
        d = {n: 100 * float(H.vix_model(v0, k, th, t)) for n, t in H.TENORS.items()}
        d["SKEW"] = H.skew_index(v0, k, th, s, r)
        d["VVIX"] = 100 * H.vvix_model(v0, k, th, s)
        return d

    rows = []
    for lab, p in [("Heston fitted to the chain", p_chain), ("Index route: triangular", p_tri),
                   ("Index route: joint six-index fit", p_joint)]:
        if p is None:
            continue
        e = E[E.model.str.startswith(lab.split(":")[0])] if lab.startswith("Heston") else E[E.model.str.contains(lab.split(": ")[1].split(" ")[0])]
        rows.append(dict(model=lab, v0=p[0], kappa=p[1], theta=p[2], sigma=p[3], rho=p[4],
                         feller=2 * p[1] * p[2] / p[3] ** 2,
                         iv_rmse_volpts=100 * np.sqrt(np.nanmean((e.iv_model - e.iv) ** 2)),
                         share_within_spread=np.nanmean(np.abs(e.iv_model - e.iv) <= e.half_spread),
                         **{f"implied_{n}": v for n, v in implied_indices(p).items()}))
    rows.append(dict(model="Observed", **{f"implied_{n}": v for n, v in {**idx, "SKEW": rep["SKEW"]}.items()}))
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "e6_chain_vs_index_route.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 250)
    print(tab.round(4).T)
    print(per_exp.round(3).to_string())
    print("median half-spread (vol pts):", 100 * otm.eval("(iv_ask - iv_bid) / 2").median())


if __name__ == "__main__":
    main()
