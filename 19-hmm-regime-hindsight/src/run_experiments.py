"""Every number in README.md for project 19. Writes CSVs to results/.

Run order: build caches (parallel) -> ladder -> attribution -> detection -> stress -> regressions
-> BIC -> stability -> robustness -> model world (simulation).
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

import hmm
from build import LADDER, build
from core import (INDICES, LABEL, RES, TDAYS, ewma_vol, holm, load_closes, load_index, matched_vol_placebo,
                  max_drawdown, nw_se_mean, realtime, sharpe, sharpe_diff_hac, spells,
                  stationary_bootstrap_idx)

N_BOOT = 999
BLOCK = 63
SEED = 19


def _build_args(a):
    name, kw = a
    build(name, **kw)
    return name


def build_all(variants):
    jobs = [(n, kw) for kw in variants for n in (kw.pop("_only", None) or INDICES)]
    with Pool(4) as p:
        p.map(_build_args, jobs)


def ann(x):
    return float(np.mean(x) * TDAYS)


def vol(x):
    return float(np.std(x, ddof=1) * np.sqrt(TDAYS))


# --------------------------------------------------------------------------- 1. ladder
STRATS = LADDER + ["Buy & hold", "Vol placebo", "MA200", "HMM weight", "HMM mean rule"]


def performance():
    rows = []
    for n in INDICES:
        b = build(n)
        yrs = len(b["ret"]["Buy & hold"].net) / TDAYS
        for k in STRATS:
            st = b["ret"][k]
            for kind in ("gross", "net"):
                x = getattr(st, kind)
                rows.append(dict(index=n, label=LABEL[n], strategy=k, returns=kind, start=b["start"].date(),
                                 end=b["end"].date(), years=round(yrs, 1), ann_excess=ann(x), ann_vol=vol(x),
                                 sharpe=float(sharpe(x.values)), max_dd=max_drawdown(x),
                                 exposure=st.exposure, turnover=st.turnover))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "performance.csv", index=False)
    return df


# ladder steps: (label, a, kind_a, b, kind_b). Differences are SR(a) - SR(b).
STEPS = [
    ("State look-ahead, same day (R0 - R1)", "R0 Hindsight", "gross", "R1 Smoothed", "gross"),
    ("State look-ahead, smoother (R1 - R2)", "R1 Smoothed", "gross", "R2 Filtered", "gross"),
    ("Parameter look-ahead (R2 - R3)", "R2 Filtered", "gross", "R3 Real-time", "gross"),
    ("Costs, 5 bp (R3 gross - net)", "R3 Real-time", "gross", "R3 Real-time", "net"),
    ("What is left (R3 net - buy & hold)", "R3 Real-time", "net", "Buy & hold", "net"),
    ("Real-time HMM vs vol placebo (net)", "R3 Real-time", "net", "Vol placebo", "net"),
    ("Hindsight HMM vs buy & hold (R1 - B&H)", "R1 Smoothed", "gross", "Buy & hold", "net"),
    ("Real-time HMM vs MA200 (net)", "R3 Real-time", "net", "MA200", "net"),
]


def ladder_tests():
    rng = np.random.default_rng(SEED)
    rows = []
    for n in INDICES:
        b = build(n)
        cols = sorted({(a, ka) for _, a, ka, _, _ in STEPS} | {(c, kc) for _, _, _, c, kc in STEPS})
        M = np.column_stack([getattr(b["ret"][c], k).values for c, k in cols])
        pos = {ck: i for i, ck in enumerate(cols)}
        T = len(M)
        boot = np.empty((N_BOOT, M.shape[1]))
        for i in range(N_BOOT):
            boot[i] = sharpe(M[stationary_bootstrap_idx(T, BLOCK, rng)])
        for lab, a, ka, c, kc in STEPS:
            ia, ic = pos[(a, ka)], pos[(c, kc)]
            d = float(sharpe(M[:, ia]) - sharpe(M[:, ic]))
            bd = boot[:, ia] - boot[:, ic]
            if (a, ka) == (c, "gross") or a == c:
                p = np.nan  # costs step: deterministic given turnover
            else:
                _, p = sharpe_diff_hac(M[:, ia], M[:, ic], lags=10)
            rows.append(dict(index=n, step=lab, diff=d, ci_lo=float(np.quantile(bd, 0.025)),
                             ci_hi=float(np.quantile(bd, 0.975)), p_hac=p))
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for s, g in df.groupby("step"):
        if g.p_hac.notna().all():
            df.loc[g.index, "p_holm"] = holm(g.p_hac.values)
    df.to_csv(RES / "ladder_tests.csv", index=False)
    # summary across markets
    summ = []
    for s, g in df.groupby("step", sort=False):
        k = int((g["diff"] > 0).sum())
        summ.append(dict(step=s, median=g["diff"].median(), mean=g["diff"].mean(), min=g["diff"].min(),
                         max=g["diff"].max(), n_positive=k, sign_test_p=binomtest(k, len(g), 0.5).pvalue,
                         n_ci_excl_zero=int(((g.ci_lo > 0) | (g.ci_hi < 0)).sum()),
                         n_holm_05=int((g.p_holm < 0.05).sum()) if g.p_holm.notna().all() else np.nan))
    pd.DataFrame(summ).to_csv(RES / "ladder_summary.csv", index=False)
    return df


# --------------------------------------------------------------------------- 2. where hindsight earns
def attribution():
    """Days where the hindsight rule (R1) and the real-time rule (R3) disagree, and what the index did."""
    rows = []
    for n in INDICES:
        b = build(n)
        pos = b["pos"]
        h, r = pos["R1 Smoothed"].shift(1), pos["R3 Real-time"].shift(1)    # positions held on each day
        ex = (b["df"].r - b["df"].rf).reindex(pos.index)
        ok = h.notna() & r.notna()
        for lab, m in [("both in", (h == 1) & (r == 1)), ("hindsight out, real-time in", (h == 0) & (r == 1)),
                       ("hindsight in, real-time out", (h == 1) & (r == 0)), ("both out", (h == 0) & (r == 0))]:
            m = m & ok
            x = ex[m]
            rows.append(dict(index=n, cell=lab, share_days=float(m.sum() / ok.sum()), n_days=int(m.sum()),
                             mean_daily_bp=float(x.mean() * 1e4) if len(x) else np.nan,
                             ann_vol=vol(x) if len(x) > 2 else np.nan,
                             contribution_ann=float(x.sum() / ok.sum() * TDAYS)))
        # split the disagreements by mechanism
        hv, rv, xv = h.values, r.values, ex.values
        cat = np.full(len(hv), "", dtype=object)
        in_h_out, seen_r_out, after_h, r_out_since = False, False, False, True
        for t in range(len(hv)):
            if not (ok.values[t]):
                continue
            if hv[t] == 0:
                if not in_h_out:
                    in_h_out, seen_r_out = True, False
                if rv[t] == 0:
                    seen_r_out = True
                else:
                    cat[t] = "re-entry inside a spell (flicker)" if seen_r_out else "late exit (before the first real-time exit)"
            else:
                if in_h_out:
                    in_h_out, r_out_since = False, True
                if rv[t] == 0:
                    cat[t] = "late return (still out after the spell)" if r_out_since else "false alarm"
                else:
                    r_out_since = False
        for c in ["late exit (before the first real-time exit)", "re-entry inside a spell (flicker)",
                  "late return (still out after the spell)", "false alarm"]:
            m = cat == c
            x = xv[m]
            rows.append(dict(index=n, cell=c, share_days=float(m.sum() / ok.sum()), n_days=int(m.sum()),
                             mean_daily_bp=float(np.mean(x) * 1e4) if len(x) else np.nan,
                             ann_vol=float(np.std(x, ddof=1) * np.sqrt(TDAYS)) if len(x) > 2 else np.nan,
                             contribution_ann=float(x.sum() / ok.sum() * TDAYS) * (-1 if "exit" in c or "flicker" in c else 1)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "attribution.csv", index=False)
    return df


# --------------------------------------------------------------------------- 3. detection lag
def detection(min_spell: int = 21, max_lag: int = 126):
    """For each hindsight (smoothed) turbulent spell of at least `min_spell` days inside the evaluation
    sample, when did the real-time filter first say 'turbulent' (predicted prob > 0.5)?"""
    rows = []
    for n in INDICES:
        b = build(n)
        pos = b["pos"]
        ix = pos.index
        ps = b["P"].p_smooth.reindex(ix).values > 0.5
        prt = b["P"].p_rt_pred.reindex(ix).values > 0.5
        ex = (b["df"].r - b["df"].rf).reindex(ix).values
        p = b["df"].p.reindex(ix).values
        for a, e in spells(ps, min_spell):
            if a < 5 or a + 1 >= len(ix):
                continue
            # real-time flag at close t means out from t+1. Search from 5 days before the hindsight start.
            w = np.flatnonzero(prt[a - 5:min(len(ix), a + max_lag)])
            if len(w) == 0:
                lag, t_flag = np.nan, None
            else:
                t_flag = a - 5 + w[0]
                lag = t_flag - a
            spell_ret = float(np.prod(1 + ex[a:e]) - 1)
            peak = np.nanmax(p[max(0, a - 63):a + 1])
            pre = float(np.prod(1 + ex[a:t_flag + 1]) - 1) if t_flag is not None and t_flag >= a else 0.0
            trough = float(np.nanmin(p[a:e]) / peak - 1)
            dd_at_flag = float(p[t_flag] / peak - 1) if t_flag is not None else np.nan
            dd_at_start = float(p[a - 1] / peak - 1)
            rows.append(dict(index=n, start=ix[a].date(), end=ix[e - 1].date(), length=e - a, lag_days=lag,
                             flag_date=ix[t_flag].date() if t_flag is not None else None,
                             spell_return=spell_ret, return_before_flag=pre,
                             drawdown_at_flag=dd_at_flag, drawdown_at_hindsight_exit=dd_at_start,
                             spell_trough_dd=trough))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "detection.csv", index=False)
    # false alarms: real-time 'turbulent' days that hindsight calls calm
    fa = []
    for n in INDICES:
        b = build(n)
        ix = b["pos"].index
        ps = b["P"].p_smooth.reindex(ix).values > 0.5
        prt = b["P"].p_rt_pred.reindex(ix).values > 0.5
        sp_rt = spells(prt, 1)
        fa.append(dict(index=n, rt_turbulent_share=prt.mean(), hindsight_turbulent_share=ps.mean(),
                       precision=float((prt & ps).sum() / max(prt.sum(), 1)),
                       recall=float((prt & ps).sum() / max(ps.sum(), 1)),
                       agreement=float((prt == ps).mean()),
                       rt_spells_per_year=len(sp_rt) / (len(ix) / TDAYS),
                       hindsight_spells_per_year=len(spells(ps, 1)) / (len(ix) / TDAYS),
                       rt_median_spell=float(np.median([e - a for a, e in sp_rt])) if sp_rt else np.nan,
                       hindsight_median_spell=float(np.median([e - a for a, e in spells(ps, 1)]))))
    pd.DataFrame(fa).to_csv(RES / "classification.csv", index=False)
    return df


# --------------------------------------------------------------------------- 4. stress windows
WINDOWS = {
    "1987 crash": ("1987-10-01", "1987-12-31"),
    "2000-02 dot-com": ("2000-03-24", "2002-10-09"),
    "2008 GFC": ("2007-10-09", "2009-03-09"),
    "2020 COVID": ("2020-02-19", "2020-04-30"),
    "2022 bear": ("2022-01-03", "2022-10-12"),
}


def stress():
    rows = []
    for n in INDICES:
        b = build(n)
        for w, (a, e) in WINDOWS.items():
            if pd.Timestamp(a) < b["start"]:
                continue
            m = (b["pos"].index >= a) & (b["pos"].index <= e)
            ix = b["pos"].index[m]
            row = dict(index=n, window=w)
            for k in ["Buy & hold", "R1 Smoothed", "R3 Real-time", "Vol placebo", "MA200"]:
                x = (b["ret"][k].net if k != "R1 Smoothed" else b["ret"][k].gross).reindex(ix).fillna(0)
                row[k] = float(np.prod(1 + x) - 1)
            prt = b["P"].p_rt_pred.reindex(ix).values > 0.5
            ps = b["P"].p_smooth.reindex(ix).values > 0.5
            p = b["df"].p.reindex(ix).values
            fr, fs = np.flatnonzero(prt), np.flatnonzero(ps)
            row["rt_first_out_day"] = int(fr[0]) if len(fr) else np.nan
            row["hindsight_first_out_day"] = int(fs[0]) if len(fs) else np.nan
            row["index_move_before_rt_exit"] = float(p[fr[0]] / p[0] - 1) if len(fr) else np.nan
            row["rt_share_out"] = float(prt.mean())
            row["hindsight_share_out"] = float(ps.mean())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "stress_windows.csv", index=False)
    return df


# --------------------------------------------------------------------------- 5. beyond volatility
def ols_nw(y, X, lags):
    X = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ b
    n = len(y)
    XtXi = np.linalg.inv(X.T @ X)
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lags + 1):
        G = Xu[L:].T @ Xu[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    V = XtXi @ S @ XtXi
    return b, np.sqrt(np.diag(V))


def regressions(h: int = 21):
    """Does the real-time turbulence probability forecast the next month beyond EWMA volatility?
    y_ret: next-h-day excess return / (EWMA daily vol at t * sqrt(h))  -> risk-adjusted return forecast
    y_vol: log realised vol over the next h days                       -> risk forecast
    regressors: P_rt(turbulent at t), log EWMA vol at t. Newey-West with 2h lags."""
    rows = []
    for n in INDICES:
        b = build(n)
        df = b["df"]
        ix = b["pos"].index
        ex = (df.r - df.rf)
        fwd = ex[::-1].rolling(h).sum()[::-1].shift(-1)
        rv = np.sqrt((df.r ** 2)[::-1].rolling(h).sum()[::-1].shift(-1) * TDAYS / h)
        ev = b["vol"]
        d = pd.DataFrame({"p": b["P"].p_rt_filt, "lv": np.log(ev), "yret": fwd / (ev / np.sqrt(TDAYS) * np.sqrt(h)),
                          "yvol": np.log(rv)}).reindex(ix).dropna()
        d = d[np.isfinite(d).all(axis=1)]
        for y in ("yret", "yvol"):
            for spec, X in [("P only", d[["p"]].values), ("P + log vol", d[["p", "lv"]].values),
                            ("log vol only", d[["lv"]].values)]:
                bb, se = ols_nw(d[y].values, X, 2 * h)
                for j, nm in enumerate(["p", "lv"] if spec == "P + log vol" else (["p"] if spec == "P only" else ["lv"])):
                    rows.append(dict(index=n, target=y, spec=spec, coef_name=nm, coef=bb[j + 1], se=se[j + 1],
                                     t=bb[j + 1] / se[j + 1], p=2 * (1 - norm.cdf(abs(bb[j + 1] / se[j + 1]))),
                                     n=len(d)))
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for _, g in df.groupby(["target", "spec", "coef_name"]):
        df.loc[g.index, "p_holm"] = holm(g.p.values)
    df.to_csv(RES / "regressions.csv", index=False)
    return df


def belief_vs_reality():
    """On the days the real-time rule is out, the model expects a strongly negative return (that is why it
    is out). What did those days actually earn? Means with Newey-West standard errors (10 lags)."""
    rows = []
    for n in INDICES:
        b = build(n)
        ix = b["pos"].index
        held_out = b["pos"]["R3 Real-time"].shift(1) == 0
        er = b["P"].er_rt.shift(1).reindex(ix) / 100          # model's predicted mean for that day (log, ~simple)
        ex = (b["df"].r - b["df"].rf).reindex(ix)
        for lab, m in [("real-time out", held_out), ("real-time in", b["pos"]["R3 Real-time"].shift(1) == 1)]:
            x = ex[m].dropna().values
            se = nw_se_mean(x, 10)
            rows.append(dict(index=n, days=lab, n_days=len(x), model_expected_ann=float(er[m].mean() * TDAYS),
                             realised_ann=float(x.mean() * TDAYS), ci_lo=float((x.mean() - 1.96 * se) * TDAYS),
                             ci_hi=float((x.mean() + 1.96 * se) * TDAYS), realised_vol=float(x.std() * np.sqrt(TDAYS)),
                             p_one_sided_mean_le_0=float(1 - norm.cdf(x.mean() / se))))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "belief_vs_reality.csv", index=False)
    return df


# --------------------------------------------------------------------------- 6. BIC and parameter stability
def bic_table():
    rows = []
    closes = load_closes()
    for n in INDICES:
        x = load_index(n, closes).x.values
        for K in (1, 2, 3, 4):
            if K == 1:
                ll = float(np.sum(-0.5 * (np.log(2 * np.pi * x.var()) + (x - x.mean()) ** 2 / x.var())))
                rows.append(dict(index=n, K=1, loglik=ll, bic=-2 * ll + 2 * np.log(len(x))))
                continue
            p = hmm.fit(x, K, n_starts=10, rng=np.random.default_rng(K))
            rows.append(dict(index=n, K=K, loglik=p.loglik, bic=hmm.bic(p, len(x))))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "bic.csv", index=False)
    return df


def stability():
    """Parameters of every real-time refit, and how full-sample parameters compare."""
    rows, full = [], []
    for n in INDICES:
        b = build(n)
        ix = b["df"].index
        dfi = load_index(n)
        for tau, p in b["fits"]:
            rows.append(dict(index=n, date=dfi.index[tau].date(), mu_calm=p.mu[0] * TDAYS, mu_turb=p.mu[-1] * TDAYS,
                             vol_calm=np.sqrt(p.var[0] * TDAYS), vol_turb=np.sqrt(p.var[-1] * TDAYS),
                             dur_calm=p.expected_duration()[0], dur_turb=p.expected_duration()[-1],
                             stat_turb=p.stationary()[-1]))
        p = b["full"]
        full.append(dict(index=n, mu_calm=p.mu[0] * TDAYS, mu_turb=p.mu[-1] * TDAYS, vol_calm=np.sqrt(p.var[0] * TDAYS),
                         vol_turb=np.sqrt(p.var[-1] * TDAYS), dur_calm=p.expected_duration()[0],
                         dur_turb=p.expected_duration()[-1], stat_turb=p.stationary()[-1]))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "realtime_fits.csv", index=False)
    f = pd.DataFrame(full)
    s = df.groupby("index").agg(share_turb_mean_negative=("mu_turb", lambda v: float((v < 0).mean())),
                                share_turb_below_calm=("mu_turb", lambda v: np.nan))
    for n in INDICES:
        d = df[df["index"] == n]
        s.loc[n, "share_turb_below_calm"] = float((d.mu_turb < d.mu_calm).mean())
        s.loc[n, "mu_turb_p10"] = d.mu_turb.quantile(0.1)
        s.loc[n, "mu_turb_p90"] = d.mu_turb.quantile(0.9)
        s.loc[n, "dur_turb_p10"] = d.dur_turb.quantile(0.1)
        s.loc[n, "dur_turb_p90"] = d.dur_turb.quantile(0.9)
    f = f.merge(s.reset_index(), on="index")
    f.to_csv(RES / "full_sample_params.csv", index=False)
    return f


# --------------------------------------------------------------------------- 7. robustness
VARIANTS = {
    "Main (K=2, monthly refit, expanding)": dict(),
    "K=3 states (out if P(most turbulent) > 0.5)": dict(K=3),
    "Quarterly refit": dict(refit=63),
    "Annual refit": dict(refit=252),
    "Rolling 10-year window": dict(window=2520),
}


def robustness():
    rows = []
    for vname, kw in VARIANTS.items():
        for n in INDICES:
            b = build(n, **kw)
            bh = sharpe(b["ret"]["Buy & hold"].net.values)
            for k in LADDER + ["Vol placebo", "HMM weight", "HMM mean rule"]:
                st = b["ret"][k]
                x = st.gross if k in ("R0 Hindsight", "R1 Smoothed", "R2 Filtered") else st.net
                rows.append(dict(variant=vname, index=n, strategy=k, sharpe=float(sharpe(x.values)),
                                 minus_bh=float(sharpe(x.values) - bh), turnover=st.turnover, exposure=st.exposure))
    # costs
    b0 = {n: build(n) for n in INDICES}
    for c in (0, 10, 20):
        for n in INDICES:
            st = b0[n]["ret"]["R3 Real-time"]
            x = st.gross
            pos_l = b0[n]["pos"]["R3 Real-time"].shift(1).reindex(x.index)
            net = x - pos_l.diff().abs().fillna(0) * c / 1e4
            bh = sharpe(b0[n]["ret"]["Buy & hold"].net.values)
            rows.append(dict(variant=f"Cost {c} bp", index=n, strategy="R3 Real-time", sharpe=float(sharpe(net.values)),
                             minus_bh=float(sharpe(net.values) - bh), turnover=st.turnover, exposure=st.exposure))
    # dividends (S&P 500 only)
    b = build("GSPC", total_return=True)
    bh = sharpe(b["ret"]["Buy & hold"].net.values)
    for k in LADDER + ["Vol placebo"]:
        st = b["ret"][k]
        x = st.gross if k in ("R0 Hindsight", "R1 Smoothed", "R2 Filtered") else st.net
        rows.append(dict(variant="S&P 500 with dividends", index="GSPC", strategy=k, sharpe=float(sharpe(x.values)),
                         minus_bh=float(sharpe(x.values) - bh), turnover=st.turnover, exposure=st.exposure))
    # common modern sample 2001-02 .. 2026 for every market
    for n in INDICES:
        b = b0[n]
        m = b["ret"]["Buy & hold"].net.index >= "2001-03-01"
        bh = sharpe(b["ret"]["Buy & hold"].net[m].values)
        for k in LADDER + ["Vol placebo"]:
            st = b["ret"][k]
            x = (st.gross if k in ("R0 Hindsight", "R1 Smoothed", "R2 Filtered") else st.net)[m]
            rows.append(dict(variant="Common sample 2001-2026", index=n, strategy=k, sharpe=float(sharpe(x.values)),
                             minus_bh=float(sharpe(x.values) - bh), turnover=np.nan, exposure=np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "robustness.csv", index=False)
    summ = (df.groupby(["variant", "strategy"], sort=False)
              .agg(median_minus_bh=("minus_bh", "median"), n_above_bh=("minus_bh", lambda v: int((v > 0).sum())),
                   n=("minus_bh", "size"), median_turnover=("turnover", "median")).reset_index())
    summ.to_csv(RES / "robustness_summary.csv", index=False)
    return df


# --------------------------------------------------------------------------- 8. model world
SIM_PATHS = 40
SIM_REFIT = 252


def _sim_one(args):
    n, p, T, seed = args
    rng = np.random.default_rng(seed)
    x, s = hmm.simulate(p, T, rng)
    r = pd.Series(np.expm1(x / 100))
    full = hmm.fit(x, 2, n_starts=3, rng=rng)
    gam = hmm.smoothed(x, full)
    ff, _ = hmm.filtered(x, full)
    rt = realtime(x, 2, refit=SIM_REFIT, seed=seed)
    P = pd.DataFrame({"s": gam[:, 1], "sp": (gam @ full.A)[:, 1], "fp": (ff @ full.A)[:, 1], "rp": rt.pred[:, 1]})
    pos = pd.DataFrame({"Oracle (true state)": pd.Series((s == 0).astype(float)).shift(-1),
                        "R0 Hindsight": (P.s.shift(-1) < .5).astype(float),
                        "R1 Smoothed": (P.sp < .5).astype(float), "R2 Filtered": (P.fp < .5).astype(float),
                        "R3 Real-time": (P.rp < .5).astype(float).where(P.rp.notna()), "Buy & hold": 1.0})
    vol_ = ewma_vol(r)
    pos["Vol placebo"] = matched_vol_placebo(pos["R3 Real-time"], vol_)
    ok = pos.notna().all(axis=1)
    pos = pos.loc[ok.idxmax():].iloc[:-1]
    out = dict(index=n, seed=seed)
    rr = r.reindex(pos.index)
    for c in pos.columns:
        pl = pos[c].shift(1)
        g = (pl * rr).iloc[1:]
        net = g - pl.diff().abs().iloc[1:].fillna(0) * 5 / 1e4
        out[c] = float(sharpe((g if c.startswith(("Oracle", "R0", "R1", "R2")) else net).values))
    # detection lag in the model world against the TRUE state
    ix = pos.index
    true_t = (s[ix] == 1)
    prt = P.rp.reindex(ix).values > 0.5
    lags = []
    for a, e in spells(true_t, 21):
        if a < 5:
            continue
        w = np.flatnonzero(prt[a - 5:a + 126])
        if len(w):
            lags.append(a - 5 + w[0] - a)
    out["median_lag_true"] = float(np.median(lags)) if lags else np.nan
    return out


def model_world():
    rows, args = [], []
    for n in INDICES:
        b = build(n)
        T = len(load_index(n))
        for i in range(SIM_PATHS):
            args.append((n, b["full"], T, 10_000 * INDICES.index(n) + i))
    with Pool(4) as pool:
        rows = pool.map(_sim_one, args, chunksize=2)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "model_world.csv", index=False)
    return df


def model_vs_real():
    sim = pd.read_csv(RES / "model_world.csv")
    perf = pd.read_csv(RES / "performance.csv")
    det = pd.read_csv(RES / "detection.csv")
    rows = []
    for n in INDICES:
        g = sim[sim["index"] == n]
        pr = perf[perf["index"] == n].set_index(["strategy", "returns"]).sharpe
        real = {"R1 Smoothed": pr[("R1 Smoothed", "gross")], "R2 Filtered": pr[("R2 Filtered", "gross")],
                "R3 Real-time": pr[("R3 Real-time", "net")], "Vol placebo": pr[("Vol placebo", "net")]}
        bh = pr[("Buy & hold", "net")]
        for k, v in real.items():
            d_sim = g[k] - g["Buy & hold"]
            rows.append(dict(index=n, strategy=k, real_minus_bh=v - bh, sim_median_minus_bh=d_sim.median(),
                             sim_p05=d_sim.quantile(0.05), sim_p95=d_sim.quantile(0.95),
                             real_percentile_in_sim=float((d_sim < v - bh).mean())))
        d = g["R3 Real-time"] - g["Vol placebo"]
        rv = real["R3 Real-time"] - real["Vol placebo"]
        rows.append(dict(index=n, strategy="R3 minus vol placebo", real_minus_bh=rv, sim_median_minus_bh=d.median(),
                         sim_p05=d.quantile(0.05), sim_p95=d.quantile(0.95), real_percentile_in_sim=float((d < rv).mean())))
        rows.append(dict(index=n, strategy="median detection lag (days)",
                         real_minus_bh=det[det["index"] == n].lag_days.median(),
                         sim_median_minus_bh=g.median_lag_true.median(), sim_p05=g.median_lag_true.quantile(0.05),
                         sim_p95=g.median_lag_true.quantile(0.95), real_percentile_in_sim=np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "model_vs_real.csv", index=False)
    return df


def main(parts=None):
    RES.mkdir(exist_ok=True)
    t0 = time.time()
    steps = dict(build=lambda: build_all([dict(v) for v in VARIANTS.values()] + [dict(total_return=True, _only=["GSPC"])]),
                 perf=performance, ladder=ladder_tests, attribution=attribution, detection=detection, stress=stress,
                 regressions=regressions, belief=belief_vs_reality, bic=bic_table, stability=stability, robustness=robustness,
                 model=model_world, model_vs_real=model_vs_real)
    for k, f in steps.items():
        if parts and k not in parts:
            continue
        f()
        print(f"{k:14s} done  {time.time() - t0:6.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
