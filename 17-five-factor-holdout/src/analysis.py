"""Every statistic in the paper. Writes results/*.csv. Run from src/: python analysis.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data as D
import stats as S

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

N_BOOT = 10_000
BLOCK = 6.0
BETA_WINDOW = 60          # months of history for out-of-sample betas
RT_START = "1983-12"      # first real-time decision (20.5 years of history)
TARGET_VOL = 0.10 / np.sqrt(12)

F = D.factors()
IS = D.window(F, D.IS_START, D.IS_END)
HO = D.window(F, D.HO_START, D.HO_END)
PERIODS = {"In-sample 1963-2013": IS, "Holdout 2014-2026": HO}


SAVE = True   # robustness.py turns this off so the main results are not overwritten


def save(df: pd.DataFrame, name: str, **kw):
    if not SAVE:
        return
    df.to_csv(RES / f"{name}.csv", float_format="%.6g", **kw)


# --------------------------------------------------------------------------------------------- premia
def premia() -> pd.DataFrame:
    rows = []
    ho_dummy = (F.index >= pd.Period(D.HO_START, "M")).astype(float)
    for f in D.FACTORS:
        row = {"factor": f}
        for lab, df in PERIODS.items():
            m, se = S.mean_nw(df[f].values)
            row[f"{lab} mean %/m"] = 100 * m
            row[f"{lab} t"] = m / se
            row[f"{lab} lo95"] = 100 * (m - 1.96 * se)
            row[f"{lab} hi95"] = 100 * (m + 1.96 * se)
            row[f"{lab} Sharpe ann"] = S.sharpe(df[f].values)
        b, se, _ = S.ols_nw(F[f].values, ho_dummy[:, None])
        row["change %/m"], row["change p"] = 100 * b[1], 2 * (1 - S.stats.norm.cdf(abs(b[1] / se[1])))
        rows.append(row)
    out = pd.DataFrame(rows).set_index("factor")
    nm = ["SMB", "HML", "RMW", "CMA", "UMD"]
    _, W, p = S.joint_shift_wald(F[nm].values, ho_dummy)
    out.attrs["joint"] = {"factors": "+".join(nm), "chi2": W, "df": len(nm), "p": p}
    pd.DataFrame([out.attrs["joint"]]).to_csv(RES / "factor_premia_joint_test.csv", index=False) if SAVE else None
    save(out, "factor_premia")
    return out


# ------------------------------------------------------------------------------------------- spanning
SPANNING_SETS = {"FF5": ["Mkt", "SMB", "HML", "RMW", "CMA"], "FF6": D.FACTORS}


def spanning_one(df: pd.DataFrame, y: str, xs: list[str]):
    b, se, e = S.ols_nw(df[y].values, df[xs].values)
    return b, se


def spanning() -> pd.DataFrame:
    rows = []
    ho = (F.index >= pd.Period(D.HO_START, "M")).astype(float)
    for setname, fs in SPANNING_SETS.items():
        for y in fs:
            if y == "Mkt":
                continue
            xs = [x for x in fs if x != y]
            row = {"set": setname, "factor": y, "regressors": "+".join(xs)}
            for lab, df in PERIODS.items():
                b, se = spanning_one(df, y, xs)
                row[f"{lab} alpha %/m"], row[f"{lab} t"] = 100 * b[0], b[0] / se[0]
                row[f"{lab} se %/m"] = 100 * se[0]
                for x, bx in zip(xs, b[1:]):
                    row[f"{lab} b_{x}"] = bx
            # pooled regression with a holdout intercept shift and holdout slope shifts
            X = F[xs].values
            Xp = np.column_stack([ho, X, X * ho[:, None]])
            b, se, _ = S.ols_nw(F[y].values, Xp)
            row["alpha change %/m"] = 100 * b[1]
            row["alpha change p"] = 2 * (1 - S.stats.norm.cdf(abs(b[1] / se[1])))
            row["holdout MDE %/m"] = 100 * S.mde(row["Holdout 2014-2026 se %/m"] / 100)
            rows.append(row)
    out = pd.DataFrame(rows)
    save(out, "spanning", index=False)
    return out


# ------------------------------------------------------------------- out-of-sample tangency (H2)
def tangency_holdout():
    w = {m: S.tangency(IS[fs].mean().values, IS[fs].cov().values, TARGET_VOL) for m, fs in D.MODELS.items()}
    ret_ho = pd.DataFrame({m: HO[fs].values @ w[m] for m, fs in D.MODELS.items()}, index=HO.index)
    ret_is = pd.DataFrame({m: IS[fs].values @ w[m] for m, fs in D.MODELS.items()}, index=IS.index)
    idx = S.stationary_bootstrap(len(HO), N_BOOT, BLOCK, seed=171)
    X = ret_ho.values
    mu = X[idx].mean(1)
    sd = X[idx].std(1, ddof=1)
    sr_boot = mu / sd * np.sqrt(12)
    sr_hat = np.array([S.sharpe(X[:, j]) for j in range(X.shape[1])])
    rows = []
    for j, m in enumerate(D.MODELS):
        fs = D.MODELS[m]
        rows.append({
            "model": m, "K": len(fs),
            "IS Sharpe": np.sqrt(12 * S.sh2(IS[fs].values)),
            "IS Sharpe adj": np.sqrt(12 * max(S.sh2_adjusted(IS[fs].values), 0)),
            "HO Sharpe": sr_hat[j], "HO lo95": np.percentile(sr_boot[:, j], 2.5),
            "HO hi95": np.percentile(sr_boot[:, j], 97.5),
            "HO Sharpe (re-estimated, in hindsight)": np.sqrt(12 * S.sh2(HO[fs].values)),
            # one-sided p that the holdout Sharpe is as high as the bias-adjusted in-sample Sharpe
            "p HO >= IS adj": float(np.mean((sr_boot[:, j] - sr_hat[j]) <= (sr_hat[j] - np.sqrt(
                12 * max(S.sh2_adjusted(IS[fs].values), 0))))),
            # two-sample version that also counts the in-sample estimate's own sampling error (Lo 2002 SEs)
            "p two-sample": float(1 - S.stats.norm.cdf(
                (np.sqrt(max(S.sh2_adjusted(IS[fs].values), 0)) - sr_hat[j] / np.sqrt(12)) /
                np.sqrt((1 + S.sh2_adjusted(IS[fs].values) / 2) / len(IS) +
                        (1 + sr_hat[j] ** 2 / 24) / len(HO)))),
            "weights": json.dumps({f: round(float(x / np.abs(w[m]).sum()), 3) for f, x in zip(fs, w[m])}),
        })
    for m in ["FF3", "FF5", "FF6"]:   # equal-weight factor portfolios: no estimated weights at all
        fs = D.MODELS[m]
        rows.append({"model": f"{m} 1/N", "K": len(fs), "IS Sharpe": S.sharpe(IS[fs].mean(1).values),
                     "IS Sharpe adj": np.nan, "HO Sharpe": S.sharpe(HO[fs].mean(1).values),
                     "HO lo95": np.percentile(HO[fs].mean(1).values[idx].mean(1) /
                                              HO[fs].mean(1).values[idx].std(1, ddof=1) * np.sqrt(12), 2.5),
                     "HO hi95": np.percentile(HO[fs].mean(1).values[idx].mean(1) /
                                              HO[fs].mean(1).values[idx].std(1, ddof=1) * np.sqrt(12), 97.5),
                     "HO Sharpe (re-estimated, in hindsight)": np.nan, "weights": "equal"})
    out = pd.DataFrame(rows).set_index("model")
    out["haircut HO/IS"] = out["HO Sharpe"] / out["IS Sharpe"]
    # pairwise differences of interest (centered bootstrap)
    names = list(D.MODELS)
    pairs = [("FF5", "FF5-HML"), ("FF5", "FF3"), ("FF6", "FF5"), ("FF6", "FF6-HML"), ("FF3", "CAPM"),
             ("FF6", "C4"), ("FF5", "CAPM")]
    prow = []
    for a, b in pairs:
        ia, ib = names.index(a), names.index(b)
        d_hat = sr_hat[ia] - sr_hat[ib]
        d_b = sr_boot[:, ia] - sr_boot[:, ib]
        prow.append({"A": a, "B": b, "HO Sharpe A-B": d_hat, "lo95": np.percentile(d_b, 2.5),
                     "hi95": np.percentile(d_b, 97.5),
                     "p two-sided": float(np.mean(np.abs(d_b - d_hat) >= abs(d_hat))),
                     "p one-sided (A>B)": float(np.mean((d_b - d_hat) >= d_hat))})
    pdiff = pd.DataFrame(prow)
    save(out, "tangency_holdout")
    save(pdiff, "tangency_holdout_diffs", index=False)
    save(ret_ho, "tangency_holdout_returns")
    return out, pdiff, ret_is, ret_ho


# --------------------------------------------------------- out-of-sample pricing errors (H3)
def oos_alphas(R: pd.DataFrame, fs: list[str], window: int = BETA_WINDOW) -> pd.DataFrame:
    """alpha_{i,t} = r_{i,t} - beta_{i,t-1}' f_t with betas from the previous `window` months."""
    Fm = F.loc[R.index, fs].values
    Rv = R.values
    T, N = Rv.shape
    out = np.full((T, N), np.nan)
    for t in range(window, T):
        X = np.column_stack([np.ones(window), Fm[t - window:t]])
        Y = Rv[t - window:t]
        ok = ~np.isnan(Y).any(0) & ~np.isnan(Rv[t])
        if not ok.any():
            continue
        B = np.linalg.lstsq(X, Y[:, ok], rcond=None)[0]
        out[t, ok] = Rv[t, ok] - Fm[t] @ B[1:]
    return pd.DataFrame(out, index=R.index, columns=R.columns)


def pricing():
    assets = {k: D.test_assets(k) for k in D.TEST_ASSETS}
    alphas = {(k, m): oos_alphas(R, fs) for k, R in assets.items() for m, fs in D.MODELS.items()}
    rows, grs_rows = [], []
    for (k, m), A in alphas.items():
        for lab, (s, e) in {"In-sample 1968-2013": ("1968-07", D.IS_END),
                            "Holdout 2014-2026": (D.HO_START, D.HO_END)}.items():
            a = D.window(A, s, e)
            r = D.window(assets[k], s, e)
            mean_a = a.mean()
            dev = (r.mean() - r.mean().mean()).abs().mean()
            # sampling variance of each mean alpha (Newey-West), to strip noise out of the squared errors
            var_a = np.array([S.mean_nw(a[c].dropna().values)[1] ** 2 for c in a.columns])
            excess = float(np.mean(mean_a.values ** 2 - var_a))
            rows.append({"assets": k, "model": m, "period": lab, "A|a| %/m": 100 * mean_a.abs().mean(),
                         "A|a|/A|r-rbar|": mean_a.abs().mean() / dev,
                         "RMS a %/m": 100 * np.sqrt((mean_a ** 2).mean()),
                         "noise RMS %/m": 100 * np.sqrt(var_a.mean()),
                         "RMS a net of noise %/m": 100 * np.sign(excess) * np.sqrt(abs(excess))})
    for k, R in assets.items():
        R = R.dropna(axis=1)
        for lab, df in PERIODS.items():
            r = R.loc[df.index]
            dev = (r.mean() - r.mean().mean()).abs().mean()
            for m, fs in D.MODELS.items():
                g, p, a, aa, s2 = S.grs(r.values, df[fs].values)
                grs_rows.append({"assets": k, "model": m, "period": lab, "GRS": g, "p": p,
                                 "A|a| %/m": 100 * aa, "A|a|/A|r-rbar|": aa / dev, "Sh2(a)": s2})
    oos = pd.DataFrame(rows)
    ins = pd.DataFrame(grs_rows)
    save(oos, "pricing_oos")
    save(ins, "pricing_grs", index=False)

    # H3: pooled 75 primary portfolios, holdout, FF5 vs FF3 (and other pairs), bootstrap over months
    def pooled(m):
        return pd.concat([D.window(alphas[(k, m)], D.HO_START, D.HO_END) for k in D.PRIMARY_ASSETS], axis=1)
    P = {m: pooled(m).values for m in D.MODELS}
    idx = S.stationary_bootstrap(len(HO), N_BOOT, BLOCK, seed=172)
    stat_b = {m: np.abs(P[m][idx].mean(1)).mean(1) for m in D.MODELS}
    stat_hat = {m: np.abs(P[m].mean(0)).mean() for m in D.MODELS}
    prow = []
    for a, b in [("FF5", "FF3"), ("FF5-HML", "FF5"), ("FF6", "FF5"), ("FF3", "CAPM"), ("C4", "FF3"),
                 ("FF5", "C4")]:
        d_hat = stat_hat[a] - stat_hat[b]
        d_b = stat_b[a] - stat_b[b]
        prow.append({"A": a, "B": b, "A|a| A-B %/m": 100 * d_hat, "lo95": 100 * np.percentile(d_b, 2.5),
                     "hi95": 100 * np.percentile(d_b, 97.5),
                     "p one-sided (A<B)": float(np.mean((d_b - d_hat) <= d_hat))})
    diffs = pd.DataFrame(prow)
    save(diffs, "pricing_oos_diffs", index=False)
    # rolling 36-month A|a| of the pooled 75 portfolios through time (for the figure)
    cum = pd.DataFrame({m: pd.concat([alphas[(k, m)] for k in D.PRIMARY_ASSETS], axis=1)
                        .rolling(36).mean().abs().mean(axis=1) for m in D.MODELS})
    save(cum, "pricing_oos_rolling36")
    return oos, ins, diffs, alphas


# ------------------------------------------------------------------ real-time selection (H4)
def real_time():
    dec = [p for p in F.index if p.month == 12 and p >= pd.Period(RT_START, "M") and p < F.index[-1]]
    rows, hist, rets = [], [], []
    for d in dec:
        past = F.loc[:d]
        crit = {m: S.sh2_adjusted(past[fs].values) for m, fs in D.MODELS.items()}
        choice = max(crit, key=crit.get)
        nxt = F.loc[d + 1:d + 12]
        w = {m: S.tangency(past[fs].mean().values, past[fs].cov().values, TARGET_VOL)
             for m, fs in D.MODELS.items()}
        for t in nxt.index:
            r = {m: float(nxt.loc[t, fs].values @ w[m]) for m, fs in D.MODELS.items()}
            r["Selector"] = r[choice]
            r["date"] = t
            r["choice"] = choice
            rets.append(r)
        # real-time spanning t-stats of HML (FF5 set and FF6 set) and of every factor in FF6
        h = {"decision": str(d), "choice": choice, **{f"crit {m}": v for m, v in crit.items()}}
        for setname, fs in SPANNING_SETS.items():
            for y in fs:
                if y == "Mkt":
                    continue
                xs = [x for x in fs if x != y]
                b, se = spanning_one(past, y, xs)
                h[f"t {y}|{setname}"] = b[0] / se[0]
                h[f"a {y}|{setname}"] = 100 * b[0]
        hist.append(h)
    R = pd.DataFrame(rets).set_index("date")
    H = pd.DataFrame(hist)
    cols = list(D.MODELS) + ["Selector"]
    idx = S.stationary_bootstrap(len(R), N_BOOT, BLOCK, seed=173)
    X = R[cols].values
    srb = X[idx].mean(1) / X[idx].std(1, ddof=1) * np.sqrt(12)
    srh = np.array([S.sharpe(X[:, j]) for j in range(len(cols))])
    summ = pd.DataFrame({"RT Sharpe": srh, "lo95": np.percentile(srb, 2.5, 0),
                         "hi95": np.percentile(srb, 97.5, 0),
                         "ann vol": X.std(0, ddof=1) * np.sqrt(12),
                         "years chosen": [int((H["choice"] == c).sum()) if c in D.MODELS else len(H)
                                          for c in cols]}, index=cols)
    js, jf = cols.index("Selector"), cols.index("FF6")
    d_hat = srh[js] - srh[jf]
    d_b = srb[:, js] - srb[:, jf]
    h4 = {"Selector-FF6 Sharpe": d_hat, "lo95": np.percentile(d_b, 2.5), "hi95": np.percentile(d_b, 97.5),
          "p one-sided (Selector<FF6)": float(np.mean((d_b - d_hat) <= d_hat))}
    # same comparisons restricted to the holdout years
    Rh = D.window(R, D.HO_START, D.HO_END)
    summ["RT Sharpe in holdout"] = [S.sharpe(Rh[c].values) for c in cols]
    save(R, "realtime_returns")
    save(H, "realtime_history", index=False)
    save(summ, "realtime_summary")
    save(pd.DataFrame([h4]), "realtime_h4", index=False)
    return R, H, summ, h4


def placebo_holdouts(R_rt: pd.DataFrame, length: int | None = None, step: int = 12):
    """Every real-time window of the holdout's length: is 2014-2026 unusual for multifactor models?"""
    length = length or len(HO)
    rows = []
    for start in range(0, len(R_rt) - length + 1, step):
        w = R_rt.iloc[start:start + length]
        rows.append({"start": str(w.index[0]), "end": str(w.index[-1]),
                     **{f"SR {m}": S.sharpe(w[m].values) for m in ["CAPM", "FF3", "FF5", "FF6", "Selector"]}})
    out = pd.DataFrame(rows)
    out["is_holdout"] = out["start"] == D.HO_START
    assert out["is_holdout"].sum() == 1, "the window grid must contain the holdout"
    for m in ["FF3", "FF5", "FF6"]:
        out[f"{m} - CAPM"] = out[f"SR {m}"] - out["SR CAPM"]
    save(out, "placebo_holdouts", index=False)
    return out


def split_holdout() -> pd.DataFrame:
    rows = []
    for lab, (s_, e_) in {"2014-2019": ("2014-01", "2019-12"), "2020-2026": ("2020-01", D.HO_END)}.items():
        df = D.window(F, s_, e_)
        for f in D.FACTORS:
            m, se = S.mean_nw(df[f].values)
            rows.append({"subperiod": lab, "factor": f, "mean %/m": 100 * m, "t": m / se})
        b, se = spanning_one(df, "HML", ["Mkt", "SMB", "RMW", "CMA"])
        rows.append({"subperiod": lab, "factor": "HML spanning alpha (FF5)", "mean %/m": 100 * b[0], "t": b[0] / se[0]})
    out = pd.DataFrame(rows)
    save(out, "holdout_split", index=False)
    return out


# ------------------------------------------------------------------------------- stress windows
def stress(R_rt: pd.DataFrame):
    rows = []
    for name, (s, e) in D.STRESS.items():
        f = D.window(F, s, e)
        r = D.window(R_rt, s, e)
        row = {"window": name, "months": len(f)}
        for x in D.FACTORS:
            row[x] = 100 * ((1 + f[x]).prod() - 1)
        for m in list(D.MODELS) + ["Selector"]:
            row[f"RT tangency {m}"] = 100 * ((1 + r[m]).prod() - 1)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("window")
    save(out, "stress_windows")
    return out


# --------------------------------------------------------------------------------- data vintage
def vintage():
    v = D.vintage_2017()
    if v is None:
        return None
    rows = []
    ff3 = D.read_raw("F-F_Research_Data_Factors.csv").rename(columns={"Mkt-RF": "Mkt"})
    now = ff3[["Mkt", "SMB", "HML"]].join(F[["UMD"]], how="inner")   # same definitions as the 2017 file
    common = v.index.intersection(now.index)
    for f in ["Mkt", "SMB", "HML", "UMD"]:
        a, b = v.loc[common, f], now.loc[common, f]
        isx = common[(common >= pd.Period(D.IS_START, "M")) & (common <= pd.Period(D.IS_END, "M"))]
        rows.append({"factor": f, "months": len(common), "first": str(common[0]), "last": str(common[-1]),
                     "corr": a.corr(b), "tracking err %/m": 100 * (a - b).std(),
                     "max |diff| %": 100 * (a - b).abs().max(),
                     "IS mean 2017 vintage %/m": 100 * v.loc[isx, f].mean(),
                     "IS mean 2026 vintage %/m": 100 * now.loc[isx, f].mean()})
    out = pd.DataFrame(rows).set_index("factor")
    save(out, "vintage_drift")
    return out


# ----------------------------------------------------------------------------------- hypotheses
def hypotheses(sp, tdiff, pdiff, h4):
    h = sp[(sp["set"] == "FF5") & (sp["factor"] == "HML")].iloc[0]
    p_h1a = 2 * (1 - S.stats.norm.cdf(abs(h["Holdout 2014-2026 t"])))
    p_h1b = h["alpha change p"]
    t = tdiff.set_index(["A", "B"]).loc[("FF5", "FF5-HML")]
    q = pdiff.set_index(["A", "B"]).loc[("FF5", "FF3")]
    rows = [
        {"H": "H1a", "claim": "HML alpha on Mkt,SMB,RMW,CMA = 0 in holdout",
         "estimate": h["Holdout 2014-2026 alpha %/m"], "unit": "%/month", "p": p_h1a,
         "supported if": "not rejected"},
        {"H": "H1b", "claim": "HML spanning alpha unchanged IS -> HO",
         "estimate": h["alpha change %/m"], "unit": "%/month", "p": p_h1b, "supported if": "not rejected"},
        {"H": "H2", "claim": "FF5-HML holdout Sharpe not below FF5",
         "estimate": -t["HO Sharpe A-B"], "unit": "Sharpe (FF5-HML minus FF5)", "p": t["p one-sided (A>B)"],
         "supported if": "not rejected"},
        {"H": "H3", "claim": "FF5 OOS A|a| below FF3 on 75 portfolios (holdout)",
         "estimate": q["A|a| A-B %/m"], "unit": "%/month (FF5 minus FF3)", "p": q["p one-sided (A<B)"],
         "supported if": "rejected"},
        {"H": "H4", "claim": "Real-time selector Sharpe not below always-FF6",
         "estimate": h4["Selector-FF6 Sharpe"], "unit": "Sharpe", "p": h4["p one-sided (Selector<FF6)"],
         "supported if": "not rejected"},
    ]
    out = pd.DataFrame(rows)
    out["p Holm"] = S.holm(out["p"].values)
    rej = out["p Holm"] < 0.05
    out["verdict"] = np.where(out["supported if"] == "rejected", np.where(rej, "supported", "not supported"),
                              np.where(rej, "rejected", "not rejected"))
    save(out, "hypotheses", index=False)
    return out


if __name__ == "__main__":
    print("sample", F.index[0], F.index[-1], "IS", len(IS), "HO", len(HO))
    pr = premia()
    print(pr.round(3).T)
    sp = spanning()
    print(sp[["set", "factor", "In-sample 1963-2013 alpha %/m", "In-sample 1963-2013 t", "Holdout 2014-2026 alpha %/m",
              "Holdout 2014-2026 t", "alpha change p", "holdout MDE %/m"]].round(3).to_string())
    tg, td, _, _ = tangency_holdout()
    print(tg.drop(columns="weights").round(3).to_string())
    print(td.round(3).to_string())
    oos, ins, pdiff, _ = pricing()
    print(oos.pivot_table(index=["assets", "model"], columns="period", values="A|a| %/m").round(3).to_string())
    print(pdiff.round(4).to_string())
    R, H, summ, h4 = real_time()
    print(summ.round(3).to_string())
    print(h4)
    print(H["choice"].value_counts())
    print(stress(R).round(2).T.to_string())
    pl = placebo_holdouts(R)
    print(pl.round(2).to_string())
    print(split_holdout().round(3).to_string())
    v = vintage()
    print(v.round(4).to_string() if v is not None else "no vintage")
    print(hypotheses(sp, td, pdiff, h4).round(4).to_string())
