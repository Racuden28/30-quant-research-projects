"""
Every table in the README. Run after backtest.py.

Part A  Forecasting: out-of-sample hedge loss of each estimator, next month (daily returns) and next
        12 months (monthly returns); Diebold-Mariano tests vs the textbook beta with Newey-West
        errors and Holm correction; a Model Confidence Set; stationary-bootstrap CIs; subperiods,
        stress windows, the concentration of the advantage in high-volatility months, and
        Mincer-Zarnowitz calibration slopes.
Part B  Pricing: does the security market line become steep when betas are forecast better?
        Quintile sorts, post-ranking betas, SML slope relative to the market premium, Fama-MacBeth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from estimators import BASELINES, ESTIMATORS, TEXTBOOK

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MODELS = ESTIMATORS + BASELINES
SEED = 20261121
N_BOOT = 2000
BLOCK = 12

SUBPERIODS = {"1929-1962": ("1929-01", "1962-12"), "1963-1999": ("1963-01", "1999-12"),
              "2000-2026": ("2000-01", "2026-12")}
STRESS = {"Crash & Depression 1929-32": ("1929-10", "1932-06"), "Oct 1987": ("1987-10", "1987-12"),
          "Dot-com bust 2000-02": ("2000-03", "2002-10"), "GFC 2008-09": ("2008-09", "2009-03"),
          "COVID 2020": ("2020-02", "2020-04"), "Inflation bear 2022": ("2022-01", "2022-10")}


# ------------------------------------------------------------------ loading and losses
def load(path=RES / "forecast_panel.csv.gz") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=[0, 1])
    df = df.reset_index()
    df["eval_month"] = (pd.PeriodIndex(df["month"], freq="M") + 1).astype(str)
    return df


def common_sample(df: pd.DataFrame, horizon: str = "d") -> pd.DataFrame:
    need = [f"beta_{m}" for m in MODELS] + [f"{horizon}_n"]
    return df.dropna(subset=need).copy()


def losses(df: pd.DataFrame, horizon: str = "d") -> pd.DataFrame:
    """Per-row hedge loss (in %^2 per period) of every model plus the zero hedge and the ex-post oracle."""
    n, sxx, sxy, syy = (df[f"{horizon}_{k}"].to_numpy() for k in ("n", "sxx", "sxy", "syy"))
    out = {}
    for m in MODELS:
        b = df[f"beta_{m}"].to_numpy()
        out[m] = (syy - 2 * b * sxy + b * b * sxx) / n * 1e4
    out["No hedge"] = syy / n * 1e4
    b_star = sxy / sxx
    out["Oracle"] = (syy - b_star * sxy) / n * 1e4
    L = pd.DataFrame(out, index=df.index)
    L["eval_month"] = df["eval_month"].to_numpy()
    L["mkt_var"] = sxx / n * 1e4
    return L


def monthly(L: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional average loss per evaluation month (each month weighs the same)."""
    return L.groupby("eval_month").mean(numeric_only=True)


# ------------------------------------------------------------------ statistics
def nw_tstat(d: np.ndarray, lags: int) -> tuple[float, float]:
    d = np.asarray(d, float)
    T = len(d)
    u = d - d.mean()
    s = u @ u / T
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / T
    se = np.sqrt(s / T)
    t = d.mean() / se
    return t, 2 * stats.norm.sf(abs(t))


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[k]))
        adj[k] = running
    return pd.Series(adj)[p.index]


def stationary_bootstrap_idx(T: int, n_boot: int, block: int, rng) -> np.ndarray:
    p = 1 / block
    idx = np.empty((n_boot, T), dtype=int)
    idx[:, 0] = rng.integers(0, T, n_boot)
    for t in range(1, T):
        new = rng.random(n_boot) < p
        idx[:, t] = np.where(new, rng.integers(0, T, n_boot), (idx[:, t - 1] + 1) % T)
    return idx


def model_confidence_set(Lm: pd.DataFrame, models, alpha=0.10, n_boot=N_BOOT, block=BLOCK, seed=SEED):
    """Hansen-Lunde-Nason MCS with the T_max statistic on the monthly loss series."""
    rng = np.random.default_rng(seed)
    X = Lm[list(models)].to_numpy()
    T = len(X)
    idx = stationary_bootstrap_idx(T, n_boot, block, rng)
    alive = list(range(len(models)))
    pvals = {}
    running_p = 0.0
    while len(alive) > 1:
        Z = X[:, alive]
        dbar_i = Z.mean(0) - Z.mean()                                  # loss of i vs average
        boot = Z[idx].mean(1)                                           # n_boot x m
        boot_d = boot - boot.mean(1, keepdims=True)
        var = ((boot_d - dbar_i) ** 2).mean(0)
        t_i = dbar_i / np.sqrt(var)
        tmax = t_i.max()
        tmax_boot = ((boot_d - dbar_i) / np.sqrt(var)).max(1)
        p = (tmax_boot >= tmax).mean()
        worst = alive[int(np.argmax(t_i))]
        running_p = max(running_p, p)
        pvals[models[worst]] = running_p
        if p >= alpha:
            break
        alive.remove(worst)
    for a in alive:
        pvals.setdefault(models[a], 1.0)
    in_set = {m: (pvals[m] >= alpha) for m in models}
    return pd.Series(pvals)[list(models)], pd.Series(in_set)[list(models)]


def forecast_table(df: pd.DataFrame, horizon: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cs = common_sample(df, horizon)
    L = losses(cs, horizon)
    Lm = monthly(L)
    lags = 6 if horizon == "d" else 18
    tb = Lm[TEXTBOOK]
    ex = Lm[MODELS].sub(Lm["Oracle"], axis=0)          # beta-error part of the loss
    rng = np.random.default_rng(SEED)
    idx = stationary_bootstrap_idx(len(Lm), N_BOOT, BLOCK if horizon == "d" else 24, rng)
    rows = []
    for m in MODELS:
        d = (Lm[m] - tb).to_numpy()
        t, p = nw_tstat(d, lags) if m != TEXTBOOK else (np.nan, np.nan)
        ratio = ex[m].sum() / ex[TEXTBOOK].sum()
        br = ex[m].to_numpy()[idx].sum(1) / ex[TEXTBOOK].to_numpy()[idx].sum(1)
        # Mincer-Zarnowitz with hedge weights: regress realised beta on forecast, weights = sxx
        b = cs[f"beta_{m}"].to_numpy()
        w = cs[f"{horizon}_sxx"].to_numpy()
        bstar = (cs[f"{horizon}_sxy"] / cs[f"{horizon}_sxx"]).to_numpy()
        if np.ptp(b) > 0:
            W = w / w.sum()
            mb, ms = W @ b, W @ bstar
            mz = (W @ ((b - mb) * (bstar - ms))) / (W @ ((b - mb) ** 2))
        else:
            mz = np.nan
        bias = (w @ (b - bstar)) / w.sum()
        rows.append(dict(model=m, loss=Lm[m].mean(), hedge_eff=1 - Lm[m].mean() / Lm["No hedge"].mean(),
                         loss_vs_textbook_pct=100 * (Lm[m].mean() / tb.mean() - 1),
                         excess_loss=ex[m].mean(), excess_ratio=ratio,
                         ratio_lo=np.quantile(br, 0.025), ratio_hi=np.quantile(br, 0.975),
                         dm_t=t, dm_p=p, mz_slope=mz, bias=bias))
    out = pd.DataFrame(rows).set_index("model")
    out["dm_p_holm"] = holm(out["dm_p"].drop(TEXTBOOK)).reindex(out.index)
    mcs_p, in_mcs = model_confidence_set(Lm, MODELS)
    out["mcs_p"], out["in_mcs90"] = mcs_p, in_mcs
    out["oracle_loss"] = Lm["Oracle"].mean()
    out["months"] = len(Lm)
    out["asset_months"] = len(L)
    return out, Lm


def window_table(Lm: pd.DataFrame, windows: dict) -> pd.DataFrame:
    ex = Lm[MODELS].sub(Lm["Oracle"], axis=0)
    rows = []
    for name, (a, b) in windows.items():
        sl = (Lm.index >= a) & (Lm.index <= b)
        e = ex[sl]
        r = {"window": name, "months": int(sl.sum()), "mkt_vol_ann": np.sqrt(Lm.loc[sl, "mkt_var"].mean() * 252)}
        for m in MODELS:
            r[m] = e[m].sum() / e[TEXTBOOK].sum()
        rows.append(r)
    return pd.DataFrame(rows).set_index("window")


def concentration(Lm: pd.DataFrame) -> pd.DataFrame:
    """Share of each estimator's cumulative advantage over the textbook earned in the most volatile months."""
    order = Lm["mkt_var"].sort_values(ascending=False).index
    rows = []
    for m in MODELS:
        if m == TEXTBOOK:
            continue
        adv = (Lm[TEXTBOOK] - Lm[m]).loc[order]
        tot = adv.sum()
        k10 = int(np.ceil(0.10 * len(adv)))
        k05 = int(np.ceil(0.05 * len(adv)))
        rows.append(dict(model=m, total_advantage=tot, share_top5pct=adv.iloc[:k05].sum() / tot,
                         share_top10pct=adv.iloc[:k10].sum() / tot,
                         share_bottom50pct=adv.iloc[len(adv) // 2:].sum() / tot,
                         win_rate=(adv > 0).mean()))
    return pd.DataFrame(rows).set_index("model")


# ------------------------------------------------------------------ Part B: the security market line
def sml_tables(df: pd.DataFrame, n_groups: int = 5):
    cs = common_sample(df, "d").dropna(subset=["r_next", "mkt_next"])
    mk = cs.groupby("eval_month")["mkt_next"].first()
    prem = mk.mean()
    rng = np.random.default_rng(SEED)
    idx = stationary_bootstrap_idx(len(mk), N_BOOT, BLOCK, rng)
    q_rows, t_rows, series = [], [], {}
    for m in ESTIMATORS:
        g = cs[["eval_month", "asset", f"beta_{m}", "r_next"]].copy()
        g["q"] = g.groupby("eval_month")[f"beta_{m}"].transform(
            lambda s: pd.qcut(s.rank(method="first"), n_groups, labels=False))
        port = g.groupby(["eval_month", "q"])["r_next"].mean().unstack().reindex(mk.index)
        ex_ante = g.groupby(["eval_month", "q"])[f"beta_{m}"].mean().unstack().mean()
        X = np.column_stack([np.ones(len(mk)), mk.to_numpy()])
        coef = np.linalg.lstsq(X, port.to_numpy(), rcond=None)[0]
        post_beta, alpha = coef[1], coef[0]
        mean_ret = port.mean().to_numpy()
        for q in range(n_groups):
            q_rows.append(dict(model=m, quintile=q + 1, ex_ante_beta=ex_ante.iloc[q], post_beta=post_beta[q],
                               mean_excess_ann=12 * mean_ret[q], alpha_ann=12 * alpha[q]))
        # SML slope from the post-ranking betas, relative to the market premium (CAPM: ratio = 1)
        hl = (port[n_groups - 1] - port[0]).to_numpy()
        spread = post_beta[-1] - post_beta[0]
        slope = hl.mean() / spread
        bh = hl[idx].mean(1)
        bm = mk.to_numpy()[idx].mean(1)
        # bootstrap re-estimates the post-ranking spread too
        P = port.to_numpy()
        bs = []
        for r in range(N_BOOT):
            ii = idx[r]
            xx = mk.to_numpy()[ii]
            c = np.cov(np.column_stack([P[ii, 0], P[ii, -1], xx]).T)
            sp = c[1, 2] / c[2, 2] - c[0, 2] / c[2, 2]
            bs.append((bh[r] / sp) / bm[r])
        bs = np.array(bs)
        # beta-neutral long-short (Frazzini-Pedersen style lever to the market): HL - spread * MKT
        neutral = hl - spread * mk.to_numpy()
        tn, pn = nw_tstat(neutral, 6)
        # Fama-MacBeth
        gam = g.groupby("eval_month").apply(
            lambda s: pd.Series(np.polyfit(s[f"beta_{m}"], s["r_next"], 1), index=["g1", "g0"]))
        gam = gam.reindex(mk.index)
        t1, _ = nw_tstat(gam["g1"].to_numpy(), 6)
        t0, _ = nw_tstat(gam["g0"].to_numpy(), 6)
        tdiff, pdiff = nw_tstat((gam["g1"] - mk).to_numpy(), 6)
        t_rows.append(dict(model=m, post_beta_spread=spread, hl_ann=12 * hl.mean(),
                           sml_slope_ann=12 * slope, mkt_prem_ann=12 * prem, slope_ratio=slope / prem,
                           ratio_lo=np.quantile(bs, 0.025), ratio_hi=np.quantile(bs, 0.975),
                           neutral_alpha_ann=-12 * neutral.mean(), neutral_t=-tn,
                           fm_g1_ann=12 * gam["g1"].mean(), fm_g1_t=t1, fm_g0_ann=12 * gam["g0"].mean(),
                           fm_g0_t=t0, fm_g1_minus_mkt_t=tdiff, fm_g1_minus_mkt_p=pdiff))
        series[m] = hl
    return pd.DataFrame(q_rows), pd.DataFrame(t_rows).set_index("model")


def main():
    df = load()
    fd, Lm_d = forecast_table(df, "d")
    fm, Lm_m = forecast_table(df, "m")
    fd.to_csv(RES / "forecast_daily_next_month.csv", float_format="%.6g")
    fm.to_csv(RES / "forecast_monthly_next_year.csv", float_format="%.6g")
    Lm_d.to_csv(RES / "monthly_losses_daily.csv", float_format="%.6g")
    Lm_m.to_csv(RES / "monthly_losses_12m.csv", float_format="%.6g")
    window_table(Lm_d, SUBPERIODS).to_csv(RES / "subperiods_daily.csv", float_format="%.4f")
    window_table(Lm_m, SUBPERIODS).to_csv(RES / "subperiods_12m.csv", float_format="%.4f")
    window_table(Lm_d, STRESS).to_csv(RES / "stress_windows.csv", float_format="%.4f")
    concentration(Lm_d).to_csv(RES / "concentration.csv", float_format="%.4f")
    q, t = sml_tables(df)
    q.to_csv(RES / "sml_quintiles.csv", index=False, float_format="%.5f")
    t.to_csv(RES / "sml_tests.csv", float_format="%.5f")
    pd.set_option("display.width", 250)
    print("== next month, daily hedge ==\n", fd.round(4).to_string())
    print("== next 12 months, monthly hedge ==\n", fm.round(4).to_string())
    print(window_table(Lm_d, STRESS).round(3).to_string())
    print(window_table(Lm_d, SUBPERIODS).round(3).to_string())
    print(window_table(Lm_m, SUBPERIODS).round(3).to_string())
    print(concentration(Lm_d).round(3).to_string())
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
