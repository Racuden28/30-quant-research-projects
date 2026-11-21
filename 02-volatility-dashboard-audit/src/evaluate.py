"""Evaluate every forecast built by forecast.py. Writes CSV tables to results/.

    python src/evaluate.py

Pre-registered choices (fixed before looking at out-of-sample results):
* evaluation origins: every trading day from 2005-01-03 to the last day with a complete target;
* primary loss QLIKE (Patton 2011: robust to a noisy but unbiased variance proxy), secondary MSE;
* primary horizon h = 22 trading days; h = 5 and 66 are robustness checks;
* inference: Diebold-Mariano with Newey-West HAC (lag 2h), stationary bootstrap (mean block 2h,
  B = 1000, seed 2) for ratio CIs and the Model Confidence Set (alpha = 0.10, T_max statistic);
* multiple testing: Holm across all asset x model DM tests in a family;
* walk-forward selection: on the first trading day of each year from 2005, pick the specification
  with the lowest mean QLIKE over all origins whose target is already observed; hold it for the year.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimators import DEFAULT, ESTIMATORS, WEIGHTINGS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
EVAL_START = "2005-01-01"
STRESS = {"GFC 2008": ("2008-09-01", "2009-03-31"), "COVID 2020": ("2020-02-15", "2020-04-30"),
          "Inflation 2022": ("2022-01-01", "2022-10-31")}
B, SEED = 1000, 2
DEF = "|".join(DEFAULT)
BENCH = ["HAR-C2C", "HAR-GK", "GJR-GARCH-t"]


# ------------------------------------------------------------------ data
def load(h: int):
    z = np.load(RES / f"forecasts_h{h}.npz", allow_pickle=True)
    dates = pd.to_datetime(z["dates"])
    models = list(z["models"])
    assets = list(z["assets"])
    out = {}
    for a in assets:
        F, RV = z[f"F_{a}"].astype(float), z[f"RV_{a}"]
        ok = (dates >= EVAL_START) & ~np.isnan(F).any(1) & ~np.isnan(RV) & (RV > 0)
        out[a] = (dates[ok], F[ok], RV[ok])
    return out, models, assets


def qlike(rv, f):
    x = rv[:, None] / f if f.ndim == 2 else rv / f
    return x - np.log(x) - 1


def mse(rv, f):
    return ((rv[:, None] if f.ndim == 2 else rv) - f) ** 2


# ------------------------------------------------------------------ inference
def nw_tstat(d: np.ndarray, lag: int) -> tuple[float, float]:
    """Diebold-Mariano t-stat of mean(d) with Newey-West (Bartlett) HAC variance; two-sided p."""
    d = d - 0.0
    T = len(d)
    m = d.mean()
    e = d - m
    v = e @ e / T
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / T
    t = m / np.sqrt(v / T)
    return t, 2 * stats.norm.sf(abs(t))


def bootstrap_counts(T: int, block: int, rng: np.random.Generator, b: int = B) -> np.ndarray:
    """Stationary bootstrap (Politis-Romano 1994) as a B x T matrix of resampling counts."""
    p = 1.0 / block
    idx = np.empty((b, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, b)
    new = rng.random((b, T)) < p
    jumps = rng.integers(0, T, (b, T))
    for t in range(1, T):
        idx[:, t] = np.where(new[:, t], jumps[:, t], (idx[:, t - 1] + 1) % T)
    counts = np.zeros((b, T), dtype=np.float32)
    rows = np.repeat(np.arange(b), T)
    np.add.at(counts, (rows, idx.ravel()), 1.0)
    return counts / T


def mcs(L: np.ndarray, W: np.ndarray, alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Model Confidence Set (Hansen, Lunde & Nason 2011), T_max statistic.

    L: T x M losses, W: B x T bootstrap weights (rows sum to 1). Returns (MCS p-values, in-set mask)."""
    Lbar = L.mean(0)
    Lb = W @ L                                    # B x M bootstrap means
    alive = list(range(L.shape[1]))
    pvals = np.ones(L.shape[1])
    running = 0.0
    while len(alive) > 1:
        a = np.array(alive)
        d = Lbar[a] - Lbar[a].mean()
        db = (Lb[:, a] - Lb[:, a].mean(1, keepdims=True)) - d
        se = np.sqrt((db ** 2).mean(0))
        t = d / se
        tmax_b = (db / se).max(1)
        p = (tmax_b >= t.max()).mean()
        running = max(running, p)
        worst = a[np.argmax(t)]
        pvals[worst] = running
        alive.remove(worst)
    pvals[alive[0]] = 1.0
    return pvals, pvals > alpha


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    adj = pd.Series(np.nan, index=p.index)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        adj[k] = run
    return adj


# ------------------------------------------------------------------ walk-forward selection
def select_walk_forward(dates, L, models, pool: list[str], h: int) -> np.ndarray:
    """Return, per origin, the index of the model chosen by yearly walk-forward selection on QLIKE.
    The selection at year start uses origins t with t + h trading days strictly before that day."""
    idx = [models.index(m) for m in pool]
    years = dates.year
    choice = np.empty(len(dates), dtype=int)
    for y in np.unique(years):
        first = np.argmax(years == y)
        past = slice(0, max(first - h, 0))
        if first - h <= 250:        # first year: no history in the evaluation window -> use default
            pick = models.index(DEF)
        else:
            pick = idx[int(np.argmin(L[past][:, idx].mean(0)))]
        choice[years == y] = pick
    return choice


def pools(models):
    grid = [m for m in models if m.count("|") == 2 and m.split("|")[0] in ESTIMATORS]
    e0, w0, s0 = DEFAULT
    return {
        "estimator only": [f"{e}|{w0}|{s0}" for e in ESTIMATORS],
        "weighting only": [f"{e0}|{w}|{s0}" for w in WEIGHTINGS],
        "scaling only": [f"{e0}|{w0}|{s}" for s in ("sqrt", "vr")],
        "estimator + weighting": [m for m in grid if m.endswith("|sqrt")],
        "full grid": grid,
        "grid + benchmarks": grid + BENCH,
    }


# ------------------------------------------------------------------ main tables
def evaluate_h(h: int, do_mcs: bool):
    data, models, assets = load(h)
    j0 = models.index(DEF)
    rng = np.random.default_rng(SEED)
    rows, wf_rows, dm_rows, mcs_rows, stress_rows, bias_rows = [], [], [], [], [], []
    pooled_loss = []
    for a in assets:
        dates, F, RV = data[a]
        Lq, Lm = qlike(RV, F), mse(RV, F)
        T = len(RV)
        W = bootstrap_counts(T, 2 * h, rng)
        # per-model scores
        vol_err = np.abs(np.sqrt(F * 252 / h) - np.sqrt(RV[:, None] * 252 / h)).mean(0) * 100
        logbias = np.log(RV[:, None] / F).mean(0)
        for j, m in enumerate(models):
            rows.append(dict(h=h, asset=a, model=m, qlike=Lq[:, j].mean(), mse=Lm[:, j].mean(),
                             qlike_ratio=Lq[:, j].mean() / Lq[:, j0].mean(),
                             mse_ratio=Lm[:, j].mean() / Lm[:, j0].mean(),
                             vol_err_pts=vol_err[j], mean_log_rv_over_f=logbias[j]))
        pooled_loss.append(pd.DataFrame(Lq, index=dates, columns=models))
        # walk-forward selection variants
        Lb0 = W @ Lq[:, j0]
        for name, pool in pools(models).items():
            ch = select_walk_forward(dates, Lq, models, pool, h)
            lsel = Lq[np.arange(T), ch]
            lb = W @ lsel
            ratio = lsel.mean() / Lq[:, j0].mean()
            lo, hi = np.quantile(lb / Lb0, [0.025, 0.975])
            t, p = nw_tstat(lsel - Lq[:, j0], 2 * h)
            picks = pd.Series([models[c] for c in ch]).value_counts()
            wf_rows.append(dict(h=h, asset=a, variant=name, qlike_ratio=ratio, ci_lo=lo, ci_hi=hi,
                                dm_t=t, p=p, n_switches=int((np.diff(ch) != 0).sum()),
                                modal_pick=picks.index[0], modal_share=picks.iloc[0] / T))
        # DM of each model vs default (whole sample)
        for j, m in enumerate(models):
            if j == j0:
                continue
            t, p = nw_tstat(Lq[:, j] - Lq[:, j0], 2 * h)
            dm_rows.append(dict(h=h, asset=a, model=m, dm_t=t, p=p))
        # MCS
        if do_mcs:
            pool = pools(models)["grid + benchmarks"]
            idx = [models.index(m) for m in pool]
            pv, inset = mcs(Lq[:, idx], W)
            for m, p_, s in zip(pool, pv, inset):
                mcs_rows.append(dict(h=h, asset=a, model=m, mcs_p=p_, in_mcs=bool(s)))
        # stress windows
        for wname, (s, e) in STRESS.items():
            msk = (dates >= s) & (dates <= e)
            if msk.sum() < 10:
                continue
            for j, m in enumerate(models):
                stress_rows.append(dict(h=h, asset=a, window=wname, model=m, n=int(msk.sum()),
                                        qlike=Lq[msk, j].mean(), qlike_ratio=Lq[msk, j].mean() / Lq[msk, j0].mean(),
                                        mean_log_rv_over_f=np.log(RV[msk] / F[msk, j]).mean()))
        # conditional bias of the default and of the best benchmark by trailing-vol quintile
        q = pd.qcut(F[:, j0], 5, labels=False)
        for m in [DEF, "C2C|E10|sqrt", "GK|E10|sqrt", "HAR-GK", "GJR-GARCH-t"]:
            j = models.index(m)
            for k in range(5):
                s_ = q == k
                bias_rows.append(dict(h=h, asset=a, model=m, quintile=k + 1,
                                      mean_log_rv_over_f=np.log(RV[s_] / F[s_, j]).mean(),
                                      ann_vol_forecast=np.sqrt(F[s_, j].mean() * 252 / h),
                                      ann_vol_realized=np.sqrt(RV[s_].mean() * 252 / h)))
        print(f"h={h} {a} done", flush=True)

    scores = pd.DataFrame(rows)
    dm = pd.DataFrame(dm_rows)
    dm["p_holm"] = holm(dm.set_index(["asset", "model"])["p"]).to_numpy()
    wf = pd.DataFrame(wf_rows)
    wf["p_holm"] = holm(wf.set_index(["asset", "variant"])["p"]).to_numpy()
    # panel DM: average loss differential across assets on common dates
    P = pd.concat(pooled_loss, keys=assets, axis=1).dropna()
    panel = []
    for m in models:
        if m == DEF:
            continue
        d = (P.xs(m, axis=1, level=1) - P.xs(DEF, axis=1, level=1)).mean(1).to_numpy()
        t, p = nw_tstat(d, 2 * h)
        panel.append(dict(h=h, model=m, mean_ratio=(P.xs(m, axis=1, level=1).mean() /
                                                   P.xs(DEF, axis=1, level=1).mean()).mean(), dm_t=t, p=p))
    panel = pd.DataFrame(panel)
    panel["p_holm"] = holm(panel.set_index("model")["p"]).to_numpy()
    return scores, dm, wf, pd.DataFrame(mcs_rows), pd.DataFrame(stress_rows), pd.DataFrame(bias_rows), panel


def factor_effects(scores: pd.DataFrame) -> pd.DataFrame:
    """Ex-post main effects on the 120-cell grid: mean QLIKE ratio (vs default) by factor level,
    averaged over the other two factors and over assets; 'spread' = worst minus best level."""
    g = scores[scores.model.str.count(r"\|") == 2].copy()
    g = g[g.model.str.split("|").str[0].isin(ESTIMATORS)]
    g[["estimator", "weighting", "scaling"]] = g.model.str.split("|", expand=True)
    out = []
    for f in ["estimator", "weighting", "scaling"]:
        lv = g.groupby(["h", f]).qlike_ratio.mean().reset_index().rename(columns={f: "level"})
        lv["factor"] = f
        out.append(lv)
    return pd.concat(out)[["h", "factor", "level", "qlike_ratio"]]


def variance_ratio_tests() -> pd.DataFrame:
    """Lo-MacKinlay (1988) variance ratios with the heteroskedasticity-robust z*, full sample and
    split at 2013-01-01. Overlapping q-period sums of daily log returns."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from forecast import ASSETS, load as load_raw
    _, data, _ = load_raw()
    rows = []
    for a in ASSETS:
        for per, (s, e) in {"1999-2026": ("1998", "2027"), "1999-2012": ("1998", "2012-12-31"),
                            "2013-2026": ("2013", "2027")}.items():
            r = data[a].r.loc[s:e].to_numpy()
            T = len(r)
            mu = r.mean()
            e_ = r - mu
            var1 = e_ @ e_ / T
            for q in [5, 22, 66, 252]:
                cs = np.cumsum(np.concatenate([[0], r]))
                rq = cs[q:] - cs[:-q]
                varq = ((rq - q * mu) ** 2).sum() / (q * (T - q + 1) * (1 - q / T))
                vr = varq / var1
                # heteroskedasticity-consistent asymptotic variance
                theta = 0.0
                for k in range(1, q):
                    delta = (e_[k:] ** 2 * e_[:-k] ** 2).sum() / (e_ @ e_) ** 2 * T
                    theta += (2 * (q - k) / q) ** 2 * delta
                z = (vr - 1) / np.sqrt(theta / T)
                rows.append(dict(asset=a, period=per, q=q, vr=vr, z_star=z, p=2 * stats.norm.sf(abs(z))))
    return pd.DataFrame(rows)


def data_quality_by_year() -> pd.DataFrame:
    """Share of days where the open equals the high or the low, and the ratio of the mean
    Parkinson+overnight proxy to the mean squared close-to-close return, per asset and year."""
    from forecast import ASSETS, load as load_raw
    from estimators import daily_proxies
    _, data, _ = load_raw()
    rows = []
    for a in ASSETS:
        df = data[a]
        px = daily_proxies(df.r, df.o, df.u, df.d, df.c)
        g = pd.DataFrame({"open_at_extreme": ((df.u < 1e-5) | (df.d > -1e-5)).astype(float),
                          "pk": px.PK, "gk": px.GK, "c2c": px.C2C}).groupby(df.index.year)
        m = g.mean()
        for y, row in m.iterrows():
            rows.append(dict(asset=a, year=y, open_at_extreme=row.open_at_extreme,
                             pk_over_c2c=row.pk / row.c2c, gk_over_c2c=row.gk / row.c2c))
    return pd.DataFrame(rows)


def scaling_on_return_target() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The sqrt-of-time rule is a statement about the variance of the h-day *return*, not about the sum
    of squared daily returns (for which h * sigma^2 is exact by construction). So the scaling factor
    is evaluated a second time against the squared cumulative return R_{t,h}^2 = (sum r_{t+1..t+h})^2,
    an unbiased (if noisy) proxy of the h-day return variance. QLIKE stays robust to the noise
    (Patton 2011); exact zeros are dropped."""
    from forecast import load as load_raw
    _, data, _ = load_raw()
    rows, panel = [], []
    for h in [5, 22, 66]:
        z = np.load(RES / f"forecasts_h{h}.npz", allow_pickle=True)
        dates = pd.to_datetime(z["dates"])
        models = list(z["models"])
        cells = [m[:-5] for m in models if m.endswith("|sqrt") and m.split("|")[0] in ESTIMATORS]
        diffs = []
        for a in z["assets"]:
            r = data[a].r.to_numpy()
            cs = np.concatenate([[0.0], np.cumsum(r)])
            R2 = np.full(len(r), np.nan)
            R2[: len(r) - h] = (cs[1 + h:] - cs[1: len(r) + 1 - h]) ** 2
            F = z[f"F_{a}"].astype(float)
            ok = (dates >= EVAL_START) & ~np.isnan(F).any(1) & ~np.isnan(R2) & (R2 > 0)
            d_cell = []
            for c in cells:
                fs, fv = F[ok, models.index(c + "|sqrt")], F[ok, models.index(c + "|vr")]
                qs, qv = qlike(R2[ok], fs), qlike(R2[ok], fv)
                ms, mv = mse(R2[ok], fs), mse(R2[ok], fv)
                rows.append(dict(h=h, asset=a, cell=c, qlike_sqrt=qs.mean(), qlike_vr=qv.mean(),
                                 qlike_ratio_vr_over_sqrt=qv.mean() / qs.mean(), mse_ratio_vr_over_sqrt=mv.mean() / ms.mean(),
                                 realized_over_forecast_sqrt=R2[ok].mean() / fs.mean(),
                                 realized_over_forecast_vr=R2[ok].mean() / fv.mean()))
                if c == "C2C|R21":
                    t, p = nw_tstat(qv - qs, 2 * h)
                    rows[-1].update(dm_t=t, p=p)
                d_cell.append(pd.Series(qv - qs, index=dates[ok]))
            diffs.append(pd.concat(d_cell, axis=1).mean(1))
        D = pd.concat(diffs, axis=1).dropna().mean(1).to_numpy()
        t, p = nw_tstat(D, 2 * h)
        panel.append(dict(h=h, mean_qlike_diff_vr_minus_sqrt=D.mean(), dm_t=t, p=p))
    return pd.DataFrame(rows), pd.DataFrame(panel)


def main():
    all_ = {k: [] for k in ["scores", "dm", "wf", "mcs", "stress", "bias", "panel"]}
    for h in [22, 5, 66]:
        out = evaluate_h(h, do_mcs=(h == 22))
        for k, v in zip(all_, out):
            all_[k].append(v)
    for k, v in all_.items():
        pd.concat(v).to_csv(RES / f"{k}.csv", index=False, float_format="%.6g")
    scores = pd.concat(all_["scores"])
    factor_effects(scores).to_csv(RES / "factor_effects.csv", index=False, float_format="%.6g")
    variance_ratio_tests().to_csv(RES / "variance_ratios.csv", index=False, float_format="%.6g")
    sr, sp = scaling_on_return_target()
    sr.to_csv(RES / "scaling_return_target.csv", index=False, float_format="%.6g")
    sp.to_csv(RES / "scaling_return_target_panel.csv", index=False, float_format="%.6g")
    data_quality_by_year().to_csv(RES / "data_quality_by_year.csv", index=False, float_format="%.6g")
    print("tables written to", RES)


if __name__ == "__main__":
    main()
