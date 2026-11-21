"""Pre-registered out-of-sample audit of 14 terminal widgets (see PREREGISTRATION.md).

    python src/audit.py           # writes results/*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import norm

from stats import bh, hac_mean_t, cw_series, cw_test, expanding_oos
from widgets import (ASSETS, FAMILY, M_TARGETS, M_WIDGETS, ROOT, TARGETS, WIDGETS, compute_targets,
                     compute_widgets, load_returns, load_shiller, monthly_targets, monthly_widgets)

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
TRAIN, STEP, Q = 1260, 21, 0.10
STRESS = {"GFC 2008-09": ("2008-09-01", "2009-03-31"), "COVID 2020": ("2020-02-15", "2020-05-31"),
          "Rates 2022": ("2022-01-01", "2022-10-31")}
RNG = np.random.default_rng(10)


def panel(h: int):
    """Per asset: widgets, targets and the OOS start index."""
    R = load_returns()
    out = {}
    for a in ASSETS:
        r = R[a].dropna()
        W, Y = compute_widgets(r), compute_targets(r, h)
        first = int(np.argmax(W.notna().all(axis=1).values))
        out[a] = (W, Y, first + TRAIN)
    return out


def run_asset(W, Y, start, h, targets=TARGETS):
    """Forecast frames for one asset: {(target, test, widget): (y, f_model, f_base)} restricted to OOS rows."""
    n = len(W)
    oos = np.zeros(n, bool)
    oos[start:] = True
    oos &= W.notna().all(axis=1).values
    res = {}
    for tg in targets:
        y = Y[tg].values
        base_mean = expanding_oos(y, None, h, start, STEP)
        lrv = np.log(W["rv21"].values)
        base_rv = expanding_oos(y, lrv, h, start, STEP)
        for w in WIDGETS + ["ALL"]:
            x = W[WIDGETS].values if w == "ALL" else W[w].values
            fm = expanding_oos(y, x, h, start, STEP)
            res[(tg, "standalone", w)] = (y, fm, base_mean)
            if tg != "ret" and w != "rv21":
                xi = np.column_stack([lrv, W[WIDGETS].drop(columns="rv21").values]) if w == "ALL" \
                    else np.column_stack([lrv, x])
                res[(tg, "incremental", w)] = (y, expanding_oos(y, xi, h, start, STEP), base_rv)
        res[(tg, "standalone", "logrv21")] = (y, base_rv, base_mean)
    for k, (y, fm, fb) in res.items():
        m = oos & np.isfinite(y) & np.isfinite(fm) & np.isfinite(fb)
        res[k] = (np.where(m, y, np.nan), np.where(m, fm, np.nan), np.where(m, fb, np.nan))
    return res


def pooled_stats(frames: dict, idx: dict, lag: int, boot: int = 0):
    """Equal-weight pooling: every asset's squared errors are scaled by its baseline MSE, then averaged by date."""
    em_list, eb_list, cw_list = [], [], []
    per_asset = {}
    for a, (y, fm, fb) in frames.items():
        mseb = np.nanmean((y - fb) ** 2)
        ix = idx[a]
        em_list.append(pd.Series((y - fm) ** 2 / mseb, ix))
        eb_list.append(pd.Series((y - fb) ** 2 / mseb, ix))
        cw_list.append(pd.Series(cw_series(y, fm, fb) / mseb, ix))
        m = np.isfinite(y)
        t, p = cw_test(cw_series(y, fm, fb)[m], lag)
        per_asset[a] = (1 - np.nansum((y - fm) ** 2) / np.nansum((y - fb) ** 2), t, p, int(m.sum()))
    EM = pd.concat(em_list, axis=1).mean(axis=1).dropna()
    EB = pd.concat(eb_list, axis=1).mean(axis=1).dropna()
    CW = pd.concat(cw_list, axis=1).mean(axis=1).dropna()
    r2 = 1 - EM.sum() / EB.sum()
    t, p = cw_test(CW.values, lag)
    _, dm_t = hac_mean_t((EB - EM).values, lag)  # Diebold–Mariano on the pooled, scaled loss difference
    dm_p = 1 - norm.cdf(dm_t)
    lo = hi = np.nan
    if boot:
        em, eb = EM.values, EB.values
        n, B = len(em), 63
        nb = int(np.ceil(n / B))
        draws = []
        for _ in range(boot):
            st = RNG.integers(0, n - B, nb)
            ii = (st[:, None] + np.arange(B)).ravel()[:n]
            draws.append(1 - em[ii].sum() / eb[ii].sum())
        lo, hi = np.percentile(draws, [5, 95])
    return dict(r2os=r2, cw_t=t, p=p, dm_t=dm_t, dm_p=dm_p, ci90_lo=lo, ci90_hi=hi, n_dates=len(EM)), per_asset, (EM, EB)


def main_h21():
    h = 21
    P = panel(h)
    runs = {a: run_asset(*P[a], h) for a in ASSETS}
    idx = {a: P[a][0].index for a in ASSETS}
    keys = list(runs["SPY"].keys())
    rows, arows, csse = [], [], {}
    for k in keys:
        frames = {a: runs[a][k] for a in ASSETS}
        s, pa, (EM, EB) = pooled_stats(frames, idx, 2 * h, boot=500)
        rows.append(dict(target=k[0], test=k[1], widget=k[2], family=FAMILY.get(k[2], k[2]), **s))
        for a, (r2, t, p, n) in pa.items():
            arows.append(dict(target=k[0], test=k[1], widget=k[2], asset=a, r2os=r2, cw_t=t, p=p, n=n))
        csse[k] = (EB - EM).cumsum()
        # stress windows (descriptive)
    pooled = pd.DataFrame(rows)
    pa = pd.DataFrame(arows)
    # pre-registered FDR families
    fam_mask = pooled.widget.isin(WIDGETS)
    for test in ["standalone", "incremental"]:
        m = fam_mask & (pooled.test == test)
        rej, adj = bh(pooled.loc[m, "p"].values, Q)
        pooled.loc[m, "p_bh"], pooled.loc[m, "survives_fdr"] = adj, rej
        rej, adj = bh(pooled.loc[m, "dm_p"].values, Q)
        pooled.loc[m, "dm_p_bh"], pooled.loc[m, "dm_survives_fdr"] = adj, rej
        m2 = pa.widget.isin(WIDGETS) & (pa.test == test)
        rej, adj = bh(pa.loc[m2, "p"].values, Q)
        pa.loc[m2, "p_bh"], pa.loc[m2, "survives_fdr"] = adj, rej
    # robustness (referee): BH within the 14 return tests alone, so that the many strong volatility rejections
    # in the pre-registered family cannot relax the threshold for the return tests
    m = fam_mask & (pooled.target == "ret")
    _, pooled.loc[m, "p_bh_return_only"] = bh(pooled.loc[m, "p"].values, Q)
    _, pooled.loc[m, "dm_p_bh_return_only"] = bh(pooled.loc[m, "dm_p"].values, Q)
    pooled.to_csv(RES / "pooled_h21.csv", index=False, float_format="%.5g")
    pa.to_csv(RES / "per_asset_h21.csv", index=False, float_format="%.5g")
    cs = pd.DataFrame({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in csse.items()})
    cs.to_csv(RES / "cumulative_sse_diff_h21.csv.gz", float_format="%.5g")

    # stress windows: pooled R²_OS inside each window
    srows = []
    for k in keys:
        if k[2] not in WIDGETS + ["ALL", "logrv21"]:
            continue
        for name, (a0, a1) in STRESS.items():
            num = den = 0.0
            for a in ASSETS:
                y, fm, fb = runs[a][k]
                ix = idx[a]
                m = (ix >= a0) & (ix <= a1) & np.isfinite(y)
                if m.sum() < 10:
                    continue
                mseb = np.nanmean((y - fb) ** 2)
                num += np.sum((y[m] - fm[m]) ** 2) / mseb
                den += np.sum((y[m] - fb[m]) ** 2) / mseb
            if den > 0:  # GFC precedes the daily OOS sample (see PREREGISTRATION.md, changes)
                srows.append(dict(target=k[0], test=k[1], widget=k[2], window=name, r2os=1 - num / den))
    pd.DataFrame(srows).to_csv(RES / "stress_h21.csv", index=False, float_format="%.5g")

    # H4: redundancy (pooled Spearman correlation, ranks within asset)
    ranks = pd.concat([P[a][0].iloc[P[a][2] - TRAIN:].rank(pct=True) for a in ASSETS])
    C = ranks.corr()
    C.to_csv(RES / "widget_rank_correlation.csv", float_format="%.4f")
    ev = np.sort(np.linalg.eigvalsh(C.values))[::-1]
    pd.DataFrame({"component": np.arange(1, len(ev) + 1), "eigenvalue": ev,
                  "share": ev / ev.sum()}).to_csv(RES / "widget_pca.csv", index=False, float_format="%.4f")

    # latest snapshot for the terminal (last close)
    snap = []
    for a in ASSETS:
        W = P[a][0]
        hist = W.iloc[P[a][2] - TRAIN:]
        last = W.iloc[-1]
        for w in WIDGETS:
            snap.append(dict(asset=a, widget=w, date=W.index[-1].date(), value=last[w],
                             pctile=(hist[w] <= last[w]).mean()))
    pd.DataFrame(snap).to_csv(RES / "terminal_snapshot.csv", index=False, float_format="%.6g")
    return runs, P


def horizon_profile():
    rows = []
    for h in [1, 5, 21, 63]:
        P = panel(h)
        runs = {a: run_asset(*P[a], h) for a in ASSETS}
        idx = {a: P[a][0].index for a in ASSETS}
        for k in runs["SPY"]:
            s, _, _ = pooled_stats({a: runs[a][k] for a in ASSETS}, idx, max(2 * h, 5))
            rows.append(dict(h=h, target=k[0], test=k[1], widget=k[2], **s))
        print("horizon", h, "done")
    pd.DataFrame(rows).to_csv(RES / "horizon_profile.csv", index=False, float_format="%.5g")


def shiller():
    r = load_shiller()
    W, Y = monthly_widgets(r), monthly_targets(r, skip=1)
    first = int(np.argmax(W.notna().all(axis=1).values))
    start, step = first + 360, 12
    eff_h = {"ret1": 2, "vol3": 4, "ret12": 13, "mdd12": 13}  # skip month + window
    rows = []
    lrv = np.log(W["rv12m"].values)
    oos = np.arange(len(W)) >= start
    for tg in M_TARGETS:
        y = Y[tg].values
        h = eff_h[tg]
        fb = expanding_oos(y, None, h, start, step)
        frv = expanding_oos(y, lrv, h, start, step)
        for w in M_WIDGETS:
            fm = expanding_oos(y, W[w].values, h, start, step)
            for test, mod, base in [("standalone", fm, fb)] + (
                    [("incremental", expanding_oos(y, np.column_stack([lrv, W[w].values]), h, start, step), frv)]
                    if tg in ("vol3", "mdd12") and w != "rv12m" else []):
                m = oos & np.isfinite(y) & np.isfinite(mod) & np.isfinite(base)
                yy, mm, bb = y[m], mod[m], base[m]
                t, p = cw_test(cw_series(yy, mm, bb), 2 * h)
                _, dmt = hac_mean_t((yy - bb) ** 2 - (yy - mm) ** 2, 2 * h)
                rows.append(dict(target=tg, test=test, widget=w, r2os=1 - np.sum((yy - mm) ** 2) / np.sum((yy - bb) ** 2),
                                 cw_t=t, p=p, dm_t=dmt, dm_p=1 - norm.cdf(dmt), n=int(m.sum()), oos_start=str(W.index[start].date())))
    df = pd.DataFrame(rows)
    for test in ["standalone", "incremental"]:
        m = df.test == test
        rej, adj = bh(df.loc[m, "p"].values, Q)
        df.loc[m, "p_bh"], df.loc[m, "survives_fdr"] = adj, rej
        rej, adj = bh(df.loc[m, "dm_p"].values, Q)
        df.loc[m, "dm_p_bh"], df.loc[m, "dm_survives_fdr"] = adj, rej
    df.to_csv(RES / "shiller_monthly.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main_h21()
    print("h21 done")
    shiller()
    print("shiller done")
    horizon_profile()
