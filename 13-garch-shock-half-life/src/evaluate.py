"""Forecast losses by horizon: QLIKE tables, Diebold-Mariano tests with Holm, bootstrap CIs, stress windows,
and the calibration-after-shocks check (realized / forecast by forecast-state quintile).

Output: results/qlike_by_index.csv, results/qlike_panel.csv, results/stress_windows.csv,
        results/calibration_by_state.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import H, MODEL_ORDER  # noqa: E402
from data import INDICES, ROOT  # noqa: E402

RES = ROOT / "results"
BENCH = "GARCH"
NBOOT = 2000
SEED = 2026
STRESS = {"GFC 2008-09": ("2008-09-01", "2009-06-30"), "Euro/US downgrade 2011": ("2011-07-01", "2011-12-31"),
          "COVID 2020": ("2020-02-15", "2020-06-30"), "Inflation bear 2022": ("2022-01-01", "2022-12-31"),
          "Black Monday 1987": ("1987-10-01", "1988-03-31")}


def qlike(y, f):
    x = y / f
    return x - np.log(x) - 1


def nw_t(d, lag):
    """Newey-West t-statistic of the mean of d (Bartlett kernel)."""
    d = d[np.isfinite(d)]
    n = len(d)
    u = d - d.mean()
    v = u @ u / n
    for L in range(1, lag + 1):
        v += 2 * (1 - L / (lag + 1)) * (u[L:] @ u[:-L]) / n
    return d.mean() / np.sqrt(v / n)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def load():
    out = {}
    for c in INDICES:
        out[c] = pd.read_pickle(RES / "forecasts" / f"{c}.pkl.gz")
    return out


def main():
    D = load()
    # ---------- per-index QLIKE and DM vs GARCH
    rows, daily = [], []
    for c, d in D.items():
        for h in H:
            y = d[f"RV|{h}"]
            ok = (y > 0) & y.notna()
            L = {m: qlike(y[ok].values, d[f"{m}|{h}"][ok].values) for m in MODEL_ORDER}
            for m in MODEL_ORDER:
                diff = L[m] - L[BENCH]
                t = nw_t(diff, max(h, 5)) if m != BENCH else np.nan
                rows.append(dict(index=c, h=h, model=m, n=int(ok.sum()), qlike=L[m].mean(),
                                 rel_to_garch=L[m].mean() / L[BENCH].mean(), dm_t=t,
                                 dm_p=2 * stats.norm.sf(abs(t)) if m != BENCH else np.nan))
            daily.append(pd.DataFrame({"index": c, "h": h, "year": y[ok].index.year,
                                       **{m: L[m] for m in MODEL_ORDER}}, index=y[ok].index))
    by = pd.DataFrame(rows)
    by.to_csv(RES / "qlike_by_index.csv", index=False, float_format="%.5g")

    # ---------- panel: equal-weight average over indices of the relative loss; year-block bootstrap
    dl = pd.concat(daily)
    years = np.sort(dl.year.unique())
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(years), size=(NBOOT, len(years)))
    panel = []
    for h in H:
        g = dl[dl.h == h]
        # per (index, year) sums of each model's loss
        S = g.groupby(["index", "year"])[MODEL_ORDER].sum()
        N = g.groupby(["index", "year"]).size()
        for m in MODEL_ORDER:
            if m == BENCH:
                continue
            ratios_point, boot = [], np.zeros(NBOOT)
            for c in INDICES:
                sm = S.loc[c, m].reindex(years).fillna(0).values
                sb = S.loc[c, BENCH].reindex(years).fillna(0).values
                ratios_point.append(sm.sum() / sb.sum())
                cnt = np.stack([np.bincount(dr, minlength=len(years)) for dr in draws]) if c == INDICES[0] else cnt
                boot += np.log((cnt @ sm) / (cnt @ sb))
            boot = np.exp(boot / len(INDICES))
            point = np.exp(np.mean(np.log(ratios_point)))
            # panel DM: t-test across indices of per-index NW t-stats is not standard; use the bootstrap p
            p_boot = 2 * min(np.mean(boot >= 1), np.mean(boot <= 1))
            sub = by[(by.h == h) & (by.model == m)]
            panel.append(dict(h=h, model=m, geo_mean_rel=point, lo=np.quantile(boot, 0.05),
                              hi=np.quantile(boot, 0.95), p_boot=max(p_boot, 1 / NBOOT),
                              n_better=int((sub.rel_to_garch < 1).sum()),
                              n_sig_better=int(((sub.rel_to_garch < 1) & (sub.dm_p < 0.05)).sum()),
                              n_sig_worse=int(((sub.rel_to_garch > 1) & (sub.dm_p < 0.05)).sum())))
    pan = pd.DataFrame(panel)
    pan["p_holm"] = holm(pan.p_boot.values)
    pan.to_csv(RES / "qlike_panel.csv", index=False, float_format="%.5g")

    # ---------- stress windows (h = 22 and 66): mean QLIKE over origins in the window, all indices
    st = []
    for name, (a, b) in STRESS.items():
        for h in (22, 66):
            g = dl[(dl.h == h) & (dl.index >= a) & (dl.index <= b)]
            if len(g) == 0:
                continue
            means = g[MODEL_ORDER].mean()
            for m in MODEL_ORDER:
                st.append(dict(window=name, h=h, model=m, n_origins=len(g), n_indices=g["index"].nunique(),
                               qlike=means[m], rel_to_garch=means[m] / means[BENCH]))
    pd.DataFrame(st).to_csv(RES / "stress_windows.csv", index=False, float_format="%.5g")

    # ---------- calibration by state: realized/forecast (ratio of sums) by quintile of GARCH s / 5y median
    cal = []
    for c, d in D.items():
        s = d["GARCH_s"]
        base = (d["r"] ** 2).rolling(22).mean().rolling(1260, min_periods=750).median()
        state = (s / base)
        for h in (22, 66, 252):
            y = d[f"RV|{h}"]
            ok = y.notna() & state.notna()
            q = pd.qcut(state[ok], 5, labels=False)
            for m in MODEL_ORDER:
                f = d[f"{m}|{h}"][ok]
                for k in range(5):
                    sel = (q == k).values
                    cal.append(dict(index=c, h=h, model=m, quintile=k + 1,
                                    ratio=y[ok][sel].sum() / f[sel].sum()))
    cal = pd.DataFrame(cal)
    cal.to_csv(RES / "calibration_by_state.csv", index=False, float_format="%.5g")

    pd.set_option("display.width", 200)
    print(pan.round(3).to_string())
    print(cal.groupby(["h", "model", "quintile"]).ratio.median().unstack().round(2))


if __name__ == "__main__":
    main()
