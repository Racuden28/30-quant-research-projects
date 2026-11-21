"""
Why do daily betas lose at the 12-month horizon? Measure, industry by industry, the gap between the
beta estimated from monthly returns and the beta estimated from daily returns over the same days,
and relate it to firm size and to the lagged reaction to the market (non-synchronous trading).

Output: results/horizon_gap.csv (one row per industry and era) and results/horizon_gap_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data import SIZE, load_panel
from estimators import monthly_excess

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ERAS = {"1929-1962": ("1929-01", "1962-12"), "1963-1999": ("1963-01", "1999-12"),
        "2000-2026": ("2000-01", "2026-09")}


def slope(y, x):
    ok = ~np.isnan(y)
    return np.polyfit(x[ok], y[ok], 1)[0] if ok.sum() > 30 else np.nan


def main():
    p = load_panel()
    assets = [c for c in p.columns if c not in ("MKT", "RF")]
    size = pd.read_csv(SIZE, index_col=0, parse_dates=True)
    lsize = np.log(size.where(size > 0))
    rel_size = lsize.sub(lsize.mean(axis=1), axis=0)        # log size relative to the cross-section
    rows = []
    for era, (a, b) in ERAS.items():
        q = p.loc[a:b]
        Y, x, rf = q[assets].to_numpy(), q["MKT"].to_numpy(), q["RF"].to_numpy()
        per = q.index.to_period("M")
        mid = np.searchsorted(per.unique().asi8, per.asi8)
        Ym, xm = monthly_excess(Y, x, rf, mid)
        xl1 = np.concatenate([[np.nan], x[:-1]])
        xl2 = np.concatenate([[np.nan] * 2, x[:-2]])
        for j, name in enumerate(assets):
            y = Y[:, j]
            ok = ~np.isnan(y) & ~np.isnan(xl2)
            if ok.sum() < 1000:
                continue
            Z = np.column_stack([np.ones(ok.sum()), x[ok], xl1[ok], xl2[ok]])
            c = np.linalg.lstsq(Z, y[ok], rcond=None)[0]
            rows.append(dict(era=era, asset=name, beta_daily=slope(y, x), beta_monthly=slope(Ym[:, j], xm),
                             lag_beta=c[2] + c[3], rel_log_size=rel_size.loc[a:b, name].mean()))
    g = pd.DataFrame(rows)
    g["gap"] = g.beta_monthly - g.beta_daily
    g.to_csv(RES / "horizon_gap.csv", index=False, float_format="%.4f")
    summ = []
    for era, h in g.groupby("era", sort=False):
        r_size = stats.spearmanr(h.gap, h.rel_log_size)
        r_lag = stats.spearmanr(h.gap, h.lag_beta)
        summ.append(dict(era=era, industries=len(h), mean_daily=h.beta_daily.mean(), mean_monthly=h.beta_monthly.mean(),
                         mean_gap=h.gap.mean(), share_gap_pos=(h.gap > 0).mean(),
                         spearman_gap_size=r_size.statistic, p_gap_size=r_size.pvalue,
                         spearman_gap_lag=r_lag.statistic, p_gap_lag=r_lag.pvalue))
    s = pd.DataFrame(summ).set_index("era")
    s.to_csv(RES / "horizon_gap_summary.csv", float_format="%.4f")
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
