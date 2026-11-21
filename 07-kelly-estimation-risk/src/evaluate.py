"""
Statistics for project 07: growth, drawdowns, pre-registered tests (Newey-West + stationary bootstrap,
Holm across the family), stress windows, knowability of the Kelly fraction and robustness checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import ASSETS, DATA, RES, STRATEGIES, load_rf, run_all  # noqa: E402
from kelly import (TRADING_DAYS, WEEKS, block_bootstrap_ci, holm,  # noqa: E402
                   knowability_years, max_drawdown, newey_west_t)

STRESS = {"GFC (Sep 2008-Mar 2009)": ("2008-09-01", "2009-03-31"),
          "COVID (Feb-Apr 2020)": ("2020-02-19", "2020-04-30"),
          "Rates shock (Jan-Oct 2022)": ("2022-01-01", "2022-10-31")}

# pre-registered primary comparisons: (hypothesis, strategy A, strategy B); test mean(A - B) weekly log growth
PRIMARY = [("H2: plug-in Kelly loses to buy & hold", "Plug-in Kelly", "Buy & hold"),
           ("H3: prior-Sharpe GARCH beats plug-in", "Prior-Sharpe GARCH", "Plug-in Kelly"),
           ("H4: volatility timing adds growth", "Prior-Sharpe GARCH", "Prior-Sharpe static"),
           ("H5: shrinkage beats plug-in", "Shrinkage Kelly", "Plug-in Kelly")]
FLOOR = np.log(1e-6)


def load():
    w = pd.read_csv(RES / "weekly.csv.gz", parse_dates=["origin"])
    w["logret"] = w.logret.fillna(0.0)  # after ruin wealth stays at the floor
    return w


def summary(w):
    rows = []
    for (s, a), g in w.groupby(["strategy", "asset"], sort=False):
        lr = g.logret.values
        logw = np.cumsum(lr)
        f = g.f.values
        rows.append(dict(strategy=s, asset=a, growth=lr.mean() * WEEKS, vol=lr.std() * np.sqrt(WEEKS),
                         max_dd=max_drawdown(np.concatenate([[0], logw])), worst_week=np.expm1(lr.min()),
                         mean_f=f.mean(), sd_f=f.std(), pct_levered=(f > 1).mean(), pct_short=(f < 0).mean(),
                         turnover=np.abs(np.diff(f)).mean() * WEEKS, ruined=bool((g.gross <= 1.01e-6).any()),
                         terminal_wealth=np.exp(logw[-1])))
    df = pd.DataFrame(rows)
    cash = df[df.strategy == "Cash"].set_index("asset").growth
    orc = df[df.strategy == "Oracle constant Kelly"].set_index("asset").growth
    df["captured"] = (df.growth - df.asset.map(cash)) / (df.asset.map(orc) - df.asset.map(cash))
    return df


def pooled_weekly(w, strategy):
    """Equal-weight average across available assets of each calendar week's log growth."""
    d = w[w.strategy == strategy].copy()
    d["wk"] = d.origin.dt.to_period("W-FRI")
    return d.pivot_table(index="wk", columns="asset", values="logret").mean(axis=1)


def tests(w):
    rows = []
    for hyp, A, B in PRIMARY:
        for a in ASSETS + ["Pooled"]:
            if a == "Pooled":
                d = (pooled_weekly(w, A) - pooled_weekly(w, B)).dropna().values
            else:
                d = (w[(w.strategy == A) & (w.asset == a)].logret.values
                     - w[(w.strategy == B) & (w.asset == a)].logret.values)
            t, se = newey_west_t(d)
            lo, hi = block_bootstrap_ci(d, mean_block=8, seed=11) * WEEKS
            # one-sided in the hypothesised direction (H2: A < B; others: A > B)
            sign = -1 if hyp.startswith("H2") else 1
            p = 1 - stats.norm.cdf(sign * t)
            rows.append(dict(hypothesis=hyp, A=A, B=B, asset=a, diff_growth=d.mean() * WEEKS,
                             ci_lo=lo, ci_hi=hi, nw_t=t, p_one_sided=p))
    df = pd.DataFrame(rows)
    m = df.asset != "Pooled"
    df.loc[m, "p_holm"] = holm(df.loc[m, "p_one_sided"].values)          # 28 asset-level tests
    df.loc[~m, "p_holm"] = holm(df.loc[~m, "p_one_sided"].values)        # 4 pooled tests
    return df


def stress(w):
    rows = []
    for name, (s, e) in STRESS.items():
        d = w[(w.origin >= s) & (w.origin <= e)]
        cum = d.groupby(["strategy", "asset"]).logret.sum()
        for (st, a), v in cum.items():
            rows.append(dict(window=name, strategy=st, asset=a, cum_return=np.expm1(v)))
    return pd.DataFrame(rows)


def knowability():
    """
    How well can 20 years pin down the Kelly fraction f = mu / sigma^2?
    The uncertainty in f comes almost entirely from mu: the standard error of the mean of daily excess
    returns is taken from Newey-West (HAC), and sigma is treated as known (its relative sampling error
    is ~1-2%, against 50-100%+ for mu). se(f) = se_HAC(mu) / sigma^2, all annualised.
    Decision-relevant horizons (years of data, at the current point estimate, until the 90% CI ...):
      * years_sign      ... excludes 0 (long or short is known)
      * years_abs_05    ... has half-width 0.5 in f (absolute tolerance; does not depend on the estimate)
      * years_rel_50    ... has half-width 50% of |f| (relative; explodes when f is near 0, appendix only)
    """
    ret = pd.read_csv(DATA / "etf_daily_log_returns.csv", index_col=0, parse_dates=True)
    z = 1.645
    rows = []
    for a in ASSETS:
        r = ret[a].dropna()
        x = np.expm1(r.values) - load_rf(r.index)
        n = len(x)
        T = n / TRADING_DAYS
        mu, sd = x.mean() * TRADING_DAYS, x.std() * np.sqrt(TRADING_DAYS)
        _, se_hac_d = newey_west_t(x)
        k = se_hac_d / (x.std() / np.sqrt(n))                # HAC inflation of the iid standard error
        theta = mu / sd
        f = mu / sd**2
        se_f = k / (sd * np.sqrt(T))
        rows.append(dict(asset=a, years=T, sharpe=theta, tstat_hac=theta * np.sqrt(T) / k, hac_factor=k,
                         kelly_f=f, se_f=se_f, f_lo90=f - z * se_f, f_hi90=f + z * se_f,
                         p_sign_wrong=stats.norm.cdf(-abs(theta) * np.sqrt(T) / k),
                         years_sign=(z * k / abs(theta)) ** 2,
                         years_abs_05=(z * k / (0.5 * sd)) ** 2,
                         years_rel_50=float(knowability_years(theta / k, 0.5))))
    return pd.DataFrame(rows)


def plugin_instability(inputs):
    rows = []
    for a, g in inputs.groupby("asset", sort=False):
        f = g["f:Plug-in Kelly"].values
        f5 = g["f:Plug-in Kelly 5y"].values
        fo = g["f:Oracle constant Kelly"].values[0]
        rows.append(dict(asset=a, oracle_f=fo, plugin_min=f.min(), plugin_max=f.max(),
                         plugin5_min=f5.min(), plugin5_max=f5.max(),
                         pct_wrong_sign=(np.sign(f) != np.sign(fo)).mean(),
                         pct_wrong_sign_5y=(np.sign(f5) != np.sign(fo)).mean(),
                         pct_at_cap=((f >= 3.999) | (f <= -1.999)).mean(),
                         pct_shrunk_to_zero=(g.shrink_c.values == 0).mean(),
                         median_tstat=np.median(g.tstat.values)))
    return pd.DataFrame(rows)


def robustness():
    KEY = ["Buy & hold", "Oracle constant Kelly", "Plug-in Kelly", "Half Kelly", "Shrinkage Kelly",
           "Prior-Sharpe static", "Prior-Sharpe GARCH", "Prior-Sharpe EWMA"]
    configs = {"Baseline (theta0=0.3, 2bp, f in [-2,4])": {},
               "theta0 = 0.2": dict(theta0=0.2), "theta0 = 0.4": dict(theta0=0.4),
               "Costs 10 bp": dict(tc=0.001), "Spread 2%": dict(spread=0.02),
               "Long-only, f in [0,2]": dict(f_bounds=(0.0, 2.0)),
               "Loose bounds, f in [-5,8]": dict(f_bounds=(-5.0, 8.0))}
    rows = []
    for name, kw in configs.items():
        wk, _ = run_all(tag="rob", **kw)
        wk["logret"] = wk.logret.fillna(0.0)
        g = wk.groupby(["strategy", "asset"]).logret.mean().unstack() * WEEKS
        for s in KEY:
            rows.append(dict(config=name, strategy=s, pooled_growth=g.loc[s].mean(),
                             assets_beating_plugin=int((g.loc[s] > g.loc["Plug-in Kelly"]).sum())))
    return pd.DataFrame(rows)


def main():
    w = load()
    s = summary(w)
    s.to_csv(RES / "summary_by_asset.csv", index=False, float_format="%.5g")
    pooled = (s.groupby("strategy", sort=False)
              .agg(growth=("growth", "mean"), max_dd=("max_dd", "mean"), worst_week=("worst_week", "min"),
                   mean_f=("mean_f", "mean"), turnover=("turnover", "mean"),
                   ruined=("ruined", "sum"))
              .reset_index())
    gc, go = pooled.set_index("strategy").growth[["Cash", "Oracle constant Kelly"]]
    pooled["captured"] = (pooled.growth - gc) / (go - gc)       # share of hindsight-Kelly excess growth
    pooled["family"] = pooled.strategy.map(lambda x: STRATEGIES[x][0])
    pooled.to_csv(RES / "summary_pooled.csv", index=False, float_format="%.5g")
    print(pooled.round(3).to_string())
    t = tests(w)
    t.to_csv(RES / "tests.csv", index=False, float_format="%.5g")
    print(t.round(4).to_string())
    stress(w).to_csv(RES / "stress_windows.csv", index=False, float_format="%.5g")
    k = knowability()
    k.to_csv(RES / "knowability.csv", index=False, float_format="%.5g")
    print(k.round(3).to_string())
    inputs = pd.read_csv(RES / "inputs.csv.gz", parse_dates=["origin"])
    pi = plugin_instability(inputs)
    pi.to_csv(RES / "plugin_instability.csv", index=False, float_format="%.5g")
    print(pi.round(3).to_string())
    rb = robustness()
    rb.to_csv(RES / "robustness.csv", index=False, float_format="%.5g")
    print(rb.round(4).to_string())


if __name__ == "__main__":
    main()
