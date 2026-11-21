"""
Calibration and scoring of the Monte Carlo forecasts produced by backtest.py.

Outputs (results/):
  summary_by_model.csv        pooled calibration + scoring per model and horizon
  summary_by_asset_model.csv  the same, per asset
  var_backtests.csv           Kupiec / Christoffersen / Berkowitz tests per asset x model (h=5)
  conditional_coverage.csv    90% interval coverage by volatility-state quintile
  stress_windows.csv          tail exceedances inside named crisis windows
  crps_dm.csv                 CRPS skill vs GBM with Diebold-Mariano (Newey-West) t-stats
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MODEL_ORDER = ["GBM", "Merton jumps", "GJR-GARCH-t", "Markov switching", "Block bootstrap", "Pool"]
CONTROL_ORDER = ["GJR-GARCH-N"]
ALL_MODELS = MODEL_ORDER + CONTROL_ORDER
ASSET_ORDER = ["SPY", "EEM", "TLT", "HYG", "GLD", "USO", "URA"]

STRESS = {
    "GFC (Sep 2008 - Mar 2009)": ("2008-09-01", "2009-03-31"),
    "Euro crisis (Jul - Dec 2011)": ("2011-07-01", "2011-12-31"),
    "Oil crash (Nov 2014 - Feb 2016)": ("2014-11-01", "2016-02-29"),
    "COVID (Feb - May 2020)": ("2020-02-15", "2020-05-31"),
    "Inflation shock (2022)": ("2022-01-01", "2022-12-31"),
}


def load() -> pd.DataFrame:
    df = pd.read_csv(RES / "forecasts.csv.gz", parse_dates=["origin"])
    # h=21 forecasts are issued weekly, so consecutive outcomes overlap. For every statistic
    # that assumes independent observations we keep every 5th origin (25-day spacing).
    keep = (df["h"] == 5) | ((df["t"] // 5) % 5 == 0)
    df = df[keep].copy()
    df["hit90"] = (df["y"] < df["q0.050"]) | (df["y"] > df["q0.950"])
    df["hit98"] = (df["y"] < df["q0.010"]) | (df["y"] > df["q0.990"])
    df["hit50"] = (df["y"] < df["q0.250"]) | (df["y"] > df["q0.750"])
    df["var1"] = df["y"] < df["q0.010"]
    df["var5"] = df["y"] < df["q0.050"]
    df["up1"] = df["y"] > df["q0.990"]
    df["width90"] = df["q0.950"] - df["q0.050"]
    return df


# ---------------------------------------------------------------- classical VaR backtests


def kupiec(hits: np.ndarray, p: float) -> float:
    n, x = len(hits), int(hits.sum())
    pi = x / n
    l0 = (n - x) * np.log(1 - p) + x * np.log(p)
    l1 = (n - x) * np.log(1 - pi) + x * np.log(pi) if 0 < x < n else 0.0
    return float(stats.chi2.sf(-2 * (l0 - l1), 1))


def christoffersen_ind(hits: np.ndarray) -> float:
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
    n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(k, p):
        return k * np.log(p) if p > 0 else 0.0

    l0 = ll(n00 + n10, 1 - pi) + ll(n01 + n11, pi)
    l1 = ll(n00, 1 - pi01) + ll(n01, pi01) + ll(n10, 1 - pi11) + ll(n11, pi11)
    return float(stats.chi2.sf(max(-2 * (l0 - l1), 0), 1))


def berkowitz(pit: np.ndarray) -> tuple[float, float]:
    """LR test that Phi^-1(PIT) is i.i.d. N(0,1) against a Gaussian AR(1) alternative (3 df)."""
    z = stats.norm.ppf(np.clip(pit, 1e-6, 1 - 1e-6))
    y, x = z[1:], z[:-1]

    def nll(th):
        c, rho, ls = th
        s = np.exp(ls)
        return -np.sum(stats.norm.logpdf(y, c + rho * x, s))

    res = optimize.minimize(nll, [0, 0, 0], method="Nelder-Mead")
    l1 = -res.fun
    l0 = np.sum(stats.norm.logpdf(y))
    lr = -2 * (l0 - l1)
    return float(lr), float(stats.chi2.sf(lr, 3))


def nw_tstat(d: np.ndarray, lags: int) -> float:
    d = d - 0  # mean differential
    n = len(d)
    m = d.mean()
    e = d - m
    v = np.dot(e, e) / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * np.dot(e[k:], e[:-k]) / n
    return float(m / np.sqrt(v / n))


# ---------------------------------------------------------------- summaries


def summarise(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n": len(g),
        "cov50": 1 - g.hit50.mean(), "cov90": 1 - g.hit90.mean(), "cov98": 1 - g.hit98.mean(),
        "var1_rate": g.var1.mean(), "var5_rate": g.var5.mean(), "up1_rate": g.up1.mean(),
        "tail_mult_98": g.hit98.mean() / 0.02,
        "pit_sd": g.pit.std(), "crps": g.crps.mean(), "width90": g.width90.mean(),
    })


def main():
    df = load()
    by_model = df.groupby(["h", "model"]).apply(summarise, include_groups=False).reset_index()
    by_model.to_csv(RES / "summary_by_model.csv", index=False, float_format="%.4f")
    by_am = df.groupby(["h", "asset", "model"]).apply(summarise, include_groups=False).reset_index()
    by_am.to_csv(RES / "summary_by_asset_model.csv", index=False, float_format="%.4f")

    # VaR backtests (h = 5, weekly non-overlapping by construction)
    rows = []
    for (h, a, m), g in df.groupby(["h", "asset", "model"]):
        g = g.sort_values("origin")
        lr, pb = berkowitz(g.pit.values)
        rows.append({
            "h": h, "asset": a, "model": m, "n": len(g),
            "var1_rate": g.var1.mean(), "kupiec1_p": kupiec(g.var1.values, 0.01),
            "christ1_p": christoffersen_ind(g.var1.values),
            "var5_rate": g.var5.mean(), "kupiec5_p": kupiec(g.var5.values, 0.05),
            "christ5_p": christoffersen_ind(g.var5.values),
            "berkowitz_lr": lr, "berkowitz_p": pb,
        })
    vb = pd.DataFrame(rows)
    vb.to_csv(RES / "var_backtests.csv", index=False, float_format="%.4g")

    # conditional coverage by volatility state (pooled across assets)
    df["vol_q"] = pd.cut(df.rv_pct, [0, .2, .4, .6, .8, 1.0001], labels=["Q1 calm", "Q2", "Q3", "Q4", "Q5 stressed"],
                         include_lowest=True)
    cc = df.groupby(["h", "model", "vol_q"], observed=True).agg(
        n=("hit90", "size"), cov90=("hit90", lambda x: 1 - x.mean()), cov98=("hit98", lambda x: 1 - x.mean()),
        var1_rate=("var1", "mean"), pit_sd=("pit", "std"), width90=("width90", "mean")).reset_index()
    cc.to_csv(RES / "conditional_coverage.csv", index=False, float_format="%.4f")

    # stress windows
    rows = []
    for w, (a, b) in STRESS.items():
        g = df[(df.origin >= a) & (df.origin <= b)]
        for (h, m), gg in g.groupby(["h", "model"]):
            rows.append({"window": w, "h": h, "model": m, "n": len(gg), "cov90": 1 - gg.hit90.mean(),
                         "exceed98_mult": gg.hit98.mean() / 0.02, "var1_rate": gg.var1.mean(),
                         "crps": gg.crps.mean()})
    pd.DataFrame(rows).to_csv(RES / "stress_windows.csv", index=False, float_format="%.4f")

    # CRPS skill vs GBM with Diebold-Mariano t-stats
    rows = []
    for (h, a), g in df.groupby(["h", "asset"]):
        piv = g.pivot_table(index="origin", columns="model", values="crps")
        for m in ALL_MODELS[1:]:
            d = (piv["GBM"] - piv[m]).dropna().values
            rows.append({"h": h, "asset": a, "model": m,
                         "skill_vs_gbm": 1 - piv[m].mean() / piv["GBM"].mean(),
                         "dm_t": nw_tstat(d, lags=4)})
    pd.DataFrame(rows).to_csv(RES / "crps_dm.csv", index=False, float_format="%.4f")
    print(by_model[by_model.h == 5].to_string())
    print(cc[cc.h == 5].pivot(index="model", columns="vol_q", values="cov90").round(3).to_string())


if __name__ == "__main__":
    main()
