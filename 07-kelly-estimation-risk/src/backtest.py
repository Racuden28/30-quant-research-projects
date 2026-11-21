"""
Walk-forward Kelly backtest, weekly rebalancing, seven ETFs.

At each weekly origin t of project 01 (close of day t-1; the held position earns days t..t+4):
  * every input to every sizing rule uses daily data strictly before t;
  * the dynamic variance forecast is project 01's out-of-sample GJR-GARCH-t predictive s.d. for the
    same week (data/gjr_garch_t_weekly_forecasts_from_project01.csv), so this project re-uses, and
    therefore tests, the economic value of #01's best-calibrated simulator;
  * positions are clipped to [F_MIN, F_MAX], leverage pays the T-bill rate plus SPREAD, turnover after
    weight drift costs TC per unit.

Writes results/weekly.csv.gz (asset, origin, strategy, f, gross, logret) and results/inputs.csv.gz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kelly import TRADING_DAYS, shrinkage_from_t  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

# ---- pre-registered design constants (README section 3) ----
F_MIN, F_MAX = -2.0, 4.0        # position bounds (fraction of wealth)
SPREAD = 0.005                  # annual borrowing spread over T-bills on leverage
TC = 0.0002                     # 2 bp per unit of turnover
THETA0 = 0.30                   # agnostic prior Sharpe ratio (annual), same for every asset
ROLL = 1260                     # rolling window (days) for the "5-year" plug-in, as in project 01
EWMA_LAMBDA = 0.94              # RiskMetrics
ASSETS = ["SPY", "EEM", "TLT", "HYG", "GLD", "USO", "URA"]

STRATEGIES = {
    # name: (family, description)
    "Cash": ("benchmark", "f = 0"),
    "Buy & hold": ("benchmark", "f = 1"),
    "Oracle constant Kelly": ("oracle", "best constant f chosen with hindsight (infeasible)"),
    "Plug-in Kelly": ("plug-in", "mu_hat / s2_hat, expanding window"),
    "Plug-in Kelly 5y": ("plug-in", "mu_hat / s2_hat, rolling 5-year window"),
    "Half Kelly": ("plug-in", "0.5 x plug-in"),
    "Shrinkage Kelly": ("plug-in", "(1 - 1/t^2)+ x plug-in"),
    "Prior-Sharpe static": ("prior", "theta0 / sigma_hat, expanding"),
    "Prior-Sharpe EWMA": ("prior", "theta0 / sigma_EWMA"),
    "Prior-Sharpe GARCH": ("prior", "theta0 / sigma_GJR-GARCH-t (from #01)"),
    "Plug-in mean, GARCH var": ("hybrid", "mu_hat / s2_GARCH"),
    "Prior mean, GARCH var": ("hybrid", "theta0 sigma_hat / s2_GARCH"),
    "Shrinkage mean, GARCH var": ("hybrid", "(1 - 1/t^2)+ mu_hat / s2_GARCH"),
    "Oracle mean, static var": ("oracle", "mu_full-sample / s2_hat (infeasible)"),
    "Oracle mean, GARCH var": ("oracle", "mu_full-sample / s2_GARCH (infeasible)"),
}


def load_rf(dates: pd.DatetimeIndex) -> np.ndarray:
    tb = pd.read_csv(DATA / "tbill_3m_annual.csv").set_index("year").tbill_3m_pct / 100
    return tb.reindex(dates.year).values / TRADING_DAYS


def build_inputs(asset: str, theta0: float = THETA0) -> pd.DataFrame:
    ret = pd.read_csv(DATA / "etf_daily_log_returns.csv", index_col=0, parse_dates=True)[asset].dropna()
    r, dates = ret.values, ret.index
    rf = load_rf(dates)
    x = np.expm1(r) - rf                                    # daily simple excess returns
    fc = pd.read_csv(DATA / "gjr_garch_t_weekly_forecasts_from_project01.csv", parse_dates=["origin"])
    fc = fc[fc.asset == asset].reset_index(drop=True)
    t_idx = fc.t.values
    assert (dates[t_idx] == fc.origin.values).all(), "origin/date misalignment"
    assert np.allclose([r[t:t + 5].sum() for t in t_idx], fc.y.values, atol=6e-4), "y != sum of r"

    # expanding moments through t-1 via cumulative sums
    cs, cs2 = np.concatenate([[0], np.cumsum(x)]), np.concatenate([[0], np.cumsum(x**2)])
    n = t_idx.astype(float)
    mu = cs[t_idx] / n
    var = (cs2[t_idx] - n * mu**2) / (n - 1)
    lo = np.maximum(t_idx - ROLL, 0)
    nr = (t_idx - lo).astype(float)
    mu_r = (cs[t_idx] - cs[lo]) / nr
    var_r = (cs2[t_idx] - cs2[lo] - nr * mu_r**2) / (nr - 1)
    # EWMA variance through t-1 (seeded with the first 60 days)
    ew = np.empty(len(x) + 1)
    ew[0] = np.var(x[:60])
    for i in range(len(x)):
        ew[i + 1] = EWMA_LAMBDA * ew[i] + (1 - EWMA_LAMBDA) * x[i] ** 2
    var_ewma = ew[t_idx]
    var_garch = fc.sd.values ** 2 / 5.0                     # weekly log-return variance -> daily
    tstat = mu / np.sqrt(var / n)

    # realised week: simple return, T-bill and spread accrued over the five days
    R = np.exp(fc.y.values) - 1.0
    rf_w = np.array([rf[t:t + 5].sum() for t in t_idx])
    eval_mask = np.zeros(len(x), bool)
    for t in t_idx:
        eval_mask[t:t + 5] = True
    mu_oracle = x[eval_mask].mean()                          # infeasible: full evaluation-sample mean
    th = theta0 / np.sqrt(TRADING_DAYS)                     # daily prior Sharpe
    c = shrinkage_from_t(tstat)

    f = {
        "Cash": np.zeros_like(mu), "Buy & hold": np.ones_like(mu),
        "Plug-in Kelly": mu / var, "Plug-in Kelly 5y": mu_r / var_r, "Half Kelly": 0.5 * mu / var,
        "Shrinkage Kelly": c * mu / var,
        "Prior-Sharpe static": th / np.sqrt(var), "Prior-Sharpe EWMA": th / np.sqrt(var_ewma),
        "Prior-Sharpe GARCH": th / np.sqrt(var_garch),
        "Plug-in mean, GARCH var": mu / var_garch, "Prior mean, GARCH var": th * np.sqrt(var) / var_garch,
        "Shrinkage mean, GARCH var": c * mu / var_garch,
        "Oracle mean, static var": mu_oracle / var, "Oracle mean, GARCH var": mu_oracle / var_garch,
    }
    out = pd.DataFrame({"asset": asset, "origin": fc.origin.values, "R": R, "rf_w": rf_w,
                        "mu_hat_ann": mu * TRADING_DAYS, "sigma_hat_ann": np.sqrt(var * TRADING_DAYS),
                        "sigma_garch_ann": np.sqrt(var_garch * TRADING_DAYS),
                        "sigma_ewma_ann": np.sqrt(var_ewma * TRADING_DAYS), "tstat": tstat,
                        "n_days": t_idx, "shrink_c": c, "mu_oracle_ann": mu_oracle * TRADING_DAYS})
    for k, v in f.items():
        out["f:" + k] = np.clip(v, F_MIN, F_MAX)
        out["fraw:" + k] = v
    return out


def run_path(f, R, rf_w, spread=SPREAD, tc=TC):
    """Weekly log returns of a strategy; weight drift within the week drives turnover; ruin is absorbing."""
    sp = spread * 5 / TRADING_DAYS
    n = len(f)
    logret = np.empty(n)
    gross = np.empty(n)
    w_prev = 0.0
    dead = False
    for i in range(n):
        if dead:
            gross[i], logret[i] = 0.0, np.nan
            continue
        cash = 1.0 - f[i]
        cash_rate = rf_w[i] if cash >= 0 else rf_w[i] + sp
        g = 1.0 + f[i] * R[i] + cash * cash_rate - tc * abs(f[i] - w_prev)
        if g <= 1e-6:
            dead, g = True, 1e-6
        gross[i], logret[i] = g, np.log(g)
        w_prev = f[i] * (1 + R[i]) / g if not dead else 0.0
    return gross, logret


def oracle_constant(R, rf_w, grid=np.round(np.arange(F_MIN, F_MAX + 1e-9, 0.01), 2), **kw):
    best, fbest = -np.inf, 0.0
    for fc in grid:
        _, lr = run_path(np.full(len(R), fc), R, rf_w, **kw)
        s = np.nansum(lr) if not np.isnan(lr).any() else -np.inf
        if s > best:
            best, fbest = s, fc
    return fbest


def run_all(theta0=THETA0, spread=SPREAD, tc=TC, f_bounds=(F_MIN, F_MAX), tag=""):
    rows, inputs = [], []
    for a in ASSETS:
        inp = build_inputs(a, theta0)
        if f_bounds != (F_MIN, F_MAX):
            for k in STRATEGIES:
                if "fraw:" + k in inp:
                    inp["f:" + k] = np.clip(inp["fraw:" + k], *f_bounds)
        R, rf_w = inp.R.values, inp.rf_w.values
        lo, hi = f_bounds
        fo = oracle_constant(R, rf_w, grid=np.round(np.arange(lo, hi + 1e-9, 0.01), 2), spread=spread, tc=tc)
        inp["f:Oracle constant Kelly"] = fo
        for s in STRATEGIES:
            f = inp["f:" + s].values
            gross, lr = run_path(f, R, rf_w, spread=spread, tc=tc)
            rows.append(pd.DataFrame({"asset": a, "origin": inp.origin, "strategy": s, "f": f,
                                      "gross": gross, "logret": lr}))
        inputs.append(inp)
        print(f"{a}: {len(inp)} weeks, oracle constant f = {fo:+.2f}", flush=True)
    weekly = pd.concat(rows, ignore_index=True)
    inputs = pd.concat(inputs, ignore_index=True)
    if not tag:
        weekly.to_csv(RES / "weekly.csv.gz", index=False, float_format="%.6g")
        inputs.drop(columns=[c for c in inputs if c.startswith("fraw:")]).to_csv(
            RES / "inputs.csv.gz", index=False, float_format="%.6g")
    return weekly, inputs


if __name__ == "__main__":
    run_all()
