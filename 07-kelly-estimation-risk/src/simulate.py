"""
The Kelly simulator: Monte Carlo experiments that separate what Kelly theory promises from what
real return dynamics deliver.

S1  Estimation risk, GBM world.   Validate E[g] = c theta^2 - c^2 (theta^2 + 1/T)/2 and c*(theta,T).
S2  Estimation risk, real world.  Same experiment on stationary-block-bootstrap paths of each ETF's
    daily excess returns, re-centred to a common true Sharpe ratio THETA_TRUE so that only the
    shape of returns (fat tails, volatility clustering) differs from the GBM case.
S3  The drawdown law.             P(wealth ever falls to x of its start) = x^(2/c - 1) under GBM.
    Tested on bootstrap paths for fixed-fraction Kelly and for volatility-timed Kelly (EWMA).

Outputs: results/sim_gbm_grid.csv, results/sim_real_multiplier.csv, results/sim_drawdown_law.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import ASSETS, DATA, RES, load_rf  # noqa: E402
from kelly import (TRADING_DAYS, drawdown_law, expected_plugin_growth,  # noqa: E402
                   optimal_multiplier, stationary_bootstrap_idx)

SEED = 7
THETA_TRUE = 0.40          # common true annual Sharpe used in S2/S3
MEAN_BLOCK = 63            # ~one quarter: long enough to keep volatility clusters intact
C_GRID = np.round(np.arange(0.0, 1.51, 0.05), 2)


def excess_returns(asset):
    ret = pd.read_csv(DATA / "etf_daily_log_returns.csv", index_col=0, parse_dates=True)[asset].dropna()
    x = np.expm1(ret.values) - load_rf(ret.index)
    return x


def recentre(x, theta):
    """Keep the shape of x (tails, clustering) but set its mean so that the daily Sharpe is theta/sqrt(252)."""
    s = x.std()
    return x - x.mean() + theta / np.sqrt(TRADING_DAYS) * s


# ------------------------------------------------------------------ S1
def s1_gbm_grid(n=20_000, H=10, rng=None):
    """Sufficient-statistic simulation: mu_hat ~ N(mu, s2/T); betting growth over H years is exact for GBM."""
    rng = rng or np.random.default_rng(SEED)
    rows = []
    for theta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        for T in [3, 5, 10, 20, 50, 100]:
            s = 1.0
            mu = theta * s
            mu_hat = mu + s / np.sqrt(T) * rng.standard_normal(n)
            for c in C_GRID:
                f = c * mu_hat / s**2
                # realised H-year log growth with a GBM path: f mu H - f^2 s^2 H/2 + f s W_H
                g_exp = f * mu - 0.5 * f**2 * s**2                      # E[growth | f]
                g = g_exp + f * s * rng.standard_normal(n) / np.sqrt(H)  # one realised H-year path
                rows.append(dict(theta=theta, T=T, c=c, sim=g_exp.mean() / (0.5 * theta**2),
                                 theory=float(expected_plugin_growth(c, theta, T)),
                                 p_below_cash=(g < 0).mean()))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "sim_gbm_grid.csv", index=False, float_format="%.5g")
    return df


# ------------------------------------------------------------------ S2
def s2_real_multiplier(n_paths=1000, H=10, Ts=(5, 10, 20), rng=None):
    rng = rng or np.random.default_rng(SEED + 1)
    rows = []
    for a in ASSETS:
        x = recentre(excess_returns(a), THETA_TRUE)
        f_true = x.mean() / x.var()
        for T in Ts:
            nT, nH = T * TRADING_DAYS, H * TRADING_DAYS
            idx = stationary_bootstrap_idx(len(x), (n_paths, nT + nH), MEAN_BLOCK, rng)
            p = x[idx]
            est, bet = p[:, :nT], p[:, nT:]
            f_hat = est.mean(1) / est.var(1)
            for c in C_GRID:
                f = np.clip(c * f_hat, -10, 10)[:, None]
                w = 1.0 + f * bet
                g = np.where((w <= 0).any(1), -np.inf, np.log(np.maximum(w, 1e-300)).sum(1)) / H
                # oracle growth on the same betting paths
                rows.append(dict(asset=a, T=T, c=c, mean_growth=np.mean(np.maximum(g, -10)),
                                 median_growth=np.median(g), p_ruin=np.mean(~np.isfinite(g)),
                                 f_true=f_true))
            print(f"S2 {a} T={T}", flush=True)
    df = pd.DataFrame(rows)
    # oracle growth (c applied to f_true) on GBM-equivalent theory: theta^2/2 per year
    df["theory_c_star"] = optimal_multiplier(THETA_TRUE, df["T"])
    df.to_csv(RES / "sim_real_multiplier.csv", index=False, float_format="%.5g")
    return df


# ------------------------------------------------------------------ S3
def s3_drawdown_law(n_paths=1500, years=40, rng=None, cs=(1.0, 0.5, 0.25), xs=(0.5, 0.3, 0.1)):
    """
    Probability that wealth ever falls to x of its starting value within `years`, for a fixed fraction
    c * f_true, and for a volatility-timed rule targeting the same average leverage (EWMA, lambda=0.94):
        f_t = c * theta_d / sigma_t   (constant-Sharpe Kelly).
    GBM paths with the same mean and variance give the finite-horizon benchmark for the x^(2/c-1) law.
    """
    rng = rng or np.random.default_rng(SEED + 2)
    nD = years * TRADING_DAYS
    rows = []
    for a in ASSETS:
        x = recentre(excess_returns(a), THETA_TRUE)
        m, v = x.mean(), x.var()
        f_true = m / v
        idx = stationary_bootstrap_idx(len(x), (n_paths, nD), MEAN_BLOCK, rng)
        boot = x[idx]
        gbm = m + np.sqrt(v) * rng.standard_normal((n_paths, nD))
        # EWMA variance along each bootstrap path (uses only past returns)
        ew = np.empty_like(boot)
        s2 = np.full(n_paths, v)
        for j in range(nD):
            ew[:, j] = s2
            s2 = 0.94 * s2 + 0.06 * boot[:, j] ** 2
        th_d = m / np.sqrt(v)
        for c in cs:
            variants = {
                "GBM, fixed fraction": (gbm, c * f_true),
                "Real dynamics, fixed fraction": (boot, c * f_true),
                "Real dynamics, vol-timed (EWMA)": (boot, c * th_d / np.sqrt(ew)),
            }
            for name, (paths, f) in variants.items():
                w = 1.0 + f * paths
                logw = np.cumsum(np.log(np.maximum(w, 1e-12)), axis=1)
                minw = np.exp(np.minimum(logw.min(1), 0.0))
                peak = np.maximum.accumulate(np.maximum(logw, 0.0), axis=1)
                mdd = 1 - np.exp((logw - peak).min(1))
                for xx in xs:
                    rows.append(dict(asset=a, c=c, variant=name, x=xx,
                                     p_ever=(minw <= xx).mean(), theory=float(drawdown_law(xx, c)),
                                     p_mdd=(mdd >= 1 - xx).mean(),
                                     growth=np.median(logw[:, -1]) / years,
                                     mean_f=float(np.mean(f))))
        print(f"S3 {a}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "sim_drawdown_law.csv", index=False, float_format="%.5g")
    return df


if __name__ == "__main__":
    s1_gbm_grid()
    s2_real_multiplier()
    s3_drawdown_law()
