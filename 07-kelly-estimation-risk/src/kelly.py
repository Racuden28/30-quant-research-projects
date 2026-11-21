"""
Core Kelly mathematics used by every script in project 07.

Conventions
-----------
* ``f`` is the fraction of wealth held in the risky asset; ``1 - f`` sits in cash (borrowed if negative).
* Continuous-time (Merton) Kelly for an asset with excess drift ``mu`` and variance ``s2`` is
  ``f* = mu / s2`` and the growth rate of fraction ``f`` is ``g(f) = r + f mu - f^2 s2 / 2``.
* Annual Sharpe ratio ``theta = mu / s``. Estimation window ``T`` is in years.
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252
WEEKS = 52


# ---------------------------------------------------------------- estimation-risk theory
def growth(f, mu, s2, r=0.0):
    """Continuous-time growth rate of a constant fraction f."""
    return r + f * mu - 0.5 * f**2 * s2


def expected_plugin_growth(c, theta, T):
    """
    Expected excess growth (in units of the oracle Kelly growth theta^2/2) of betting
    c * mu_hat / s2 when mu_hat ~ N(mu, s2 / T) and s2 is known.

        E[g] = c theta^2 - c^2 (theta^2 + 1/T) / 2,   oracle = theta^2 / 2
    """
    theta = np.asarray(theta, float)
    T = np.asarray(T, float)
    eg = c * theta**2 - 0.5 * c**2 * (theta**2 + 1.0 / T)
    return eg / (0.5 * theta**2)


def optimal_multiplier(theta, T):
    """c*(theta, T) = theta^2 T / (theta^2 T + 1): the fraction of plug-in Kelly that maximises E[g]."""
    x = np.asarray(theta, float) ** 2 * np.asarray(T, float)
    return x / (x + 1.0)


def shrinkage_from_t(tstat):
    """
    Feasible version of c*: replace theta^2 T by its unbiased estimate t^2 - 1, giving
    c_hat = 1 - 1/t^2 (positive part). Bet nothing unless the mean is at least one standard error from 0.
    """
    t2 = np.asarray(tstat, float) ** 2
    with np.errstate(divide="ignore"):
        c = 1.0 - 1.0 / t2
    return np.clip(np.nan_to_num(c, nan=0.0, neginf=0.0), 0.0, 1.0)


def knowability_years(theta, rel_err=0.5, z=1.645):
    """
    Years of data needed so that the plug-in Kelly fraction is within +/- rel_err of the truth
    with two-sided confidence 2*Phi(z)-1 (z=1.645 -> 90%). Since se(f_hat)/f* = 1/(theta sqrt(T)):
        T = (z / (rel_err * theta))^2
    """
    return (z / (rel_err * np.abs(np.asarray(theta, float)))) ** 2


def drawdown_law(x, c):
    """
    GBM, fraction c of the true Kelly bet, infinite horizon: probability that wealth EVER falls to
    a fraction x of its starting value is x^(2/c - 1) (e.g. full Kelly: P(ever halving) = 1/2).
    """
    return np.asarray(x, float) ** (2.0 / np.asarray(c, float) - 1.0)


# ---------------------------------------------------------------- path utilities
def wealth_step(f, R, rf, spread=0.0, tc=0.0, df=0.0):
    """
    One-period gross wealth multiplier for holding fraction f of the risky asset:
    cash leg earns rf, a borrowed cash leg (f > 1) pays rf + spread, turnover |df| costs tc.
    """
    cash = 1.0 - f
    cash_rate = np.where(cash >= 0, rf, rf + spread)
    return 1.0 + f * R + cash * cash_rate - tc * np.abs(df)


def max_drawdown(logw):
    """Maximum drawdown (as a positive fraction) of a log-wealth path, along the last axis."""
    peak = np.maximum.accumulate(logw, axis=-1)
    return 1.0 - np.exp(np.min(logw - peak, axis=-1))


def stationary_bootstrap_idx(n, size, mean_block, rng):
    """Politis-Romano stationary bootstrap indices, shape (size[0], size[1])."""
    n_paths, length = size
    idx = np.empty((n_paths, length), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_paths)
    new_block = rng.random((n_paths, length)) < 1.0 / mean_block
    starts = rng.integers(0, n, (n_paths, length))
    for j in range(1, length):
        idx[:, j] = np.where(new_block[:, j], starts[:, j], (idx[:, j - 1] + 1) % n)
    return idx


# ---------------------------------------------------------------- statistics
def newey_west_t(x, lags=None):
    """HAC (Bartlett) t-statistic and standard error of the mean of x."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * (e[k:] @ e[:-k]) / n
    se = np.sqrt(v / n)
    return x.mean() / se, se


def block_bootstrap_ci(x, stat=np.mean, n_boot=2000, mean_block=8, alpha=0.05, seed=0):
    """Stationary-bootstrap percentile CI for stat(x)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(len(x), (n_boot, len(x)), mean_block, rng)
    draws = np.apply_along_axis(stat, 1, x[idx]) if stat is not np.mean else x[idx].mean(axis=1)
    return np.quantile(draws, [alpha / 2, 1 - alpha / 2])


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
