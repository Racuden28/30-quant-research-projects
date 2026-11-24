"""Small statistics helpers: Sharpe ratios, stationary bootstrap, HAC t-stats, Holm."""
from __future__ import annotations
import numpy as np


def sharpe(x, periods):
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def stationary_bootstrap_idx(n, mean_block, reps, rng):
    """Politis-Romano stationary bootstrap indices (reps x n)."""
    p = 1.0 / mean_block
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    new = rng.random((reps, n)) < p
    starts = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def sharpe_ci(x, periods, reps=1000, mean_block=20, seed=0, level=0.95):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(len(x), mean_block, reps, rng)
    xb = x[idx]
    sr = xb.mean(1) / xb.std(1, ddof=1) * np.sqrt(periods)
    a = (1 - level) / 2
    return float(np.quantile(sr, a)), float(np.quantile(sr, 1 - a))


def hac_t(x, lags=None):
    """Newey-West t-statistic of the mean."""
    x = np.asarray(x, float); n = len(x)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    u = x - x.mean()
    s = u @ u / n
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        s += 2 * w * (u[l:] @ u[:-l]) / n
    return float(x.mean() / np.sqrt(s / n))


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj
