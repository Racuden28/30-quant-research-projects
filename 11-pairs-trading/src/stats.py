"""Small inference helpers: Newey-West t-statistic for a mean, stationary block-bootstrap CI, Holm."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _boot_means(x, mean_block, draws, seed):
    np.random.seed(seed)
    n = len(x)
    p = 1.0 / mean_block
    out = np.empty(draws)
    for d in range(draws):
        i = np.random.randint(n)
        tot = x[i]
        for t in range(1, n):
            if np.random.random() < p:
                i = np.random.randint(n)
            else:
                i = (i + 1) % n
            tot += x[i]
        out[d] = tot / n
    return out


def nw_t(x, lags: int = 6) -> tuple[float, float]:
    """Mean and Newey-West (Bartlett) t-statistic of the mean of x."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    m = x.mean()
    e = x - m
    v = e @ e / n
    for L in range(1, min(lags, n - 1) + 1):
        w = 1 - L / (lags + 1)
        v += 2 * w * (e[L:] @ e[:-L]) / n
    se = np.sqrt(max(v, 1e-300) / n)
    return m, m / se


def block_boot_ci(x, stat=np.mean, mean_block: float = 12, draws: int = 5000, seed: int = 0,
                  level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of stat(x) from a Politis-Romano stationary bootstrap."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if stat is np.mean:
        out = _boot_means(x, float(mean_block), int(draws), int(seed))
    else:
        rng = np.random.default_rng(seed)
        out = np.empty(draws)
        for d in range(draws):
            idx = np.empty(n, np.int64)
            idx[0] = rng.integers(n)
            new = rng.random(n) < 1.0 / mean_block
            st = rng.integers(0, n, n)
            for t in range(1, n):
                idx[t] = st[t] if new[t] else (idx[t - 1] + 1) % n
            out[d] = stat(x[idx])
    a = (1 - level) / 2
    return float(np.quantile(out, a)), float(np.quantile(out, 1 - a))


def sharpe(x, periods: int = 12) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(x.mean() / x.std(ddof=1) * np.sqrt(periods)) if len(x) > 2 and x.std() > 0 else np.nan


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        adj[j] = min(1.0, run)
    return adj
