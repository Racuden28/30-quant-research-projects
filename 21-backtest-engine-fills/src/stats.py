"""Statistics helpers. The stationary bootstrap, Holm and the Ledoit-Wolf HAC Sharpe-difference test are
copied from projects 03/19 so this folder runs on its own; the joint (panel) bootstrap is new."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, binomtest

TDAYS = 252


def sharpe(x, axis=0):
    x = np.asarray(x, float)
    return np.nanmean(x, axis=axis) / np.nanstd(x, axis=axis, ddof=1) * np.sqrt(TDAYS)


def stationary_bootstrap_idx(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices (circular)."""
    new = rng.random(n) < 1.0 / mean_block
    new[0] = True
    block_id = np.cumsum(new) - 1
    starts = rng.integers(n, size=block_id[-1] + 1)
    first = np.flatnonzero(new)
    offset = np.arange(n) - first[block_id]
    return (starts[block_id] + offset) % n


def boot_mean_ci(x: np.ndarray, B: int = 2000, block: float = 20.0, seed: int = 0, alpha: float = 0.05):
    """Mean of a daily series with a stationary-bootstrap percentile CI and a two-sided p-value
    (share of centred bootstrap means beyond the observed one)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    m = x.mean()
    bm = np.array([x[stationary_bootstrap_idx(len(x), block, rng)].mean() for _ in range(B)])
    lo, hi = np.quantile(bm, [alpha / 2, 1 - alpha / 2])
    p = float(np.mean(np.abs(bm - m) >= abs(m)))
    return float(m), float(lo), float(hi), max(p, 1.0 / B)


def panel_boot(X: np.ndarray, stat, B: int = 2000, block: float = 20.0, seed: int = 0) -> np.ndarray:
    """Resample whole dates (rows) of a (T, K) panel jointly, so cross-market correlation on the same
    day (e.g. March 2020) is kept. `stat` maps a (T, K) array to a vector; returns (B, len(stat))."""
    rng = np.random.default_rng(seed)
    T = X.shape[0]
    return np.array([stat(X[stationary_bootstrap_idx(T, block, rng)]) for _ in range(B)])


def sharpe_diff_hac(a: np.ndarray, b: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Ledoit-Wolf (2008) HAC test of SR(a) - SR(b) via the delta method. Returns (annualised diff, p)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m1, m2 = a.mean(), b.mean()
    g1, g2 = (a ** 2).mean(), (b ** 2).mean()
    d = m1 / np.sqrt(g1 - m1 ** 2) - m2 / np.sqrt(g2 - m2 ** 2)
    y = np.column_stack([a - m1, b - m2, a ** 2 - g1, b ** 2 - g2])
    n = len(a)
    S = y.T @ y / n
    for L in range(1, lags + 1):
        G = y[L:].T @ y[:-L] / n
        S += (1 - L / (lags + 1)) * (G + G.T)
    grad = np.array([g1 / (g1 - m1 ** 2) ** 1.5, -g2 / (g2 - m2 ** 2) ** 1.5,
                     -0.5 * m1 / (g1 - m1 ** 2) ** 1.5, 0.5 * m2 / (g2 - m2 ** 2) ** 1.5])
    se = np.sqrt(grad @ S @ grad / n)
    return float(d * np.sqrt(TDAYS)), float(2 * (1 - norm.cdf(abs(d / se))))


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, (m - i) * p[k])
        adj[k] = min(1.0, running)
    return adj


def sign_test(values) -> tuple[int, int, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & (v != 0)]
    k = int((v > 0).sum())
    return k, len(v), float(binomtest(k, len(v), 0.5).pvalue) if len(v) else np.nan
