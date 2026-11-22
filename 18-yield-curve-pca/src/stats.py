"""Inference helpers: HAC Diebold-Mariano, moving-block bootstrap of a ratio of means, Holm."""
from __future__ import annotations

import numpy as np
from scipy import stats


def nw_var(x: np.ndarray, lags: int) -> float:
    """Newey-West long-run variance of a series (Bartlett kernel)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    v = x @ x / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        v += 2 * w * (x[L:] @ x[:-L]) / n
    return v


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = 21):
    """Two-sided Diebold-Mariano test of equal mean loss. Returns (t, p). Negative t: a is better."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    se = np.sqrt(nw_var(d, lags) / len(d))
    t = d.mean() / se if se > 0 else 0.0
    return float(t), float(2 * stats.norm.sf(abs(t)))


def block_indices(n: int, block: int, reps: int, rng) -> np.ndarray:
    """Moving-block bootstrap index matrix (reps x n)."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    return idx


def ratio_ci(a: np.ndarray, b: np.ndarray, block: int = 63, reps: int = 1000, seed: int = 18,
             level: float = 0.95):
    """Point estimate and percentile CI of mean(a)/mean(b), resampling days in blocks jointly."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = block_indices(len(a), block, reps, rng)
    r = a[idx].mean(1) / b[idx].mean(1)
    lo, hi = np.quantile(r, [(1 - level) / 2, (1 + level) / 2])
    return float(a.mean() / b.mean()), float(lo), float(hi)


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj
