"""Stationary bootstrap and multiple-testing helpers."""
from __future__ import annotations

import numpy as np

SEED = 20261122
N_BOOT = 2000
MEAN_BLOCK = 12          # months


def stationary_indices(T: int, n_boot: int = N_BOOT, mean_block: float = MEAN_BLOCK,
                       seed: int = SEED) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap index matrix of shape (n_boot, T)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    idx = np.empty((n_boot, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, n_boot)
    new_block = rng.random((n_boot, T)) < p
    jumps = rng.integers(0, T, (n_boot, T))
    for t in range(1, T):
        idx[:, t] = np.where(new_block[:, t], jumps[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def boot_means(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Bootstrap means of a 1-d series (NaNs treated as missing)."""
    v = x[idx]
    return np.nanmean(v, axis=1)


def ci(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    d = draws[np.isfinite(draws)]
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a))


def p_greater(draws: np.ndarray, point: float, null: float = 0.0) -> float:
    """One-sided bootstrap p-value for H1: stat > null, by re-centring the draws at the null."""
    d = draws[np.isfinite(draws)]
    centred = d - point + null
    return float((np.sum(centred >= point) + 1) / (len(d) + 1))


def holm(p: dict[str, float]) -> dict[str, float]:
    keys = sorted(p, key=p.get)
    m = len(keys)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(12)) if len(x) > 2 else np.nan


def boot_sharpe(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    v = x[idx]
    return v.mean(axis=1) / v.std(axis=1, ddof=1) * np.sqrt(12)
