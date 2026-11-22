"""Statistics helpers: Sharpe ratios, stationary bootstrap, paired tests, Holm, Newey-West."""
from __future__ import annotations

import numpy as np


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(12))


def max_drawdown(x: np.ndarray) -> float:
    w = np.cumprod(1 + np.asarray(x, float))
    peak = np.maximum.accumulate(w)
    return float((w / peak - 1).min())


def stationary_bootstrap_idx(n: int, n_boot: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap: blocks of geometric length with mean `block`, wrapping around."""
    p = 1.0 / block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    new = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _sharpe_rows(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=1) / X.std(axis=1, ddof=1) * np.sqrt(12)


def paired_sharpe_test(a: np.ndarray, b: np.ndarray, n_boot: int, block: float, seed: int) -> dict:
    """Sharpe(a) - Sharpe(b) with a paired stationary bootstrap. CI: percentile. p-value: two-sided, from the
    re-centred bootstrap distribution (Ledoit & Wolf 2008 style)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(len(a), n_boot, block, rng)
    d = sharpe(a) - sharpe(b)
    db = _sharpe_rows(a[idx]) - _sharpe_rows(b[idx])
    lo, hi = np.percentile(db, [2.5, 97.5])
    p = float(np.mean(np.abs(db - d) >= abs(d)))
    return {"diff": d, "ci_lo": lo, "ci_hi": hi, "p": max(p, 1.0 / n_boot), "se": float(db.std(ddof=1))}


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def newey_west_ols(y: np.ndarray, x: np.ndarray, lags: int) -> dict:
    """OLS of y on [1, x] with Newey-West (Bartlett) standard errors."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(x, float)])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xu[L:].T @ Xu[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))
    return {"alpha": beta[0], "beta": beta[1], "se_beta": se[1], "t_beta": beta[1] / se[1], "r2": r2, "n": len(y)}
