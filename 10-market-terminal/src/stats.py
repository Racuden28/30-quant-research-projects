"""Out-of-sample forecasting machinery: expanding OLS with embargo, R²_OS, Clark–West with HAC, BH-FDR."""
from __future__ import annotations

import numpy as np
from scipy import stats


def expanding_oos(y: np.ndarray, X: np.ndarray | None, h: int, start: int, step: int) -> np.ndarray:
    """One-step-ahead-style forecasts of y[t] for t >= start.

    y[t] is the target observed at t+h (forward window t+1..t+h). At refit date k only rows i with i+h <= k
    (target fully observed) enter the fit. X=None gives the prevailing mean. Forecasts for t in [k, k+step)
    use the model fitted at k. Rows with NaN y or X are skipped in fitting and get NaN forecasts.
    """
    n = len(y)
    f = np.full(n, np.nan)
    Z = np.ones((n, 1)) if X is None else np.column_stack([np.ones(n), X])
    ok = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    for k in range(start, n, step):
        m = ok.copy()
        m[max(k - h + 1, 0):] = False  # embargo: i + h <= k
        if m.sum() < 50:
            continue
        beta, *_ = np.linalg.lstsq(Z[m], y[m], rcond=None)
        sl = slice(k, min(k + step, n))
        zz = Z[sl]
        good = np.all(np.isfinite(zz), axis=1)
        fc = np.full(len(zz), np.nan)
        fc[good] = zz[good] @ beta
        f[sl] = fc
    return f


def r2_os(y, f_model, f_base) -> float:
    m = np.isfinite(y) & np.isfinite(f_model) & np.isfinite(f_base)
    return 1 - np.sum((y[m] - f_model[m]) ** 2) / np.sum((y[m] - f_base[m]) ** 2)


def cw_series(y, f_model, f_base) -> np.ndarray:
    """Clark–West adjusted loss differential; positive values favour the (nesting) model."""
    eb, em = y - f_base, y - f_model
    return eb ** 2 - (em ** 2 - (f_base - f_model) ** 2)


def hac_mean_t(x: np.ndarray, lag: int) -> tuple[float, float]:
    """Mean of x and its Newey–West (Bartlett) t-statistic."""
    x = x[np.isfinite(x)]
    n = len(x)
    mu = x.mean()
    e = x - mu
    v = e @ e / n
    for L in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n
    se = np.sqrt(max(v, 1e-300) / n)
    return mu, mu / se


def cw_test(fseries: np.ndarray, lag: int) -> tuple[float, float]:
    """One-sided Clark–West: returns (t-stat, p-value) for H0 'model does not improve'."""
    _, t = hac_mean_t(fseries, lag)
    return t, 1 - stats.norm.cdf(t)


def bh(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Benjamini–Hochberg: boolean mask of rejections and adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    return adj <= q, adj
