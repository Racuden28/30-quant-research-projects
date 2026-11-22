"""Small, tested statistical toolkit: OLS with Newey-West errors, GRS, tangency portfolios,
bias-adjusted squared Sharpe ratios, the stationary bootstrap and Holm's correction."""
from __future__ import annotations

import numpy as np
from scipy import stats

NW_LAGS = 6


def ols_nw(y: np.ndarray, X: np.ndarray, lags: int = NW_LAGS, const: bool = True):
    """OLS coefficients and Newey-West (Bartlett) standard errors. Returns (coef, se, resid)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if const:
        X = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    u = X * e[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return b, np.sqrt(np.diag(V)), e


def mean_nw(x: np.ndarray, lags: int = NW_LAGS):
    b, se, _ = ols_nw(x, np.empty((len(x), 0)), lags)
    return b[0], se[0]


def grs(R: np.ndarray, F: np.ndarray):
    """Gibbons-Ross-Shanken test. R: T x N excess returns, F: T x K factors.
    Returns (GRS statistic, p-value, alphas, A|a|, Sh2(a))."""
    R, F = np.asarray(R, float), np.asarray(F, float)
    T, N = R.shape
    K = F.shape[1]
    X = np.column_stack([np.ones(T), F])
    B = np.linalg.lstsq(X, R, rcond=None)[0]
    a = B[0]
    E = R - X @ B
    Sigma = E.T @ E / (T - K - 1)
    mu_f = F.mean(0)
    Omega = np.atleast_2d(np.cov(F, rowvar=False, ddof=1))
    sh2_f = mu_f @ np.linalg.solve(Omega, mu_f)
    sh2_a = a @ np.linalg.solve(Sigma, a)
    stat = (T / N) * ((T - N - K) / (T - K - 1)) * sh2_a / (1 + sh2_f)
    p = 1 - stats.f.cdf(stat, N, T - N - K)
    return stat, p, a, np.mean(np.abs(a)), sh2_a


def tangency(mu: np.ndarray, Sigma: np.ndarray, target_vol: float | None = None):
    """Unnormalised tangency weights Sigma^-1 mu, optionally scaled to an ex-ante volatility."""
    w = np.linalg.solve(np.atleast_2d(Sigma), np.atleast_1d(mu))
    if target_vol is not None:
        w = w * target_vol / np.sqrt(w @ np.atleast_2d(Sigma) @ w)
    return w


def sh2(F: np.ndarray) -> float:
    F = np.atleast_2d(np.asarray(F, float).T).T
    mu = F.mean(0)
    S = np.atleast_2d(np.cov(F, rowvar=False, ddof=1))
    return float(mu @ np.linalg.solve(S, mu))


def sh2_adjusted(F: np.ndarray) -> float:
    """Unbiased estimator of the population squared Sharpe ratio of the tangency portfolio
    (Kan & Zhou 2007; Ferson & Siegel 2003): ((T-N-2)/T) * sh2 - N/T."""
    F = np.atleast_2d(np.asarray(F, float).T).T
    T, N = F.shape
    return (T - N - 2) / T * sh2(F) - N / T


def sharpe(x: np.ndarray, periods: int = 12) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(periods))


def stationary_bootstrap(T: int, n: int, mean_block: float = 6.0, seed: int = 17) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix of shape (n, T)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    idx = np.empty((n, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, n)
    new = rng.random((n, T)) < p
    jumps = rng.integers(0, T, (n, T))
    for t in range(1, T):
        idx[:, t] = np.where(new[:, t], jumps[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def mde(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable effect of a two-sided z-test with standard error se."""
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)


def joint_shift_wald(Y: np.ndarray, d: np.ndarray, lags: int = NW_LAGS):
    """Wald test that the mean shift of every column of Y between d=0 and d=1 is zero, with a
    cross-equation Newey-West covariance. Returns (shifts, chi2, p)."""
    Y = np.asarray(Y, float)
    T, J = Y.shape
    X = np.column_stack([np.ones(T), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    B = XtX_inv @ X.T @ Y
    E = Y - X @ B
    U = np.concatenate([X * E[:, [j]] for j in range(J)], axis=1)
    Sm = U.T @ U
    for L in range(1, lags + 1):
        G = U[L:].T @ U[:-L]
        Sm += (1 - L / (lags + 1)) * (G + G.T)
    A = np.kron(np.eye(J), XtX_inv)
    V = A @ Sm @ A
    sel = [2 * j + 1 for j in range(J)]
    b = B[1]
    W = float(b @ np.linalg.solve(V[np.ix_(sel, sel)], b))
    return b, W, float(1 - stats.chi2.cdf(W, J))
