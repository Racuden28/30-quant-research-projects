"""Estimators and long-only optimisers used everywhere in project 05.

All optimisers are long-only and fully invested (w >= 0, sum w = 1), which is how mean-variance is
used in practice and keeps every portfolio implementable without leverage.
"""
from __future__ import annotations

import numpy as np
import quadprog
from scipy import stats

# ------------------------------------------------------------------ estimators


def sample_mean(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0)


def sample_cov(x: np.ndarray) -> np.ndarray:
    return np.cov(x, rowvar=False, ddof=1)


def ledoit_wolf(x: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) shrinkage towards the scaled identity, analytic intensity.

    Same estimator as sklearn.covariance.LedoitWolf (tested against it), written out so the
    project has no hidden dependency and the formula is visible.
    """
    t, n = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / t
    mu = np.trace(s) / n
    delta = s.copy()
    delta[np.diag_indices(n)] -= mu
    d2 = (delta ** 2).sum() / n
    x2 = xc ** 2
    b2_bar = ((x2.T @ x2) / t - s ** 2).sum() / (n * t)
    b2 = min(b2_bar, d2)
    shrink = b2 / d2 if d2 > 0 else 1.0
    return shrink * mu * np.eye(n) + (1 - shrink) * s


def bayes_stein_mean(x: np.ndarray) -> np.ndarray:
    """Jorion (1986) Bayes-Stein mean: shrink towards the minimum-variance portfolio's mean."""
    t, n = x.shape
    m = x.mean(axis=0)
    s = sample_cov(x) * (t - 1) / max(t - n - 2, 1)
    si = np.linalg.inv(s)
    one = np.ones(n)
    m0 = one @ si @ m / (one @ si @ one)
    d = m - m0
    phi = (n + 2) / ((n + 2) + t * d @ si @ d)
    return (1 - phi) * m + phi * m0


# ------------------------------------------------------------------ optimisers


def _qp(g: np.ndarray, a: np.ndarray, c: np.ndarray, b: np.ndarray, meq: int) -> np.ndarray:
    g = 0.5 * (g + g.T)
    for ridge in (1e-12, 1e-8, 1e-5):          # near-singular sample covariances when N ~ T
        try:
            return quadprog.solve_qp(g + np.eye(len(g)) * ridge * np.trace(g) / len(g), a, c, b, meq)[0]
        except ValueError as e:
            if "positive definite" not in str(e):
                raise
    raise ValueError("covariance not positive definite even after ridge")


def min_variance(cov: np.ndarray) -> np.ndarray:
    n = len(cov)
    c = np.column_stack([np.ones(n), np.eye(n)])
    b = np.r_[1.0, np.zeros(n)]
    w = _qp(cov, np.zeros(n), c, b, 1)
    return _clean(w)


def max_sharpe(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Long-only tangency portfolio (excess-return means).

    Solved as min y'Σy s.t. μ'y = 1, y >= 0, then w = y / sum(y), which is exact for long-only
    max-Sharpe when some μ_i > 0. If no asset has a positive expected excess return the problem has
    no positive-Sharpe solution; we then hold the minimum-variance portfolio (stated in the README).
    Equal means for all assets give exactly the minimum-variance portfolio, which is the
    "no-information" row of the factorial.
    """
    n = len(mu)
    if np.ptp(mu) < 1e-14 * max(1.0, np.abs(mu).max()):
        return min_variance(cov)
    if mu.max() <= 0:
        return min_variance(cov)
    scale = np.abs(mu).max()
    c = np.column_stack([mu / scale, np.eye(n)])
    b = np.r_[1.0, np.zeros(n)]
    y = _qp(cov, np.zeros(n), c, b, 1)
    return _clean(y)


def max_sharpe_short(mu: np.ndarray, cov: np.ndarray, short: float = 0.3) -> np.ndarray:
    """Max-Sharpe with limited shorting: w_i >= -short, sum w = 1 (short=0.3 allows up to 30% short per asset).

    Homogeneous form: min y'Σy s.t. μ'y = 1, y_i + short * 1'y >= 0, 1'y >= 0; then w = y / 1'y.
    Falls back to the long-only solution if the problem is infeasible (no positive-Sharpe portfolio).
    """
    n = len(mu)
    if np.ptp(mu) < 1e-14 * max(1.0, np.abs(mu).max()) or mu.max() <= 0:
        return max_sharpe(mu, cov)
    scale = np.abs(mu).max()
    c = np.column_stack([mu / scale, np.eye(n) + short * np.ones((n, n)), np.ones(n)])
    b = np.r_[1.0, np.zeros(n), 1e-9]
    try:
        y = _qp(cov, np.zeros(n), c, b, 1)
    except ValueError:
        return max_sharpe(mu, cov)
    return y / y.sum()


def resampled_max_sharpe(mu: np.ndarray, cov: np.ndarray, t: int, n_draws: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Michaud (1998) resampled efficiency, max-Sharpe variant.

    Each draw takes the sampling distribution of (mean, covariance) for a Gaussian sample of
    length t: mean ~ N(mu, cov/t), cov ~ Wishart(t-1, cov/(t-1)). Weights are averaged.
    """
    w = np.zeros(len(mu))
    covs = stats.wishart(df=t - 1, scale=cov / (t - 1)).rvs(size=n_draws, random_state=rng)
    chol = np.linalg.cholesky(cov / t)
    for k in range(n_draws):
        m = mu + chol @ rng.standard_normal(len(mu))
        w += max_sharpe(m, covs[k])
    return w / n_draws


def inverse_vol(cov: np.ndarray) -> np.ndarray:
    iv = 1 / np.sqrt(np.diag(cov))
    return iv / iv.sum()


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def _clean(w: np.ndarray) -> np.ndarray:
    w = np.where(w < 1e-10, 0.0, w)
    return w / w.sum()


def sharpe(mu: np.ndarray, cov: np.ndarray, w: np.ndarray) -> float:
    return float(w @ mu / np.sqrt(w @ cov @ w))
