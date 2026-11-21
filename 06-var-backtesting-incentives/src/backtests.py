"""Statistical backtests, scoring functions and the Basel / FRTB capital rules."""
from __future__ import annotations

import numpy as np
from scipy import stats

# ------------------------------------------------------------------ coverage tests


def kupiec(hits: np.ndarray, p: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures LR test. Returns (LR, p-value)."""
    n, x = len(hits), int(hits.sum())
    ph = x / n
    ll0 = (n - x) * np.log1p(-p) + x * np.log(p)
    ll1 = ((n - x) * np.log1p(-ph) if x < n else 0.0) + (x * np.log(ph) if x > 0 else 0.0)
    lr = -2 * (ll0 - ll1)
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen_ind(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) first-order Markov independence LR test."""
    h0, h1 = hits[:-1].astype(int), hits[1:].astype(int)
    n00 = np.sum((h0 == 0) & (h1 == 0)); n01 = np.sum((h0 == 0) & (h1 == 1))
    n10 = np.sum((h0 == 1) & (h1 == 0)); n11 = np.sum((h0 == 1) & (h1 == 1))

    def xlogy(a, b):
        return a * np.log(b) if a > 0 else 0.0

    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    l0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    lr = max(-2 * (l0 - l1), 0.0)
    return lr, float(stats.chi2.sf(lr, 1))


def dq_test(hits: np.ndarray, var: np.ndarray, p: float, lags: int = 4) -> tuple[float, float]:
    """Engle-Manganelli (2004) dynamic quantile test: regress Hit_t - p on a constant,
    `lags` lagged hits and the VaR forecast. DQ ~ chi2(lags + 2) under H0."""
    hit = hits.astype(float) - p
    y = hit[lags:]
    X = [np.ones_like(y)] + [hit[lags - k:-k] for k in range(1, lags + 1)] + [var[lags:]]
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    dq = float(beta @ X.T @ X @ beta / (p * (1 - p)))
    return dq, float(stats.chi2.sf(dq, X.shape[1]))


def newey_west_se(x: np.ndarray, lags: int | None = None) -> float:
    """HAC standard error of the mean of x (Bartlett kernel)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    g0 = x @ x / n
    s = g0
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    return float(np.sqrt(max(s, 1e-300) / n))


def acerbi_szekely_z2(ret: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> tuple[float, float]:
    """Acerbi-Szekely (2014) Z2 = 1 - mean(L_t 1{L_t > VaR_t} / (alpha ES_t)); E[Z2]=0 under H0,
    Z2 < 0 means ES is too small. One-sided p-value from a HAC t-statistic (no simulation needed,
    so it works for non-parametric models too)."""
    L = -ret
    x = L * (L > var) / (alpha * es) - 1.0
    z2 = -float(np.mean(x))
    t = z2 / newey_west_se(x)
    return z2, float(stats.norm.cdf(t))


# ------------------------------------------------------------------ scoring functions


def tick_loss(ret: np.ndarray, var: np.ndarray, alpha: float) -> np.ndarray:
    """Quantile (pinball) loss of the alpha-quantile -var. Strictly consistent for VaR."""
    q = -var
    return (alpha - (ret < q)) * (ret - q)


def fz0_loss(ret: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> np.ndarray:
    """Fissler-Ziegel FZ0 loss (Patton, Ziegel & Chen 2019) for the pair (VaR_alpha, ES_alpha),
    strictly consistent and homogeneous of degree zero. var, es are positive loss numbers."""
    v, e = -var, -es
    return -1.0 / (alpha * e) * (ret <= v) * (v - ret) + v / e + np.log(-e) - 1.0


def dm_test(d: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano on a loss differential d (HAC). Returns (t, two-sided p)."""
    t = float(np.mean(d) / newey_west_se(d))
    return t, float(2 * stats.norm.sf(abs(t)))


# ------------------------------------------------------------------ capital rules

# Basel Committee (1996), "Supervisory framework for the use of backtesting", Table 2.
# Plus factor added to the multiplier of 3, by number of 99% exceptions in the last 250 days.
BASEL_PLUS = np.array([0, 0, 0, 0, 0, 0.40, 0.50, 0.65, 0.75, 0.85] + [1.00] * 241)
# Basel FRTB, MAR99.  Plus factor added to the multiplier of 1.5 (applied to ES).
FRTB_PLUS = np.array([0, 0, 0, 0, 0, 0.20, 0.26, 0.33, 0.38, 0.42] + [0.50] * 241)
RED = 10
QUARTER = 63


def trailing_exceptions(hits: np.ndarray, window: int = 250) -> np.ndarray:
    """N[t] = exceptions in days t-window .. t-1 (known before day t). hits: (..., T)."""
    c = np.concatenate([np.zeros(hits.shape[:-1] + (1,)), np.cumsum(hits, axis=-1)], axis=-1)
    t = np.arange(hits.shape[-1])
    return c[..., t] - c[..., np.maximum(t - window, 0)]


def capital_path(ret: np.ndarray, var99: np.ndarray, risk10: np.ndarray, lam: np.ndarray,
                 base: float, plus_table: np.ndarray, burn: int = 250) -> dict[str, np.ndarray]:
    """Internal-models market-risk charge for a bank that reports lam * (its model's numbers).

    ret, var99, risk10 : (T,) realised return, model 1-day 99% VaR, and the 10-day risk measure
                         the charge is based on (sqrt(10) * VaR99 for Basel 2.5, sqrt(10) * ES97.5 for FRTB).
    lam                : (L,) grid of reporting scale factors.
    The supervisor counts exceptions of the *reported* VaR over the last 250 days and resets the
    plus factor once a quarter (every 63 trading days), as in the 1996 framework.

    charge_t = max(lam*R_{t-1}, (base + plus_t) * mean(lam*R_{t-60..t-1}))
    Returns per-lambda arrays: mean charge after burn-in, share of quarters in the yellow / red zone.
    """
    lam = np.asarray(lam)[:, None]
    hits = (ret[None, :] < -lam * var99[None, :]).astype(float)
    N = trailing_exceptions(hits)
    T = len(ret)
    q_idx = np.arange(T) - (np.arange(T) - burn) % QUARTER       # last quarterly reset date
    q_idx = np.clip(q_idx, 0, T - 1)
    Nq = N[:, q_idx]
    mult = base + plus_table[Nq.astype(int)]
    R = lam * risk10[None, :]
    cs = np.concatenate([np.zeros((lam.shape[0], 1)), np.cumsum(R, axis=1)], axis=1)
    t = np.arange(T)
    lo = np.maximum(t - 60, 0)
    avg60 = (cs[:, t] - cs[:, lo]) / np.maximum(t - lo, 1)
    prev = np.concatenate([R[:, :1], R[:, :-1]], axis=1)
    charge = np.maximum(prev, mult * avg60)
    sl = slice(burn, None)
    resets = np.arange(burn, T, QUARTER)
    return dict(charge=charge[:, sl].mean(axis=1),
                yellow=((N[:, resets] >= 5) & (N[:, resets] < RED)).mean(axis=1),
                red=(N[:, resets] >= RED).mean(axis=1),
                exc_rate=hits[:, sl].mean(axis=1))
