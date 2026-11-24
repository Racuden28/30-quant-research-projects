"""Probabilistic and Deflated Sharpe ratios, and estimators of the effective number of trials.

All Sharpe ratios here are PER PERIOD (not annualised) unless a name says otherwise.
References: Bailey & Lopez de Prado (2012, 2014); Lopez de Prado & Lewis (2019);
Nyholt (2004); Li & Ji (2005); Mertens (2002).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

EULER = 0.5772156649015329


# ---------------------------------------------------------------- moments
def moments(x: np.ndarray, axis: int = 0):
    """Mean, sd (ddof=1), skewness and raw (non-excess) kurtosis along `axis`."""
    x = np.asarray(x, dtype=float)
    m = x.mean(axis=axis, keepdims=True)
    d = x - m
    m2 = (d ** 2).mean(axis=axis)
    n = x.shape[axis]
    sd = np.sqrt(m2 * n / (n - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = (d ** 3).mean(axis=axis) / m2 ** 1.5
        kurt = (d ** 4).mean(axis=axis) / m2 ** 2
    return np.squeeze(m, axis=axis), sd, skew, kurt


def sr_se(sr, T, skew=0.0, kurt=3.0):
    """Standard error of a per-period Sharpe ratio under non-normal IID returns (Mertens 2002)."""
    v = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    return np.sqrt(np.maximum(v, 1e-12) / (T - 1.0))


# ---------------------------------------------------------------- PSR / DSR
def psr(sr, T, skew=0.0, kurt=3.0, sr_star=0.0):
    """Probabilistic Sharpe ratio: P(true SR > sr_star) given the estimate."""
    return stats.norm.cdf((sr - sr_star) / sr_se(sr, T, skew, kurt))


def expected_max_sr(n_trials, var_sr):
    """Expected maximum of n_trials IID N(0, var_sr) Sharpe estimates (False Strategy Theorem)."""
    n = np.asarray(n_trials, dtype=float)
    out = np.zeros(np.broadcast(n, np.asarray(var_sr)).shape)
    ok = n > 1.0
    nn = np.where(ok, n, 2.0)
    e = (1 - EULER) * stats.norm.ppf(1 - 1 / nn) + EULER * stats.norm.ppf(1 - 1 / (nn * np.e))
    out = np.where(ok, np.sqrt(var_sr) * e, 0.0)
    return out if out.shape else float(out)


def dsr(sr, T, skew, kurt, n_trials, var_sr):
    """Deflated Sharpe ratio and the benchmark SR0 it deflates against."""
    sr0 = expected_max_sr(n_trials, var_sr)
    return psr(sr, T, skew, kurt, sr0), sr0


def min_track_record(sr, sr_star, skew=0.0, kurt=3.0, alpha=0.05):
    """Minimum track record length (periods) for PSR(sr_star) >= 1-alpha."""
    z = stats.norm.ppf(1 - alpha)
    v = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    with np.errstate(divide="ignore"):
        return np.where(sr > sr_star, 1 + v * (z / (sr - sr_star)) ** 2, np.inf)


# ---------------------------------------------------------------- effective number of trials
def corr_eigs(R: np.ndarray):
    """Eigenvalues of the correlation matrix of the columns of R (T x N)."""
    C = np.corrcoef(R, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    return np.clip(np.linalg.eigvalsh(C), 0.0, None), C


def neff_nyholt(lam):
    n = lam.size
    return float(np.clip(1 + (n - 1) * (1 - np.var(lam, ddof=1) / n), 1, n))


def neff_liji(lam):
    lam = np.abs(lam)
    return float(np.clip(np.sum((lam >= 1) + (lam - np.floor(lam))), 1, lam.size))


def neff_participation(lam):
    return float(lam.sum() ** 2 / np.sum(lam ** 2))


def cluster_trials(C: np.ndarray, kmax: int = 60, absolute: bool = False):
    """Hierarchical (average-linkage) clustering of trials, k chosen by mean silhouette.

    Distance sqrt((1-rho)/2) as in Lopez de Prado (2019). With absolute=True a strategy and its
    mirror image (rho=-1) land in the same cluster.
    Returns (k, labels in 0..k-1).
    """
    n = C.shape[0]
    if n <= 3:
        return n, np.arange(n)
    rho = np.abs(C) if absolute else C
    D = np.sqrt(np.clip((1 - rho) / 2, 0, None))
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    best = (-np.inf, 1, np.zeros(n, int))
    for k in range(2, min(kmax, n - 1) + 1):
        lab = fcluster(Z, k, criterion="maxclust") - 1
        kk = lab.max() + 1
        if kk < 2:
            continue
        s = silhouette(D, lab, kk)
        if s > best[0]:
            best = (s, kk, lab)
    return best[1], best[2]


def silhouette(D, lab, k):
    n = D.shape[0]
    onehot = np.zeros((n, k))
    onehot[np.arange(n), lab] = 1.0
    sizes = onehot.sum(0)
    sums = D @ onehot  # n x k: sum of distances to each cluster
    own = sizes[lab]
    a = np.where(own > 1, sums[np.arange(n), lab] / np.maximum(own - 1, 1), 0.0)
    mean_other = sums / np.where(sizes > 0, sizes, np.inf)
    mean_other[np.arange(n), lab] = np.inf
    b = mean_other.min(1)
    s = np.where(own > 1, (b - a) / np.maximum(a, b), 0.0)
    return float(s.mean())


def cluster_sharpes(R: np.ndarray, lab: np.ndarray):
    """Sharpe of each cluster's equal-risk (inverse-vol) composite return."""
    sd = R.std(0, ddof=1)
    Z = R / np.where(sd > 0, sd, 1.0)
    k = lab.max() + 1
    out = np.empty(k)
    for j in range(k):
        c = Z[:, lab == j].mean(1)
        s = c.std(ddof=1)
        out[j] = c.mean() / s if s > 0 else 0.0
    return out


def implied_n(sr, T, skew, kurt, var_sr, target=0.05, thr=0.95, nmax=1e7):
    """Smallest N for which the share of worlds with DSR > thr is <= target (bisection on log N)."""
    def fpr(n):
        return np.mean(dsr(sr, T, skew, kurt, n, var_sr)[0] > thr)
    if fpr(1.0) <= target:
        return 1.0
    lo, hi = 0.0, np.log(nmax)
    if fpr(nmax) > target:
        return np.inf
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fpr(np.exp(mid)) > target:
            lo = mid
        else:
            hi = mid
    return float(np.exp(hi))
