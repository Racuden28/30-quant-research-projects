"""Hedge-ratio models for a book of key-rate DV01 exposures.

Notation. A book holds DV01 exposures w_T (dollars per basis point) at the *target* maturities T.
The desk hedges with positions h at the *hedge* maturities H (think 2Y/5Y/10Y/30Y futures or
on-the-run notes). With daily yield changes dy (bp), the hedged P&L is

    pnl_t = -(dy_T,t' w_T + dy_H,t' h).

Every model below returns a matrix A (|H| x |T|) such that h = -A w_T, so the hedged book's
exposure to dy is the "residual design" row  e_t = dy_T,t - dy_H,t A  and pnl_t = -e_t' w_T.
Because everything is linear in w, one residual design serves every possible book.

Models (k = number of hedge instruments = number of factors neutralised):
  interp   key-rate bucketing: each target DV01 is split between the two neighbouring hedge points by
           linear interpolation in maturity (flat beyond the ends). Model-free desk standard.
  ns       Nelson-Siegel factor neutrality with the Diebold-Li decay (lambda = 0.0609 per month).
           Loadings are fixed, nothing is estimated.
  pca      neutralise the first k principal components of the trailing covariance of dy (bp).
  pca_vol  same, but the PCA is run on EWMA-volatility-standardised changes and the loadings are
           rescaled by today's EWMA volatilities, so the factor shapes follow the volatility regime.
  fm<j>    minimum-variance hedge under a j-factor PCA covariance model, Sigma = V_j L_j V_j' + Psi
           (Psi diagonal = what the j factors leave). j is chosen independently of k: PCA used as a
           covariance model rather than as a list of risks to neutralise.
  fmv<j>   same on EWMA-standardised changes, rescaled by today's EWMA volatilities.
  ols      minimum-variance regression hedge on the trailing window (uses all k instruments); this
           equals fm<n>, the factor model with every component kept.
"""
from __future__ import annotations

import numpy as np

NS_LAMBDA = 0.0609 * 12  # Diebold & Li (2006) decay, per year of maturity


# ---------------------------------------------------------------- loadings
def ns_loadings(mats_years: np.ndarray, k: int = 3, lam: float = NS_LAMBDA) -> np.ndarray:
    """Nelson-Siegel level, slope and curvature loadings (n x k); k = 4 adds a Svensson hump."""
    x = lam * np.asarray(mats_years, dtype=float)
    f1 = (1 - np.exp(-x)) / x
    x2 = x / 5  # Svensson second hump (peaks near 12 years), used only when k = 4
    f2 = (1 - np.exp(-x2)) / x2
    L = np.column_stack([np.ones_like(x), f1, f1 - np.exp(-x), f2 - np.exp(-x2)])
    return L[:, :k]


def pca(cov: np.ndarray, k: int):
    """Top-k eigenvectors (n x k) and all eigenvalues (descending) of a symmetric matrix.

    Signs are normalised so that PC1 is positive on average (a level shift up), PC2 rises with
    maturity (steepening) and PC3 is positive in the belly (higher in the middle than at the ends).
    """
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    n = cov.shape[0]
    ramp = np.linspace(-1, 1, n)
    hump = 1 - ramp ** 2 - np.mean(1 - ramp ** 2)
    refs = [np.ones(n), ramp, hump]
    for j in range(min(k, 3)):
        if vecs[:, j] @ refs[j] < 0:
            vecs[:, j] *= -1
    return vecs[:, :k], vals


def ewma_vol(dy: np.ndarray, lam: float = 0.97, warm: int = 63) -> np.ndarray:
    """EWMA volatility known at the close of day t-1 for day t (no look-ahead), shape like dy.

    sigma2_t = lam * sigma2_{t-1} + (1 - lam) * dy_{t-1}^2, seeded with the mean square of the
    first `warm` days. Rows before `warm` use that seed (callers drop them). A floor of 0.25 bp
    stops zero-lower-bound days (bill yields that do not move for weeks) from producing huge
    standardised changes.
    """
    dy = np.asarray(dy, dtype=float)
    s2 = np.empty_like(dy)
    s2[0] = np.mean(dy[:warm] ** 2, axis=0)
    for t in range(1, len(dy)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * dy[t - 1] ** 2
    return np.sqrt(np.maximum(s2, 0.25 ** 2))


# ---------------------------------------------------------------- hedge matrices
def solve_factor_hedge(L: np.ndarray, iT, iH) -> np.ndarray:
    """A such that h = -A w_T makes the book neutral to the factors in L (n x k), k = |H|."""
    LH, LT = L[iH], L[iT]
    return np.linalg.solve(LH.T, LT.T)


def interp_matrix(mT: np.ndarray, mH: np.ndarray) -> np.ndarray:
    """Linear key-rate bucketing in maturity, flat extrapolation beyond the outermost hedges."""
    A = np.zeros((len(mH), len(mT)))
    for j, m in enumerate(mT):
        if m <= mH[0]:
            A[0, j] = 1
        elif m >= mH[-1]:
            A[-1, j] = 1
        else:
            i = np.searchsorted(mH, m) - 1
            a = (mH[i + 1] - m) / (mH[i + 1] - mH[i])
            A[i, j], A[i + 1, j] = a, 1 - a
    return A


def ols_matrix(dyT: np.ndarray, dyH: np.ndarray) -> np.ndarray:
    """Minimum-variance hedge: regress each target column on the hedge columns (no intercept)."""
    return np.linalg.lstsq(dyH, dyT, rcond=None)[0]


def factor_cov(S: np.ndarray, j: int) -> np.ndarray:
    """j-factor PCA covariance with a diagonal residual: V_j L_j V_j' + diag(S - V_j L_j V_j')."""
    vals, vecs = np.linalg.eigh(S)
    o = np.argsort(vals)[::-1][:j]
    C = (vecs[:, o] * vals[o]) @ vecs[:, o].T
    psi = np.maximum(np.diag(S) - np.diag(C), 1e-6 * np.diag(S))
    return C + np.diag(psi)


def minvar_matrix(S: np.ndarray, iT, iH) -> np.ndarray:
    """A for the minimum-variance hedge under covariance S: h = -S_HH^{-1} S_HT w."""
    return np.linalg.solve(S[np.ix_(iH, iH)], S[np.ix_(iH, iT)])


def hedge_matrix(method: str, k: int, win: np.ndarray, sig_win: np.ndarray | None,
                 sig_now: np.ndarray | None, mats: np.ndarray, iT, iH) -> np.ndarray:
    """Dispatch. `win` is the trailing window of dy (days x n), all maturities."""
    if method == "interp":
        return interp_matrix(mats[iT], mats[iH])
    if method == "ns":
        return solve_factor_hedge(ns_loadings(mats, k), iT, iH)
    if method == "pca":
        L, _ = pca(np.cov(win, rowvar=False), k)
        return solve_factor_hedge(L, iT, iH)
    if method == "pca_vol":
        z = win / sig_win
        L, _ = pca(np.corrcoef(z, rowvar=False), k)
        return solve_factor_hedge(L * sig_now[:, None], iT, iH)
    if method == "ols":
        return ols_matrix(win[:, iT], win[:, iH])
    if method.startswith("fmv"):
        R = factor_cov(np.corrcoef(win / sig_win, rowvar=False), int(method[3:]))
        return minvar_matrix(R * np.outer(sig_now, sig_now), iT, iH)
    if method.startswith("fm"):
        return minvar_matrix(factor_cov(np.cov(win, rowvar=False), int(method[2:])), iT, iH)
    raise ValueError(method)


# ---------------------------------------------------------------- geometry
def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between the column spaces of A and B."""
    qa, _ = np.linalg.qr(A)
    qb, _ = np.linalg.qr(B)
    s = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1, 1)
    return np.degrees(np.arccos(s))


def explained_share(dy: np.ndarray, L: np.ndarray) -> float:
    """Share of the sum of squares of dy (days x n) inside the column space of L."""
    q, _ = np.linalg.qr(L)
    proj = dy @ q @ q.T
    return float(np.sum(proj ** 2) / np.sum(dy ** 2))


def nonparallel(dy: np.ndarray) -> np.ndarray:
    """Remove the cross-maturity mean from each day: the part of the move that is not a parallel shift."""
    return dy - dy.mean(axis=1, keepdims=True)
