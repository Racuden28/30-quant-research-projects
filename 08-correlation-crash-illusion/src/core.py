"""Core estimators shared by the conditioning study (part A) and the forecast horse race (part B)."""
from __future__ import annotations

import numpy as np
from scipy import optimize, special

EPS = 1e-12


# ----------------------------------------------------------------------------- volatility filter
VOL_FLOOR = 0.002  # 0.2% a day: guards against stale-price stretches (rounded 1990s quotes)


def ewma_vol(r: np.ndarray, lam: float = 0.94, warm: int = 63, floor: float = VOL_FLOOR) -> np.ndarray:
    """Causal RiskMetrics volatility: sigma[t] uses returns up to t-1 only (ex-ante).

    r is T x N. The first `warm` rows are initialised with the std of those rows (marked as warm-up
    by the caller, never evaluated). A floor on sigma stops a run of unchanged (rounded) prices from
    producing near-zero volatility and absurd standardised returns.
    """
    r = np.asarray(r, float)
    T = r.shape[0]
    var = np.empty_like(r)
    v = np.nanvar(r[:warm], axis=0) + EPS
    for t in range(T):
        var[t] = np.maximum(v, floor * floor)
        x = r[t]
        v = lam * v + (1 - lam) * np.where(np.isnan(x), v, x * x)
    return np.sqrt(var)


# ----------------------------------------------------------------------------- correlation helpers
def corr(x: np.ndarray) -> np.ndarray:
    c = np.corrcoef(np.asarray(x, float), rowvar=False)
    return np.clip(c, -1, 1)


def offdiag(c: np.ndarray, cols=None) -> np.ndarray:
    if cols is not None:
        c = c[np.ix_(cols, cols)]
    iu = np.triu_indices(c.shape[0], 1)
    return c[iu]


def fisher_mean(c: np.ndarray, cols=None) -> float:
    """Average pairwise correlation, averaged on Fisher's z scale."""
    v = np.clip(offdiag(c, cols), -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(v))))


def mean_corr_subset(x: np.ndarray, mask: np.ndarray, cols=None) -> float:
    return fisher_mean(corr(x[mask]), cols)


def nearest_corr(c: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Symmetric eigenvalue floor and unit diagonal (cheap PSD repair)."""
    c = (c + c.T) / 2
    w, v = np.linalg.eigh(c)
    c = (v * np.maximum(w, floor)) @ v.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


# ----------------------------------------------------------------------------- null simulators
def mvt_unit(rng, R: np.ndarray, T: int, nu: float | None) -> np.ndarray:
    """T draws from a Gaussian (nu=None) or Student-t (unit variance) with correlation R."""
    L = np.linalg.cholesky(R)
    g = rng.standard_normal((T, R.shape[0])) @ L.T
    if nu is None:
        return g
    w = rng.chisquare(nu, size=(T, 1)) / nu
    return g / np.sqrt(w) * np.sqrt((nu - 2) / nu)


def fit_mvt_nu(z: np.ndarray, R: np.ndarray, grid=None) -> float:
    """Profile MLE of the multivariate-t degrees of freedom, correlation fixed at R, unit variance."""
    grid = np.arange(3.0, 30.01, 0.25) if grid is None else grid
    z = z / z.std(axis=0)
    N = z.shape[1]
    Ri = np.linalg.inv(R)
    q = np.einsum("ti,ij,tj->t", z, Ri, z)
    best, arg = -np.inf, None
    for nu in grid:
        s = (nu - 2) / nu  # unit-variance scaling: Sigma = R * (nu-2)/nu
        ll = (special.gammaln((nu + N) / 2) - special.gammaln(nu / 2) - N / 2 * np.log(nu * np.pi)
              - N / 2 * np.log(s) - (nu + N) / 2 * np.log1p(q / (s * nu))).sum()
        if ll > best:
            best, arg = ll, nu
    return float(arg)


# ----------------------------------------------------------------------------- forecasting estimators
def ewma_corr_path(z: np.ndarray, lam: float = 0.97, warm: int = 252) -> np.ndarray:
    """Causal EWMA correlation; out[t] uses z up to t-1. Returns T x N x N."""
    T, N = z.shape
    Q = np.cov(z[:warm], rowvar=False)
    out = np.empty((T, N, N))
    for t in range(T):
        d = np.sqrt(np.diag(Q))
        out[t] = Q / np.outer(d, d)
        Q = lam * Q + (1 - lam) * np.outer(z[t], z[t])
    return out


def rmt_clip(c: np.ndarray, T: int) -> np.ndarray:
    """Laloux et al. (1999) eigenvalue clipping: noise-band eigenvalues replaced by their mean."""
    N = c.shape[0]
    lam_plus = (1 + np.sqrt(N / T)) ** 2
    w, v = np.linalg.eigh(c)
    noise = w < lam_plus
    if noise.sum() > 0:
        w = w.copy()
        w[noise] = w[noise].mean()
    c2 = (v * w) @ v.T
    d = np.sqrt(np.diag(c2))
    return c2 / np.outer(d, d)


def lw_constant_corr(x: np.ndarray) -> np.ndarray:
    """Ledoit & Wolf (2004) shrinkage toward the constant-correlation target, returned as a correlation."""
    x = np.asarray(x, float)
    T, N = x.shape
    x = x - x.mean(0)
    S = x.T @ x / T
    s = np.sqrt(np.diag(S))
    Rm = S / np.outer(s, s)
    rbar = (Rm.sum() - N) / (N * (N - 1))
    F = rbar * np.outer(s, s)
    np.fill_diagonal(F, np.diag(S))
    y = x ** 2
    pi_mat = y.T @ y / T - S ** 2
    pi_hat = pi_mat.sum()
    theta_ii = (x ** 3).T @ x / T - np.diag(S)[:, None] * S  # theta_ii,ij
    theta_jj = theta_ii.T
    term = (np.sqrt(np.diag(S))[None, :] / np.sqrt(np.diag(S))[:, None]) * theta_ii \
        + (np.sqrt(np.diag(S))[:, None] / np.sqrt(np.diag(S))[None, :]) * theta_jj
    np.fill_diagonal(term, 0)
    rho_hat = np.trace(pi_mat) + rbar / 2 * term.sum()
    gamma_hat = ((F - S) ** 2).sum()
    kappa = (pi_hat - rho_hat) / gamma_hat
    delta = float(np.clip(kappa / T, 0, 1))
    Sig = delta * F + (1 - delta) * S
    d = np.sqrt(np.diag(Sig))
    return Sig / np.outer(d, d)


# ----------------------------------------------------------------------------- DCC(1,1)
def dcc_filter(z: np.ndarray, a: float, b: float, Qbar: np.ndarray) -> np.ndarray:
    """Engle (2002) DCC recursion. Returns Q path (T+1 x N x N); Q[t] is the ex-ante Q for day t."""
    T, N = z.shape
    Q = np.empty((T + 1, N, N))
    Q[0] = Qbar
    c = (1 - a - b) * Qbar
    for t in range(T):
        Q[t + 1] = c + a * np.outer(z[t], z[t]) + b * Q[t]
    return Q


def q_to_r(Q: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.einsum("...ii->...i", Q))
    return Q / (d[..., :, None] * d[..., None, :])


def dcc_negll(params, z, Qbar):
    a, b = params
    if a <= 0 or b <= 0 or a + b >= 0.9995:
        return 1e12
    R = q_to_r(dcc_filter(z, a, b, Qbar)[:-1])
    sign, logdet = np.linalg.slogdet(R)
    if np.any(sign <= 0):
        return 1e12
    sol = np.linalg.solve(R, z[..., None])[..., 0]
    quad = np.einsum("ti,ti->t", z, sol)
    return 0.5 * float(np.sum(logdet + quad - np.einsum("ti,ti->t", z, z)))


def dcc_fit(z: np.ndarray, start=(0.02, 0.95)):
    """Two-step DCC with correlation targeting (Qbar = sample covariance of z)."""
    Qbar = np.cov(z, rowvar=False)
    best = None
    for x0 in [start, (0.01, 0.98), (0.05, 0.90)]:
        res = optimize.minimize(dcc_negll, x0, args=(z, Qbar), method="Nelder-Mead",
                                options=dict(xatol=1e-5, fatol=1e-3, maxiter=400))
        if best is None or res.fun < best.fun:
            best = res
    return tuple(best.x), Qbar


def dcc_forecast_avg(Q_next: np.ndarray, Qbar: np.ndarray, a: float, b: float, h: int) -> np.ndarray:
    """Average correlation over the next h days (Engle & Sheppard mean-reverting approximation)."""
    p = a + b
    Rs = [q_to_r(Qbar + p ** k * (Q_next - Qbar)) for k in range(h)]
    return nearest_corr(np.mean(Rs, axis=0))


# ----------------------------------------------------------------------------- inference helpers
def stationary_bootstrap_idx(rng, T: int, mean_block: int) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap indices."""
    idx = np.empty(T, dtype=np.int64)
    p = 1.0 / mean_block
    idx[0] = rng.integers(T)
    jumps = rng.random(T) < p
    starts = rng.integers(T, size=T)
    for t in range(1, T):
        idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % T
    return idx


def newey_west_t(d: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """Diebold-Mariano style t-stat of mean(d) with Newey-West HAC variance; two-sided normal p-value."""
    from scipy import stats
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    T = len(d)
    lags = int(np.floor(4 * (T / 100) ** (2 / 9))) if lags is None else lags
    u = d - d.mean()
    s = u @ u / T
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / T
    t = d.mean() / np.sqrt(s / T)
    return float(t), float(2 * stats.norm.sf(abs(t)))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for k in range(m - 1, -1, -1):
        i = o[k]
        prev = min(prev, p[i] * m / (k + 1))
        q[i] = prev
    return q
