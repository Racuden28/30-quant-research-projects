"""Heston (1993) model: characteristic function, cumulants, VIX-family indices, VVIX, SKEW, option prices.

All quantities are risk-neutral and written on the forward, X = ln(S_T / F_0), so rates and dividends
never enter (the CBOE indices are defined on the forward too).

    dS/S = sqrt(v) dW_1,    dv = kappa (theta - v) dt + sigma sqrt(v) dW_2,    d<W_1, W_2> = rho dt

Closed forms used here (all exact under Heston, no approximation):

* VIX_tau^2 = theta + (v0 - theta) (1 - e^{-kappa tau}) / (kappa tau)          (expected average variance)
* SKEW = 100 - 10 * Skew[X_30d],   skewness from the cumulants c2, c3 of X, obtained by a Cauchy
  contour integral of the cumulant generating function log E[e^{zX}] (spectrally accurate).
* VVIX: under Heston VIX_T^2 = a + b v_T is affine in v_T. The VIX futures price F_t = E_t[VIX_T] is a
  continuous martingale, so the model-free variance that CBOE computes from VIX options equals
  E[<ln F>_T] = -2 E[ln(VIX_T / F_0)]. Hence
      VVIX^2 * T = 2 ( ln E[sqrt(Y)] - 0.5 E[ln Y] ),   Y = a + b v_T,  T = 30 days,
  and both expectations are one-dimensional integrals of the CIR Laplace transform.
* Option prices by the COS method (Fang & Oosterlee 2008) with cumulant-based truncation.
"""
from __future__ import annotations

import numpy as np

DAY = 1.0 / 365.0
TENORS = {"VIX9D": 9 * DAY, "VIX": 30 * DAY, "VIX3M": 93 * DAY, "VIX6M": 183 * DAY}
T30 = 30 * DAY


# --------------------------------------------------------------------------------------------- term
def term_b(kappa, tau):
    """b(tau) = (1 - e^{-kappa tau}) / (kappa tau): weight of v0 in the expected average variance."""
    kt = np.asarray(kappa, float) * tau
    return np.where(kt < 1e-8, 1.0 - kt / 2, -np.expm1(-kt) / np.where(kt < 1e-8, 1.0, kt))


def vix_model(v0, kappa, theta, tau):
    """Model VIX-type index (annualised vol, decimal) for horizon tau."""
    b = term_b(kappa, tau)
    return np.sqrt(np.maximum(theta + (v0 - theta) * b, 0.0))


# --------------------------------------------------------------------------------------------- CF
def log_cf(u, T, v0, kappa, theta, sigma, rho):
    """log E[exp(i u X)], X = ln(S_T/F_0). 'Little trap' form (Albrecher et al. 2007), u may be complex."""
    u = np.asarray(u, complex)
    iu = 1j * u
    beta = kappa - rho * sigma * iu
    d = np.sqrt(beta * beta + sigma * sigma * (iu + u * u))
    g = (beta - d) / (beta + d)
    edT = np.exp(-d * T)
    C = kappa * theta / sigma**2 * ((beta - d) * T - 2.0 * np.log((1.0 - g * edT) / (1.0 - g)))
    D = (beta - d) / sigma**2 * (1.0 - edT) / (1.0 - g * edT)
    return C + D * v0


def cumulants(T, v0, kappa, theta, sigma, rho, n_max=4, r=0.5, npts=64):
    """Wrapper: shrink the contour radius if the moment-generating function explodes inside it."""
    for rr in (r, r / 4, r / 16):
        c = _cumulants(T, v0, kappa, theta, sigma, rho, n_max, rr, npts)
        if np.all(np.isfinite(c)) and c[1] > 0:
            return c
    return c


def _cumulants(T, v0, kappa, theta, sigma, rho, n_max, r, npts):
    """Cumulants c1..c_n_max of X by the Cauchy integral c_n = n! / (2 pi i) oint K(z) z^{-n-1} dz,
    with K(z) = log E[e^{zX}] = log_cf(-i z). Exact to ~1e-12 for the radii used (tested)."""
    th = 2 * np.pi * np.arange(npts) / npts
    z = r * np.exp(1j * th)
    K = log_cf(-1j * z, T, v0, kappa, theta, sigma, rho)
    out = []
    fact = 1.0
    for n in range(1, n_max + 1):
        fact *= n
        out.append(float(np.real(np.mean(K * np.exp(-1j * n * th))) * fact / r**n))
    return np.array(out)


def skewness(T, v0, kappa, theta, sigma, rho):
    c = cumulants(T, v0, kappa, theta, sigma, rho, n_max=3)
    return c[2] / c[1] ** 1.5


def skew_index(v0, kappa, theta, sigma, rho, T=T30):
    """Model counterpart of the CBOE SKEW index."""
    return 100.0 - 10.0 * skewness(T, v0, kappa, theta, sigma, rho)


# --------------------------------------------------------------------------------------------- VVIX
# Gauss-Laguerre-type quadrature for the Laplace integrals, on a log grid in s.
_S_LO, _S_HI = 1e-8, 1e8
_S = np.exp(np.linspace(np.log(_S_LO), np.log(_S_HI), 2400))
_W = np.gradient(np.log(_S)) * _S  # ds on the log grid


def _cir_laplace(lam, h, v0, kappa, theta, sigma):
    """E[exp(-lam v_h) | v_0] for the CIR process (lam >= 0, vectorised over lam)."""
    e = np.exp(-kappa * h)
    c = sigma**2 * (1 - e) / (2 * kappa)
    den = 1.0 + lam * c
    return den ** (-2 * kappa * theta / sigma**2) * np.exp(-lam * v0 * e / den)


def _expectations_sqrt_log(a, b, h, v0, kappa, theta, sigma):
    """E[sqrt(Y)] and E[ln Y] for Y = a + b v_h, a > 0, via
       sqrt(y) = (1/(2 sqrt(pi))) int_0^inf (1 - e^{-s y}) s^{-3/2} ds,
       ln y    = int_0^inf (e^{-s} - e^{-s y}) / s ds."""
    L = np.exp(-_S * a) * _cir_laplace(_S * b, h, v0, kappa, theta, sigma)  # E[e^{-sY}]
    EY = a + b * (theta + (v0 - theta) * np.exp(-kappa * h))
    # analytic end corrections: (1-L) ~ s E[Y] near 0, and 1 - L ~ 1 beyond _S_HI
    e_sqrt = (np.sum((1.0 - L) * _S ** (-1.5) * _W) + 2 * EY * np.sqrt(_S_LO) + 2 / np.sqrt(_S_HI)) / (2 * np.sqrt(np.pi))
    e_log = np.sum((np.exp(-_S) - L) / _S * _W) + (EY - 1.0) * _S_LO
    return e_sqrt, e_log


def vvix_model(v0, kappa, theta, sigma, T=T30, tau=T30):
    """Model VVIX (annualised vol of VIX in decimal) over horizon T for the tau-day VIX."""
    b = term_b(kappa, tau)
    a = theta * (1 - b)
    a = max(float(a), 1e-12)
    e_sqrt, e_log = _expectations_sqrt_log(a, b, T, v0, kappa, theta, sigma)
    qv = 2.0 * (np.log(e_sqrt) - 0.5 * e_log)
    return np.sqrt(max(qv, 0.0) / T)


def vix_futures(v0, kappa, theta, sigma, T, tau=T30):
    b = term_b(kappa, tau)
    a = max(float(theta * (1 - b)), 1e-12)
    return _expectations_sqrt_log(a, b, T, v0, kappa, theta, sigma)[0]


# --------------------------------------------------------------------------------------------- COS
def cos_put_over_F(k, T, params, N=512, L=12.0):
    """Undiscounted put price divided by F for log-strikes k = ln(K/F) (array), maturity T. COS method,
    vectorised over strikes. Truncation [c1 - L w, c1 + L w], w = sqrt(c2 + sqrt|c4|)."""
    v0, kappa, theta, sigma, rho = params
    c = cumulants(T, v0, kappa, theta, sigma, rho, n_max=4)
    c1, c2, c4 = c[0], c[1], abs(c[3])
    w = L * np.sqrt(c2 + np.sqrt(c4))
    a, b = c1 - w, c1 + w
    k = np.atleast_1d(np.asarray(k, float))
    n = np.arange(N)
    u = n * np.pi / (b - a)
    phi = np.exp(log_cf(u, T, v0, kappa, theta, sigma, rho))
    A = np.real(phi * np.exp(-1j * u * a))
    A[0] *= 0.5
    d = np.clip(k, a, b)[None, :]                       # (1, K)
    U = u[:, None]                                      # (N, 1)
    ud = U * (d - a)
    chi = (np.cos(ud) * np.exp(d) - np.exp(a) + U * np.sin(ud) * np.exp(d)) / (1 + U * U)
    psi = np.where(U == 0, d - a, np.sin(ud) / np.where(U == 0, 1.0, U))
    V = 2.0 / (b - a) * (np.exp(k)[None, :] * psi - chi)
    out = A @ V
    return np.maximum(out, 0.0)


def otm_price_over_F(k, T, params, N=512):
    """OTM option price / F (put for k < 0, call for k >= 0), undiscounted."""
    k = np.atleast_1d(np.asarray(k, float))
    p = cos_put_over_F(k, T, params, N=N)
    call = p + 1.0 - np.exp(k)
    return np.where(k < 0, p, call)


# --------------------------------------------------------------------------------------------- simulation
def simulate(params, T, n_steps, n_paths, seed=0, antithetic=True):
    """Andersen (2008) QE scheme for v with the martingale-correct log-price step. Returns (X_T, v_T, v path)."""
    v0, kappa, theta, sigma, rho = params
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    m_ = n_paths // 2 if antithetic else n_paths
    v = np.full(n_paths, v0)
    x = np.zeros(n_paths)
    e = np.exp(-kappa * dt)
    k0 = -rho * kappa * theta / sigma * dt
    g1 = g2 = 0.5
    k1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    k2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    k3 = g1 * dt * (1 - rho**2)
    k4 = g2 * dt * (1 - rho**2)
    for _ in range(n_steps):
        z = rng.standard_normal(m_)
        uu = rng.random(m_)
        zx = rng.standard_normal(m_)
        if antithetic:
            z = np.concatenate([z, -z]); uu = np.concatenate([uu, 1 - uu]); zx = np.concatenate([zx, -zx])
        m = theta + (v - theta) * e
        s2 = v * sigma**2 * e / kappa * (1 - e) + theta * sigma**2 / (2 * kappa) * (1 - e) ** 2
        psi = s2 / m**2
        vn = np.empty_like(v)
        lo = psi <= 1.5
        b2 = 2 / psi[lo] - 1 + np.sqrt(2 / psi[lo]) * np.sqrt(np.maximum(2 / psi[lo] - 1, 0))
        aa = m[lo] / (1 + b2)
        vn[lo] = aa * (np.sqrt(b2) + z[lo]) ** 2
        hi = ~lo
        p = (psi[hi] - 1) / (psi[hi] + 1)
        beta = (1 - p) / m[hi]
        vn[hi] = np.where(uu[hi] <= p, 0.0, np.log(np.maximum(1 - p, 1e-300) / np.maximum(1 - uu[hi], 1e-300)) / beta)
        # martingale correction for x (Andersen sec. 3.2.2 / simple drift fix)
        x = x + k0 + k1 * v + k2 * vn + np.sqrt(np.maximum(k3 * v + k4 * vn, 0)) * zx
        v = vn
    # exact martingale: recentre so E[e^X] = 1 (removes the small discretisation drift)
    x = x - np.log(np.mean(np.exp(x)))
    return x, v
