"""GJR-GARCH(1,1) with Gaussian innovations, fitted to daily S&P 500 excess log returns, and its
risk-neutral version (Duan 1995 locally risk-neutral valuation relationship) for Part III.

Physical:      x_t = ln(S_t/S_{t-1}) - (r_t - q_t)/252 = lam*sqrt(h_t) - h_t/2 + eps_t,  eps_t = sqrt(h_t) z_t
               h_{t+1} = omega + (alpha + gamma*1[eps_t<0]) eps_t^2 + beta h_t
Risk-neutral:  eps*_t = eps_t + lam*sqrt(h_t) ~ N(0, h_t), so x_t = -h_t/2 + eps*_t and
               h_{t+1} = omega + (alpha + gamma*1[eps*_t - lam sqrt(h_t) < 0]) (eps*_t - lam sqrt(h_t))^2 + beta h_t
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]


@njit(cache=True)
def _nll(params, x):
    omega, alpha, gamma, beta, lam = params
    n = x.shape[0]
    h = np.var(x)
    ll = 0.0
    for t in range(n):
        if h <= 0:
            return 1e10
        e = x[t] - lam * np.sqrt(h) + 0.5 * h
        ll += -0.5 * (np.log(2 * np.pi) + np.log(h) + e * e / h)
        h = omega + (alpha + (gamma if e < 0 else 0.0)) * e * e + beta * h
    return -ll


def fit(x):
    """Gaussian QMLE. omega is optimised in units of 1e-6 so all parameters are O(1)."""
    x = np.asarray(x, float)

    def obj(p):
        th = np.array([p[0] * 1e-6, p[1], p[2], p[3], p[4]])
        if th[1] + 0.5 * th[2] + th[3] >= 0.9999:
            return 1e10
        return _nll(th, x)

    p0 = np.array([np.var(x) * 0.02 * 1e6, 0.03, 0.08, 0.90, 0.05])
    bounds = [(1e-4, 1e3), (0.0, 0.5), (0.0, 0.5), (0.5, 0.9999), (-1.0, 1.0)]
    res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds)
    res = minimize(obj, res.x, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-8, fatol=1e-6))
    th = np.array([res.x[0] * 1e-6, *res.x[1:]])
    return th, -res.fun


@njit(cache=True)
def simulate_q(S0, h0, r, q, n_paths, n_steps, params, seed):
    """Risk-neutral GJR-GARCH paths (daily). Returns S (n_paths, n_steps+1), h (n_paths, n_steps+1)."""
    omega, alpha, gamma, beta, lam = params
    np.random.seed(seed)
    S = np.empty((n_paths, n_steps + 1))
    H = np.empty((n_paths, n_steps + 1))
    drift = (r - q) / 252.0
    for i in range(n_paths):
        s = S0
        h = h0
        S[i, 0] = s
        H[i, 0] = h
        for t in range(n_steps):
            z = np.random.standard_normal()
            es = np.sqrt(h) * z
            s = s * np.exp(drift - 0.5 * h + es)
            e = es - lam * np.sqrt(h)
            h = omega + (alpha + (gamma if e < 0 else 0.0)) * e * e + beta * h
            S[i, t + 1] = s
            H[i, t + 1] = h
    return S, H


def expected_avg_var(h, steps, params):
    """Risk-neutral expected average daily variance over the next `steps` days given h (vectorised)."""
    omega, alpha, gamma, beta, lam = params
    # under Q, E[h_{t+1}|h_t] = omega + h_t * (alpha (1+lam^2) + gamma*E[(z-lam)^2 1(z<lam)] + beta)
    from scipy.stats import norm
    g = (1 + lam ** 2) * norm.cdf(lam) + lam * norm.pdf(lam)   # E[(z-lam)^2 1(z<lam)]
    pers = alpha * (1 + lam ** 2) + gamma * g + beta
    hbar = omega / (1 - pers)
    tau = np.asarray(steps, float)
    wmean = (1.0 - pers ** tau) / (tau * (1.0 - pers))      # mean of pers^k, k = 0..tau-1
    return hbar + (np.asarray(h) - hbar) * wmean, pers, hbar


def load_returns():
    d = pd.read_csv(ROOT / "data" / "sp500_option_inputs.csv", parse_dates=["date"], index_col="date")
    x = np.log(d.S).diff() - (d.r - d.q) / 252.0
    return x.dropna()
