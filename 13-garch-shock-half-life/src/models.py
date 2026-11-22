"""Volatility models with own MLE, filters and closed-form multi-horizon forecasts.

Every model is zero-mean on percent log returns. Each has:
  fit(r)                    -> parameter vector (QMLE / MLE on the given sample)
  filter(r, p, s0)          -> state arrays aligned so that row t holds the forecast for day t+1
                               made with information up to and including day t
  cum_forecast(state, p, h) -> E_t[sum_{k=1..h} r^2_{t+k}] for every origin t
  decay(p, k)               -> how much of a variance shock at t is left at day t+k (k=1 -> 1)

Models: EWMA (RiskMetrics 0.94), GARCH(1,1)-normal, GJR-GARCH(1,1)-t, and the Engle-Lee (1999)
component GARCH (CGARCH). HAR and FIGARCH live in backtest.py / irf.py.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import gammaln

LOG2PI = math.log(2 * math.pi)
BIG = 1e10


# ----------------------------------------------------------------------------- recursions
@njit(cache=True)
def _garch_var(r2, omega, alpha, beta, s0):
    """s[t] = variance of day t+1 given info to t. s[-1] (prior) = s0."""
    n = r2.shape[0]
    s = np.empty(n)
    prev = s0
    for t in range(n):
        prev = omega + alpha * r2[t] + beta * prev
        s[t] = prev
    return s


@njit(cache=True)
def _gjr_var(r, omega, alpha, gamma, beta, s0):
    n = r.shape[0]
    s = np.empty(n)
    prev = s0
    for t in range(n):
        a = alpha + (gamma if r[t] < 0 else 0.0)
        prev = omega + a * r[t] * r[t] + beta * prev
        s[t] = prev
    return s


@njit(cache=True)
def _cgarch_var(r2, omega, rho, phi, alpha, beta, s0):
    """Engle-Lee: q_{t+1} = w + rho (q_t - w) + phi (r_t^2 - s_t);
    s_{t+1} = q_{t+1} + alpha (r_t^2 - q_t) + beta (s_t - q_t). Returns (s, q) for day t+1."""
    n = r2.shape[0]
    s = np.empty(n)
    q = np.empty(n)
    sp, qp = s0, s0
    for t in range(n):
        qn = omega + rho * (qp - omega) + phi * (r2[t] - sp)
        sn = qn + alpha * (r2[t] - qp) + beta * (sp - qp)
        if sn < 1e-8:
            sn = 1e-8
        s[t] = sn
        q[t] = qn
        sp, qp = sn, qn
    return s, q


@njit(cache=True)
def simulate_garch(w, a, b, z):
    """Simulate GARCH(1,1) returns from a vector of standardized shocks z (first 500 are burn-in)."""
    n = z.shape[0]
    out = np.empty(n)
    s = w / (1 - a - b)
    for t in range(n):
        out[t] = math.sqrt(s) * z[t]
        s = w + a * out[t] * out[t] + b * s
    return out[500:]


@njit(cache=True)
def simulate_cgarch(w, rho, phi, a, b, z):
    """Simulate Engle-Lee component GARCH returns from standardized shocks z (first 500 burn-in)."""
    n = z.shape[0]
    out = np.empty(n)
    s, q = w, w
    for t in range(n):
        out[t] = math.sqrt(s) * z[t]
        r2 = out[t] * out[t]
        qn = w + rho * (q - w) + phi * (r2 - s)
        sn = qn + a * (r2 - q) + b * (s - q)
        if sn < 1e-8:
            sn = 1e-8
        s, q = sn, qn
    return out[500:]


@njit(cache=True)
def _ewma_var(r2, lam, s0):
    n = r2.shape[0]
    s = np.empty(n)
    prev = s0
    for t in range(n):
        prev = lam * prev + (1 - lam) * r2[t]
        s[t] = prev
    return s


# ----------------------------------------------------------------------------- likelihoods
def _nll_normal(r2, s_pred):
    # s_pred[t] is the variance of r[t]
    return 0.5 * np.sum(LOG2PI + np.log(s_pred) + r2 / s_pred)


def _shift(s, s0):
    """Variance of day t (made at t-1), from the filter output that is variance of day t+1."""
    return np.concatenate(([s0], s[:-1]))


def start_var(r):
    return float(np.mean(r[: min(len(r), 252)] ** 2))


class GARCH:
    name = "GARCH"
    k = 3

    @staticmethod
    def nll(theta, r, r2, s0):
        w, a, b = theta
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.99999:
            return BIG
        s = _shift(_garch_var(r2, w, a, b, s0), s0)
        return _nll_normal(r2, s)

    @classmethod
    def fit(cls, r, x0=None, quick=False):
        r2 = r * r
        s0 = start_var(r)
        v = np.mean(r2)
        starts = [x0] if x0 is not None else []
        if not (quick and x0 is not None):
            starts += [np.array([v * 0.01, 0.08, 0.91]), np.array([v * 0.05, 0.12, 0.83])]
        best = None
        for x in starts:
            o = minimize(cls.nll, x, args=(r, r2, s0), method="Nelder-Mead",
                         options=dict(xatol=1e-7, fatol=1e-6, maxiter=4000))
            o = minimize(cls.nll, o.x, args=(r, r2, s0), method="L-BFGS-B",
                         bounds=[(1e-8, None), (0, 1), (0, 1)])
            if best is None or o.fun < best.fun:
                best = o
        return best.x, -best.fun

    @staticmethod
    def persistence(p):
        return p[1] + p[2]

    @staticmethod
    def long_run(p):
        return p[0] / (1 - GARCH.persistence(p))

    @staticmethod
    def filter(r, p, s0):
        return {"s": _garch_var(r * r, *p, s0)}

    @classmethod
    def cum_forecast(cls, st, p, h):
        phi, vbar = cls.persistence(p), cls.long_run(p)
        return h * vbar + (st["s"] - vbar) * (1 - phi ** h) / (1 - phi)

    @classmethod
    def path(cls, st, p, k):
        phi, vbar = cls.persistence(p), cls.long_run(p)
        return vbar + np.outer(st["s"] - vbar, phi ** (np.asarray(k) - 1))

    @classmethod
    def decay(cls, p, k):
        return cls.persistence(p) ** (np.asarray(k, float) - 1)


class GJRt:
    """GJR-GARCH(1,1) with standardized Student-t innovations (the model of project 01)."""
    name = "GJR-t"
    k = 5

    @staticmethod
    def nll(theta, r, r2, s0):
        w, a, g, b, nu = theta
        if w <= 0 or a < 0 or b < 0 or a + g < 0 or a + g / 2 + b >= 0.99999 or nu <= 2.05 or nu > 200:
            return BIG
        s = _shift(_gjr_var(r, w, a, g, b, s0), s0)
        c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
        ll = c - 0.5 * np.log(s) - (nu + 1) / 2 * np.log1p(r2 / (s * (nu - 2)))
        return -np.sum(ll)

    @classmethod
    def fit(cls, r, x0=None):
        r2 = r * r
        s0 = start_var(r)
        v = np.mean(r2)
        starts = [x0] if x0 is not None else []
        starts += [np.array([v * 0.01, 0.02, 0.10, 0.90, 7.0]), np.array([v * 0.04, 0.05, 0.08, 0.85, 5.0])]
        best = None
        for x in starts:
            o = minimize(cls.nll, x, args=(r, r2, s0), method="Nelder-Mead",
                         options=dict(xatol=1e-7, fatol=1e-6, maxiter=6000))
            if best is None or o.fun < best.fun:
                best = o
        return best.x, -best.fun

    @staticmethod
    def persistence(p):
        return p[1] + p[2] / 2 + p[3]

    @classmethod
    def long_run(cls, p):
        return p[0] / (1 - cls.persistence(p))

    @staticmethod
    def filter(r, p, s0):
        return {"s": _gjr_var(r, p[0], p[1], p[2], p[3], s0)}

    cum_forecast = classmethod(GARCH.cum_forecast.__func__)
    path = classmethod(GARCH.path.__func__)
    decay = classmethod(GARCH.decay.__func__)


class CGARCH:
    """Engle & Lee (1999) component GARCH, normal QMLE. theta = (omega, rho, phi, alpha, beta)."""
    name = "CGARCH"
    k = 5

    @staticmethod
    def valid(theta):
        w, rho, phi, a, b = theta
        return (w > 0 and 0 < rho < 0.99999 and phi > 0 and a > 0 and b >= 0
                and a + b < rho and phi < 0.5)

    @classmethod
    def nll(cls, theta, r, r2, s0):
        if not cls.valid(theta):
            return BIG
        s, _ = _cgarch_var(r2, *theta, s0)
        return _nll_normal(r2, _shift(s, s0))

    @classmethod
    def fit(cls, r, x0=None):
        r2 = r * r
        s0 = start_var(r)
        v = np.mean(r2)
        starts = [x0] if x0 is not None else []
        starts += [np.array([v, 0.995, 0.03, 0.06, 0.80]), np.array([v, 0.98, 0.05, 0.10, 0.60]),
                   np.array([v, 0.999, 0.02, 0.08, 0.85])]
        best = None
        for x in starts:
            o = minimize(cls.nll, x, args=(r, r2, s0), method="Nelder-Mead",
                         options=dict(xatol=1e-8, fatol=1e-6, maxiter=8000))
            if best is None or o.fun < best.fun:
                best = o
        return best.x, -best.fun

    @staticmethod
    def persistence(p):
        """Long-run (trend) persistence rho; the transitory one is alpha + beta."""
        return p[1]

    @staticmethod
    def long_run(p):
        return p[0]

    @staticmethod
    def filter(r, p, s0):
        s, q = _cgarch_var(r * r, *p, s0)
        return {"s": s, "q": q}

    @staticmethod
    def cum_forecast(st, p, h):
        w, rho, _, a, b = p
        psi = a + b
        return (h * w + (st["q"] - w) * (1 - rho ** h) / (1 - rho)
                + (st["s"] - st["q"]) * (1 - psi ** h) / (1 - psi))

    @staticmethod
    def path(st, p, k):
        w, rho, _, a, b = p
        k = np.asarray(k)
        return w + np.outer(st["q"] - w, rho ** (k - 1)) + np.outer(st["s"] - st["q"], (a + b) ** (k - 1))

    @staticmethod
    def decay(p, k, share_long=None):
        """Response at t+k to a unit r^2 shock at t, normalized to 1 at k=1 (the 'news impact decay')."""
        w, rho, phi, a, b = p
        k = np.asarray(k, float)
        # a unit r^2 innovation at t raises q_{t+1} by phi and s_{t+1} - q_{t+1} by alpha
        lng, sht = phi, a
        tot = lng + sht
        return (lng * rho ** (k - 1) + sht * (a + b) ** (k - 1)) / tot


class EWMA:
    name = "EWMA"
    k = 0
    LAM = 0.94

    @staticmethod
    def fit(r, x0=None):
        return np.array([EWMA.LAM]), np.nan

    @staticmethod
    def filter(r, p, s0):
        return {"s": _ewma_var(r * r, p[0], s0)}

    @staticmethod
    def cum_forecast(st, p, h):
        return h * st["s"]

    @staticmethod
    def decay(p, k):
        return np.ones_like(np.asarray(k, float))

    @staticmethod
    def persistence(p):
        return 1.0


def half_life(phi):
    """Days until a shock decays to half under geometric decay phi."""
    phi = np.asarray(phi, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((phi > 0) & (phi < 1), np.log(0.5) / np.log(phi), np.inf)


def curve_half_life(curve, k):
    """First horizon (linearly interpolated) where a decay curve normalized to 1 at k[0] crosses 0.5."""
    c = np.asarray(curve, float)
    k = np.asarray(k, float)
    below = np.where(c <= 0.5)[0]
    if len(below) == 0:
        return np.inf
    i = below[0]
    if i == 0:
        return k[0]
    return k[i - 1] + (c[i - 1] - 0.5) / (c[i - 1] - c[i]) * (k[i] - k[i - 1])


MODELS = {m.name: m for m in (EWMA, GARCH, GJRt, CGARCH)}
