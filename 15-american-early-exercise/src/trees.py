"""Lattice pricers for American and European options with a continuous dividend yield.

Every method prices one contract (S, K, T, r, q, sigma, is_call, n steps) and is compiled with numba.
All lattices share one backward-induction kernel; they differ only in (u, d, p) and in the
two "smoothing" variants (BBS) that replace the last step by the Black-Scholes value.

Methods (name -> function):
    crr        Cox-Ross-Rubinstein (1979)
    jr         Jarrow-Rudd (1983) equal-probability tree
    tian       Tian (1993) third-moment-matching tree
    trig       Trigeorgis (1991) log-transformed tree
    lr         Leisen-Reimer (1996), Peizer-Pratt inversion method 2, n forced odd
    crr_avg    average of CRR(n) and CRR(n+1)  (kills the odd-even oscillation)
    bbs        Broadie-Detemple (1996) binomial Black-Scholes: CRR with a BS last step
    bbsr       BBS with two-point Richardson extrapolation  2*BBS(n) - BBS(n/2)
    lr_r       Leisen-Reimer with Richardson assuming O(1/n):  2*LR(2n) - LR(n)
    lr_r2      Leisen-Reimer with Richardson assuming O(1/n^2): (4*LR(2n) - LR(n)) / 3  (textbook)
    tri        Boyle/Kamrad-Ritchken trinomial (lambda = sqrt(3))
    crr_cv     CRR with the Hull-White (1988) control variate: Am_tree - Eu_tree + BS
    lr_cv      Leisen-Reimer with the same control variate
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

SQRT2 = math.sqrt(2.0)


# ----------------------------------------------------------------------------------------------
# Black-Scholes-Merton (European) on spot with yield q, used by BBS and the control variates.
# Prices the out-of-the-money side and uses parity, as recommended in project 04.
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def _ncdf(x):
    return 0.5 * math.erfc(-x / SQRT2)


@njit(cache=True)
def bs_price(S, K, T, r, q, sigma, is_call):
    if T <= 0.0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    sd = sigma * math.sqrt(T)
    F = S * math.exp((r - q) * T)
    D = math.exp(-r * T)
    if sd < 1e-14:
        return D * (max(F - K, 0.0) if is_call else max(K - F, 0.0))
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    call = D * (F * _ncdf(d1) - K * _ncdf(d2))
    put = D * (K * _ncdf(-d2) - F * _ncdf(-d1))
    # compute the OTM side directly, the ITM side by parity
    if F >= K:
        call = put + D * (F - K)
    else:
        put = call - D * (F - K)
    return call if is_call else put


# ----------------------------------------------------------------------------------------------
# Generic recombining binomial kernel.
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def _binomial(S, K, T, r, q, sigma, is_call, american, n, u, d, p, bbs):
    dt = T / n
    disc = math.exp(-r * dt)
    pu = disc * p
    pd = disc * (1.0 - p)
    sgn = 1.0 if is_call else -1.0
    last = n - 1 if bbs else n
    v = np.empty(last + 1)
    ratio = u / d
    s = S * d ** last
    for j in range(last + 1):
        if bbs:
            e = bs_price(s, K, dt, r, q, sigma, is_call)
            if american:
                e = max(e, sgn * (s - K))
            v[j] = e
        else:
            v[j] = max(sgn * (s - K), 0.0)
        s *= ratio
    for i in range(last - 1, -1, -1):
        s = S * d ** i
        for j in range(i + 1):
            c = pd * v[j] + pu * v[j + 1]
            if american:
                ex = sgn * (s - K)
                if ex > c:
                    c = ex
            v[j] = c
            s *= ratio
    return v[0]


@njit(cache=True)
def _crr_params(T, r, q, sigma, n):
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    return u, d, p


@njit(cache=True)
def _pp2(z, n):
    """Peizer-Pratt inversion, method 2 (Leisen-Reimer 1996)."""
    if z == 0.0:
        return 0.5
    a = z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))
    h = 0.5 + math.copysign(0.5, z) * math.sqrt(max(0.0, 1.0 - math.exp(-a * a * (n + 1.0 / 6.0))))
    return h


@njit(cache=True)
def _lr_params(S, K, T, r, q, sigma, n):
    dt = T / n
    sd = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sd
    d2 = d1 - sd
    p = _pp2(d2, n)
    ps = _pp2(d1, n)
    m = math.exp((r - q) * dt)
    u = m * ps / p
    d = (m - p * u) / (1.0 - p)
    return u, d, p


@njit(cache=True)
def crr(S, K, T, r, q, sigma, is_call, american, n):
    u, d, p = _crr_params(T, r, q, sigma, n)
    return _binomial(S, K, T, r, q, sigma, is_call, american, n, u, d, p, False)


@njit(cache=True)
def jr(S, K, T, r, q, sigma, is_call, american, n):
    dt = T / n
    nu = (r - q - 0.5 * sigma * sigma) * dt
    u = math.exp(nu + sigma * math.sqrt(dt))
    d = math.exp(nu - sigma * math.sqrt(dt))
    # risk-neutral p for this (u, d) so the tree is arbitrage-free (Jarrow-Rudd with exact drift)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    return _binomial(S, K, T, r, q, sigma, is_call, american, n, u, d, p, False)


@njit(cache=True)
def tian(S, K, T, r, q, sigma, is_call, american, n):
    dt = T / n
    v = math.exp(sigma * sigma * dt)
    m = math.exp((r - q) * dt)
    root = math.sqrt(v * v + 2.0 * v - 3.0)
    u = 0.5 * m * v * (v + 1.0 + root)
    d = 0.5 * m * v * (v + 1.0 - root)
    p = (m - d) / (u - d)
    return _binomial(S, K, T, r, q, sigma, is_call, american, n, u, d, p, False)


@njit(cache=True)
def trig(S, K, T, r, q, sigma, is_call, american, n):
    dt = T / n
    nu = r - q - 0.5 * sigma * sigma
    dx = math.sqrt(sigma * sigma * dt + nu * nu * dt * dt)
    u = math.exp(dx)
    d = 1.0 / u
    p = 0.5 + 0.5 * nu * dt / dx
    return _binomial(S, K, T, r, q, sigma, is_call, american, n, u, d, p, False)


@njit(cache=True)
def lr(S, K, T, r, q, sigma, is_call, american, n):
    if n % 2 == 0:
        n += 1
    u, d, p = _lr_params(S, K, T, r, q, sigma, n)
    return _binomial(S, K, T, r, q, sigma, is_call, american, n, u, d, p, False)


@njit(cache=True)
def crr_avg(S, K, T, r, q, sigma, is_call, american, n):
    return 0.5 * (crr(S, K, T, r, q, sigma, is_call, american, n)
                  + crr(S, K, T, r, q, sigma, is_call, american, n + 1))


@njit(cache=True)
def bbs(S, K, T, r, q, sigma, is_call, american, n):
    u, d, p = _crr_params(T, r, q, sigma, n)
    return _binomial(S, K, T, r, q, sigma, is_call, american, n, u, d, p, True)


@njit(cache=True)
def bbsr(S, K, T, r, q, sigma, is_call, american, n):
    m = n // 2
    return 2.0 * bbs(S, K, T, r, q, sigma, is_call, american, 2 * m) - bbs(S, K, T, r, q, sigma, is_call, american, m)


@njit(cache=True)
def _lr_pair(S, K, T, r, q, sigma, is_call, american, n, power):
    m = n // 2
    if m % 2 == 0:
        m += 1
    m2 = 2 * m + 1  # LR needs odd n; 2m+1 ~ 2m
    a = lr(S, K, T, r, q, sigma, is_call, american, m)
    b = lr(S, K, T, r, q, sigma, is_call, american, m2)
    w = (m2 / m) ** power  # Richardson with the exact step ratio
    return (w * b - a) / (w - 1.0)


@njit(cache=True)
def lr_r(S, K, T, r, q, sigma, is_call, american, n):
    """LR + Richardson assuming the American O(1/n) error (what the data show, section 2)."""
    return _lr_pair(S, K, T, r, q, sigma, is_call, american, n, 1.0)


@njit(cache=True)
def lr_r2(S, K, T, r, q, sigma, is_call, american, n):
    """LR + Richardson assuming the European O(1/n^2) error (the textbook choice)."""
    return _lr_pair(S, K, T, r, q, sigma, is_call, american, n, 2.0)


@njit(cache=True)
def tri(S, K, T, r, q, sigma, is_call, american, n):
    dt = T / n
    dx = sigma * math.sqrt(3.0 * dt)
    nu = r - q - 0.5 * sigma * sigma
    a = (sigma * sigma * dt + nu * nu * dt * dt) / (dx * dx)
    b = nu * dt / dx
    pu = 0.5 * (a + b)
    pd = 0.5 * (a - b)
    pm = 1.0 - a
    disc = math.exp(-r * dt)
    pu *= disc
    pd *= disc
    pm *= disc
    sgn = 1.0 if is_call else -1.0
    m = 2 * n + 1
    v = np.empty(m)
    eu = math.exp(dx)
    s = S * math.exp(-n * dx)
    for j in range(m):
        v[j] = max(sgn * (s - K), 0.0)
        s *= eu
    for i in range(n - 1, -1, -1):
        s = S * math.exp(-i * dx)
        for j in range(2 * i + 1):
            c = pd * v[j] + pm * v[j + 1] + pu * v[j + 2]
            if american:
                ex = sgn * (s - K)
                if ex > c:
                    c = ex
            v[j] = c
            s *= eu
    return v[0]


@njit(cache=True)
def crr_cv(S, K, T, r, q, sigma, is_call, american, n):
    a = crr(S, K, T, r, q, sigma, is_call, True, n)
    e = crr(S, K, T, r, q, sigma, is_call, False, n)
    return a - e + bs_price(S, K, T, r, q, sigma, is_call)


@njit(cache=True)
def lr_cv(S, K, T, r, q, sigma, is_call, american, n):
    a = lr(S, K, T, r, q, sigma, is_call, True, n)
    e = lr(S, K, T, r, q, sigma, is_call, False, n)
    return a - e + bs_price(S, K, T, r, q, sigma, is_call)


METHODS = {
    "crr": crr, "jr": jr, "tian": tian, "trig": trig, "lr": lr, "crr_avg": crr_avg,
    "bbs": bbs, "bbsr": bbsr, "lr_r": lr_r, "lr_r2": lr_r2, "tri": tri, "crr_cv": crr_cv, "lr_cv": lr_cv,
}
LABELS = {
    "crr": "CRR", "jr": "Jarrow-Rudd", "tian": "Tian", "trig": "Trigeorgis", "lr": "Leisen-Reimer",
    "crr_avg": "CRR avg(n, n+1)", "bbs": "BBS", "bbsr": "BBS + Richardson", "lr_r": "LR + Richardson (1/n)", "lr_r2": "LR + Richardson (1/n², textbook)",
    "tri": "Trinomial", "crr_cv": "CRR + control variate", "lr_cv": "LR + control variate",
}


# ----------------------------------------------------------------------------------------------
# Exercise boundary and policy evaluation (used in Part III).
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def crr_boundary(S, K, T, r, q, sigma, is_call, n):
    """Critical price b(t_i), i = 0..n-1, of an American option on a CRR lattice.

    For a put, b is the midpoint (in log space) between the highest node where exercise is optimal and
    the node above it; NaN if no node exercises at that step. For a call, the mirror image.
    Returns (times, boundary, price).
    """
    u, d, p = _crr_params(T, r, q, sigma, n)
    dt = T / n
    disc = math.exp(-r * dt)
    sgn = 1.0 if is_call else -1.0
    v = np.empty(n + 1)
    ratio = u / d
    s = S * d ** n
    for j in range(n + 1):
        v[j] = max(sgn * (s - K), 0.0)
        s *= ratio
    b = np.full(n, np.nan)
    for i in range(n - 1, -1, -1):
        s = S * d ** i
        hi = -1
        lo = -1
        for j in range(i + 1):
            c = disc * (p * v[j + 1] + (1.0 - p) * v[j])
            ex = sgn * (s - K)
            if ex > c and ex > 0.0:
                c = ex
                if hi < 0 or j > hi:
                    hi = j
                if lo < 0:
                    lo = j
            v[j] = c
            s *= ratio
        if is_call:
            if lo >= 0:
                b[i] = S * d ** i * ratio ** lo / math.sqrt(ratio)
        else:
            if hi >= 0:
                b[i] = S * d ** i * ratio ** hi * math.sqrt(ratio)
    times = np.arange(n) * dt
    return times, b, v[0]


@njit(cache=True)
def crr_policy_value(S, K, T, r, q, sigma, is_call, n, bnd):
    """Value, on the TRUE CRR lattice (sigma, r, q), of the policy 'exercise when S crosses bnd[i]'.

    bnd has length n (one critical price per step, NaN = never exercise at that step). The policy
    is evaluated exactly by backward induction, so there is no Monte Carlo noise.
    """
    u, d, p = _crr_params(T, r, q, sigma, n)
    disc = math.exp(-r * T / n)
    sgn = 1.0 if is_call else -1.0
    v = np.empty(n + 1)
    ratio = u / d
    s = S * d ** n
    for j in range(n + 1):
        v[j] = max(sgn * (s - K), 0.0)
        s *= ratio
    for i in range(n - 1, -1, -1):
        s = S * d ** i
        bi = bnd[i]
        for j in range(i + 1):
            c = disc * (p * v[j + 1] + (1.0 - p) * v[j])
            if bi == bi:  # not NaN
                ex = sgn * (s - K)
                if ex > 0.0 and ((is_call and s >= bi) or ((not is_call) and s <= bi)):
                    c = ex
            v[j] = c
            s *= ratio
    return v[0]


@njit(cache=True)
def crr_delta(S, K, T, r, q, sigma, is_call, american, n):
    """Delta and gamma from the first two steps of a CRR tree (extended by two steps back)."""
    m = n + 2
    Tm = T * m / n
    u, d, p = _crr_params(Tm, r, q, sigma, m)
    dt = Tm / m
    disc = math.exp(-r * dt)
    S0 = S / (u * d)  # centre node at step 2 equals S
    sgn = 1.0 if is_call else -1.0
    v = np.empty(m + 1)
    ratio = u / d
    s = S0 * d ** m
    for j in range(m + 1):
        v[j] = max(sgn * (s - K), 0.0)
        s *= ratio
    for i in range(m - 1, 1, -1):
        s = S0 * d ** i
        for j in range(i + 1):
            c = disc * (p * v[j + 1] + (1.0 - p) * v[j])
            if american:
                c = max(c, sgn * (s - K))
            v[j] = c
            s *= ratio
    s_lo, s_mid, s_hi = S0 * d * d, S, S0 * u * u
    dl = (v[1] - v[0]) / (s_mid - s_lo)
    dh = (v[2] - v[1]) / (s_hi - s_mid)
    delta = (v[2] - v[0]) / (s_hi - s_lo)
    gamma = (dh - dl) / (0.5 * (s_hi - s_lo))
    return delta, gamma, v[1]
