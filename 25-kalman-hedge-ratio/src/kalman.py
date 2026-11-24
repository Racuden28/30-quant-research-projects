"""
Kalman filters for a time-varying hedge ratio, and the exponentially weighted regressions they are compared with.

Notation (scalar hedging model)
    y_t = beta_t x_t + eps_t,   eps_t ~ N(0, R)
    beta_t = beta_{t-1} + eta_t, eta_t ~ N(0, q)
Only the signal-to-noise ratio r = q / R moves the hedge ratio: scaling q and R together leaves every beta
unchanged. All filters therefore run "in units of R" (P is stored as P / R) and R is concentrated out of the
likelihood: for a given r, R_hat = mean(v_t^2 / f_t) and the profile log-likelihood is
    -0.5 * (n log R_hat + sum log f_t + n),
where v_t is the one-step prediction error and f_t = x_t^2 p_t^- + 1 its variance in units of R.

Every function returns *predicted* quantities: the beta used on day t is computed from days < t only.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# 0 (no drift in beta: recursive OLS through the origin) plus a log grid from 1e-5 to 1e3. With daily market
# variance s2 ~ 1e-4 this spans steady-state half-lives from about 70 years down to about 2 days.
R_GRID = np.r_[0.0, np.logspace(-5, 3, 40)]
P0 = 1e4          # diffuse prior variance of beta in units of R (R ~ 1e-4 a day, so P0 ~ 1, i.e. sd 1)
B0 = 1.0          # prior mean of a market beta
BURN = 20         # first observations left out of the likelihood (diffuse start)


# ----------------------------------------------------------------------------------------------------------
# Scalar filter, many signal-to-noise ratios at once
# ----------------------------------------------------------------------------------------------------------
@njit(cache=True)
def kf_scalar_grid(y, x, rgrid, b0=B0, p0=P0, burn=BURN):
    """Run one scalar random-walk-beta filter per value of r in rgrid.

    y, x: (T,) with NaN in y for missing days (the filter then only adds process noise).
    Returns
      B   (T, G) predicted beta for day t (uses days < t),
      LL  (T, G) profile log-likelihood of days < t (cumulative), NaN before enough data,
      PM  (T, G) predicted variance p_t^- in units of R.
    """
    T, G = len(y), len(rgrid)
    B = np.empty((T, G))
    LL = np.full((T, G), np.nan)
    PM = np.empty((T, G))
    b = np.full(G, b0)
    p = np.full(G, p0)
    slogf = np.zeros(G)
    sv2f = np.zeros(G)
    n = 0
    seen = 0
    for t in range(T):
        for g in range(G):
            pm = p[g] + rgrid[g]
            B[t, g] = b[g]
            PM[t, g] = pm
            if n > 10:
                Rh = sv2f[g] / n
                LL[t, g] = -0.5 * (n * np.log(Rh) + slogf[g] + n)
            p[g] = pm
        yt, xt = y[t], x[t]
        if np.isnan(yt) or np.isnan(xt):
            continue
        seen += 1
        for g in range(G):
            pm = p[g]
            f = xt * xt * pm + 1.0
            v = yt - b[g] * xt
            k = pm * xt / f
            b[g] += k * v
            p[g] = pm / f
            if seen > burn:
                slogf[g] += np.log(f)
                sv2f[g] += v * v / f
        if seen > burn:
            n += 1
    return B, LL, PM


@njit(cache=True)
def kf_scalar_hr_grid(y, x, rgrid, hl_r=21.0, b0=B0, p0=P0, burn=BURN):
    """Heteroskedastic variant: R_t is an EWMA (half-life hl_r days) of past squared prediction errors and
    q_t = r * Rbar_t, Rbar_t the expanding mean of past squared errors. Not scale-free, so the plain
    Gaussian log-likelihood is accumulated. Returns B, LL as kf_scalar_grid."""
    T, G = len(y), len(rgrid)
    lam = 0.5 ** (1.0 / hl_r)
    B = np.empty((T, G))
    LL = np.full((T, G), np.nan)
    b = np.full(G, b0)
    P = np.full(G, -1.0)          # set once R is initialised
    Rt = np.zeros(G)
    sv2 = np.zeros(G)
    ll = np.zeros(G)
    nv = np.zeros(G)
    n = 0
    seen = 0
    init_y2 = 0.0
    for t in range(T):
        for g in range(G):
            B[t, g] = b[g]
            if n > 10:
                LL[t, g] = ll[g]
        yt, xt = y[t], x[t]
        if np.isnan(yt) or np.isnan(xt):
            for g in range(G):
                if P[g] > 0:
                    P[g] += rgrid[g] * sv2[g] / max(nv[g], 1.0)
            continue
        seen += 1
        if seen <= burn:                     # warm-up: collect a scale for R, plain recursive LS
            init_y2 += yt * yt
            if seen == burn:
                for g in range(G):
                    Rt[g] = 0.5 * init_y2 / burn
                    sv2[g] = Rt[g] * burn
                    nv[g] = burn
                    P[g] = p0 * Rt[g]
            continue
        for g in range(G):
            Rbar = sv2[g] / nv[g]
            pm = P[g] + rgrid[g] * Rbar
            F = xt * xt * pm + Rt[g]
            v = yt - b[g] * xt
            k = pm * xt / F
            b[g] += k * v
            P[g] = pm - k * xt * pm
            ll[g] += -0.5 * (np.log(2 * np.pi * F) + v * v / F)
            Rt[g] = lam * Rt[g] + (1 - lam) * v * v
            sv2[g] += v * v
            nv[g] += 1
        n += 1
    return B, LL


def steady_state_pm(r, s2):
    """Steady-state predicted variance p^- (units of R) for a constant regressor second moment s2."""
    r = np.asarray(r, float)
    return (r * s2 + np.sqrt((r * s2) ** 2 + 4.0 * s2 * r)) / (2.0 * s2)


def matched_lambda(r, s2):
    """Decay factor of the EWMA regression whose per-day weight on the newest observation equals the
    Kalman filter's steady-state gain share s2 p^- / (1 + s2 p^-)."""
    a = s2 * steady_state_pm(r, s2)
    return 1.0 / (1.0 + a)


def matched_halflife(r, s2):
    lam = matched_lambda(r, s2)
    with np.errstate(divide="ignore"):
        return np.where(lam < 1, np.log(0.5) / np.log(lam), np.inf)


def r_from_halflife(h, s2):
    """Inverse of matched_halflife: the r whose steady state forgets like an EWMA of half-life h days."""
    a = 0.5 ** (-1.0 / np.asarray(h, float)) - 1.0      # a = s2 p^-
    pm = a / s2
    # s2 pm^2 = r (1 + s2 pm)
    return s2 * pm * pm / (1.0 + s2 * pm)


# ----------------------------------------------------------------------------------------------------------
# Exponentially weighted regressions through the origin, with a time-varying clock
# ----------------------------------------------------------------------------------------------------------
@njit(cache=True)
def ew_beta(y, x, decay, b0=B0, prior_w=0.0):
    """beta_t = sum_s w_ts x_s y_s / sum_s w_ts x_s^2 over s < t, with weights that shrink by decay[s] at
    each new observation s (decay may vary by day: calendar, variance or volatility clock).
    prior_w adds prior_w * (x^2 weight) of a beta of b0 (0 = none). Returns predicted betas (T,)."""
    T = len(y)
    out = np.empty(T)
    sxy = prior_w * b0
    sxx = prior_w
    for t in range(T):
        out[t] = sxy / sxx if sxx > 0 else b0
        yt, xt = y[t], x[t]
        if np.isnan(yt) or np.isnan(xt):
            continue
        d = decay[t]
        sxy = d * sxy + xt * yt
        sxx = d * sxx + xt * xt
    return out


@njit(cache=True)
def rolling_ols(y, x, w, minobs):
    """Predicted beta from an OLS with intercept on the last w valid observations before t."""
    T = len(y)
    out = np.full(T, np.nan)
    bufx = np.empty(w)
    bufy = np.empty(w)
    k = 0
    cnt = 0
    sx = sy = sxx = sxy = 0.0
    for t in range(T):
        if cnt >= minobs:
            m = min(cnt, w)
            cxx = sxx - sx * sx / m
            if cxx > 0:
                out[t] = (sxy - sx * sy / m) / cxx
        yt, xt = y[t], x[t]
        if np.isnan(yt) or np.isnan(xt):
            continue
        if cnt >= w:
            ox, oy = bufx[k], bufy[k]
            sx -= ox
            sy -= oy
            sxx -= ox * ox
            sxy -= ox * oy
        bufx[k] = xt
        bufy[k] = yt
        k = (k + 1) % w
        cnt += 1
        sx += xt
        sy += yt
        sxx += xt * xt
        sxy += xt * yt
    return out


# ----------------------------------------------------------------------------------------------------------
# Two-state filter (intercept and slope) on log prices, for pairs
# ----------------------------------------------------------------------------------------------------------
@njit(cache=True)
def kf2_loglik_grid(y, x, ra, rb, p0=P0, burn=BURN):
    """Profile log-likelihood of y_t = a_t + b_t x_t + e_t with independent random-walk a and b
    (variances ra R and rb R) for every (ra[i], rb[j]). Diffuse start at a = b = 0. Returns (len(ra), len(rb))."""
    T = len(y)
    out = np.empty((len(ra), len(rb)))
    for i in range(len(ra)):
        for j in range(len(rb)):
            a = 0.0
            b = 0.0
            p11 = p0
            p22 = p0
            p12 = 0.0
            slogf = 0.0
            sv2f = 0.0
            n = 0
            for t in range(T):
                p11 += ra[i]
                p22 += rb[j]
                xt = x[t]
                f = p11 + 2 * xt * p12 + xt * xt * p22 + 1.0
                v = y[t] - a - b * xt
                k1 = (p11 + xt * p12) / f
                k2 = (p12 + xt * p22) / f
                a += k1 * v
                b += k2 * v
                q11 = p11 - k1 * (p11 + xt * p12)
                q12 = p12 - k1 * (p12 + xt * p22)
                q22 = p22 - k2 * (p12 + xt * p22)
                p11, p12, p22 = q11, q12, q22
                if t >= burn:
                    slogf += np.log(f)
                    sv2f += v * v / f
                    n += 1
            out[i, j] = -0.5 * (n * np.log(sv2f / n) + slogf + n)
    return out


@njit(cache=True)
def kf2_run(y, x, ra, rb, p0=P0):
    """Run the two-state filter. Returns predicted a, b (T,), prediction error v (T,), its variance f (T,) in
    units of R, and the filtered (end-of-day) a, b (T,)."""
    T = len(y)
    A = np.empty(T)
    Bp = np.empty(T)
    V = np.empty(T)
    Fv = np.empty(T)
    Af = np.empty(T)
    Bf = np.empty(T)
    a = 0.0
    b = 0.0
    p11 = p0
    p22 = p0
    p12 = 0.0
    for t in range(T):
        p11 += ra
        p22 += rb
        xt = x[t]
        A[t] = a
        Bp[t] = b
        f = p11 + 2 * xt * p12 + xt * xt * p22 + 1.0
        v = y[t] - a - b * xt
        V[t] = v
        Fv[t] = f
        k1 = (p11 + xt * p12) / f
        k2 = (p12 + xt * p22) / f
        a += k1 * v
        b += k2 * v
        q11 = p11 - k1 * (p11 + xt * p12)
        q12 = p12 - k1 * (p12 + xt * p22)
        q22 = p22 - k2 * (p12 + xt * p22)
        p11, p12, p22 = q11, q12, q22
        Af[t] = a
        Bf[t] = b
    return A, Bp, V, Fv, Af, Bf


@njit(cache=True)
def kf2_run_abs(y, x, qa, qb, Rv, a0, b0, p0):
    """Two-state filter with absolute (not concentrated) noise variances, as in Chan (2013): Q = diag(qa, qb),
    R = Rv, start (a0, b0) with variance p0 I. Returns the same arrays as kf2_run, f in absolute units."""
    T = len(y)
    A = np.empty(T)
    Bp = np.empty(T)
    V = np.empty(T)
    Fv = np.empty(T)
    Af = np.empty(T)
    Bf = np.empty(T)
    a, b = a0, b0
    p11 = p0
    p22 = p0
    p12 = 0.0
    for t in range(T):
        p11 += qa
        p22 += qb
        xt = x[t]
        A[t] = a
        Bp[t] = b
        f = p11 + 2 * xt * p12 + xt * xt * p22 + Rv
        v = y[t] - a - b * xt
        V[t] = v
        Fv[t] = f
        k1 = (p11 + xt * p12) / f
        k2 = (p12 + xt * p22) / f
        a += k1 * v
        b += k2 * v
        q11 = p11 - k1 * (p11 + xt * p12)
        q12 = p12 - k1 * (p12 + xt * p22)
        q22 = p22 - k2 * (p12 + xt * p22)
        p11, p12, p22 = q11, q12, q22
        Af[t] = a
        Bf[t] = b
    return A, Bp, V, Fv, Af, Bf
