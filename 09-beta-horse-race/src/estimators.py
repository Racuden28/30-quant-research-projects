"""
Nine ex-ante market-beta estimators, evaluated at month-ends.

Every function takes
    Y   (T x N) daily excess returns of the test assets, NaN where missing
    x   (T,)    daily excess return of the market (never missing)
    E   (K,)    integer positions of the month-end days at which a forecast is made
and returns a (K x N) array of betas that use only days 0..E[k] (inclusive). No function ever
reads a day after E[k]; tests/test_estimators.py checks this by perturbing the future.

Parameters are fixed in advance (see README, "Pre-registered design"); none was tuned on the
evaluation sample.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- pre-registered parameters
W_DAILY = 252          # one year of daily data (OLS-1Y, Vasicek, Blume, Dimson)
MIN_DAILY = 200
W_MONTHLY = 60         # five years of monthly data (the textbook beta)
MIN_MONTHLY = 36
EWMA_HALFLIFE = 126    # six-month half-life
FP_CORR_WINDOW, FP_CORR_MIN = 1260, 750   # Frazzini-Pedersen: 5y of 3-day returns for correlation
FP_SHRINK = 0.6        # beta = 0.6 * beta_TS + 0.4 * 1
WELCH_WINDOW, WELCH_MIN = 756, 252
WELCH_RHO = 2 / 252    # daily age-decay of Welch (2022)
WELCH_BOUNDS = (-2.0, 4.0)
BLUME_A, BLUME_B = 0.33, 0.67
DIMSON_LAGS = 2

ESTIMATORS = ["OLS-5Y-M", "OLS-1Y-D", "Blume", "Vasicek", "EWMA", "Dimson", "Frazzini-Pedersen",
              "Welch-SWA", "Combination"]
BASELINES = ["One"]           # beta = 1 for everything (the "no information" CAPM prior)
TEXTBOOK = "OLS-5Y-M"


def _csum(a: np.ndarray) -> np.ndarray:
    """Cumulative sum along axis 0 with a leading zero row, NaN treated as 0."""
    c = np.cumsum(np.nan_to_num(a), axis=0)
    return np.concatenate([np.zeros((1,) + a.shape[1:]), c], axis=0)


def _win(c: np.ndarray, E: np.ndarray, w: int) -> np.ndarray:
    """Sum over the window (E-w, E] from a leading-zero cumulative sum."""
    hi = E + 1
    lo = np.maximum(hi - w, 0)
    return c[hi] - c[lo]


def ols_daily(Y, x, E, window=W_DAILY, min_obs=MIN_DAILY, return_se=False):
    m = ~np.isnan(Y)
    X = np.where(m, x[:, None], 0.0)
    Yz = np.where(m, Y, 0.0)
    n = _win(_csum(m.astype(float)), E, window)
    sx, sy = _win(_csum(X), E, window), _win(_csum(Yz), E, window)
    sxx, sxy, syy = (_win(_csum(X * X), E, window), _win(_csum(X * Yz), E, window),
                     _win(_csum(Yz * Yz), E, window))
    with np.errstate(invalid="ignore", divide="ignore"):
        cxx = sxx - sx * sx / n
        cxy = sxy - sx * sy / n
        cyy = syy - sy * sy / n
        beta = cxy / cxx
        beta[n < min_obs] = np.nan
        if not return_se:
            return beta
        s2 = (cyy - beta * cxy) / (n - 2)
        se2 = s2 / cxx
    return beta, se2


def monthly_excess(Y, x, rf, month_id):
    """Compound daily excess returns into monthly excess returns (total compounded minus rf compounded)."""
    K = month_id.max() + 1
    out_y = np.full((K, Y.shape[1]), np.nan)
    out_x = np.full(K, np.nan)
    for k in range(K):
        sl = month_id == k
        r_f = np.prod(1 + rf[sl]) - 1
        tot = Y[sl] + rf[sl, None]
        valid = ~np.isnan(tot).any(axis=0)
        out_y[k, valid] = np.prod(1 + tot[:, valid], axis=0) - 1 - r_f
        out_x[k] = np.prod(1 + x[sl] + rf[sl]) - 1 - r_f
    return out_y, out_x


def ols_monthly(Ym, xm, window=W_MONTHLY, min_obs=MIN_MONTHLY):
    """Rolling OLS on monthly returns; row k uses months k-window+1..k."""
    K = np.arange(len(xm))
    return ols_daily(Ym, xm, K, window, min_obs)


def blume(beta_1y):
    return BLUME_A + BLUME_B * beta_1y


def vasicek(beta_1y, se2_1y):
    """Empirical-Bayes shrinkage to the cross-sectional mean (Vasicek 1973), month by month."""
    out = np.full_like(beta_1y, np.nan)
    for k in range(beta_1y.shape[0]):
        b, s = beta_1y[k], se2_1y[k]
        ok = ~np.isnan(b)
        if ok.sum() < 5:
            continue
        prior_mean = b[ok].mean()
        prior_var = max(b[ok].var(ddof=1) - s[ok].mean(), 1e-4)
        w = prior_var / (prior_var + s[ok])
        out[k, ok] = w * b[ok] + (1 - w) * prior_mean
    return out


def ewma(Y, x, E, halflife=EWMA_HALFLIFE, min_obs=MIN_DAILY):
    lam = 0.5 ** (1 / halflife)
    T, N = Y.shape
    sw = np.zeros(N); sx = np.zeros(N); sy = np.zeros(N); sxx = np.zeros(N); sxy = np.zeros(N)
    cnt = np.zeros(N)
    out = np.full((len(E), N), np.nan)
    e_pos = {int(e): k for k, e in enumerate(E)}
    for t in range(T):
        sw *= lam; sx *= lam; sy *= lam; sxx *= lam; sxy *= lam
        v = ~np.isnan(Y[t])
        xt = x[t]
        sw[v] += 1; sx[v] += xt; sy[v] += Y[t, v]; sxx[v] += xt * xt; sxy[v] += xt * Y[t, v]
        cnt[v] += 1
        if t in e_pos:
            with np.errstate(invalid="ignore", divide="ignore"):
                mx, my = sx / sw, sy / sw
                b = (sxy / sw - mx * my) / (sxx / sw - mx * mx)
            b[cnt < min_obs] = np.nan
            out[e_pos[t]] = b
    return out


def dimson(Y, x, E, lags=DIMSON_LAGS, window=W_DAILY, min_obs=MIN_DAILY):
    """Sum of slopes on the market return and its first `lags` lags (Dimson 1979, lags only)."""
    T, N = Y.shape
    Z = [np.ones(T)] + [np.concatenate([np.full(l, np.nan), x[:T - l]]) for l in range(lags + 1)]
    Z = np.array(Z).T                                   # T x (lags+2)
    m = ~np.isnan(Y) & ~np.isnan(Z).any(axis=1)[:, None]
    Zz = np.nan_to_num(Z)
    p = Z.shape[1]
    A = np.zeros((len(E), N, p, p))
    bvec = np.zeros((len(E), N, p))
    mf = m.astype(float)
    Yz = np.where(m, Y, 0.0)
    for i in range(p):
        bvec[:, :, i] = _win(_csum(Zz[:, i, None] * Yz), E, window)
        for j in range(i, p):
            s = _win(_csum(Zz[:, i, None] * Zz[:, j, None] * mf), E, window)
            A[:, :, i, j] = s
            A[:, :, j, i] = s
    n = A[:, :, 0, 0]
    out = np.full((len(E), N), np.nan)
    ok = n >= min_obs
    coef = np.linalg.solve(A[ok], bvec[ok][..., None])[..., 0]
    out[ok] = coef[:, 1:].sum(axis=1)
    return out


def frazzini_pedersen(Y, x, E):
    """Correlation from overlapping 3-day log returns (5y), volatilities from 1-day log returns (1y)."""
    ly, lx = np.log1p(Y), np.log1p(x)
    r3y = ly + np.vstack([np.full((1, Y.shape[1]), np.nan), ly[:-1]]) + \
        np.vstack([np.full((2, Y.shape[1]), np.nan), ly[:-2]])
    r3x = lx + np.concatenate([[np.nan], lx[:-1]]) + np.concatenate([[np.nan] * 2, lx[:-2]])
    m = ~np.isnan(r3y) & ~np.isnan(r3x)[:, None]
    X = np.where(m, r3x[:, None], 0.0)
    Yz = np.where(m, r3y, 0.0)
    w = FP_CORR_WINDOW
    n = _win(_csum(m.astype(float)), E, w)
    sx, sy = _win(_csum(X), E, w), _win(_csum(Yz), E, w)
    sxx, syy, sxy = _win(_csum(X * X), E, w), _win(_csum(Yz * Yz), E, w), _win(_csum(X * Yz), E, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (sxy - sx * sy / n) / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))
    rho[n < FP_CORR_MIN] = np.nan
    # 1-day volatilities over one year
    m1 = ~np.isnan(ly)
    n1 = _win(_csum(m1.astype(float)), E, W_DAILY)
    s1, s2 = _win(_csum(np.where(m1, ly, 0)), E, W_DAILY), _win(_csum(np.where(m1, ly * ly, 0)), E, W_DAILY)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd_y = np.sqrt((s2 - s1 * s1 / n1) / (n1 - 1))
    sd_y[n1 < MIN_DAILY] = np.nan
    cx = _csum(lx[:, None])[:, 0]
    cxx = _csum((lx * lx)[:, None])[:, 0]
    nx = np.minimum(E + 1, W_DAILY)
    sxm, sxxm = _win(cx, E, W_DAILY), _win(cxx, E, W_DAILY)
    sd_x = np.sqrt((sxxm - sxm * sxm / nx) / (nx - 1))
    beta_ts = rho * sd_y / sd_x[:, None]
    return FP_SHRINK * beta_ts + (1 - FP_SHRINK) * 1.0


def welch_swa(Y, x, E, window=WELCH_WINDOW, rho=WELCH_RHO, bounds=WELCH_BOUNDS, min_obs=WELCH_MIN):
    """Slope-winsorized, age-weighted beta (Welch 2022): returns clipped to [lo*x, hi*x] around the
    market's own move, then exponentially age-weighted OLS with intercept."""
    lo_k, hi_k = bounds
    out = np.full((len(E), Y.shape[1]), np.nan)
    for k, e in enumerate(E):
        s = max(0, e - window + 1)
        y, xx = Y[s:e + 1], x[s:e + 1]
        a, b = lo_k * xx, hi_k * xx
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        yw = np.clip(y, lo[:, None], hi[:, None])
        age = (e - np.arange(s, e + 1)).astype(float)
        w = np.exp(-rho * age)[:, None] * (~np.isnan(y))
        yw = np.nan_to_num(yw)
        sw = w.sum(0)
        mx = (w * xx[:, None]).sum(0) / sw
        my = (w * yw).sum(0) / sw
        cxy = (w * (xx[:, None] - mx) * (yw - my)).sum(0)
        cxx = (w * (xx[:, None] - mx) ** 2).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            bk = cxy / cxx
        bk[(~np.isnan(y)).sum(0) < min_obs] = np.nan
        out[k] = bk
    return out


def all_betas(Y, x, rf, dates_month_id, E):
    """Return dict name -> (K x N) betas at the month-ends E (E[k] = last day of month k)."""
    b1y, se2 = ols_daily(Y, x, E, return_se=True)
    Ym, xm = monthly_excess(Y, x, rf, dates_month_id)
    b5m = ols_monthly(Ym, xm)
    out = {
        "OLS-5Y-M": b5m,
        "OLS-1Y-D": b1y,
        "Blume": blume(b1y),
        "Vasicek": vasicek(b1y, se2),
        "EWMA": ewma(Y, x, E),
        "Dimson": dimson(Y, x, E),
        "Frazzini-Pedersen": frazzini_pedersen(Y, x, E),
        "Welch-SWA": welch_swa(Y, x, E),
    }
    stack = np.stack([out[k] for k in ESTIMATORS[:-1]])
    out["Combination"] = stack.mean(axis=0)          # NaN if any component is missing
    out["One"] = np.where(np.isnan(b1y), np.nan, 1.0)
    return out, Ym, xm
