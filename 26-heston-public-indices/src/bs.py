"""A Black-Scholes / Black-76 pricer that stays accurate in the wings.

Everything is written on the forward F = S exp((r - q) T) with discount factor D = exp(-r T), so the
same code prices equity (Black-Scholes-Merton), futures (Black-76) and FX (Garman-Kohlhagen) options.

Two implementation choices matter for accuracy and are tested against a 50-digit mpmath reference
in tests/test_bs.py and src/accuracy.py:

1. Always price the out-of-the-money side and obtain the in-the-money side by put-call parity.
   An ITM value is "intrinsic + small number", so computing the small number (the OTM value) to
   full relative precision is what matters.
2. Price the OTM option with the scaled complementary error function erfcx. The textbook
   F N(d1) - K N(d2) subtracts two nearly equal tiny numbers in the wings and loses every digit;
   with N(-x) = erfcx(x / sqrt 2) exp(-x^2 / 2) / 2 and the identity K exp(-d2^2/2) = F exp(-d1^2/2)
   the OTM value becomes F exp(-d1^2/2) / 2 * [erfcx(-d1/sqrt2) - erfcx(-d2/sqrt2)], a difference
   of two O(1) numbers whose relative error stays bounded.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfcx, ndtr

SQRT2 = np.sqrt(2.0)
SQRT2PI = np.sqrt(2.0 * np.pi)


def _d1d2(F, K, v):
    """v = sigma * sqrt(T) is the total volatility."""
    k = np.log(F / K)
    d1 = k / v + 0.5 * v
    return d1, d1 - v


def npdf(x):
    return np.exp(-0.5 * x * x) / SQRT2PI


# ----------------------------------------------------------------------------------------------
# prices
# ----------------------------------------------------------------------------------------------
def _otm_call_normalised(x, v):
    """Undiscounted OTM call value divided by F, for x = ln(F/K) <= 0 (strike at or above forward).

    Stable form: exp(-d1^2/2)/2 * [erfcx(-d1/sqrt2) - erfcx(-d2/sqrt2)].
    """
    d1 = x / v + 0.5 * v
    d2 = d1 - v
    return 0.5 * np.exp(-0.5 * d1 * d1) * (erfcx(-d1 / SQRT2) - erfcx(-d2 / SQRT2))


def black_price(F, K, T, sigma, D=1.0, call=True):
    """Discounted Black-76 price. Vectorised over all arguments.

    The OTM side is computed directly with the erfcx form; the ITM side by parity.
    """
    F, K, T, sigma, D, call = np.broadcast_arrays(*(np.asarray(a, float) for a in (F, K, T, sigma, D, call)))
    call = call.astype(bool)
    v = sigma * np.sqrt(T)
    x = np.log(F / K)
    # OTM call value (strike >= forward) and OTM put value (strike < forward) via symmetry:
    # put(F, K) / K = call(K, F) / K -> normalised by the larger of the two
    otm_is_call = x <= 0
    xc = np.where(otm_is_call, x, -x)  # for the put, swap roles of F and K: put(F,K) = K * c(ln(K/F))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        c_norm = _otm_call_normalised(xc, np.maximum(v, 1e-300))
    otm = np.where(otm_is_call, F * c_norm, K * c_norm)
    otm = np.where(v > 0, otm, 0.0)
    intrinsic_call = np.maximum(F - K, 0.0)
    intrinsic_put = np.maximum(K - F, 0.0)
    # value of the requested side
    call_val = np.where(otm_is_call, otm, otm + (F - K))  # ITM call = OTM put + (F - K)
    put_val = np.where(otm_is_call, otm + (K - F), otm)
    call_val = np.where(v > 0, call_val, intrinsic_call)
    put_val = np.where(v > 0, put_val, intrinsic_put)
    out = D * np.where(call, call_val, put_val)
    return out if out.ndim else float(out)


def black_price_naive(F, K, T, sigma, D=1.0, call=True):
    """Textbook formula, for comparison only: D [F N(d1) - K N(d2)] and parity for the put."""
    v = sigma * np.sqrt(T)
    d1, d2 = _d1d2(F, K, v)
    c = D * (F * ndtr(d1) - K * ndtr(d2))
    return c if call else c - D * (F - K)


def bs_price(S, K, T, r, sigma, q=0.0, call=True):
    """Black-Scholes-Merton on spot with continuous dividend yield q."""
    F = S * np.exp((r - q) * T)
    return black_price(F, K, T, sigma, np.exp(-r * T), call)


# ----------------------------------------------------------------------------------------------
# Greeks (analytic, spot convention, BSM with r and q)
# ----------------------------------------------------------------------------------------------
def bs_greeks(S, K, T, r, sigma, q=0.0, call=True):
    """Return dict with delta, gamma, vega, theta (per year), rho, vanna, volga, cash_gamma (Gamma S^2)."""
    S, K, T, r, sigma, q = (np.asarray(a, float) for a in (S, K, T, r, sigma, q))
    sqT = np.sqrt(T)
    v = sigma * sqT
    F = S * np.exp((r - q) * T)
    d1, d2 = _d1d2(F, K, v)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    pdf1 = npdf(d1)
    sign = 1.0 if call else -1.0
    delta = sign * dq * ndtr(sign * d1)
    gamma = dq * pdf1 / (S * v)
    vega = S * dq * pdf1 * sqT
    theta = (-S * dq * pdf1 * sigma / (2 * sqT)
             - sign * r * K * dr * ndtr(sign * d2)
             + sign * q * S * dq * ndtr(sign * d1))
    rho = sign * K * T * dr * ndtr(sign * d2)
    vanna = -dq * pdf1 * d2 / sigma
    volga = vega * d1 * d2 / sigma
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho, vanna=vanna, volga=volga,
                cash_gamma=gamma * S * S)


# ----------------------------------------------------------------------------------------------
# implied volatility
# ----------------------------------------------------------------------------------------------
def implied_vol(price, F, K, T, D=1.0, call=True, tol=1e-14, maxiter=100):
    """Implied Black volatility, robust over the whole (moneyness, total-vol) plane.

    * converts the quote to the OTM side (parity), normalises by F, works in total vol v = sigma sqrt T;
    * keeps a bracket [lo, hi] at all times and takes a Newton step on log(price) when it lands inside
      the bracket, a bisection step otherwise (log-price Newton converges in the far wings where plain
      Newton on price overshoots to negative volatility);
    * returns NaN for quotes outside the no-arbitrage bounds.
    """
    price, F, K, T, D, call = np.broadcast_arrays(*(np.asarray(a, float) for a in (price, F, K, T, D, call)))
    call = call.astype(bool)
    undisc = price / D
    x = np.log(F / K)
    otm_is_call = x <= 0
    # OTM value
    otm = np.where(call == otm_is_call, undisc, np.where(call, undisc - (F - K), undisc - (K - F)))
    scale = np.where(otm_is_call, F, K)
    xc = np.where(otm_is_call, x, -x)
    target = otm / scale  # normalised OTM call value with x <= 0; no-arbitrage bounds (0, 1)
    bad = (target <= 0) | (target >= 1.0) | ~np.isfinite(target)
    tgt = np.where(bad, 0.5, target)
    lo = np.full(tgt.shape, 1e-8)
    hi = np.full(tgt.shape, 50.0)
    # initial guess: Brenner-Subrahmanyam near the money, wing-aware lower bound elsewhere
    v = np.clip(np.sqrt(2 * np.pi) * tgt + np.sqrt(2 * np.abs(xc)), 1e-4, 10.0)
    lt = np.log(tgt)
    for _ in range(maxiter):
        with np.errstate(all="ignore"):
            c = _otm_call_normalised(xc, v)
            f = np.log(c) - lt
            d1 = xc / v + 0.5 * v
            # d log c / d v = vega / c, vega_normalised = pdf(d1)
            dfdv = npdf(d1) / c
        lo = np.where(f < 0, v, lo)
        hi = np.where(f > 0, v, hi)
        vn = v - f / dfdv
        inside = np.isfinite(vn) & (vn > lo) & (vn < hi)
        vn = np.where(inside, vn, 0.5 * (lo + hi))
        done = np.abs(vn - v) <= tol * np.maximum(v, 1e-8)
        v = vn
        if done.all():
            break
    sigma = v / np.sqrt(T)
    sigma = np.where(bad, np.nan, sigma)
    return sigma if sigma.ndim else float(sigma)
