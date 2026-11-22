"""A small, reusable discrete delta-hedging simulator.

The engine takes a batch of price paths (one row per option, one column per trading day, the first
column is the day the option is sold), a strike, a volatility to compute deltas with, a rebalancing
rule and a cost model, and returns the P&L of the hedged short-option book, broken into its parts.

Conventions
-----------
* The book is SHORT one European option and LONG `pos` units of the underlying. It is the position of a
  market maker who sold the option and hedges it.
* Rates and dividends are set to zero. The hedge instrument is then the forward (or a future), and
  `S` is its price. Over one month the carry changes the P&L variance by a negligible amount
  (checked in tests/test_hedgesim.py::test_carry_is_negligible).
* Time is counted in trading days, with `dt = 1/252` years per day.
* P&L is reported per unit of the option's vega at inception and multiplied by 100, so a P&L of +1.0
  means "one volatility point". A hedging error with a standard deviation of 2.0 is as large as
  mis-quoting the option by 2 vol points.

This module is the hedging core that project 30 (deep hedging) reuses: any delta rule can be passed
in as `delta_fn`, and any rebalancing rule as a boolean schedule or a band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

DT = 1.0 / 252.0
SQRT2PI = np.sqrt(2.0 * np.pi)


# ----------------------------------------------------------------------------------------------
# Black-Scholes pieces with r = q = 0 (forward measure), vectorised
# ----------------------------------------------------------------------------------------------
def _d1(S, K, tau, sigma):
    v = sigma * np.sqrt(tau)
    return np.log(S / K) / v + 0.5 * v, v


def call_price(S, K, tau, sigma):
    S, K, tau, sigma = np.broadcast_arrays(*(np.asarray(a, float) for a in (S, K, tau, sigma)))
    out = np.maximum(S - K, 0.0)
    live = tau > 0
    if live.any():
        d1, v = _d1(S[live], K[live], tau[live], sigma[live])
        out = out.copy()
        out[live] = S[live] * ndtr(d1) - K[live] * ndtr(d1 - v)
    return out


def call_delta(S, K, tau, sigma):
    S, K, tau, sigma = np.broadcast_arrays(*(np.asarray(a, float) for a in (S, K, tau, sigma)))
    out = (S > K).astype(float)
    live = tau > 0
    if live.any():
        d1, _ = _d1(S[live], K[live], tau[live], sigma[live])
        out = out.copy()
        out[live] = ndtr(d1)
    return out


def call_vega(S, K, tau, sigma):
    """dC/dsigma (per 1.00 of volatility)."""
    d1, _ = _d1(S, K, tau, sigma)
    return S * np.exp(-0.5 * d1 * d1) / SQRT2PI * np.sqrt(tau)


def call_gamma(S, K, tau, sigma):
    d1, v = _d1(S, K, tau, sigma)
    return np.exp(-0.5 * d1 * d1) / SQRT2PI / (S * v)


# ----------------------------------------------------------------------------------------------
# rebalancing schedules
# ----------------------------------------------------------------------------------------------
def every_h_days(n_days: int, h: int) -> np.ndarray:
    """Boolean vector of length n_days: True on the days the hedge is reset (day 0 always)."""
    sched = np.zeros(n_days, bool)
    sched[::h] = True
    return sched


@dataclass
class HedgeResult:
    pnl: np.ndarray          # total P&L per path, in vol points (see module docstring)
    hedge_gains: np.ndarray  # sum of pos * dS, in vol points
    option_leg: np.ndarray   # premium - payoff, in vol points
    cost: np.ndarray         # transaction costs paid (positive number), in vol points
    n_trades: np.ndarray     # number of rebalances after inception
    turnover: np.ndarray     # sum |d pos|, units of underlying per unit of option
    vega0: np.ndarray        # vega at inception per 1.00 vol (in price units)
    premium: np.ndarray      # option premium received, in price units


def hedge_short_call(
    S: np.ndarray,
    K: np.ndarray | float,
    sigma_quote: np.ndarray | float,
    sigma_delta: np.ndarray | float | None = None,
    schedule: np.ndarray | None = None,
    band: float | None = None,
    delta_fn=None,
    cost_bps: float = 0.0,
    unwind: bool = True,
) -> HedgeResult:
    """Sell a European call at `sigma_quote` and delta-hedge it along each row of `S`.

    Parameters
    ----------
    S : (n, T+1) prices. Column 0 is inception, column T is expiry.
    K : strike(s), scalar or (n,).
    sigma_quote : volatility the option is sold at, scalar or (n,).
    sigma_delta : volatility used inside the delta. Scalar, (n,) constant per path, or (n, T) time
        varying (column i is used on day i). Defaults to sigma_quote.
    schedule : boolean (T,) rebalancing days for a fixed calendar rule. Day 0 must be True.
    band : if given, rebalance on any day where |target delta - held delta| > band (day 0 always).
        `schedule` and `band` are mutually exclusive; with neither, the hedge is reset daily.
    delta_fn : optional callable(S_i, K, tau_i, i) -> target delta (n,), overrides the BS delta.
        Used for the minimum-variance delta and by project 30.
    cost_bps : proportional cost per unit of notional traded, in basis points (half-spread + fees).
    unwind : if True the hedge is closed at expiry and pays the cost on the final position.
    """
    S = np.asarray(S, float)
    n, T1 = S.shape
    T = T1 - 1
    K = np.broadcast_to(np.asarray(K, float), (n,))
    sq = np.broadcast_to(np.asarray(sigma_quote, float), (n,))
    if sigma_delta is None:
        sigma_delta = sq
    sd = np.asarray(sigma_delta, float)
    if sd.ndim <= 1:
        sd = np.broadcast_to(sd.reshape(-1, 1) if sd.ndim == 1 else sd, (n, T))
    if schedule is not None and band is not None:
        raise ValueError("use either a calendar schedule or a band, not both")
    if schedule is None:
        schedule = np.ones(T, bool)
    schedule = np.asarray(schedule, bool)
    if not schedule[0]:
        raise ValueError("the hedge must be put on at inception (schedule[0] = True)")

    tau0 = T * DT
    premium = call_price(S[:, 0], K, tau0, sq)
    vega0 = call_vega(S[:, 0], K, tau0, sq)
    k = cost_bps * 1e-4

    pos = np.zeros(n)
    gains = np.zeros(n)
    cost = np.zeros(n)
    turnover = np.zeros(n)
    n_trades = np.zeros(n)
    for i in range(T):
        tau = (T - i) * DT
        if delta_fn is not None:
            target = delta_fn(S[:, i], K, tau, i)
        else:
            target = call_delta(S[:, i], K, tau, sd[:, i])
        if band is not None:
            trade = np.abs(target - pos) > band if i > 0 else np.ones(n, bool)
        else:
            trade = np.full(n, schedule[i])
        new = np.where(trade, target, pos)
        dpos = np.abs(new - pos)
        turnover += dpos
        cost += k * dpos * S[:, i]
        if i > 0:
            n_trades += trade & (dpos > 0)
        pos = new
        gains += pos * (S[:, i + 1] - S[:, i])
    if unwind:
        cost += k * np.abs(pos) * S[:, T]
        turnover += np.abs(pos)
    payoff = np.maximum(S[:, T] - K, 0.0)
    unit = vega0 / 100.0  # price of one vol point at inception
    opt = premium - payoff
    return HedgeResult(
        pnl=(opt + gains - cost) / unit,
        hedge_gains=gains / unit,
        option_leg=opt / unit,
        cost=cost / unit,
        n_trades=n_trades,
        turnover=turnover,
        vega0=vega0,
        premium=premium,
    )


# ----------------------------------------------------------------------------------------------
# path windows from a daily price series
# ----------------------------------------------------------------------------------------------
def windows(prices: np.ndarray, T: int, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """All overlapping (T+1)-day windows of a 1-d price array, normalised to start at 100.

    Returns (paths (m, T+1), start_index (m,)).
    """
    p = np.asarray(prices, float)
    m = len(p) - T
    idx = np.arange(0, m, step)
    W = np.lib.stride_tricks.sliding_window_view(p, T + 1)[idx]
    return 100.0 * W / W[:, :1], idx


# ----------------------------------------------------------------------------------------------
# theory: the Boyle-Emanuel / Kamal-Derman discrete-hedging law under GBM
# ----------------------------------------------------------------------------------------------
def derman_std_volpts(sigma: float, n_rebalances: int) -> float:
    """Standard deviation of the hedging error of an ATM option hedged N times under GBM, in vol points.

    std(P&L) ~= sqrt(pi/4) * vega * sigma / sqrt(N)  (Kamal & Derman 1999; Boyle & Emanuel 1980).
    Dividing by vega/100 gives vol points.
    """
    return 100.0 * np.sqrt(np.pi / 4.0) * sigma / np.sqrt(n_rebalances)


def floor_fit(h: np.ndarray, var: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    """Least-squares fit of the linear frequency law Var(h) = a + b h. Returns (a, b)."""
    h = np.asarray(h, float)
    var = np.asarray(var, float)
    w = np.ones_like(h) if w is None else np.asarray(w, float)
    X = np.column_stack([np.ones_like(h), h]) * np.sqrt(w)[:, None]
    a, b = np.linalg.lstsq(X, var * np.sqrt(w), rcond=None)[0]
    return float(a), float(b)
