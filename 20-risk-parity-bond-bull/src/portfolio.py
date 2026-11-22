"""Weights, the break-even hurdle, and the monthly backtest engine for project 20."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------- estimators


def sample_cov(X: np.ndarray) -> np.ndarray:
    return np.cov(X, rowvar=False, ddof=1)


def ewma_cov(X: np.ndarray, halflife: float = 12.0) -> np.ndarray:
    n = len(X)
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(n - 1, -1, -1)
    w /= w.sum()
    mu = w @ X
    Z = X - mu
    return (Z * w[:, None]).T @ Z


# ---------------------------------------------------------------- weights


def inverse_vol(S: np.ndarray) -> np.ndarray:
    s = np.sqrt(np.diag(S))
    w = 1.0 / s
    return w / w.sum()


def risk_contributions(w: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Share of portfolio variance from each asset: w_i (S w)_i / w'Sw. Sums to one."""
    m = S @ w
    return w * m / (w @ m)


def erc(S: np.ndarray, tol: float = 1e-12, max_iter: int = 10_000) -> np.ndarray:
    """Equal-risk-contribution weights (long-only, sum to one), by the cyclical coordinate method of
    Griveau-Billion, Richard & Roncalli (2013)."""
    n = S.shape[0]
    x = inverse_vol(S).copy()
    b = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sig = np.sqrt(x @ S @ x)
            a = S[i, i]
            c = S[i] @ x - a * x[i]
            x[i] = (-c + np.sqrt(c * c + 4 * a * b[i] * sig)) / (2 * a)
        if np.max(np.abs(x - x_old)) < tol:
            break
    return x / x.sum()


def port_vol(w: np.ndarray, S: np.ndarray) -> float:
    return float(np.sqrt(w @ S @ w))


def scale_to_vol(w: np.ndarray, S: np.ndarray, target: float) -> np.ndarray:
    return w * (target / port_vol(w, S))


def max_sharpe_long_only(mu: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Long-only tangency direction (weights sum to one). Falls back to inverse-vol if no mean is positive."""
    n = len(mu)
    if np.all(mu <= 0):
        return inverse_vol(S)

    def neg(w):
        return -(w @ mu) / np.sqrt(w @ S @ w)

    best = None
    for w0 in [np.full(n, 1 / n), inverse_vol(S)]:
        r = minimize(neg, w0, method="SLSQP", bounds=[(0, 1)] * n,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"ftol": 1e-12})
        if best is None or r.fun < best.fun:
            best = r
    w = np.clip(best.x, 0, None)
    return w / w.sum()


# ---------------------------------------------------------------- the hurdle


def sharpe_of_direction(k: float, r: float, rho: float) -> float:
    """Sharpe of a stock/bond portfolio, divided by the stock Sharpe S_s, when the bond position carries k times
    the stand-alone volatility of the stock position. r = S_b / S_s.  (1 + k r) / sqrt(1 + k^2 + 2 rho k)."""
    return (1 + k * r) / np.sqrt(1 + k * k + 2 * rho * k)


def hurdle(rho: float, a: float) -> float:
    """Break-even ratio r* = S_b / S_s above which risk parity (k = 1) has a higher Sharpe than a portfolio whose
    bond position carries a times the stock position's stand-alone volatility (60/40: a = 0.4 s_b / (0.6 s_s)).
    Valid when S_s > 0. Derived from (1 + r)/A = (1 + a r)/B with A = sqrt(2 + 2 rho), B = sqrt(1 + a^2 + 2 rho a)."""
    A = np.sqrt(2 + 2 * rho)
    B = np.sqrt(1 + a * a + 2 * rho * a)
    return (A - B) / (B - a * A)


# ---------------------------------------------------------------- backtest

WeightFn = Callable[[np.ndarray, np.ndarray, dict], np.ndarray]


def backtest(X: pd.DataFrame, rf: pd.Series, weight_fn: WeightFn, window: int, tc: float, spread: float,
             cov_fn: Callable = sample_cov, start: pd.Period | None = None, hist: pd.DataFrame | None = None,
             ) -> pd.DataFrame:
    """Monthly rebalanced backtest on excess returns X (columns = assets, futures-style positions).

    At month t the weights use only rows of X strictly before t (the trailing `window` months for the covariance,
    and `hist` = all earlier rows for rules that need means). Portfolio return:
        r_p = rf + w.x - spread/12 * max(sum(w) - 1, 0) - tc * turnover,
    where turnover is measured against the weights after last month's drift.
    """
    cols = list(X.columns)
    Xv = X.values
    out = []
    w_prev = None
    idx = X.index
    t0 = window if start is None else max(window, idx.get_loc(start))
    for t in range(t0, len(X)):
        win = Xv[t - window:t]
        if np.isnan(win).any():
            continue
        S = cov_fn(win) * 12.0
        ctx = {"S": S, "past": Xv[:t] if hist is None else hist.values[:t], "t": idx[t]}
        w = np.asarray(weight_fn(win, S, ctx), float)
        x = Xv[t]
        if np.isnan(x).any():
            continue
        gross = float(w @ x)
        lev = float(w.sum())
        fund = spread / 12.0 * max(lev - 1.0, 0.0)
        if w_prev is None:
            turn = float(np.abs(w).sum())
        else:
            turn = float(np.abs(w - w_prev).sum())
        cost = tc * turn
        rp = rf.iloc[t] + gross - fund - cost
        row = {"month": idx[t], "ret": rp, "xret": rp - rf.iloc[t], "gross_x": gross, "lev": lev,
               "turnover": turn, "fund_cost": fund, "tc_cost": cost, "ex_ante_vol": port_vol(w, S)}
        for c, wi, xi in zip(cols, w, x):
            row[f"w_{c}"] = wi
            row[f"pnl_{c}"] = wi * xi
        out.append(row)
        # weights after drift (positions move with the underlying total return; NAV moves with r_p)
        w_prev = w * (1 + rf.iloc[t] + x) / (1 + rp)
    return pd.DataFrame(out).set_index("month")


# ---------------------------------------------------------------- strategy library


def strat_6040(win, S, ctx):
    return np.array([0.6, 0.4] + [0.0] * (S.shape[0] - 2))


def _bench_vol(S):
    w = np.zeros(S.shape[0])
    w[:2] = [0.6, 0.4]
    return port_vol(w, S)


def strat_rp_matched(win, S, ctx):
    """Inverse-volatility on stocks and bonds, scaled to the ex-ante volatility of 60/40."""
    S2 = S[:2, :2]
    w = np.zeros(S.shape[0])
    w[:2] = scale_to_vol(inverse_vol(S2), S2, _bench_vol(S))
    return w


def strat_rp_unlevered(win, S, ctx):
    S2 = S[:2, :2]
    w = np.zeros(S.shape[0])
    w[:2] = inverse_vol(S2)
    return w


def strat_iv_all_matched(win, S, ctx):
    return scale_to_vol(inverse_vol(S), S, _bench_vol(S))


def strat_erc_all_matched(win, S, ctx):
    return scale_to_vol(erc(S), S, _bench_vol(S))


def make_target_vol(target: float):
    def f(win, S, ctx):
        S2 = S[:2, :2]
        w = np.zeros(S.shape[0])
        w[:2] = scale_to_vol(inverse_vol(S2), S2, target)
        return w
    return f


def make_mvo_hist(mean_window: int | None):
    """Long-only max-Sharpe on trailing mean excess returns (expanding if mean_window is None), scaled to the
    ex-ante vol of 60/40. Link to project 05: does knowing (estimating) the means help?"""
    def f(win, S, ctx):
        past = ctx["past"][:, :S.shape[0]]
        if mean_window is not None:
            past = past[-mean_window:]
        mu = np.nanmean(past, axis=0) * 12.0
        w = max_sharpe_long_only(mu, S)
        return scale_to_vol(w, S, _bench_vol(S))
    return f


def make_switch(mean_window: int = 120):
    """Hold risk parity when trailing estimates say it has the higher Sharpe than 60/40, else hold 60/40.
    Uses trailing means (mean_window months) and the trailing covariance: an ex-ante version of the hurdle."""
    def f(win, S, ctx):
        past = ctx["past"][-mean_window:, :2]
        mu = np.nanmean(past, axis=0) * 12.0
        S2 = S[:2, :2]
        w_rp = scale_to_vol(inverse_vol(S2), S2, _bench_vol(S))
        w_b = np.array([0.6, 0.4])
        sr = lambda w: (w @ mu) / port_vol(w, S2)  # noqa: E731
        w = w_rp if sr(w_rp) >= sr(w_b) else w_b
        out = np.zeros(S.shape[0])
        out[:2] = w
        return out
    return f


def make_rp_capped(cap: float):
    """Risk parity matched to 60/40's ex-ante vol, but with total notional capped at `cap` times NAV."""
    def f(win, S, ctx):
        w = strat_rp_matched(win, S, ctx)
        lev = w.sum()
        return w * (cap / lev) if lev > cap else w
    return f
