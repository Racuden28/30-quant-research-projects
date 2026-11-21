"""Daily variance proxies and the weighting schemes a volatility dashboard can apply to them.

All inputs are log quantities for one asset on one day t:
    r  close-to-close log return on adjusted closes   ln(A_t / A_{t-1})
    o  overnight log return                          r - c   (dividend-adjusted by construction)
    u  ln(H_t / O_t)   (>= 0)
    d  ln(L_t / O_t)   (<= 0)
    c  ln(C_t / O_t)

Every range-based estimator below adds the overnight term o^2, so all of them target the same
close-to-close daily variance as r^2. The one deliberate exception is ``PK_noON`` (Parkinson on the
intraday range only), kept as an ablation because it is what many dashboards actually plot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LN2 = np.log(2.0)
ESTIMATORS = ["C2C", "PK", "GK", "RS", "YZ"]           # the selection pool
ABLATIONS = ["PK_noON", "GK_cal"]                       # reported, never selected
ROLL_WINDOWS = [5, 10, 21, 63, 126, 252]
EWMA_HALFLIVES = [5, 10, 21, 42, 63, 126]
WEIGHTINGS = [f"R{n}" for n in ROLL_WINDOWS] + [f"E{h}" for h in EWMA_HALFLIVES]
DEFAULT = ("C2C", "R21", "sqrt")                        # the textbook dashboard number


def daily_proxies(r, o, u, d, c) -> pd.DataFrame:
    """Per-day variance proxies (YZ is window-level, handled in ``weighted_variance``)."""
    pk_intra = (u - d) ** 2 / (4 * LN2)
    gk_intra = 0.5 * (u - d) ** 2 - (2 * LN2 - 1) * c ** 2
    rs_intra = u * (u - c) + d * (d - c)
    return pd.DataFrame({
        "C2C": r ** 2,
        "PK": pk_intra + o ** 2,
        "GK": gk_intra + o ** 2,
        "RS": rs_intra + o ** 2,
        "PK_noON": pk_intra,
        "RS_intra": rs_intra,
    })


def _weighted(series: pd.Series, w: str, fn: str = "mean") -> pd.Series:
    kind, n = w[0], int(w[1:])
    if kind == "R":
        roll = series.rolling(n, min_periods=n)
        return roll.mean() if fn == "mean" else roll.var(ddof=1)
    ew = series.ewm(halflife=n, min_periods=2 * n, adjust=True)
    return ew.mean() if fn == "mean" else ew.var(bias=False)


def effective_n(w: str) -> float:
    kind, n = w[0], int(w[1:])
    if kind == "R":
        return float(n)
    lam = 0.5 ** (1.0 / n)
    return (1 + lam) / (1 - lam)


def weighted_variance(px: pd.DataFrame, o: pd.Series, c: pd.Series, est: str, w: str) -> pd.Series:
    """Daily-variance estimate known at the close of each day, for estimator ``est`` and weighting ``w``."""
    if est == "YZ":
        n = effective_n(w)
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        return _weighted(o, w, "var") + k * _weighted(c, w, "var") + (1 - k) * _weighted(px["RS_intra"], w)
    if est == "GK_cal":
        # GK shape, rescaled so its trailing one-year mean matches the C2C mean (known at t, no look-ahead)
        ratio = px["C2C"].rolling(252, min_periods=252).mean() / px["GK"].rolling(252, min_periods=252).mean()
        return _weighted(px["GK"], w) * ratio
    return _weighted(px[est], w)


def variance_ratio_hat(r: pd.Series, h: int, window: int = 756) -> pd.Series:
    """Trailing estimate of VR(h) = Var(sum of h daily returns) / (h Var(daily)), known at t.

    Uses overlapping h-day sums inside a ``window``-day trailing sample and is shrunk towards 1 with
    weight 0.5 because overlapping VR estimates are noisy (pre-registered, not tuned).
    """
    rs = r.rolling(h).sum()
    num = rs.rolling(window, min_periods=window).var()
    den = h * r.rolling(window, min_periods=window).var()
    vr = (num / den).clip(0.5, 2.0)
    return 0.5 + 0.5 * vr
