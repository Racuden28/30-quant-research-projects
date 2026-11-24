"""
Why the likelihood picks a hedge ratio that forgets too fast: a controlled simulation.

The random-walk-beta model says the one-step error variance is R (1 + x_t^2 p^-): errors are larger on days the
market moves a lot. Real hedge errors have exactly that property for another reason: idiosyncratic and market
volatility cluster together. The likelihood cannot tell the two apart, so it buys the heteroskedasticity with a
larger process noise, i.e. a shorter memory. Four worlds, 200 simulations of 6,000 days each:

  A  constant beta, homoskedastic errors and market         (true memory: infinite)
  B  constant beta, market and idiosyncratic volatility share one GARCH(1,1) factor
  C  random-walk beta with a 63-day steady-state half-life, homoskedastic errors
  D  like B, but the idiosyncratic volatility follows its own, independent GARCH factor
  E  transient beta noise: beta_t = 1 + u_t with u_t IID N(0, 0.3^2) (a random coefficient, not a random walk)
  F  B and E together, with Student-t(4) shocks
  G  constant beta; market and idiosyncratic volatility share one slow log-volatility factor (AR(1), half-life
     500 days, s.d. of log volatility 0.6: decade-like swings such as the 1930s against the 1960s)
  H  G plus a fast GARCH factor and Student-t(4) shocks
  I  C's random-walk beta (true half-life 63 days) inside H's volatility and tails

In each simulation the first 4,000 days choose r (likelihood and past hedging loss) and the last 2,000 days score
the hedge. Output: results/mechanism.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import RES
from kalman import R_GRID, kf_scalar_grid, matched_halflife, r_from_halflife

T, T_FIT, NSIM = 6000, 4000, 200
S2 = 1e-4            # market variance a day (1 % daily volatility)
IDIO = 1.0           # idiosyncratic / market variance ratio


def garch(T, rng, omega=0.02, a=0.08, b=0.90):
    h = np.empty(T)
    z = rng.standard_normal(T)
    h[0] = omega / (1 - a - b)
    for t in range(1, T):
        h[t] = omega + a * h[t - 1] * z[t - 1] ** 2 + b * h[t - 1]
    return np.sqrt(h / h.mean()), z


def slow_vol(T, rng, hl=500.0, sd=0.6):
    phi = 0.5 ** (1 / hl)
    e = rng.standard_normal(T) * sd * np.sqrt(1 - phi * phi)
    lv = np.empty(T)
    lv[0] = rng.standard_normal() * sd
    for t in range(1, T):
        lv[t] = phi * lv[t - 1] + e[t]
    v = np.exp(lv)
    return v / np.sqrt((v * v).mean())


def world(name, rng):
    if name in ("A", "C", "E"):
        sx = np.ones(T)
        se = np.ones(T)
    elif name in ("B", "F"):
        sx, _ = garch(T, rng)
        se = sx
    elif name == "D":
        sx, _ = garch(T, rng)
        se, _ = garch(T, rng)
    elif name == "G":
        sx = slow_vol(T, rng)
        se = sx
    else:  # H and I
        sx = slow_vol(T, rng) * garch(T, rng)[0]
        se = sx
    if name in ("F", "H", "I"):
        sh = lambda: rng.standard_t(4, T) / np.sqrt(2.0)          # unit variance
    else:
        sh = lambda: rng.standard_normal(T)
    x = np.sqrt(S2) * sx * sh()
    eps = np.sqrt(S2 * IDIO) * se * sh()
    beta = np.ones(T)
    if name in ("E", "F"):
        beta = 1.0 + 0.3 * rng.standard_normal(T)
    if name in ("C", "I"):
        r = float(r_from_halflife(63.0, S2))                      # q / R for a 63-day half-life
        q = r * S2 * IDIO
        beta = 1.0 + np.cumsum(np.sqrt(q) * rng.standard_normal(T))
    return beta * x + eps, x


def run():
    rows = []
    for name in "ABCDEFGHI":
        for s in range(NSIM):
            rng = np.random.default_rng([25, ord(name), s])
            y, x = world(name, rng)
            B, LL, _ = kf_scalar_grid(y, x, R_GRID)
            g_ml = int(np.nanargmax(LL[T_FIT]))
            e2 = (y[:, None] - B * x[:, None]) ** 2
            g_sse = int(np.argmin(e2[:T_FIT].sum(0)))
            oos = e2[T_FIT:].mean(0)
            g_best = int(np.argmin(oos))
            s2 = (x[:T_FIT] ** 2).mean()
            hl = matched_halflife(R_GRID, s2)
            rows.append(dict(world=name, sim=s, hl_ml=hl[g_ml], hl_sse=hl[g_sse], hl_best_oos=hl[g_best],
                             loss_ml_over_best=oos[g_ml] / oos[g_best], loss_sse_over_best=oos[g_sse] / oos[g_best]))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "mechanism_sims.csv", index=False, float_format="%.6g")
    summ = df.replace(np.inf, 1e9).groupby("world").agg(
        hl_ml_median=("hl_ml", "median"), hl_sse_median=("hl_sse", "median"),
        hl_best_oos_median=("hl_best_oos", "median"),
        share_ml_finite=("hl_ml", lambda v: float((v < 1e5).mean())),
        loss_ml_over_best=("loss_ml_over_best", "mean"), loss_sse_over_best=("loss_sse_over_best", "mean"))
    summ.to_csv(RES / "mechanism.csv", float_format="%.6g")
    return summ


if __name__ == "__main__":
    print(run().to_string())
