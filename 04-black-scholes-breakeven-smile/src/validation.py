"""Does the break-even estimator recover a known answer? Simulation study.

1. Geometric Brownian motion with constant 20% vol: the true break-even smile is flat at m* = 1
   (sigma_ref is estimated from the simulated trailing window, so m* = 1 up to estimation noise).
2. Merton jump-diffusion with negative mean jumps: the estimator must produce a negative skew.
3. GARCH(1,1) with leverage (GJR) and no jumps: vol clustering plus spot-vol correlation alone
   should also generate a skew and convexity.
Writes results/validation.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from breakeven import RES, X_GRID, Spec, build_paths, pnl_grid, root_on_grid

ANN = 252


def sim_gbm(n, sigma, rng):
    return rng.normal(-0.5 * sigma ** 2 / ANN, sigma / np.sqrt(ANN), n)


def sim_merton(n, sigma, lam, mu_j, sd_j, rng):
    dt = 1 / ANN
    nj = rng.poisson(lam * dt, n)
    jumps = rng.normal(mu_j * nj, sd_j * np.sqrt(nj))
    return rng.normal(-0.5 * sigma ** 2 * dt, sigma * np.sqrt(dt), n) + jumps


def sim_gjr(n, rng, omega=2e-6, alpha=0.02, gamma=0.15, beta=0.89):
    r = np.empty(n)
    h = omega / (1 - alpha - gamma / 2 - beta)
    for t in range(n):
        z = rng.standard_normal()
        r[t] = np.sqrt(h) * z
        h = omega + (alpha + gamma * (r[t] < 0)) * r[t] ** 2 + beta * h
    return r


def curve(logret):
    s = pd.Series(logret, index=pd.bdate_range("2000-01-03", periods=len(logret)))
    _, sig, logS = build_paths(s, Spec(), start="2000-01-01")
    return root_on_grid(pnl_grid(sig, logS, Spec()).mean(0))


def main(n=20000, seed=7):
    rng = np.random.default_rng(seed)
    out = {"GBM 20% (truth: flat at 1)": curve(sim_gbm(n, 0.20, rng)),
           "Merton, jumps -5% x 3/yr": curve(sim_merton(n, 0.15, 3.0, -0.05, 0.03, rng)),
           "GJR-GARCH with leverage": curve(sim_gjr(n, rng))}
    df = pd.DataFrame(out, index=X_GRID)
    df.index.name = "x"
    df.to_csv(RES / "validation.csv")
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
