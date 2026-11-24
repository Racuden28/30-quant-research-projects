"""Part 0: DSR in a world we control. Gaussian trials with a one-factor correlation structure.

Trial i's Sharpe estimate is theta_i + (sqrt(rho) f + sqrt(1-rho) e_i) / sqrt(T), with f, e_i ~ N(0,1).
Under the null every theta_i = 0. In the 'signal-rich' alternative a share pi of the trials have
theta = mu and as many again have theta = -mu (a strategy and its mirror image), as in any zoo that
contains both a rule and its reverse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import dsr as D
from data import RES

T = 2520           # ten years of daily data
REPS = 20000
RHOS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]


def neff_equicorr(N, rho):
    lam = np.r_[1 + (N - 1) * rho, np.full(N - 1, 1 - rho)]
    return D.neff_nyholt(lam), D.neff_liji(lam), D.neff_participation(lam)


def simulate(N, rho, rng, theta=None):
    f = rng.standard_normal((REPS, 1))
    e = rng.standard_normal((REPS, N))
    sr = (np.sqrt(rho) * f + np.sqrt(1 - rho) * e) / np.sqrt(T)
    if theta is not None:
        sr = sr + theta
    return sr


def variants(sr_mat, N, rho):
    w = sr_mat.max(1)
    v_xs = sr_mat.var(1, ddof=1)
    ny, lj, pr = neff_equicorr(N, rho)
    out = {"PSR (N = 1)": D.psr(w, T)}
    out["DSR, raw N, cross-sectional V"] = D.dsr(w, T, 0, 3, N, v_xs)[0]
    out["DSR, raw N, null V = 1/T"] = D.dsr(w, T, 0, 3, N, 1 / T)[0]
    out["DSR, Li-Ji N, cross-sectional V"] = D.dsr(w, T, 0, 3, lj, v_xs)[0]
    out["DSR, Li-Ji N, null V"] = D.dsr(w, T, 0, 3, lj, 1 / T)[0]
    out["DSR, participation N, null V"] = D.dsr(w, T, 0, 3, pr, 1 / T)[0]
    out["DSR, Nyholt N, null V"] = D.dsr(w, T, 0, 3, ny, 1 / T)[0]
    p = D.psr(w, T)
    # Sidak-calibrated PSR: reject when PSR > 0.95 ** (1 / N); mapped to the same "> 0.95" scale
    for lab, n in (("raw", N), ("Li-Ji", lj), ("Nyholt", ny), ("participation", pr)):
        out[f"Sidak max-PSR, {lab} N"] = 0.95 + 0.05 * (p > 0.95 ** (1 / n)) - 0.0001 * (p <= 0.95 ** (1 / n))
    return out, dict(ny=ny, lj=lj, pr=pr, v_ratio=float(np.mean(v_xs) * T))


def main():
    rng = np.random.default_rng(23)
    rows = []
    for N in [100, 1000]:
        for rho in RHOS:
            sr = simulate(N, rho, rng)
            # exact benchmark: the 95th percentile of the max under the null, by simulation
            out, ne = variants(sr, N, rho)
            for k, v in out.items():
                rows.append(dict(world="all-zero", N=N, rho=rho, pi=0.0, mu_ann=0.0, method=k,
                                 reject=float(np.mean(v > 0.95)), **ne))
            # signal-rich alternative: 10% of trials at +mu, 10% at -mu (their mirror images)
            for mu_ann in [0.5, 1.0]:
                th = np.zeros(N)
                m = int(0.1 * N)
                th[:m] = mu_ann / np.sqrt(252)
                th[m:2 * m] = -mu_ann / np.sqrt(252)
                sra = simulate(N, rho, rng, th)
                out, ne = variants(sra, N, rho)
                for k, v in out.items():
                    rows.append(dict(world="signal-rich", N=N, rho=rho, pi=0.1, mu_ann=mu_ann, method=k,
                                     reject=float(np.mean(v > 0.95)), **ne))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "synthetic_size_power.csv", index=False, float_format="%.5g")
    piv = df[df.world == "all-zero"].pivot_table(index=["N", "method"], columns="rho", values="reject")
    print(piv.round(3).to_string())
    piv = df[df.world != "all-zero"].pivot_table(index=["N", "mu_ann", "method"], columns="rho", values="reject")
    print(piv.round(3).to_string())
    print(df.drop_duplicates(["N", "rho"])[["N", "rho", "ny", "lj", "pr"]].to_string())


if __name__ == "__main__":
    main()
