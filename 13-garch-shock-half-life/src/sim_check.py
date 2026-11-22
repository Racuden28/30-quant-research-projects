"""Placebo for the method: if returns really were GARCH, would the local projection find the GARCH half-life?

Simulate GARCH(1,1) with fat-tailed shocks (standardized t with 5 d.o.f.) at three persistence levels,
the sample length of the S&P 500 (25,000 days). Fit GARCH by QMLE on each path, then run the same
local projection as irf.py on the fitted variance. If the estimator is sound, realized and implied
half-lives agree. A second design simulates a two-component world (fast + slow), which is what the
data look like. Output: results/simulation_check.csv
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT  # noqa: E402
from models import CGARCH, GARCH, curve_half_life, half_life, simulate_garch, start_var  # noqa: E402

K = np.arange(1, 253)
N = 25000
REPS = 40


def lp_norm(r2, s):
    n = len(r2)
    b = []
    for k in K:
        x, y = s[: n - k], r2[k:]
        xc = x - x.mean()
        b.append(xc @ (y - y.mean()) / (xc @ xc))
    b = np.array(b)
    return b / b[0]


def tshock(rng, n, nu=5):
    z = rng.standard_t(nu, n)
    return z / np.sqrt(nu / (nu - 2))


def one(args):
    design, phi, rep = args
    rng = np.random.default_rng(10_000 * int(phi * 1000) + rep + (0 if design == "GARCH" else 7))
    if design == "GARCH":
        a = 0.08
        b = phi - a
        w = 1.0 * (1 - phi)
        r = simulate_garch(w, a, b, tshock(rng, N + 500))
        true_hl = float(half_life(phi))
    else:
        # two components: transitory (alpha+beta = 0.95, HL ~13.5 days) and permanent-ish rho = phi
        p = np.array([1.0, phi, 0.02, 0.07, 0.88])
        z = tshock(rng, N + 500)
        r = np.empty(N + 500)
        s = q = 1.0
        for t in range(N + 500):
            r[t] = np.sqrt(s) * z[t]
            qn = p[0] + p[1] * (q - p[0]) + p[2] * (r[t] ** 2 - s)
            s = max(qn + p[3] * (r[t] ** 2 - q) + p[4] * (s - q), 1e-6)
            q = qn
        r = r[500:]
        true_hl = curve_half_life(CGARCH.decay(p, K), K)
    pg, _ = GARCH.fit(r)
    sg = GARCH.filter(r, pg, start_var(r))["s"]
    emp = curve_half_life(lp_norm(r * r, sg), K)
    return dict(design=design, phi=phi, rep=rep, true_half_life=true_hl,
                garch_fitted_half_life=float(half_life(GARCH.persistence(pg))), lp_half_life=emp)


def main():
    jobs = [(d, phi, i) for d in ("GARCH", "Two-component") for phi in (0.98, 0.99, 0.995) for i in range(REPS)]
    with Pool(4) as pool:
        rows = pool.map(one, jobs)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "simulation_check_draws.csv", index=False, float_format="%.5g")
    summ = df.replace(np.inf, np.nan).groupby(["design", "phi"]).agg(
        true_half_life=("true_half_life", "median"), garch_fitted=("garch_fitted_half_life", "median"),
        lp_median=("lp_half_life", "median"), lp_p05=("lp_half_life", lambda x: x.quantile(0.05)),
        lp_p95=("lp_half_life", lambda x: x.quantile(0.95))).reset_index()
    summ.to_csv(ROOT / "results" / "simulation_check.csv", index=False, float_format="%.4g")
    print(summ.round(1).to_string())


if __name__ == "__main__":
    main()
