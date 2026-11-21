"""N/T sweep: where does the frontier break as the ratio of assets to observations grows?

For each universe and estimation window W, rerun the walk-forward factorial (no resampling) and record
how much of the value of perfect information comes from the means vs. the covariance, and what
covariance shrinkage (Ledoit-Wolf) is worth, for the max-Sharpe and the minimum-variance portfolio.
"""
from __future__ import annotations

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import run
from config import STRATEGIES, SWEEP, UNIVERSES
from evaluate import boot_sharpes, sharpe

RES = Path(__file__).resolve().parents[1] / "results"


def one(args):
    u, W = args
    cfg = UNIVERSES[u]
    out = run(u, window=W, draws=0)
    n = out["gross"].shape[1] and len([c for c in out["weights"].columns if c not in ("date", "strategy")])
    ppy, block = cfg["ppy"], cfg["block"]
    g = out["gross"]
    cells = g[[c for c in g.columns if c.startswith("cell|")]].dropna()
    hr = out["net"][[s for s in STRATEGIES if s in out["net"].columns]].dropna()
    bs = boot_sharpes(pd.concat([cells, hr.add_prefix("hr|")], axis=1).dropna(), ppy, block, reps=1000)
    pt = pd.Series(sharpe(pd.concat([cells, hr.add_prefix("hr|")], axis=1).dropna().to_numpy(), ppy),
                   index=bs.columns)

    def c(m, v):
        return f"cell|{m}|{v}"

    def shap(s):
        pm = 0.5 * ((s[c("Clairvoyant", "Sample")] - s[c("Sample", "Sample")])
                    + (s[c("Clairvoyant", "Clairvoyant")] - s[c("Sample", "Clairvoyant")]))
        pc = 0.5 * ((s[c("Sample", "Clairvoyant")] - s[c("Sample", "Sample")])
                    + (s[c("Clairvoyant", "Clairvoyant")] - s[c("Clairvoyant", "Sample")]))
        return pm, pc

    row = dict(universe=u, N=n, window=W, n_over_t=n / W, periods=len(cells))
    pm, pc = shap(pt)
    bm, bc = shap(bs)
    row.update(phi_mean=pm, phi_cov=pc, phi_cov_lo=np.quantile(bc, .025), phi_cov_hi=np.quantile(bc, .975),
               phi_mean_lo=np.quantile(bm, .025), phi_mean_hi=np.quantile(bm, .975))
    for name, hi, lo in [("lw_given_sample_mean", c("Sample", "Ledoit-Wolf"), c("Sample", "Sample")),
                         ("lw_given_no_mean", c("None", "Ledoit-Wolf"), c("None", "Sample")),
                         ("clairv_cov_given_sample_mean", c("Sample", "Clairvoyant"), c("Sample", "Sample")),
                         ("clairv_cov_given_no_mean", c("None", "Clairvoyant"), c("None", "Sample")),
                         ("no_mean_minus_sample_mean", c("None", "Sample"), c("Sample", "Sample")),
                         ("msr_sample_minus_1N_net", "hr|Max-Sharpe (sample)", "hr|1/N"),
                         ("minvar_lw_minus_1N_net", "hr|Min-var (LW)", "hr|1/N")]:
        d = bs[hi] - bs[lo]
        row[name] = pt[hi] - pt[lo]
        row[name + "_lo"], row[name + "_hi"] = np.quantile(d, [0.025, 0.975])
    row["markowitz_sharpe"] = pt[c("Sample", "Sample")]
    return row


def main():
    t0 = time.time()
    jobs = [(u, W) for u in ["A", "C", "B"] for W in SWEEP[u]]
    with Pool(3) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    df = pd.DataFrame(rows).sort_values(["universe", "n_over_t"])
    df.to_csv(RES / "nt_sweep.csv", index=False, float_format="%.4f")
    print(df[["universe", "N", "window", "n_over_t", "phi_mean", "phi_cov", "lw_given_sample_mean",
              "lw_given_no_mean", "clairv_cov_given_no_mean", "no_mean_minus_sample_mean"]].round(3).to_string())
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
