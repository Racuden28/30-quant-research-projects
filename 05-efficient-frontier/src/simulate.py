"""Stationary counterfactuals.

1. Walk-forward on simulated i.i.d. Gaussian paths whose true moments equal the full-sample moments of
   each universe. Same windows, same code, same strategies. Whatever gap between promised and realized
   Sharpe survives here is pure estimation error; the rest of the real-data gap is non-stationarity.
2. "How many years of data would you need?": exact sampling distribution of (mean, covariance) for a
   Gaussian sample of W years, expected true Sharpe of the estimated portfolios.
"""
from __future__ import annotations

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import portfolio as pf
from backtest import run
from config import N_SIM_PATHS, SEED, UNIVERSES
from data import load

RES = Path(__file__).resolve().parents[1] / "results"


def _one_path(args):
    u, p = args
    df = load(u)
    mu, cov = df.mean().to_numpy(), df.cov().to_numpy()
    rng = np.random.default_rng([SEED, ord(u), p])
    x = rng.multivariate_normal(mu, cov, size=len(df))
    sim = pd.DataFrame(x, index=df.index, columns=df.columns)
    out = run(u, x=sim, draws=0, cells=True, seed=SEED + p)
    ppy = UNIVERSES[u]["ppy"]
    g = out["gross"]
    row = {k: g[k].dropna().mean() / g[k].dropna().std() * np.sqrt(ppy) for k in g.columns}
    row["promised"] = out["promised"].mean() * np.sqrt(ppy)
    row["path"] = p
    return u, row


def counterfactual(n_paths=N_SIM_PATHS, procs=3):
    jobs = [(u, p) for u in UNIVERSES for p in range(n_paths if u != "C" else n_paths // 2)]
    with Pool(procs) as pool:
        res = pool.map(_one_path, jobs, chunksize=4)
    for u in UNIVERSES:
        pd.DataFrame([r for uu, r in res if uu == u]).to_csv(RES / f"{u}_sim_paths.csv", index=False)


def years_needed(k_draws=2000):
    rows = []
    years = [1, 2, 3, 5, 10, 20, 40, 80, 160, 320, 640, 1280]
    for u, cfg in UNIVERSES.items():
        df = load(u)
        mu, cov = df.mean().to_numpy(), df.cov().to_numpy()
        n, ppy = len(mu), cfg["ppy"]
        ann = np.sqrt(ppy)
        best = pf.sharpe(mu, cov, pf.max_sharpe(mu, cov)) * ann
        ew = pf.sharpe(mu, cov, pf.equal_weight(n)) * ann
        mv = pf.sharpe(mu, cov, pf.min_variance(cov)) * ann
        rng = np.random.default_rng([SEED, ord(u), 99])
        chol = np.linalg.cholesky(cov)
        for y in years:
            t = int(y * ppy)
            if t <= n + 3:
                continue
            covs = stats.wishart(df=t - 1, scale=cov / (t - 1)).rvs(size=k_draws, random_state=rng)
            acc = {"Sample mean, sample cov": [], "Sample mean, true cov": [], "Bayes-Stein mean, sample cov": [],
                   "True mean, sample cov": [], "No mean (min-var), sample cov": []}
            for k in range(k_draws):
                m = mu + chol @ rng.standard_normal(n) / np.sqrt(t)
                s = covs[k]
                si = np.linalg.inv(s * (t - 1) / max(t - n - 2, 1))
                one = np.ones(n)
                m0 = one @ si @ m / (one @ si @ one)
                d = m - m0
                phi = (n + 2) / ((n + 2) + t * d @ si @ d)
                m_bs = (1 - phi) * m + phi * m0
                for key, w in [("Sample mean, sample cov", pf.max_sharpe(m, s)),
                               ("Sample mean, true cov", pf.max_sharpe(m, cov)),
                               ("Bayes-Stein mean, sample cov", pf.max_sharpe(m_bs, s)),
                               ("True mean, sample cov", pf.max_sharpe(mu, s)),
                               ("No mean (min-var), sample cov", pf.min_variance(s))]:
                    acc[key].append(pf.sharpe(mu, cov, w) * ann)
            for key, v in acc.items():
                v = np.asarray(v)
                rows.append(dict(universe=u, years=y, estimator=key, mean_sharpe=v.mean(),
                                 p10=np.quantile(v, 0.1), p90=np.quantile(v, 0.9),
                                 prob_beats_1N=(v > ew).mean(), best=best, one_over_n=ew, min_var_true=mv))
            print(u, y, "done", flush=True)
    pd.DataFrame(rows).to_csv(RES / "years_needed.csv", index=False)


if __name__ == "__main__":
    t0 = time.time()
    years_needed()
    print(f"years_needed {time.time() - t0:.0f}s", flush=True)
    counterfactual()
    print(f"counterfactual {time.time() - t0:.0f}s")
