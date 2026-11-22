"""Is GARCH persistence a constant of nature or a property of the sample?

For each index: fit GARCH(1,1)-normal on the full sample and on each calendar decade (>= 5 years of
data). Test parameter constancy with an LR statistic, 2 * (sum of decade log-likelihoods - full-sample
log-likelihood), whose null distribution is built by a parametric bootstrap: simulate the full-sample
GARCH with i.i.d. resampled standardized residuals (keeps the fat tails), refit everything on the
same decade split, and recompute the statistic. The same simulations give the spread of decade
half-lives that estimation noise alone would produce, and the small-sample bias of short windows.

Output: results/decade_params.csv, results/stability_tests.csv
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import INDICES, ROOT, load_returns  # noqa: E402
from models import GARCH, half_life, simulate_garch  # noqa: E402

NSIM = 200
SEED = 1313
RES = ROOT / "results"


def decades(idx: pd.DatetimeIndex):
    dec = (idx.year // 10) * 10
    out = []
    for d in np.unique(dec):
        pos = np.where(dec == d)[0]
        if len(pos) >= 5 * 250:
            out.append((int(d), pos[0], pos[-1] + 1))
    return out


def fit_split(r, splits, x0=None, quick=False):
    pf, llf = GARCH.fit(r, x0, quick=quick)
    res = []
    for d, a, b in splits:
        p, ll = GARCH.fit(r[a:b], pf, quick=quick)
        res.append((d, p, ll))
    return pf, llf, res


def lr_stat(r, splits, pf, res):
    # log-likelihood of the decades' data at the full-sample parameters, same start rule as fit
    ll0 = sum(-GARCH.nll(pf, r[a:b], r[a:b] ** 2, float(np.mean(r[a:b][:252] ** 2))) for _, a, b in splits)
    return 2 * (sum(x[2] for x in res) - ll0)


def simulate(p, z, n, rng):
    return simulate_garch(p[0], p[1], p[2], rng.choice(z, size=n + 500, replace=True))


def run_index(code):
    r_s = load_returns()[code]
    r = r_s.values
    splits = decades(r_s.index)
    if len(splits) < 2:
        return None
    pf, llf, res = fit_split(r, splits)
    lr = lr_stat(r, splits, pf, res)
    s = GARCH.filter(r, pf, float(np.mean(r[:252] ** 2)))["s"]
    z = r / np.sqrt(np.r_[np.mean(r[:252] ** 2), s[:-1]])
    z = (z - z.mean()) / z.std()
    rng = np.random.default_rng(SEED + INDICES.index(code))
    sim_lr, sim_hl, sim_full_hl = [], [], []
    for _ in range(NSIM):
        rs = simulate(pf, z, len(r), rng)
        pfs, _, ress = fit_split(rs, splits, pf)  # same optimizer as the real data
        sim_lr.append(lr_stat(rs, splits, pfs, ress))
        sim_hl.append([float(half_life(GARCH.persistence(p))) for _, p, _ in ress])
        sim_full_hl.append(float(half_life(GARCH.persistence(pfs))))
    sim_hl = np.array(sim_hl)
    rows = []
    for j, (d, p, ll) in enumerate(res):
        lo, med, hi = np.quantile(np.minimum(sim_hl[:, j], 5000), [0.05, 0.5, 0.95])
        rows.append(dict(index=code, decade=d, n=splits[j][2] - splits[j][1], omega=p[0], alpha=p[1], beta=p[2],
                         persistence=GARCH.persistence(p), half_life=float(half_life(GARCH.persistence(p))),
                         long_run_vol=float(np.sqrt(252 * GARCH.long_run(p))) if GARCH.persistence(p) < 1 else np.inf,
                         null_hl_lo=lo, null_hl_med=med, null_hl_hi=hi))
    hl_full = float(half_life(GARCH.persistence(pf)))
    dec_hl = np.array([x["half_life"] for x in rows])
    test = dict(index=code, n_decades=len(splits), full_persistence=GARCH.persistence(pf), full_half_life=hl_full,
                median_decade_half_life=float(np.median(dec_hl)),
                null_median_decade_half_life=float(np.median(np.minimum(sim_hl, 5000))),
                null_full_half_life=float(np.median(sim_full_hl)),
                lr=lr, lr_p=(1 + np.sum(np.array(sim_lr) >= lr)) / (NSIM + 1),
                lr_df_naive=3 * (len(splits) - 1),
                share_decades_outside_null90=float(np.mean([(x["half_life"] < x["null_hl_lo"]) or
                                                            (x["half_life"] > x["null_hl_hi"]) for x in rows])))
    print(code, f"LR={lr:.1f} p={test['lr_p']:.3f} fullHL={hl_full:.0f} medDecHL={np.median(dec_hl):.0f}", flush=True)
    return pd.DataFrame(rows), test


def main():
    order = sorted(INDICES, key=lambda c: -len(load_returns()[c]))  # longest first for load balance
    with Pool(4) as pool:
        out = pool.map(run_index, order)
    out = [o for o in out if o is not None]
    pd.concat([o[0] for o in out]).to_csv(RES / "decade_params.csv", index=False, float_format="%.6g")
    pd.DataFrame([o[1] for o in out]).to_csv(RES / "stability_tests.csv", index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
