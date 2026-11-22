"""The fair test: compare the decay measured in the data with the decay the SAME measurement finds in
worlds where the model is true.

The local-projection half-life is a finite-sample estimator. With fat tails and high persistence it is
biased towards short half-lives (see sim_check.py: in a true GARCH world with a 69-day half-life it
reports about 37). So the right null is not the model's population half-life but the distribution of
the measured half-life in simulated model worlds.

For each index: fit GARCH(1,1) and component GARCH on the full sample; simulate NSIM paths of the same
length from each fit, with shocks resampled from that model's own standardized residuals (filtered
historical simulation, which keeps the fat tails and the skew); refit the model on each path; run the
in-sample local projection of r^2_{t+k} on the fitted variance; and compare with the same statistic on
the real data. Pooled curves combine the indices with index fixed effects, rep by rep.
Output: results/null_world_curves.csv, results/null_world_tests.csv
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import INDICES, ROOT, load_returns  # noqa: E402
from models import CGARCH, GARCH, curve_half_life, simulate_cgarch, simulate_garch, start_var  # noqa: E402

K = np.arange(1, 253)
NSIM = 100
SEED = 99
RES = ROOT / "results"
WORLDS = {"GARCH": GARCH, "CGARCH": CGARCH}


def lp_stats(r2, s):
    """(len(K), 5) sufficient statistics n, sx, sy, sxx, sxy of r2[t+k] on s[t], scaled by mean r2."""
    sc = r2.mean()
    r2, s = r2 / sc, s / sc
    n = len(r2)
    out = np.empty((len(K), 5))
    for j, k in enumerate(K):
        x, y = s[: n - k], r2[k:]
        out[j] = (n - k, x.sum(), y.sum(), x @ x, x @ y)
    return out


def slope(S):
    n, sx, sy, sxx, sxy = np.moveaxis(S, -1, 0)
    return (sxy - sx * sy / n) / (sxx - sx * sx / n)


def pooled_norm(stats_list):
    num = den = 0.0
    for S in stats_list:
        n, sx, sy, sxx, sxy = S.T
        num = num + (sxy - sx * sy / n)
        den = den + (sxx - sx * sx / n)
    b = num / den
    return b / b[0]


def fit_filter(M, r):
    p, _ = M.fit(r)
    return p, M.filter(r, p, start_var(r))["s"]


def run_index(code):
    r = load_returns()[code].values
    out = {"index": code, "data": {}, "sims": {}}
    rng = np.random.default_rng(SEED + INDICES.index(code))
    for wname, M in WORLDS.items():
        p, s = fit_filter(M, r)
        out["data"][wname] = lp_stats(r * r, s)
        z = r / np.sqrt(np.r_[start_var(r), s[:-1]])
        z = (z - z.mean()) / z.std()
        sims = []
        for _ in range(NSIM):
            zz = rng.choice(z, size=len(r) + 500, replace=True)
            rs = simulate_garch(*p, zz) if wname == "GARCH" else simulate_cgarch(*p, zz)
            ps, ss = fit_filter(M, rs)
            sims.append(lp_stats(rs * rs, ss))
        out["sims"][wname] = np.stack(sims)
        out[f"{wname}_params"] = p
    print(code, "done", flush=True)
    return out


def main():
    order = sorted(INDICES, key=lambda c: -len(load_returns()[c]))
    with Pool(4) as pool:
        res = pool.map(run_index, order)
    curves, tests = [], []
    for wname in WORLDS:
        units = [(o["index"], [o["data"][wname]], [o["sims"][wname][i] for i in range(NSIM)]) for o in res]
        units.append(("POOLED", [o["data"][wname] for o in res],
                      [[o["sims"][wname][i] for o in res] for i in range(NSIM)]))
        for name, dstats, sstats in units:
            if name == "POOLED":
                dn = pooled_norm(dstats)
                sn = np.stack([pooled_norm(x) for x in sstats])
            else:
                b = slope(dstats[0])
                dn = b / b[0]
                sb = np.stack([slope(x) for x in sstats])
                sn = sb / sb[:, :1]
            dhl = curve_half_life(dn, K)
            shl = np.array([curve_half_life(x, K) for x in sn])
            curves.append(pd.DataFrame(dict(world=wname, index=name, k=K, data=dn,
                                            null_med=np.median(sn, 0), null_lo=np.quantile(sn, 0.05, 0),
                                            null_hi=np.quantile(sn, 0.95, 0))))
            # area between curves over k = 1..126 (a shape statistic that does not hinge on one crossing)
            area_d = dn[:126].mean()
            area_s = sn[:, :126].mean(1)
            tests.append(dict(world=wname, index=name, data_half_life=dhl,
                              null_half_life_median=np.median(shl), null_hl_p05=np.quantile(shl, 0.05),
                              null_hl_p95=np.quantile(shl, 0.95),
                              p_hl=(1 + np.sum(shl <= dhl)) / (NSIM + 1),
                              data_mean_share_1_126=area_d, null_mean_share_1_126=np.median(area_s),
                              p_area=(1 + np.sum(area_s <= area_d)) / (NSIM + 1),
                              data_share_at_252=dn[-1], null_share_at_252=np.median(sn[:, -1]),
                              p_tail_high=(1 + np.sum(sn[:, -1] >= dn[-1])) / (NSIM + 1)))
    pd.concat(curves).to_csv(RES / "null_world_curves.csv", index=False, float_format="%.5g")
    t = pd.DataFrame(tests)
    t.to_csv(RES / "null_world_tests.csv", index=False, float_format="%.5g")
    pd.DataFrame([dict(index=o["index"], **{f"G_{k}": v for k, v in zip(["omega", "alpha", "beta"], o["GARCH_params"])},
                       **{f"CG_{k}": v for k, v in zip(["omega", "rho", "phi", "alpha", "beta"], o["CGARCH_params"])})
                  for o in res]).to_csv(RES / "full_sample_params.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 220)
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
