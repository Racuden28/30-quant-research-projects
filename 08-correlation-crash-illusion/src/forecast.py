"""
Part B. Which heatmap should you use for next month?

Every 21 trading days (after a 5-year burn-in) each estimator forecasts the correlation of the next 21
devolatised returns. All estimators share the same causal EWMA volatility, so only the correlation
differs. Two out-of-sample losses:

  * GMV: realised variance of the global minimum-variance portfolio built from D R_hat D
         (D = ex-ante EWMA vols). The economically relevant loss: a correlation error you can trade on.
  * LOGSCORE: Gaussian negative log-likelihood of the next 21 devolatised return vectors under R_hat
         (proper scoring rule for the correlation matrix; lower is better).

Diebold-Mariano tests vs the 1-year sample heatmap (the default "correlation heatmap"), Newey-West HAC,
Holm-adjusted across estimators. DCC(1,1) is re-estimated every 252 trading days on an expanding window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core import (corr, dcc_filter, dcc_fit, dcc_forecast_avg, ewma_corr_path, ewma_vol, holm,
                  lw_constant_corr, nearest_corr, newey_west_t, q_to_r, rmt_clip)
from data import load

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
H = 21
BURN = 1260
WARM = 63
REFIT = 252
BASELINE = "Sample 1y"
EST_ORDER = ["Sample 3m", "Sample 1y", "Sample 5y", "Expanding", "EWMA 0.97", "Shrink 1y (LW-CC)",
             "RMT clip 1y", "DCC(1,1)", "Identity"]


def run(universe: str):
    df = load(universe)
    r_all = df.values
    sig_all = ewma_vol(r_all)
    r, sig, dates = r_all[WARM:], sig_all[WARM:], df.index[WARM:]
    z = r / sig
    T, N = z.shape
    ew = ewma_corr_path(z, 0.97)
    origins = list(range(BURN, T - H + 1, H))
    dcc_par, dcc_rows = None, []
    rows = []
    for k, t in enumerate(origins):
        if dcc_par is None or (t - BURN) % REFIT < H:
            (a, b), _ = dcc_fit(z[:t])
            dcc_par = (a, b)
            dcc_rows.append(dict(universe=universe, date=str(dates[t].date()), a=a, b=b))
        a, b = dcc_par
        Qbar = np.cov(z[:t], rowvar=False)
        Q = dcc_filter(z[:t], a, b, Qbar)
        est = {
            "Sample 3m": corr(z[t - 63:t]),
            "Sample 1y": corr(z[t - 252:t]),
            "Sample 5y": corr(z[t - 1260:t]),
            "Expanding": corr(z[:t]),
            "EWMA 0.97": ew[t],
            "Shrink 1y (LW-CC)": lw_constant_corr(z[t - 252:t]),
            "RMT clip 1y": rmt_clip(corr(z[t - 252:t]), 252),
            "DCC(1,1)": dcc_forecast_avg(Q[-1], Qbar, a, b, H),
            "Identity": np.eye(N),
        }
        zf, rf = z[t:t + H], r[t:t + H]
        D = sig[t]
        realised = corr(zf)
        for name, R in est.items():
            R = nearest_corr(R)
            S = R * np.outer(D, D)
            w = np.linalg.solve(S, np.ones(N))
            w /= w.sum()
            rp = rf @ w
            sign, logdet = np.linalg.slogdet(R)
            Ri = np.linalg.inv(R)
            ls = float(np.mean(logdet + np.einsum("ti,ij,tj->t", zf, Ri, zf)))
            rows.append(dict(universe=universe, origin=dates[t], estimator=name, gmv_var=float(np.mean(rp ** 2)),
                             logscore=ls, frob=float(np.linalg.norm(R - realised)),
                             mean_corr=float(R[np.triu_indices(N, 1)].mean()),
                             gross_leverage=float(np.abs(w).sum()),
                             mvol=float(np.mean(D))))
    return pd.DataFrame(rows), pd.DataFrame(dcc_rows), pd.DataFrame(
        dict(date=dates[BURN:], dcc_mean_corr=_dcc_path_mean(z, dcc_rows, dates)))


def _dcc_path_mean(z, dcc_rows, dates):
    """Full-sample-parameter DCC path of the average correlation (descriptive figure only)."""
    (a, b), Qbar = dcc_fit(z)
    R = q_to_r(dcc_filter(z, a, b, Qbar)[1:])
    N = z.shape[1]
    iu = np.triu_indices(N, 1)
    return np.tanh(np.arctanh(np.clip(R[:, iu[0], iu[1]], -.999, .999)).mean(1))[BURN:]


def summarise(f: pd.DataFrame) -> pd.DataFrame:
    out = []
    for u, g in f.groupby("universe"):
        piv = {m: g.pivot(index="origin", columns="estimator", values=m) for m in ["gmv_var", "logscore", "frob"]}
        mv = g[g.estimator == BASELINE].set_index("origin").mvol
        stress = mv >= mv.quantile(0.8)
        for subset, sel in [("all", slice(None)), ("stressed (top-20% vol origins)", stress[stress].index)]:
            pv = []
            tmp = []
            for e in EST_ORDER:
                gv = piv["gmv_var"].loc[sel]
                ls = piv["logscore"].loc[sel]
                d_g = (gv[e] - gv[BASELINE]).values
                d_l = (ls[e] - ls[BASELINE]).values
                tg, pg = newey_west_t(d_g) if e != BASELINE else (0.0, 1.0)
                tl, pl = newey_west_t(d_l) if e != BASELINE else (0.0, 1.0)
                tmp.append(dict(universe=u, subset=subset, estimator=e, n=len(gv),
                                gmv_vol_ann=float(np.sqrt(gv[e].mean() * 252)),
                                gmv_rel=float(np.sqrt(gv[e].mean() / gv[BASELINE].mean()) - 1),
                                logscore=float(ls[e].mean()), logscore_diff=float(d_l.mean()),
                                frob=float(piv["frob"].loc[sel][e].mean()),
                                dm_t_gmv=tg, p_gmv=pg, dm_t_logscore=tl, p_logscore=pl))
            ests = [x for x in tmp if x["estimator"] != BASELINE]
            hg = holm(np.array([x["p_gmv"] for x in ests]))
            hl = holm(np.array([x["p_logscore"] for x in ests]))
            for x, a, b in zip(ests, hg, hl):
                x["p_gmv_holm"], x["p_logscore_holm"] = a, b
            for x in tmp:
                x.setdefault("p_gmv_holm", np.nan)
                x.setdefault("p_logscore_holm", np.nan)
            out += tmp
    return pd.DataFrame(out)


def main():
    fs, ds, ps = [], [], []
    for u in ["stocks", "etfs"]:
        f, d, p = run(u)
        p["universe"] = u
        fs.append(f), ds.append(d), ps.append(p)
        print(u, "origins", f.origin.nunique())
    f = pd.concat(fs)
    f.to_csv(RES / "forecast_losses.csv.gz", index=False, float_format="%.6g")
    pd.concat(ds).to_csv(RES / "dcc_parameters.csv", index=False, float_format="%.5f")
    pd.concat(ps).to_csv(RES / "dcc_mean_corr_path.csv", index=False, float_format="%.4f")
    s = summarise(f)
    s.to_csv(RES / "forecast_summary.csv", index=False, float_format="%.5f")
    print(s[["universe", "subset", "estimator", "gmv_vol_ann", "gmv_rel", "logscore_diff", "p_gmv_holm",
             "p_logscore_holm"]].round(4).to_string())


if __name__ == "__main__":
    main()
