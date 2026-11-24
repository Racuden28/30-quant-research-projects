"""Aggregate results/runs/*.pkl into the tables used in the README.

Writes results/*.csv. Every number quoted in the README comes from one of these files.
"""
from __future__ import annotations
import os, sys, glob, pickle
import numpy as np
import pandas as pd
from scipy import stats as sst

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats as st
from experiments import UNIV, OUT as RUNS

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SIG_NAMES = {"ou": "s-score + filter (Avellaneda-Lee)", "ou_nf": "s-score, no filter", "res5": "5-day residual z"}

STRESS = {
    "equity": [("Quant quake Aug 2007", "2007-07-25", "2007-08-15"),
               ("GFC Sep 2008-Mar 2009", "2008-09-01", "2009-03-31"),
               ("COVID Feb-Apr 2020", "2020-02-19", "2020-04-30"),
               ("2022 bear market", "2022-01-03", "2022-10-12"),
               ("SVB Mar 2023", "2023-03-01", "2023-03-31")],
    "crypto": [("COVID crash Mar 2020", "2020-03-01", "2020-04-15"),
               ("May 2021 crash", "2021-05-10", "2021-06-30"),
               ("LUNA collapse May-Jun 2022", "2022-05-05", "2022-06-30"),
               ("FTX collapse Nov 2022", "2022-11-06", "2022-11-30"),
               ("Oct 2025 liquidation cascade", "2025-10-08", "2025-10-20")],
}
ERAS = {
    "equity": [("2001-2007 (pre-publication)", "2001-01-01", "2007-12-31"),
               ("2008-2009 (crisis)", "2008-01-01", "2009-12-31"),
               ("2010-2019", "2010-01-01", "2019-12-31"),
               ("2020-2026", "2020-01-01", "2026-12-31")],
    "crypto": [(str(y), f"{y}-01-01", f"{y}-12-31") for y in range(2019, 2027)],
}


def load_runs(path=RUNS):
    runs = {}
    for f in sorted(glob.glob(os.path.join(path, "*.pkl"))):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        runs[(r["universe"], r["label"])] = r
    return runs


def common_start(runs, universe):
    """First date every run of this universe trades, so all comparisons share one sample."""
    starts = [r["pnl"]["ou"].index[0] for (u, _), r in runs.items() if u == universe]
    return max(starts)


def metrics(p: pd.DataFrame, u: str, delay: int, cost=None, borrow=None):
    P = UNIV[u]["periods"]
    cost = UNIV[u]["cost"] if cost is None else cost
    borrow = UNIV[u]["borrow"] if borrow is None else borrow
    g = p[f"gross_d{delay}"].to_numpy(float)
    turn = p[f"turn_d{delay}"].to_numpy(float)
    net = g - turn * cost * 1e-4 - p["short"].to_numpy(float) * borrow / P
    be = g.mean() / turn.mean() * 1e4 if turn.mean() > 0 else np.nan
    return dict(ann_gross=g.mean() * P, vol=g.std(ddof=1) * np.sqrt(P), sr_gross=st.sharpe(g, P),
                t_hac=st.hac_t(g), turnover_day=turn.mean(), breakeven_bps=be,
                ann_net=net.mean() * P, sr_net=st.sharpe(net, P),
                borrow_drag=(p["short"].to_numpy(float) * borrow).mean(), gexp=p["gexp"].mean())


def main():
    runs = load_runs()
    os.makedirs(RES, exist_ok=True)
    out_main, out_abl, out_filter, out_ic, out_era, out_stress, out_cost = [], [], [], [], [], [], []
    for u in ("equity", "crypto"):
        t0 = common_start(runs, u)
        P = UNIV[u]["periods"]
        plac = [r for (uu, l), r in runs.items() if uu == u and l.startswith("placebo")]
        real = runs[(u, "main")]
        # ---- main table: real vs placebo, by signal and delay
        for s in ("ou", "ou_nf", "res5"):
            for d in (0, 1):
                m = metrics(real["pnl"][s].loc[t0:], u, d)
                lo, hi = st.sharpe_ci(real["pnl"][s].loc[t0:, f"gross_d{d}"].to_numpy(float), P)
                psr = np.array([metrics(r["pnl"][s].loc[t0:], u, d)["sr_gross"] for r in plac])
                pbe = np.array([metrics(r["pnl"][s].loc[t0:], u, d)["sr_net"] for r in plac])
                z = (m["sr_gross"] - psr.mean()) / psr.std(ddof=1)
                out_main.append(dict(universe=u, signal=s, delay=d, **m, sr_ci_lo=lo, sr_ci_hi=hi,
                                     placebo_sr_mean=psr.mean(), placebo_sr_sd=psr.std(ddof=1),
                                     placebo_sr_max=psr.max(), placebo_net_sr_mean=pbe.mean(),
                                     placebo_rank=int((psr >= m["sr_gross"]).sum()), n_placebo=len(psr),
                                     z_vs_placebo=z, p_vs_placebo=float(sst.norm.sf(z)),
                                     p_vs_zero=float(sst.norm.sf(m["t_hac"])),
                                     start=str(t0.date()), end=str(real["pnl"][s].index[-1].date())))
        # ---- filter: pass share and median mean-reversion time, real vs placebo
        def fstats(r):
            dg = r["diag"].loc[t0:]; dg = dg[dg.n_assets > 0]
            oc = r["ou_cross"]; oc = oc[oc.date >= t0]
            tau = 1 / oc.kappa
            return dict(pass_share=dg.pass_filter.mean(), median_tau=np.nanmedian(tau),
                        tau_lt_10=np.nanmean(tau < 10), mean_k=dg.k.mean(), mean_assets=dg.n_assets.mean(),
                        mean_open=dg.open_ou.mean())
        fr = fstats(real); fp = pd.DataFrame([fstats(r) for r in plac])
        out_filter.append(dict(universe=u, world="real", **fr))
        out_filter.append(dict(universe=u, world="placebo (mean of %d)" % len(plac), **fp.mean().to_dict()))
        out_filter.append(dict(universe=u, world="placebo (min-max)",
                               pass_share=f"{fp.pass_share.min():.3f}-{fp.pass_share.max():.3f}",
                               median_tau=f"{fp.median_tau.min():.2f}-{fp.median_tau.max():.2f}"))
        # ---- information coefficient term structure
        for name in ("ou", "res5", "res1"):
            for h in (1, 2, 3, 5, 10, 20):
                col = f"ic_{name}_h{h}"
                x = real["ic"].loc[t0:, col].dropna().to_numpy()
                pm = [r["ic"].loc[t0:, col].dropna().mean() for r in plac]
                out_ic.append(dict(universe=u, signal=name, horizon=h, ic_mean=x.mean(), t_hac=st.hac_t(x, lags=h + 5),
                                   placebo_ic_mean=np.mean(pm), placebo_ic_sd=np.std(pm, ddof=1)))
        # ---- ablations (all on the common sample)
        for (uu, l), r in sorted(runs.items()):
            if uu != u or l.startswith("placebo"):
                continue
            for s in ("ou", "res5"):
                for d in (0, 1):
                    m = metrics(r["pnl"][s].loc[t0:], u, d)
                    out_abl.append(dict(universe=u, spec=l, signal=s, delay=d, mean_k=r["diag"].loc[t0:].k.mean(),
                                        p_vs_zero=float(sst.norm.sf(m["t_hac"])), **m))
        # ---- eras and stress windows (main spec, s-score, both delays) with placebo percentile
        for lab, a, b in ERAS[u]:
            for s in ("ou", "res5"):
                for d in (0, 1):
                    x = real["pnl"][s].loc[max(pd.Timestamp(a), t0):b]
                    if len(x) < 60:
                        continue
                    m = metrics(x, u, d)
                    out_era.append(dict(universe=u, era=lab, signal=s, delay=d, days=len(x),
                                        ci=st.sharpe_ci(x[f"gross_d{d}"].to_numpy(float), P), **m))
        for lab, a, b in STRESS[u]:
            if pd.Timestamp(b) < t0:
                continue
            for s in ("ou", "res5"):
                for d in (0, 1):
                    g = real["pnl"][s].loc[a:b, f"gross_d{d}"]
                    pg = np.array([r["pnl"][s].loc[a:b, f"gross_d{d}"].sum() for r in plac])
                    full = real["pnl"][s].loc[t0:, f"gross_d{d}"]
                    out_stress.append(dict(universe=u, window=lab, start=a, end=b, signal=s, delay=d, days=len(g),
                                           cum_gross=g.sum(), worst_day=g.min(),
                                           cum_in_sd=g.sum() / (full.std() * np.sqrt(len(g))),
                                           placebo_cum_mean=pg.mean(), placebo_cum_min=pg.min(), placebo_cum_max=pg.max()))
        # ---- cost sweep for the main spec
        for s in ("ou", "res5"):
            for d in (0, 1):
                for c in (0, 1, 2, 3, 5, 10, 20):
                    m = metrics(real["pnl"][s].loc[t0:], u, d, cost=c, borrow=0.0)
                    out_cost.append(dict(universe=u, signal=s, delay=d, cost_bps=c, sr_net=m["sr_net"], ann_net=m["ann_net"]))
    main_df = pd.DataFrame(out_main)
    main_df["p_holm_vs_placebo"] = st.holm(main_df.p_vs_placebo)
    main_df["p_holm_vs_zero"] = st.holm(main_df.p_vs_zero)
    abl = pd.DataFrame(out_abl)
    abl["p_holm"] = st.holm(abl.p_vs_zero)
    era = pd.DataFrame(out_era)
    era[["sr_ci_lo", "sr_ci_hi"]] = pd.DataFrame(era.pop("ci").tolist(), index=era.index)
    main_df.to_csv(os.path.join(RES, "main_real_vs_placebo.csv"), index=False)
    pd.DataFrame(out_filter).to_csv(os.path.join(RES, "filter_real_vs_placebo.csv"), index=False)
    pd.DataFrame(out_ic).to_csv(os.path.join(RES, "ic_term_structure.csv"), index=False)
    abl.to_csv(os.path.join(RES, "ablations.csv"), index=False)
    era.to_csv(os.path.join(RES, "eras.csv"), index=False)
    pd.DataFrame(out_stress).to_csv(os.path.join(RES, "stress_windows.csv"), index=False)
    pd.DataFrame(out_cost).to_csv(os.path.join(RES, "cost_sweep.csv"), index=False)
    # ---- crypto loss decomposition: contribution by asset of the main s-score book
    ba = runs[("crypto", "main")]["by_asset"].sort_values()
    ba.rename("cum_contribution").to_csv(os.path.join(RES, "crypto_contribution_by_asset.csv"))
    placebo_versions(runs)
    pd.set_option("display.width", 250)
    print(main_df[["universe", "signal", "delay", "ann_gross", "sr_gross", "sr_ci_lo", "sr_ci_hi", "breakeven_bps",
                   "sr_net", "placebo_sr_mean", "placebo_sr_sd", "placebo_rank", "z_vs_placebo", "p_holm_vs_placebo"]].round(3).to_string())
    print(pd.DataFrame(out_filter).to_string())


def placebo_versions(runs):
    """The three placebo designs tried, side by side (README section 7). The rejected ones are
    rebuilt by `python src/experiments.py --placebo permute|bootstrap` into results/runs_<method>/."""
    rows = []
    versions = [("v1 permute days within quarter", os.path.join(RES, "runs_permute")),
                ("v2 bootstrap days within quarter", os.path.join(RES, "runs_bootstrap")),
                ("v3 random sign per day (used)", RUNS)]
    for name, path in versions:
        pr = load_runs(path)
        for u in ("equity", "crypto"):
            t0 = common_start(runs, u)
            plac = [r for (uu, l), r in pr.items() if uu == u and l.startswith("placebo")]
            if not plac:
                continue
            for d in (0, 1):
                sr = np.array([metrics(r["pnl"]["ou"].loc[t0:], u, d)["sr_gross"] for r in plac])
                ic1 = np.mean([r["ic"].loc[t0:, "ic_ou_h1"].mean() for r in plac])
                ic5 = np.mean([r["ic"].loc[t0:, "ic_ou_h5"].mean() for r in plac])
                rows.append(dict(version=name, universe=u, delay=d, n=len(plac), placebo_sr_mean=sr.mean(),
                                 placebo_sr_sd=sr.std(ddof=1), placebo_ic_h1=ic1, placebo_ic_h5=ic5))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "placebo_versions.csv"), index=False)


if __name__ == "__main__":
    main()
