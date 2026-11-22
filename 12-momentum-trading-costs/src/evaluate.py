"""
Every table in the README, from results/runs.csv.gz.

Outputs (results/):
  summary.csv            per universe x signal x design x era: gross premium, Sharpe, turnover,
                         break-even costs (constant and volatility-proportional) with 95% CIs,
                         cost-timing multipliers, net Sharpe at 5-50 bps
  marginal_breakeven.csv extra premium per extra $ traded of the textbook over each slower design
  vol_vs_base.csv        net Sharpe of volatility-scaled momentum minus textbook, by cost level
  hypotheses.csv         the confirmatory family with raw and Holm-adjusted p-values
  stress_windows.csv     gross, turnover, cost and kappa inside each stress window
  long_only_active.csv   the long-only version measured against the equal-weight universe
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data
import engine as E
import stats as S
from designs import ERAS, GRID, PRIMARY_SIGNAL, SLOW, STRESS

COSTS = [5, 10, 20, 30, 50]
RES = data.RES


def load_runs() -> pd.DataFrame:
    return pd.read_csv(RES / "runs.csv.gz", parse_dates=["month"])


def window(runs: pd.DataFrame, u: str, s: str, a: str, b: str) -> dict[str, pd.DataFrame]:
    sub = runs[(runs.universe == u) & (runs.signal == s)]
    out = {}
    for d, g in sub.groupby("design"):
        out[d] = g.set_index("month").loc[a:b]
    # align on common months (every design starts at the same date by construction)
    common = sorted(set.intersection(*[set(x.index) for x in out.values()]))
    return {d: x.loc[common] for d, x in out.items()}


def arrays(x: pd.DataFrame, ref: float) -> dict[str, np.ndarray]:
    return {"g": (x["gross"] - x["tick_cost"]).values, "tau": x["turnover"].values,
            "vt": (x["vturn_raw"] / ref).values, "vraw": x["vturn_raw"].values,
            "sb": x["sigma_bar"].values, "gross": x["gross"].values}


def summarise(runs: pd.DataFrame):
    rows, marg, volrows = [], [], []
    for u in ("A", "B"):
        for s in sorted(runs.signal.unique()):
            for era, (a, b) in ERAS[u].items():
                W = window(runs, u, s, a, b)
                T = len(next(iter(W.values())))
                idx = S.stationary_indices(T)
                ref = W["base"]["sigma_bar"].mean()
                arr = {d: arrays(x, ref) for d, x in W.items()}
                boots = {}
                for d, x in W.items():
                    A = arr[d]
                    bg, bt, bv = (S.boot_means(A["g"], idx), S.boot_means(A["tau"], idx),
                                  S.boot_means(A["vt"], idx))
                    bsb = S.boot_means(A["tau"] * A["sb"], idx)
                    bref = S.boot_means(A["sb"], idx)
                    bk = S.boot_means(A["vraw"], idx) / (bt * bref)
                    boots[d] = (bg, bt, bv)
                    k = E.kappa(x)
                    row = {"universe": u, "signal": s, "design": d, "era": era, "months": T,
                           "gross_pct_yr": 1200 * A["gross"].mean(),
                           "tick_cost_bps_mo": 1e4 * x["tick_cost"].mean(),
                           "t_gross": A["gross"].mean() / A["gross"].std(ddof=1) * np.sqrt(T),
                           "sharpe_gross": S.sharpe(A["gross"]),
                           "turnover_mo": A["tau"].mean(),
                           "be_const_bps": E.breakeven(x, ref, "const"),
                           "be_vol_bps": E.breakeven(x, ref, "vol"),
                           "kappa": k["kappa"], "kappa_time": k["kappa_time"],
                           "kappa_cross": k["kappa_cross"],
                           "avg_gross_exposure": x["gross_exposure"].mean(),
                           "avg_scale": x["scale"].mean()}
                    lo, hi = S.ci(1e4 * bg / bv)
                    row.update(be_vol_lo=lo, be_vol_hi=hi)
                    lo, hi = S.ci(1e4 * bg / bt)
                    row.update(be_const_lo=lo, be_const_hi=hi)
                    lo, hi = S.ci(bk)
                    row.update(kappa_lo=lo, kappa_hi=hi,
                               p_kappa_gt1=S.p_greater(bk, k["kappa"], 1.0))
                    lo, hi = S.ci(1200 * bg)
                    row.update(gross_lo=lo, gross_hi=hi)
                    for c in COSTS:
                        row[f"sharpe_net_{c}"] = S.sharpe(A["g"] - c / 1e4 * A["vt"])
                        row[f"sharpe_netconst_{c}"] = S.sharpe(A["g"] - c / 1e4 * A["tau"])
                    rows.append(row)
                # marginal break-even: (gross_base - gross_slow) / (vturn_base - vturn_slow)
                bg0, bt0, bv0 = boots["base"]
                be0 = 1e4 * bg0 / bv0
                for d in SLOW:
                    if d not in boots:
                        continue
                    bg1, bt1, bv1 = boots[d]
                    dg = arr["base"]["g"].mean() - arr[d]["g"].mean()
                    dv = arr["base"]["vt"].mean() - arr[d]["vt"].mean()
                    point = 1e4 * dg / dv
                    draws = 1e4 * (bg0 - bg1) / (bv0 - bv1)
                    lo, hi = S.ci(draws)
                    avg = 1e4 * arr["base"]["g"].mean() / arr["base"]["vt"].mean()
                    # H2: marginal < average  <=>  (avg - marg) > 0
                    gap = be0 - draws
                    marg.append({"universe": u, "signal": s, "era": era, "slow_design": d,
                                 "d_gross_pct_yr": 1200 * dg, "d_vturn_mo": dv,
                                 "marginal_be_bps": point, "marginal_lo": lo, "marginal_hi": hi,
                                 "average_be_base_bps": avg,
                                 "share_of_turnover_saved": dv / arr["base"]["vt"].mean(),
                                 "share_of_premium_lost": dg / arr["base"]["g"].mean()
                                 if arr["base"]["g"].mean() != 0 else np.nan,
                                 "p_marginal_lt_average": S.p_greater(gap, avg - point, 0.0)})
                # volatility scaling vs textbook, paired
                if "vol" in arr:
                    for c in [0] + COSTS:
                        n0 = arr["base"]["g"] - c / 1e4 * arr["base"]["vt"]
                        n1 = arr["vol"]["g"] - c / 1e4 * arr["vol"]["vt"]
                        d0, d1 = S.boot_sharpe(n0, idx), S.boot_sharpe(n1, idx)
                        diff = S.sharpe(n1) - S.sharpe(n0)
                        lo, hi = S.ci(d1 - d0)
                        volrows.append({"universe": u, "signal": s, "era": era, "cost_bps": c,
                                        "sharpe_base": S.sharpe(n0), "sharpe_vol": S.sharpe(n1),
                                        "diff": diff, "lo": lo, "hi": hi,
                                        "p_vol_better": S.p_greater(d1 - d0, diff, 0.0)})
    return pd.DataFrame(rows), pd.DataFrame(marg), pd.DataFrame(volrows)


def hypotheses(summ, marg, volr) -> pd.DataFrame:
    s0 = PRIMARY_SIGNAL
    P, desc, est = {}, {}, {}
    r = summ[(summ.universe == "A") & (summ.signal == s0) & (summ.design == "base") &
             (summ.era == "1928-2026")].iloc[0]
    P["H1"] = r.p_kappa_gt1
    desc["H1"] = "A, 12-1 textbook: cost-timing multiplier kappa > 1"
    est["H1"] = f"kappa = {r.kappa:.3f} [{r.kappa_lo:.3f}, {r.kappa_hi:.3f}]"
    for d in ("partial50", "band50", "hold3"):
        m = marg[(marg.universe == "A") & (marg.signal == s0) & (marg.era == "1928-2026") &
                 (marg.slow_design == d)].iloc[0]
        k = f"H2-{d}"
        P[k] = m.p_marginal_lt_average
        desc[k] = f"A, 12-1: marginal break-even of textbook over {d} < its average break-even"
        est[k] = (f"marginal {m.marginal_be_bps:.0f} bps [{m.marginal_lo:.0f}, {m.marginal_hi:.0f}] vs "
                  f"average {m.average_be_base_bps:.0f}")
    for c in (10, 30):
        v = volr[(volr.universe == "A") & (volr.signal == s0) & (volr.era == "1928-2026") &
                 (volr.cost_bps == c)].iloc[0]
        k = f"H3-{c}bps"
        P[k] = v.p_vol_better
        desc[k] = f"A, 12-1: vol-scaled net Sharpe > textbook net Sharpe at {c} bps"
        est[k] = f"diff {v['diff']:+.2f} [{v.lo:+.2f}, {v.hi:+.2f}]"
    r = summ[(summ.universe == "B") & (summ.signal == s0) & (summ.design == "base") &
             (summ.era == "1998-2026")].iloc[0]
    # H4: positive gross premium in the tradable ETF universe (bootstrap of mean gross)
    W = window(load_runs(), "B", s0, *ERAS["B"]["1998-2026"])
    g = W["base"]["gross"].values
    idx = S.stationary_indices(len(g))
    bm = S.boot_means(g, idx)
    P["H4"] = S.p_greater(bm, g.mean(), 0.0)
    desc["H4"] = "B, 12-1 textbook: gross premium > 0"
    est["H4"] = f"{1200 * g.mean():+.2f}%/yr [{S.ci(1200 * bm)[0]:+.2f}, {S.ci(1200 * bm)[1]:+.2f}]"
    adj = S.holm(P)
    return pd.DataFrame([{"hypothesis": k, "description": desc[k], "estimate": est[k],
                          "p_raw": P[k], "p_holm": adj[k]} for k in P])


def stress(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for u in ("A", "B"):
        sub = runs[(runs.universe == u) & (runs.signal == PRIMARY_SIGNAL)]
        full = {d: g.set_index("month") for d, g in sub.groupby("design")}
        a, b = next(iter(ERAS[u].values()))
        ref = full["base"].loc[a:b]["sigma_bar"].mean()
        for name, (s0, s1) in STRESS.items():
            for d in ("base", "band50", "partial33", "vol"):
                x = full[d].loc[s0:s1]
                if len(x) == 0:
                    continue
                vt = (x["vturn_raw"] / ref)
                k = E.kappa(x) if x["turnover"].sum() > 0 else {"kappa": np.nan}
                allx = full[d].loc[a:b]
                rows.append({"universe": u, "window": name, "design": d, "months": len(x),
                             "gross_pct": 100 * ((1 + x["gross"]).prod() - 1),
                             "turnover_per_mo": x["turnover"].mean(),
                             "turnover_vs_full": x["turnover"].mean() / allx["turnover"].mean(),
                             "sigma_vs_full": x["sigma_bar"].mean() / allx["sigma_bar"].mean(),
                             "cost_pct_at_10bps": 100 * (10e-4 * vt + x["tick_cost"]).sum(),
                             "cost_pct_at_30bps": 100 * (30e-4 * vt + x["tick_cost"]).sum(),
                             "cost_per_mo_vs_full_30bps": (vt.mean() / (allx["vturn_raw"] / ref).mean()),
                             "kappa_in_window": k["kappa"]})
    return pd.DataFrame(rows)


def long_only(runs: pd.DataFrame) -> pd.DataFrame:
    ew = pd.read_csv(RES / "ew_benchmark.csv", parse_dates=["month"])
    rows = []
    for u in ("A", "B"):
        bm = ew[ew.universe == u].set_index("month")["ew"]
        for s in sorted(runs.signal.unique()):
            x = runs[(runs.universe == u) & (runs.signal == s) & (runs.design == "long_only")]
            x = x.set_index("month")
            for era, (a, b) in ERAS[u].items():
                y = x.loc[a:b]
                act = (y["gross"] - bm.reindex(y.index)).dropna()
                y = y.loc[act.index]
                ref = runs[(runs.universe == u) & (runs.signal == s) & (runs.design == "base")]
                ref = ref.set_index("month").loc[a:b]["sigma_bar"].mean()
                vt = y["vturn_raw"] / ref
                T = len(act)
                idx = S.stationary_indices(T)
                g = (act - y["tick_cost"]).values
                bg, bv = S.boot_means(g, idx), S.boot_means(vt.values, idx)
                lo, hi = S.ci(1e4 * bg / bv)
                rows.append({"universe": u, "signal": s, "era": era,
                             "active_pct_yr": 1200 * act.mean(),
                             "t_active": act.mean() / act.std(ddof=1) * np.sqrt(T),
                             "turnover_mo": y["turnover"].mean(),
                             "be_vol_bps": 1e4 * g.mean() / vt.mean(), "be_lo": lo, "be_hi": hi,
                             "info_ratio_gross": S.sharpe(act.values),
                             "info_ratio_net_10": S.sharpe(g - 10e-4 * vt.values),
                             "info_ratio_net_30": S.sharpe(g - 30e-4 * vt.values)})
    return pd.DataFrame(rows)


def main():
    runs = load_runs()
    summ, marg, volr = summarise(runs)
    summ.to_csv(RES / "summary.csv", index=False, float_format="%.5g")
    marg.to_csv(RES / "marginal_breakeven.csv", index=False, float_format="%.5g")
    volr.to_csv(RES / "vol_vs_base.csv", index=False, float_format="%.5g")
    h = hypotheses(summ, marg, volr)
    h.to_csv(RES / "hypotheses.csv", index=False, float_format="%.4g")
    stress(runs).to_csv(RES / "stress_windows.csv", index=False, float_format="%.4g")
    long_only(runs).to_csv(RES / "long_only_active.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print(h.to_string(index=False))


if __name__ == "__main__":
    main()
