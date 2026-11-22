"""Experiment 4: what rebalancing is worth once you pay for it.

S&P 500 one-month ATM calls sold at VIX (1990-2026) and the 7 ETFs sold at their EWMA forecast
(2006-2026), hedged with the fixed-quote delta under two families of rules:

  calendar  rebalance every h days, h in {1, 2, 3, 5, 10, 21}
  band      rebalance whenever |BS delta - held delta| exceeds a band b, b in {0.02, ..., 0.30}
            (the shape of the Whalley-Wilmott no-trade region, with a constant width)

with a proportional cost of k bps of notional per trade (k = 1, 5, 20 for the S&P 500; 5 for the ETFs).
For every rule we report the mean cost and the standard deviation of the total P&L, both in vol points,
and pick the rule that minimises  mean cost + lambda * std  for lambda in {0.5, 1, 2}.

Writes results/costs_frontier.csv, results/costs_best.csv, results/costs_band_vs_calendar.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import ETFS, RES, T_DAYS, etf_prices, spx
from frontier import block_indices
from hedgesim import every_h_days, hedge_short_call, windows

H_RULES = [1, 2, 3, 5, 10, 21]
BANDS = [0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.30]
LAMBDAS = [0.5, 1.0, 2.0]
SEED = 1604
B = 1000


def run_rules(P, sq, k):
    rows, pnls = [], {}
    for h in H_RULES:
        r = hedge_short_call(P, 100.0, sq, schedule=every_h_days(T_DAYS, h), cost_bps=k)
        pnls[f"h{h}"] = r.pnl
        rows.append(dict(family="calendar", rule=f"every {h}d", param=h, mean_cost=r.cost.mean(),
                         std_pnl=r.pnl.std(ddof=1), std_gross=(r.pnl + r.cost).std(ddof=1),
                         trades=r.n_trades.mean() + 1, mean_pnl=r.pnl.mean()))
    for b in BANDS:
        r = hedge_short_call(P, 100.0, sq, band=b, cost_bps=k)
        pnls[f"b{b}"] = r.pnl
        rows.append(dict(family="band", rule=f"band {b:.2f}", param=b, mean_cost=r.cost.mean(),
                         std_pnl=r.pnl.std(ddof=1), std_gross=(r.pnl + r.cost).std(ddof=1),
                         trades=r.n_trades.mean() + 1, mean_pnl=r.pnl.mean()))
    return pd.DataFrame(rows), pnls


def main():
    rng = np.random.default_rng(SEED)
    books = {}
    d = spx().loc["1990":]
    d = d[d.vix.notna()]
    P, idx = windows(d.close.to_numpy(), T_DAYS)
    books["SPX"] = (P, d.vix.to_numpy()[idx], [1.0, 5.0, 20.0])
    for a, e in etf_prices().items():
        P, idx = windows(e.close.to_numpy(), T_DAYS)
        q = e.ewma.to_numpy()[idx]
        ok = np.isfinite(q)
        books[a] = (P[ok], q[ok], [5.0])

    frontier, best, bvc = [], [], []
    for asset, (P, sq, ks) in books.items():
        for k in ks:
            fr, pnls = run_rules(P, sq, k)
            fr.insert(0, "cost_bps", k)
            fr.insert(0, "asset", asset)
            frontier.append(fr)
            for lam in LAMBDAS:
                obj = fr.mean_cost + lam * fr.std_pnl
                i = obj.idxmin()
                cal = fr[fr.family == "calendar"]
                j = (cal.mean_cost + lam * cal.std_pnl).idxmin()
                best.append(dict(asset=asset, cost_bps=k, lam=lam, best_rule=fr.rule[i],
                                 best_obj=obj[i], best_calendar=fr.rule[j], best_calendar_obj=obj[j],
                                 daily_obj=obj[fr.index[fr.rule == "every 1d"][0]]))
            # band vs calendar at (about) equal cost: compare each calendar rule with the band rule whose
            # mean cost is closest, on std of total P&L, with a paired block bootstrap
            cal = fr[fr.family == "calendar"]
            bnd = fr[fr.family == "band"]
            n = len(P)
            boots = [block_indices(n, rng) for _ in range(B)]
            for _, c in cal.iterrows():
                if c.param == 21:
                    continue
                j = (bnd.mean_cost - c.mean_cost).abs().idxmin()
                bb = bnd.loc[j]
                x, y = pnls[f"h{int(c.param)}"], pnls[f"b{bb.param}"]
                dstd = y.std() - x.std()
                bt = np.array([y[s].std() - x[s].std() for s in boots])
                bvc.append(dict(asset=asset, cost_bps=k, calendar=c.rule, band=bb.rule,
                                cost_calendar=c.mean_cost, cost_band=bb.mean_cost,
                                std_calendar=c.std_pnl, std_band=bb.std_pnl, d_std=dstd,
                                d_std_lo=np.percentile(bt, 2.5), d_std_hi=np.percentile(bt, 97.5)))
    pd.concat(frontier).to_csv(RES / "costs_frontier.csv", index=False, float_format="%.4f")
    pd.DataFrame(best).to_csv(RES / "costs_best.csv", index=False, float_format="%.4f")
    pd.DataFrame(bvc).to_csv(RES / "costs_band_vs_calendar.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(best).round(2).to_string())
    print(pd.DataFrame(bvc).query("asset=='SPX'").round(2).to_string())


if __name__ == "__main__":
    main()
