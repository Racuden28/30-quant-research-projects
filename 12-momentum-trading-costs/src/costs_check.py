"""
Can trading costs be measured from daily prices? A check on the ETF universe.

Two popular estimators recover the bid-ask spread from daily high, low and close prices:
Corwin & Schultz (2012) and Abdi & Ranaldo (2017). Both are used to put costs into backtests when no
quote data exist. We compute them month by month for SPY, the most liquid security in the world,
whose quoted spread is almost always one cent, and for the 42 country ETFs.

The tick size is a hard, *measured* lower bound on the quoted spread: no quote can be narrower than
one tick. Before decimalisation (April 2001) the tick was $1/16 (and $1/8 before June 1997), which on
a $10-20 fund is 30-60 bps for the full spread. That floor is what backtest.py charges as the
measured part of the cost in universe B.

Output: results/spread_estimators.csv (monthly, by fund), results/spread_estimator_summary.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data
from backtest import tick_size

K = 3 - 2 * np.sqrt(2)


def monthly_estimators(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date")
    h = np.maximum(g["high"].values, g["close"].values)
    l = np.minimum(g["low"].values, g["close"].values)
    c = g["close"].values
    m = g["date"].dt.to_period("M").values
    same = m[:-1] == m[1:]
    lc, eta = np.log(c), (np.log(h) + np.log(l)) / 2
    ar = (lc[:-1] - eta[:-1]) * (lc[:-1] - eta[1:])
    beta = np.log(h[:-1] / l[:-1]) ** 2 + np.log(h[1:] / l[1:]) ** 2
    gam = np.log(np.maximum(h[:-1], h[1:]) / np.minimum(l[:-1], l[1:])) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / K - np.sqrt(gam / K)
    cs = np.clip(2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)), 0, None)
    df = pd.DataFrame({"m": m[:-1][same], "ar": ar[same], "cs": cs[same]})
    out = df.groupby("m").agg(ar=("ar", "mean"), cs=("cs", "mean"), n=("ar", "size"))
    out["ar_spread"] = np.sqrt(np.clip(4 * out["ar"], 0, None))
    px = g.groupby(g["date"].dt.to_period("M")).agg(close=("close", "last"), date=("date", "last"))
    out = out.join(px)
    out["one_tick_spread"] = tick_size(pd.DatetimeIndex(out["date"])).values / out["close"]
    return out.drop(columns="ar")


def main():
    d = data.etf_raw()
    d = d[d.ticker.isin(list(data.COUNTRY) + ["SPY"])]
    rows = []
    for t, g in d.groupby("ticker"):
        e = monthly_estimators(g)
        e["ticker"] = t
        rows.append(e.reset_index())
    E = pd.concat(rows)
    E = E[E["m"].astype(str) <= data.LAST_MONTH_B]
    E.to_csv(data.RES / "spread_estimators.csv", index=False, float_format="%.6g")
    E["year"] = E["m"].dt.year
    summ = []
    for (grp, era), x in E.assign(grp=np.where(E.ticker == "SPY", "SPY", "Country ETFs"),
                                 era=pd.cut(E.year, [1992, 2001, 2008, 2019, 2026],
                                            labels=["1993-2001", "2002-2008", "2009-2019",
                                                    "2020-2026"])).groupby(["grp", "era"],
                                                                           observed=True):
        summ.append({"group": grp, "era": str(era), "fund_months": len(x),
                     "median_cs_bps": 1e4 * x["cs"].median(),
                     "median_ar_bps": 1e4 * x["ar_spread"].median(),
                     "share_ar_zero": float((x["ar_spread"] == 0).mean()),
                     "median_one_tick_bps": 1e4 * x["one_tick_spread"].median(),
                     "median_cs_over_tick": float((x["cs"] / x["one_tick_spread"]).median()),
                     "median_ar_over_tick": float((x["ar_spread"] / x["one_tick_spread"]).median())})
    S = pd.DataFrame(summ)
    S.to_csv(data.RES / "spread_estimator_summary.csv", index=False, float_format="%.4g")
    print(S.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
