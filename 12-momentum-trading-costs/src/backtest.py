"""Run every (universe, signal, design) and save the monthly book records to results/runs.csv.gz."""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import data
import engine as E
from designs import GRID, LONG_ONLY, SIGNALS

STALE_WINDOW, STALE_MIN = 63, 0.80      # universe B: traded on >= 80% of the last 63 sessions


def tick_size(dates: pd.DatetimeIndex) -> pd.Series:
    """Minimum price increment on the AMEX/NYSE Arca for these funds (USD)."""
    t = pd.Series(0.01, index=dates)
    t[dates < "2001-04-09"] = 1 / 16          # AMEX completed decimalisation on 9 April 2001
    t[dates < "1997-06-24"] = 1 / 8           # sixteenths adopted in June 1997
    return t


def universe_a() -> E.Universe:
    return E.Universe("A", data.industries())


def universe_b() -> E.Universe:
    b = data.etf_clean()
    U = E.Universe("B", b["ret"])
    traded = (~b["stale"]).astype(float).where(b["ret"].notna())
    share = traded.rolling(STALE_WINDOW, min_periods=STALE_WINDOW).mean()
    U.extra_eligible = share.loc[U.month_end].set_axis(U.monthly.index).ge(STALE_MIN)
    close = b["close"].loc[U.month_end]
    half = (0.5 * tick_size(close.index)).values[:, None] / close
    U.halftick = half.set_axis(U.monthly.index)
    return U


_U: dict = {}


def _job(args):
    uname, sname, dkey = args
    U = _U[uname]
    lb, sk = SIGNALS[sname]
    S = E.momentum_signal(U, lb, sk)
    D = {d.key: d for d in GRID + [LONG_ONLY]}[dkey]
    if uname == "A":
        out = E.run(U, S, D, start="1928-01", min_assets=15)["monthly"]
    else:
        out = E.run(U, S, D, start="1998-01", min_assets=10)["monthly"]
        out = out.loc[: data.LAST_MONTH_B]
    out = out.assign(universe=uname, signal=sname, design=dkey)
    return out.reset_index()


def main(n_proc: int = 4):
    t0 = time.time()
    _U["A"], _U["B"] = universe_a(), universe_b()
    jobs = [(u, s, d.key) for u in ("A", "B") for s in SIGNALS for d in GRID + [LONG_ONLY]]
    with Pool(n_proc) as p:
        res = p.map(_job, jobs, chunksize=1)
    runs = pd.concat(res, ignore_index=True)
    runs.to_csv(data.RES / "runs.csv.gz", index=False, float_format="%.8g")
    # equal-weight benchmark of the eligible universe (for the long-only active return)
    rows = []
    for uname, U in _U.items():
        lb, sk = SIGNALS["12-1"]
        S = E.momentum_signal(U, lb, sk)
        elig = S.notna() & U.present
        if U.extra_eligible is not None:
            elig &= U.extra_eligible.reindex_like(elig).fillna(False)
        nxt = U.monthly.shift(-1)
        ew = nxt.where(elig).mean(axis=1).shift(1).dropna()
        rows.append(pd.DataFrame({"month": ew.index, "ew": ew.values, "universe": uname}))
    pd.concat(rows).to_csv(data.RES / "ew_benchmark.csv", index=False, float_format="%.8g")
    # cost-state reference: mean over months of the cross-sectional mean 21-day vol, per universe
    print(f"{len(jobs)} runs in {time.time() - t0:.0f}s -> results/runs.csv.gz ({len(runs):,} rows)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
