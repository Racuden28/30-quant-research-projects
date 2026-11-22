"""Sensitivity of the five pre-registered tests to every free parameter. Writes results/robustness.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

import analysis as A
import data as D
import stats as S

A.SAVE = False
A.N_BOOT = 4000


def run() -> dict:
    sp = A.spanning()
    _, td, _, _ = A.tangency_holdout()
    _, _, pdiff, _ = A.pricing()
    _, _, summ, h4 = A.real_time()
    hy = A.hypotheses(sp, td, pdiff, h4)
    out = {f"{h} est": e for h, e in zip(hy["H"], hy["estimate"])}
    out.update({f"{h} p": p for h, p in zip(hy["H"], hy["p"])})
    return out


def set_holdout(start: str):
    D.HO_START = start
    A.HO = D.window(A.F, D.HO_START, D.HO_END)
    A.IS = D.window(A.F, D.IS_START, f"{int(start[:4]) - 1}-12")
    A.PERIODS = {"In-sample 1963-2013": A.IS, "Holdout 2014-2026": A.HO}


if __name__ == "__main__":
    rows = [{"variant": "baseline", **run()}]
    for lags in (3, 12):
        S.NW_LAGS = lags
        A.S.ols_nw.__defaults__ = (lags, True)
        A.S.mean_nw.__defaults__ = (lags,)
        rows.append({"variant": f"Newey-West lags {lags}", **run()})
    A.S.ols_nw.__defaults__ = (6, True)
    A.S.mean_nw.__defaults__ = (6,)
    for b in (3.0, 12.0):
        A.BLOCK = b
        rows.append({"variant": f"bootstrap mean block {b:.0f}m", **run()})
    A.BLOCK = 6.0
    for w in (36, 120):
        A.BETA_WINDOW = w
        A.oos_alphas.__defaults__ = (w,)
        rows.append({"variant": f"beta window {w}m", **run()})
    A.oos_alphas.__defaults__ = (60,)
    for rt in ("1973-12", "1993-12"):
        A.RT_START = rt
        rows.append({"variant": f"real-time start {rt}", **run()})
    A.RT_START = "1983-12"
    set_holdout("2015-01")
    rows.append({"variant": "holdout starts 2015-01 (publication year)", **run()})
    set_holdout("2014-01")
    out = pd.DataFrame(rows).set_index("variant")
    out.to_csv(A.RES / "robustness.csv", float_format="%.5g")
    print(out.round(3).T.to_string())
