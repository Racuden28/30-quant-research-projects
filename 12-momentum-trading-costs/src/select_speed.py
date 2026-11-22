"""
Can an investor learn the right trading speed in real time?

Each January from 1948 in A (20-year burn-in; 2003 in B, 5 years) an investor who pays a one-way cost of c bps (volatility-
proportional model) picks, from the 17 non-scaled long-short designs, the one with the highest mean
net return over all past months (expanding window). The cost normalisation sigma_ref is also an
expanding mean, so no future information enters the choice. Switching design is charged as a full
textbook rebalance (the books differ), which overstates the true switching cost.

Output: results/speed_selection.csv (net Sharpe of the selected-speed strategy, the textbook and the
best fixed design in hindsight, with paired bootstrap CIs) and results/speed_choices.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data
import stats as S
from designs import FAMILY, GRID, PRIMARY_SIGNAL

START = {"A": "1948-01", "B": "2003-01"}
MENU = [d.key for d in GRID if FAMILY[d.key] != "vol"]


def main(universe: str = "A", signal: str = PRIMARY_SIGNAL):
    runs = pd.read_csv(data.RES / "runs.csv.gz", parse_dates=["month"])
    sub = runs[(runs.universe == universe) & (runs.signal == signal) & runs.design.isin(MENU)]
    G = sub.pivot(index="month", columns="design", values="gross")
    V = sub.pivot(index="month", columns="design", values="vturn_raw")
    TK = sub.pivot(index="month", columns="design", values="tick_cost").fillna(0)
    sb = sub[sub.design == "base"].set_index("month")["sigma_bar"]
    ref = sb.expanding().mean().shift(1)               # known before the month
    out, choices = [], []
    for c in (10, 20, 30, 50):
        cost = V.div(ref, axis=0) * c / 1e4 + TK
        net = G - cost
        hist = net.expanding().mean().shift(1)
        sel = pd.Series(index=net.index, dtype=object)
        cur = "base"
        for m in net.index:
            if m < pd.Timestamp(START[universe]):
                continue
            if m.month == 1:
                cur = hist.loc[m].idxmax()
            sel[m] = cur
        sel = sel.dropna()
        r = pd.Series([net.loc[m, d] for m, d in sel.items()], index=sel.index)
        switch = sel.ne(sel.shift(1)) & sel.shift(1).notna()
        r = r - switch * (V.loc[sel.index, "base"] / ref.loc[sel.index] * c / 1e4)
        base = net.loc[sel.index, "base"]
        fixed_best = net.loc[sel.index].mean().idxmax()
        best = net.loc[sel.index, fixed_best]
        idx = S.stationary_indices(len(r))
        bs_r, bs_b, bs_f = (S.boot_sharpe(x.values, idx) for x in (r, base, best))
        lo, hi = S.ci(bs_r - bs_b)
        out.append({"universe": universe, "signal": signal, "cost_bps": c,
                    "months": len(r), "n_switches": int(switch.sum()),
                    "sharpe_selected": S.sharpe(r.values), "sharpe_textbook": S.sharpe(base.values),
                    "best_fixed_in_hindsight": fixed_best, "sharpe_best_fixed": S.sharpe(best.values),
                    "diff_selected_minus_textbook": S.sharpe(r.values) - S.sharpe(base.values),
                    "diff_lo": lo, "diff_hi": hi,
                    "p_selected_better": S.p_greater(bs_r - bs_b, S.sharpe(r.values) - S.sharpe(base.values)),
                    "most_chosen": sel.value_counts().index[0],
                    "share_most_chosen": sel.value_counts().iloc[0] / len(sel)})
        choices.append(sel.rename(f"c{c}"))
    res = pd.DataFrame(out)
    res.to_csv(data.RES / f"speed_selection_{universe}.csv", index=False, float_format="%.4g")
    pd.concat(choices, axis=1).to_csv(data.RES / f"speed_choices_{universe}.csv")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main("A")
    main("B")
