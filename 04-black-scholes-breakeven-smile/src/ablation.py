"""Robustness of the break-even smile to every modelling choice, plus a split-sample stability check.

Variants: baseline (21d, daily hedge, gamma at quoted vol, simple returns); gamma evaluated at the
trailing vol; weekly hedging; 3-month options; log returns; first half (2007-2016) and second half
(2017-2026) of the sample. Writes results/ablation.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import XW, ix, stats_from_curve
from breakeven import ASSETS, RES, Spec, build_paths, load_returns, pnl_grid, root_on_grid

VARIANTS = {
    "Baseline": (Spec(), None),
    "Gamma at trailing vol": (Spec(gamma_at_ref=True), None),
    "Weekly hedging": (Spec(step=5), None),
    "3-month options": (Spec(horizon=63), None),
    "Log returns": (Spec(simple=False), None),
    "First half 2007-2016": (Spec(), ("2007-01-01", "2016-12-31")),
    "Second half 2017-2026": (Spec(), ("2017-01-01", "2026-12-31")),
}


def main():
    lr = load_returns()
    rows = []
    for asset in ASSETS:
        for name, (spec, win) in VARIANTS.items():
            dates, sig, logS = build_paths(lr[asset], spec)
            if win is not None:
                mk = (dates >= win[0]) & (dates <= win[1])
                sig, logS = sig[mk], logS[mk]
            m = root_on_grid(pnl_grid(sig, logS, spec).mean(0))
            s, c, lv = stats_from_curve(m)
            rows.append(dict(asset=asset, variant=name, m_put=m[ix(-XW)], m_atm=m[ix(0)], m_call=m[ix(XW)],
                             skew=s, convexity=c, n_paths=len(sig)))
        print(asset)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "ablation.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
