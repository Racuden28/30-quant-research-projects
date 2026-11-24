"""Crypto loss diagnostic: where does the s-score book lose money?

Re-runs the crypto main specification (seconds) and decomposes its gross P&L into the long
and short legs and into what is left when daily returns are capped at +-c (the tail share).
Output: results/crypto_tail_diagnostic.csv
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
import engine as en
from evaluate import RES, load_runs, common_start


def main():
    r, q, _ = data.load_crypto()
    res = en.run(r, q, en.Config(k="mp", N=100))
    w = res.weights["ou"]
    first = common_start(load_runs(), "crypto")     # same sample as every other crypto table
    ws = w.shift(1).fillna(0.0); R = r.fillna(0.0)
    rows = []
    for cap in (None, 1.0, 0.5, 0.3, 0.2, 0.1):
        Rc = R if cap is None else R.clip(-cap, cap)
        g = (ws * Rc).sum(axis=1).loc[first:]
        L = (ws.clip(lower=0) * Rc).sum(axis=1).loc[first:]
        S = (ws.clip(upper=0) * Rc).sum(axis=1).loc[first:]
        rows.append(dict(return_cap=("none" if cap is None else cap), ann_gross=g.mean() * 365,
                         sr_gross=g.mean() / g.std() * np.sqrt(365), positive_weights_ann=L.mean() * 365,
                         negative_weights_ann=S.mean() * 365))
    held = (ws != 0) & (R.abs() > 0.3)
    df = pd.DataFrame(rows)
    df["share_held_asset_days_abs_ret_gt_30pct"] = float(held.loc[first:].to_numpy().sum() / (ws.loc[first:] != 0).to_numpy().sum())
    df.to_csv(os.path.join(RES, "crypto_tail_diagnostic.csv"), index=False)
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
