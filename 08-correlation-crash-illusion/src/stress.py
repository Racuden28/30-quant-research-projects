"""
Stress windows (GFC, COVID, 2022 rate shock, plus 1998 and 2011 for the stocks): raw vs devolatised
average correlation in the window against the year before, and the out-of-sample GMV result inside
each window. Output: results/stress_windows.csv, results/stress_forecast.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from conditioning import WARM, blocks_of
from core import corr, ewma_vol, fisher_mean
from data import load

RES = Path(__file__).resolve().parents[1] / "results"
WINDOWS = {"LTCM / Russia 1998": ("1998-08-01", "1998-10-31"),
           "Global Financial Crisis": ("2008-09-01", "2009-03-31"),
           "Euro crisis / US downgrade 2011": ("2011-08-01", "2011-10-31"),
           "COVID crash": ("2020-02-20", "2020-04-30"),
           "2022 rate shock": ("2022-01-01", "2022-10-31")}


def main():
    rows = []
    for u in ["stocks", "etfs"]:
        df = load(u)
        cols = list(df.columns)
        sig = ewma_vol(df.values)
        r = df.iloc[WARM:]
        z = pd.DataFrame(df.values[WARM:] / sig[WARM:], index=r.index, columns=cols)
        for name, (a, b) in WINDOWS.items():
            w = (r.index >= a) & (r.index <= b)
            if w.sum() < 30:
                continue
            start = np.argmax(w)
            pre = slice(max(0, start - 252), start)
            for blk, ix in blocks_of(u, cols).items():
                if u == "etfs" and blk == "SPY-GLD":
                    continue
                rows.append(dict(universe=u, window=name, block=blk, days=int(w.sum()),
                                 raw_pre=fisher_mean(corr(r.values[pre]), ix),
                                 raw_window=fisher_mean(corr(r.values[w]), ix),
                                 devol_pre=fisher_mean(corr(z.values[pre]), ix),
                                 devol_window=fisher_mean(corr(z.values[w]), ix)))
    out = pd.DataFrame(rows)
    out["raw_change"] = out.raw_window - out.raw_pre
    out["devol_change"] = out.devol_window - out.devol_pre
    out.to_csv(RES / "stress_windows.csv", index=False, float_format="%.4f")

    f = pd.read_csv(RES / "forecast_losses.csv.gz", parse_dates=["origin"])
    fr = []
    for name, (a, b) in WINDOWS.items():
        # origins whose 21-day forecast window overlaps the stress window
        g = f[(f.origin >= pd.Timestamp(a) - pd.Timedelta(days=30)) & (f.origin <= b)]
        for u, gu in g.groupby("universe"):
            v = gu.groupby("estimator").gmv_var.mean()
            for e, x in v.items():
                fr.append(dict(universe=u, window=name, estimator=e, n_origins=int(gu.origin.nunique()),
                               gmv_vol_ann=np.sqrt(x * 252), gmv_rel=np.sqrt(x / v["Sample 1y"]) - 1))
    pd.DataFrame(fr).to_csv(RES / "stress_forecast.csv", index=False, float_format="%.4f")
    print(out.round(3).to_string())
    print(pd.DataFrame(fr).pivot_table(index=["universe", "window"], columns="estimator", values="gmv_rel").round(3).to_string())


if __name__ == "__main__":
    main()
