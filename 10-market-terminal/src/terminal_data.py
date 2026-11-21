"""Build the data behind the terminal page (terminal/terminal_data.json + embedded copy in terminal/index.html).

For each asset, at the last close in the data:
* every widget's value and its percentile in the asset's own history;
* the widget's audited information (from results/pooled_h21.csv): DM = survives BH-FDR on Diebold–Mariano,
  CW = survives on Clark–West only, none = neither;
* what the widget's current value does to next month's volatility forecast, from a full-sample OLS of log forward
  21-day vol on [log RV21, widget] (the last refit of the out-of-sample procedure), relative to the widget at its median;
* a headline next-month volatility forecast from log RV21 + Bollinger %b (the best single incremental widget),
  with a 90% range from the empirical residual quantiles.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from widgets import ASSETS, FAMILY, ROOT, TARGET_LABEL, WIDGET_LABEL, WIDGETS, compute_targets, compute_widgets, \
    load_returns

RES = ROOT / "results"
OUT = ROOT / "terminal"
NAMES = {"SPY": "US equities", "EEM": "Emerging-market equities", "TLT": "20y+ Treasuries",
         "HYG": "High-yield credit", "GLD": "Gold", "USO": "Crude oil", "URA": "Uranium miners"}


def badge(P, tg, test, w):
    r = P[(P.target == tg) & (P.test == test) & (P.widget == w)]
    if not len(r):
        return None
    r = r.iloc[0]
    # "Weak" requires a lenient-test (Clark–West) rejection AND a positive out-of-sample R²: CW tests a nested model and
    # can reject while the estimated forecast loses to the baseline, which a screen user would misread as "signal".
    lvl = "dm" if r.dm_survives_fdr else ("cw" if (r.survives_fdr and r.r2os > 0) else "none")
    return {"level": lvl, "r2os": round(float(r.r2os) * 100, 2)}


def ols(y, X):
    Z = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return b, y - Z @ b


def main():
    P = pd.read_csv(RES / "pooled_h21.csv")
    R = load_returns()
    assets, slopes = [], []
    for a in ASSETS:
        r = R[a].dropna()
        W, Y = compute_widgets(r), compute_targets(r, 21)
        D = W.join(Y).dropna()
        lrv = np.log(D.rv21.values)
        last = W.iloc[-1]
        hist = W.dropna()
        # headline forecast: log RV21 + Bollinger %b
        b, e = ols(D.vol.values, np.column_stack([lrv, D.boll.values]))
        mu = b[0] + b[1] * np.log(last.rv21) + b[2] * last.boll
        q05, q95 = np.quantile(e, [0.05, 0.95])
        b0, _ = ols(D.vol.values, lrv)
        rv_only = b0[0] + b0[1] * np.log(last.rv21)
        rows = []
        for w in WIDGETS:
            if w == "rv21":
                eff = None
            else:
                bw, _ = ols(D.vol.values, np.column_stack([lrv, D[w].values]))
                med = hist[w].median()
                eff = float(np.expm1(bw[2] * (last[w] - med)) * 100)
                lo, hi = hist[w].quantile([0.1, 0.9])
                slopes.append(dict(asset=a, widget=w, beta=bw[2],
                                   vol_change_p10_to_p90_pct=np.expm1(bw[2] * (hi - lo)) * 100))
            rows.append({
                "id": w, "label": WIDGET_LABEL[w], "family": FAMILY[w], "value": float(last[w]),
                "pctile": float((hist[w] <= last[w]).mean()),
                "ret": badge(P, "ret", "standalone", w),
                "vol": badge(P, "vol", "standalone" if w == "rv21" else "incremental", w),
                "mdd": badge(P, "mdd", "standalone" if w == "rv21" else "incremental", w),
                "vol_effect_pct": None if eff is None else round(eff, 1),
            })
        lp = r.cumsum().iloc[-252:]
        assets.append({
            "ticker": a, "name": NAMES[a], "date": str(W.index[-1].date()),
            "chg1d": float(last.chg1d), "spark": [round(float(v), 4) for v in (lp - lp.iloc[0]).values],
            "rv21": float(last.rv21), "vol_fc": float(np.exp(mu)), "vol_lo": float(np.exp(mu + q05)),
            "vol_hi": float(np.exp(mu + q95)), "vol_rv_only": float(np.exp(rv_only)),
            "vol_hist_median": float(np.exp(np.median(D.vol.values))),
            "widgets": rows,
        })
    pd.DataFrame(slopes).to_csv(RES / "vol_slopes_full_sample.csv", index=False, float_format="%.4g")
    meta = {"targets": TARGET_LABEL, "asof": assets[0]["date"],
            "n_oos_days": int(P.n_dates.max()), "period": "2012–2026"}
    OUT.mkdir(exist_ok=True)
    payload = {"meta": meta, "assets": assets}
    (OUT / "terminal_data.json").write_text(json.dumps(payload, separators=(",", ":")))
    tpl = (ROOT / "src" / "terminal_template.html").read_text(encoding="utf-8")
    body = tpl.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           + body.replace("<div class=\"wrap\">", "</head>\n<body>\n<div class=\"wrap\">", 1) + "\n</body>\n</html>\n")
    (OUT / "index.html").write_text(doc, encoding="utf-8")
    print("terminal written", OUT / "index.html")


if __name__ == "__main__":
    main()
