"""Event study: what each model predicted at the start of every large volatility shock, and what came.

Event trigger (known at t, no look-ahead): the trailing 22-day realized variance RV22_t rises above
4x its trailing 5-year median (volatility doubles relative to its recent norm), for the first time
after at least 126 trading days below the trigger. Origins are restricted to the out-of-sample
period of backtest.py, so every model only uses parameters estimated before the event.

For each event and model: log(realized / forecast) cumulative variance at 5..252 days, and the
predicted vs realized "halfway time" of the excess variance. Inference treats events in the same
calendar month across indices as one cluster (2008 hits every market at once).
Output: results/events.csv, results/event_errors.csv, results/event_paths.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import H, MODEL_ORDER  # noqa: E402
from data import INDICES, ROOT, load_returns  # noqa: E402
from irf import model_path  # noqa: E402
from models import half_life  # noqa: E402

RES = ROOT / "results"
TRIG, COOL, BASE_WIN = 4.0, 126, 1260
KMAX = 252
NBOOT = 2000
SEED = 7


def find_events(d: pd.DataFrame, full: pd.Series):
    """Trigger computed on the full return history (so the 5-year baseline exists from day one of
    the out-of-sample period), then aligned to the backtest origins."""
    rv = (full ** 2).rolling(22).mean()
    bs = rv.rolling(BASE_WIN, min_periods=750).median()
    rv22, base = rv.reindex(d.index).values, bs.reindex(d.index).values
    r2 = d["r"].values ** 2
    above = rv22 > TRIG * base
    ev, last_above = [], -10 ** 9
    for t in range(len(r2)):
        if not np.isfinite(base[t]):
            continue
        if above[t]:
            if t - last_above > COOL and t + KMAX < len(r2):
                ev.append(t)
            last_above = t
    return ev, rv22, base


def main():
    events, errs, paths = [], [], []
    R = load_returns()
    for c in INDICES:
        d = pd.read_pickle(RES / "forecasts" / f"{c}.pkl.gz")
        r2 = d["r"].values ** 2
        ev, rv22, base = find_events(d, R[c])
        for t in ev:
            date = d.index[t]
            exc0 = rv22[t] - base[t]
            # realized smoothed path: 21-day centred mean of r^2 (one-sided at the start)
            sm = np.array([r2[t + max(1, k - 10): t + k + 11].mean() for k in range(1, KMAX + 1)])
            rel = (sm - base[t]) / exc0
            k_half = int(np.argmax(rel <= 0.5)) + 1 if np.any(rel <= 0.5) else np.nan
            row = dict(index=c, date=date.date(), t=t, rv22_vol=np.sqrt(252 * rv22[t]),
                       base_vol=np.sqrt(252 * base[t]), ratio=rv22[t] / base[t],
                       garch_vol_tomorrow=np.sqrt(252 * d["GARCH_s"].iloc[t]),
                       garch_hl=float(half_life(d["GARCH_phi"].iloc[t])),
                       gjr_hl=float(half_life(d["GJR_phi"].iloc[t])),
                       cg_short_hl=float(half_life(d["CG_psi"].iloc[t])),
                       cg_long_hl=float(half_life(d["CG_rho"].iloc[t])),
                       realized_halfway=k_half)
            # predicted halfway time of the same excess, from each model's own path
            for spec in ("GARCH", "GJR-t", "CGARCH"):
                pth = np.array([model_path(d.iloc[[t]], spec, k)[0] for k in range(1, KMAX + 1)])
                relm = (pth - base[t]) / exc0
                row[f"pred_halfway_{spec}"] = int(np.argmax(relm <= 0.5)) + 1 if np.any(relm <= 0.5) else np.inf
                paths.append(pd.DataFrame(dict(index=c, date=date.date(), spec=spec, k=np.arange(1, KMAX + 1),
                                               pred=pth / base[t], real=sm / base[t])))
            events.append(row)
            for h in H:
                y = d[f"RV|{h}"].iloc[t]
                for m in MODEL_ORDER:
                    f = d[f"{m}|{h}"].iloc[t]
                    errs.append(dict(index=c, date=date.date(), h=h, model=m, y=y, f=f, log_ratio=np.log(y / f),
                                     qlike=y / f - np.log(y / f) - 1))
    ev = pd.DataFrame(events)
    er = pd.DataFrame(errs)
    ev.to_csv(RES / "events.csv", index=False, float_format="%.5g")
    pd.concat(paths).to_csv(RES / "event_paths.csv.gz", index=False, float_format="%.5g")

    # cluster bootstrap by calendar month
    er["cluster"] = pd.to_datetime(er["date"]).dt.to_period("M").astype(str)
    clusters = er["cluster"].unique()
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(clusters), size=(NBOOT, len(clusters)))
    summ = []
    for (h, m), g in er.groupby(["h", "model"]):
        cm = g.groupby("cluster")["log_ratio"].agg(["sum", "count"]).reindex(clusters).fillna(0)
        s, n = cm["sum"].values, cm["count"].values
        boot = s[draws].sum(1) / n[draws].sum(1)
        # ratio of sums: unbiased under a correct model (the geometric mean is pulled below 1 by Jensen)
        cy = g.groupby("cluster")["y"].sum().reindex(clusters).fillna(0).values
        cf = g.groupby("cluster")["f"].sum().reindex(clusters).fillna(0).values
        boot_rs = cy[draws].sum(1) / cf[draws].sum(1)
        summ.append(dict(h=h, model=m, n_events=len(g), n_clusters=int((n > 0).sum()),
                         mean_log_ratio=g.log_ratio.mean(), lo=np.quantile(boot, 0.05), hi=np.quantile(boot, 0.95),
                         realized_over_forecast=np.exp(g.log_ratio.mean()),
                         share_over_forecast=np.mean(g.log_ratio < 0), mean_qlike=g.qlike.mean(),
                         ratio_of_sums=g.y.sum() / g.f.sum(), rs_lo=np.quantile(boot_rs, 0.05),
                         rs_hi=np.quantile(boot_rs, 0.95), median_ratio=np.median(g.y / g.f)))
    pd.DataFrame(summ).to_csv(RES / "event_errors.csv", index=False, float_format="%.5g")
    print(f"{len(ev)} events in {ev['date'].astype(str).str[:7].nunique()} calendar months")
    print(pd.DataFrame(summ).pivot(index="h", columns="model", values="ratio_of_sums").round(3))
    print(ev[["realized_halfway", "pred_halfway_GARCH", "pred_halfway_GJR-t", "pred_halfway_CGARCH"]]
          .median())


if __name__ == "__main__":
    main()
