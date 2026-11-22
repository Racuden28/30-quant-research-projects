"""Walk-forward backtest: refit every January, forecast cumulative variance at 5 horizons every day.

Models
  EWMA      RiskMetrics lambda = 0.94 (no mean reversion: a shock never decays)
  GARCH     GARCH(1,1)-normal, expanding window
  GARCH-R10 GARCH(1,1)-normal, rolling 10-year window (the parameter-instability ablation)
  GJR-t     GJR-GARCH(1,1)-t, expanding window (project 01's model)
  CGARCH    Engle-Lee component GARCH, expanding window (two decay speeds)
  HAR       log-HAR direct projection per horizon (Corsi 2009 style, on daily squared returns)

All parameters at an origin t come from data that ended before the refit date <= t. The variance
state used at t is filtered through day t only. Output: results/forecasts/<index>.pkl.gz
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import INDICES, ROOT, load_returns  # noqa: E402
from models import CGARCH, EWMA, GARCH, GJRt, start_var  # noqa: E402

H = [5, 22, 66, 126, 252]
BURN = 2520          # 10 years of daily data before the first forecast
ROLL = 2520
OUT = ROOT / "results" / "forecasts"
MODEL_ORDER = ["EWMA", "GARCH", "GARCH-R10", "GJR-t", "CGARCH", "HAR"]


def trailing_mean(x, w):
    c = np.concatenate(([0.0], np.cumsum(x)))
    out = np.full(len(x), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def forward_sum(x, h):
    """y[t] = x[t+1] + ... + x[t+h] (NaN where not fully observed)."""
    c = np.concatenate(([0.0], np.cumsum(x)))
    n = len(x)
    out = np.full(n, np.nan)
    idx = np.arange(n - h)
    out[idx] = c[idx + h + 1] - c[idx + 1]
    return out


def har_features(r2):
    f = [trailing_mean(r2, w) for w in (5, 22, 66, 252)]
    return np.log(np.maximum(np.column_stack(f), 1e-4))


def refit_dates(idx: pd.DatetimeIndex):
    """Positions of the first trading day of each calendar year after the burn-in."""
    years = idx.year
    first = np.r_[0, np.where(np.diff(years) != 0)[0] + 1]
    return [int(p) for p in first if p >= BURN]


def run_index(code: str):
    r_s = load_returns()[code]
    r = r_s.values
    r2 = r * r
    n = len(r)
    dates = r_s.index
    s0 = start_var(r)
    refits = refit_dates(dates) + [n]
    X = har_features(r2)
    Y = {h: forward_sum(r2, h) for h in H}

    first = refits[0]
    rows = np.arange(first, n)
    fc = {f"{m}|{h}": np.full(n, np.nan) for m in MODEL_ORDER for h in H}
    state = {k: np.full(n, np.nan) for k in
             ["GARCH_s", "GARCH_vbar", "GARCH_phi", "R10_s", "R10_vbar", "R10_phi",
              "GJR_s", "GJR_vbar", "GJR_phi", "CG_s", "CG_q", "CG_w", "CG_rho", "CG_psi", "CG_phi", "CG_a",
              "EWMA_s"]}
    params = []
    x0 = {"GARCH": None, "R10": None, "GJR-t": None, "CGARCH": None}
    st_e = EWMA.filter(r, [EWMA.LAM], s0)

    for a, b in zip(refits[:-1], refits[1:]):
        seg = slice(a - 1, b - 1)      # origins t = a-1 .. b-2 use info to t (refit uses data < a)
        tr = r[:a]
        pg, llg = GARCH.fit(tr, x0["GARCH"]); x0["GARCH"] = pg
        pr, llr = GARCH.fit(tr[-ROLL:], x0["R10"]); x0["R10"] = pr
        pj, llj = GJRt.fit(tr, x0["GJR-t"]); x0["GJR-t"] = pj
        pc, llc = CGARCH.fit(tr, x0["CGARCH"]); x0["CGARCH"] = pc
        sg = GARCH.filter(r, pg, s0)
        sr = GARCH.filter(r, pr, s0)
        sj = GJRt.filter(r, pj, s0)
        sc = CGARCH.filter(r, pc, s0)
        for h in H:
            fc[f"EWMA|{h}"][seg] = EWMA.cum_forecast(st_e, None, h)[seg]
            fc[f"GARCH|{h}"][seg] = GARCH.cum_forecast(sg, pg, h)[seg]
            fc[f"GARCH-R10|{h}"][seg] = GARCH.cum_forecast(sr, pr, h)[seg]
            fc[f"GJR-t|{h}"][seg] = GJRt.cum_forecast(sj, pj, h)[seg]
            fc[f"CGARCH|{h}"][seg] = CGARCH.cum_forecast(sc, pc, h)[seg]
            # HAR: train on origins whose whole target window ended before the refit date
            ok = np.arange(251, a - 1 - h)
            y = np.log(np.maximum(Y[h][ok] / h, 1e-4))
            Z = np.column_stack([np.ones(len(ok)), X[ok]])
            beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
            smear = np.mean(np.exp(y - Z @ beta))
            Zf = np.column_stack([np.ones(b - a), X[a - 1:b - 1]])
            fc[f"HAR|{h}"][seg] = h * smear * np.exp(Zf @ beta)
        state["EWMA_s"][seg] = st_e["s"][seg]
        state["GARCH_s"][seg] = sg["s"][seg]
        state["GARCH_vbar"][seg] = GARCH.long_run(pg)
        state["GARCH_phi"][seg] = GARCH.persistence(pg)
        state["R10_s"][seg] = sr["s"][seg]
        state["R10_vbar"][seg] = GARCH.long_run(pr)
        state["R10_phi"][seg] = GARCH.persistence(pr)
        state["GJR_s"][seg] = sj["s"][seg]
        state["GJR_vbar"][seg] = GJRt.long_run(pj)
        state["GJR_phi"][seg] = GJRt.persistence(pj)
        state["CG_s"][seg] = sc["s"][seg]
        state["CG_q"][seg] = sc["q"][seg]
        state["CG_w"][seg] = pc[0]
        state["CG_rho"][seg] = pc[1]
        state["CG_psi"][seg] = pc[3] + pc[4]
        state["CG_phi"][seg] = pc[2]
        state["CG_a"][seg] = pc[3]
        params.append(dict(index=code, refit=dates[a].date(), n_train=a,
                           **{f"GARCH_{k}": v for k, v in zip(["omega", "alpha", "beta"], pg)},
                           **{f"R10_{k}": v for k, v in zip(["omega", "alpha", "beta"], pr)},
                           **{f"GJR_{k}": v for k, v in zip(["omega", "alpha", "gamma", "beta", "nu"], pj)},
                           **{f"CG_{k}": v for k, v in zip(["omega", "rho", "phi", "alpha", "beta"], pc)},
                           ll_GARCH=llg, ll_GJR=llj, ll_CG=llc))

    keep = slice(first - 1, n)
    df = pd.DataFrame({**fc, **state}, index=dates)
    for h in H:
        df[f"RV|{h}"] = Y[h]
    df["r"] = r
    df = df.iloc[keep]
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUT / f"{code}.pkl.gz")
    print(f"{code}: {len(df)} origins, {len(params)} refits", flush=True)
    return pd.DataFrame(params)


def main():
    with Pool(4) as pool:
        tabs = pool.map(run_index, INDICES)
    pd.concat(tabs).to_csv(ROOT / "results" / "refit_params.csv", index=False)


if __name__ == "__main__":
    main()
