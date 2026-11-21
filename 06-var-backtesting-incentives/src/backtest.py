"""Run every forecaster on every series, strictly out of sample. Writes results/forecasts.csv.gz."""
from __future__ import annotations

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from models import MODEL_ORDER, WINDOW, rolling_forecasts

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
RES.mkdir(exist_ok=True)

# Multi-asset book: a fixed-weight, daily-rebalanced 40/10/35/15 equity/EM/duration/gold portfolio
PORT_W = {"SPY": 0.40, "EEM": 0.10, "TLT": 0.35, "GLD": 0.15}
SERIES = ["SPY", "EEM", "TLT", "HYG", "GLD", "USO", "URA", "BOOK"]


def load_returns() -> dict[str, pd.Series]:
    d = pd.read_csv(DATA / "etf_daily_log_returns.csv", index_col=0, parse_dates=True)
    out = {c: d[c].dropna() for c in d.columns}
    sub = d[list(PORT_W)].dropna()
    out["BOOK"] = np.log((np.exp(sub) * pd.Series(PORT_W)).sum(axis=1))
    return {k: out[k] for k in SERIES}


def run_one(args):
    name, s = args
    t0 = time.time()
    f = rolling_forecasts(s.values, WINDOW)
    idx = s.index[WINDOW:]
    frames = []
    for m in MODEL_ORDER:
        frames.append(pd.DataFrame({"date": idx, "series": name, "model": m, "ret": s.values[WINDOW:],
                                    "var99": f[m][:, 0], "var975": f[m][:, 1],
                                    "es975": f[m][:, 2]}))
    print(f"{name}: {len(idx)} days, {time.time() - t0:.0f}s", flush=True)
    return pd.concat(frames)


if __name__ == "__main__":
    rets = load_returns()
    with Pool(4) as p:
        out = p.map(run_one, list(rets.items()))
    df = pd.concat(out, ignore_index=True)
    df.to_csv(RES / "forecasts.csv.gz", index=False, float_format="%.6f", compression="gzip")
    print(df.groupby("model")[["var99", "var975", "es975"]].describe().round(4))
