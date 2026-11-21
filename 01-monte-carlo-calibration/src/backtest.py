"""
Rolling, strictly out-of-sample forecast backtest.

At every weekly forecast origin t (close of day t-1):
  * the estimation window is the last <= 1,260 trading days (~5 years) of daily log returns,
    with a minimum of 500 days before the first forecast;
  * model parameters are re-estimated every `REFIT_EVERY` origins (~monthly); between refits
    the conditional models still filter their latent state (GARCH variance, regime
    probabilities) through the newest data, so no forecast ever uses information after t-1;
  * each simulator draws N_PATHS paths of cumulative log returns for h = 5 and h = 21 days;
  * we record the probability integral transform (PIT) of the realised outcome, a set of
    predictive quantiles, the CRPS and the predictive standard deviation.

An equal-weight linear pool of the five simulators ("Pool") is evaluated as a sixth forecaster.
"""
from __future__ import annotations

import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import CONTROLS, MODELS, Merton  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "etf_daily_log_returns.csv"
OUT = ROOT / "results" / "forecasts.csv.gz"

H = (5, 21)
WINDOW, MIN_WINDOW, STEP, REFIT_EVERY = 1260, 500, 5, 4
N_PATHS = 10_000
QLEVELS = np.array([0.005, 0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.99, 0.995])


def crps_from_samples(x: np.ndarray, y: float) -> float:
    """CRPS(F, y) = E|X - y| - 0.5 E|X - X'| with the O(n log n) order-statistic identity."""
    x = np.sort(x)
    n = len(x)
    i = np.arange(1, n + 1)
    e_xy = np.mean(np.abs(x - y))
    e_xx = 2.0 * np.sum((2 * i - n - 1) * x) / n**2
    return e_xy - 0.5 * e_xx


def pit(x: np.ndarray, y: float, u: float) -> float:
    """Randomised PIT (handles ties for the discrete bootstrap)."""
    below = np.mean(x < y)
    equal = np.mean(x == y)
    return below + u * equal


def run_asset(asset: str, n_paths: int = N_PATHS, max_origins: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    s = df[asset].dropna()
    r, dates = s.values, s.index
    T = len(r)
    origins = list(range(MIN_WINDOW, T - max(H) + 1, STEP))
    if max_origins:
        origins = origins[:max_origins]

    # trailing 21-day realised volatility at each origin (annualised), used for conditional analysis
    rv = pd.Series(r).rolling(21).std().values * np.sqrt(252)

    rows, params, merton_x0 = [], {}, None
    t0 = time.time()
    for k, t in enumerate(origins):
        win = r[max(0, t - WINDOW):t]
        if k % REFIT_EVERY == 0:
            for M in MODELS + CONTROLS:
                if M is Merton:
                    params[M.name] = M.fit(win, x0=merton_x0)
                    merton_x0 = params[M.name]["x"]
                else:
                    params[M.name] = M.fit(win)
        rng = np.random.default_rng([zlib.crc32(asset.encode()), t])
        u = rng.random()
        realised = [r[t:t + h].sum() for h in H]
        sims = {M.name: M.simulate(params[M.name], win, H, n_paths, rng) for M in MODELS}
        sims["Pool"] = np.vstack(list(sims.values()))
        sims.update({M.name: M.simulate(params[M.name], win, H, n_paths, rng) for M in CONTROLS})
        hist_rv = rv[MIN_WINDOW - 1:t]  # expanding history of the vol state, no look-ahead
        rv_pct = float(np.mean(hist_rv[~np.isnan(hist_rv)] <= rv[t - 1]))
        for name, X in sims.items():
            for j, h in enumerate(H):
                x, y = X[:, j], realised[j]
                q = np.quantile(x, QLEVELS)
                rows.append({
                    "asset": asset, "origin": dates[t], "t": t, "model": name, "h": h,
                    "y": y, "pit": pit(x, y, u), "crps": crps_from_samples(x, y),
                    "sd": float(np.std(x)), "mean": float(np.mean(x)),
                    "rv21": rv[t - 1], "rv_pct": rv_pct,
                    **{f"q{lv:.3f}": v for lv, v in zip(QLEVELS, q)},
                })
        if k % 100 == 0:
            print(f"[{asset}] origin {k + 1}/{len(origins)} {dates[t].date()} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main():
    assets = pd.read_csv(DATA, index_col=0, nrows=1).columns.tolist()
    with Pool(min(4, len(assets))) as pool:
        frames = pool.map(run_asset, assets)
    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, float_format="%.6g", compression="gzip")
    print("saved", OUT, out.shape)


if __name__ == "__main__":
    main()
