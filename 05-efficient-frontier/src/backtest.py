"""Walk-forward backtests: the horse race and the 5x3 information factorial.

At every rebalance date t the estimation window is x[t-W:t]; the holding period is x[t:t+step].
"Clairvoyant" inputs use x[t:t+H] (the future, including the holding period) and are the only
place where look-ahead is used, on purpose, to measure the value of perfect information.
"Full-sample" means use the 1949-2017 (A) or 1990-2022 (B) average: what a perfectly informed
investor would know if expected returns were constant.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import portfolio as pf
from config import COST_BPS, COV_LEVELS, MEAN_LEVELS, RESAMPLE_DRAWS, SEED, UNIVERSES
from data import load

RES = Path(__file__).resolve().parents[1] / "results"
RES.mkdir(exist_ok=True)


def hold(r: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Buy-and-hold w over the rows of r. Returns per-period portfolio returns and end weights."""
    growth = np.cumprod(1 + r, axis=0)
    value = growth @ w
    prev = np.r_[1.0, value[:-1]]
    end = w * growth[-1]
    return value / prev - 1, end / end.sum()


def weights_at(x_est, x_fut, full_mean, rng, draws, cells: bool, short: float = 0.0):
    m_s, s_s = pf.sample_mean(x_est), pf.sample_cov(x_est)
    s_lw = pf.ledoit_wolf(x_est)
    m_bs = pf.bayes_stein_mean(x_est)
    n = x_est.shape[1]
    out = {
        "1/N": pf.equal_weight(n),
        "Inverse vol": pf.inverse_vol(s_s),
        "Min-var (sample)": pf.min_variance(s_s),
        "Min-var (LW)": pf.min_variance(s_lw),
        "Max-Sharpe (sample)": pf.max_sharpe(m_s, s_s),
        "Max-Sharpe (BS + LW)": pf.max_sharpe(m_bs, s_lw),
    }
    if draws:
        out["Max-Sharpe (resampled)"] = pf.resampled_max_sharpe(m_s, s_s, len(x_est), draws, rng)
    promised = pf.sharpe(m_s, s_s, out["Max-Sharpe (sample)"])
    if cells and x_fut is not None:
        means = {"None": np.zeros(n), "Sample": m_s, "Bayes-Stein": m_bs, "Full-sample": full_mean,
                 "Clairvoyant": x_fut.mean(0)}
        covs = {"Sample": s_s, "Ledoit-Wolf": s_lw, "Clairvoyant": pf.sample_cov(x_fut)}
        for a in MEAN_LEVELS:
            for b in COV_LEVELS:
                out[f"cell|{a}|{b}"] = (pf.max_sharpe_short(means[a], covs[b], short) if short
                                        else pf.max_sharpe(means[a], covs[b]))
    return out, promised


def run(universe: str, window: int | None = None, horizon: int | None = None, draws=RESAMPLE_DRAWS,
        cells=True, x: pd.DataFrame | None = None, seed=SEED, short: float = 0.0, cfg: dict | None = None):
    cfg = cfg or UNIVERSES[universe]
    W = window or cfg["window"]
    H = horizon or cfg["horizon"]
    step = cfg["step"]
    df = load(universe) if x is None else x
    X = df.to_numpy()
    T, n = X.shape
    full_mean = X.mean(0)
    rng = np.random.default_rng(seed)
    cost = COST_BPS / 1e4

    rets, turns, effn, promised, wlog = {}, {}, {}, [], []
    prev_end: dict[str, np.ndarray] = {}
    dates_reb = []
    for t in range(W, T, step):
        x_est = X[t - W:t]
        x_fut = X[t:t + H] if t + H <= T else None
        ws, prom = weights_at(x_est, x_fut, full_mean, rng, draws, cells, short)
        r_hold = X[t:t + step]
        dates_reb.append(df.index[t])
        promised.append(prom)
        for k, w in ws.items():
            pr, end = hold(r_hold, w)
            to = np.abs(w - prev_end[k]).sum() if k in prev_end else 1.0
            prev_end[k] = end
            gross = pr.copy()
            net = pr.copy()
            net[0] -= cost * to
            rets.setdefault(k, {"gross": [], "net": [], "idx": []})
            rets[k]["gross"].append(gross)
            rets[k]["net"].append(net)
            rets[k]["idx"].append(np.arange(t, t + len(pr)))
            turns.setdefault(k, []).append((df.index[t], to))
            effn.setdefault(k, []).append((df.index[t], 1.0 / np.sum(w ** 2)))
            if k in ("Max-Sharpe (sample)", "Max-Sharpe (BS + LW)", "Max-Sharpe (resampled)",
                     "Min-var (sample)", "1/N"):
                wlog.append(dict(date=df.index[t], strategy=k, **dict(zip(df.columns, w))))

    def frame(kind):
        cols = {}
        for k, v in rets.items():
            idx = np.concatenate(v["idx"])
            cols[k] = pd.Series(np.concatenate(v[kind]), index=df.index[idx])
        return pd.DataFrame(cols)

    gross, net = frame("gross"), frame("net")
    turnover = pd.DataFrame({k: pd.Series(dict(v)) for k, v in turns.items()})
    eff = pd.DataFrame({k: pd.Series(dict(v)) for k, v in effn.items()})
    prom = pd.Series(promised, index=dates_reb, name="promised_sharpe_per_period")
    return dict(gross=gross, net=net, turnover=turnover, promised=prom, weights=pd.DataFrame(wlog), effective_n=eff,
                W=W, H=H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip robustness runs")
    args = ap.parse_args()
    for u, cfg in UNIVERSES.items():
        t0 = time.time()
        out = run(u)
        out["gross"].to_csv(RES / f"{u}_returns_gross.csv.gz", float_format="%.7g")
        out["net"].to_csv(RES / f"{u}_returns_net.csv.gz", float_format="%.7g")
        out["turnover"].to_csv(RES / f"{u}_turnover.csv.gz", float_format="%.6g")
        out["promised"].to_csv(RES / f"{u}_promised.csv")
        out["effective_n"].to_csv(RES / f"{u}_effective_n.csv.gz", float_format="%.5g")
        out["weights"].to_csv(RES / f"{u}_weights.csv.gz", index=False, float_format="%.5g")
        print(f"[{u}] main run {time.time() - t0:.0f}s, {out['gross'].shape}")
        if args.quick:
            continue
        for W in cfg["robust_windows"]:
            o = run(u, window=W, draws=RESAMPLE_DRAWS)
            o["gross"].to_csv(RES / f"{u}_returns_gross_W{W}.csv.gz", float_format="%.7g")
            o["net"].to_csv(RES / f"{u}_returns_net_W{W}.csv.gz", float_format="%.7g")
            print(f"[{u}] W={W} done {time.time() - t0:.0f}s")
        for H in cfg["robust_horizons"]:
            o = run(u, horizon=H, draws=0)  # noqa: E501
            o["gross"].to_csv(RES / f"{u}_returns_gross_H{H}.csv.gz", float_format="%.7g")
            print(f"[{u}] H={H} done {time.time() - t0:.0f}s")
        o = run(u, draws=0, short=0.3)
        o["gross"].to_csv(RES / f"{u}_returns_gross_short.csv.gz", float_format="%.7g")
        print(f"[{u}] shorts allowed done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
