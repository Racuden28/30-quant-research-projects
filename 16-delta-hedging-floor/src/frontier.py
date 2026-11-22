"""Experiment 1: the rebalancing frontier and its floor, on real paths.

For every trading day t and every underlying, sell a one-month at-the-money call at a quoted volatility
known at t, and delta-hedge it every h trading days for h in H_GRID. Two quotes:

* forecast : the EWMA volatility known at the close of t (VIX for the S&P 500 from 1990, in vix_*),
             i.e. what a hedger could actually have used;
* oracle   : the realised volatility of the next 21 daily returns (ex post). Not a strategy: it removes
             the level error of the forecast, so whatever floor is left comes from the shape of the path.

The variance of the P&L across start dates is fitted by the linear frequency law Var(h) = a + b h on
h = 1..5. Under GBM with a correct quote, a = 0 and Var(1) = b. The floor share a / (a + b) is the part
of the daily-hedging variance (h is replaced by the effective interval T / ceil(T / h)) that would remain if the law held all the way to continuous hedging.

Writes results/frontier_paths.csv.gz (one row per option) and results/frontier_summary.csv,
results/floor_shares.csv, results/stress_windows.csv, results/robustness_floor.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import H_FIT, H_GRID, RES, T_DAYS, universe
from hedgesim import every_h_days, floor_fit, hedge_short_call, windows

B_BOOT = 2000
BLOCK = 126
SEED = 16

STRESS = {
    "1929 crash (Sep-Dec 1929)": ("1929-09-01", "1929-12-31"),
    "1987 crash (Sep-Nov 1987)": ("1987-09-01", "1987-11-30"),
    "GFC (Sep-Dec 2008)": ("2008-09-01", "2008-12-31"),
    "COVID (Feb-Apr 2020)": ("2020-02-01", "2020-04-30"),
    "2022 bear (Jan-Oct 2022)": ("2022-01-01", "2022-10-31"),
}


def realised_vol_windows(r: np.ndarray, T: int) -> np.ndarray:
    """sqrt(252 * mean r^2) over the T returns that follow each start index."""
    W = np.lib.stride_tricks.sliding_window_view(r, T)
    return np.sqrt(252.0 * (W ** 2).mean(1))


def run_one(close: np.ndarray, sigma_q: np.ndarray, T: int = T_DAYS, h_grid=H_GRID, x_strike: float = 0.0,
            cost_bps: float = 0.0) -> np.ndarray:
    """P&L matrix (n_starts, len(h_grid)) in vol points for given paths and quotes."""
    P, _ = windows(close, T)
    sigma_q = np.asarray(sigma_q, float)[: len(P)]
    K = 100.0 * np.exp(x_strike * sigma_q * np.sqrt(T / 252.0))
    out = np.empty((len(P), len(h_grid)))
    for j, h in enumerate(h_grid):
        out[:, j] = hedge_short_call(P, K, sigma_q, schedule=every_h_days(T, h), cost_bps=cost_bps).pnl
    return out


def build_paths(T: int = T_DAYS) -> pd.DataFrame:
    rows = []
    for label, d in universe().items():
        close = d.close.to_numpy()
        r = np.log(close[1:] / close[:-1])
        n = len(close) - T
        dates = d.index[:n]
        rv = realised_vol_windows(r, T)[:n]
        quotes = {"forecast": d.ewma.to_numpy()[:n], "oracle": rv}
        if label == "SPX":
            quotes["vix"] = d.vix.to_numpy()[:n]
        for qname, q in quotes.items():
            ok = np.isfinite(q) & (q > 0)
            pnl = np.full((n, len(H_GRID)), np.nan)
            idx = np.flatnonzero(ok)
            P, _ = windows(close, T)
            for j, h in enumerate(H_GRID):
                pnl[idx, j] = hedge_short_call(P[idx], 100.0, q[idx], schedule=every_h_days(T, h)).pnl
            df = pd.DataFrame(pnl[idx], columns=[f"h{h}" for h in H_GRID])
            df.insert(0, "date", dates[idx])
            df.insert(0, "quote", qname)
            df.insert(0, "asset", label)
            df["sigma_q"] = q[idx]
            df["rv"] = rv[idx]
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def block_indices(n: int, rng: np.random.Generator, block: int = BLOCK) -> np.ndarray:
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=k)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def floor_stats(M: np.ndarray, h_grid=H_GRID, h_fit=H_FIT, T: int = T_DAYS) -> dict:
    """Floor share and related numbers from a P&L matrix (rows: options, cols: h_grid)."""
    var = M.var(0, ddof=1)
    cols = [h_grid.index(h) for h in h_fit]
    # effective interval T / N: with N = ceil(T / h) rebalances the textbook variance is exactly
    # proportional to T / N, which is what makes a = 0 under GBM (h = 4 and h = 5 give 6 and 5 hedges)
    a, b = floor_fit(np.array([T / np.ceil(T / h) for h in h_fit]), var[cols])
    v1 = var[h_grid.index(1)]
    return dict(a=a, b=b, share=a / (a + b), var1=v1,
                std1=np.sqrt(v1), std5=np.sqrt(var[h_grid.index(5)]),
                std21=np.sqrt(var[h_grid.index(21)]),
                gain_5_to_1=1 - v1 / var[h_grid.index(5)])


def boot_floor(M: np.ndarray, rng: np.random.Generator, B: int = B_BOOT) -> np.ndarray:
    out = np.empty((B, 3))
    n = len(M)
    for bb in range(B):
        s = floor_stats(M[block_indices(n, rng)])
        out[bb] = s["share"], s["std1"], s["gain_5_to_1"]
    return out


def excess_kurtosis_standardised(close: np.ndarray, T: int = T_DAYS) -> float:
    """Pooled excess kurtosis of daily returns divided by their own month's realised vol.

    Uses non-overlapping 21-day blocks. This is the ek in the iid prediction share = ek / (2 + ek).
    """
    r = np.log(close[1:] / close[:-1])
    m = len(r) // T
    R = r[: m * T].reshape(m, T)
    s = np.sqrt((R ** 2).mean(1, keepdims=True))
    z = (R / s)[s[:, 0] > 0].ravel()
    return float((z ** 4).mean() / (z ** 2).mean() ** 2 - 3.0)


def raw_excess_kurtosis(close: np.ndarray) -> float:
    r = np.log(close[1:] / close[:-1])
    r = r - r.mean()
    return float((r ** 4).mean() / (r ** 2).mean() ** 2 - 3.0)


def main():
    rng = np.random.default_rng(SEED)
    paths = build_paths()
    paths.to_csv(RES / "frontier_paths.csv.gz", index=False, float_format="%.5g")
    hcols = [f"h{h}" for h in H_GRID]
    U = universe()

    # ---------- frontier summary: std by h ----------
    summ = (paths.groupby(["asset", "quote"])[hcols].agg(lambda x: x.std(ddof=1)).reset_index())
    summ.insert(2, "n_options", paths.groupby(["asset", "quote"]).size().values)
    summ.to_csv(RES / "frontier_summary.csv", index=False, float_format="%.4f")

    # ---------- floor shares with block-bootstrap CIs ----------
    rows = []
    for (a, q), g in paths.groupby(["asset", "quote"], sort=False):
        M = g[hcols].to_numpy()
        s = floor_stats(M)
        bt = boot_floor(M, rng)
        lo, hi = np.percentile(bt, [2.5, 97.5], axis=0)
        close = U[a].close.to_numpy()
        rows.append(dict(asset=a, quote=q, start=g.date.min().date(), end=g.date.max().date(), n=len(g),
                         std1=s["std1"], std5=s["std5"], std21=s["std21"],
                         share=s["share"], share_lo=lo[0], share_hi=hi[0],
                         gain_5_to_1=s["gain_5_to_1"], gain_lo=lo[2], gain_hi=hi[2],
                         # Bonferroni bounds over the 8 underlyings (one-sided 0.05 / 8), stricter than Holm
                         share_lo_bonf=np.percentile(bt[:, 0], 100 * 0.05 / 8),
                         gain_hi_bonf=np.percentile(bt[:, 2], 100 - 100 * 0.05 / 8),
                         p_floor_le_0=max(np.mean(bt[:, 0] <= 0), 1 / B_BOOT),
                         ek_std=excess_kurtosis_standardised(close), ek_raw=raw_excess_kurtosis(close),
                         mean1=g["h1"].mean()))
    fl = pd.DataFrame(rows)
    fl["pred_share_iid"] = fl.ek_std / (2 + fl.ek_std)
    fl.to_csv(RES / "floor_shares.csv", index=False, float_format="%.4f")

    # ---------- SPX by decade (oracle + forecast) ----------
    rows = []
    spx = paths[paths.asset == "SPX"].copy()
    spx["decade"] = (spx.date.dt.year // 10) * 10
    close = U["SPX"]
    for (dec, q), g in spx.groupby(["decade", "quote"]):
        if len(g) < 500:
            continue
        s = floor_stats(g[hcols].to_numpy())
        seg = close.loc[str(dec):str(dec + 9)].close.to_numpy()
        ek = excess_kurtosis_standardised(seg)
        rows.append(dict(decade=dec, quote=q, n=len(g), share=s["share"], std1=s["std1"],
                         gain_5_to_1=s["gain_5_to_1"], ek_std=ek, pred_share_iid=ek / (2 + ek)))
    pd.DataFrame(rows).to_csv(RES / "spx_by_decade.csv", index=False, float_format="%.4f")

    # ---------- stress windows ----------
    rows = []
    for name, (a0, a1) in STRESS.items():
        for asset in ["SPX"] + [x for x in U if x != "SPX"]:
            for q in ["forecast", "oracle", "vix"]:
                g = paths[(paths.asset == asset) & (paths.quote == q) & (paths.date >= a0) & (paths.date <= a1)]
                if len(g) < 20:
                    continue
                std = g[hcols].std(ddof=1)
                rows.append(dict(window=name, asset=asset, quote=q, n=len(g),
                                 **{f"std_h{h}": std[f"h{h}"] for h in H_GRID},
                                 mean_h1=g.h1.mean(),
                                 gain_5_to_1=1 - std["h1"] ** 2 / std["h5"] ** 2))
    pd.DataFrame(rows).to_csv(RES / "stress_windows.csv", index=False, float_format="%.4f")

    # ---------- robustness: strikes and maturity (SPX forecast and oracle, SPY forecast) ----------
    rows = []
    for asset in ["SPX", "SPY", "USO", "TLT"]:
        d = U[asset]
        close = d.close.to_numpy()
        r = np.log(close[1:] / close[:-1])
        for T, hg in [(21, H_GRID), (63, [1, 2, 3, 4, 5, 10, 21, 63])]:
            n = len(close) - T
            rv = realised_vol_windows(r, T)[:n]
            for q, sig in [("forecast", d.ewma.to_numpy()[:n]), ("oracle", rv)]:
                ok = np.isfinite(sig)
                for x in [-1.0, 0.0, 1.0]:
                    P, _ = windows(close, T)
                    P, s_ = P[ok], sig[ok]
                    K = 100.0 * np.exp(x * s_ * np.sqrt(T / 252.0))
                    M = np.column_stack([hedge_short_call(P, K, s_, schedule=every_h_days(T, h)).pnl for h in hg])
                    st = floor_stats(M, h_grid=hg, T=T)
                    rows.append(dict(asset=asset, T=T, quote=q, strike_sd=x, n=len(M), share=st["share"],
                                     std1=st["std1"], gain_5_to_1=st["gain_5_to_1"]))
    pd.DataFrame(rows).to_csv(RES / "robustness_floor.csv", index=False, float_format="%.4f")
    print(fl[["asset", "quote", "n", "std1", "std5", "share", "share_lo", "share_hi", "gain_5_to_1",
              "ek_std", "pred_share_iid"]].round(3).to_string())


if __name__ == "__main__":
    main()
