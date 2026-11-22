"""Run every strategy and variant. Writes results/bt_<variant>.csv.gz (one block of columns per strategy)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import portfolio as pf
from config import ERAS, SPREAD, TC, WINDOW, period

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "monthly_panel.csv", index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


def era_of(idx: pd.PeriodIndex) -> pd.Series:
    lab = pd.Series(index=idx, dtype=object)
    for k, (a, b) in ERAS.items():
        lab[(idx >= period(a)) & (idx <= period(b))] = k
    return lab


def windfall_free(df: pd.DataFrame, bonds: bool = True, stocks: bool = True) -> pd.DataFrame:
    """Remove, era by era, the average capital gain from the change in bond yields and the average equity re-rating
    (change in log price/dividend). Monthly shocks, carry, dividends and earnings growth are kept."""
    out = df.copy()
    era = era_of(df.index)
    logpd = -np.log(df["dy"])
    d_logpd = logpd.diff()
    # first month: use the dividend yield of the month before the sample (not available) -> drop it from the mean
    for k in ERAS:
        m = era == k
        if bonds:
            out.loc[m, "bond_x"] = df.loc[m, "bond_x"] - df.loc[m, "bond_price"].mean()
        if stocks:
            out.loc[m, "eq_x"] = df.loc[m, "eq_x"] - d_logpd[m].dropna().mean()
    return out


def run_set(df: pd.DataFrame, assets: list[str], strategies: dict, window=WINDOW, tc=TC, spread=SPREAD,
            cov_fn=pf.sample_cov, start=None) -> dict[str, pd.DataFrame]:
    X = df[assets].copy()
    X.columns = [a.replace("_x", "") for a in assets]
    return {name: pf.backtest(X, df["rf"], fn, window, tc, spread, cov_fn=cov_fn, start=start)
            for name, fn in strategies.items()}


def save(results: dict[str, pd.DataFrame], name: str):
    frames = []
    for s, r in results.items():
        r = r.copy()
        r.columns = [f"{s}|{c}" for c in r.columns]
        frames.append(r)
    out = pd.concat(frames, axis=1)
    out.index = out.index.astype(str)
    out.to_csv(RES / f"bt_{name}.csv.gz", float_format="%.7g")


MAIN = {
    "60/40": pf.strat_6040,
    "RP": pf.strat_rp_matched,
    "RP-unlevered": pf.strat_rp_unlevered,
    "MVO-expanding-means": pf.make_mvo_hist(None),
    "MVO-10y-means": pf.make_mvo_hist(120),
    "Switch-10y": pf.make_switch(120),
}


def main():
    df = load_panel()
    two = ["eq_x", "bond_x"]

    # 1. main universe, main settings
    save(run_set(df, two, MAIN), "main")

    # 2. windfall-free counterfactual (H3)
    pair = {"60/40": pf.strat_6040, "RP": pf.strat_rp_matched}
    save(run_set(windfall_free(df), two, pair), "windfall_free")
    save(run_set(windfall_free(df, stocks=False), two, pair), "windfall_free_bonds_only")
    save(run_set(windfall_free(df, bonds=False), two, pair), "windfall_free_stocks_only")

    # 3. robustness: estimation window, EWMA, financing spread, costs, bond data
    core = {"60/40": pf.strat_6040, "RP": pf.strat_rp_matched}
    rob = {}
    for w in (12, 24, 60, 120):
        rob[f"window {w}m"] = run_set(df, two, core, window=w)
    rob["EWMA (half-life 12m)"] = run_set(df, two, core, cov_fn=pf.ewma_cov)
    for s in (0.0, 0.005, 0.01, 0.02):
        rob[f"spread {s * 1e4:.0f} bp"] = run_set(df, two, core, spread=s)
    for c in (0.0, 0.002):
        rob[f"costs {c * 1e4:.0f} bp"] = run_set(df, two, core, tc=c)
    d_avg = df.copy()
    d_avg["bond_x"] = df["bond_x_avg"]
    rob["bonds from averaged yields only"] = run_set(d_avg, two, core)
    d_resc = df.copy()
    pre = df.index < period("1962-01")
    ratio = 0.0779 / 0.0664  # month-end vs averaged bond vol, 1962-2026 (results/data_smoothing_check.csv)
    m = df.loc[pre, "bond_x"].mean()
    d_resc.loc[pre, "bond_x"] = m + (df.loc[pre, "bond_x"] - m) * ratio
    rob["pre-1962 bond vol rescaled x1.17"] = run_set(d_resc, two, core)
    rob["leverage cap 3x"] = run_set(df, two, {"60/40": pf.strat_6040, "RP": pf.make_rp_capped(3.0)})
    rob["RP at 10% vol target"] = run_set(df, two, {"60/40": pf.strat_6040, "RP": pf.make_target_vol(0.10)})
    for k, v in rob.items():
        save(v, "rob_" + k.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "").replace(".", ""))
    pd.Series(list(rob)).to_csv(RES / "robustness_variants.csv", index=False, header=["variant"])

    # 4. three-asset universe with gold (H6), 1972 onwards
    d3 = df[df.index >= period("1972-01")]
    three = ["eq_x", "bond_x", "gold_x"]
    save(run_set(d3, three, {"60/40": pf.strat_6040, "RP": pf.strat_rp_matched,
                             "IV-3 (stocks, bonds, gold)": pf.strat_iv_all_matched,
                             "ERC-3 (stocks, bonds, gold)": pf.strat_erc_all_matched}), "gold")
    print("done")


if __name__ == "__main__":
    main()
