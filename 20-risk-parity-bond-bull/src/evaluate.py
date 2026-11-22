"""Every table in the paper. Run after data.py and backtest.py. Writes results/*.csv."""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import portfolio as pf
import stats as st
from backtest import era_of, load_panel, windfall_free
from config import BLOCK, ERAS, N_BOOT, SEED, STRESS, period

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load_bt(name: str) -> pd.DataFrame:
    d = pd.read_csv(RES / f"bt_{name}.csv.gz", index_col=0)
    d.index = pd.PeriodIndex(d.index, freq="M")
    return d


def strategies(d: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(c.split("|")[0] for c in d.columns))


def segments(idx: pd.PeriodIndex) -> dict[str, np.ndarray]:
    e = era_of(idx)
    seg = {"Full sample": np.ones(len(idx), bool)}
    for k in ERAS:
        seg[k] = (e == k).values
    return seg


def perf_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, m in segments(d.index).items():
        for s in strategies(d):
            x = d.loc[m, f"{s}|xret"].values
            r = d.loc[m, f"{s}|ret"].values
            rows.append(dict(segment=name, strategy=s, start=str(d.index[m][0]), end=str(d.index[m][-1]), months=m.sum(),
                             ann_excess=x.mean() * 12, ann_vol=x.std(ddof=1) * np.sqrt(12), sharpe=st.sharpe(x),
                             cagr_total=np.prod(1 + r) ** (12 / len(r)) - 1, max_dd_total=st.max_drawdown(r),
                             avg_leverage=d.loc[m, f"{s}|lev"].mean(),
                             turnover_pa=d.loc[m, f"{s}|turnover"].mean() * 12))
    return pd.DataFrame(rows)


def diff_tests(d: pd.DataFrame, pairs: list[tuple[str, str]], seed: int) -> pd.DataFrame:
    rows = []
    for name, m in segments(d.index).items():
        for k, (a, b) in enumerate(pairs):
            t = st.paired_sharpe_test(d.loc[m, f"{a}|xret"].values, d.loc[m, f"{b}|xret"].values, N_BOOT, BLOCK,
                                      seed + 17 * k + zlib.crc32(name.encode()) % 1000)
            rows.append(dict(segment=name, a=a, b=b, months=m.sum(), **t))
    return pd.DataFrame(rows)


def hurdle_table(panel: pd.DataFrame, idx: pd.PeriodIndex, d: pd.DataFrame) -> pd.DataFrame:
    """For each era: stock and bond Sharpe, correlation, the 60/40 risk ratio a, the break-even ratio r*, and the
    static-formula prediction of the Sharpe gap against the realised (dynamic) backtest gap."""
    rows = []
    P = panel.loc[idx]
    for name, m in segments(idx).items():
        x = P.loc[m, ["eq_x", "bond_x"]]
        ss, sb = st.sharpe(x.eq_x), st.sharpe(x.bond_x)
        vs, vb = x.eq_x.std(), x.bond_x.std()
        rho = x.corr().iloc[0, 1]
        a = 0.4 * vb / (0.6 * vs)
        rstar = pf.hurdle(rho, a)
        r = sb / ss
        A, B = np.sqrt(2 + 2 * rho), np.sqrt(1 + a * a + 2 * rho * a)
        pred = (ss + sb) / A - (ss + a * sb) / B
        real = st.sharpe(d.loc[m, "RP|xret"]) - st.sharpe(d.loc[m, "60/40|xret"])
        rows.append(dict(segment=name, sharpe_stocks=ss, sharpe_bonds=sb, ratio_r=r, corr=rho, vol_stocks=vs * np.sqrt(12),
                         vol_bonds=vb * np.sqrt(12), a_6040=a, hurdle_rstar=rstar,
                         rp_should_win=bool((ss > 0 and r > rstar) or (ss <= 0 and pred > 0)),
                         predicted_gap_static=pred, realised_gap_backtest=real,
                         sign_correct=bool(np.sign(pred) == np.sign(real))))
    return pd.DataFrame(rows)


def hurdle_bootstrap(panel: pd.DataFrame, idx: pd.PeriodIndex) -> pd.DataFrame:
    """Bootstrap clouds of (Sharpe stocks, Sharpe bonds) per era, for the star figure."""
    rng = np.random.default_rng(SEED)
    out = []
    P = panel.loc[idx]
    for name, m in segments(idx).items():
        if name == "Full sample":
            continue
        X = P.loc[m, ["eq_x", "bond_x"]].values
        bi = st.stationary_bootstrap_idx(len(X), 1000, BLOCK, rng)
        S = X[bi]
        sr = S.mean(axis=1) / S.std(axis=1, ddof=1) * np.sqrt(12)
        out.append(pd.DataFrame({"segment": name, "sharpe_stocks": sr[:, 0], "sharpe_bonds": sr[:, 1]}))
    return pd.concat(out)


def hurdle_curve() -> pd.DataFrame:
    rows = []
    for a in (0.25, 0.29, 0.35):
        for rho in np.round(np.arange(-0.5, 0.71, 0.05), 2):
            rows.append(dict(a=a, corr=rho, rstar=pf.hurdle(rho, a)))
    return pd.DataFrame(rows)


def windfall_table(panel: pd.DataFrame, idx: pd.PeriodIndex) -> pd.DataFrame:
    """Split each era's excess returns into a windfall (yield change for bonds, change in log P/D for stocks) and the
    rest, annualised."""
    rows = []
    P = panel.loc[idx]
    dlpd = (-np.log(panel["dy"])).diff().loc[idx]
    for name, m in segments(idx).items():
        p = P.loc[m]
        yrs = m.sum() / 12
        rows.append(dict(segment=name, years=yrs,
                         y10_start=panel["y10"].shift(1).loc[idx][m].iloc[0], y10_end=p.y10.iloc[-1],
                         dy_start=panel["dy"].shift(1).loc[idx][m].iloc[0], dy_end=p.dy.iloc[-1],
                         bond_excess=p.bond_x.mean() * 12, bond_carry=p.bond_carry.mean() * 12,
                         bond_yield_change=p.bond_price.mean() * 12,
                         stock_excess=p.eq_x.mean() * 12, stock_rerating=dlpd[m].mean() * 12,
                         stock_excl_rerating=(p.eq_x.mean() - dlpd[m].mean()) * 12))
    return pd.DataFrame(rows)


def break_even_spread(d: pd.DataFrame) -> pd.DataFrame:
    """Financing spread (per year, on top of the 25 bp already charged) at which RP's Sharpe equals 60/40's."""
    rows = []
    for name, m in segments(d.index).items():
        x = d.loc[m, "RP|xret"].values
        extra = np.maximum(d.loc[m, "RP|lev"].values - 1, 0) / 12
        target = st.sharpe(d.loc[m, "60/40|xret"].values)
        f = lambda s: st.sharpe(x - s * extra) - target  # noqa: E731
        if f(0) <= 0:
            be = np.nan if f(-0.2) < 0 else -_bisect(lambda s: -f(-s), 0, 0.2)
        else:
            be = _bisect(f, 0, 0.5) if f(0.5) < 0 else np.inf
        rows.append(dict(segment=name, extra_spread_break_even=be, avg_leverage=d.loc[m, "RP|lev"].mean()))
    return pd.DataFrame(rows)


def _bisect(f, lo, hi, n=80):
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def stress_table(d: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (a, b) in STRESS.items():
        for src, dd, strats in [("main", d, ["60/40", "RP"]),
                                ("gold", gold, ["ERC-3 (stocks, bonds, gold)"])]:
            m = (dd.index >= period(a)) & (dd.index <= period(b))
            if m.sum() == 0:
                continue
            for s in strats:
                x = dd.loc[m, f"{s}|xret"]
                pnl = {c.split("|")[1][4:]: dd.loc[m, c].sum() for c in dd.columns if c.startswith(f"{s}|pnl_")}
                rows.append(dict(window=name, start=a, end=b, strategy=s, cum_excess=np.prod(1 + x) - 1,
                                 **{f"pnl_{k}": v for k, v in pnl.items()},
                                 avg_leverage=dd.loc[m, f"{s}|lev"].mean()))
    return pd.DataFrame(rows)


def risk_share(d: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Did risk parity deliver equal risk? Ex-post share of RP's variance from bonds, per era, using the era's realised
    covariance and the actual (time-varying) weights: mean over months of w_b (S w)_b / w'Sw."""
    rows = []
    for name, m in segments(d.index).items():
        P = panel.loc[d.index[m], ["eq_x", "bond_x"]]
        S = P.cov().values
        W = d.loc[m, ["RP|w_eq", "RP|w_bond"]].values
        sh = np.array([pf.risk_contributions(w, S)[1] for w in W])
        W6 = np.tile([0.6, 0.4], (len(W), 1))
        sh6 = np.array([pf.risk_contributions(w, S)[1] for w in W6])
        rows.append(dict(segment=name, rp_bond_risk_share=sh.mean(), b6040_bond_risk_share=sh6.mean(),
                         rp_bond_weight=W[:, 1].mean(), rp_stock_weight=W[:, 0].mean()))
    return pd.DataFrame(rows)


def rolling_table(d: pd.DataFrame, panel: pd.DataFrame, h: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    gap = (d["RP|xret"] - d["60/40|xret"]).rolling(h).mean() * 12
    srgap = (d["RP|xret"].rolling(h).mean() / d["RP|xret"].rolling(h).std()
             - d["60/40|xret"].rolling(h).mean() / d["60/40|xret"].rolling(h).std()) * np.sqrt(12)
    y = panel["y10"].reindex(d.index)
    y_lag = panel["y10"].shift(h).reindex(d.index)
    dy = y - y_lag
    corr = panel[["eq_x", "bond_x"]].rolling(h).corr().xs("eq_x", level=1)["bond_x"].reindex(d.index)
    roll = pd.DataFrame({"ret_gap": gap, "sharpe_gap": srgap, "yield_change_10y": dy, "stock_bond_corr_10y": corr}).dropna()
    reg = []
    for xname in ("yield_change_10y", "stock_bond_corr_10y"):
        for yname in ("sharpe_gap", "ret_gap"):
            r = st.newey_west_ols(roll[yname].values, roll[xname].values, lags=h)
            reg.append(dict(y=yname, x=xname, **r))
        # non-overlapping check: one observation every 10 years
    nono = roll.iloc[::h]
    for xname in ("yield_change_10y",):
        r = st.newey_west_ols(nono["sharpe_gap"].values, nono[xname].values, lags=0)
        reg.append(dict(y="sharpe_gap (non-overlapping)", x=xname, **r))
    return roll, pd.DataFrame(reg)


def robustness_table() -> pd.DataFrame:
    rows = []
    variants = pd.read_csv(RES / "robustness_variants.csv")["variant"].tolist()
    names = ["main (36m sample cov, 25 bp spread, 5 bp costs)"] + variants
    files = ["main"] + ["rob_" + v.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "").replace(".", "")
                        for v in variants]
    for n, f in zip(names, files):
        d = load_bt(f)
        for seg, m in segments(d.index).items():
            rows.append(dict(variant=n, segment=seg, sharpe_6040=st.sharpe(d.loc[m, "60/40|xret"]),
                             sharpe_rp=st.sharpe(d.loc[m, "RP|xret"]),
                             gap=st.sharpe(d.loc[m, "RP|xret"]) - st.sharpe(d.loc[m, "60/40|xret"])))
    return pd.DataFrame(rows)


GOLD_SEGS = {"1975-1981 (bond bear, gold boom)": ("1975-01", "1981-09"),
             "E3 1981-2020 (bond bull)": ("1981-10", "2020-07"),
             "E4 2020-2026 (bond bear)": ("2020-08", "2026-09"),
             "Full 1975-2026": ("1975-01", "2026-09")}


def gold_tables(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    perf, tests = [], []
    for k, (seg, (a, b)) in enumerate(GOLD_SEGS.items()):
        m = (g.index >= period(a)) & (g.index <= period(b))
        for s in strategies(g):
            x = g.loc[m, f"{s}|xret"].values
            perf.append(dict(segment=seg, strategy=s, months=m.sum(), sharpe=st.sharpe(x), ann_excess=x.mean() * 12,
                             ann_vol=x.std(ddof=1) * np.sqrt(12), avg_leverage=g.loc[m, f"{s}|lev"].mean(),
                             avg_w_gold=g.loc[m, f"{s}|w_gold"].mean()))
        for j, (a_, b_) in enumerate([("ERC-3 (stocks, bonds, gold)", "RP"), ("ERC-3 (stocks, bonds, gold)", "60/40"),
                                      ("RP", "60/40")]):
            t = st.paired_sharpe_test(g.loc[m, f"{a_}|xret"].values, g.loc[m, f"{b_}|xret"].values, N_BOOT, BLOCK,
                                      SEED + 100 * k + j)
            tests.append(dict(segment=seg, a=a_, b=b_, months=m.sum(), **t))
    tests = pd.DataFrame(tests)
    # H6 family: ERC-3 vs RP in the two bond-bear stretches
    fam = tests[(tests.a.str.startswith("ERC")) & (tests.b == "RP") & tests.segment.str.contains("bear")].index
    tests["p_holm_H6"] = np.nan
    tests.loc[fam, "p_holm_H6"] = st.holm(tests.loc[fam, "p"].values)
    return pd.DataFrame(perf), tests


def main():
    panel = load_panel()
    d = load_bt("main")
    idx = d.index

    perf = perf_table(d)
    perf.to_csv(RES / "performance_by_era.csv", index=False, float_format="%.4f")

    # H1/H2: RP vs 60/40 by era (Holm over the four eras)
    t = diff_tests(d, [("RP", "60/40"), ("RP-unlevered", "60/40")], SEED)
    t["family"] = np.where((t.a == "RP") & (t.segment != "Full sample"), "H1/H2 eras", "")
    fam = t.family == "H1/H2 eras"
    t["p_holm"] = np.nan
    t.loc[fam, "p_holm"] = st.holm(t.loc[fam, "p"].values)
    # H5: ex-ante mean-based rules vs RP over the full sample (Holm over three)
    t5 = diff_tests(d, [("MVO-expanding-means", "RP"), ("MVO-10y-means", "RP"), ("Switch-10y", "RP"),
                        ("MVO-10y-means", "60/40"), ("Switch-10y", "60/40")], SEED + 5)
    t5["family"] = np.where((t5.b == "RP") & (t5.segment == "Full sample"), "H5", "")
    f5 = t5.family == "H5"
    t5["p_holm"] = np.nan
    t5.loc[f5, "p_holm"] = st.holm(t5.loc[f5, "p"].values)
    # extra windows: the sample of Asness-Frazzini-Pedersen (to 2010) and E4 without 2022
    extra = []
    for lab, m in [("1929-07 to 2010-12 (ends where AFP 2012 ends)", d.index <= period("2010-12")),
                   ("E4 without 2022", (d.index >= period("2020-08")) & ((d.index < period("2022-01")) | (d.index > period("2022-12"))))]:
        r = st.paired_sharpe_test(d.loc[m, "RP|xret"].values, d.loc[m, "60/40|xret"].values, N_BOOT, BLOCK, SEED + 77)
        extra.append(dict(segment=lab, a="RP", b="60/40", months=int(m.sum()), family="extra window", **r))
    tests = pd.concat([t, t5, pd.DataFrame(extra)], ignore_index=True)
    tests.to_csv(RES / "sharpe_tests.csv", index=False, float_format="%.4f")

    # H4: the hurdle
    h = hurdle_table(panel, idx, d)
    h.to_csv(RES / "hurdle_by_era.csv", index=False, float_format="%.4f")
    hurdle_bootstrap(panel, idx).to_csv(RES / "hurdle_bootstrap_clouds.csv.gz", index=False, float_format="%.4f")
    hurdle_curve().to_csv(RES / "hurdle_curve.csv", index=False, float_format="%.4f")

    # H3: windfalls
    w = windfall_table(panel, idx)
    w.to_csv(RES / "windfall_decomposition.csv", index=False, float_format="%.5f")
    wf = load_bt("windfall_free")
    tw = diff_tests(wf, [("RP", "60/40")], SEED + 3)
    base = t[t.a == "RP"].set_index("segment")["diff"]
    tw["gap_actual"] = tw.segment.map(base)
    tw["share_of_gap_removed"] = 1 - tw["diff"] / tw["gap_actual"]
    for k, nm in enumerate(["windfall_free_bonds_only", "windfall_free_stocks_only"]):
        tx = diff_tests(load_bt(nm), [("RP", "60/40")], SEED + 30 + k)
        sfx = nm.replace('windfall_free_', '')
        for c in ("diff", "ci_lo", "ci_hi", "p"):
            tw[f"{c}_{sfx}"] = tx[c].values
    tw.to_csv(RES / "windfall_free_tests.csv", index=False, float_format="%.4f")
    hw = hurdle_table(windfall_free(panel), idx, wf)
    hw.to_csv(RES / "hurdle_by_era_windfall_free.csv", index=False, float_format="%.4f")

    break_even_spread(d).to_csv(RES / "break_even_spread.csv", index=False, float_format="%.5f")
    risk_share(d, panel).to_csv(RES / "risk_share_by_era.csv", index=False, float_format="%.4f")
    g = load_bt("gold")
    gp, gt = gold_tables(g)
    gp.to_csv(RES / "gold_performance.csv", index=False, float_format="%.4f")
    gt.to_csv(RES / "gold_tests.csv", index=False, float_format="%.4f")
    stress_table(d, g).to_csv(RES / "stress_windows.csv", index=False, float_format="%.4f")
    roll, reg = rolling_table(d, panel)
    roll.index = roll.index.astype(str)
    roll.to_csv(RES / "rolling_10y.csv", float_format="%.5f")
    reg.to_csv(RES / "rolling_regressions.csv", index=False, float_format="%.4f")
    robustness_table().to_csv(RES / "robustness.csv", index=False, float_format="%.4f")

    # headline numbers
    s = {}
    for _, r in t[t.a == "RP"].iterrows():
        s[f"gap {r.segment}"] = f"{r['diff']:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] p={r.p:.3f} holm={r.p_holm if r.p_holm == r.p_holm else float('nan'):.3f}"
    for _, r in h.iterrows():
        s[f"hurdle {r.segment}"] = f"r={r.ratio_r:.2f} r*={r.hurdle_rstar:.2f} corr={r['corr']:+.2f} pred={r.predicted_gap_static:+.2f} real={r.realised_gap_backtest:+.2f}"
    for _, r in tw.iterrows():
        s[f"windfall-free gap {r.segment}"] = f"{r['diff']:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] p={r.p:.3f} (actual {r.gap_actual:+.2f})"
    for _, r in t5[t5.family == "H5"].iterrows():
        s[f"H5 {r.a} - RP"] = f"{r['diff']:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] p={r.p:.3f} holm={r.p_holm:.3f}"
    pd.Series(s).to_csv(RES / "summary.csv", header=["value"])
    print(pd.Series(s).to_string())


if __name__ == "__main__":
    main()
