"""Publication figures for project 20. Run after data.py, backtest.py and evaluate.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import portfolio as pf  # noqa: E402
from backtest import load_panel  # noqa: E402
from config import ERAS, STRESS, period  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens shared with projects 01 and 05 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5bd6"]
C6040, CRP, CERC, CMVO = CAT[1], CAT[0], CAT[2], CAT[3]
BULL, BEAR = "#e8f0fb", "#fbeee8"   # era bands: yields falling (blue tint) / rising (orange tint)
ERA_KEYS = list(ERAS)
ERA_LAB = {k: k.split(" ", 1)[1].replace(" (", "\n(") for k in ERA_KEYS}
FALLING = {ERA_KEYS[0]: True, ERA_KEYS[1]: False, ERA_KEYS[2]: True, ERA_KEYS[3]: False}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def title(fig, t, sub, y=1.0):
    fig.text(0.01, y + 0.035 * sub.count("\n"), t, fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(0.01, y - 0.035, sub, fontsize=10, color=INK2, ha="left", va="bottom")


def bt(name):
    d = pd.read_csv(RES / f"bt_{name}.csv.gz", index_col=0)
    d.index = pd.PeriodIndex(d.index, freq="M")
    return d


def era_bands(ax, x_of=lambda p: p.to_timestamp(), labels=True, ytxt=None):
    for k, (a, b) in ERAS.items():
        x0, x1 = x_of(period(a)), x_of(period(b))
        ax.axvspan(x0, x1, color=BULL if FALLING[k] else BEAR, lw=0, zorder=0)
        if labels:
            ax.text(x0 + (x1 - x0) / 2, ytxt, ERA_LAB[k], ha="center", va="top", fontsize=8.5, color=INK2)


# ------------------------------------------------------------------ fig 1 (star)


def fig1():
    panel = load_panel()
    t = pd.read_csv(RES / "sharpe_tests.csv")
    t = t[(t.a == "RP") & (t.b == "60/40") & t.segment.isin(ERA_KEYS)].set_index("segment").loc[ERA_KEYS]
    w = pd.read_csv(RES / "windfall_free_tests.csv").set_index("segment").loc[ERA_KEYS]

    fig = plt.figure(figsize=(11, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=0.38)
    ax = fig.add_subplot(gs[0])
    y = panel["y10"] * 100
    x = y.index.to_timestamp()
    era_bands(ax, ytxt=17.3)
    ax.plot(x, y.values, color=INK, lw=1.6)
    ax.set_ylim(0, 18)
    ax.set_ylabel("10-year Treasury yield (%)")
    ax.set_xlim(x[0], x[-1])
    ax.set_title("The four eras are cut at the yield's turning points (blue = yields falling, orange = rising)",
                 loc="left", fontsize=10.5)

    ax2 = fig.add_subplot(gs[1])
    pos = np.arange(4)
    for i, k in enumerate(ERA_KEYS):
        ax2.axvspan(i - 0.5, i + 0.5, color=BULL if FALLING[k] else BEAR, lw=0, zorder=0)
    ax2.axhline(0, color=INK2, lw=1)
    off = 0.14
    ax2.errorbar(pos - off, t["diff"], yerr=[t["diff"] - t.ci_lo, t.ci_hi - t["diff"]], fmt="o", ms=9, color=CRP,
                 ecolor=CRP, elinewidth=2, capsize=0, label="Actual returns", zorder=3)
    ax2.errorbar(pos + off, w["diff_bonds_only"],
                 yerr=[w["diff_bonds_only"] - w.ci_lo_bonds_only, w.ci_hi_bonds_only - w["diff_bonds_only"]],
                 fmt="D", ms=8, mfc=SURFACE, mec=CRP, mew=2, color=CRP, ecolor=CRP, elinewidth=1, capsize=0,
                 label="Bond-yield windfall removed (era-average capital gain or loss from yield changes taken out)",
                 zorder=3)
    for i, k in enumerate(ERA_KEYS):
        ax2.annotate(f"{t.loc[k, 'diff']:+.2f}", (i - off, t.loc[k, "diff"]), xytext=(-10, 0),
                     textcoords="offset points", ha="right", va="center", fontsize=9, color=INK)
        ax2.annotate(f"{w.loc[k, 'diff_bonds_only']:+.2f}", (i + off, w.loc[k, "diff_bonds_only"]), xytext=(10, 0),
                     textcoords="offset points", ha="left", va="center", fontsize=9, color=INK2)
    ax2.set_xticks(pos, [ERA_LAB[k] for k in ERA_KEYS])
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylabel("Sharpe ratio of risk parity\nminus Sharpe ratio of 60/40")
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="lower left", fontsize=9, handletextpad=0.4)
    ax2.set_title("Risk parity (levered to 60/40's volatility) beat 60/40 only when yields fell. Bars: 95% bootstrap CI",
                  loc="left", fontsize=10.5)
    ax2.text(0.995, 0.97, "above 0: risk parity better", transform=ax2.transAxes, ha="right", va="top",
             fontsize=8.5, color=MUTED)
    title(fig, "Risk parity's edge over 60/40 followed the direction of bond yields, 1929–2026",
          "Monthly US stocks and 10-year Treasuries. The 1981–2020 edge disappears once the falling-yield windfall is removed.",
          y=0.955)
    save(fig, "fig1_edge_follows_yields")


# ------------------------------------------------------------------ fig 2: the hurdle plane


def fig2():
    h = pd.read_csv(RES / "hurdle_by_era.csv").set_index("segment")
    cl = pd.read_csv(RES / "hurdle_bootstrap_clouds.csv.gz")
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.4), sharex=True, sharey=True)
    lim = (-1.6, 1.9)
    xs = np.linspace(lim[0], lim[1], 50)
    for ax, k in zip(axes, ERA_KEYS):
        r = h.loc[k]
        c = cl[cl.segment == k]
        ax.fill_between(xs, r.hurdle_rstar * xs, lim[1] + 1, color=BULL, lw=0, zorder=0)
        ax.scatter(c.sharpe_stocks, c.sharpe_bonds, s=4, color=MUTED, alpha=0.25, lw=0, zorder=1)
        ax.plot(xs, r.hurdle_rstar * xs, color=CRP, lw=2, zorder=2)
        ax.plot(r.sharpe_stocks, r.sharpe_bonds, "o", ms=9, color=INK, mec=SURFACE, mew=2, zorder=4)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_title(ERA_LAB[k].replace("\n", " "), fontsize=10)
        tl = k != ERA_KEYS[0]
        ax.text(0.03 if tl else 0.97, 0.97 if tl else 0.03, f"corr = {r['corr']:+.2f}\nhurdle r* = {r.hurdle_rstar:.2f}\n"
                f"actual gap {r.realised_gap_backtest:+.2f}", transform=ax.transAxes, va="top" if tl else "bottom",
                ha="left" if tl else "right", fontsize=8.5,
                color=INK2, bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID))
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Sharpe ratio of stocks")
    axes[0].set_ylabel("Sharpe ratio of 10-year bonds")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hand = [Line2D([], [], color=CRP, lw=2, label="Break-even line: bond Sharpe = r* × stock Sharpe"),
            Patch(color=BULL, label="Region where risk parity beats 60/40"),
            Line2D([], [], marker="o", color=INK, lw=0, ms=8, label="The era's actual Sharpe ratios"),
            Line2D([], [], marker="o", color=MUTED, lw=0, ms=4, alpha=0.6, label="1,000 bootstrap resamples")]
    fig.legend(handles=hand, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    title(fig, "The hurdle: how good bonds must be for risk parity to win",
          "Risk parity beats 60/40 when the bond Sharpe ratio exceeds r*(corr) times the stock Sharpe ratio. "
          "The point landed on the predicted side of the line in all four eras.", y=0.93)
    save(fig, "fig2_hurdle_plane")


# ------------------------------------------------------------------ fig 3: hurdle vs correlation


def fig3():
    hc = pd.read_csv(RES / "hurdle_curve.csv")
    h = pd.read_csv(RES / "hurdle_by_era.csv").set_index("segment")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    c = hc[hc.a == 0.29]
    ax.plot(c["corr"], c.rstar, color=CRP, lw=2, label="Hurdle r* for 60/40 with a typical risk ratio (a = 0.29)")
    for a_, ls in [(0.25, ":"), (0.35, "--")]:
        c = hc[hc.a == a_]
        ax.plot(c["corr"], c.rstar, color=CRP, lw=1, ls=ls, label=f"a = {a_}")
    for k in ERA_KEYS:
        r = h.loc[k]
        ax.plot(r["corr"], r.hurdle_rstar, "o", ms=9, color=INK, mec=SURFACE, mew=2, zorder=4)
        ax.annotate(ERA_LAB[k].split("\n")[0], (r["corr"], r.hurdle_rstar), xytext=(8, -12), textcoords="offset points",
                    fontsize=8.5, color=INK2)
    ax.set_xlabel("Correlation between stock and bond excess returns")
    ax.set_ylabel("Hurdle r* = break-even bond Sharpe / stock Sharpe")
    ax.legend(fontsize=8.5, loc="upper left")
    title(fig, "A positive stock–bond correlation raises the bar for bonds",
          "Each era's hurdle (dots) uses its own correlation and volatilities. In 2020–26 the correlation rose to +0.47.",
          y=0.96)
    save(fig, "fig3_hurdle_vs_correlation")


# ------------------------------------------------------------------ fig 4: rolling


def fig4():
    r = pd.read_csv(RES / "rolling_10y.csv", index_col=0)
    r.index = pd.PeriodIndex(r.index, freq="M")
    reg = pd.read_csv(RES / "rolling_regressions.csv")
    b = reg[(reg.y == "sharpe_gap") & (reg.x == "yield_change_10y")].iloc[0]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bins = [(1939, 1959), (1960, 1979), (1980, 1999), (2000, 2026)]
    for (y0, y1), c in zip(bins, [CAT[0], CAT[1], CAT[2], CAT[6]]):
        m = (r.index.year >= y0) & (r.index.year <= y1)
        ax.scatter(r.yield_change_10y[m] * 100, r.sharpe_gap[m], s=10, color=c, lw=0, alpha=0.8,
                   label=f"windows ending {y0}–{y1}")
    xs = np.linspace(r.yield_change_10y.min(), r.yield_change_10y.max(), 10)
    ax.plot(xs * 100, b.alpha + b.beta * xs, color=INK, lw=1.6,
            label=f"OLS slope {b.beta / 100:+.3f} Sharpe per 1 pp (Newey–West t = {b.t_beta:.1f})")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Change in the 10-year yield over the window (percentage points)")
    ax.set_ylabel("Sharpe of risk parity minus Sharpe of 60/40")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    title(fig, "Over any 10 years, rising yields meant risk parity lagged",
          "Every rolling 10-year window, 1939–2026 (overlapping, so the t-statistic uses Newey–West errors).", y=0.95)
    save(fig, "fig4_rolling_gap_vs_yields")


# ------------------------------------------------------------------ fig 5: wealth


def fig5():
    d = bt("main")
    x = d.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for s, c, lab in [("60/40", C6040, "60/40"), ("RP", CRP, "Risk parity, levered to 60/40's ex-ante volatility"),
                      ("RP-unlevered", MUTED, "Risk parity, unlevered")]:
        ax.plot(x, np.cumprod(1 + d[f"{s}|xret"]), color=c, lw=1.6, label=lab)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    for k, (a, b) in ERAS.items():
        ax.axvspan(max(period(a), d.index[0]).to_timestamp(), period(b).to_timestamp(),
                   color=BULL if FALLING[k] else BEAR, lw=0, zorder=0)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel("Growth of $1 in excess of T-bills (log scale)")
    ax.legend(fontsize=9, loc="upper left")
    title(fig, "Same volatility, different century: risk parity vs 60/40",
          "Excess returns over T-bills, net of 5 bp trading costs and T-bill + 25 bp financing. "
          "Blue bands: yields falling; orange: rising.", y=0.95)
    save(fig, "fig5_wealth")


# ------------------------------------------------------------------ fig 6: windfall decomposition


def fig6():
    w = pd.read_csv(RES / "windfall_decomposition.csv").set_index("segment").loc[ERA_KEYS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    pos = np.arange(4)
    wd = 0.36
    ax = axes[0]
    ax.bar(pos - wd / 2, w.bond_carry * 100, wd, color=CRP, label="Carry: coupon minus T-bill")
    ax.bar(pos + wd / 2, w.bond_yield_change * 100, wd, color=CAT[6], label="Windfall: capital gain/loss from yield changes")
    ax.set_title("10-year bonds: excess return per year", loc="left")
    ax.set_ylabel("% per year")
    ax = axes[1]
    ax.bar(pos - wd / 2, w.stock_excl_rerating * 100, wd, color=C6040, label="Dividends + growth (excess of T-bills)")
    ax.bar(pos + wd / 2, w.stock_rerating * 100, wd, color=CAT[3], label="Windfall: re-rating (change in log price/dividend)")
    ax.set_title("Stocks: excess return per year", loc="left")
    for ax in axes:
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(pos, [ERA_LAB[k] for k in ERA_KEYS], fontsize=8.5)
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    title(fig, "Where the returns came from, era by era",
          "Both assets had windfalls. Bonds' windfall flips sign with the yield cycle and is what risk parity overweights.",
          y=1.01)
    save(fig, "fig6_windfalls")


# ------------------------------------------------------------------ fig 7: stress


def fig7():
    s = pd.read_csv(RES / "stress_windows.csv")
    names = list(STRESS)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    pos = np.arange(len(names))
    wd = 0.27
    for j, (st_, c, lab) in enumerate([("60/40", C6040, "60/40"), ("RP", CRP, "Risk parity (stocks + bonds)"),
                                       ("ERC-3 (stocks, bonds, gold)", CERC, "Equal risk: stocks, bonds, gold (from 1975)")]):
        v = [s[(s.window == n) & (s.strategy == st_)].cum_excess.values for n in names]
        v = [x[0] * 100 if len(x) else np.nan for x in v]
        ax.bar(pos + (j - 1) * wd, v, wd, color=c, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(pos, [n.replace(" ", "\n", 1) for n in names], fontsize=8.5)
    ax.set_ylabel("Cumulative excess return in the window (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9, loc="lower right")
    title(fig, "Risk parity protects in equity crashes and fails in inflation shocks",
          "1979–81 and 2022 were the two windows where bonds, not stocks, did the damage.", y=0.95)
    save(fig, "fig7_stress_windows")


# ------------------------------------------------------------------ fig 8: robustness


def fig8():
    r = pd.read_csv(RES / "robustness.csv")
    r = r[r.segment.isin(ERA_KEYS + ["Full sample"])]
    variants = list(dict.fromkeys(r.variant))
    segs = ERA_KEYS + ["Full sample"]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for j, sg in enumerate(segs):
        v = r[r.segment == sg].set_index("variant").loc[variants, "gap"]
        y = np.arange(len(variants))
        c = INK if sg == "Full sample" else (CRP if FALLING[sg] else C6040)
        ax.scatter(np.full(len(v), j) + np.linspace(-0.25, 0.25, len(v)), v, s=22, color=c, lw=0, alpha=0.9)
        main = v.iloc[0]
        ax.plot([j - 0.32, j + 0.32], [main, main], color=c, lw=2.2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(segs)), [ERA_LAB[k] for k in ERA_KEYS] + ["Full sample\n1929–2026"], fontsize=9)
    ax.set_ylabel("Sharpe of risk parity minus 60/40")
    ax.grid(axis="x", visible=False)
    ax.text(0.0, -0.2, f"{len(variants)} specifications per column: estimation window 12–120 months, EWMA, financing "
            "spread 0–200 bp,\ncosts 0–20 bp, leverage cap 3×, 10% vol target, two alternative bond series. Bar = main specification.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK2)
    title(fig, "Within each era, the sign of the edge does not depend on the specification",
          "Each dot is one robustness variant. Blue: yields falling; orange: yields rising; black: full sample, "
          "which hovers around zero.", y=0.95)
    save(fig, "fig8_robustness")


# ------------------------------------------------------------------ fig 9: gold


def fig9():
    g = pd.read_csv(RES / "gold_performance.csv")
    segs = list(dict.fromkeys(g.segment))
    strats = [("60/40", C6040), ("RP", CRP), ("IV-3 (stocks, bonds, gold)", CMVO), ("ERC-3 (stocks, bonds, gold)", CERC)]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    pos = np.arange(len(segs))
    wd = 0.2
    for j, (s, c) in enumerate(strats):
        v = [g[(g.segment == sg) & (g.strategy == s)].sharpe.values[0] for sg in segs]
        ax.bar(pos + (j - 1.5) * wd, v, wd, color=c, label={"RP": "Risk parity (stocks + bonds)"}.get(s, s))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(pos, [s.replace(" (", "\n(") for s in segs], fontsize=8.5)
    ax.set_ylabel("Sharpe ratio")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    title(fig, "Gold as the third leg: insurance for bond bears, paid for in the bull",
          "All risk-parity variants levered to 60/40's ex-ante volatility. 1975–2026 (36-month warm-up from 1972).", y=0.95)
    save(fig, "fig9_gold")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
        f()
    print("figures done")
