"""Publication figures for project 21. Run after run_experiments.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import ETF_OF, INDICES, LABEL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# design tokens shared with projects 01, 09 and 12
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
KIND_COLOR = {"index": BLUE, "ETF": ORANGE}
KIND_LABEL = {"index": "Price index (what projects 03 and 19 backtested)", "ETF": "US-listed ETF on that index (what you can trade)"}

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
    fig.text(0.01, y, t, fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(0.01, y - 0.012, sub, fontsize=10, color=INK2, ha="left", va="top")


def fig1_star():
    """Per-trade cost of filling one close later, MA200, by era and instrument."""
    e = pd.read_csv(RES / "era_pooled.csv")
    e = e[e.strategy == "MA200"]
    eras = ["1928-1989", "1990-2005", "2006-2026"]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.axhspan(-5, 5, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.text(2.45, 72, "Grey band: ±5 bp, the entire trading cost\nassumed per trade in projects 03 and 19",
            fontsize=8, color=INK2, va="top", ha="right")
    ax.axhline(0, color=AXIS, lw=1)
    rng = np.random.default_rng(0)
    for k, (kind, dx) in enumerate([("index", -0.17), ("ETF", 0.17)]):
        for i, era in enumerate(eras):
            q = e[(e.kind == kind) & (e.era == era)]
            if q.empty:
                continue
            pts = q[~q.pooled]
            ax.scatter(i + dx + rng.uniform(-0.05, 0.05, len(pts)), pts.bp_per_trade, s=16, color=KIND_COLOR[kind],
                       alpha=0.35, lw=0, zorder=2)
            m = q[q.pooled].iloc[0]
            ax.errorbar(i + dx, m.bp_per_trade, yerr=[[m.bp_per_trade - m.lo], [m.hi - m.bp_per_trade]], fmt="D",
                        ms=8, color=KIND_COLOR[kind], ecolor=KIND_COLOR[kind], elinewidth=2, capsize=4, zorder=3,
                        label=KIND_LABEL[kind] if i == (0 if kind == "index" else 1) else None)
            up = m.bp_per_trade >= 0
            ax.annotate(f"{m.bp_per_trade:+.0f} bp\n{int(m.n_positive)}/{int(m.n)} > 0", (i + dx, m.hi if up else m.lo),
                        xytext=(0, 6 if up else -6), textcoords="offset points", ha="center",
                        va="bottom" if up else "top", fontsize=8.5,
                        color=KIND_COLOR[kind], fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Before 1990\n(indices only)", "1990–2005\n(ETFs from 1993–2001)", "2006–2026"])
    ax.set_ylabel("Return lost per trade by filling one close later (bp)\n> 0: the same-close backtest flatters the rule")
    ax.set_ylim(-75, 75)
    ax.set_xlim(-0.5, 2.5)
    ax.legend(loc="lower left", fontsize=9)
    fig.text(0.99, -0.02, "Diamonds: mean over markets with 95% CI (stationary bootstrap, all markets resampled on the same "
             "dates). Dots: single markets with ≥ 25 trades in the era.", ha="right", fontsize=8, color=MUTED)
    title(fig, "A backtest's fill convention is a bet on microstructure",
          "200-day moving-average rule, 10 equity indices and their US-listed ETFs. Old index prices are stale, so "
          "the day after a\ncrossing continues the move; early ETF prices were noisy, so it reversed; today it is "
          "close to zero.", y=1.0)
    fig.subplots_adjust(top=0.86)
    save(fig, "fig1_fill_convention_by_era")


def fig2_profile():
    p = pd.read_csv(RES / "profile.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, strat in zip(axes, ["MA200", "HMM"]):
        q = p[(p.strategy == strat) & (p.h >= 1)]
        for kind in ["index", "ETF"]:
            g = q[q.kind == kind].pivot(index="h", columns="sym", values="bp")
            for c in g.columns:
                ax.plot(g.index, g[c], color=KIND_COLOR[kind], alpha=0.18, lw=0.8)
            ax.plot(g.index, g.median(axis=1), color=KIND_COLOR[kind], lw=2.4, marker="o", ms=4,
                    label=f"{'Indices' if kind == 'index' else 'ETFs'} (median of 10)")
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xlabel("Days after the trade (h)")
        ax.set_title("200-day MA (project 03)" if strat == "MA200" else "Real-time HMM (project 19)", loc="left")
        ax.set_xticks(range(1, 11))
        trig = p[(p.strategy == strat) & (p.h == 0)].groupby("kind").bp.median()
        ax.text(0.98, 0.97, f"Trade-day return (h = 0), median:\nindex {trig['index']:.0f} bp, ETF {trig['ETF']:.0f} bp",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    axes[0].set_ylabel("Mean return in the direction of the trade (bp)")
    axes[0].set_ylim(-40, 40)
    axes[0].legend(loc="lower left", fontsize=8.5)
    title(fig, "The day after the signal", "Each step of fill delay gives up the return of day h; the same-close "
          "convention keeps all of them. Faint lines: single markets.", y=1.06)
    save(fig, "fig2_day_after_profile")


def fig3_mechanism():
    m = pd.read_csv(RES / "mechanism.csv")
    f = pd.read_csv(RES / "mechanism_fit.csv").set_index("sample")
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    for strat, mk in [("MA200", "o"), ("HMM", "^")]:
        for kind in ["index", "ETF"]:
            q = m[(m.strategy == strat) & (m.kind == kind)]
            ax.scatter(q.predicted_bp, q.actual_bp, s=q.trades / 3, marker=mk, color=KIND_COLOR[kind],
                       alpha=0.6 if strat == "MA200" else 0.35, lw=0,
                       label=f"{'MA200' if strat == 'MA200' else 'HMM'}, {'index' if kind == 'index' else 'ETF'}")
    x = np.linspace(-40, 60, 10)
    ax.plot(x, x, color=MUTED, ls="--", lw=1, label="Actual = predicted")
    a = f.loc["all cells"]
    ax.plot(x, a.intercept + a.slope * x, color=INK, lw=1.6,
            label=f"Fit, all cells: slope {a.slope:.2f} (se {a.slope_se:.2f})")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("Predicted: lag-1 autocorrelation × trade-day return (bp)")
    ax.set_ylabel("Actual return of the day after the trade (bp)")
    ax.set_xlim(-40, 60)
    ax.set_ylim(-90, 90)
    ax.legend(fontsize=8, loc="upper left")
    title(fig, "Autocorrelation predicts who pays", "One point per market × era (pre-1990, 1990–2005, 2006–2026); "
          "size = number of trades.", y=1.04)
    save(fig, "fig3_mechanism")


def fig4_paired():
    g = pd.read_csv(RES / "paired_gap.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    pairs = [f"{n}-{ETF_OF[n]}" for n in INDICES]
    y = np.arange(len(pairs))
    for strat, dy, col in [("MA200", -0.18, BLUE), ("HMM", 0.18, MUTED)]:
        q = g[g.strategy == strat].set_index("pair")
        ax.barh(y + dy, q.loc[pairs, "gap_bp"], height=0.34, color=col,
                label=f"{'200-day MA' if strat == 'MA200' else 'Real-time HMM'}: mean {q.loc['mean', 'gap_bp']:+.1f} bp "
                      f"[{q.loc['mean', 'lo']:+.1f}, {q.loc['mean', 'hi']:+.1f}], {int(q.loc['mean', 'n_positive'])}/10 > 0")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{LABEL[n]} vs {ETF_OF[n]}" for n in INDICES])
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("Delay cost on the index minus delay cost on its ETF, same dates (bp per trade)")
    ax.legend(loc="lower right", fontsize=8.5)
    title(fig, "Same signal, same years, opposite answer", "Positive: the index makes filling late look more costly "
          "than it is on the instrument you would actually trade.", y=1.05)
    save(fig, "fig4_index_vs_etf")


def fig5_curve():
    c = pd.read_csv(RES / "delay_curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, strat in zip(axes, ["MA200", "HMM"]):
        for kind in ["index", "ETF"]:
            q = c[(c.strategy == strat) & (c.kind == kind)].pivot(index="k", columns="sym", values="d_sharpe")
            ax.fill_between(q.index, q.quantile(0.25, axis=1), q.quantile(0.75, axis=1), color=KIND_COLOR[kind], alpha=0.15, lw=0)
            ax.plot(q.index, q.median(axis=1), color=KIND_COLOR[kind], lw=2.2, marker="o", ms=4,
                    label=f"{'Indices' if kind == 'index' else 'ETFs'}: median, interquartile band")
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xlabel("Fill lag k (trade at the close k days after the signal)")
        ax.set_title("200-day MA" if strat == "MA200" else "Real-time HMM", loc="left")
    axes[0].set_ylabel("Change in net Sharpe vs same-close fill")
    axes[0].legend(fontsize=8.5, loc="lower left")
    title(fig, "How fast does the edge decay with fill delay?", "Net of 5 bp costs, full sample of each instrument.", y=1.05)
    save(fig, "fig5_delay_curve")


def fig6_claims():
    c = pd.read_csv(RES / "claims.csv")
    q = c[(c.strategy == "MA200") & (c.kind == "ETF")].pivot(index="sym", columns="fill", values="diff")
    order = [ETF_OF[n] for n in INDICES]
    q = q.loc[order]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    y = np.arange(len(order))
    for i, s in enumerate(order):
        ax.plot([q.loc[s, "close"], q.loc[s, "close+1"]], [i, i], color=AXIS, lw=2, zorder=1)
    ax.scatter(q["close"], y, color=MUTED, s=40, zorder=2,
               label=f"Fill at the signal's close (as in 03): {int((q['close'] > 0).sum())}/10 beat buy & hold")
    ax.scatter(q["close+1"], y, color=ORANGE, s=40, zorder=3,
               label=f"Fill at the next close: {int((q['close+1'] > 0).sum())}/10 beat buy & hold")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("Net Sharpe of the 200-day MA rule minus buy & hold (5 bp costs)")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    title(fig, "On tradable ETFs, the fill flips the verdict's sign", "Same signal, same sample, one close apart. "
          "No single difference is significant after Holm.", y=1.05)
    save(fig, "fig6_ma_vs_buy_hold_etf")


def fig7_momentum():
    m = pd.read_csv(RES / "momentum.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    labs, vals, errs, cols = [], [], [], []
    for (u, s), g in m.groupby(["universe", "sample"], sort=False):
        for _, r in g.iterrows():
            if r.fill == "close":
                continue
            labs.append(f"{'49 industries' if u == 'A' else '42 country ETFs'}{' (real opens)' if s == 'real opens' else ''}: {r.fill}")
            vals.append(r.d_pct_yr)
            errs.append([r.d_pct_yr - r.d_lo, r.d_hi - r.d_pct_yr])
            cols.append(BLUE if u == "A" else ORANGE)
    y = np.arange(len(labs))
    ax.barh(y, vals, color=cols, height=0.6)
    ax.errorbar(vals, y, xerr=np.array(errs).T, fmt="none", ecolor=INK2, capsize=3, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("Change in the gross 12-1 momentum premium vs same-close fill (%/yr, 95% CI)")
    a = m[(m.universe == "A") & (m.fill == "close+1")].iloc[0]
    fig.text(0.01, -0.03, f"Industries: one close later costs {a.delay_cost_bp_per_trade:.1f} bp per unit of turnover, "
            f"against a break-even of {m[(m.universe == 'A') & (m.fill == 'close')].be_bps.iloc[0]:.0f} bp.",
            fontsize=8.5, color=INK2)
    title(fig, "Industry momentum barely notices the fill; country-ETF momentum does", "Textbook 12-1, top/bottom 30%, equal weights, "
          "monthly; the decision is unchanged, only the fill moves.", y=1.06)
    save(fig, "fig7_momentum")


if __name__ == "__main__":
    fig1_star()
    fig2_profile()
    fig3_mechanism()
    fig4_paired()
    fig5_curve()
    fig6_claims()
    fig7_momentum()
    print("figures written")
