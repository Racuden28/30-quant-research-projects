"""Publication figures for project 12. Run after backtest.py, evaluate.py, decay.py, costs_check.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from designs import ERAS, FAMILY, STRESS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# design tokens shared with projects 01 and 09 (validated: blue, orange, green; green needs labels)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
FAM_COLOR = {"partial": BLUE, "band": ORANGE, "hold": GREEN, "smooth": MUTED}
FAM_LABEL = {"partial": "Partial rebalance (trade θ of the gap)", "band": "Band (exit only past a wider cut)",
             "hold": "Overlapping hold (K months)", "smooth": "Rank smoothed over K months"}
ERA_COLOR = {"1928-1962": BLUE, "1963-1999": ORANGE, "2000-2026": GREEN}

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


def _frontier_points(runs, era, universe="A", signal="12-1"):
    a, b = ERAS[universe][era]
    sub = runs[(runs.universe == universe) & (runs.signal == signal)]
    base = sub[sub.design == "base"].set_index("month").loc[a:b]
    ref = base["sigma_bar"].mean()
    pts = {}
    for d, g in sub.groupby("design"):
        if d not in FAMILY or FAMILY[d] == "vol":
            continue
        x = g.set_index("month").loc[a:b]
        pts[d] = ((x["vturn_raw"] / ref).mean(), 1200 * (x["gross"] - x["tick_cost"]).mean())
    return pts


def fig1_frontier(runs, summ, marg):
    eras = ["1928-1962", "1963-1999", "2000-2026"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    order = {"partial": ["base", "partial75", "partial50", "partial33", "partial25", "partial15"],
             "band": ["base", "band35", "band40", "band50", "band60"],
             "hold": ["base", "hold2", "hold3", "hold6", "hold12"],
             "smooth": ["base", "smooth2", "smooth3", "smooth6"]}
    for ax, era in zip(axes, eras):
        P = _frontier_points(runs, era)
        x0, y0 = P["base"]
        be = summ[(summ.universe == "A") & (summ.signal == "12-1") & (summ.design == "base") &
                  (summ.era == era)].iloc[0]
        xs = np.linspace(0, 1.0, 50)
        ax.plot(xs, 12 * be.be_vol_bps / 100 * xs, ls=(0, (4, 3)), color=INK2, lw=1.2, zorder=1,
                label="Textbook's average break-even (line through origin)" if era == eras[0] else None)
        for fam, keys in order.items():
            pts = np.array([P[k] for k in keys])
            ax.plot(pts[:, 0], pts[:, 1], color=FAM_COLOR[fam], lw=2, marker="o", ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
                    label=FAM_LABEL[fam] if era == eras[0] else None)
        ax.plot([x0], [y0], marker="*", ms=17, color=INK, markeredgecolor=SURFACE, zorder=5,
                label="Textbook 12-1 (full monthly rebalance)" if era == eras[0] else None)
        m = marg[(marg.universe == "A") & (marg.signal == "12-1") & (marg.era == era) &
                 (marg.slow_design == "partial50")].iloc[0]
        ax.text(0.03, 0.97, f"Average trade earns {be.be_vol_bps:.0f} bps\n"
                f"Last trades (textbook vs θ = 0.5)\nearn {m.marginal_be_bps:.0f} bps "
                f"[95% CI {round(m.marginal_lo) + 0:.0f} to {round(m.marginal_hi) + 0:.0f}]",
                transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=AXIS, lw=0.8))
        ax.set_title(era, loc="left")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-1, 12.5)
        ax.axhline(0, color=AXIS, lw=0.8)
    fig.supxlabel("Turnover per month (dollars traded per dollar of capital, volatility-weighted)",
                  fontsize=10, color=INK2, y=0.02)
    axes[0].set_ylabel("Gross long-short premium (% per year)")
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.14), fontsize=9)
    title(fig, "Momentum's turnover frontier: the last trades earn far less than the average trade",
          "49 US industries, 12-1 momentum. Each dot is a slower way to run the same signal. The dashed "
          "line is the textbook's average break-even cost.\nThe slope from the star to its nearest "
          "neighbours is what the textbook's last trades earn: much less in 1928-1999, closer to it after 2000.", y=1.07)
    save(fig, "fig1_turnover_frontier")


def fig2_decay(decay):
    d = decay[(decay.universe == "A") & (decay.signal == "12-1")]
    fig, ax = plt.subplots(figsize=(10, 5))
    for era, col in ERA_COLOR.items():
        x = d[d.era == era].sort_values("age")
        ax.fill_between(x.age, x.lo, x.hi, color=col, alpha=0.12, lw=0)
        ax.plot(x.age, x.alpha_pct_yr, color=col, lw=2, marker="o", ms=4, markeredgecolor=SURFACE)
        first = x[x.age == 0].iloc[0]
        ax.text(-0.5, first.alpha_pct_yr, era, color=INK, fontsize=9.5, fontweight="bold",
                ha="right", va="center")
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.set_xlabel("Signal age: months between ranking and the month the return is earned")
    ax.set_ylabel("Long-short premium of that signal (% per year)")
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlim(-4.2, 24.3)
    title(fig, "Why slowing down was cheap before 2000: a 1-3 month old signal kept most of the premium",
          "Mean return of the textbook portfolio formed k months earlier; shaded = 95% stationary-bootstrap CI.",
          y=0.99)
    save(fig, "fig2_signal_decay")


def fig3_net_sharpe(summ):
    s = summ[(summ.universe == "A") & (summ.signal == "12-1") & (summ.era == "1928-2026")].set_index("design")
    costs = [0, 5, 10, 20, 30, 50]
    show = [("base", INK, "Textbook"), ("partial50", BLUE, "Partial θ = 0.5"),
            ("band50", ORANGE, "Band 30/50"), ("vol", GREEN, "Vol-scaled")]
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, col, lab in show:
        y = [s.loc[k, "sharpe_gross"]] + [s.loc[k, f"sharpe_net_{c}"] for c in costs[1:]]
        ax.plot(costs, y, color=col, lw=2, marker="o", ms=5, markeredgecolor=SURFACE, label=lab)
        ax.text(51.5, y[-1], lab, color=INK, va="center", fontsize=9)
    ax.set_xlim(0, 62)
    ax.set_xlabel("One-way trading cost for an asset of average volatility (bps per $ traded)")
    ax.set_ylabel("Net Sharpe ratio, 1928-2026")
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.legend(loc="upper right")
    title(fig, "Volatility scaling wins at every cost; trading slower only pays above ~20 bps",
          "49 US industries, 12-1 momentum, long-short. Costs scale with each asset's 21-day volatility.",
          y=0.99)
    save(fig, "fig3_net_sharpe_vs_cost")


def fig4_cost_timing(runs):
    sub = runs[(runs.universe == "A") & (runs.signal == "12-1")]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    base = sub[sub.design == "base"].set_index("month")
    ref = base["sigma_bar"].mean()
    for k, col, lab in (("base", INK, "Textbook"), ("vol", GREEN, "Vol-scaled")):
        x = sub[sub.design == k].set_index("month")
        cost = 100 * 30e-4 * x["vturn_raw"] / ref * 12
        ax.plot(cost.rolling(12).mean(), color=col, lw=1.4, label=lab)
    for name, (a, b) in STRESS.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b) + pd.offsets.MonthEnd(0), color=ORANGE, alpha=0.18, lw=0)
    avg = (100 * 30e-4 * base["vturn_raw"] / ref * 12).mean()
    ax.axhline(avg, color=INK2, lw=0.9, ls=(0, (4, 3)))
    ax.text(pd.Timestamp("1952-01-01"), avg - 0.75, f"Textbook average: {avg:.1f}% a year", color=INK2, fontsize=9)
    ax.set_ylabel("Annualised cost at 30 bps (%)\n12-month rolling mean")
    ax.legend(loc="upper right")
    ax.set_ylim(0, None)
    title(fig, "Costs spike in crises, but crises are rare: the timing penalty is only 8%",
          "Shaded = stress windows (1932, 1933, 1987, 2000-01, 2008, 2009, 2020, 2022). Vol-scaling trades about 65% "
          "more on average,\nbut less than the textbook in the worst spikes (1932-33, 2009), because it shrinks the book "
          "when volatility, and so cost, is high.", y=1.06)
    save(fig, "fig4_cost_timing")


def fig5_etf(summ, spreads):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    rows = []
    for u, era in (("A", "1928-1962"), ("A", "1963-1999"), ("A", "2000-2026"),
                   ("B", "1998-2008"), ("B", "2009-2026")):
        r = summ[(summ.universe == u) & (summ.signal == "12-1") & (summ.design == "base") &
                 (summ.era == era)].iloc[0]
        rows.append((("US industries " if u == "A" else "Country ETFs ") + era, r))
    for i, (lab, r) in enumerate(rows[::-1]):
        col = BLUE if lab.startswith("US") else ORANGE
        ax.plot([r.be_vol_lo, r.be_vol_hi], [i, i], color=col, lw=2.5, solid_capstyle="round")
        ax.plot([r.be_vol_bps], [i], "o", color=col, ms=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(r.be_vol_hi + 4, i, f"{r.be_vol_bps:.0f} bps", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([l for l, _ in rows[::-1]])
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("Average break-even one-way cost, textbook 12-1 (bps), 95% CI")
    ax.set_title("No premium to pay costs with in the tradable country ETFs", loc="left")
    ax = axes[1]
    sp = spreads[spreads.ticker == "SPY"].copy()
    sp["date"] = pd.to_datetime(sp["date"])
    sp = sp.set_index("date").sort_index()
    ax.semilogy(sp.index, 1e4 * sp["cs"].rolling(12, min_periods=6).median(), color=BLUE, lw=1.6,
                label="Corwin-Schultz estimate")
    ax.semilogy(sp.index, 1e4 * sp["ar_spread"].replace(0, np.nan).rolling(12, min_periods=6).median(),
                color=ORANGE, lw=1.6, label="Abdi-Ranaldo estimate (non-zero months)")
    ax.semilogy(sp.index, 1e4 * sp["one_tick_spread"], color=INK, lw=1.6, label="One tick / price (the floor)")
    ax.set_ylabel("SPY bid-ask spread (bps, log scale)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Daily-price spread estimators say SPY costs ~20-30 bps", loc="left")
    title(fig, "The tradable test: country ETFs have no momentum premium, and their costs cannot be read off prices",
          "Left: break-even with 95% stationary-bootstrap CI (tick floors included for ETFs). Right: SPY trades "
          "at a one-cent spread almost always; the estimators overstate it 30-100 times since 2002.", y=1.05)
    save(fig, "fig5_etf_test")


def main():
    runs = pd.read_csv(RES / "runs.csv.gz", parse_dates=["month"])
    summ = pd.read_csv(RES / "summary.csv")
    marg = pd.read_csv(RES / "marginal_breakeven.csv")
    decay = pd.read_csv(RES / "decay_alpha.csv")
    spreads = pd.read_csv(RES / "spread_estimators.csv")
    fig1_frontier(runs, summ, marg)
    fig2_decay(decay)
    fig3_net_sharpe(summ)
    fig4_cost_timing(runs)
    fig5_etf(summ, spreads)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
