"""Publication figures for project 11. Run after backtest.py, evaluate.py and diagnostics.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import ROOT  # noqa: E402
from evaluate import Raw  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens shared with projects 01-10 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
REAL, PLACEBO, DRIFT = "#2a78d6", "#eb6834", "#8a8984"
GOOD, BAD = "#1baf7a", "#a51c1c"
U = {"industries": "49 US industries, 1927–2026", "countries": "22 country ETFs, 2002–2026"}

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
    fig.text(0.01, y + 0.045, t, fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(0.01, y + 0.005, sub, fontsize=10, color=INK2, ha="left", va="bottom")


def fig1_star():
    conv = pd.read_csv(RES / "convergence.csv")
    perf = pd.read_csv(RES / "performance.csv")
    full = {"industries": "Full 1927-2026", "countries": "Full 2002-2026"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"wspace": 0.12, "width_ratios": [1, 1.05]})
    # A: convergence share
    ax = axes[0]
    y = np.arange(2)
    for i, u in enumerate(("industries", "countries")):
        c = conv[(conv.universe == u) & (conv.period == full[u]) & (conv.config == "D20_w1")].set_index("source")
        r, p = 100 * c.loc["real", "converged_share"], 100 * c.loc["N1", "converged_share"]
        ax.barh(i + 0.18, r, 0.34, color=REAL)
        ax.barh(i - 0.18, p, 0.34, color=PLACEBO)
        ax.text(r + 0.8, i + 0.18, f"Real market  {r:.0f}%", va="center", fontsize=9, color=REAL, fontweight="bold")
        ax.text(p + 0.8, i - 0.18, f"Placebo market  {p:.0f}%", va="center", fontsize=9, color=PLACEBO,
                fontweight="bold")
    ax.set_yticks(y, [U["industries"], U["countries"]])
    ax.set_xlim(0, 62)
    ax.set_xlabel("Share of opened trades that converge\nbefore the trading period ends (%)")
    ax.set_title("A. Trades converge as often in the placebo", loc="left")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    # B: gross return with 95% CI
    ax = axes[1]
    for i, u in enumerate(("industries", "countries")):
        for off, src, col, lab in ((0.18, "real", REAL, "Real market"), (-0.18, "N1", PLACEBO, "Placebo market")):
            r = perf[(perf.universe == u) & (perf.config == "D20_w1") & (perf.series == src) &
                     (perf.period == full[u])].iloc[0]
            lo, hi = 12 * r.ci95_lo_pct, 12 * r.ci95_hi_pct
            ax.plot([lo, hi], [i + off] * 2, color=col, lw=2.2, solid_capstyle="round")
            ax.scatter(r.ann_mean_pct, i + off, color=col, s=55, zorder=3, edgecolor=SURFACE, lw=1.2)
            ax.text(max(hi, 0) + 0.12, i + off, f"{lab}  {r.ann_mean_pct:+.1f}%/yr", va="center", fontsize=9,
                    color=col, fontweight="bold")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, ["", ""])
    ax.set_xlim(-2.2, 4.2)
    ax.set_xlabel("Gross return on committed capital, % per year\n(dot = mean, bar = 95% block-bootstrap CI)")
    ax.set_title("B. …but only country pairs make money", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(axes[0].get_ylim())
    title(fig, "Convergence is not evidence",
          "Textbook distance pairs (top 20, 2-sd trigger, one-day wait). Placebo market: same volatilities, fat tails and "
          "correlations, no relative mean reversion")
    save(fig, "fig1_convergence_is_not_evidence")


def fig2_century():
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    for u, col in (("industries", REAL), ("countries", GOOD)):
        raw = Raw(u)
        m = raw.monthly("real", "D20_w1")
        p = raw.monthly("N1", "D20_w1")
        idx = m.index.to_timestamp()
        roll = 1200 * m.rolling(120, min_periods=120).mean()
        proll = 1200 * p.rolling(120, min_periods=120).mean()
        ax.plot(idx, roll.values, color=col, lw=2, label=f"{U[u]}: real market")
        if u == "industries":
            ax.plot(idx, proll.values, color=PLACEBO, lw=1.5, ls="--", label=f"{U[u]}: placebo market")
    ax.axhline(0, color=INK2, lw=1)
    ax.axvspan(pd.Timestamp("1962-01-01"), pd.Timestamp("2002-12-31"), color=GRID, alpha=0.45, lw=0)
    ax.text(pd.Timestamp("1982-06-01"), 3.0, "1962–2002: the sample in which pairs trading\nwas made famous on single stocks (GGR 2006)",
            ha="center", fontsize=9, color=INK2)
    ax.set_ylabel("Trailing 10-year mean, % per year (gross)")
    ax.set_ylim(-5, 4)
    ax.legend(loc="lower right", fontsize=9)
    title(fig, "A century of industry pairs: mostly negative",
          "Textbook distance rule, trailing 10-year average return; the country-ETF line starts once 10 years of "
          "tradable history exist")
    save(fig, "fig2_century")


def fig3_mechanism():
    d = pd.read_csv(RES / "reversion_by_distance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True, gridspec_kw={"wspace": 0.08})
    labels = ["0–0.5", "0.5–1", "1–1.5", "1.5–2", "2–3", "> 3"]
    for ax, u in zip(axes, ("industries", "countries")):
        x = np.arange(len(labels))
        for j, (h, col) in enumerate(((5, "#9ec5f4"), (21, "#5598e7"), (63, "#184f95"))):
            y = d[(d.universe == u) & (d.h == h)].sort_values("abs_z_from").excess_move.values
            ax.bar(x + (j - 1) * 0.26, y, 0.26, color=col, label=f"next {h} days", zorder=3)
        ax.axhline(0, color=INK2, lw=1)
        ax.axvspan(3.5, 5.5, color=GRID, alpha=0.5, lw=0, zorder=0)
        ax.text(4.5, ax.get_ylim()[1] if u == "countries" else 0.2, "the rule trades here", ha="center", va="top",
                fontsize=9, color=INK2)
        ax.set_xticks(x, labels)
        ax.set_xlabel("Distance of the spread from zero today\n(formation standard deviations)")
        ax.set_title(U[u], loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Move back toward zero, real minus placebo\n(formation sd; > 0 = mean reversion)")
    axes[1].legend(loc="upper left", fontsize=9)
    for ax in axes:
        ax.set_ylim(-0.32, 0.32)
    title(fig, "Industry spreads keep drifting apart; country spreads come back",
          "Average move of the spread over the next 5, 21 and 63 days (after a one-day wait), in excess of the same "
          "measure in the placebo market")
    save(fig, "fig3_where_reversion_lives")


def fig4_variance_ratios():
    v = pd.read_csv(RES / "variance_ratios.csv")
    full = {"industries": "Full 1927-2026", "countries": "Full 2002-2026"}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for u, col in (("industries", REAL), ("countries", GOOD)):
        x = v[(v.universe == u) & (v.period == full[u]) & (v.config == "D20_w1")].sort_values("h")
        ax.plot(x.h, x.real_over_placebo, "o-", color=col, lw=2, label=U[u])
        ax.text(x.h.iloc[-1] * 1.04, x.real_over_placebo.iloc[-1], f"{x.real_over_placebo.iloc[-1]:.2f}",
                color=col, va="center", fontsize=9, fontweight="bold")
    ax.axhline(1, color=PLACEBO, lw=1.5, ls="--")
    ax.text(1.05, 1.01, "placebo market = 1 (no dependence)", color=PLACEBO, fontsize=9, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 21, 42, 63], ["1", "2", "5", "10", "21", "42", "63"])
    ax.set_xlabel("Horizon h (trading days)")
    ax.set_ylabel("Variance ratio of traded spreads,\nreal ÷ placebo")
    ax.set_ylim(0.55, 1.08)
    ax.legend(loc="lower left", fontsize=9)
    title(fig, "Both universes mean-revert at short horizons; only countries keep reverting",
          "VR(h) = variance of h-day spread changes ÷ h × 1-day variance, pooled over the 20 traded pairs; "
          "flat after 2 weeks = no further reversion")
    save(fig, "fig4_variance_ratios")


def fig5_screen():
    s = pd.read_csv(RES / "coint_screen.csv")
    full = {"industries": "Full 1927-2026", "countries": "Full 2002-2026"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"wspace": 0.35})
    for i, u in enumerate(("industries", "countries")):
        x = s[(s.universe == u) & (s.period == full[u])].set_index("source")
        r, p = 100 * x.loc["real", "share_sig_p05"], 100 * x.loc["N1", "share_sig_p05"]
        ax = axes[0]
        ax.barh(i + 0.18, r, 0.34, color=REAL)
        ax.barh(i - 0.18, p, 0.34, color=PLACEBO)
        ax.text(r + 0.15, i + 0.18, f"Real {r:.1f}%", va="center", color=REAL, fontsize=9, fontweight="bold")
        ax.text(p + 0.15, i - 0.18, f"Placebo {p:.1f}%  (≈{100 * p / r:.0f}% of real hits expected by chance)",
                va="center", color=PLACEBO, fontsize=9, fontweight="bold")
        ax = axes[1]
        for off, src, col, lab in ((0.18, "real", REAL, "Real"), (-0.18, "N1", PLACEBO, "Placebo")):
            a, b = 100 * x.loc[src, "persist_if_sig"], 100 * x.loc[src, "persist_if_not"]
            ax.plot([b, a], [i + off] * 2, color=col, lw=2)
            ax.scatter([b], [i + off], color=SURFACE, edgecolor=col, s=45, zorder=3, lw=1.5)
            ax.scatter([a], [i + off], color=col, s=45, zorder=3)
            ax.text(max(a, b) + 0.4, i + off, f"{lab}: {b:.1f}% → {a:.1f}%", va="center", color=col, fontsize=9,
                    fontweight="bold")
    axes[0].invert_yaxis()
    for ax in axes:
        ax.set_yticks([0, 1], [U["industries"], U["countries"]] if ax is axes[0] else ["", ""])
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(0, 21)
    axes[0].set_xlabel("Pairs with Engle–Granger p < 0.05 in the formation year (%)")
    axes[0].set_title("A. Most 'cointegrated' pairs are false positives", loc="left")
    axes[1].set_xlim(0, 22)
    axes[1].set_xlabel("Chance of p < 0.05 again next year (%)\n○ if not significant this year, ● if significant")
    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].set_title("B. Significance barely persists", loc="left")
    title(fig, "Cointegration screens mostly find noise",
          "One test per pair per non-overlapping year; placebo = same assets with relative mean reversion removed")
    save(fig, "fig5_cointegration_screen")


def fig6_costs():
    fig, ax = plt.subplots(figsize=(9, 4.3))
    raw = Raw("countries")
    cs = np.arange(0, 26, 1.0)
    for (a, b, lab, col) in (("2002-01", "2026-11", "2002–2026", GOOD), ("2010-01", "2026-11", "2010–2026", REAL)):
        from evaluate import sl
        from stats import nw_t
        means, lo, hi = [], [], []
        for c in cs:
            x = sl(raw.monthly("real", "D20_w1", cost_bp=c), a, b)
            m, t = nw_t(x.values)
            se = abs(m / t)
            means.append(1200 * m)
            lo.append(1200 * (m - 1.96 * se))
            hi.append(1200 * (m + 1.96 * se))
        ax.plot(cs, means, color=col, lw=2, label=f"Country ETFs, {lab}")
        ax.fill_between(cs, lo, hi, color=col, alpha=0.12, lw=0)
    ax.axhline(0, color=INK2, lw=1)
    ax.axvline(5, color=MUTED, lw=1, ls=":")
    ax.text(5.3, 2.6, "pre-registered\nrealistic cost (5 bp)", fontsize=9, color=INK2, va="top")
    ax.set_xlabel("Cost per leg per one-way trade (basis points)")
    ax.set_ylabel("Net return, % per year\n(band = 95% Newey–West CI)")
    ax.legend(loc="upper right", fontsize=9)
    title(fig, "What survives costs", "Distance rule on 22 country ETFs; each round trip trades four legs")
    save(fig, "fig6_costs")


def fig7_stress():
    s = pd.read_csv(RES / "stress_windows.csv")
    s = s[~s.window.str.startswith("All")]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    rows = []
    for u in ("industries", "countries"):
        for _, r in s[s.universe == u].iterrows():
            rows.append((f"{r.window}  ({'industries' if u == 'industries' else 'countries'})", r.edge_cum_pct))
    y = np.arange(len(rows))
    vals = [v for _, v in rows]
    ax.barh(y, vals, color=[GOOD if v > 0 else BAD for v in vals], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + (0.25 if v >= 0 else -0.25), i, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right",
                fontsize=9)
    ax.set_yticks(y, [r for r, _ in rows])
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("Cumulative return over the window, real minus placebo (%)")
    ax.grid(axis="y", visible=False)
    title(fig, "Crises do not rescue pairs trading", "Distance rule, one-day wait, gross")
    save(fig, "fig7_stress_windows")


if __name__ == "__main__":
    fig1_star()
    fig2_century()
    fig3_mechanism()
    fig4_variance_ratios()
    fig5_screen()
    fig6_costs()
    fig7_stress()
    print("figures written to", FIG)
