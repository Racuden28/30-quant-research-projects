"""Publication figures for project 16. Run after frontier.py, simulation.py, which_delta.py, costs.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import FIG, H_GRID, RES, T_DAYS  # noqa: E402

# ---- design tokens shared with projects 01 and 04 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN, GOLD, PINK, DKGREEN, PURPLE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                     "#008300", "#7b5bd6")
ASSETS = ["SPX", "SPY", "EEM", "HYG", "USO", "TLT", "GLD", "URA"]
ASSET_LABEL = {"SPX": "S&P 500 index 1928-2026", "SPY": "US equities (SPY)", "EEM": "EM equities (EEM)",
               "TLT": "20y+ Treasuries (TLT)", "HYG": "High yield (HYG)", "GLD": "Gold (GLD)",
               "USO": "Crude oil (USO)", "URA": "Uranium (URA)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
})


def no_minor(ax):
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def title(fig, t, sub, y=1.0):
    fig.text(0.01, y, t, fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(0.01, y - 0.035, sub, fontsize=10, color=INK2, ha="left", va="bottom")


def n_hedges(h):
    return int(np.ceil(T_DAYS / h))


def fig1_star():
    """The star chart: what daily hedging buys, textbook vs real, and the floor by asset."""
    fs = pd.read_csv(RES / "frontier_summary.csv")
    sim = pd.read_csv(RES / "sim_factorial.csv")
    fl = pd.read_csv(RES / "floor_shares.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    title(fig, "Hedging more often barely helps: most delta-hedging risk is a floor",
          "One-month ATM calls sold at a volatility forecast and delta-hedged every h days. "
          "Left: variance of the hedging error relative to hedging once a week.", y=1.04)
    hs = [h for h in H_GRID if h <= 10]
    N = np.array([n_hedges(h) for h in hs])
    # textbook GBM law: variance proportional to 1/N
    xs = np.linspace(2, 21, 100)
    ax.plot(xs, 5 / xs, color=MUTED, lw=2, ls="--", label="Textbook law (GBM, correct vol): variance ∝ 1/N")
    spx = fs[(fs.asset == "SPX") & (fs.quote == "forecast")].iloc[0]
    v = np.array([spx[f"h{h}"] ** 2 for h in hs]) / spx["h5"] ** 2
    ax.plot(N, v, color=BLUE, lw=2.4, marker="o", label="S&P 500 1928-2026, real paths (EWMA quote)")
    vx = fs[(fs.asset == "SPX") & (fs.quote == "vix")].iloc[0]
    v2 = np.array([vx[f"h{h}"] ** 2 for h in hs]) / vx["h5"] ** 2
    ax.plot(N, v2, color=GREEN, lw=2.0, marker="s", label="S&P 500 1990-2026, real paths (sold at VIX)")
    g = sim[(sim.world == "GJR-t") & (sim.quote == "EWMA")].iloc[0]
    v3 = np.array([g[f"std_h{h}"] ** 2 for h in hs]) / g["std_h5"] ** 2
    ax.plot(N, v3, color=ORANGE, lw=1.6, marker="^", ls=":", label="Simulated GJR-GARCH-t world fit to the S&P")
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 7, 11, 21])
    ax.set_xticklabels(["3", "5\n(weekly)", "7", "11", "21\n(daily)"])
    no_minor(ax)
    ax.set_xlim(2.7, 23)
    ax.set_xlabel("Number of hedges over the option's month (N)")
    ax.set_ylabel("Variance of hedging error / variance with weekly hedging")
    ax.set_ylim(0, 2.0)
    ax.axhline(1, color=AXIS, lw=0.8)
    ax.annotate(f"daily keeps {v[0]:.0%} of the weekly\nvariance on the S&P 500;\nthe law promises {5/21:.0%}",
                xy=(21, v[0]), xytext=(7.5, 0.15), fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
    ax.legend(loc="upper right", fontsize=8.6)
    ax.set_title("A. Going from weekly to daily hedging", loc="left")

    f = fl[fl.quote == "forecast"].set_index("asset").loc[ASSETS]
    y = np.arange(len(ASSETS))[::-1]
    bx.barh(y, f.share, color=BLUE, alpha=0.85, height=0.6, label="Floor share (EWMA quote)")
    bx.errorbar(f.share, y, xerr=[f.share - f.share_lo, f.share_hi - f.share], fmt="none", ecolor=INK, lw=1,
                capsize=3, label="95% block-bootstrap CI")
    vxr = fl[(fl.asset == "SPX") & (fl.quote == "vix")].iloc[0]
    bx.plot([vxr.share], [y[0]], marker="D", color=GREEN, ms=8, ls="none", label="S&P 500 sold at VIX (1990-2026)")
    gb = sim[(sim.world == "GBM") & (sim.quote == "model-optimal")].iloc[0]
    bx.axvline(gb.share, color=MUTED, ls="--", lw=1.5, label=f"Textbook GBM world: {gb.share:+.2f} (≈ 0)")
    bx.set_yticks(y)
    bx.set_yticklabels([ASSET_LABEL[a] for a in ASSETS])
    bx.set_xlim(-0.15, 1.1)
    bx.set_xlabel("Share of daily-hedging variance left with continuous hedging\n(extrapolated with the fitted law Var = a + b·T/N)")
    bx.legend(loc="lower left", fontsize=8.2, bbox_to_anchor=(0.0, -0.36), ncol=2)
    bx.set_title("B. The floor, by underlying", loc="left")
    fig.tight_layout()
    save(fig, "fig1_rebalancing_floor")


def fig2_factorial():
    sim = pd.read_csv(RES / "sim_factorial.csv")
    real = pd.read_csv(RES / "floor_shares.csv")
    r = real[(real.asset == "SPX") & (real.quote == "forecast")].iloc[0]
    worlds = ["GBM", "iid-t5", "Merton", "GARCH-N", "GJR-t"]
    wl = {"GBM": "GBM", "iid-t5": "iid Student-t(5)", "Merton": "Merton jumps", "GARCH-N": "GARCH, normal",
          "GJR-t": "GJR-GARCH-t\n(fit to S&P)"}
    quotes = [("true integrated vol", PURPLE, "Quote = vol the month actually delivered (ablation)"),
              ("model-optimal", GREEN, "Quote = best forecast inside that world"),
              ("EWMA", BLUE, "Quote = EWMA forecast (what the real hedger uses)")]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    title(fig, "What the floor is made of: jumps, tails and an unknowable month of volatility",
          "Floor share in simulated worlds (50,000 one-month options each), against the real S&P 500", y=1.02)
    x = np.arange(len(worlds))
    w = 0.26
    for j, (q, c, lab) in enumerate(quotes):
        vals, lo, hi, xs = [], [], [], []
        for i, wd in enumerate(worlds):
            s = sim[(sim.world == wd) & (sim.quote == q)]
            if q == "true integrated vol" and s.empty:
                s = sim[(sim.world == wd) & (sim.quote == "model-optimal")]  # constant-vol worlds: same thing
            s = s.iloc[0]
            vals.append(s.share), lo.append(s.share_lo), hi.append(s.share_hi), xs.append(i + (j - 1) * w)
        vals, lo, hi = map(np.array, (vals, lo, hi))
        ax.bar(xs, vals, width=w, color=c, alpha=0.85, label=lab)
        ax.errorbar(xs, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=INK, lw=0.9, capsize=2)
    ax.axhline(r.share, color=ORANGE, lw=2, label=f"Real S&P 500 1928-2026, EWMA quote: {r.share:.2f}")
    ax.axhspan(r.share_lo, r.share_hi, color=ORANGE, alpha=0.12, lw=0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([wl[w_] for w_ in worlds])
    ax.set_ylabel("Floor share  a / (a + b)")
    ax.set_ylim(-0.75, 1.1)
    ax.text(0, -0.6, "constant-vol worlds: the three quotes\nfor GBM, t5 and Merton are the true σ\nexcept EWMA",
            fontsize=7.5, color=MUTED, ha="center")
    ax.legend(loc="upper left", fontsize=8.3, ncol=2)
    fig.tight_layout()
    save(fig, "fig2_what_makes_the_floor")


def fig3_stress():
    st = pd.read_csv(RES / "stress_windows.csv")
    fs = pd.read_csv(RES / "frontier_summary.csv")
    s = st[(st.asset == "SPX") & (st.quote == "forecast")]
    full = fs[(fs.asset == "SPX") & (fs.quote == "forecast")].iloc[0]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    title(fig, "In a crash, hedging daily instead of weekly buys almost nothing",
          "S&P 500, options sold at the EWMA forecast with start dates inside each window", y=1.03)
    names = list(s.window)
    gain = list(s.gain_5_to_1)
    names = ["All 1928-2026"] + names
    gain = [1 - full.h1 ** 2 / full.h5 ** 2] + gain
    y = np.arange(len(names))[::-1]
    cols = [MUTED] + [ORANGE] * (len(names) - 1)
    ax.barh(y, gain, color=cols, height=0.6)
    ax.axvline(1 - 5 / 21, color=INK2, ls="--", lw=1.2)
    ax.text(1 - 5 / 21, y[0] + 0.55, "textbook: 76%", fontsize=8.5, color=INK2, ha="center")
    for yi, g_ in zip(y, gain):
        ax.text(g_ + 0.01 if g_ >= 0 else 0.01, yi, f"{g_:.0%}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(-0.05, 0.9)
    ax.set_xlabel("Variance removed by hedging daily instead of weekly")
    ax.set_title("A. Gain from daily vs weekly hedging", loc="left")
    hs = [1, 2, 3, 5, 10, 21]
    for _, row in s.iterrows():
        bx.plot([n_hedges(h) for h in hs], [row[f"std_h{h}"] for h in hs], marker="o", lw=1.6,
                label=row.window)
    bx.plot([n_hedges(h) for h in hs], [full[f"h{h}"] for h in hs], marker="o", color=INK, lw=2.4,
            label="All 1928-2026")
    bx.set_xscale("log")
    bx.set_xticks([1, 3, 5, 11, 21])
    bx.set_xticklabels(["1", "3", "5", "11", "21"])
    no_minor(bx)
    bx.set_xlabel("Number of hedges over the month (N)")
    bx.set_ylabel("Std of hedging error (vol points)")
    bx.set_title("B. Hedging-error std by frequency", loc="left")
    bx.legend(fontsize=7.8)
    fig.tight_layout()
    save(fig, "fig3_stress_windows")


def fig4_which_delta():
    t = pd.read_csv(RES / "which_delta_tests.csv")
    t = t[t.baseline == "floating"].set_index("rule").loc[["fixed", "ewma", "mv"]]
    lab = {"fixed": "Delta at the vol\nthe option was sold at", "ewma": "Delta at today's\nEWMA realised vol",
           "mv": "Minimum-variance delta\n(today's VIX + leverage slope)"}
    fig, ax = plt.subplots(figsize=(10, 4.4))
    title(fig, "Which volatility belongs inside the delta?",
          "S&P 500 one-month ATM calls sold at VIX, hedged daily, 1992-2026 (8,718 options). "
          "Baseline: Black-Scholes delta at today's VIX.", y=1.03)
    y = np.arange(3)[::-1]
    for k, (col, c, name, off) in enumerate([("terminal", BLUE, "P&L at expiry (model-free)", 0.17),
                                            ("daily", GREEN, "Daily mark-to-market (VIX marks)", -0.17)]):
        g = t[f"gain_{col}"].to_numpy()
        lo, hi = t[f"gain_{col}_lo"].to_numpy(), t[f"gain_{col}_hi"].to_numpy()
        ax.barh(y + off, g, height=0.32, color=c, alpha=0.85, label=name)
        ax.errorbar(g, y + off, xerr=[g - lo, hi - g], fmt="none", ecolor=INK, lw=1, capsize=3)
        for yi, gi, l_, h_, p in zip(y + off, g, lo, hi, t[f"p_{col}_holm"]):
            xt = h_ + 0.015 if gi >= 0 else l_ - 0.015
            ax.text(xt, yi, f"{gi:+.0%}  (Holm p = {p:.3f})", va="center", ha="left" if gi >= 0 else "right",
                    fontsize=8.5)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([lab[r] for r in t.index])
    ax.set_xlim(-0.55, 0.85)
    ax.set_xlabel("Reduction in hedging-error variance vs the baseline (95% block-bootstrap CI)")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig4_which_delta")


def fig5_costs():
    fr = pd.read_csv(RES / "costs_frontier.csv")
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    title(fig, "Hedge by delta, not by calendar",
          "S&P 500 one-month ATM calls sold at VIX, 1990-2026. Each point is one rebalancing rule.", y=1.03)
    for ax, k in zip(axs, [5.0, 20.0]):
        d = fr[(fr.asset == "SPX") & (fr.cost_bps == k)]
        c = d[d.family == "calendar"]
        b = d[d.family == "band"]
        ax.plot(c.mean_cost, c.std_pnl, marker="o", color=BLUE, lw=2, label="Every h days")
        for _, r in c.iterrows():
            ax.annotate(f"{int(r.param)}d", (r.mean_cost, r.std_pnl), xytext=(5, 3), textcoords="offset points",
                        fontsize=8, color=BLUE)
        ax.plot(b.mean_cost, b.std_pnl, marker="s", color=ORANGE, lw=2, label="When |Δ − held Δ| > band")
        for _, r in b.iterrows():
            ax.annotate(f"{r.param:.2f}", (r.mean_cost, r.std_pnl), xytext=(5, -10), textcoords="offset points",
                        fontsize=8, color=ORANGE)
        ax.set_title(f"{k:.0f} bps per unit of notional traded", loc="left")
        ax.set_xlabel("Mean transaction cost (vol points)")
        ax.legend(loc="upper right", fontsize=8.5)
    axs[0].set_ylabel("Std of total P&L (vol points)")
    fig.tight_layout()
    save(fig, "fig5_cost_frontier")


def fig6_decades():
    d = pd.read_csv(RES / "spx_by_decade.csv")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    title(fig, "The floor is not a recent accident",
          "S&P 500 by decade of the option's start date: floor share and daily-hedging error", y=1.03)
    f = d[d.quote == "forecast"]
    ax.plot(f.decade, f.share, marker="o", color=BLUE, lw=2.2, label="Floor share (EWMA quote), left axis")
    v = d[d.quote == "vix"]
    ax.plot(v.decade, v.share, marker="s", color=GREEN, lw=2, label="Floor share (sold at VIX), left axis")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Floor share")
    ax.set_xticks(f.decade)
    ax.set_xticklabels([f"{x}s" for x in f.decade])
    bx = ax.twinx()
    bx.bar(f.decade, f.std1, width=4, color=AXIS, alpha=0.6, zorder=0, label="Daily-hedging std (vol pts), right axis")
    bx.set_ylabel("Std of daily-hedging error (vol points)")
    bx.grid(False)
    bx.spines["right"].set_visible(True)
    ax.set_zorder(bx.get_zorder() + 1)
    ax.patch.set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = bx.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8.3)
    fig.tight_layout()
    save(fig, "fig6_floor_by_decade")


def fig7_validation():
    d = pd.read_csv(RES / "derman_check.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    title(fig, "Validation: the engine reproduces the textbook law under GBM",
          "Std of the hedging error of an ATM call, 100,000 GBM paths, against sqrt(π/4)·σ/√N", y=1.03)
    for s_, c in zip([0.1, 0.2, 0.4], [BLUE, ORANGE, GREEN]):
        g = d[d.sigma == s_]
        ax.plot(g.n_rebalances, g.sim_std, marker="o", ls="none", color=c, label=f"simulated, σ = {s_:.0%}")
        n = np.linspace(1, 21, 100)
        ax.plot(n, 100 * np.sqrt(np.pi / 4) * s_ / np.sqrt(n), color=c, lw=1.2, ls="--")
    ax.plot([], [], color=MUTED, ls="--", label="Kamal-Derman approximation")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 3, 5, 11, 21])
    ax.set_xticklabels(["1", "3", "5", "11", "21"])
    no_minor(ax)
    ax.set_xlabel("Number of hedges (N)")
    ax.set_ylabel("Std of hedging error (vol points)")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig7_validation")


def main():
    fig1_star()
    fig2_factorial()
    fig3_stress()
    fig4_which_delta()
    fig5_costs()
    fig6_decades()
    fig7_validation()


if __name__ == "__main__":
    main()
