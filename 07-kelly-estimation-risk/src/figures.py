"""Publication figures for project 07. Run after backtest.py, simulate.py and evaluate.py."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import ASSETS  # noqa: E402
from evaluate import STRESS, load  # noqa: E402
from kelly import expected_plugin_growth, optimal_multiplier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens shared with project 01 (validated categorical order) ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#a51c1c", "#f08a89", "#f0efec", "#6da7ec", "#184f95"])
ASSET_LABEL = {"SPY": "US equities (SPY)", "EEM": "EM equities (EEM)", "TLT": "20y+ Treasuries (TLT)",
               "HYG": "High yield credit (HYG)", "GLD": "Gold (GLD)", "USO": "Crude oil (USO)",
               "URA": "Uranium (URA)"}
KEY = {  # strategy -> (colour, linestyle, short label)
    "Prior-Sharpe GARCH": (CAT[0], "-", "Prior-Sharpe, GARCH vol"),
    "Plug-in Kelly": (CAT[1], "-", "Plug-in Kelly"),
    "Prior-Sharpe static": (CAT[2], "-", "Prior-Sharpe, static vol"),
    "Half Kelly": (CAT[3], "-", "Half Kelly"),
    "Shrinkage Kelly": (CAT[4], "-", "Shrinkage Kelly (1-1/t²)"),
    "Buy & hold": (MUTED, "-", "Buy & hold"),
    "Oracle constant Kelly": (INK, (0, (4, 3)), "Oracle constant Kelly (hindsight)"),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
})
PCT = matplotlib.ticker.PercentFormatter(1, decimals=0)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def title(fig, t, sub, y=1.0):
    fig.text(0.01, y, t, fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(0.01, y - 0.035, sub, fontsize=10, color=INK2, ha="left", va="bottom")


# ---------------------------------------------------------------- fig 1: the fraction you cannot know
def fig_knowability(k):
    k = k.set_index("asset").loc[ASSETS[::-1]]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    y = np.arange(len(k))
    ax.axvspan(-2, 0, color="#f0efec", zorder=0)
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(1, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(1.06, -0.75, "buy & hold", fontsize=8.5, color=INK2)
    ax.text(-1.9, -0.75, "short", fontsize=8.5, color=INK2)
    ax.set_ylim(-0.95, len(k) - 0.5)
    for i, (a, r) in enumerate(k.iterrows()):
        ax.plot([r.f_lo90, r.f_hi90], [i, i], color=CAT[0], lw=3, solid_capstyle="round", alpha=0.35)
        ax.plot(r.kelly_f, i, "o", color=CAT[0], ms=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(r.f_hi90 + 0.12, i, f"{r.kelly_f:+.1f}  [{r.f_lo90:+.1f}, {r.f_hi90:+.1f}]",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [ASSET_LABEL[a] for a in k.index])
    ax.set_xlim(-2, 8.6)
    ax.set_xlabel("Full-sample Kelly fraction f = μ/σ², 90% CI (Newey–West s.e. of μ, σ treated as known)")
    ax.set_title("Best estimate of the Kelly bet, 2006–2026", loc="left")
    ax.grid(axis="y", visible=False)
    ys = k.years_sign.clip(upper=1e4)
    ya = k.years_abs_05
    for i, a in enumerate(k.index):
        ax2.plot([min(ys[a], ya[a]), max(ys[a], ya[a])], [i, i], color=AXIS, lw=1.5, zorder=1)
        ax2.plot(ys[a], i, "o", color=CAT[1], ms=8, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax2.plot(ya[a], i, "D", color=CAT[0], ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        lab = f"{ys[a]:,.0f}" if ys[a] < 1e4 else ">10,000"
        box = dict(facecolor=SURFACE, edgecolor="none", pad=0.6)
        ax2.text(ys[a], i + 0.17, lab, ha="center", va="bottom", fontsize=8, color=INK2, bbox=box, zorder=4)
        ax2.text(ya[a], i - 0.17, f"{ya[a]:,.0f}", ha="center", va="top", fontsize=8, color=INK2, bbox=box, zorder=4)
    ax2.axvline(k.years.max(), color=INK, lw=1, ls=(0, (4, 3)))
    ax2.text(k.years.max() * 1.1, -0.75, "data we have (~20 yrs)", fontsize=8.5, color=INK2, va="center")
    ax2.set_ylim(-0.95, len(k) - 0.5)
    ax2.set_xscale("log")
    ax2.set_xlim(3, 3e4)
    ax2.set_yticks(y, [""] * len(y))
    ax2.set_xlabel("Years of data needed (90% confidence, HAC)")
    ax2.set_title("Years to know it", loc="left")
    ax2.grid(axis="y", visible=False)
    ax2.legend(handles=[plt.Line2D([], [], color=CAT[1], marker="o", lw=0, label="the sign (long or short)"),
                        plt.Line2D([], [], color=CAT[0], marker="D", lw=0, label="f to within ±0.5")],
               loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    title(fig, "The Kelly fraction you cannot know",
          "Twenty years of daily data pin down the sign of the bet for only two of seven assets, and its size for none", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig1_knowability")


# ---------------------------------------------------------------- fig 2: theory map
def fig_theory_map(k):
    th = np.linspace(0.05, 1.0, 200)
    T = np.logspace(np.log10(2), np.log10(100), 200)
    TH, TT = np.meshgrid(th, T)
    Z = expected_plugin_growth(1.0, TH, TT)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1)
    pc = ax.pcolormesh(TH, TT, np.clip(Z, -1.5, 1), cmap=DIVERGING, norm=norm, shading="auto", rasterized=True)
    cs = ax.contour(TH, TT, optimal_multiplier(TH, TT), levels=[0.25, 0.5, 0.75, 0.9], colors=INK, linewidths=0.8)
    ax.clabel(cs, fmt=lambda v: f"c* = {v:.2f}", fontsize=8)
    ax.contour(TH, TT, Z, levels=[0], colors=INK, linewidths=2)
    ax.text(0.13, 3.0, "full Kelly expected\nto grow slower\nthan cash", fontsize=9, color=INK)
    for a, r in k.set_index("asset").iterrows():
        if abs(r.sharpe) < 0.05:  # oil and uranium
            continue
        ax.plot(abs(r.sharpe), r.years, "o", color=SURFACE, ms=9, markeredgecolor=INK, markeredgewidth=1.5)
        ax.annotate(a, (abs(r.sharpe), r.years), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_yscale("log")
    ax.set_yticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("True annual Sharpe ratio θ of the asset")
    ax.set_ylabel("Years of data used to estimate the mean (T)")
    cb = fig.colorbar(pc, ax=ax, pad=0.02)
    cb.set_label("Expected growth of plug-in full Kelly, as share of the oracle growth θ²/2")
    cb.ax.yaxis.set_major_formatter(PCT)
    ax.grid(False)
    title(fig, "Where full Kelly is worse than cash",
          "E[g] = θ² − (θ² + 1/T)/2 per unit variance. Thin lines: the optimal fraction of Kelly c* = θ²T/(θ²T+1).",
          y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig2_theory_map")


# ---------------------------------------------------------------- fig 3: plug-in fraction paths
def fig_fraction_paths(inp):
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharex=True)
    for ax, a in zip(axes.flat, ASSETS):
        g = inp[inp.asset == a]
        ax.axhline(0, color=AXIS, lw=1)
        ax.plot(g.origin, g["f:Plug-in Kelly 5y"], color=CAT[3], lw=1.2, label="Plug-in, 5-year window")
        ax.plot(g.origin, g["f:Plug-in Kelly"], color=CAT[1], lw=2, label="Plug-in, expanding window")
        ax.plot(g.origin, g["f:Prior-Sharpe GARCH"], color=CAT[0], lw=1.2, label="Prior-Sharpe, GARCH vol")
        ax.axhline(g["f:Oracle constant Kelly"].iloc[0], color=INK, lw=1.2, ls=(0, (4, 3)),
                   label="Oracle constant Kelly")
        ax.set_ylim(-2.15, 4.15)
        ax.set_title(ASSET_LABEL[a], loc="left", fontsize=10)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(6))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    axes.flat[-1].axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h, l, loc="center left", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of wealth in the asset")
    title(fig, "Walk-forward Kelly bets swing from maximum short to maximum long",
          "Weekly position chosen with data up to the previous close, clipped to [−2, 4]", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig3_fraction_paths")


# ---------------------------------------------------------------- fig 4: realised growth
def fig_growth(summ, pooled):
    order = pooled.set_index("strategy").loc[[s for s in pooled.strategy if s != "Cash"]].growth.sort_values()
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(order))
    fam_col = {"benchmark": MUTED, "oracle": INK, "plug-in": CAT[1], "prior": CAT[0], "hybrid": CAT[2]}
    fam = pooled.set_index("strategy").family
    cash = pooled.set_index("strategy").growth["Cash"]
    ax.axvline(cash, color=AXIS, lw=1)
    ax.text(cash, len(order) - 0.3, " cash", fontsize=8.5, color=INK2)
    for i, (s, v) in enumerate(order.items()):
        pts = summ[summ.strategy == s].growth.values
        ax.plot(pts, np.full(len(pts), i), "|", color=MUTED, ms=9, mew=1.2)
        ax.plot([cash, v], [i, i], color=fam_col[fam[s]], lw=2)
        ax.plot(v, i, "o", color=fam_col[fam[s]], ms=9, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(0.245, i, f"{v:+.1%}", va="center", ha="right", fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks(y, order.index)
    ax.set_xlim(-0.16, 0.25)
    ax.axvspan(0.205, 0.25, color=SURFACE, zorder=0)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("Annualised log growth of wealth, average of 7 assets (ticks: individual assets)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=c, marker="o", lw=2, label=l) for l, c in
               [("Plug-in mean", CAT[1]), ("Prior Sharpe, no mean estimate", CAT[0]),
                ("Hybrid", CAT[2]), ("Infeasible oracle", INK), ("Benchmark", MUTED)]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=5, fontsize=8.5)
    title(fig, "Not estimating the mean beats estimating it",
          "Weekly walk-forward, 2008–2026, 2 bp costs, leverage at T-bill + 0.5%", y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "fig4_realised_growth")


# ---------------------------------------------------------------- fig 5: wealth paths
def fig_wealth(w):
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.8), sharex=True)
    show = ["Oracle constant Kelly", "Buy & hold", "Plug-in Kelly", "Half Kelly", "Prior-Sharpe GARCH"]
    for ax, a in zip(axes.flat, ASSETS):
        for s in show:
            g = w[(w.asset == a) & (w.strategy == s)]
            c, ls, lab = KEY[s]
            ax.plot(g.origin, np.exp(np.cumsum(g.logret.values)), color=c, ls=ls,
                    lw=2 if s in ("Plug-in Kelly", "Prior-Sharpe GARCH") else 1.2, label=lab)
        ax.set_yscale("log")
        ax.set_title(ASSET_LABEL[a], loc="left", fontsize=10)
        fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(6))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    axes.flat[-1].axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h, l, loc="center left", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Wealth (log scale)")
    title(fig, "Growth of $1 under five sizing rules", "Weekly walk-forward, all positions out of sample", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig5_wealth_paths")


# ---------------------------------------------------------------- fig 6: multiplier under real dynamics
def fig_multiplier(sim):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    theta = 0.40
    for ax, T in zip(axes, [5, 10, 20]):
        d = sim[sim["T"] == T]
        for a in ASSETS:
            g = d[d.asset == a]
            ax.plot(g.c, g.mean_growth, color=MUTED, lw=1, alpha=0.8)
        pooled = d.groupby("c").mean_growth.mean()
        ax.plot(pooled.index, pooled.values, color=CAT[0], lw=2.2, label="Real dynamics, 7-asset mean")
        cg = np.linspace(0, 1.5, 100)
        ax.plot(cg, expected_plugin_growth(cg, theta, T) * theta**2 / 2, color=INK, lw=1.4,
                ls=(0, (4, 3)), label="GBM theory")
        cs = optimal_multiplier(theta, T)
        ax.axvline(cs, color=INK, lw=0.8)
        best = pooled.idxmax()
        ax.plot(best, pooled.max(), "o", color=CAT[0], ms=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(cs + 0.03, -0.11, f"theory c* = {cs:.2f}\nsimulated c* = {best:.2f}", fontsize=8.5, color=INK2)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_ylim(-0.13, 0.1)
        ax.set_title(f"{T} years of estimation data", loc="left")
        ax.set_xlabel("Fraction c of the plug-in Kelly bet")
        ax.yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("Mean annual log growth over the next 10 years")
    axes[0].legend(loc="upper left", fontsize=8.5)
    title(fig, "How much of the estimated Kelly bet to take",
          "Block-bootstrap paths of each ETF's daily returns, re-centred to a true Sharpe of 0.4 (grey: single assets)",
          y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig6_multiplier_real_dynamics")


# ---------------------------------------------------------------- fig 7: drawdown law
def fig_drawdown(dd):
    fig, ax = plt.subplots(figsize=(8, 5.4))
    marks = {"GBM, fixed fraction": (MUTED, "o"), "Real dynamics, fixed fraction": (CAT[1], "s"),
             "Real dynamics, vol-timed (EWMA)": (CAT[0], "D")}
    lim = (1e-4, 1)
    ax.plot(lim, lim, color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(2e-4, 1.6e-4, "law holds", rotation=33, fontsize=8.5, color=INK2)
    for v, (c, m) in marks.items():
        d = dd[(dd.variant == v) & (dd.p_ever > 0)]
        ax.scatter(d.theory, d.p_ever, s=26, color=c, marker=m, edgecolor=SURFACE, linewidth=0.8,
                   label=v, alpha=0.9, zorder=3)
    for _, r in dd[(dd.variant == "Real dynamics, fixed fraction") & (dd.asset == "USO") & (dd.c == 0.5) & (dd.x == 0.1)].iterrows():
        ax.annotate("crude oil, half Kelly,\nlosing 90%: 9× the law", (r.theory, r.p_ever), xytext=(8, 2), textcoords="offset points",
                    fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Kelly drawdown law: P(wealth ever falls to x) = x^(2/c − 1)")
    ax.set_ylabel("Simulated probability within 40 years")
    ax.legend(loc="upper left", fontsize=8.5)
    title(fig, "The drawdown law holds at the centre and breaks in the tail",
          "7 ETFs × fractions c ∈ {1, ½, ¼} × thresholds x ∈ {0.5, 0.3, 0.1}, 1,500 paths each", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig7_drawdown_law")


# ---------------------------------------------------------------- fig 8: stress windows
def fig_stress(st):
    show = ["Buy & hold", "Plug-in Kelly", "Half Kelly", "Shrinkage Kelly", "Prior-Sharpe static", "Prior-Sharpe GARCH"]
    pooled = st.groupby(["window", "strategy"]).cum_return.mean().unstack().loc[list(STRESS), show]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    width = 0.13
    x = np.arange(len(pooled))
    for i, s in enumerate(show):
        c = KEY[s][0]
        vals = pooled[s].values
        ax.bar(x + (i - 2.5) * width, vals, width * 0.88, color=c, label=KEY[s][2])
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x, pooled.index)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Cumulative return, average of 7 assets")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, fontsize=8.5, loc="lower left")
    title(fig, "Volatility timing cut the 2008 loss by two thirds, and did not help in 2020 or 2022",
          "Cumulative return over each stress window, equal-weight average of the seven single-asset strategies", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig8_stress_windows")


def main():
    k = pd.read_csv(RES / "knowability.csv")
    inp = pd.read_csv(RES / "inputs.csv.gz", parse_dates=["origin"])
    w = load()
    fig_knowability(k)
    fig_theory_map(k)
    fig_fraction_paths(inp)
    fig_growth(pd.read_csv(RES / "summary_by_asset.csv"), pd.read_csv(RES / "summary_pooled.csv"))
    fig_wealth(w)
    fig_multiplier(pd.read_csv(RES / "sim_real_multiplier.csv"))
    fig_drawdown(pd.read_csv(RES / "sim_drawdown_law.csv"))
    fig_stress(pd.read_csv(RES / "stress_windows.csv"))
    print("figures saved to", FIG)


if __name__ == "__main__":
    main()
