"""Publication figures for project 01. Run after backtest.py and evaluate.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from evaluate import ALL_MODELS, ASSET_ORDER, MODEL_ORDER, STRESS, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens (validated categorical order; see README "Figure notes") ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
COLORS = dict(zip(MODEL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MARKERS = dict(zip(MODEL_ORDER, ["o", "s", "D", "^", "v", "P"]))
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f08a89", "#a51c1c"])
ASSET_LABEL = {"SPY": "US equities (SPY)", "EEM": "EM equities (EEM)", "TLT": "20y+ Treasuries (TLT)",
               "HYG": "High yield credit (HYG)", "GLD": "Gold (GLD)", "USO": "Crude oil (USO)",
               "URA": "Uranium (URA)"}

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


def title(fig, t, sub):
    fig.text(0.01, 1.0, t, fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(0.01, 0.965, sub, fontsize=10, color=INK2, ha="left", va="bottom")


def fig_conditional(df_cc, ci):
    cc = df_cc[df_cc.h == 5]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    offs = np.linspace(-0.12, 0.12, len(MODEL_ORDER))
    for ax, col, nominal, lab in [(axes[0], "cov90", 0.90, "90% interval"), (axes[1], "cov98", 0.98, "98% interval")]:
        for k, m in enumerate(MODEL_ORDER):
            g = cc[cc.model == m]
            c = ci[(ci.model == m) & (ci.level == nominal) & (ci.vol_q != "gap Q1-Q5")].set_index("vol_q").loc[g.vol_q]
            x = np.arange(len(g)) + offs[k]
            ax.vlines(x, c.ci_low.values, c.ci_high.values, color=COLORS[m], lw=1, alpha=0.7)
            ax.plot(x, g[col].values, color=COLORS[m], lw=2, marker=MARKERS[m], ms=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=m)
        ax.axhline(nominal, color=INK, lw=1, ls=(0, (4, 3)), label=f"Nominal {nominal:.0%}")
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02 if nominal < 0.95 else 0.01))
        ax.set_xticks(range(5), ["Q1\ncalm", "Q2", "Q3", "Q4", "Q5\nstressed"])
        ax.set_xlabel("Volatility state at forecast origin (quintile of trailing 21-day realised vol)")
        ax.set_ylabel(f"Realised coverage of the {lab}")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.set_title(lab, loc="left")
        ax.set_xlim(-0.3, 4.9)
    axes[0].legend(loc="lower left", fontsize=8.5, ncol=2)
    axes[1].legend(handles=[axes[1].get_lines()[-1]], loc="lower left", fontsize=8.5)
    title(fig, "Static simulators are right on average and wrong at both extremes",
          "1-week-ahead coverage by volatility state at the origin, 7 assets pooled, 2008-2026. "
          "Whiskers: 95% block-bootstrap CI")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig1_conditional_coverage")


def fig_pit(df):
    d = df[df.h == 5]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, m in zip(axes.flat, MODEL_ORDER):
        p = d[d.model == m].pit.values
        cnt, _ = np.histogram(p, bins)
        dens = cnt / cnt.sum() * 20
        ax.bar(bins[:-1], dens, width=0.05 - 0.004, align="edge", color=COLORS[m], edgecolor=SURFACE, lw=0)
        ax.axhline(1, color=INK, lw=1, ls=(0, (4, 3)))
        ax.set_title(f"{m}   (PIT s.d. {p.std():.3f}; uniform 0.289)", loc="left", fontsize=9.5)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    for ax in axes[1]:
        ax.set_xlabel("PIT of realised 1-week return")
    title(fig, "PIT histograms: a calibrated simulator is flat",
          "U-shape = intervals too narrow (tails under-predicted); hump = too wide. 1-week horizon, all assets")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig2_pit_histograms")


def fig_heat(by_am):
    d = by_am[by_am.h == 5].pivot(index="asset", columns="model", values="tail_mult_98").loc[ASSET_ORDER, MODEL_ORDER]
    z = np.log2(d.values)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    lim = np.nanmax(np.abs(z))
    im = ax.imshow(z, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = d.values[i, j]
            ax.text(j, i, f"{v:.2f}x", ha="center", va="center", fontsize=9.5,
                    color="white" if abs(z[i, j]) > 0.65 * lim else INK)
    ax.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER, rotation=0, fontsize=9)
    ax.set_yticks(range(len(ASSET_ORDER)), [ASSET_LABEL[a] for a in ASSET_ORDER])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(MODEL_ORDER)), minor=True)
    ax.set_yticks(np.arange(-.5, len(ASSET_ORDER)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ticks = [k for k in (-2, -1, 0, 1, 2) if abs(k) <= lim + 1e-9]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{2.0 ** k:g}x" for k in ticks])
    cb.outline.set_visible(False)
    cb.set_label("Observed / nominal frequency of moves outside the 98% band", color=INK2)
    title(fig, "Tail honesty: how often reality escapes the 98% band",
          "1.00x = honest. Red: tails under-predicted. Blue: bands wider than needed. 1-week horizon")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig3_tail_multiplier_heatmap")


def fig_bands(df, asset="SPY"):
    d = df[(df.h == 5) & (df.asset == asset)]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    for m, alpha in [("GBM", 0.28), ("GJR-GARCH-t", 0.35)]:
        g = d[d.model == m].sort_values("origin")
        ax.fill_between(g.origin, g["q0.010"], g["q0.990"], color=COLORS[m], alpha=alpha, lw=0,
                        label=f"{m} 98% band", step="post")
    g = d[d.model == "GBM"].sort_values("origin")
    out = (g.y < g["q0.010"]) | (g.y > g["q0.990"])
    ax.plot(g.origin, g.y, color=INK2, lw=0.6, label="Realised 1-week log return")
    ax.scatter(g.origin[out], g.y[out], s=22, color="#d03b3b", zorder=5, edgecolor=SURFACE, lw=0.8,
               label=f"Outside GBM band ({out.sum()} weeks vs {0.02 * len(g):.0f} expected)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_ylabel("1-week log return")
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    ax.set_xlim(g.origin.min(), g.origin.max())
    title(fig, f"{ASSET_LABEL[asset]}: a static band breathes too little",
          "Rolling 1-week 98% bands. GBM breaches arrive in clusters (2008, 2020, 2022) while its band is too wide in calm years")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, f"fig4_bands_{asset.lower()}")


def fig_crps(dm):
    d = dm[dm.h == 5]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    models = MODEL_ORDER[1:]
    off = np.linspace(-0.3, 0.3, len(models))
    for k, m in enumerate(models):
        g = d[d.model == m].set_index("asset").loc[ASSET_ORDER]
        y = np.arange(len(ASSET_ORDER)) + off[k]
        sig = np.abs(g.dm_t.values) > 1.96
        ax.scatter(g.skill_vs_gbm.values[sig], y[sig], color=COLORS[m], marker=MARKERS[m], s=55,
                   edgecolor=SURFACE, lw=1, label=m, zorder=3)
        ax.scatter(g.skill_vs_gbm.values[~sig], y[~sig], facecolor=SURFACE, edgecolor=COLORS[m],
                   marker=MARKERS[m], s=55, lw=1.5, zorder=3)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(ASSET_ORDER)), [ASSET_LABEL[a] for a in ASSET_ORDER])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=1))
    ax.set_xlabel("CRPS skill vs GBM (positive = better full-distribution forecast)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    title(fig, "Scoring the whole distribution, not just the tails",
          "Filled = Diebold-Mariano |t| > 1.96 (Newey-West); hollow = not significant. 1-week horizon")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig5_crps_skill")


def fig_stress(st):
    d = st[st.h == 5].pivot(index="window", columns="model", values="exceed98_mult")
    d = d.loc[list(STRESS), MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    x = np.arange(len(d))
    w = 0.8 / len(MODEL_ORDER)
    for k, m in enumerate(MODEL_ORDER):
        ax.bar(x - 0.4 + k * w, d[m].values, width=w - 0.02, color=COLORS[m], label=m, edgecolor=SURFACE, lw=0)
    ax.axhline(1, color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(len(d) - 0.45, 1.05, "honest = 1x", fontsize=8.5, color=INK2, ha="right", va="bottom")
    ax.set_xticks(x, [s.replace(" (", "\n(") for s in d.index], fontsize=8.5)
    ax.set_ylabel("98%-band breaches, observed / nominal (x)")
    ax.legend(ncol=3, fontsize=8.5, loc="upper left")
    ax.grid(axis="x", visible=False)
    title(fig, "Crisis windows: who saw it coming?",
          "Tail-exceedance multiplier inside named stress windows, all assets pooled, 1-week horizon")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig6_stress_windows")


def fig_factorial(cc, st):
    """2x2 design: static vs dynamic volatility  x  thin vs fat tails."""
    q5 = cc[(cc.h == 5) & (cc.vol_q == "Q5 stressed")].set_index("model")
    q1 = cc[(cc.h == 5) & (cc.vol_q == "Q1 calm")].set_index("model")
    gfc = st[(st.h == 5) & st.window.str.startswith("GFC")].set_index("model")
    cov = st[(st.h == 5) & st.window.str.startswith("COVID")].set_index("model")
    grid = [[("GBM", "Static vol, thin tails"), ("GJR-GARCH-N", "Dynamic vol, thin tails")],
            [("Merton jumps", "Static vol, fat tails"), ("GJR-GARCH-t", "Dynamic vol, fat tails")]]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    norm = TwoSlopeNorm(0, -0.15, 0.15)
    for i, row in enumerate(grid):
        for j, (m, lab) in enumerate(row):
            err = q5.loc[m, "cov90"] - 0.90
            ax.add_patch(plt.Rectangle((j + 0.02, 1 - i + 0.02), 0.96, 0.96, color=DIVERGING(norm(-err)), lw=0))
            dark = abs(err) > 0.09
            c = "white" if dark else INK
            ax.text(j + 0.5, 1 - i + 0.86, m, ha="center", fontsize=12, fontweight="bold", color=c)
            ax.text(j + 0.5, 1 - i + 0.78, lab, ha="center", fontsize=9, color=c)
            ax.text(j + 0.5, 1 - i + 0.50, f"{q5.loc[m, 'cov90']:.1%}", ha="center", fontsize=24,
                    fontweight="bold", color=c)
            ax.text(j + 0.5, 1 - i + 0.40, "90%-interval coverage in stressed weeks", ha="center", fontsize=8.5, color=c)
            ax.text(j + 0.5, 1 - i + 0.22,
                    f"Calm weeks: {q1.loc[m, 'cov90']:.1%}\n98%-band breaches vs nominal: "
                    f"GFC {gfc.loc[m, 'exceed98_mult']:.1f}x, COVID {cov.loc[m, 'exceed98_mult']:.1f}x",
                    ha="center", va="center", fontsize=9, color=c, linespacing=1.5)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.axis("off")
    ax.text(0.5, 2.04, "STATIC volatility", ha="center", fontsize=10, color=INK2, fontweight="bold")
    ax.text(1.5, 2.04, "DYNAMIC volatility (GJR-GARCH)", ha="center", fontsize=10, color=INK2, fontweight="bold")
    ax.text(-0.04, 1.5, "THIN tails\n(Gaussian)", ha="right", va="center", fontsize=10, color=INK2, fontweight="bold")
    ax.text(-0.04, 0.5, "FAT tails\n(jumps / Student-t)", ha="right", va="center", fontsize=10, color=INK2,
            fontweight="bold")
    title(fig, "It is the volatility dynamics, not the fat tails",
          "Factorial comparison. Red = intervals too narrow in stress, blue = too wide; nominal 90%")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig7_factorial_dynamics_vs_tails")


def fig_dumbbell(ci):
    d = ci[(ci.level == 0.90) & ci.vol_q.isin(["Q1 calm", "Q5 stressed"])]
    order = (d[d.vol_q == "Q5 stressed"].set_index("model").coverage.sort_values()).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    calm_c, stress_c = "#2a78d6", "#eb6834"
    for i, m in enumerate(order):
        c1 = d[(d.model == m) & (d.vol_q == "Q1 calm")].iloc[0]
        c5 = d[(d.model == m) & (d.vol_q == "Q5 stressed")].iloc[0]
        ax.plot([c1.coverage, c5.coverage], [i, i], color=AXIS, lw=2.5, zorder=1, solid_capstyle="round")
        for c, col, mk in [(c1, calm_c, "o"), (c5, stress_c, "D")]:
            ax.hlines(i, c.ci_low, c.ci_high, color=col, lw=1.2, alpha=0.8, zorder=2)
            ax.scatter(c.coverage, i, s=70, color=col, marker=mk, edgecolor=SURFACE, lw=1.2, zorder=3)
        ax.text(1.015, i, f"{c5.coverage:.1%}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=9, color=INK, clip_on=False)
        ax.text(1.13, i, f"{c1.coverage:.1%}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=9, color=INK2, clip_on=False)
    ax.axvline(0.90, color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(0.901, len(order) - 0.45, "nominal 90%", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(order)), order)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Realised coverage of the 1-week 90% interval (95% block-bootstrap CI)")
    ax.scatter([], [], color=stress_c, marker="D", s=50, label="Most stressed 20% of weeks")
    ax.scatter([], [], color=calm_c, marker="o", s=50, label="Calmest 20% of weeks")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.72, 1.0)
    for x, lab in [(1.015, "stressed"), (1.13, "calm")]:
        ax.text(x, len(order) - 0.4, lab, transform=ax.get_yaxis_transform(), fontsize=8.5,
                color=INK2, fontweight="bold", clip_on=False)
    title(fig, "Calm vs stress: the gap is the problem",
          "Every forecaster, including three control arms. A calibrated simulator puts both dots on the 90% line")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig8_calm_vs_stress_all_models")


def main():
    df = load()
    ci = pd.read_csv(RES / "conditional_coverage_ci.csv")
    fig_conditional(pd.read_csv(RES / "conditional_coverage.csv"), ci)
    fig_dumbbell(ci)
    fig_pit(df)
    fig_heat(pd.read_csv(RES / "summary_by_asset_model.csv"))
    fig_bands(df, "SPY")
    fig_bands(df, "USO")
    fig_crps(pd.read_csv(RES / "crps_dm.csv"))
    fig_stress(pd.read_csv(RES / "stress_windows.csv"))
    fig_factorial(pd.read_csv(RES / "conditional_coverage.csv"), pd.read_csv(RES / "stress_windows.csv"))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
