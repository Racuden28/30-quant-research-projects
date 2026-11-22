"""Publication figures for project 18. Run after analysis.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import changes, panel  # noqa: E402
from models import pca  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens shared with project 01 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN, YELLOW, PINK, DKGREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
RED = "#c23b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
})

PANEL_LABEL = {"A": "Panel A · 1976–2026 · hedges 2Y, 5Y, 10Y",
               "B": "Panel B · 2006–2026 · hedges 2Y, 5Y, 10Y, 30Y"}
METHOD_LABEL = {"ns": "Nelson–Siegel factor-neutral", "pca": "PCA factor-neutral (textbook)",
                "pca_vol": "PCA factor-neutral, vol-scaled", "interp": "Key-rate bucketing (baseline)",
                "fm3": "PCA as a risk model, 3 factors", "ols": "PCA as a risk model, all factors (= OLS)",
                "clair": "Hindsight bound (clairvoyant OLS)"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def title(fig, t, sub, y=1.0):
    fig.text(0.01, y, t, fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(0.01, y - 0.012, sub, fontsize=10, color=INK2, ha="left", va="top")


def fmt_x(v):
    return f"{v:,.0f}×" if v >= 10 else f"{v:.2f}×"


# ---------------------------------------------------------------- 1. star chart
def fig_star(hr):
    order = ["ns", "pca", "pca_vol", "interp", "fm3", "ols", "clair"]
    XMAX = 3.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    for ax, (p, K) in zip(axes, [("A", 3), ("B", 4)]):
        d = hr[(hr.panel == p) & (hr.K == K) & (hr.family == "curve")].set_index("method")
        y = np.arange(len(order))[::-1]
        ax.axvspan(0, 1, color=GREEN, alpha=0.07, lw=0)
        ax.axvspan(1, XMAX, color=RED, alpha=0.05, lw=0)
        ax.axvline(1, color=INK2, lw=1, ls="--")
        for yi, m in zip(y, order):
            r = d.loc[m]
            col = RED if r.ratio_vs_base > 1.0001 else (MUTED if m in ("interp", "clair") else BLUE)
            if r.ratio_vs_base > XMAX:  # off the chart: arrow and a label
                ax.annotate("", xy=(XMAX, yi), xytext=(XMAX - 0.45, yi),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
                ax.text(XMAX - 0.5, yi + 0.02, f"{r.ratio_vs_base:,.0f}× (median month {r.median_month_ratio:,.0f}×,\n"
                        f"better than bucketing in {r.share_months_better:.0%} of months)",
                        ha="right", va="center", fontsize=8.5, color=col)
                continue
            if m != "interp":
                ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=col, lw=3, alpha=0.45, solid_capstyle="round")
            ax.plot(r.ratio_vs_base, yi, "o", ms=9, color=col, mec="white", mew=1.2, zorder=3)
            ax.text(r.ratio_vs_base, yi + 0.3, f"{r.ratio_vs_base:.2f}×", ha="center", fontsize=9, color=col)
        ax.set_xlim(0, XMAX)
        ax.set_yticks(y, [METHOD_LABEL[m] for m in order])
        ax.set_title(PANEL_LABEL[p], loc="left", fontsize=10.5)
        ax.set_xlabel("Out-of-sample residual variance of DV01-neutral curve books,\n"
                      "relative to key-rate bucketing (< 1 = less risk left than bucketing)")
        ax.text(0.5, -0.62, "less risk", color=DKGREEN, fontsize=8.5, ha="center")
        ax.text(2.0, -0.62, "more risk", color=RED, fontsize=8.5, ha="center")
        ax.set_ylim(-0.75, len(order) - 0.4)
        ax.grid(axis="y", visible=False)
    title(fig, "Three PCA factors explain 98% of daily Treasury moves. Neutralising them is the worst hedge here.",
          "The same trailing two-year PCA used two ways. As a list of risks to neutralise, it blows up. "
          "As a covariance model, it beats the desk baseline.\nDots: pooled ratio with 95% block-bootstrap CI. "
          "500 random DV01-neutral books, monthly re-estimation, no look-ahead.", y=1.0)
    fig.subplots_adjust(top=0.84, wspace=0.06)
    save(fig, "fig1_two_uses_of_pca")


# ---------------------------------------------------------------- 2. where the 99% comes from
def fig_where_99(vt):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    data = ["levels", "monthly changes (21d)", "weekly changes (5d)", "daily changes"]
    lab = ["Yield\nlevels", "Monthly\nchanges", "Weekly\nchanges", "Daily\nchanges"]
    x = np.arange(len(data))
    for off, p, col, name in [(-0.2, "A", BLUE, "Treasury CMT curve (FRED)"),
                              (0.2, "G", ORANGE, "Fed GSW fitted curve (Nelson–Siegel–Svensson)")]:
        d = vt[vt.panel == p].set_index("data").loc[data]
        a1.bar(x + off, d.curve_left_k3 * 100, 0.38, color=col, label=name)
        for xi, (v, s) in enumerate(zip(d.curve_left_k3, d.share_k3)):
            a1.text(xi + off, v * 100 + 0.4, f"{v:.1%}", ha="center", fontsize=8.5, color=col)
            a1.text(xi + off, -1.6, f"{s:.2%}", ha="center", fontsize=7.5, color=col)
    a1.text(-0.62, -1.6, "all-variance\nshare of 3 PCs:", fontsize=7.5, color=INK2, ha="right", va="center")
    a1.set_xticks(x, lab)
    a1.set_ylim(-3, 19)
    a1.set_ylabel("% of non-parallel variance\nleft outside 3 PCs (in sample)")
    a1.set_title("A. The '99%' is a statement about levels and fitted curves", loc="left")
    a1.legend(loc="upper left", fontsize=8.5)
    a1.grid(axis="x", visible=False)
    for p, col, name in [("A", BLUE, "CMT"), ("G", ORANGE, "GSW fitted")]:
        e = np.array(vt[(vt.panel == p) & (vt.data == "daily changes")].eigs.iloc[0].split(), float)
        a2.plot(np.arange(1, len(e) + 1), e, "o-", color=col, lw=2, label=f"{name}, daily changes 1Y–10Y")
    a2.set_yscale("log")
    a2.set_xlabel("Principal component")
    a2.set_ylabel("Share of total variance (log)")
    a2.set_title("B. The fitted curve has almost nothing beyond PC3", loc="left")
    a2.legend(fontsize=8.5)
    title(fig, "Why 'three factors explain 99%' oversells the model for hedgers",
          "1976–2026, maturities 1Y–10Y. What a curve trade cares about is the non-parallel part of the move, "
          "and the fitted curve shows barely a quarter of it.", y=1.0)
    fig.subplots_adjust(top=0.84, wspace=0.25)
    save(fig, "fig2_where_the_99_comes_from")


# ---------------------------------------------------------------- 3. how many factors
def fig_factor_count(fc):
    fig, ax = plt.subplots(figsize=(10, 5))
    spec = [("A", 3, BLUE, "Panel A, CMT 1976–2026, 3 hedges"), ("B", 4, ORANGE, "Panel B, CMT 2006–2026, 4 hedges"),
            ("G", 3, MUTED, "GSW fitted curve, 3 hedges"), ("E", 3, GREEN, "Treasury ETFs, 3 hedges")]
    for p, K, col, name in spec:
        d = fc[(fc.panel == p) & (fc.K == K) & (fc.family == "curve") & (fc.model == "fm")].sort_values("j")
        ax.plot(d.j, d.ratio_vs_ols, "o-", color=col, lw=2, label=name)
    ax.axvline(3, color=INK2, ls=":", lw=1)
    ax.text(3.08, 2.6, "the textbook\n3 factors", fontsize=8.5, color=INK2, va="top")
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_yscale("log")
    ax.set_yticks([1, 1.1, 1.25, 1.5, 2, 2.5], ["1.00", "1.10", "1.25", "1.50", "2.00", "2.50"])
    ax.set_ylim(0.97, 2.8)
    ax.set_xlabel("Number of PCA factors kept in the covariance model (the rest is treated as diagonal noise)")
    ax.set_ylabel("Residual variance relative to keeping\nall factors (OLS), log scale")
    ax.legend(fontsize=9, loc="upper right")
    title(fig, "For hedging, the best number of yield-curve factors is all of them",
          "Minimum-variance hedges of DV01-neutral books under a j-factor PCA covariance. "
          "The last point of each line keeps every factor (= OLS).\nStopping at 3 factors never beats keeping them all.", y=0.98)
    fig.subplots_adjust(top=0.85)
    save(fig, "fig3_how_many_factors")


# ---------------------------------------------------------------- 4. mechanism
def fig_mechanism(rot):
    d = rot[(rot.panel == "A")].dropna(subset=["angle_max_k3"])
    era = np.where(d.level_2y < 0.5, "2Y yield below 0.5% (zero lower bound)", "2Y yield above 0.5%")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for e, col in [("2Y yield above 0.5%", BLUE), ("2Y yield below 0.5% (zero lower bound)", ORANGE)]:
        m = era == e
        a1.scatter(d.pca_gross_k3[m], d.pca_vs_interp_k3[m], s=14, color=col, alpha=0.65, label=e, lw=0)
        a2.scatter(d.angle_max_k3[m], d.pca_vs_interp_k3[m], s=14, color=col, alpha=0.65, lw=0)
    for ax in (a1, a2):
        ax.set_yscale("log")
        ax.axhline(1, color=INK2, lw=0.9, ls="--")
    a1.set_xscale("log")
    a1.axvline(2, color=INK2, lw=0.8, ls=":")
    a1.text(2.1, 1e4, "gross hedge = 2×\nthe book's DV01", fontsize=8, color=INK2)
    c1 = np.corrcoef(np.log(d.pca_gross_k3), np.log(d.pca_vs_interp_k3))[0, 1]
    c2 = np.corrcoef(d.angle_max_k3, np.log(d.pca_vs_interp_k3))[0, 1]
    a1.set_title(f"A. Size of the hedge it asks for (corr of logs {c1:.2f})", loc="left")
    a2.set_title(f"B. How much the factors rotated over the next year (corr {c2:.2f})", loc="left")
    a1.set_xlabel("Gross PCA-neutral hedge, DV01 per unit of book DV01 (log)")
    a2.set_xlabel("Largest principal angle between this window's 3-PC space and next year's (degrees)")
    a1.set_ylabel("That month's residual variance,\nPCA-neutral / bucketing (log)")
    a1.legend(fontsize=8.5, loc="lower right")
    title(fig, "The textbook hedge fails because of geometry, not because the factors move",
          "Panel A, 574 monthly rebalances, 1978–2026. When the 2Y/5Y/10Y loadings are nearly collinear the "
          "neutral hedge needs huge offsetting positions.", y=1.0)
    fig.subplots_adjust(top=0.84, wspace=0.06)
    save(fig, "fig4_geometry_not_rotation")


# ---------------------------------------------------------------- 5. loadings through time
def fig_loadings():
    lv = panel("B")
    dy = changes(lv)
    ends = [("2007-06-29", "2005–07, pre-GFC", BLUE), ("2013-06-28", "2011–13, zero lower bound", ORANGE),
            ("2019-06-28", "2017–19, normal", GREEN), ("2023-06-30", "2021–23, hiking", RED)]
    mats = [1, 2, 3, 5, 7, 10, 20, 30]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharex=True)
    for end, lab, col in ends:
        w = dy.loc[:end].iloc[-504:].to_numpy()
        S = np.cov(w, rowvar=False)
        L, vals = pca(S, 3)
        sd = np.sqrt(vals[:3])
        for j, ax in enumerate(axes):
            ax.plot(mats, L[:, j] * sd[j], "o-", color=col, lw=2, ms=4, label=lab)
    for j, (ax, t) in enumerate(zip(axes, ["PC1 'level'", "PC2 'slope'", "PC3 'curvature'"])):
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(mats, [f"{m}Y" for m in mats])
        ax.minorticks_off()
        ax.set_title(t, loc="left")
        ax.set_xlabel("Maturity")
    axes[0].set_ylabel("Loading × factor volatility (bp/day)")
    axes[0].legend(fontsize=8.5)
    title(fig, "The factors keep their names but not their shapes",
          "Panel B, PCA of daily changes on trailing two-year windows. At the zero lower bound the front end "
          "stops moving and 'level' becomes a long-end factor.", y=1.0)
    fig.subplots_adjust(top=0.82, wspace=0.18)
    save(fig, "fig5_loadings_through_time")


# ---------------------------------------------------------------- 6. stress windows
def fig_stress(st):
    show = [("interp", "Bucketing", MUTED), ("fm3", "Risk model, 3 factors", BLUE),
            ("fmv3", "Risk model, 3 factors, vol-scaled", YELLOW), ("ols", "Risk model, all factors (OLS)", DKGREEN),
            ("clair", "Hindsight bound", INK2)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharex=True)
    for ax, (p, K) in zip(axes, [("A", 3), ("B", 4)]):
        d = st[(st.panel == p) & (st.K == K) & (st.family == "curve")].reset_index(drop=True)
        y = np.arange(len(d))[::-1]
        for off, (m, lab, col) in zip(np.linspace(-0.24, 0.24, len(show)), show):
            ax.scatter(d[f"he_{m}"].clip(lower=-0.45), y + off, s=26, color=col, label=lab, zorder=3)
        ax.set_yticks(y, [f"{w} ({n}d)" for w, n in zip(d.window, d.days)])
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlim(-0.5, 1)
        ax.set_title(PANEL_LABEL[p], loc="left", fontsize=10.5)
        ax.set_xlabel("Hedge effectiveness: 1 − residual / unhedged variance\n(curve books; points below −0.45 are clipped)")
        ax.grid(axis="y", visible=False)
    axes[1].legend(fontsize=8.5, loc="lower left")
    title(fig, "Estimated hedges win in most stress windows, but not in a front-end shock",
          "SVB (March 2023): the 2Y yield fell 102 bp in three trading days (8 to 13 March), and every estimated hedge did worse than bucketing.",
          y=1.0)
    fig.subplots_adjust(top=0.86, wspace=0.62)
    save(fig, "fig6_stress_windows")


# ---------------------------------------------------------------- 7. ETFs
def fig_etf(hr):
    d = hr[(hr.panel == "E") & (hr.family == "curve")]
    meths = [("pca", "PCA factor-neutral", RED), ("pca_vol", "PCA neutral, vol-scaled", PINK),
             ("fm3", "Risk model, 3 factors", BLUE), ("fmv3", "Risk model, 3 factors, vol-scaled", YELLOW)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, K in enumerate([1, 2, 3]):
        for j, (m, lab, col) in enumerate(meths):
            r = d[(d.K == K) & (d.method == m)].iloc[0]
            x = i + (j - 1.5) * 0.19
            v = min(r.ratio_vs_base, 1.6)
            ax.bar(x, v - 1, 0.17, bottom=1, color=col, label=lab if i == 0 else None)
            ax.plot([x, x], [max(r.ci_lo, 0.7), min(r.ci_hi, 1.6)], color=INK2, lw=1)
            if r.ratio_vs_base > 1.6:
                ax.text(x, 1.585, f"{r.ratio_vs_base:,.0f}×", ha="center", va="top", fontsize=7.5, color="white",
                        rotation=90, fontweight="bold")
    ax.axhline(1, color=INK2, lw=1)
    ax.set_xticks([0, 1, 2], ["1 hedge (IEF)", "2 hedges (SHY, TLT)", "3 hedges (SHY, IEF, TLT)"])
    ax.set_ylim(0.7, 1.6)
    ax.set_ylabel("Residual variance relative to\nOLS on the same ETFs")
    ax.legend(fontsize=8.5, ncol=2, loc="upper left")
    ax.grid(axis="x", visible=False)
    title(fig, "Same result in traded total returns",
          "Hedging IEI (3–7Y) and TLH (10–20Y) books with Treasury ETFs, 2009–2026. Bars above 1.6 are cut and "
          "labelled.", y=0.98)
    fig.subplots_adjust(top=0.84)
    save(fig, "fig7_treasury_etfs")


if __name__ == "__main__":
    hr = pd.read_csv(RES / "horse_race.csv")
    fig_star(hr)
    fig_where_99(pd.read_csv(RES / "variance_explained.csv"))
    fig_factor_count(pd.read_csv(RES / "factor_count.csv"))
    fig_mechanism(pd.read_csv(RES / "rotation.csv"))
    fig_loadings()
    fig_stress(pd.read_csv(RES / "stress_windows.csv"))
    fig_etf(hr)
    print("figures done")
