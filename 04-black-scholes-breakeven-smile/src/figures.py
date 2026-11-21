"""Publication figures for project 04. Run after accuracy.py, analysis.py, validation.py, ablation.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens shared with project 01 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7b5bd6"]
ASSETS = ["SPY", "EEM", "HYG", "USO", "TLT", "GLD", "URA"]
COLORS = dict(zip(ASSETS, PALETTE))
MARKERS = dict(zip(ASSETS, ["o", "s", "D", "^", "v", "P", "X"]))
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f08a89", "#a51c1c"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#a51c1c", "#f08a89", "#f0efec", "#6da7ec", "#184f95"])
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
XLAB = "Strike, in trailing standard deviations from spot  (x = ln(K/S) / σ_ref √T)"


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def title(fig, t, sub, y=1.0):
    fig.text(0.01, y, t, fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(0.01, y - 0.035, sub, fontsize=10, color=INK2, ha="left", va="bottom")


def fig_accuracy():
    g = pd.read_csv(RES / "accuracy_grid.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    ks, vs = np.sort(g.log_moneyness.unique()), np.sort(g.total_vol.unique())
    for ax, col, lab in [(axes[0], "rel_err_naive", "Textbook  F·N(d1) − K·N(d2)"),
                         (axes[1], "rel_err_stable", "Stable  (OTM side + erfcx)")]:
        e = g.pivot(index="total_vol", columns="log_moneyness", values=col).reindex(index=vs, columns=ks)
        rep = g.pivot(index="total_vol", columns="log_moneyness", values="representable").reindex(index=vs, columns=ks)
        dig = -np.log10(np.clip(e.fillna(1.0).values, 1e-16, 1.0))
        dig = np.where(rep.values.astype(bool), dig, np.nan)
        im = ax.pcolormesh(ks, vs, dig, cmap=SEQ, vmin=0, vmax=16, shading="nearest")
        ax.set_yscale("log")
        ax.set_title(lab, loc="left")
        ax.set_xlabel("Log-moneyness  ln(K/F)")
        ax.grid(False)
    axes[0].set_ylabel("Total volatility  σ√T")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label("Correct significant digits (vs 60-digit reference)")
    s = pd.read_csv(RES / "accuracy_summary.csv")
    axes[0].text(0.02, 0.03, f"{s.share_cells_no_correct_digit[0]:.0%} of cells: no correct digit",
                 transform=axes[0].transAxes, fontsize=9, color=INK, va="bottom")
    axes[1].text(0.02, 0.03, f"worst cell: {s.worst_correct_digits[1]:.1f} digits",
                 transform=axes[1].transAxes, fontsize=9, color=INK, va="bottom")
    title(fig, "The textbook formula loses every digit in the put wing", "Accuracy of the out-of-the-money option value, "
          "grid of 81 strikes × 61 total volatilities; blank = value below 1e-300 (not representable)", y=1.02)
    save(fig, "fig1_pricer_accuracy")


def fig_smiles():
    c = pd.read_csv(RES / "breakeven_smiles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    groups = [("Risk assets: history charged a put skew", ["SPY", "EEM", "HYG", "USO"]),
              ("Havens and uranium: a symmetric smile", ["TLT", "GLD", "URA"])]
    for ax, (lab, assets) in zip(axes, groups):
        for a in assets:
            d = c[c.asset == a]
            ax.fill_between(d.x, d.ci_lo, d.ci_hi, color=COLORS[a], alpha=0.13, lw=0)
            ax.plot(d.x, d.m_star, color=COLORS[a], lw=2.2, label=ASSET_LABEL[a])
            ax.plot(d.x[::4], d.m_star[::4], ls="none", marker=MARKERS[a], color=COLORS[a], ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.1)
        ax.axhline(1, color=INK, lw=1, ls=(0, (4, 3)))
        ax.text(-2.5, 0.985, "m* = 1: Black–Scholes at trailing vol breaks even", ha="left", va="top", fontsize=8.5, color=INK2)
        ax.set_title(lab, loc="left")
        ax.set_xlabel(XLAB)
        ax.text(-2.5, 0.72, "← low strikes: OTM puts          high strikes: OTM calls →", fontsize=8, color=MUTED)
        ax.legend(loc="upper center", fontsize=8.5)
        ax.set_xlim(-2.6, 2.6)
    axes[0].set_ylabel("Break-even vol ÷ trailing 63-day vol  (m*)")
    title(fig, "The smile history would have charged",
          "Volatility at which a daily delta-hedged 1-month option seller broke even, pooled over all start dates "
          "2007–2026; bands: 95% block bootstrap", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig2_breakeven_smiles")


def fig_yearly():
    y = pd.read_csv(RES / "yearly_smile_stats.csv")
    p = y.pivot(index="asset", columns="year", values="skew").reindex(ASSETS)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(p.values, cmap=DIVERGING, norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
    ax.set_yticks(range(len(p)), [ASSET_LABEL[a] for a in p.index])
    ax.set_xticks(range(p.shape[1]), p.columns, rotation=0, fontsize=8.5)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}".replace("+", "") if v < 0 else f"{v:.2f}", ha="center", va="center",
                        fontsize=7.2, color="white" if abs(v) > 0.55 else INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
    cb.set_label("Skew  m*(−1.5σ) − m*(+1.5σ)")
    title(fig, "US equity and high-yield skew never switched off",
          "Break-even skew by calendar year of option start; red = puts were under-priced by flat Black–Scholes", y=1.02)
    save(fig, "fig3_yearly_skew")


def fig_stress():
    s = pd.read_csv(RES / "stress_smiles.csv")
    wins = ["All other start dates", "GFC (Sep-Dec 2008)", "COVID (Feb 15-Apr 30 2020)", "2022 rate shock"]
    wcol = dict(zip(wins, [MUTED, "#a51c1c", "#eb6834", "#2a78d6"]))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), sharey=True)
    for ax, a in zip(axes, ["SPY", "HYG", "TLT", "GLD"]):
        for w in wins:
            d = s[(s.asset == a) & (s.window == w)]
            if len(d):
                ax.plot(d.x, d.m_star, color=wcol[w], lw=2.4 if w != wins[0] else 1.8,
                        ls="-" if w != wins[0] else (0, (4, 2)), label=w)
        ax.axhline(1, color=INK, lw=0.8, ls=(0, (2, 3)))
        ax.set_title(ASSET_LABEL[a], loc="left", fontsize=10)
        ax.set_xlabel("Strike (trailing s.d.)")
    axes[0].set_ylabel("Break-even vol ÷ trailing vol")
    axes[0].legend(loc="upper right", fontsize=7.8)
    title(fig, "In a crash, every strike breaks even above trailing vol",
          "Pooled break-even smile for options started inside each stress window", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig4_stress_smiles")


def fig_vix():
    v = pd.read_csv(RES / "spy_vix_vs_breakeven.csv", parse_dates=["date"]).set_index("date")
    r = v[["vix", "atm_breakeven_vol"]].rolling(63, min_periods=40).mean() * 100
    gap = (v.vix - v.atm_breakeven_vol).rolling(63, min_periods=40).mean() * 100
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.2]))
    axes[0].plot(r.index, r.vix, color="#2a78d6", lw=1.6, label="VIX at option start")
    axes[0].plot(r.index, r.atm_breakeven_vol, color="#eb6834", lw=1.6, label="ATM break-even vol, same option")
    axes[0].set_ylabel("Volatility, %  (63-day mean)")
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[1].fill_between(gap.index, 0, gap, where=gap >= 0, color="#2a78d6", alpha=0.55, lw=0)
    axes[1].fill_between(gap.index, 0, gap, where=gap < 0, color="#a51c1c", alpha=0.75, lw=0)
    axes[1].axhline(0, color=INK, lw=0.8)
    axes[1].set_ylabel("VIX − break-even\n(vol points)")
    s = pd.read_csv(RES / "spy_vix_summary.csv")
    f = s.iloc[0]
    axes[1].text(0.01, 0.92, f"Full sample: VIX above break-even on {f.share_vix_above:.0%} of start dates, "
                 f"mean gap {f.mean_gap_vol_pts:.1f} vol pts", transform=axes[1].transAxes, fontsize=8.5, va="top")
    title(fig, "The volatility premium, measured with Black–Scholes' own accounting",
          "SPY 1-month ATM options: VIX at inception vs the vol a daily delta hedger actually needed", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig5_vix_vs_breakeven")


def fig_oos():
    o = pd.read_csv(RES / "oos_bias_by_strike.csv")
    t = pd.read_csv(RES / "oos_tests.csv").set_index("asset")
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.8), sharex=True)
    mcol = {"Textbook BS (flat trailing vol)": "#a51c1c", "Flat, level-corrected": MUTED,
            "Break-even smile (expanding)": "#1baf7a"}
    for ax, a in zip(axes.flat, ASSETS):
        for m, col in mcol.items():
            d = o[(o.asset == a) & (o.model == m)]
            ax.plot(d.x, d.mean_pnl, color=col, lw=2.2 if "smile" in m else 1.6,
                    ls="-" if "level" not in m else (0, (4, 2)), label=m)
            if "smile" in m:
                ax.fill_between(d.x, d.mean_pnl - 1.96 * d.hac_se, d.mean_pnl + 1.96 * d.hac_se, color=col, alpha=0.15, lw=0)
        ax.axhline(0, color=INK, lw=0.8)
        r = t.loc[a]
        ax.set_title(f"{a}   bias −{(1 - r.mab_smile / r.mab_textbook):.0%}", loc="left", fontsize=10)
    axes.flat[-1].axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h, lab, loc="center", fontsize=9)
    for ax in list(axes[1]) + [axes[0, 3]]:
        ax.set_xlabel("Strike (trailing s.d.)")
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Seller's mean hedged P&L\n(vega × trailing vol)")
    title(fig, "Out of sample, the historical smile removes most of Black–Scholes' wing bias",
          "Options started 2012–2026, smile re-estimated monthly from options that had already expired; "
          "0 = fairly priced; band: ±1.96 HAC s.e.", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig6_oos_bias")


def fig_validation():
    v = pd.read_csv(RES / "validation.csv", index_col="x")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col, c, mk in zip(v.columns, ["#2a78d6", "#eb6834", "#1baf7a"], ["o", "s", "D"]):
        ax.plot(v.index, v[col], color=c, lw=2, label=col)
        ax.plot(v.index[::4], v[col].values[::4], ls="none", marker=mk, color=c, ms=6, markeredgecolor=SURFACE)
    ax.axhline(1, color=INK, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("Strike (trailing s.d.)")
    ax.set_ylabel("Estimated break-even multiplier m*")
    ax.legend(fontsize=8.5, loc="upper center")
    title(fig, "The estimator recovers known answers", "Simulated 80-year daily paths, same pipeline as the data", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig7_validation")


def main():
    fig_accuracy()
    fig_smiles()
    fig_yearly()
    fig_stress()
    fig_vix()
    fig_oos()
    if (RES / "validation.csv").exists():
        fig_validation()


if __name__ == "__main__":
    main()
