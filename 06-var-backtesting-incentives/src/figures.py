"""Publication figures for project 06. Run after theory.py and evaluate.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from models import MODEL_ORDER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens shared with project 01 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
COLORS = dict(zip(MODEL_ORDER, CAT))
MARKERS = dict(zip(MODEL_ORDER, ["o", "s", "D", "^", "v", "P"]))
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#f6b48f", "#e2603a", "#a51c1c"])
RED, AMBER, GREEN = "#c62828", "#eda100", "#1baf7a"
SERIES = ["SPY", "EEM", "TLT", "HYG", "GLD", "USO", "URA", "BOOK"]
LABEL = {"SPY": "US equities", "EEM": "EM equities", "TLT": "Long Treasuries", "HYG": "High yield",
         "GLD": "Gold", "USO": "Crude oil", "URA": "Uranium", "BOOK": "Multi-asset book"}

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
    fig.text(0.01, y - 0.035, sub, fontsize=10, color=INK2, ha="left", va="bottom")


# ------------------------------------------------------------------ fig 1: the headline

def fig_break_even():
    be = pd.read_csv(RES / "theory_break_even.csv")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    n = np.arange(3, 16)
    basel = be[np.isinf(be.nu)].plus_basel.values
    ax.step(n, basel, where="mid", color=INK, lw=2.6, label="Basel plus factor (1996 table)", zorder=5)
    styles = [(np.inf, CAT[0], "Gaussian"), (10.0, CAT[2], "Student t, ν = 10"),
              (5.0, CAT[3], "Student t, ν = 5"), (3.0, CAT[1], "Student t, ν = 3")]
    for nu, c, lab in styles:
        g = be[be.nu == nu] if np.isfinite(nu) else be[np.isinf(be.nu)]
        ax.plot(g.n_exc, g.plus_required, color=c, lw=2, marker="o", ms=5.5, markeredgecolor=SURFACE,
                markeredgewidth=1, label=f"break-even plus, {lab}")
    ax.axvspan(2.5, 4.5, color=GREEN, alpha=0.07, lw=0)
    ax.axvspan(4.5, 9.5, color=AMBER, alpha=0.08, lw=0)
    ax.axvspan(9.5, 15.5, color=RED, alpha=0.06, lw=0)
    for x, t in ((3.5, "green"), (7, "yellow"), (12.5, "red")):
        ax.text(x, 0.06, t, ha="center", color=INK2, fontsize=9, fontweight="bold")
    ax.set_xlim(2.5, 15.5); ax.set_ylim(0, 2.8)
    ax.set_xlabel("expected exceptions per 250 days produced by shading the reported VaR")
    ax.set_ylabel("plus factor added to the multiplier of 3")
    ax.legend(loc="upper left", fontsize=9)
    title(fig, "The Basel traffic light is the Gaussian break-even point, and nothing more",
          "Plus factor that exactly offsets the capital a bank saves by reporting λ·VaR, vs the plus factor the rule charges",
          y=0.99)
    save(fig, "fig1_break_even")


# ------------------------------------------------------------------ fig 2: theory curves + safe shading

def fig_theory():
    cur = pd.read_csv(RES / "theory_curves.csv")
    opt = pd.read_csv(RES / "theory_optimum.csv")
    mc = pd.read_csv(RES / "theory_mc_garch.csv")
    cur = cur[cur.regime == "Basel 2.5"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    for nu, c, lab in ((np.inf, CAT[0], "Gaussian"), (5.0, CAT[3], "t, ν = 5"), (3.0, CAT[1], "t, ν = 3")):
        g = cur[np.isinf(cur.nu)] if np.isinf(nu) else cur[cur.nu == nu]
        g = g[g.lam >= 0.6]
        safe = g.p_red <= 0.05
        ax.plot(g.lam[safe], g.rel_charge[safe], color=c, lw=2.2, label=lab)
        ax.plot(g.lam[~safe], g.rel_charge[~safe], color=c, lw=1.4, ls=(0, (2, 2)))
        j = g[safe].rel_charge.idxmin()
        ax.plot(g.lam[j], g.rel_charge[j], "o", color=c, ms=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axhline(1, color=INK, lw=0.8); ax.axvline(1, color=INK, lw=0.8)
    ax.set_xlabel("reporting scale λ (reported VaR / true VaR)")
    ax.set_ylabel("expected capital charge / honest charge")
    ax.set_title("Expected charge, i.i.d. losses", loc="left")
    ax.text(0.62, 1.07, "dashed: >5% chance of a red-zone year", color=INK2, fontsize=8.5)
    ax.legend(loc="lower right", fontsize=9)
    ax = axes[1]
    o = opt[(opt.regime == "Basel 2.5")].copy()
    o["nu_plot"] = np.where(np.isinf(o.nu), 60, o.nu)
    ax.plot(o.nu_plot, o.saving_red5 * 100, color=CAT[0], lw=2.2, marker="o", ms=6, markeredgecolor=SURFACE,
            label="closed form, i.i.d. Student t")
    m = mc.copy()
    ax.errorbar(m.nu, m.saving * 100, yerr=[(m.saving - m.saving_lo) * 100, (m.saving_hi - m.saving) * 100],
                fmt="s", color=CAT[1], ms=7, capsize=3, label="Monte Carlo, GARCH(1,1)-t, quarterly reviews")
    for _, r in o.iterrows():
        if r.nu in (3.0, 5.0, 10.0) or np.isinf(r.nu):
            ax.annotate(f"λ = {r.lam_star_red5:.2f}", (r.nu_plot, r.saving_red5 * 100), textcoords="offset points",
                        xytext=(6, 6), fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks([3, 5, 10, 30, 60], ["3", "5", "10", "30", "Gauss"])
    ax.set_xlabel("tail index ν (lower = fatter tails)")
    ax.set_ylabel("capital saved by optimal 'safe' shading, %")
    ax.set_title("How much shading pays without a red year (≤5%)", loc="left")
    ax.legend(fontsize=8.5, loc="upper right")
    title(fig, "Fatter tails make understatement both cheaper and harder to detect",
          "A shaded VaR breaches less often when the tail is heavy, so the plus factor bites later", y=1.03)
    save(fig, "fig2_theory_safe_shading")


# ------------------------------------------------------------------ fig 3: empirical saving heatmap

def fig_saving_heatmap(sh):
    s = sh[sh.regime == "Basel 2.5"].pivot(index="model", columns="series", values="saving").loc[MODEL_ORDER, SERIES]
    lam = sh[sh.regime == "Basel 2.5"].pivot(index="model", columns="series", values="lam_capital").loc[MODEL_ORDER, SERIES]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    im = ax.imshow(s.values * 100, cmap=SEQ, vmin=0, vmax=max(20, np.nanmax(s.values) * 100), aspect="auto")
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            v = s.values[i, j] * 100
            ax.text(j, i - 0.12, f"{v:.1f}%", ha="center", va="center", fontsize=9.5,
                    color="white" if v > 13 else INK, fontweight="bold")
            ax.text(j, i + 0.24, f"λ={lam.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 13 else INK2)
    ax.set_xticks(range(len(SERIES)), [LABEL[x] for x in SERIES], rotation=0, fontsize=8.5)
    ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("capital saved, %")
    cb.outline.set_visible(False)
    title(fig, "In 45 of 48 books a cheaper reporting scale exists that the traffic light does not punish",
          "Basel 2.5 charge saved by the best ex-post reporting scale λ with no more red-zone quarters than honest reporting",
          y=1.0)
    save(fig, "fig3_saving_heatmap")


# ------------------------------------------------------------------ fig 4: score vs capital

def fig_score_vs_capital(cur, sh, series="BOOK", model="GJR-skew-t"):
    g = cur[(cur.series == series) & (cur.model == model) & (cur.regime == "Basel 2.5")].sort_values("lam")
    row = sh[(sh.series == series) & (sh.model == model) & (sh.regime == "Basel 2.5")].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(wspace=0.42))
    ax = axes[0]
    ax.plot(g.lam, g.rel_charge, color=CAT[0], lw=2.2, label="Basel 2.5 charge / honest")
    ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax2.fill_between(g.lam, 0, g.red * 100, color=RED, alpha=0.15, lw=0, label="red-zone quarters")
    ax2.plot(g.lam, g.red * 100, color=RED, lw=1)
    ax2.set_ylabel("quarters in red zone, %", color=RED); ax2.set_ylim(0, 100)
    ax.axvline(row.lam_capital, color=CAT[0], ls=(0, (3, 2)), lw=1.2)
    ax.text(row.lam_capital, 1.32, f" capital-optimal\n λ = {row.lam_capital:.2f}", color=CAT[0], fontsize=8.5, va="top")
    ax.set_xlabel("reporting scale λ"); ax.set_ylabel("charge relative to λ = 1")
    ax.set_title("What the capital rule rewards", loc="left")
    ax.set_ylim(0.5, 1.4); ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax = axes[1]
    f = g.fz0 - g.fz0.min()
    t = (g.tick - g.tick.min()) / (g.tick.max() - g.tick.min()) * f.max()
    ax.plot(g.lam, f, color=CAT[1], lw=2.2, label="FZ0 joint VaR/ES score (97.5%)")
    ax.plot(g.lam, t, color=CAT[2], lw=1.8, ls=(0, (4, 2)), label="quantile score, VaR 99% (rescaled)")
    ax.axvline(row.lam_honest_fz0, color=CAT[1], ls=(0, (3, 2)), lw=1.2)
    ax.text(row.lam_honest_fz0, f.max() * 0.55, f" score-optimal\n λ = {row.lam_honest_fz0:.2f}", color=CAT[1], fontsize=8.5)
    ax.set_xlabel("reporting scale λ"); ax.set_ylabel("mean loss above its minimum")
    ax.set_title("What a strictly consistent score rewards", loc="left")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    title(fig, f"Same forecasts, opposite incentives ({LABEL[series]}, {model})",
          "The charge keeps falling as λ shrinks until red-zone quarters appear; the score is U-shaped around honest reporting",
          y=1.03)
    save(fig, "fig4_score_vs_capital")


# ------------------------------------------------------------------ fig 5: gap dumbbell

def fig_gap(sh):
    s = sh[sh.regime == "Basel 2.5"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    for ax, model in zip(axes, ["GJR-skew-t", "HS-250"]):
        g = s[s.model == model].set_index("series").loc[SERIES[::-1]]
        y = np.arange(len(g))
        ax.hlines(y, g.lam_capital, g.lam_honest_tick, color=AXIS, lw=3, zorder=1)
        ax.errorbar(g.lam_capital, y, xerr=[g.lam_capital - g.lam_capital_lo, g.lam_capital_hi - g.lam_capital],
                    fmt="o", color=RED, ms=8, capsize=3, label="capital-optimal λ (95% block-bootstrap CI)", zorder=3)
        ax.plot(g.lam_honest_tick, y, "D", color=CAT[0], ms=7, label="honest λ (minimises quantile score)", zorder=3)
        ax.axvline(1, color=INK, lw=0.8)
        ax.set_yticks(y, [LABEL[x] for x in g.index])
        ax.set_title(model, loc="left"); ax.set_xlabel("reporting scale λ")
    axes[0].legend(loc="lower left", fontsize=8.5, bbox_to_anchor=(0, -0.32))
    title(fig, "The gaming gap: what the score calls honest vs what the capital rule calls optimal",
          "Each line is the understatement a bank could hide from the traffic light, 2008 to 2026", y=1.0)
    save(fig, "fig5_gaming_gap")


# ------------------------------------------------------------------ fig 6: ex-ante shading

def fig_adaptive(ad, bt):
    a = ad[(ad.regime == "Basel 2.5")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    for b, c in ((4, CAT[2]), (7, CAT[3]), (9, CAT[1])):
        g = a[a.budget == b]
        ax.scatter(g.red * 100, (1 - g.rel_charge) * 100, color=c, s=34, edgecolor=SURFACE, lw=0.8,
                   label=f"budget {b} exceptions")
    hon = ad[(ad.regime == "Basel 2.5") & (ad.budget == 4)][["series", "model"]].merge(
        HONEST_RED, on=["series", "model"])
    ax.scatter(hon.red_honest * 100, np.zeros(len(hon)), color=INK, marker="|", s=80, label="honest reporting")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("quarters in the red zone, %"); ax.set_ylabel("capital saved vs honest, %")
    ax.set_title("Ex-ante 'exception budget' rule, 48 series×model", loc="left")
    ax.legend(fontsize=8.5)
    ax = axes[1]
    p = pd.read_csv(RES / "adaptive_lambda_paths.csv.gz", parse_dates=["date"])
    for s, c in (("BOOK", CAT[0]), ("USO", CAT[1]), ("TLT", CAT[2])):
        g = p[(p.series == s) & (p.model == "GJR-skew-t")]
        ax.plot(g.date, g.lam, color=c, lw=1.6, label=LABEL[s], drawstyle="steps-post")
    ax.set_ylim(0.45, 1.05); ax.set_ylabel("chosen reporting scale λ")
    ax.set_title("λ chosen each quarter, budget 7, GJR-skew-t", loc="left")
    ax.legend(fontsize=8.5, loc="lower left")
    title(fig, "Without hindsight, shading still pays, but the bill arrives in the red zone",
          "Each quarter the bank picks the smallest λ that would have kept its last 250 days under the budget", y=1.03)
    save(fig, "fig6_adaptive_shading")


# ------------------------------------------------------------------ fig 7: tails vs saving

def fig_tails(sh, tail):
    s = sh[sh.regime == "Basel 2.5"].groupby("series").saving.median().reindex(SERIES)
    t = tail.set_index("series").hill_alpha.reindex(SERIES)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter(t, s * 100, s=70, color=CAT[0], edgecolor=SURFACE, lw=1.2, zorder=3)
    for k in SERIES:
        off = (6, -12) if k == "USO" else (6, 4)
        ax.annotate(LABEL[k], (t[k], s[k] * 100), textcoords="offset points", xytext=off, fontsize=8.5, color=INK2)
    rho = pd.Series(t.values).corr(pd.Series(s.values), method="spearman")
    ax.set_xlabel("Hill tail index of GARCH-standardised losses (lower = fatter)")
    ax.set_ylabel("median capital saved across 6 models, %")
    ax.text(0.98, 0.95, f"Spearman ρ = {rho:.2f} (n = 8)", transform=ax.transAxes, ha="right", color=INK2)
    title(fig, "On real books, tail thickness does not predict the room to shade",
          "Cross-section of the eight books, Basel 2.5, ex-post safe shading (a null result; see section 6.3)", y=1.0)
    save(fig, "fig7_tails_vs_saving")


# ------------------------------------------------------------------ fig 8: stress

def fig_stress(st):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)
    for ax, (w, g) in zip(axes, st.groupby("window", sort=False)):
        piv = g.pivot(index="series", columns="model", values="exceptions").reindex(SERIES)[MODEL_ORDER]
        x = np.arange(len(SERIES)); wdt = 0.13
        for i, m in enumerate(MODEL_ORDER):
            ax.bar(x + (i - 2.5) * wdt, piv[m], wdt, color=COLORS[m], label=m)
        exp = g.groupby("series").expected.first().reindex(SERIES)
        ax.scatter(x, exp, marker="_", s=300, color=INK, zorder=4, label="expected at 99%")
        ax.set_xticks(x, SERIES, fontsize=8.5); ax.set_title(w, loc="left")
    axes[0].set_ylabel("99% VaR exceptions")
    axes[1].legend(fontsize=8.5, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    title(fig, "In crises every model breaches, which is when shading is most dangerous",
          "Exceptions of honest 99% VaR in three stress windows (HYG and URA forecasts start after 2009)", y=1.03)
    save(fig, "fig8_stress_exceptions")


if __name__ == "__main__":
    sh = pd.read_csv(RES / "shading.csv")
    cur = pd.read_csv(RES / "capital_curves.csv.gz")
    ad = pd.read_csv(RES / "adaptive.csv")
    bt = pd.read_csv(RES / "backtests.csv")
    HONEST_RED = sh[sh.regime == "Basel 2.5"][["series", "model", "red_share_honest"]].rename(
        columns={"red_share_honest": "red_honest"})
    fig_break_even()
    fig_theory()
    fig_saving_heatmap(sh)
    fig_score_vs_capital(cur, sh)
    fig_gap(sh)
    fig_adaptive(ad, bt)
    fig_tails(sh, pd.read_csv(RES / "tail_index.csv"))
    fig_stress(pd.read_csv(RES / "stress.csv"))
    print("figures written to", FIG)
