"""Publication figures for project 17. Run after analysis.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# design tokens shared with projects 01 and 09 (validated categorical order: blue, orange, green)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
C_IS, C_HO = BLUE, ORANGE
L_IS, L_HO = "In-sample 1963–2013 (the data FF 2015 used)", "Holdout 2014–2026 (data that did not exist yet)"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
})
IS_LAB, HO_LAB = "In-sample 1963-2013", "Holdout 2014-2026"


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def title(fig, t, sub, y=1.0):
    fig.text(0.01, y, t, fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(0.01, y - 0.045, sub, fontsize=10, color=INK2, ha="left", va="bottom")


def period_legend(ax, loc="lower right"):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C_IS, marker="o", ms=8, lw=2, label=L_IS),
         Line2D([], [], color=C_HO, marker="o", ms=8, lw=2, label=L_HO)]
    ax.legend(handles=h, loc=loc, fontsize=9)


def fig1_report_card():
    sp = pd.read_csv(RES / "spanning.csv")
    rows = [("FF5", "SMB"), ("FF5", "HML"), ("FF5", "RMW"), ("FF5", "CMA"), ("FF6", "UMD")]
    tg = pd.read_csv(RES / "tangency_holdout.csv", index_col=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.05]})
    labels = []
    for i, (s, f) in enumerate(rows):
        r = sp[(sp["set"] == s) & (sp["factor"] == f)].iloc[0]
        y = len(rows) - 1 - i
        for lab, col, dy in [(IS_LAB, C_IS, 0.14), (HO_LAB, C_HO, -0.14)]:
            m, se = r[f"{lab} alpha %/m"], r[f"{lab} se %/m"]
            a1.plot([m - 1.96 * se, m + 1.96 * se], [y + dy] * 2, color=col, lw=2, solid_capstyle="round")
            a1.plot(m, y + dy, "o", color=col, ms=8, mec=SURFACE, mew=2)
        others = "Mkt, SMB, RMW, CMA" if f == "HML" else ("other five" if s == "FF6" else "other four")
        labels.append(f"{f} (on {others})\nchange p = {r['alpha change p']:.3f}")
    a1.axvline(0, color=AXIS, lw=1)
    a1.set_yticks(range(len(rows)))
    a1.set_yticklabels(labels[::-1])
    a1.set_xlim(-1.05, 1.3)
    a1.set_xlabel("spanning-regression alpha, % per month (95% CI, Newey–West)")
    a1.set_title("A. What each factor adds beyond the others", loc="left")
    a1.grid(axis="y", visible=False)

    models = ["CAPM", "FF3", "C4", "FF5", "FF6"]
    x = np.arange(len(models))
    for j, m in enumerate(models):
        r = tg.loc[m]
        a2.plot(j - 0.12, r["IS Sharpe adj"], "o", color=C_IS, ms=9, mec=SURFACE, mew=2)
        a2.plot([j + 0.12] * 2, [r["HO lo95"], r["HO hi95"]], color=C_HO, lw=2, solid_capstyle="round")
        a2.plot(j + 0.12, r["HO Sharpe"], "o", color=C_HO, ms=9, mec=SURFACE, mew=2)
        a2.text(j + 0.2, r["HO Sharpe"], f"{r['HO Sharpe']:.2f}", fontsize=8.5, color=INK2, va="center")
        a2.text(j - 0.22, r["IS Sharpe adj"], f"{r['IS Sharpe adj']:.2f}", fontsize=8.5, color=INK2,
                va="center", ha="right")
    a2.axhline(0, color=AXIS, lw=1)
    a2.set_xticks(x)
    a2.set_xticklabels(["CAPM\n(Mkt)", "FF3", "Carhart 4", "FF5", "FF6\n(FF5+UMD)"])
    a2.set_ylabel("annualised Sharpe ratio of the model's tangency portfolio")
    a2.set_title("B. The best mix of each model's factors, weights fixed in 2013", loc="left")
    a2.grid(axis="x", visible=False)
    a2.text(0.01, -0.2, "Blue: in-sample Sharpe, bias-adjusted (Kan–Zhou). Orange: same weights, holdout, "
            "with 95% stationary-bootstrap CI.", transform=a2.transAxes, fontsize=8.5, color=MUTED)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=C_IS, marker="o", ms=8, lw=2, label=L_IS),
                        Line2D([], [], color=C_HO, marker="o", ms=8, lw=2, label=L_HO)],
               loc="upper left", bbox_to_anchor=(0.01, 0.985), ncol=2, fontsize=9.5)
    fig.subplots_adjust(wspace=0.32)
    title(fig, "The five-factor model's holdout exam",
          "HML stayed redundant after 2013, but only because SMB, RMW and CMA stopped adding anything too. "
          "The 2013 factor mix kept 30% of its Sharpe; the market alone did best.", y=1.06)
    save(fig, "fig1_holdout_report_card")


def fig2_realtime_t():
    H = pd.read_csv(RES / "realtime_history.csv")
    yr = H["decision"].str[:4].astype(int)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axvspan(2013.5, yr.max() + 0.5, color=GRID, alpha=0.5, lw=0)
    ax.text(2013.8, 6.6, "holdout", fontsize=9, color=INK2)
    ax.axhline(1.96, color=MUTED, lw=1, ls="--")
    ax.axhline(0, color=AXIS, lw=1)
    ax.text(1983.2, 2.05, "t = 1.96", fontsize=8.5, color=MUTED)
    series = [("t HML|FF5", "HML | FF5", ORANGE, 2.4), ("t HML|FF6", "HML | FF6 (with UMD)", ORANGE, 1.2),
              ("t RMW|FF5", "RMW | FF5", BLUE, 1.6), ("t CMA|FF5", "CMA | FF5", GREEN, 1.6),
              ("t SMB|FF5", "SMB | FF5", INK2, 1.2)]
    for col, lab, c, lw in series:
        ls = ":" if "FF6" in lab else "-"
        ax.plot(yr, H[col], color=c, lw=lw, ls=ls, label=lab)
        ax.text(yr.iloc[-1] + 0.4, H[col].iloc[-1], lab, fontsize=8.5, color=INK, va="center")
    ax.set_xlim(1983, yr.max() + 6.5)
    ax.set_xticks(range(1985, 2030, 5))
    ax.set_ylabel("t-statistic of the spanning alpha (expanding window)")
    ax.set_xlabel("data available up to December of year")
    ax.legend(loc="upper left", ncol=3, fontsize=8.5)
    title(fig, "When would you have known HML was redundant?",
          "In real time HML's spanning alpha had t between 1.7 and 2.3 every year from 1985 to 1998. "
          "Its redundancy is a post-1999 fact; with momentum in the model (dotted) HML stays positive.", y=1.0)
    save(fig, "fig2_realtime_redundancy")


def fig3_premia():
    pr = pd.read_csv(RES / "factor_premia.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(pr))
    for dx, lab, c in [(-0.13, IS_LAB, C_IS), (0.13, HO_LAB, C_HO)]:
        m, lo, hi = pr[f"{lab} mean %/m"], pr[f"{lab} lo95"], pr[f"{lab} hi95"]
        for xi, a, b, mm in zip(x + dx, lo, hi, m):
            ax.plot([xi, xi], [a, b], color=c, lw=2, solid_capstyle="round")
            ax.plot(xi, mm, "o", color=c, ms=8, mec=SURFACE, mew=2)
    for xi, p in zip(x, pr["change p"]):
        ax.text(xi, -0.95, f"p = {p:.2f}", ha="center", fontsize=8.5, color=INK2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pr.index)
    ax.set_ylim(-1.05, 1.65)
    ax.set_ylabel("average return, % per month (95% CI)")
    ax.grid(axis="x", visible=False)
    period_legend(ax, "upper right")
    jt = pd.read_csv(RES / "factor_premia_joint_test.csv").iloc[0]
    title(fig, "Only the market kept its premium",
          f"p-values test each change; jointly, the five non-market premia shifted (Wald χ²(5) = {jt['chi2']:.1f}, "
          f"p = {jt['p']:.3f}).", y=1.0)
    save(fig, "fig3_factor_premia")


def fig4_placebo():
    pl = pd.read_csv(RES / "placebo_holdouts.csv")
    s = pd.to_datetime(pl["start"]).dt.year
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axhline(0, color=AXIS, lw=1)
    for col, c, lab in [("FF5 - CAPM", BLUE, "FF5 minus CAPM"), ("FF6 - CAPM", GREEN, "FF6 minus CAPM")]:
        ax.plot(s, pl[col], color=c, lw=2, label=lab, marker="o", ms=4)
        h = pl[pl["is_holdout"]]
        ax.plot(pd.to_datetime(h["start"]).dt.year, h[col], "o", color=c, ms=11, mec=INK, mew=1.2)
        ax.text(s.iloc[-1] + 0.4, pl[col].iloc[-1], lab, fontsize=8.5, va="center")
    ax.annotate("the holdout\n(2014-01 to 2026-09)", (2014, pl["FF5 - CAPM"].iloc[-1]), (2016.5, -0.62),
                fontsize=8.5, color=INK2, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set_xlim(1983, 2021)
    ax.set_ylim(-0.75, 1.45)
    ax.set_xlabel("start of a 153-month window (real-time weights, re-estimated every December)")
    ax.set_ylabel("Sharpe ratio gap vs the market alone")
    ax.legend(loc="upper right", fontsize=9)
    title(fig, "Was the holdout just a bad draw?",
          "Every 153-month real-time window since 1984. The holdout is the worst one for FF5; the windows overlap, "
          "so there are only about three independent draws.", y=1.0)
    save(fig, "fig4_placebo_windows")


def fig5_pricing():
    o = pd.read_csv(RES / "pricing_oos.csv")
    fams = ["25 Size-B/M", "25 Size-OP", "25 Size-Inv", "10 Momentum", "49 Industries"]
    models = ["CAPM", "FF3", "C4", "FF5", "FF5-HML", "FF6"]
    col = "RMS a net of noise %/m"
    fig, axes = plt.subplots(1, 5, figsize=(14, 4.6), sharey=True)
    for ax, fam in zip(axes, fams):
        d = o[o["assets"] == fam]
        for j, m in enumerate(models):
            a = d[(d["model"] == m) & d["period"].str.startswith("In")][col].iloc[0]
            b = d[(d["model"] == m) & d["period"].str.startswith("Hold")][col].iloc[0]
            ax.plot([max(a, 0), max(b, 0)], [j, j], color=GRID, lw=2.5, zorder=1)
            for v, c in [(a, C_IS), (b, C_HO)]:
                ax.plot(max(v, 0), j, "o", ms=8, zorder=2, mew=2,
                        color=c if v > 0 else SURFACE, mec=c if v <= 0 else SURFACE)
        ax.set_title(fam, loc="left", fontsize=10)
        ax.set_xlim(-0.01, 0.45)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("RMS α net of noise, %/month")
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels(models)
    axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=C_IS, marker="o", ms=8, lw=0, label="In-sample 1968–2013"),
                        Line2D([], [], color=C_HO, marker="o", ms=8, lw=0, label=L_HO),
                        Line2D([], [], color=INK2, marker="o", ms=8, lw=0, mfc=SURFACE, mew=2,
                               label="hollow: no larger than sampling noise (plotted at 0)")],
               loc="upper left", bbox_to_anchor=(0.01, 0.985), ncol=3, fontsize=9.5)
    title(fig, "Out-of-sample pricing errors, after removing sampling noise",
          "Betas from the previous 60 months only. Root-mean-square alpha minus its Newey–West sampling variance. "
          "Many holdout errors are within noise; FF5's in-sample edge on size–profitability is gone.", y=1.07)
    save(fig, "fig5_oos_pricing")


def fig6_stress():
    st = pd.read_csv(RES / "stress_windows.csv", index_col=0)
    cols = ["Mkt", "SMB", "HML", "RMW", "CMA", "UMD", "RT tangency CAPM", "RT tangency FF5", "RT tangency FF6"]
    names = ["Mkt", "SMB", "HML", "RMW", "CMA", "UMD", "CAPM*", "FF5*", "FF6*"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, (w, r) in zip(axes, st.iterrows()):
        v = r[cols].values.astype(float)
        cs = [INK2] * 6 + [BLUE] * 3
        ax.bar(range(len(v)), v, color=cs, width=0.7)
        for i, val in enumerate(v):
            ax.text(i, val + (1.5 if val >= 0 else -1.5), f"{val:.0f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=8, color=INK2)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(f"{w} ({int(r['months'])} months)", loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("cumulative return, %")
    title(fig, "Factors as crash insurance",
          "Gray: single factors. Blue (*): each model's real-time tangency portfolio, weights estimated before "
          "the window, scaled to 10% ex-ante volatility.", y=1.02)
    save(fig, "fig6_stress_windows")


if __name__ == "__main__":
    fig1_report_card()
    fig2_realtime_t()
    fig3_premia()
    fig4_placebo()
    fig5_pricing()
    fig6_stress()
    print("figures written to", FIG)
