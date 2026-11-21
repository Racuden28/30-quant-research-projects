"""Publication figures for project 10. Run after audit.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: E402

from widgets import M_LABEL, ROOT, WIDGET_LABEL, WIDGETS, FAMILY  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens shared with project 01 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
FAM_COLOR = {"return": "#2a78d6", "trend": "#eb6834", "risk": "#1baf7a"}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f08a89", "#a51c1c"])
POS = LinearSegmentedColormap.from_list("pos", ["#a51c1c", "#f08a89", "#f0efec", "#6da7ec", "#184f95"])
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


def lab(w):
    return WIDGET_LABEL[w]


def fig_information_map(P):
    cols = [("ret", "standalone", "Return\nvs. historical mean"), ("vol", "standalone", "Volatility\nvs. historical mean"),
            ("mdd", "standalone", "Drawdown\nvs. historical mean"), ("vol", "incremental", "Volatility\nbeyond realised vol"),
            ("mdd", "incremental", "Drawdown\nbeyond realised vol")]
    order = sorted(WIDGETS, key=lambda w: (["return", "trend", "risk"].index(FAMILY[w]), w))
    M = np.full((len(order), len(cols)), np.nan)
    DM = np.zeros_like(M, bool)
    for j, (tg, te, _) in enumerate(cols):
        for i, w in enumerate(order):
            r = P[(P.target == tg) & (P.test == te) & (P.widget == w)]
            if len(r):
                M[i, j] = r.r2os.iloc[0] * 100
                DM[i, j] = bool(r.dm_survives_fdr.iloc[0])
    fig, ax = plt.subplots(figsize=(9.6, 7.4))
    vmax = 5
    im = ax.imshow(np.clip(M, -vmax, vmax), cmap=POS, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color=MUTED)
                continue
            mark = " ●" if DM[i, j] else ""
            col = SURFACE if abs(M[i, j]) > 3.2 else INK
            ax.text(j, i, f"{M[i, j]:+.1f}{mark}", ha="center", va="center", fontsize=9, color=col,
                    fontweight="bold" if DM[i, j] else "normal")
    ax.set_xticks(range(len(cols)), [c[2] for c in cols], fontsize=9)
    ax.set_yticks(range(len(order)), [lab(w) for w in order])
    for t, w in zip(ax.get_yticklabels(), order):
        t.set_color(FAM_COLOR[FAMILY[w]])
    ax.xaxis.tick_top()
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(2.5, color=INK, lw=1.2)
    for y in (5.5, 10.5):
        ax.axhline(y, color=SURFACE, lw=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Out-of-sample R² (%), clipped at ±5")
    title(fig, "The screen forecasts risk, not returns",
          "Pooled out-of-sample R² for the next 21 trading days, 7 ETFs, 2012–2026. ● = beats the baseline on the strict "
          "Diebold–Mariano test after BH-FDR (q = 0.10).\nLabel colour: blue = read as return signal, orange = trend, "
          "green = risk. Clark–West results: results/pooled_h21.csv.", y=1.06)
    save(fig, "fig1_information_map")


def fig_incremental(P):
    d = P[(P.target == "vol") & (P.test == "incremental") & P.widget.isin(WIDGETS)].copy()
    d = d.sort_values("r2os")
    fig, ax = plt.subplots(figsize=(9, 5.6))
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        c = FAM_COLOR[r.family]
        ax.plot([r.ci90_lo * 100, r.ci90_hi * 100], [yi, yi], color=c, lw=2.2, alpha=.55, solid_capstyle="round")
        ax.plot(r.r2os * 100, yi, "o", color=c, ms=8, markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.text(r.ci90_hi * 100 + 0.15, yi, "DM ●" if r.dm_survives_fdr else "", va="center", fontsize=8, color=INK2)
    a = P[(P.target == "vol") & (P.test == "incremental") & (P.widget == "ALL")].iloc[0]
    ax.axvline(a.r2os * 100, color=MUTED, ls=(0, (2, 2)), lw=1)
    ax.text(a.r2os * 100, len(d) - .3, f" all 13 together: {a.r2os * 100:+.1f}%", fontsize=8, color=MUTED, va="bottom")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [lab(w) for w in d.widget])
    ax.set_xlabel("Incremental out-of-sample R² for log next-month volatility (%) over a log-RV21 model, 90% block-bootstrap CI")
    for fam, c in FAM_COLOR.items():
        ax.plot([], [], "o", color=c, label={"return": "read as return signal", "trend": "trend", "risk": "risk"}[fam])
    ax.legend(loc="lower right", fontsize=9)
    title(fig, "What the screen knows that realised volatility doesn't",
          "Short-horizon 'return' widgets (Bollinger %b, 5-day change, RSI) carry the most extra volatility information: "
          "the leverage effect.")
    save(fig, "fig2_incremental_vol")


def fig_cumulative(P):
    cs = pd.read_csv(RES / "cumulative_sse_diff_h21.csv.gz", index_col=0, parse_dates=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharex=True)
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for ax, keys, ttl in [
        (axes[0], ["ret|standalone|chg5d", "ret|standalone|streak", "ret|standalone|hi52", "ret|standalone|ma200",
                   "ret|standalone|mom12_1"], "Return: no widget beats the historical mean for long"),
        (axes[1], ["vol|incremental|boll", "vol|incremental|chg5d", "vol|incremental|rsi14", "vol|incremental|dd",
                   "vol|incremental|ma200"], "Volatility beyond RV21: the gains accumulate steadily")]:
        for c, k in zip(pal, keys):
            s = cs[k].dropna()
            ax.plot(s.index, s.values, color=c, lw=1.8, label=lab(k.split("|")[2]))
        ax.axhline(0, color=INK, lw=1)
        for a0, a1 in [("2020-02-15", "2020-05-31"), ("2022-01-01", "2022-10-31")]:
            ax.axvspan(pd.Timestamp(a0), pd.Timestamp(a1), color=GRID, alpha=.6, lw=0)
        ax.set_title(ttl, loc="left")
        ax.legend(fontsize=8, loc="lower left" if ax is axes[0] else "upper left")
    axes[0].set_ylabel("Cumulative scaled SSE(baseline) − SSE(widget)\n(up = widget winning)")
    title(fig, "Where the information shows up, and when",
          "Goyal–Welch cumulative squared-error differences, pooled across 7 ETFs; shaded: COVID 2020 and 2022 rate shock.",
          y=1.04)
    save(fig, "fig3_cumulative_sse")


def fig_horizon():
    H = pd.read_csv(RES / "horizon_profile.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    pal = dict(zip(["chg5d", "boll", "rsi14", "hi52", "dd", "ma200"],
                   ["#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#e87ba4", "#008300"]))
    for ax, tg, te, ttl in [(axes[0], "ret", "standalone", "Return, vs. historical mean"),
                            (axes[1], "vol", "incremental", "Volatility, beyond log RV21")]:
        for w, c in pal.items():
            g = H[(H.target == tg) & (H.test == te) & (H.widget == w)].sort_values("h")
            ax.plot(range(len(g)), g.r2os * 100, "-o", color=c, lw=2, ms=6, markeredgecolor=SURFACE, label=lab(w))
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(range(4), ["1 day", "1 week", "1 month", "3 months"])
        ax.set_title(ttl, loc="left")
        ax.set_ylabel("Pooled out-of-sample R² (%)")
    axes[0].set_ylim(top=1.5)
    axes[1].legend(fontsize=8, ncol=2)
    title(fig, "How long a screen's information lasts",
          "Same pipeline at four forecast horizons. Return information never appears; volatility information peaks at "
          "one week to one month and decays.", y=1.04)
    save(fig, "fig4_horizon_profile")


def fig_shiller():
    S = pd.read_csv(RES / "shiller_monthly.csv")
    cols = [("ret1", "standalone", "Next month's\nreturn"), ("ret12", "standalone", "Next 12 months'\nreturn"),
            ("vol3", "standalone", "Next 3 months'\nvolatility"), ("vol3", "incremental", "Volatility beyond\n12-month vol"),
            ("mdd12", "incremental", "12-month drawdown\nbeyond 12-month vol")]
    ws = list(M_LABEL)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.8 / len(ws)
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cc4"]
    for k, (w, c) in enumerate(zip(ws, pal)):
        vals, cw = [], []
        for tg, te, _ in cols:
            r = S[(S.target == tg) & (S.test == te) & (S.widget == w)]
            vals.append(r.r2os.iloc[0] * 100 if len(r) else np.nan)
            cw.append(bool(r.dm_survives_fdr.iloc[0]) if len(r) else False)
        x = np.arange(len(cols)) + (k - len(ws) / 2 + .5) * width
        ax.bar(x, np.clip(vals, -8, 12), width * .92, color=c, label=M_LABEL[w])
        for xi, v, s in zip(x, vals, cw):
            if s and np.isfinite(v):
                ax.text(xi, max(v, 0) + .3, "●", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(range(len(cols)), [c[2] for c in cols])
    ax.set_ylabel("Out-of-sample R² (%), clipped to [−8, 12]")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    title(fig, "122 years later, the same answer",
          "S&P 500 monthly, out of sample 1904–2026 (30-year burn-in, one-month skip for averaged prices). "
          "● beats the historical mean on Diebold–Mariano after BH-FDR (q = 0.10).", y=1.04)
    save(fig, "fig5_shiller_1871_2026")


def fig_redundancy():
    C = pd.read_csv(RES / "widget_rank_correlation.csv", index_col=0)
    E = pd.read_csv(RES / "widget_pca.csv")
    o = leaves_list(linkage(1 - np.abs(C.values[np.triu_indices(len(C), 1)]), "average"))
    C = C.iloc[o, o]
    fig = plt.figure(figsize=(12, 5.4))
    ax = fig.add_axes([0.05, 0.08, 0.5, 0.82])
    im = ax.imshow(C.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(C)), [lab(w) for w in C.columns], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(C)), [lab(w) for w in C.index], fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Spearman correlation (pooled)")
    ax2 = fig.add_axes([0.72, 0.15, 0.26, 0.7])
    ax2.bar(E.component, E.eigenvalue, color=["#2a78d6" if v > 1 else AXIS for v in E.eigenvalue])
    ax2.axhline(1, color=INK, ls=(0, (4, 3)), lw=1)
    ax2.text(14.4, 1.05, "Kaiser rule", ha="right", fontsize=8, color=INK2)
    ax2.set_xlabel("Principal component")
    ax2.set_xticks(range(1, 15, 2))
    ax2.set_ylabel("Eigenvalue")
    ax2.set_title(f"{(E.eigenvalue > 1).sum()} independent signals, "
                  f"{E.share[:4].sum():.0%} of the variance", loc="left", fontsize=10)
    title(fig, "Fourteen numbers, four ideas", "Rank correlations of the widgets across 7 ETFs (out-of-sample period).",
          y=0.99)
    save(fig, "fig6_redundancy")


def fig_stress():
    S = pd.read_csv(RES / "stress_h21.csv")
    d = S[(S.target == "vol") & (S.test == "incremental") & S.widget.isin(WIDGETS)]
    piv = d.pivot(index="widget", columns="window", values="r2os") * 100
    full = pd.read_csv(RES / "pooled_h21.csv")
    full = full[(full.target == "vol") & (full.test == "incremental")].set_index("widget").r2os * 100
    piv["Full sample"] = full
    piv = piv.sort_values("Full sample")[["Full sample", "COVID 2020", "Rates 2022"]]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(piv))
    for c, (col, m) in zip(["#0b0b0b", "#eb6834", "#2a78d6"], [("Full sample", "o"), ("COVID 2020", "s"),
                                                              ("Rates 2022", "D")]):
        ax.plot(piv[col], y, m, color=c, ms=7, markeredgecolor=SURFACE, label=col, ls="none")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [lab(w) for w in piv.index])
    ax.set_xlabel("Incremental out-of-sample R² for next-month volatility (%)")
    ax.legend(fontsize=9, loc="lower right")
    title(fig, "Crash versus grind", "The leverage-type widgets help most in a fast crash (COVID) and little in a slow "
          "rate-driven bear market (2022).")
    save(fig, "fig7_stress")


if __name__ == "__main__":
    P = pd.read_csv(RES / "pooled_h21.csv")
    fig_information_map(P)
    fig_incremental(P)
    fig_cumulative(P)
    fig_shiller()
    fig_redundancy()
    fig_stress()
    if (RES / "horizon_profile.csv").exists():
        fig_horizon()
    print("figures written to", FIG)
