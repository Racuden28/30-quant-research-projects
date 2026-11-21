"""Publication figures for project 02. Run after evaluate.py.

    python src/figures.py
"""
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
from estimators import ESTIMATORS, WEIGHTINGS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens shared with project 01 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f08a89", "#a51c1c"])
ASSETS = ["SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
DEF = "C2C|R21|sqrt"

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


def pct(x):
    return (x - 1) * 100


def fig1_which_choice(wf):
    """Star chart: out-of-sample loss change from letting each dashboard choice be data-driven."""
    w = wf[wf.h == 22]
    order = ["estimator only", "weighting only", "estimator + weighting", "full grid", "grid + benchmarks"]
    labels = ["Estimator only\n(close vs 4 range estimators)", "Window only\n(12 rolling / EWMA)",
              "Estimator + window\n(60 cells)", "All three choices\n(+ √t vs variance ratio, 120 cells)",
              "120 cells + HAR & GJR-GARCH"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, v in enumerate(order):
        g = w[w.variant == v]
        med = pct(g.qlike_ratio.median())
        ax.barh(i, med, color=CAT[0] if i < 2 else (CAT[3] if i < 4 else CAT[2]), height=0.6, alpha=0.9)
        ax.scatter(pct(g.qlike_ratio), np.full(len(g), i) + np.linspace(-0.2, 0.2, len(g)), s=14,
                   color=INK, zorder=3, alpha=0.7, linewidth=0)
        ax.text(min(pct(g.qlike_ratio).min(), med) - 1.2, i, f"{med:+.0f}%", va="center", ha="right",
                fontsize=10, fontweight="bold")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(order)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Change in out-of-sample QLIKE loss vs the textbook dashboard (%)  ·  bars: median across 10 ETFs, dots: each ETF")
    lo = pct(w[w.variant.isin(order)].qlike_ratio).min()
    ax.set_xlim(lo - 9, max(3, pct(w[w.variant.isin(order)].qlike_ratio).max() + 2))
    title(fig, "Which hidden dashboard choice matters?",
          "Walk-forward selection each January using only past data, 22-day horizon, 2005–2026 (negative = better). Blue: one choice; amber: choices combined; green: with forecasting models", y=0.99)
    fig.subplots_adjust(top=0.86)
    save(fig, "fig1_which_choice_matters")


def fig2_grid_heatmap(scores):
    s = scores[(scores.h == 22) & scores.model.str.endswith("|sqrt")]
    s = s[s.model.str.split("|").str[0].isin(ESTIMATORS)].copy()
    s[["e", "w", "_"]] = s.model.str.split("|", expand=True)
    m = s.groupby(["e", "w"]).qlike_ratio.mean().unstack()[WEIGHTINGS].loc[ESTIMATORS]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    v = pct(m.to_numpy())
    norm = TwoSlopeNorm(vcenter=0, vmin=min(v.min(), -1), vmax=max(v.max(), 1))
    ax.imshow(v, cmap=DIVERGING, norm=norm, aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{v[i, j]:+.0f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(norm(v[i, j]) - 0.5) > 0.35 else INK)
    wl = [w.replace("R", "roll ").replace("E", "EWMA hl ") for w in WEIGHTINGS]
    ax.set_xticks(range(len(WEIGHTINGS)), wl, rotation=35, ha="right")
    ax.set_yticks(range(len(ESTIMATORS)), ["Close-to-close", "Parkinson", "Garman–Klass", "Rogers–Satchell", "Yang–Zhang"])
    ax.grid(False)
    i0, j0 = ESTIMATORS.index("C2C"), WEIGHTINGS.index("R21")
    ax.add_patch(plt.Rectangle((j0 - .5, i0 - .5), 1, 1, fill=False, ec=INK, lw=2))
    title(fig, "A short EWMA on any range estimator beats the textbook cell by ~16%",
          "QLIKE loss vs the textbook cell (boxed), %, mean over 10 ETFs, 22-day horizon, √t scaling, 2005–2026", y=1.02)
    save(fig, "fig2_estimator_window_grid")


def fig3_conditional_bias(bias):
    b = bias[bias.h == 22]
    lab = {DEF: "Textbook 21d close-to-close", "C2C|E10|sqrt": "EWMA 10d close-to-close",
           "GK|E10|sqrt": "EWMA 10d Garman–Klass", "HAR-GK": "HAR on range", "GJR-GARCH-t": "GJR-GARCH-t"}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, (m, l) in enumerate(lab.items()):
        bb = b[b.model == m].assign(r=lambda d: (d.ann_vol_realized / d.ann_vol_forecast) ** 2)
        g = bb.groupby("quintile").r.median()
        ax.plot(g.index, (g.values - 1) * 100, marker="o", lw=2.2 if m == DEF else 1.6,
                color=INK if m == DEF else CAT[i], label=l, ms=6, markeredgecolor=SURFACE)
    ax.axhline(0, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(range(1, 6), ["Q1\ncalmest", "Q2", "Q3", "Q4", "Q5\nmost volatile"])
    ax.set_xlabel("Quintile of the textbook number at the forecast date")
    ax.set_ylabel("Mean realized / mean forecast variance − 1 (%)")
    ax.legend(loc="lower left", fontsize=9)
    title(fig, "The textbook number is too low when calm and too high after spikes",
          "Next-22-day realized variance relative to the forecast, by volatility state at the forecast date, median over 10 ETFs", y=1.0)
    fig.subplots_adjust(top=0.86)
    save(fig, "fig3_conditional_bias")


def fig4_data_quality(dq, scores_year):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for i, a in enumerate(ASSETS):
        g = dq[dq.asset == a].set_index("year")
        g = g[g.index >= 1999]
        ax.plot(g.index, g.open_at_extreme * 100, color=INK if a == "SPY" else CAT[0], alpha=1 if a == "SPY" else 0.35,
                lw=2 if a == "SPY" else 1.2, label="SPY" if a == "SPY" else ("sector ETFs" if i == 1 else None))
    ax.set_ylabel("% of days where the open is the day's high or low")
    ax.set_title("Free daily OHLC: stale opens before ~2010", loc="left")
    ax.legend()
    ax = axes[1]
    ax.scatter(scores_year.open_at_extreme * 100, pct(scores_year.ratio), s=14, color=CAT[1], alpha=0.6, linewidth=0)
    x, y = scores_year.open_at_extreme * 100, pct(scores_year.ratio)
    b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 10)
    ax.plot(xx, np.polyval(b, xx), color=INK, lw=1.5)
    ax.axhline(0, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_ylim(-75, 110)
    n_out = int((y > 110).sum())
    if n_out:
        ax.text(0.99, 0.98, f"{n_out} point above 110% not shown", transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=INK2)
    ax.set_xlabel("% of days with the open at the high or low (asset-year)")
    ax.set_ylabel("Range (GK) loss vs close-to-close, %")
    ax.set_title(f"Dirtier years: a smaller edge, but still an edge (slope {b[0]:+.2f} pp per pp)", loc="left")
    title(fig, "Free OHLC data are dirty, and range estimators survive it",
          "Left: data quality by year. Right: QLIKE of 21-day Garman–Klass relative to 21-day close-to-close, per ETF-year, 22-day horizon", y=1.02)
    save(fig, "fig4_data_quality")


def fig5_sqrt_time(vr, sr):
    v = vr[vr.period == "1999-2026"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    qs = [5, 22, 66, 252]
    wdt = 0.8 / len(qs)
    for k, q in enumerate(qs):
        g = v[v.q == q].set_index("asset").loc[ASSETS]
        x = np.arange(len(ASSETS)) + (k - 1.5) * wdt
        ax.bar(x, g.vr, width=wdt, color=CAT[k], label=f"q = {q} days")
        for xi, (vv, p) in zip(x, zip(g.vr, g.p)):
            if p < 0.05:
                ax.text(xi, vv + 0.01, "*", ha="center", fontsize=9)
    ax.axhline(1, color=INK, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(range(len(ASSETS)), ASSETS)
    ax.set_ylim(0.4, 1.2)
    ax.set_ylabel("Variance ratio VR(q)")
    ax.legend(ncol=4, loc="upper right", fontsize=8.5)
    ax.set_title("Lo–MacKinlay variance ratios, 1999–2026 (* robust p < 0.05)", loc="left")
    ax = axes[1]
    g = sr.groupby("h")[["realized_over_forecast_sqrt", "realized_over_forecast_vr"]].median()
    x = np.arange(len(g))
    ax.bar(x - 0.18, g.realized_over_forecast_sqrt, 0.36, color=INK2, label="√t rule")
    ax.bar(x + 0.18, g.realized_over_forecast_vr, 0.36, color=CAT[0], label="variance-ratio corrected")
    for xi, val in zip(x - 0.18, g.realized_over_forecast_sqrt):
        ax.text(xi, val + 0.01, f"{val:.2f}", ha="center", fontsize=8.5)
    for xi, val in zip(x + 0.18, g.realized_over_forecast_vr):
        ax.text(xi, val + 0.01, f"{val:.2f}", ha="center", fontsize=8.5)
    ax.axhline(1, color=INK, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(x, [f"{h} days" for h in g.index])
    ax.set_ylim(0.5, 1.15)
    ax.set_ylabel("Mean squared h-day return / mean forecast")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax.set_title("Out of sample, 2005–2026, median of 600 cells", loc="left")
    title(fig, "√t overstates multi-week risk in almost every sector",
          "Left: variance of q-day returns relative to q × daily variance. Right: how much of the forecast h-day return variance actually arrived", y=1.02)
    save(fig, "fig5_sqrt_time")


def fig6_stress(stress):
    s = stress[stress.h == 22]
    models = {"C2C|E10|sqrt": "EWMA 10d C2C", "GK|E10|sqrt": "EWMA 10d GK", "HAR-GK": "HAR on range",
              "GJR-GARCH-t": "GJR-GARCH-t", "C2C|R252|sqrt": "1-year C2C"}
    wins = list(s.window.unique())
    fig, ax = plt.subplots(figsize=(10, 4.2))
    wdt = 0.8 / len(models)
    for k, (m, l) in enumerate(models.items()):
        vals = [pct(s[(s.window == w) & (s.model == m)].qlike_ratio.median()) for w in wins]
        x = np.arange(len(wins)) + (k - (len(models) - 1) / 2) * wdt
        ax.bar(x, vals, width=wdt, color=CAT[k], label=l)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(range(len(wins)), wins)
    ax.set_ylabel("QLIKE vs textbook (%), median over ETFs")
    ax.legend(ncol=3, fontsize=9, loc="best")
    title(fig, "In crises the gap widens",
          "Forecast origins inside each stress window, 22-day horizon (negative = better than the textbook number)", y=1.0)
    fig.subplots_adjust(top=0.86)
    save(fig, "fig6_stress_windows")


def gk_vs_c2c_by_year():
    """Per asset-year QLIKE ratio GK|R21 over C2C|R21 at h = 22, joined with data quality."""
    z = np.load(RES / "forecasts_h22.npz", allow_pickle=True)
    dates = pd.to_datetime(z["dates"])
    models = list(z["models"])
    i0, i1 = models.index(DEF), models.index("GK|R21|sqrt")
    rows = []
    for a in ASSETS:
        F, RV = z[f"F_{a}"].astype(float), z[f"RV_{a}"]
        ok = ~np.isnan(F[:, [i0, i1]]).any(1) & ~np.isnan(RV) & (RV > 0)
        d = pd.DataFrame({"y": dates[ok].year, "l0": qlike(RV[ok], F[ok, i0]), "l1": qlike(RV[ok], F[ok, i1])})
        g = d.groupby("y").mean()
        for y, r in g.iterrows():
            rows.append(dict(asset=a, year=y, ratio=r.l1 / r.l0))
    return pd.DataFrame(rows)


def qlike(rv, f):
    x = rv / f
    return x - np.log(x) - 1


def main():
    wf = pd.read_csv(RES / "wf.csv")
    scores = pd.read_csv(RES / "scores.csv")
    bias = pd.read_csv(RES / "bias.csv")
    dq = pd.read_csv(RES / "data_quality_by_year.csv")
    vr = pd.read_csv(RES / "variance_ratios.csv")
    stress = pd.read_csv(RES / "stress.csv")
    fig1_which_choice(wf)
    fig2_grid_heatmap(scores)
    fig3_conditional_bias(bias)
    sy = gk_vs_c2c_by_year().merge(dq, on=["asset", "year"])
    sy = sy[(sy.year >= 2001) & (sy.year <= 2025)]
    sy.to_csv(RES / "gk_vs_c2c_by_year.csv", index=False, float_format="%.6g")
    fig4_data_quality(dq, sy)
    fig5_sqrt_time(vr, pd.read_csv(RES / "scaling_return_target.csv"))
    fig6_stress(stress)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
