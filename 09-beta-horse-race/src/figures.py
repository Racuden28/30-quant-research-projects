"""Publication figures for project 09. Run after backtest.py, evaluate.py, horizon_gap.py, robustness.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from estimators import BASELINES, ESTIMATORS, TEXTBOOK  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens shared with project 01 (validated: blue, orange, green; green needs direct labels)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f08a89", "#a51c1c"])
MODELS = ESTIMATORS + BASELINES
LABEL = {"OLS-5Y-M": "OLS 5y monthly (textbook)", "OLS-1Y-D": "OLS 1y daily", "Blume": "Blume",
         "Vasicek": "Vasicek", "EWMA": "EWMA (hl 6m)", "Dimson": "Dimson (2 lags)",
         "Frazzini-Pedersen": "Frazzini–Pedersen", "Welch-SWA": "Welch slope-winsorized",
         "Combination": "Equal-weight combination", "One": "β = 1"}

SHORT = {"OLS-5Y-M": "OLS 5y M", "OLS-1Y-D": "OLS 1y D", "Blume": "Blume", "Vasicek": "Vasicek",
         "EWMA": "EWMA", "Dimson": "Dimson", "Frazzini-Pedersen": "Frazzini–\nPedersen",
         "Welch-SWA": "Welch\nSWA", "Combination": "Combi-\nnation", "One": "β = 1"}

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


def fig1_horizon_flip():
    d = pd.read_csv(RES / "forecast_daily_next_month.csv", index_col=0)
    m = pd.read_csv(RES / "forecast_monthly_next_year.csv", index_col=0)
    order = d.drop(index=TEXTBOOK)["excess_ratio"].sort_values().index.tolist() + [TEXTBOOK]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    y = np.arange(len(order))[::-1]
    for yi, mod in zip(y, order):
        a, b = d.loc[mod, "excess_ratio"], m.loc[mod, "excess_ratio"]
        ax.plot([a, b], [yi, yi], color=AXIS, lw=2, zorder=1)
    for df, col, mk, lab in [(d, BLUE, "o", "Hedge next month (daily returns)"),
                             (m, ORANGE, "s", "Hedge next 12 months (monthly returns)")]:
        v = df.loc[order]
        ax.errorbar(v.excess_ratio, y, xerr=[v.excess_ratio - v.ratio_lo, v.ratio_hi - v.excess_ratio],
                    fmt=mk, color=col, ms=8, mec=SURFACE, mew=1.2, elinewidth=1.4, capsize=0, label=lab, zorder=3)
    ax.axvline(1, color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(1.005, len(order) - 0.45, "textbook = 1", fontsize=8.5, color=INK2, va="bottom")
    ax.set_yticks(y, [LABEL[o] for o in order])
    ax.set_xlabel("Beta-error hedge loss relative to the 5-year monthly OLS beta  (lower is better; 95% bootstrap CI)")
    ax.set_xlim(0.45, 1.45)
    ax.legend(loc="upper left", bbox_to_anchor=(0.66, 0.42), fontsize=9)
    ax.grid(axis="y", visible=False)
    title(fig, "Daily betas cut the textbook's beta error by ~40% next month, then lose that edge over a year",
          "49 US industry portfolios, 1929–2026, 54,245 out-of-sample forecasts. Same forecasts, two horizons.", y=0.99)
    save(fig, "fig1_horizon_flip")


def fig2_concentration():
    Lm = pd.read_csv(RES / "monthly_losses_daily.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    idx = pd.PeriodIndex(Lm.index, freq="M").to_timestamp()
    for mod, col, lab in [("Welch-SWA", BLUE, "Welch"), ("EWMA", GREEN, "EWMA"),
                          ("Frazzini-Pedersen", ORANGE, "Frazzini–Pedersen")]:
        cum = (Lm[TEXTBOOK] - Lm[mod]).cumsum()
        ax.plot(idx, cum, color=col, lw=2, label=lab)
        off = {"Welch-SWA": 2, "EWMA": -2}.get(mod, 0)
        ax.text(idx[-1], cum.iloc[-1] + off, "  " + lab, color=INK2, fontsize=8.5, va="center")
    vol = np.sqrt(Lm["mkt_var"] * 252)
    hi = vol >= vol.quantile(0.9)
    for t in idx[hi.to_numpy()]:
        ax.axvspan(t, t + pd.offsets.MonthEnd(1), color="#f08a89", alpha=0.45, lw=0)
    ax.set_ylabel("Cumulative hedge-loss saved vs textbook (%²)")
    ax.set_title("Cumulative gain; shaded = top-10% volatility months", loc="left")
    ax.set_xlim(idx[0], idx[-1] + pd.Timedelta(days=3000))
    ax = axes[1]
    order = vol.sort_values(ascending=False).index
    for mod, col in [("Welch-SWA", BLUE), ("EWMA", GREEN), ("Frazzini-Pedersen", ORANGE)]:
        adv = (Lm[TEXTBOOK] - Lm[mod]).loc[order]
        share = adv.cumsum() / adv.sum()
        ax.plot(np.arange(1, len(adv) + 1) / len(adv), share, color=col, lw=2)
    ax.plot([0, 1], [0, 1], color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(0.55, 0.47, "if every month\ncounted equally", fontsize=8.5, color=INK2, rotation=0)
    ax.axvline(0.1, color=MUTED, lw=0.8)
    ax.text(0.11, 0.08, "top 10% of months\n≈ half the gain", fontsize=8.5, color=INK2)
    ax.set_xlabel("Share of months, most volatile first")
    ax.set_ylabel("Share of total advantage")
    ax.set_title("Share of the gain vs share of months", loc="left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.15)
    title(fig, "A better beta is worth most exactly when the market is wild",
          "Hedge loss is (β̂ − β)² × market variance: estimation error is priced by volatility.", y=1.02)
    save(fig, "fig2_concentration")


def fig3_horizon_gap():
    g = pd.read_csv(RES / "horizon_gap.csv")
    s = pd.read_csv(RES / "horizon_gap_summary.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for era, col, mk in [("1929-1962", ORANGE, "s"), ("1963-1999", BLUE, "o"), ("2000-2026", GREEN, "D")]:
        h = g[g.era == era]
        ax.scatter(h.rel_log_size, h.gap, s=46, color=col, marker=mk, edgecolor=SURFACE, lw=1,
                   label=f"{era}: mean gap {s.loc[era, 'mean_gap']:+.2f}, Spearman with size {s.loc[era, 'spearman_gap_size']:+.2f}")
        c = np.polyfit(h.rel_log_size, h.gap, 1)
        xs = np.linspace(h.rel_log_size.min(), h.rel_log_size.max(), 10)
        ax.plot(xs, np.polyval(c, xs), color=col, lw=2)
    for _, r in g[g.era == "1963-1999"].nlargest(3, "gap").iterrows():
        ax.annotate(r.asset, (r.rel_log_size, r.gap), xytext=(5, 3), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("Average firm size of the industry, log relative to the cross-section")
    ax.set_ylabel("β from monthly returns − β from daily returns")
    ax.legend(fontsize=8.5, loc="upper right")
    title(fig, "Small-firm industries look less risky in daily data than they are over a month",
          "The gap that sinks daily betas at long horizons is non-synchronous trading, and it has shrunk since 2000.", y=0.99)
    save(fig, "fig3_horizon_gap")


def fig4_sml():
    q = pd.read_csv(RES / "sml_quintiles.csv")
    t = pd.read_csv(RES / "sml_tests.csv", index_col=0)
    prem = t["mkt_prem_ann"].iloc[0]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    xs = np.linspace(0.5, 1.5, 10)
    ax.plot(xs, 100 * prem * xs, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(1.42, 100 * prem * 1.42 + 0.3, "CAPM:\nslope = market premium", fontsize=8.5, color=INK2, ha="right")
    for mod, col, mk in [(TEXTBOOK, ORANGE, "s"), ("Welch-SWA", BLUE, "o"), ("Frazzini-Pedersen", GREEN, "D")]:
        h = q[q.model == mod]
        ax.plot(h.post_beta, 100 * h.mean_excess_ann, color=col, lw=2, marker=mk, ms=8, mec=SURFACE, mew=1.2,
                label=f"{LABEL[mod]}: slope = {t.loc[mod, 'slope_ratio']:.2f} × premium "
                      f"[{t.loc[mod, 'ratio_lo']:.2f}, {t.loc[mod, 'ratio_hi']:.2f}]")
    ax.set_xlabel("Post-ranking beta of the quintile portfolio (estimated on its realised returns)")
    ax.set_ylabel("Average excess return, % per year")
    ax.legend(fontsize=8.5, loc="upper left")
    title(fig, "A better beta does not rescue the CAPM: the security market line stays flat",
          "Industries sorted monthly into beta quintiles by each estimator, 1929–2026. 95% stationary-bootstrap CIs.", y=0.99)
    save(fig, "fig4_sml")


def fig5_stress():
    s = pd.read_csv(RES / "stress_windows.csv", index_col=0)
    sub = pd.read_csv(RES / "subperiods_daily.csv", index_col=0)
    tab = pd.concat([sub, s])[MODELS].drop(columns=[TEXTBOOK])
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(tab.to_numpy(), cmap=DIVERGING, norm=TwoSlopeNorm(1.0, 0.3, 1.8), aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if (v < 0.5 or v > 1.55) else INK)
    ax.set_xticks(range(tab.shape[1]), [SHORT[c] for c in tab.columns], fontsize=8.5)
    ax.set_yticks(range(tab.shape[0]), [f"{w}  ({int(n)}m)" for w, n in zip(tab.index, pd.concat([sub, s])["months"])])
    ax.axhline(len(sub) - 0.5, color=SURFACE, lw=4)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Beta-error loss vs textbook (1 = same)")
    title(fig, "The ranking holds in every era and every crash",
          "Next-month daily hedge. Blue: better than the 5-year monthly beta; red: worse.", y=0.99)
    save(fig, "fig5_stress_heatmap")


if __name__ == "__main__":
    fig1_horizon_flip()
    fig2_concentration()
    fig3_horizon_gap()
    fig4_sml()
    fig5_stress()
    print("figures written to", FIG)
