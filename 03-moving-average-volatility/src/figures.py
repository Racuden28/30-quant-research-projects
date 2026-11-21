"""Publication figures for project 03. Run after run_experiments.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from core import INDICES, LABEL, ROOT, TDAYS, load_closes, strategy_returns  # noqa: E402
from run_experiments import build  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens shared with project 01 (validated categorical order) ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
C = {"Buy & hold": "#52514e", "MA200": "#2a78d6", "Vol placebo": "#eb6834", "Vol target": "#1baf7a"}
MK = {"Buy & hold": "o", "MA200": "D", "Vol placebo": "s", "Vol target": "^"}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f08a89", "#a51c1c"])

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


def order_by(df, col):
    return df.sort_values(col)["index"].tolist()


def fig1_sharpe():
    p = pd.read_csv(RES / "performance.csv")
    p = p[p.returns == "net"]
    piv = p.pivot(index="index", columns="strategy", values="sharpe")
    order = piv.sort_values("MA200").index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(order))
    for i, ix in enumerate(order):
        ax.plot([piv.loc[ix].min(), piv.loc[ix].max()], [i, i], color=GRID, lw=2, zorder=1)
    for k in ["Vol placebo", "Buy & hold", "MA200"]:
        ax.scatter(piv.loc[order, k], y, s=70, color=C[k], marker=MK[k], edgecolor=SURFACE, lw=1.4, zorder=3,
                   label={"Vol placebo": "Volatility placebo (out when vol is high, same exposure)",
                          "Buy & hold": "Buy & hold", "MA200": "200-day moving average"}[k])
    ax.set_yticks(y, [LABEL[i] for i in order])
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("Sharpe ratio, net of 5 bp per trade (excess over cash, annualised)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    n_lose = int((piv["Vol placebo"] < piv["Buy & hold"]).sum())
    n_beat = int((piv["MA200"] > piv["Vol placebo"]).sum())
    title(fig, f"Stepping out when volatility is high lowers the Sharpe ratio in {n_lose} of 10 markets",
          f"Placebo has the same exposure as the 200-day rule but sees only volatility. The rule beats it in {n_beat} of 10",
          y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig1_sharpe_by_index")


def fig2_decomposition():
    d = pd.read_csv(RES / "stratified_decomposition.csv")
    order = order_by(d, "analytic_direction_part")
    d = d.set_index("index").loc[order]
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    y = np.arange(len(order)) * 1.25
    h = 0.32
    rows = [("analytic_total", "total_ci_lo", "total_ci_hi", +h, "#52514e", "Total: MA200 minus buy & hold"),
            ("analytic_direction_part", "dir_ci_lo", "dir_ci_hi", 0.0, C["MA200"], "Information beyond current volatility"),
            ("analytic_vol_timing_part", "vt_ci_lo", "vt_ci_hi", -h, C["Vol placebo"], "Volatility-timing part")]
    for col, lo, hi, off, colr, lab in rows:
        ax.barh(y + off, d[col], height=h * 0.86, color=colr, label=lab + " (bar = estimate, whisker = its 95% CI)")
        ax.errorbar(d[col], y + off, xerr=[d[col] - d[lo], d[hi] - d[col]], fmt="none", ecolor=INK,
                    elinewidth=1, capsize=2.5)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [LABEL[i] for i in order])
    ax.set_xlabel("Sharpe ratio contribution, 200-day rule vs buy & hold (gross of costs; 95% stationary block-bootstrap CIs)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=9)
    ax.grid(axis="y", visible=False)
    n_neg = int((d.analytic_vol_timing_part < 0).sum())
    n_ci = int((d.vt_ci_hi < 0).sum())
    n_info = int((d.dir_ci_lo > 0).sum())
    n_holm = int((d.p_boot_direction_holm < 0.05).sum())
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.text(0.01, 0.985, "The moving average's edge is not its correlation with volatility", fontsize=14,
             fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.945,
             f"Volatility-timing part: negative point estimate in {n_neg}/10 markets (sign test p = 0.001), its own CI excludes 0 in {n_ci}/10.\n"
             f"Information part: CI excludes 0 in {n_info}/10 unadjusted; significant after Holm across the 10 markets in {n_holm}/10 (S&P 500).",
             fontsize=10, color=INK2, ha="left", va="top")
    save(fig, "fig2_decomposition")


def fig3_mean_vs_vol():
    d = pd.read_csv(RES / "stratified_decomposition.csv")
    order = order_by(d, "analytic_mean_gap")
    d = d.set_index("index").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    y = np.arange(len(order))
    for ax, col, lo, hi, holm, lab, col_c in [
        (axes[0], "analytic_mean_gap", "mean_gap_ci_lo", "mean_gap_ci_hi", "p_boot_mean_holm",
         "Return forecast: extra mean excess return\n(percentage points per year)", C["MA200"]),
        (axes[1], "analytic_vol_gap", "vol_gap_ci_lo", "vol_gap_ci_hi", "p_boot_vol_holm",
         "Risk forecast: change in strategy volatility\n(percentage points per year)", C["MA200"])]:
        sig = d[holm] < 0.05
        ax.errorbar(d[col] * 100, y, xerr=[(d[col] - d[lo]) * 100, (d[hi] - d[col]) * 100], fmt="none",
                    ecolor=MUTED, elinewidth=1.5, capsize=3)
        ax.scatter(d[col][sig] * 100, y[sig.values], s=70, color=col_c, edgecolor=SURFACE, lw=1.4, zorder=3,
                   label="Significant after Holm (10 markets, 5%)")
        ax.scatter(d[col][~sig] * 100, y[~sig.values], s=70, facecolor=SURFACE, edgecolor=col_c, lw=1.6, zorder=3,
                   label="Not significant after Holm")
        ax.axvline(0, color=INK, lw=1)
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
        n = int(sig.sum())
        ax.set_title(f"{n} of 10 markets significant", loc="left", fontsize=10, color=INK2, fontweight="normal")
    axes[0].set_yticks(y, [LABEL[i] for i in order])
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    title(fig, "It forecasts risk everywhere, returns almost nowhere",
          "200-day rule minus its volatility-matched benchmark, 95% stationary-bootstrap intervals (63-day blocks)", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    save(fig, "fig3_return_vs_risk_forecast")


def fig4_mechanism():
    d = pd.read_csv(RES / "stratified_decomposition.csv")
    m = pd.read_csv(RES / "mechanism.csv")
    m = m[m.signal >= 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    E = np.array([[float(v) for v in s.split(";")] for s in d.exposure_by_decile])
    x = np.arange(1, 11)
    for row in E:
        ax.plot(x, 1 - row, color=GRID, lw=1.2)
    ax.plot(x, 1 - E.mean(0), color=C["MA200"], lw=2.4, marker="D", ms=7, markeredgecolor=SURFACE,
            label="Average of 10 markets")
    ax.plot([], [], color=GRID, lw=1.2, label="Each market")
    ax.set_xticks(x)
    ax.set_xlabel("Volatility state today (decile of trailing 21-day vol within the past 5 years)")
    ax.set_ylabel("Share of days the 200-day rule is out of the market")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_title("A: The rule is out mostly when volatility is high", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    ax = axes[1]
    g = m.groupby(["vol_quintile", "signal"]).mean_future_vol.mean().unstack()
    xq = np.arange(1, 6)
    ax.plot(xq, g[1] * 100, color=C["MA200"], lw=2, marker="o", ms=8, markeredgecolor=SURFACE, label="Price above its 200-day average")
    ax.plot(xq, g[0] * 100, color=C["Vol placebo"], lw=2, marker="s", ms=8, markeredgecolor=SURFACE, label="Price below its 200-day average")
    ax.set_xticks(xq, ["Q1\ncalm", "Q2", "Q3", "Q4", "Q5\nstressed"])
    ax.set_xlabel("Volatility state today (quintile)")
    ax.set_ylabel("Realised volatility over the next month (%)")
    ax.set_title("B: Same volatility today, more volatility tomorrow below trend", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    title(fig, "Why: a falling market is a market whose risk is about to rise",
          "10 indices pooled (equal weight). Panel B compares days in the same quintile of current volatility; the regression in Table 4 controls for it continuously", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig4_mechanism")


def fig5_spx_wealth():
    closes = load_closes()
    df, vol, sig, plc, _ = build("GSPC", closes)
    st = {"Buy & hold": strategy_returns(pd.Series(1.0, index=df.index), df.r, df.rf).net,
          "MA200": strategy_returns(sig["MA200"], df.r, df.rf).net,
          "Vol placebo": strategy_returns(plc["MA200"], df.r, df.rf).net}
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for k, x in st.items():
        w = (1 + x).cumprod()
        lab = {"Buy & hold": "Buy & hold", "MA200": "200-day moving average", "Vol placebo": "Volatility placebo"}[k]
        axes[0].plot(w.index, w, color=C[k], lw=1.6 if k != "MA200" else 2, label=lab)
        axes[1].plot(w.index, (w / w.cummax() - 1) * 100, color=C[k], lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Growth of $1 in excess of cash (log)")
    axes[0].axvline(pd.Timestamp("1992-12-01"), color=INK, lw=1, ls=(0, (4, 3)))
    axes[0].text(pd.Timestamp("1993-06-01"), axes[0].get_ylim()[0] * 1.08,
                 "Brock, Lakonishok & LeBaron\npublish the rule (1992)", fontsize=8.5, color=INK2)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].set_ylabel("Drawdown (%)")
    title(fig, "S&P 500 since 1933: the rule's money was made before it was published",
          "Price index excess of T-bills, net of 5 bp per trade. After 1992 the rule matches buy & hold's Sharpe with half the drawdown",
          y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig5_sp500_wealth")


def fig6_grid():
    g = pd.read_csv(RES / "grid_romano_wolf.csv")
    rules = list(dict.fromkeys(g.rule))
    order = INDICES
    M = g.pivot(index="index", columns="rule", values="diff_vs_placebo").loc[order, rules]
    P = g.pivot(index="index", columns="rule", values="p_rw_vs_placebo").loc[order, rules]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    lim = np.nanmax(np.abs(M.values))
    im = ax.imshow(M.values, cmap=DIVERGING.reversed(), norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v, p = M.values[i, j], P.values[i, j]
            star = "**" if p < 0.05 else ("*" if p < 0.10 else "")
            ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=8.5,
                    color=SURFACE if abs(v) > 0.6 * lim else INK)
    ax.set_xticks(range(len(rules)), rules)
    ax.set_yticks(range(len(order)), [LABEL[i] for i in order])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Sharpe(rule) minus Sharpe(its volatility placebo), net")
    cb.outline.set_visible(False)
    n_pos = int((M.values > 0).sum())
    title(fig, f"Every trend rule beats its volatility placebo in {n_pos} of {M.size} cases",
          "9 rules x 10 markets. ** / * : Romano-Wolf step-down p < 0.05 / 0.10 across the 9 rules within each market", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig6_rule_grid")


def fig7_rolling():
    closes = load_closes()
    df, vol, sig, plc, _ = build("GSPC", closes)
    ma = strategy_returns(sig["MA200"], df.r, df.rf).net
    bh = strategy_returns(pd.Series(1.0, index=df.index), df.r, df.rf).net.reindex(ma.index)
    pl = strategy_returns(plc["MA200"], df.r, df.rf).net.reindex(ma.index).fillna(0)
    w = 10 * TDAYS
    rs = lambda x: x.rolling(w).mean() / x.rolling(w).std() * np.sqrt(TDAYS)  # noqa: E731
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(ma.index, rs(ma) - rs(bh), color=C["MA200"], lw=2, label="200-day rule minus buy & hold")
    ax.plot(ma.index, rs(pl) - rs(bh), color=C["Vol placebo"], lw=1.6, label="Volatility placebo minus buy & hold")
    ax.axhline(0, color=INK, lw=1)
    ax.axvline(pd.Timestamp("1992-12-01"), color=INK, lw=1, ls=(0, (4, 3)))
    ax.set_ylabel("Difference in trailing 10-year Sharpe")
    ax.legend(loc="upper right", fontsize=9)
    title(fig, "S&P 500: the trend premium is episodic, and the volatility placebo is almost always below zero",
          "Trailing 10-year Sharpe ratio difference, net of costs; dashed line: publication of Brock, Lakonishok & LeBaron (1992)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig7_rolling_sharpe_sp500")


if __name__ == "__main__":
    for f in [fig1_sharpe, fig2_decomposition, fig3_mean_vs_vol, fig4_mechanism, fig5_spx_wealth, fig6_grid, fig7_rolling]:
        f()
        print("ok", f.__name__)
