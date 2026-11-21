"""Publication figures for project 05. Run after backtest.py, simulate.py and evaluate.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from config import COV_LEVELS, MEAN_LEVELS, STRATEGIES, UNIVERSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens shared with project 01 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5bd6"]
SC = dict(zip(STRATEGIES, CAT))
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#9cc3ef", "#2a78d6", "#123f78"])
ORDER = ["A", "B", "C"]
SHORT = {"A": "A · 12 US industries", "C": "C · 49 US industries", "B": "B · 19 S&P 500 stocks"}


def _span(u, cells: bool):
    g = pd.read_csv(RES / f"{u}_returns_gross.csv.gz", index_col=0, parse_dates=True)
    g = g[[c for c in g.columns if c.startswith("cell|") == cells]].dropna()
    return f"{g.index[0]:%Y}–{g.index[-1]:%Y}"


UNAME = {u: f"{SHORT[u]}, {_span(u, False)}" for u in ORDER}
HOR = {"A": "3-year oracle", "B": "3-year oracle", "C": "10-year oracle"}
UNAME_FACT = {u: f"{SHORT[u]} ({_span(u, True)}, {HOR[u]})" for u in ORDER}
CLAB = {"Sample": "Sample", "Ledoit-Wolf": "Ledoit–Wolf", "Clairvoyant": "Clairvoyant\n(future)"}
MLAB = {"None": "None (equal means)\n= min-variance", "Sample": "Sample mean", "Bayes-Stein": "Bayes–Stein",
        "Full-sample": "Full-sample mean\n(stationary oracle)", "Clairvoyant": "Clairvoyant\n(future)"}

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
    fig.text(0.01, y + 0.035 * sub.count("\n"), t, fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(0.01, y - 0.035, sub, fontsize=10, color=INK2, ha="left", va="bottom")


def csv(name, **kw):
    return pd.read_csv(RES / name, **kw)


def fig_factorial():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, u in zip(axes, ORDER):
        g = csv(f"{u}_factorial_grid.csv", keep_default_na=False).pivot(index="mean", columns="cov", values="sharpe")
        g = g.reindex(MEAN_LEVELS)[COV_LEVELS]
        base = g.loc["Sample", "Sample"]
        v = g.to_numpy()
        ax.imshow(v, cmap=SEQ, vmin=v.min() - 0.15 * np.ptp(v), vmax=v.max(), aspect="auto")
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                dark = v[i, j] > v.min() + 0.6 * np.ptp(v)
                ax.text(j, i - 0.1, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=13, fontweight="bold",
                        color=SURFACE if dark else INK)
                d = v[i, j] - base
                ax.text(j, i + 0.25, "Markowitz" if (i, j) == (1, 0) else f"{d:+.2f}", ha="center",
                        va="center", fontsize=8.5, color=SURFACE if dark else INK2)
        ax.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False, ec=INK, lw=2))
        ax.set_xticks(range(3), [CLAB[c] for c in COV_LEVELS])
        ax.set_yticks(range(5), [MLAB[m] for m in MEAN_LEVELS] if u == "A" else [""] * 5)
        ax.set_xlabel("covariance input")
        ax.grid(False)
        ax.set_title(UNAME_FACT[u], loc="left", fontsize=10.5)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("expected-return input")
    title(fig, "The frontier breaks in the means; covariances only pay once the means are right",
          "Out-of-sample Sharpe ratio (gross, annualised) of the long-only max-Sharpe portfolio fed each pair of inputs. "
          "Small numbers: change vs. classic Markowitz (boxed).\nEach panel starts after its estimation window (10 years for A and C, 3 for B) and ends when the oracle horizon runs out of data (data end 2026-09; C needs a 10-year oracle because a 49-asset covariance needs more than 49 months).", y=1.02)
    fig.tight_layout()
    save(fig, "fig1_information_factorial")


def fig_cov_value():
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharey=False)
    for ax, u in zip(axes, ORDER):
        e = csv(f"{u}_factorial_effects.csv")
        e = e[e.effect == "cov: Sample -> Clairvoyant"].copy()
        e["m"] = e.given.str.replace("mean=", "")
        e = e.set_index("m").reindex(MEAN_LEVELS)
        y = np.arange(len(e))[::-1]
        col = ["#a51c1c" if hi < 0 else ("#1baf7a" if lo > 0 else MUTED) for lo, hi in zip(e.ci_lo, e.ci_hi)]
        ax.hlines(y, e.ci_lo, e.ci_hi, color=col, lw=2.5)
        ax.scatter(e.value, y, color=col, s=70, zorder=3, edgecolor=SURFACE, lw=1.2)
        ax.axvline(0, color=INK, lw=1)
        for yy, (v, p) in zip(y, zip(e.value, e.p)):
            ax.text(e.ci_hi.max() + 0.01, yy, f"{v:+.3f}  (p={p:.2f})", va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(y, [MLAB[m].replace("\n", " ") for m in MEAN_LEVELS] if u == "A" else [""] * 5)
        ax.set_xlabel("Δ Sharpe: true future covariance instead of sample covariance")
        ax.set_title(UNAME_FACT[u], loc="left", fontsize=10.5)
        ax.set_xlim(min(e.ci_lo.min(), -0.02) - 0.02, e.ci_hi.max() + 0.14)
    title(fig, "Covariance information pays only once the means are right",
          "Value of a perfect covariance forecast (realised over the clairvoyance horizon), by quality of the expected-return input. "
          "95% paired stationary-bootstrap intervals, raw p-values. Only C with clairvoyant means clears p < 0.01 (Holm p = 0.16); the sign follows the quality of the means.", y=1.02)
    fig.tight_layout()
    save(fig, "fig2_value_of_covariance_information")


def fig_concentration():
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for ax, u in zip(axes, ORDER):
        en = csv(f"{u}_effective_n.csv.gz", index_col=0)
        n = {"A": 12, "B": 19, "C": 49}[u]
        vals = {m: [en[f"cell|{m}|{c}"].mean() for c in COV_LEVELS] for m in MEAN_LEVELS}
        x = np.arange(len(MEAN_LEVELS))
        for j, c in enumerate(COV_LEVELS):
            ax.bar(x + (j - 1) * 0.26, [vals[m][j] for m in MEAN_LEVELS], 0.24, color=CAT[j], label=CLAB[c].replace("\n", " "))
        ax.set_xticks(x, [{"None":"None\n(min-var)","Sample":"Sample","Bayes-Stein":"Bayes–Stein","Full-sample":"Full-\nsample","Clairvoyant":"Clair-\nvoyant"}[m] for m in MEAN_LEVELS], fontsize=8.5)
        ax.set_ylabel(f"effective number of holdings (of {n})")
        ax.set_title(UNAME_FACT[u], loc="left", fontsize=10.5)
    axes[0].legend(title="covariance input", fontsize=8.5, title_fontsize=8.5, ncol=3, loc="lower left",
                   bbox_to_anchor=(0, 1.08))
    title(fig, "Mechanism: with noisy means, a sharper covariance concentrates the portfolio further",
          "Average effective number of holdings, 1/Σw², of the max-Sharpe portfolio in each cell.", y=1.08)
    fig.tight_layout()
    save(fig, "fig3_concentration_mechanism")


def fig_promise():
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    s = csv("summary.csv", index_col=0)
    for ax, u in zip(axes, ORDER):
        sim = csv(f"{u}_sim_paths.csv")
        gap = sim["promised"] - sim["Max-Sharpe (sample)"]
        real = float(s.loc["optimism_gap_real", u])
        ax.hist(gap, bins=30, color="#9cc3ef", edgecolor=SURFACE)
        ax.axvline(real, color="#a51c1c", lw=2.5)
        ax.text(real, ax.get_ylim()[1] * 0.92, f"  real data: {real:.2f}\n  (sim. p = {float(s.loc['optimism_p_gap', u]):.3f})",
                color="#a51c1c", va="top", fontsize=9)
        ax.text(gap.min(), ax.get_ylim()[1] * 0.92, f"stationary worlds\nmean gap {gap.mean():.2f}", ha="left",
                va="top", fontsize=9, color=INK2)
        ax.set_xlabel("promised minus delivered Sharpe of the Markowitz max-Sharpe portfolio")
        ax.set_ylabel("simulated paths")
        ax.set_title(UNAME[u], loc="left", fontsize=10.5)
        ax.set_xlim(min(gap.min(), real) - 0.1, max(gap.max(), real) + 0.35)
    title(fig, "How much of the broken promise is estimation error?",
          "Real-data gap between in-sample and out-of-sample Sharpe vs. i.i.d. Gaussian worlds (200 for A and B, 100 for C) with the same "
          "moments, windows and code.", y=1.02)
    fig.tight_layout()
    save(fig, "fig4_promise_vs_reality")


def fig_years():
    d = csv("years_needed.csv")
    s = csv("summary.csv", index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    styles = {"Sample mean, sample cov": (CAT[0], "-"), "Bayes-Stein mean, sample cov": (CAT[1], "-"),
              "No mean (min-var), sample cov": (CAT[2], "-"), "True mean, sample cov": (MUTED, ":")}
    for ax, u in zip(axes, ORDER):
        g = d[d.universe == u]
        for k, (c, ls) in styles.items():
            h = g[g.estimator == k]
            ax.plot(h.years, h.mean_sharpe, color=c, ls=ls, lw=2, marker="o" if ls == "-" else None, ms=4, label=k)
            if k == "Sample mean, sample cov":
                ax.fill_between(h.years, h.p10, h.p90, color=c, alpha=0.12, lw=0)
        ax.axhline(g.one_over_n.iloc[0], color=INK, lw=1, ls=(0, (4, 3)))
        ax.text(1300, g.one_over_n.iloc[0], "1/N", va="bottom", ha="right", fontsize=9)
        ax.axhline(g.best.iloc[0], color=INK, lw=0.8)
        ax.text(1300, g.best.iloc[0], "true optimum", va="bottom", ha="right", fontsize=9)
        win = UNIVERSES[u]["window"] / UNIVERSES[u]["ppy"]
        ax.axvline(win, color="#a51c1c", lw=1)
        ax.text(win * 1.08, ax.get_ylim()[0] + 0.01, f"window used\n({win:g} yrs)", color="#a51c1c", fontsize=8.5, va="bottom")
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30, 100, 300, 1000], ["1", "3", "10", "30", "100", "300", "1000"])
        ax.set_xlabel("years of i.i.d. data used to estimate the inputs")
        ax.set_ylabel("expected true Sharpe (annualised)")
        ax.set_title(SHORT[u] + " · true moments = full-sample", loc="left", fontsize=10.5)
        if u == "A":
            ax.legend(fontsize=8.5, loc="lower right", bbox_to_anchor=(1, 0.08))
    title(fig, "Even in a stationary world, estimated means need years to decades of data to pay off",
          "Exact Gaussian sampling distribution of the inputs, 2,000 draws per point; band = 10th–90th percentile "
          "for the sample estimator.", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_years_of_data_needed")


def fig_horse_race():
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for ax, u in zip(axes, ORDER):
        h = csv(f"{u}_horse_race.csv", index_col=0).reindex(STRATEGIES)
        y = np.arange(len(h))[::-1]
        ax.hlines(y, h.sharpe_ci_lo, h.sharpe_ci_hi, color=[SC[s] for s in h.index], lw=2.5, alpha=0.6)
        ax.scatter(h.sharpe_net, y, color=[SC[s] for s in h.index], s=70, zorder=3, edgecolor=SURFACE, lw=1.2)
        ax.axvline(h.loc["1/N", "sharpe_net"], color=INK, lw=1, ls=(0, (4, 3)))
        for yy, s in zip(y, h.index):
            if s != "1/N":
                ax.text(h.sharpe_ci_hi.max() + 0.03, yy,
                        f"{h.loc[s, 'diff_vs_1N']:+.2f} vs 1/N, Holm p={h.loc[s, 'p_holm']:.2f}", va="center",
                        fontsize=8.3, color=INK2)
        ax.set_yticks(y, h.index if u == "A" else [""] * len(h))
        ax.set_xlabel("out-of-sample Sharpe, net of 10 bp costs (95% bootstrap CI)")
        ax.set_title(UNAME[u], loc="left", fontsize=10.5)
        ax.set_xlim(h.sharpe_ci_lo.min() - 0.05, h.sharpe_ci_hi.max() + 0.75)
    title(fig, "No investable estimator beats 1/N with statistical confidence",
          "Walk-forward, long-only, monthly rebalancing. Differences vs. 1/N tested with a paired stationary bootstrap, "
          "Holm-adjusted.", y=1.02)
    fig.tight_layout()
    save(fig, "fig6_horse_race")


def fig_weights():
    w = csv("A_weights.csv.gz", parse_dates=["date"])
    cols = [c for c in w.columns if c not in ("date", "strategy")]
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5bd6", "#9cc3ef", "#a51c1c",
           "#52514e", "#c3c2b7", "#123f78"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, s in zip(axes, ["Max-Sharpe (sample)", "Min-var (sample)"]):
        g = w[w.strategy == s].set_index("date")[cols]
        ax.stackplot(g.index, g.T.values, colors=pal, labels=cols, lw=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel("weight")
        ax.set_title(s + (" · inputs: sample means + sample covariance" if "Sharpe" in s else " · no means used"),
                     loc="left", fontsize=10.5)
        ax.grid(False)
    axes[0].legend(ncol=12, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.12), handlelength=1)
    title(fig, "What estimated means do to a portfolio", "Universe A, 12 industries, 10-year rolling window, "
          "monthly rebalancing, 1959–2017.", y=1.06)
    fig.tight_layout()
    save(fig, "fig7_weights_over_time")


def fig_nt():
    d = csv("nt_sweep.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    panels = [("phi_cov", "Shapley value of a perfect covariance\n(max-Sharpe, Δ Sharpe)"),
              ("lw_given_no_mean", "Ledoit–Wolf vs sample covariance\n(minimum variance, Δ Sharpe)"),
              ("lw_given_sample_mean", "Ledoit–Wolf vs sample covariance\n(max-Sharpe with sample means, Δ Sharpe)")]
    for ax, (col, lab) in zip(axes, panels):
        for k, u in enumerate(ORDER):
            g = d[d.universe == u].sort_values("n_over_t")
            ax.fill_between(g.n_over_t, g[col + "_lo"], g[col + "_hi"], color=CAT[k], alpha=0.12, lw=0)
            ax.plot(g.n_over_t, g[col], color=CAT[k], lw=2, marker="o", ms=5, label=SHORT[u])
        ax.axhline(0, color=INK, lw=1)
        ax.set_xlabel("N / T  (assets ÷ observations in the estimation window)")
        ax.set_title(lab, loc="left", fontsize=10.5)
    axes[0].legend(fontsize=8.5, loc="upper left")
    title(fig, "Where the covariance starts to matter: the N/T sweep",
          "Same walk-forward code, only the estimation window changes. Bands: 95% paired stationary-bootstrap intervals.",
          y=1.03)
    fig.tight_layout()
    save(fig, "fig8_nt_sweep")


if __name__ == "__main__":
    fig_factorial()
    fig_cov_value()
    fig_concentration()
    fig_years()
    fig_horse_race()
    fig_weights()
    if (RES / "summary.csv").exists():
        fig_promise()
    if (RES / "nt_sweep.csv").exists():
        fig_nt()
