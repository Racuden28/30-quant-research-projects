"""Publication figures for project 23. Run after analyze.py and audit.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import FIG, RES  # noqa: E402

FIG.mkdir(exist_ok=True)

# ---- design tokens shared with projects 01, 03 and 19 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN, RED, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b", "#7a5bd1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
})

SHOW = [  # (method key in the CSVs, label on the chart)
    ("PSR, no deflation", "PSR, no deflation"),
    ("DSR textbook (raw N, cross-sectional V)", "DSR, textbook\n(raw N, cross-sectional V)"),
    ("DSR, raw N, null V = 1/T", "DSR, raw N, V = 1/T"),
    ("DSR, Li-Ji N, cross-sectional V", "DSR, Li-Ji N"),
    ("DSR, clusters (K, cluster V)", "DSR, clusters"),
    ("Sidak max-PSR, raw N", "Šidák max-PSR, raw N"),
    ("Sidak max-PSR, Li-Ji N", "Šidák max-PSR, Li-Ji N"),
    ("Sidak max-PSR, participation N", "Šidák max-PSR,\nparticipation-ratio N"),
]
GROUPS = [("A pooled (all indices)", "Zoo A, all indices pooled", BLUE),
          ("A single index", "Zoo A, one index (114 rules)", PURPLE),
          ("B industries", "Zoo B, 49 industries (176 rules)", ORANGE)]


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def heading(fig, t, sub, y=0.995):
    fig.text(0.01, y, t, fontsize=14, fontweight="bold", ha="left", va="top")
    fig.text(0.01, y - 0.045, sub, fontsize=10, color=INK2, ha="left", va="top")


def fig1_star():
    size = pd.read_csv(RES / "size.csv").set_index(["group", "method"])
    power = pd.read_csv(RES / "power.csv")
    power = power[power.planted_sr == 1.0].set_index(["group", "method"]).power
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 7.4), sharey=True, gridspec_kw=dict(wspace=0.12, width_ratios=[1.25, 1]))
    ys = np.arange(len(SHOW))[::-1]
    off = {"A pooled (all indices)": 0.22, "A single index": 0.0, "B industries": -0.22}
    for g, lab, col in GROUPS:
        for y, (m, _) in zip(ys, SHOW):
            s = size.loc[(g, m)]
            v = max(s["size"], 2e-4)
            a1.plot([max(s.lo, 2e-4), max(s.hi, 2e-4)], [y + off[g]] * 2, color=col, lw=1.4, alpha=0.6)
            a1.plot(v, y + off[g], "o", color=col, ms=6.5, label=lab if m == SHOW[0][0] else None)
            if g != "A single index":
                a2.plot(power.loc[(g, m)], y + off[g], "s", color=col, ms=6.5)
    a1.axvline(0.05, color=INK, lw=1.2, ls="--")
    a1.text(0.056, ys[-1] - 0.45, "nominal 5%", fontsize=9, color=INK)
    a1.set_xscale("log")
    a1.set_xlim(1.5e-4, 1.3)
    a1.set_xticks([1e-3, 1e-2, 0.05, 0.2, 1])
    a1.set_xticklabels(["0.1%", "1%", "5%", "20%", "100%"])
    a1.set_yticks(ys)
    a1.set_yticklabels([l for _, l in SHOW])
    a1.set_xlabel("False-positive rate, every rule's true Sharpe = 0 (log scale)\n300 sign-flip worlds × 99 searches; bars: 95% CI over worlds;\ndots on the left edge: 0 flags in every search")
    a1.set_title("Size: flags a strategy that is not there", loc="left")
    a2.set_xlim(-0.02, 1.02)
    a2.set_xlabel("Power: share of searches flagged when one rule with a\ntrue Sharpe of 1.0 a year is added to the null zoo\n(squares; zoo A pooled in blue, zoo B in orange)")
    a2.set_title("Power: flags one that is", loc="left")
    a2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a1.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, fontsize=9, handletextpad=0.3)
    heading(fig, "“DSR > 0.95” is not a 5% test",
            "Real index and industry data, sign-flipped so that no rule can predict. The textbook Deflated Sharpe Ratio never fired in 29,700 null searches "
            "and caught a real\nSharpe-1.0 rule 15–18% of the time. A Šidák bound on the plain PSR with the raw count N is valid (2–3%) and has twice the power.", y=1.07)
    save(fig, "fig1_dsr_is_not_a_p_value")


def fig2_synthetic():
    d = pd.read_csv(RES / "synthetic_size_power.csv")
    d = d[d.N == 1000]
    sel = [("DSR, raw N, cross-sectional V", "DSR textbook", RED, "-"),
           ("DSR, raw N, null V = 1/T", "DSR, V = 1/T", ORANGE, "-"),
           ("PSR (N = 1)", "PSR, no deflation", MUTED, ":"),
           ("Sidak max-PSR, raw N", "Šidák, raw N", PURPLE, "--"),
           ("Sidak max-PSR, Li-Ji N", "Šidák, Li-Ji N", GREEN, "-"),
           ("Sidak max-PSR, participation N", "Šidák, participation N", BLUE, "--")]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw=dict(wspace=0.22))
    panels = [("all-zero", 0.0, "Null: all 1,000 trials have Sharpe 0"),
              ("signal-rich", 0.5, "10% of trials at +0.5, 10% at −0.5"),
              ("signal-rich", 1.0, "10% of trials at +1.0, 10% at −1.0")]
    for ax, (w, mu, title) in zip(axs, panels):
        for m, lab, col, ls in sel:
            q = d[(d.world == w) & (d.mu_ann == mu) & (d.method == m)].sort_values("rho")
            yv = np.maximum(q.reject, 1e-4) if w == "all-zero" else q.reject
            ax.plot(q.rho, yv, color=col, ls=ls, lw=2, marker="o", ms=4, label=lab)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Correlation between trials (one-factor ρ)")
        if w == "all-zero":
            ax.set_yscale("log")
            ax.axhline(0.05, color=INK, ls="--", lw=1)
            ax.text(0.0, 0.058, "nominal 5%", fontsize=8.5)
            ax.set_ylim(8e-5, 1.5)
            ax.set_ylabel("Share of 20,000 simulations flagged\n(log scale; 0 plotted at 0.01%)")
        else:
            ax.set_ylim(-0.02, 1.02)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    heading(fig, "In a world we control: textbook DSR is a ~0% test that cannot see a zoo full of real strategies",
            "Gaussian trials, 10 years of daily data, N = 1,000. Annual Sharpe ratios. With mirror-image strategies in the zoo, "
            "the cross-sectional V[SR] counts the real dispersion as noise.", y=1.08)
    save(fig, "fig2_synthetic_size_power")


def fig3_neff():
    im = pd.read_csv(RES / "implied_n.csv").set_index("group")
    cols = [("mean_raw_N", "Raw count N", AXIS), ("mean_nyholt", "Nyholt", MUTED), ("mean_liji", "Li-Ji", GREEN),
            ("mean_participation", "Participation ratio", BLUE), ("mean_clusters", "Silhouette clusters", PURPLE),
            ("implied_N_sidak", "Implied: Šidák exact 5%", INK), ("implied_N_dsr_textbook", "Implied: DSR exact 5%", RED)]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(GROUPS))
    wd = 0.11
    for i, (c, lab, col) in enumerate(cols):
        v = [im.loc[g, c] for g, _, _ in GROUPS]
        b = ax.bar(x + (i - 3) * wd, v, wd, color=col, label=lab, edgecolor=SURFACE)
        for xx, vv in zip(x + (i - 3) * wd, v):
            ax.text(xx, vv * 1.08, f"{vv:,.0f}" if vv >= 10 else f"{vv:.1f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l, _ in GROUPS])
    ax.set_ylabel("Number of trials (log scale, mean over splits)")
    ax.legend(ncol=4, fontsize=8.5, loc="upper right", bbox_to_anchor=(1, 1.02))
    ax.set_ylim(0.8, 6000)
    heading(fig, "How many trials did we run? Four estimators of the effective N disagree by up to 40×",
            "Bars: effective-trial estimators on the same correlation matrices. Black and red: the N that makes each test exactly 5% "
            "in the sign-flip placebo.\nTo reach a 5% size the textbook DSR needs N understated 17–33× (red); the Šidák test needs N between Li-Ji and the raw count (black).", y=1.1)
    save(fig, "fig3_effective_trials")


def fig4_real():
    d = pd.read_csv(RES / "real_decisions.csv")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw=dict(wspace=0.2))
    for ax, grp, title in ((axs[0], ["A pooled (all indices)", "A single index"], "Zoo A winners (pooled and per index)"),
                           (axs[1], ["B industries"], "Zoo B winners (industry long-short)")):
        q = d[d.group.isin(grp)]
        fl_t = q["DSR textbook (raw N, cross-sectional V)"].astype(bool)
        fl_s = q["Sidak max-PSR, raw N"].astype(bool)
        ax.scatter(q.sr_ann[~fl_s], q.oos_ann_win[~fl_s], s=22, color=AXIS, label="Šidák max-PSR, raw N: not significant", zorder=2)
        ax.scatter(q.sr_ann[fl_s], q.oos_ann_win[fl_s], s=26, color=GREEN, label="Šidák max-PSR, raw N: significant", zorder=3)
        if fl_t.any():
            ax.scatter(q.sr_ann[fl_t], q.oos_ann_win[fl_t], s=90, facecolor="none", edgecolor=RED, lw=1.4,
                       label="DSR textbook > 0.95", zorder=4)
        ax.axhline(0, color=INK2, lw=0.8)
        lim = [min(q.sr_ann.min(), q.oos_ann_win.min()) - 0.2, max(q.sr_ann.max(), q.oos_ann_win.max()) + 0.2]
        ax.plot(lim, lim, color=MUTED, ls=":", lw=1)
        ax.text(lim[1] - 0.05, lim[1] - 0.25, "OOS = IS", color=MUTED, fontsize=8.5, ha="right")
        ax.set_xlabel("In-sample Sharpe of the winner (10 years, net, annual)")
        ax.set_ylabel("Out-of-sample Sharpe of the same rule (next 5 years, net)")
        ax.set_title(title, loc="left")
        ax.legend(fontsize=8.5, loc="upper left")
    heading(fig, "On real data the textbook DSR flagged none of 99 winners. The ones a Šidák test flags did better out of sample",
            "Each dot is one search at one split (1940-2020, every 5 years). The 28 winners the Šidák test flags earned a mean 1.10 Sharpe in the next 5 years, the other 71 earned 0.27.\n"
            "Price indices: part of zoo A's persistence before 2000 is stale-price autocorrelation in the index itself (see §4.3).", y=1.09)
    save(fig, "fig4_real_oos")


def fig5_audit():
    a = pd.read_csv(RES / "audit.csv")
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ys = np.arange(len(a))[::-1]
    marks = [("psr", "PSR (no deflation)", MUTED, "o"), ("dsr_textbook", "DSR textbook", RED, "X"),
             ("dsr_nullV", "DSR, V = 1/T", ORANGE, "D")]
    for k, lab, col, mk in marks:
        ax.scatter(a[k], ys, color=col, marker=mk, s=60, label=lab, zorder=3)
    ax.scatter(1 - a.sidak_lj_p, ys, color=GREEN, marker="s", s=70, label="Šidák max-PSR, Li-Ji N  (1 − p)", zorder=4)
    ax.axvline(0.95, color=INK, ls="--", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{p}  {c}\n{t:.0f} years, Sharpe {s:.2f}, N = {n:.0f} (Li-Ji {lj:.0f})"
                        for p, c, t, s, n, lj in zip(a.project, a.claim, a.years, a.sr_ann, a.n_raw, a.n_lj)], fontsize=8.5)
    ax.set_xlabel("Probability-scale score (pass = right of the dashed line at 0.95)")
    ax.set_xlim(-0.03, 1.03)
    ax.legend(fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    heading(fig, "Deflating our own portfolio", "Headline claims from earlier projects, with the trials each project itself reports "
            "(a lower bound on the trials actually run).", y=1.12)
    save(fig, "fig5_portfolio_audit")


def fig6_variance():
    pl = pd.read_csv(RES / "placebo_searches.csv.gz", usecols=["basis", "search", "v_xs", "T"])
    pl = pl[pl.basis == "gross"]
    re = pd.read_csv(RES / "real_searches.csv")
    re = re[re.basis == "net"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(wspace=0.18))
    for ax, s, lab in ((axs[0], "A-pooled", "Zoo A pooled"), (axs[1], "B", "Zoo B industries")):
        v0 = (pl[pl.search == s].v_xs * pl[pl.search == s]["T"])
        v1 = (re[re.search == s].v_xs * re[re.search == s]["T"])
        bins = np.linspace(0, max(v0.quantile(0.995), v1.max()) * 1.05, 40)
        ax.hist(v0, bins=bins, color=AXIS, label="Placebo worlds (true SR = 0)", density=True)
        for v in v1:
            ax.axvline(v, color=RED, lw=1.1, alpha=0.8)
        ax.plot([], [], color=RED, label="Real data, one line per split")
        ax.axvline(1, color=INK, ls="--", lw=1)
        ax.text(1.05, ax.get_ylim()[1] * 0.55, "1/T: pure noise,\nindependent trials", fontsize=8, color=INK)
        ax.set_xlabel("Cross-sectional variance of trial Sharpe ratios × T")
        ax.set_title(lab, loc="left")
        ax.legend(fontsize=8.5, loc="upper right")
    heading(fig, "Why textbook DSR loses power: V[SR] on real data is inflated by real signal",
            "The DSR plugs the cross-sectional variance of the trials' Sharpe ratios into the expected-maximum benchmark. "
            "Most real splits (red) sit\nto the right of their own placebo distribution (grey), so the benchmark rises with every real strategy in the zoo, and with its mirror image.", y=1.1)
    save(fig, "fig6_variance_contamination")


if __name__ == "__main__":
    fig2_synthetic()
    fig5_audit()
    for f in (fig1_star, fig3_neff, fig4_real, fig6_variance):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
