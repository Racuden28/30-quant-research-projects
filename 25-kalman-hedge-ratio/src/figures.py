"""Publication figures for project 25. Run after hedging.py, pairs.py, mechanism.py and evaluate.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import FIG, RES  # noqa: E402

# ---- design tokens shared with projects 01-24 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
KAL, EWM, CHOICE, OTHER = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
REAL, PLACEBO = "#2a78d6", "#eb6834"
U = {"industries": "49 US industries vs the market, 1929–2026", "countries": "21 country ETFs vs SPY, 2006–2026"}

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
    fig.text(0.01, y + 0.05, t, fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(0.01, y + 0.005, sub, fontsize=10, color=INK2, ha="left", va="bottom")


def fig1_star():
    """Tuning curves: hedge loss against memory, for the Kalman filter and the EWMA with the same memory."""
    cur = pd.read_csv(RES / "tuning_curves.csv")
    rh = {u: pd.read_csv(RES / f"hedge_rhat_{u}.csv") for u in U}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), gridspec_kw={"wspace": 0.16})
    for ax, u in zip(axes, U):
        c = cur[(cur.universe == u) & np.isfinite(cur.halflife_days) & (cur.halflife_days < 6000)
                & (cur.halflife_days > 2.5)].sort_values("halflife_days")
        best = min(c.loss_kalman.min(), c.loss_ewma.min())
        ex_k = 100 * (c.loss_kalman / best - 1)
        ex_e = 100 * (c.loss_ewma / best - 1)
        ax.plot(c.halflife_days, ex_k, color=KAL, lw=2.2, label="Kalman filter, fixed noise ratio")
        ax.plot(c.halflife_days, ex_e, color=EWM, lw=2.0, ls=(0, (4, 2)),
                label="EWMA regression with the same memory")
        ml = rh[u].halflife_days.replace(np.inf, np.nan).median()
        ss = rh[u].halflife_sse.replace(np.inf, np.nan).median()

        def at(h):
            return float(np.interp(np.log(h), np.log(c.halflife_days), ex_k))
        pos = {"industries": {"ml": (16, 5.6), "ss": (22, 3.2), "126": (130, 5.2)},
               "countries": {"ml": (14, 3.4), "ss": (40, 6.6), "126": (140, 9.6)}}[u]
        for h, key, txt, col in ((ml, "ml", f"Chosen by likelihood\n(median {ml:.0f} days)", KAL),
                                 (ss, "ss", f"Chosen by past hedging loss\n(median {ss:.0f} days)", CHOICE),
                                 (126, "126", "Project 09's EWMA\n(126 days)", OTHER)):
            ax.plot([h], [at(h)], "o", ms=9, color=col, mec=SURFACE, mew=2, zorder=5)
            ax.annotate(txt, (h, at(h)), xytext=pos[key], textcoords="data", ha="left", va="bottom",
                        fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.set_xscale("log")
        ax.set_xlim(2.5, 6000)
        ax.set_ylim(-0.3, 9.5 if u == "industries" else 12)
        ax.set_xticks([3, 10, 30, 100, 300, 1000, 3000], ["3", "10", "30", "100", "300", "1,000", "3,000"])
        ax.set_xlabel("Memory of the hedge ratio: steady-state half-life (trading days, log scale)")
        ax.set_title(U[u], loc="left", fontsize=10.5)
        if u == "industries":
            ax.set_ylabel("Out-of-sample hedge loss above the best memory (%)")
            ax.legend(loc="upper left", fontsize=9)
    title(fig, "A Kalman hedge ratio is an EWMA, and its likelihood picks the wrong memory",
          "Next-day hedge error, daily-updated betas. The two curves overlap at every memory. The likelihood "
          "chooses a memory 2.5–6× too short; choosing it by past hedging loss lands near the minimum.")
    save(fig, "fig1_kalman_is_an_ewma")


def fig2_p09():
    p = pd.read_csv(RES / "p09_comparison.csv").set_index("estimator")
    rows = [("p09|OLS-5Y-M", "Textbook: 5-year monthly OLS (project 09)", OTHER),
            ("p09|Welch-SWA", "Welch age-weighted (project 09's best)", OTHER),
            ("p09|EWMA", "EWMA 126 days (project 09)", OTHER),
            ("mine_M|KF", "Kalman, likelihood memory, month-end", KAL),
            ("mine_D|KF", "Kalman, likelihood memory, updated daily", KAL),
            ("mine_D|EWMA-126", "EWMA 126 days, updated daily", EWM),
            ("mine_D|KF-SSE", "Kalman, memory by past loss, updated daily", CHOICE)]
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    for i, (k, lab, col) in enumerate(rows[::-1]):
        r = p.loc[k]
        ax.barh(i, r.excess_loss_ratio, 0.6, color=col)
        ax.errorbar(r.excess_loss_ratio, i, xerr=[[r.excess_loss_ratio - r.ci_lo], [r.ci_hi - r.excess_loss_ratio]],
                    fmt="none", ecolor=INK, lw=1, capsize=3)
        ax.text(r.ci_hi + 0.015, i, f"{r.excess_loss_ratio:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows[::-1]])
    ax.set_xlim(0, 1.15)
    ax.axvline(1, color=AXIS, lw=1)
    ax.set_xlabel("Share of the textbook beta's avoidable hedge error that remains (project 09's metric; lower is better)")
    title(fig, "On project 09's scoreboard, the likelihood-tuned Kalman beta loses to a plain EWMA",
          "49 industries, 54,245 industry-months, Aug 1929 – Sep 2026, bars with 95% block-bootstrap CIs. "
          "Daily updating helps every estimator.", y=0.98)
    save(fig, "fig2_project09_scoreboard")


def fig3_mechanism():
    m = pd.read_csv(RES / "mechanism.csv").set_index("world")
    rh = pd.read_csv(RES / "hedge_rhat_industries.csv")
    rh["dec"] = pd.to_datetime(rh.date).dt.year // 10 * 10
    g = rh.replace(np.inf, np.nan).groupby("dec")[["halflife_days", "halflife_sse"]].median()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5), gridspec_kw={"wspace": 0.25, "width_ratios": [1, 1.1]})
    ax = axes[0]
    worlds = [("C", "True random-walk beta,\nconstant volatility"), ("I", "Same beta, volatility that\nclusters, fat tails")]
    for i, (w, lab) in enumerate(worlds):
        r = m.loc[w]
        for j, (col, val, name) in enumerate(((OTHER, 63.0, "true half-life"), (KAL, r.hl_ml_median, "likelihood"),
                                              (CHOICE, r.hl_sse_median, "past hedging loss"),
                                              (EWM, r.hl_best_oos_median, "best out of sample"))):
            ax.bar(i * 5 + j, val, 0.85, color=col, label=name if i == 0 else None)
            ax.text(i * 5 + j, val + 2, f"{val:.0f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks([1.5, 6.5], [w[1] for w in worlds])
    ax.set_ylabel("Median half-life chosen (days, 200 simulations)")
    ax.legend(fontsize=8.5, loc="upper left", ncol=2)
    ax.set_ylim(0, 115)
    ax.set_title("Simulation: clustered volatility shortens\nthe likelihood's memory", loc="left", fontsize=10.5)
    ax = axes[1]
    ax.plot(g.index, g.halflife_days, "o-", color=KAL, lw=2, ms=7, label="chosen by likelihood")
    ax.plot(g.index, g.halflife_sse, "o-", color=CHOICE, lw=2, ms=7, label="chosen by past hedging loss")
    ax.set_xticks(g.index, [f"{d}s" for d in g.index], rotation=0, fontsize=8.5)
    ax.set_ylabel("Median half-life across 49 industries (days)")
    ax.set_ylim(0, 75)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Real industries: the likelihood picks 6–13 days\nin every decade since 1930", loc="left", fontsize=10.5)
    title(fig, "Why the likelihood is fooled",
          "The random-walk model explains bigger errors on volatile days as beta uncertainty, so it buys "
          "heteroskedasticity with a shorter memory.")
    save(fig, "fig3_why_the_likelihood_is_fooled")


def fig4_stress():
    s = pd.read_csv(RES / "hedge_stress_windows.csv")
    c = pd.read_csv(RES / "hedge_buckets.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), gridspec_kw={"wspace": 0.35, "width_ratios": [1.6, 1]})
    ax = axes[0]
    labs = [f"{w} ({'ind.' if u == 'industries' else 'ETFs'})" for u, w in zip(s.universe, s.window)]
    x = np.arange(len(s))
    for j, (k, col, lab) in enumerate((("KF", KAL, "Kalman, likelihood memory"),
                                      ("EW-match", EWM, "EWMA, same memory"),
                                      ("KF-SSE", CHOICE, "Kalman, memory by past loss"))):
        ax.bar(x + (j - 1) * 0.27, s[f"{k}_vs_EWMA126_pct"], 0.25, color=col, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, labs, fontsize=8, rotation=35, ha="right")
    ax.set_ylabel("Hedge loss vs EWMA 126 days (%)")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("Stress windows", loc="left", fontsize=10.5)
    ax = axes[1]
    for i, u in enumerate(U):
        r = c[c.universe == u].set_index("bucket")
        for j, (b, col) in enumerate((("other 90% of months", OTHER), ("most volatile 10% of months", INK2))):
            v = r.loc[b, "kf_vs_ewmatch_pct"]
            ax.bar(i * 3 + j, v, 0.85, color=col, label=b if i == 0 else None)
            ax.text(i * 3 + j, v + (0.05 if v >= 0 else -0.05), f"{v:+.2f}%", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8.5)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks([0.5, 3.5], ["Industries", "Country ETFs"])
    ax.set_ylabel("Kalman vs EWMA with the same memory (%)")
    ax.set_ylim(-1.6, 2.6)
    ax.legend(fontsize=8.5, loc="upper center", ncol=1)
    ax.set_title("Kalman wins calm months, loses crashes", loc="left", fontsize=10.5)
    title(fig, "Where the Kalman filter differs from its EWMA twin: it overreacts on crash days",
          "Its gain grows with the market move, so one crash day can rewrite the hedge ratio. Net over the "
          "whole sample: zero.")
    save(fig, "fig4_stress_and_crash_days")


def fig5_pairs_mechanics():
    p = pd.read_csv(RES / "pairs_summary.csv")
    p = p[(p.selection == "dist") & (p.cost_bp == 0)]
    rules = [("S", "Static"), ("R", "Rolling\n63d"), ("K", "Kalman")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), gridspec_kw={"wspace": 0.32})
    ur = pd.read_csv(RES / "pairs_unit_root.csv")
    for ax, metric, ylab, ttl in ((axes[0], "convergence_share", "Trades that converge (%)",
                                    "Convergence: real ≈ placebo"),
                                   (axes[1], "price_share_of_closure", "Share of the spread's closing\ndue to prices (%)",
                                    "Kalman spreads close by redefinition")):
        for i, u in enumerate(("countries", "industries")):
            q = p[p.universe == u].set_index("rule")
            for j, (r, lab) in enumerate(rules):
                xx = i * 4 + j
                ax.bar(xx - 0.2, 100 * q.loc[r, f"real_{metric}"], 0.38, color=REAL,
                       label="real market" if (i, j) == (0, 0) else None)
                ax.bar(xx + 0.2, 100 * q.loc[r, f"placebo_{metric}"], 0.38, color=PLACEBO,
                       label="placebo market" if (i, j) == (0, 0) else None)
        ax.set_xticks([0, 1, 2, 4, 5, 6], [r[1] for r in rules] * 2, fontsize=8)
        ax.text(1, -0.22, "Country ETFs", transform=ax.get_xaxis_transform(), ha="center", fontsize=8.5, color=INK2)
        ax.text(5, -0.22, "Industries", transform=ax.get_xaxis_transform(), ha="center", fontsize=8.5, color=INK2)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 108)
        ax.set_title(ttl, loc="left", fontsize=10.5)
    axes[0].legend(fontsize=8.5, loc="upper left")
    ax = axes[2]
    for i, u in enumerate(("countries", "industries")):
        q = ur[ur.universe == u].set_index("source")
        for j, (col, lab) in enumerate((("eg_residual_sig_share", "Static residual"),
                                        ("kalman_innovation_sig_share", "Kalman innovation"))):
            xx = i * 3 + j
            ax.bar(xx - 0.2, 100 * q.loc["real", col], 0.38, color=REAL)
            ax.bar(xx + 0.2, 100 * q.loc["N1", col], 0.38, color=PLACEBO)
    ax.set_xticks([0, 1, 3, 4], ["Static\nresidual", "Kalman\ninnovation"] * 2, fontsize=8)
    ax.text(0.5, -0.25, "Country ETFs", transform=ax.get_xaxis_transform(), ha="center", fontsize=8.5, color=INK2)
    ax.text(3.5, -0.25, "Industries", transform=ax.get_xaxis_transform(), ha="center", fontsize=8.5, color=INK2)
    ax.set_ylabel("Selected pairs whose formation spread\nrejects a unit root at 5% (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Every Kalman spread looks stationary", loc="left", fontsize=10.5)
    title(fig, "Three kinds of pairs-trading evidence the Kalman filter produces on pure noise",
          "Same 20 distance-selected pairs per window, project 11's design and placebo market (10 replications "
          "per window).", y=1.02)
    save(fig, "fig5_pairs_mechanical_evidence")


def fig6_pairs_returns():
    p = pd.read_csv(RES / "pairs_summary.csv")
    p = p[p.selection == "dist"]
    rules = [("S", "Static"), ("R", "Rolling 63d"), ("K", "Kalman"), ("K-wait5", "Kalman,\n5-day wait"),
             ("REV1", "1-day\nreversal")]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4), gridspec_kw={"wspace": 0.18})
    for ax, u in zip(axes, ("countries", "industries")):
        for j, (r, lab) in enumerate(rules):
            g = p[(p.universe == u) & (p.rule == r) & (p.cost_bp == 0)].iloc[0]
            n = p[(p.universe == u) & (p.rule == r) & (p.cost_bp == 5)].iloc[0]
            ax.bar(j - 0.27, g.real_pct_yr, 0.25, color=REAL, label="real, gross" if j == 0 else None)
            ax.bar(j, n.real_pct_yr, 0.25, color="#9fc3ee", label="real, 5 bp per leg" if j == 0 else None)
            ax.bar(j + 0.27, g.placebo_pct_yr, 0.25, color=PLACEBO, label="placebo, gross" if j == 0 else None)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(rules)), [r[1] for r in rules], fontsize=8.5)
        ax.set_title({"countries": "22 country ETFs, 2002–2026", "industries": "49 industries, 1927–2026"}[u],
                     loc="left", fontsize=10.5)
        if u == "countries":
            ax.set_ylabel("Return on committed capital (% a year)")
            ax.legend(fontsize=8.5, loc="lower left")
    title(fig, "The Kalman pairs trade is a short-term reversal trade, and costs eat it",
          "Same pairs for every rule. The Kalman rule holds a position ~2 days, tracks the 1-day reversal "
          "(monthly correlation 0.71), dies with a 5-day wait and loses money at 5 bp per leg.")
    save(fig, "fig6_pairs_returns")


def fig7_paths():
    b = pd.read_csv(RES / "hedge_betas_countries.csv.gz", index_col=0, parse_dates=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2), gridspec_kw={"wspace": 0.15})
    for ax, (a, w) in zip(axes, (("EWJ", ("2008-06-01", "2009-06-30")), ("EWZ", ("2020-01-01", "2020-12-31")))):
        s = b.loc[w[0]:w[1]]
        ax.plot(s.index, s[f"{a}|KF"], color=KAL, lw=1.6, label="Kalman, likelihood memory")
        ax.plot(s.index, s[f"{a}|EW-match"], color=EWM, lw=1.4, ls=(0, (4, 2)), label="EWMA, same memory")
        ax.plot(s.index, s[f"{a}|EWMA-126"], color=INK2, lw=1.4, label="EWMA, 126 days")
        ax.set_title({"EWJ": "Japan ETF on SPY, 2008–09", "EWZ": "Brazil ETF on SPY, 2020"}[a], loc="left",
                     fontsize=10.5)
        ax.set_ylabel("Hedge ratio (beta on SPY)")
        if a == "EWJ":
            ax.legend(fontsize=8.5, loc="upper left")
    title(fig, "What a two-week memory looks like",
          "Daily predicted hedge ratios. The Kalman beta and its EWMA twin mostly move together but part ways "
          "after crash days; both chase a few weeks of noise.")
    save(fig, "fig7_beta_paths")


if __name__ == "__main__":
    for f in (fig1_star, fig2_p09, fig3_mechanism, fig4_stress, fig5_pairs_mechanics, fig6_pairs_returns, fig7_paths):
        f()
        print(f.__name__)
