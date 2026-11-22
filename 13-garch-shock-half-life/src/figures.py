"""Publication figures for project 13. Run after backtest, irf, stability, episodes, evaluate."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import INDICES, LABEL, ROOT  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens shared with project 01 (palette validated: CVD and normal-vision checks pass;
# contrast WARN on green/amber/pink is relieved by markers and direct labels)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
BAND = "#d9d8d2"
C = {"GARCH": "#2a78d6", "GJR-t": "#eb6834", "CGARCH": "#1baf7a", "HAR": "#eda100", "EWMA": "#e87ba4",
     "GARCH-R10": "#008300", "Empirical": INK}
MK = {"GARCH": "o", "GJR-t": "s", "CGARCH": "D", "HAR": "^", "EWMA": "v", "GARCH-R10": "P"}
NAME = {"GARCH": "GARCH(1,1)", "GJR-t": "GJR-GARCH-t", "CGARCH": "Component GARCH", "HAR": "HAR (direct)",
        "EWMA": "EWMA 0.94", "GARCH-R10": "GARCH(1,1), 10y rolling"}

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
    fig.text(0.01, y + 0.045, t, fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(0.01, y + 0.005, sub, fontsize=10, color=INK2, ha="left", va="bottom")


# --------------------------------------------------------------------------- fig 1 (star)
def fig1_decay(nc, nt):
    cv = nc[nc["index"] == "POOLED"]
    t = nt[nt["index"] == "POOLED"].set_index("world")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for w, lab in [("GARCH", "GARCH(1,1)"), ("CGARCH", "component GARCH")]:
        g = cv[cv.world == w]
        ax.fill_between(g.k, g.null_lo, g.null_hi, color=C[w], alpha=0.16, lw=0)
        ax.plot(g.k, g.null_med, color=C[w], lw=2, ls=(0, (5, 2)),
                label=f"If the world were {lab}: same measurement, 100 simulations (median, 90% band)")
        ax.plot(g.k[::42], g.null_med.values[::42], ls="none", marker=MK[w], ms=7, color=C[w],
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    d = cv[cv.world == "GARCH"]
    ax.plot(d.k, d.data, color=INK, lw=2.8, label="Real data, 10 equity indices pooled")
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.axhline(0, color=AXIS, lw=1)
    he, hg = t.loc["GARCH", "data_half_life"], t.loc["GARCH", "null_half_life_median"]
    ax.plot([he], [0.5], "o", ms=9, color=INK, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
    ax.plot([hg], [0.5], "o", ms=9, color=C["GARCH"], markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
    ax.annotate(f"Real data: half gone after {he:.0f} trading days", (he, 0.5), xytext=(he + 8, 0.80), fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.annotate(f"GARCH world: {hg:.0f} days (90%: {t.loc['GARCH', 'null_hl_p05']:.0f}–{t.loc['GARCH', 'null_hl_p95']:.0f}),"
                f" p = {t.loc['GARCH', 'p_hl']:.2f}", (hg, 0.5), xytext=(hg + 10, 0.60), fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.annotate(f"After a year {t.loc['GARCH', 'data_share_at_252']:.0%} of the shock is still there.\n"
                f"GARCH worlds: {max(t.loc['GARCH', 'null_share_at_252'], 0):.1%} (p = {t.loc['GARCH', 'p_tail_high']:.2f})",
                (252, t.loc["GARCH", "data_share_at_252"]), xytext=(150, 0.30), fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlim(1, 252)
    ax.set_ylim(-0.08, 1.05)
    ax.set_xticks([1, 22, 63, 126, 189, 252])
    ax.set_xticklabels(["1 day", "1 month", "3 months", "6 months", "9 months", "1 year"])
    ax.set_ylabel("Share of today's variance news still visible k days later")
    ax.set_xlabel("Days after the shock (k)")
    ax.legend(loc="upper right", fontsize=8.8, bbox_to_anchor=(1.0, 1.02))
    title(fig, "Volatility shocks fade faster, and linger longer, than GARCH allows",
          "Local projection of r²(t+k) on the model's variance for tomorrow. Each band applies the identical "
          "measurement to simulated data in which the model is true.", y=0.9)
    save(fig, "fig1_shock_decay")


# --------------------------------------------------------------------------- fig 2 per index
def fig2_per_index(nt):
    t = nt[(nt.world == "GARCH") & (nt["index"] != "POOLED")].set_index("index")
    fp = pd.read_csv(RES / "full_sample_params.csv").set_index("index")
    pop = np.log(0.5) / np.log(fp.G_alpha + fp.G_beta)
    order = sorted(t.index, key=lambda c: t.loc[c, "data_half_life"])
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    y = np.arange(len(order))
    for i, c in enumerate(order):
        ax.hlines(i, t.loc[c, "null_hl_p05"], t.loc[c, "null_hl_p95"], color=C["GARCH"], lw=5, alpha=0.3)
    ax.plot(t.loc[order, "null_half_life_median"], y, "|", color=C["GARCH"], ms=16, mew=2,
            label="Measured in 100 simulated GARCH worlds (median, 90% range)")
    ax.plot(pop.loc[order], y, "o", mfc="none", mec=C["GARCH"], mew=1.6, ms=8,
            label="GARCH's own half-life, ln 0.5 / ln(α+β)")
    ax.plot(t.loc[order, "data_half_life"], y, "o", color=INK, ms=8, markeredgecolor=SURFACE,
            label="Measured in the real data")
    for i, c in enumerate(order):
        pv = t.loc[c, "p_hl"]
        ax.text(300, i, f"p = {pv:.2f}", va="center", fontsize=8.5, color=INK if pv < 0.05 else MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[c] for c in order])
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 22, 63, 126, 252])
    ax.set_xticklabels(["5", "10", "22", "63", "126", "252"])
    ax.set_xlim(5, 420)
    ax.set_xlabel("Half-life of a variance shock, trading days (log scale)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.36), fontsize=8.8, ncol=1)
    title(fig, "Half the naive gap is measurement bias. The rest is real in 5 of 10 markets",
          "Comparing the data with GARCH's own half-life (hollow) overstates the gap. "
          "The fair yardstick is the blue range. p = share of GARCH worlds at least as fast.", y=0.9)
    save(fig, "fig2_half_life_by_index")


# --------------------------------------------------------------------------- fig 3 stability
def fig3_decades(dec, tests):
    order = tests.sort_values("n_decades", ascending=False)["index"].tolist()
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), sharey=True)
    for ax, c in zip(axes.flat, order):
        g = dec[dec["index"] == c]
        t = tests[tests["index"] == c].iloc[0]
        x = g.decade.values
        ax.fill_between(x, np.minimum(g.null_hl_lo, 1500), np.minimum(g.null_hl_hi, 1500), color=BAND, lw=0,
                        step="mid", label="90% range if parameters were constant")
        ax.plot(x, np.minimum(g.half_life, 2000), "o-", color=C["GARCH"], lw=1.6, ms=6, markeredgecolor=SURFACE,
                label="GARCH(1,1) fitted on that decade")
        ax.axhline(t.full_half_life, color=INK, lw=1.2, ls=(0, (4, 3)), label="Fitted on the whole sample")
        ax.set_yscale("log")
        ax.set_ylim(4, 1000)
        ax.set_title(f"{LABEL[c]}  (LR p={t.lr_p:.3f})", fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d % 100:02d}s" for d in x], fontsize=8)
    axes[0, 0].set_ylabel("Half-life, trading days")
    axes[1, 0].set_ylabel("Half-life, trading days")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=9.5)
    title(fig, "Persistence is borrowed from the decades, not from the shocks",
          "GARCH(1,1) refitted on each calendar decade. Grey: 90% range of decade estimates if the whole-sample "
          "model were true (200 simulations; tops above 1,000 days are cut).", y=0.93)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    save(fig, "fig3_decade_stability")


# --------------------------------------------------------------------------- fig 4 events
def fig4_events(err, ev):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
    hs = sorted(err.h.unique())
    xs = np.arange(len(hs))
    models = ["EWMA", "GARCH", "GJR-t", "CGARCH", "HAR"]
    offs = np.linspace(-0.2, 0.2, len(models))
    for j, m in enumerate(models):
        g = err[err.model == m].set_index("h").loc[hs]
        a1.vlines(xs + offs[j], g.rs_lo, g.rs_hi, color=C[m], lw=1.2)
        a1.plot(xs + offs[j], g.ratio_of_sums, MK[m], color=C[m], ms=8, markeredgecolor=SURFACE,
                label=NAME[m])
    a1.axhline(1, color=INK, lw=1)
    a1.set_xticks(xs)
    a1.set_xticklabels([f"{h} days" for h in hs])
    a1.set_ylabel("Realized ÷ forecast variance (sum over events)")
    a1.set_yscale("log")
    a1.set_yticks([0.5, 0.7, 1, 1.5, 2, 3])
    a1.set_yticklabels(["0.5", "0.7", "1", "1.5", "2", "3"])
    a1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_title("Forecast made on the day the shock is detected", loc="left")
    a1.legend(fontsize=8.5, ncol=2, loc="lower left")
    a1.set_ylim(0.45, 3.2)
    a1.text(0.99, 0.97, "1 = right on average · below 1 = forecast too high · above 1 = too low", transform=a1.transAxes,
            ha="right", va="top", fontsize=8.5, color=INK2)
    v = ev.realized_halfway.dropna().values
    bins = [0, 5, 10, 15, 22, 33, 44, 63, 126, 260]
    cnt, _ = np.histogram(v, bins)
    a2.bar(np.arange(len(cnt)), cnt, color=INK2, width=0.8)
    a2.set_xticks(np.arange(len(cnt)))
    a2.set_xticklabels([f"{bins[i]}–{bins[i + 1]}" for i in range(len(cnt))], fontsize=8, rotation=0)
    a2.set_xlabel("Days until the excess variance is half gone")
    a2.set_ylabel("Number of shocks")
    a2.set_title(f"Most shocks are short. A few are very long (median {np.median(v):.0f} days)", loc="left")
    title(fig, f"After {len(ev)} detected shocks: the next week is underpredicted, the next year overpredicted",
          "Volatility doubled vs its 5-year norm (trigger known on the day). Bars: 90% month-cluster bootstrap.", y=0.92)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig4_event_study")


# --------------------------------------------------------------------------- fig 5 QLIKE by horizon
def fig5_qlike(panel):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    hs = sorted(panel.h.unique())
    xs = np.arange(len(hs))
    models = ["EWMA", "GARCH-R10", "GJR-t", "CGARCH", "HAR"]
    offs = np.linspace(-0.2, 0.2, len(models))
    for j, m in enumerate(models):
        g = panel[panel.model == m].set_index("h").loc[hs]
        ax.vlines(xs + offs[j], g.lo, g.hi, color=C[m], lw=1.2)
        ax.plot(xs + offs[j], g.geo_mean_rel, MK[m], color=C[m], ms=8, markeredgecolor=SURFACE, label=NAME[m])
        for x, (_, rr) in zip(xs + offs[j], g.iterrows()):
            if rr.p_holm < 0.05:
                ax.text(x, rr.hi + 0.01, "*", ha="center", fontsize=12, color=INK)
    ax.axhline(1, color=C["GARCH"], lw=1.4, ls=(0, (4, 3)), label="GARCH(1,1) = 1")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{h} days" for h in hs])
    ax.set_ylim(0.75, 1.6)
    ax.set_ylabel("QLIKE loss relative to GARCH(1,1)")
    ax.legend(fontsize=8.5, ncol=3, loc="upper left")
    title(fig, "The decay error costs little on average, and more at long horizons",
          "Geometric mean over 10 indices, 90% year-block bootstrap. * = Holm-adjusted p < 0.05. Lower is better.",
          y=0.9)
    save(fig, "fig5_qlike_by_horizon")


# --------------------------------------------------------------------------- fig 6 eras
def fig6_eras(curves, hl):
    tags = [t for t in curves.tag.unique() if t.startswith("era")]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    cols = [C["HAR"], C["EWMA"], INK]
    for t, col in zip(tags, cols):
        cv = curves[(curves.tag == t) & (curves["index"] == "POOLED") & (curves.spec == "GARCH") & (curves.kind == "emp")]
        h = hl[(hl.tag == t) & (hl["index"] == "POOLED") & (hl.spec == "GARCH") & (hl.kind == "emp")].iloc[0]
        n_idx = curves[(curves.tag == t) & (curves["index"] != "POOLED")]["index"].nunique()
        ax.plot(cv.k, cv.norm, color=col, lw=2.2,
                label=f"{t[3:7]}–{min(int(t[8:12]), 2026)} ({n_idx} {'index' if n_idx == 1 else 'indices'}): "
                      f"half-life {h.half_life:.0f} d (90% CI {h.hl_lo:.0f}–{h.hl_hi:.0f})")
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks([1, 22, 63, 126, 189, 252])
    ax.set_xticklabels(["1 day", "1 month", "3 months", "6 months", "9 months", "1 year"])
    ax.set_xlim(1, 252)
    ax.set_ylabel("Share of shock still visible (out of sample)")
    ax.legend(fontsize=9, loc="upper right")
    title(fig, "No sign that shocks have become shorter or longer over 90 years",
          "Out-of-sample local projections on the expanding-window GARCH variance, by period of the forecast origin",
          y=0.9)
    save(fig, "fig6_eras")


# --------------------------------------------------------------------------- fig 7 example episodes
def fig7_examples(paths, ev):
    picks = []
    # four well-known shocks that the rule-based trigger detected (S&P 500 2008 and 2020 were already
    # inside an earlier episode under the 126-day cool-down, so other markets stand in for them)
    for idx, lo, hi in [("GSPC", "1987-10-01", "1987-12-31"), ("BVSP", "2008-09-01", "2008-12-31"),
                        ("GDAXI", "2020-02-15", "2020-04-30"), ("N225", "2024-07-15", "2024-09-30")]:
        g = ev[(ev["index"] == idx) & (pd.to_datetime(ev.date) >= lo) & (pd.to_datetime(ev.date) <= hi)]
        if len(g):
            picks.append((idx, str(g.date.iloc[0])))
    fig, axes = plt.subplots(1, len(picks), figsize=(3.4 * len(picks), 4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (idx, dt) in zip(axes, picks):
        p = paths[(paths["index"] == idx) & (paths.date == dt)]
        real = p[p.spec == "GARCH"]
        ax.plot(real.k, np.sqrt(real.real), color=INK, lw=2.2, label="Realized (21-day centred)")
        for spec in ["GARCH", "CGARCH"]:
            q = p[p.spec == spec]
            ax.plot(q.k, np.sqrt(q.pred), color=C[spec], lw=2, ls=(0, (5, 2)), label=f"{NAME[spec]} forecast")
        ax.axhline(1, color=MUTED, lw=1, ls=":")
        ax.set_title(f"{LABEL[idx]}, {dt}", loc="left", fontsize=10)
        ax.set_xticks([1, 63, 126, 252])
        ax.set_xticklabels(["1d", "3m", "6m", "1y"])
    axes[0].set_ylabel("Volatility ÷ its 5-year norm")
    axes[0].legend(fontsize=8, loc="upper right")
    title(fig, "Four shocks, as the models saw them on the day",
          "Forecast path from the trigger day (parameters estimated before it) vs what followed. "
          "Dotted line = the 5-year norm.", y=0.93)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig7_example_episodes")


def main():
    curves = pd.read_csv(RES / "irf_curves.csv.gz")
    hl = pd.read_csv(RES / "half_lives.csv")
    nc, nt = pd.read_csv(RES / "null_world_curves.csv"), pd.read_csv(RES / "null_world_tests.csv")
    fig1_decay(nc, nt)
    fig2_per_index(nt)
    fig3_decades(pd.read_csv(RES / "decade_params.csv"), pd.read_csv(RES / "stability_tests.csv"))
    ev = pd.read_csv(RES / "events.csv")
    fig4_events(pd.read_csv(RES / "event_errors.csv"), ev)
    fig5_qlike(pd.read_csv(RES / "qlike_panel.csv"))
    fig6_eras(curves, hl)
    fig7_examples(pd.read_csv(RES / "event_paths.csv.gz"), ev)
    print("figures written")


if __name__ == "__main__":
    main()
