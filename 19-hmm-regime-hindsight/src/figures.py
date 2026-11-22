"""Publication figures for project 19. Run after run_experiments.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build import build  # noqa: E402
from core import INDICES, LABEL, RES, ROOT  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens shared with projects 01 and 03 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN, RED, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b", "#7a5bd1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
})

RUNGS = [("R0 Hindsight", "gross", "R0\nSmoothed state,\nsame day"),
         ("R1 Smoothed", "gross", "R1\nSmoothed state,\ntraded next day"),
         ("R2 Filtered", "gross", "R2\nFiltered state,\nfull-sample params"),
         ("R3 Real-time", "gross", "R3\nFiltered state,\nreal-time params"),
         ("R3 Real-time", "net", "R3 net\n+ 5 bp costs")]


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def heading(fig, t, sub, y=0.995):
    fig.text(0.01, y, t, fontsize=14, fontweight="bold", ha="left", va="top")
    fig.text(0.01, y - 0.05, sub, fontsize=10, color=INK2, ha="left", va="top")


def fig1_ladder():
    perf = pd.read_csv(RES / "performance.csv").set_index(["index", "strategy", "returns"]).sharpe
    fig, ax = plt.subplots(figsize=(12, 7.2))
    xs = np.arange(len(RUNGS))
    M = np.array([[perf[(n, s, k)] - perf[(n, "Buy & hold", "net")] for s, k, _ in RUNGS] for n in INDICES])
    pl = np.array([perf[(n, "Vol placebo", "net")] - perf[(n, "Buy & hold", "net")] for n in INDICES])
    for i, n in enumerate(INDICES):
        ax.plot(xs, M[i], color=AXIS, lw=1.1, zorder=1)
    # labels at the right end, nudged apart so they do not overlap
    order = np.argsort(M[:, -1])
    ys = M[order, -1].copy()
    gap = 0.06
    for j in range(1, len(ys)):
        ys[j] = max(ys[j], ys[j - 1] + gap)
    ys -= max(0.0, (ys.mean() - M[:, -1].mean()) * 0.5)   # re-centre the column on the points
    for j, i in enumerate(order):
        ax.plot([xs[-1] + 0.04, xs[-1] + 0.3], [M[i, -1], ys[j]], color=AXIS, lw=0.7, zorder=1)
        ax.text(xs[-1] + 0.33, ys[j], LABEL[INDICES[i]], fontsize=8, color=INK2, va="center")
    med = np.median(M, axis=0)
    ax.plot(xs, med, color=BLUE, lw=3, marker="o", ms=9, mec=SURFACE, mew=1.5, zorder=3,
            label="HMM timing rule, median of the 10 markets (grey lines: each market)")
    for x, m in zip(xs, med):
        ax.text(x - 0.07, m + 0.07, f"{m:+.2f}", ha="right", va="bottom", fontsize=10, color=BLUE, fontweight="bold",
                bbox=dict(fc=SURFACE, ec="none", pad=1.5), zorder=4)
    ax.scatter([xs[-1] + 0.0] * 0 , [], s=0)
    ax.axhline(0, color=INK, lw=1.2, zorder=2)
    ax.text(-0.35, 0.02, "Buy & hold", fontsize=9, color=INK, va="bottom")
    ax.scatter(np.full(len(pl), xs[-1] + 1.35), pl, marker="s", s=36, color=ORANGE, edgecolor=SURFACE, zorder=3,
               label="Volatility-only placebo with the same exposure, net (one square per market)")
    ax.set_xticks(xs, [r[2] for r in RUNGS])
    ax.set_xlim(-0.45, xs[-1] + 1.55)
    ax.set_ylabel("Sharpe ratio minus buy & hold's\n(annualised, excess over cash)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=9)
    n_below = int((M[:, -1] < 0).sum())
    share = (med[0] - med[-1]) / med[0] * 100 if med[0] > 0 else np.nan
    heading(fig, "Almost all of an HMM regime strategy's backtest edge is hindsight",
            f"Each step removes one source of look-ahead. In real time and after costs the HMM rule trails buy & hold in "
            f"{n_below} of 10 markets.\nThe step that matters is the smoother (R1 to R2), not the parameters (R2 to R3).")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "fig1_hindsight_ladder")


def fig2_episodes():
    b = build("GSPC")
    eps = [("2008", "2008-05-01", "2009-06-30"), ("2020", "2020-01-02", "2020-07-31"), ("2022", "2021-11-01", "2022-12-30")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), gridspec_kw=dict(height_ratios=[1.3, 1]), sharex="col")
    for j, (lab, a, e) in enumerate(eps):
        P = b["P"].loc[a:e]
        p = b["df"].p.loc[a:e]
        ax = axes[0, j]
        ax.plot(p.index, p / p.iloc[0] * 100, color=INK, lw=1.4)
        for flag, colr, alpha, yy in [(P.p_smooth > 0.5, RED, 0.16, None)]:
            ax.fill_between(p.index, 0, 1, where=flag.reindex(p.index).values, color=colr, alpha=alpha,
                            transform=ax.get_xaxis_transform(), lw=0)
        rt = (P.p_rt_pred > 0.5).reindex(p.index)
        ax.fill_between(p.index, 0, 0.07, where=rt.values, color=BLUE, alpha=0.85,
                        transform=ax.get_xaxis_transform(), lw=0)
        ax.set_title(f"S&P 500, {lab}", loc="left")
        if j == 0:
            ax.set_ylabel("Index (start = 100)")
        ax2 = axes[1, j]
        ax2.plot(P.index, P.p_smooth, color=RED, lw=1.8, label="Hindsight: smoothed P(turbulent), uses the whole sample")
        ax2.plot(P.index, P.p_rt_filt, color=BLUE, lw=1.2, label="Real time: filtered P(turbulent), data up to that day only")
        ax2.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        ax2.set_ylim(-0.03, 1.03)
        if j == 0:
            ax2.set_ylabel("Probability")
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3 if lab != "2020" else 2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    h, l = axes[1, 0].get_legend_handles_labels()
    import matplotlib.patches as mp
    h += [mp.Patch(color=RED, alpha=0.25), mp.Patch(color=BLUE)]
    l += ["Shaded: hindsight says turbulent", "Blue strip: real-time rule is out of the market"]
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    heading(fig, "Hindsight draws clean regimes; in real time the same model flickers and arrives late",
            "Red: smoothed probability from the model fitted on 1928–2026, what a typical backtest shows. "
            "Blue: filtered probability with parameters estimated only on data up to that day", y=1.02)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    save(fig, "fig2_episodes_sp500")


def fig3_attribution():
    a = pd.read_csv(RES / "attribution.csv")
    comps = [("late exit (before the first real-time exit)", RED, "Late exit: still in during the first days of a sell-off"),
             ("re-entry inside a spell (flicker)", ORANGE, "Flicker: back in for a day or two inside a turbulent spell"),
             ("late return (still out after the spell)", GREEN, "Late return: still out in the first days of the rebound"),
             ("false alarm", PURPLE, "False alarm: out while hindsight says calm")]
    keep = [c for c, _, _ in comps]
    a = a[a.cell.isin(keep)]
    piv = a.pivot(index="index", columns="cell", values="contribution_ann").loc[INDICES, keep] * 100
    sh = a.pivot(index="index", columns="cell", values="share_days").loc[INDICES] * 100
    bp = a.pivot(index="index", columns="cell", values="mean_daily_bp").loc[INDICES]
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    y = np.arange(len(INDICES))
    left = np.zeros(len(INDICES))
    neg = np.zeros(len(INDICES))
    for c, colr, lab in comps:
        v = piv[c].values
        start = np.where(v >= 0, left, neg)        # negative pieces are drawn to the left of zero
        ax.barh(y, v, left=start, color=colr, height=0.62, edgecolor=SURFACE, lw=0.8, label=lab)
        left += np.where(v > 0, v, 0)
        neg += np.where(v < 0, v, 0)
    for yi, n in enumerate(INDICES):
        c0 = comps[0][0]
        ax.text(left[yi] + 0.15, yi, f"late exit: {sh.loc[n, c0]:.1f}% of days at {bp.loc[n, c0]:+.0f} bp/day",
                va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [LABEL[n] for n in INDICES])
    ax.invert_yaxis()
    ax.set_xlim(neg.min() - 0.4, left.max() + 3.6)
    ax.set_xlabel("Hindsight rule's gain over the real-time rule on the days they disagree (percentage points of return per year)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.38), ncol=2, fontsize=8.8)
    med = np.median(piv[comps[0][0]] / piv.clip(lower=0).sum(axis=1)) * 100
    heading(fig, "The hindsight edge comes mostly from the first days of each sell-off",
            f"Days on which the smoothed and the real-time rule disagree, split by why. The late exit is the largest piece "
            f"(median {med:.0f}% of the positive total)", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig3_where_hindsight_earns")


def fig4_detection():
    d = pd.read_csv(RES / "detection.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    lag = d.lag_days.dropna()
    ax = axes[0]
    bins = np.arange(-5, max(40, lag.max()) + 2, 2)
    ax.hist(lag.clip(upper=bins[-1] - 1), bins=bins, color=BLUE, edgecolor=SURFACE)
    ax.axvline(lag.median(), color=INK, lw=1.4)
    ax.text(lag.median() + 0.6, ax.get_ylim()[1] * 0.92, f"median {lag.median():.0f} trading days", fontsize=9.5)
    ax.set_xlabel("Trading days from the hindsight start of a turbulent spell to the real-time exit")
    ax.set_ylabel("Number of spells")
    ax.set_title(f"{len(d)} turbulent spells (≥ 21 days) in 10 markets; {d.lag_days.isna().sum()} never flagged", loc="left",
                 fontsize=10, color=INK2, fontweight="normal")
    ax = axes[1]
    ok = d.dropna(subset=["lag_days"])
    frac = (ok.return_before_flag / ok.spell_trough_dd.where(ok.spell_trough_dd < -0.05))
    sel = ok.spell_trough_dd < -0.10
    ax.scatter(-ok.spell_trough_dd[~sel] * 100, -ok.drawdown_at_flag[~sel] * 100, s=18, color=AXIS,
               label="Spells with a trough under 10%")
    ax.scatter(-ok.spell_trough_dd[sel] * 100, -ok.drawdown_at_flag[sel] * 100, s=30, color=BLUE, edgecolor=SURFACE,
               label="Spells with a trough of 10% or more")
    m = max(ok.spell_trough_dd.min() * -100, 5) * 1.05
    ax.plot([0, m], [0, m], color=MUTED, lw=1, ls="--")
    ax.text(m * 0.62, m * 0.66, "exit at the bottom", rotation=0, fontsize=8.5, color=MUTED)
    ax.plot([0, m], [0, m / 2], color=MUTED, lw=0.8, ls=":")
    ax.text(m * 0.72, m * 0.33, "half the fall\nalready taken", fontsize=8.5, color=MUTED)
    ax.set_xlabel("Worst drawdown in the spell (% from the prior 3-month high)")
    ax.set_ylabel("Drawdown already taken when the\nreal-time rule exits (%)")
    big = ok[sel]
    ax.set_title(f"Spells with a 10%+ trough: median share of the fall taken before the exit is "
                 f"{(big.drawdown_at_flag / big.spell_trough_dd).median() * 100:.0f}% in real time\n"
                 f"(hindsight exits with {(big.drawdown_at_hindsight_exit / big.spell_trough_dd).median() * 100:.0f}% taken)",
                 loc="left", fontsize=10, color=INK2, fontweight="normal")
    ax.legend(loc="upper left", fontsize=8.5)
    heading(fig, "The real-time filter is only a couple of days late, but they are the worst days",
            "Real-time detection of the turbulent spells that hindsight finds (lags of -5 mean the real-time rule was already out 5+ days before)", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig4_detection_lag")


def fig5_model_world():
    mv = pd.read_csv(RES / "model_vs_real.csv")
    sim = pd.read_csv(RES / "model_world.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    y = np.arange(len(INDICES))
    for ax, strat, ttl in [(axes[0], "R3 Real-time", "Real-time HMM rule minus buy & hold (Sharpe, net)"),
                           (axes[1], "R3 minus vol placebo", "Real-time HMM rule minus volatility placebo (Sharpe, net)")]:
        g = mv[mv.strategy == strat].set_index("index").loc[INDICES]
        ax.hlines(y, g.sim_p05, g.sim_p95, color=AXIS, lw=6, label="Model world: 90% range over 40 simulated histories")
        ax.scatter(g.sim_median_minus_bh, y, marker="|", s=160, color=INK2, label="Model world: median")
        ax.scatter(g.real_minus_bh, y, s=55, color=BLUE, edgecolor=SURFACE, zorder=3, label="Real data")
        ax.axvline(0, color=INK, lw=1)
        ax.set_title(ttl, loc="left", fontsize=10.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [LABEL[n] for n in INDICES])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", fontsize=8.5, bbox_to_anchor=(0, -0.32), ncol=3)
    g = mv[mv.strategy == "R3 Real-time"]
    below = int((g.real_percentile_in_sim < 0.05).sum())
    heading(fig, "Real markets deliver less than the fitted HMM promises",
            f"If returns really came from each market's fitted two-state HMM, real-time timing would usually beat buy & hold. "
            f"Real data falls below the model's 5th percentile in {below} of 10 markets", y=1.04)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    save(fig, "fig5_model_world")


def fig6_stability():
    f = pd.read_csv(RES / "realtime_fits.csv", parse_dates=["date"])
    g = f[f["index"] == "GSPC"]
    full = pd.read_csv(RES / "full_sample_params.csv").set_index("index").loc["GSPC"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (c1, c2, lab, sc) in zip(axes, [("mu_calm", "mu_turb", "Mean log return by state (% per year)", 1),
                                            ("vol_calm", "vol_turb", "Volatility by state (% per year)", 1),
                                            ("dur_calm", "dur_turb", "Expected spell length (trading days, log)", 1)]):
        ax.plot(g.date, g[c1] * sc, color=GREEN, lw=1.4, label="Calm state")
        ax.plot(g.date, g[c2] * sc, color=RED, lw=1.4, label="Turbulent state")
        ax.axhline(full[c1] * sc, color=GREEN, lw=0.8, ls="--")
        ax.axhline(full[c2] * sc, color=RED, lw=0.8, ls="--")
        ax.set_title(lab, loc="left", fontsize=10)
        if c1 == "dur_calm":
            ax.set_yscale("log")
        if c1 == "mu_calm":
            ax.axhline(0, color=INK, lw=0.8)
    axes[0].legend(loc="lower left", fontsize=8.5)
    heading(fig, "What the real-time S&P 500 model believed, refit every month since 1933",
            "Solid: expanding-window estimate at each refit. Dashed: the full-sample (1928–2026) estimate a backtest would use", y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "fig6_parameter_paths_sp500")


def fig7_stress():
    s = pd.read_csv(RES / "stress_windows.csv")
    wins = ["1987 crash", "2000-02 dot-com", "2008 GFC", "2020 COVID", "2022 bear"]
    strats = [("Buy & hold", INK2), ("R1 Smoothed", RED), ("R3 Real-time", BLUE), ("Vol placebo", ORANGE), ("MA200", GREEN)]
    lab = {"R1 Smoothed": "HMM, hindsight (R1)", "R3 Real-time": "HMM, real time (R3, net)",
           "Vol placebo": "Vol placebo (net)", "MA200": "200-day MA (net)", "Buy & hold": "Buy & hold"}
    fig, axes = plt.subplots(1, len(wins), figsize=(14, 4.6), sharey=True)
    for ax, w in zip(axes, wins):
        g = s[s.window == w]
        for j, (k, c) in enumerate(strats):
            v = g[k].values * 100
            ax.scatter(np.full(len(v), j) + np.linspace(-0.18, 0.18, len(v)), v, s=16, color=c, alpha=0.8)
            ax.hlines(np.median(v), j - 0.3, j + 0.3, color=c, lw=2.5)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(len(strats)), [lab[k] for k, _ in strats], rotation=60, ha="right", fontsize=8)
        ax.set_title(f"{w} ({len(g)} mkts)", loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Cumulative excess return in the window (%)")
    heading(fig, "In the crashes, real-time HMM timing helps when volatility explodes and not in slow bears",
            "Dots: markets; bar: median across markets. Windows run from the S&P 500's pre-crash high to its trough", y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "fig7_stress_windows")


def fig8_regressions():
    r = pd.read_csv(RES / "regressions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    y = np.arange(len(INDICES))
    for ax, tgt, ttl in [(axes[0], "yret", "Next-month risk-adjusted return"), (axes[1], "yvol", "Next-month log realised volatility")]:
        for off, spec, colr, lab in [(-0.16, "P only", AXIS, "P(turbulent) alone"),
                                     (0.16, "P + log vol", BLUE, "P(turbulent) controlling for EWMA volatility")]:
            g = r[(r.target == tgt) & (r.spec == spec) & (r.coef_name == "p")].set_index("index").loc[INDICES]
            sig = g.p_holm < 0.05
            ax.errorbar(g.coef, y + off, xerr=1.96 * g.se, fmt="none", ecolor=colr, elinewidth=1.4)
            ax.scatter(g.coef[sig], (y + off)[sig.values], color=colr, s=36, zorder=3, label=lab + " (filled: Holm 5%)")
            ax.scatter(g.coef[~sig], (y + off)[~sig.values], facecolor=SURFACE, edgecolor=colr, s=36, zorder=3, lw=1.4)
        ax.axvline(0, color=INK, lw=1)
        ax.set_title(ttl, loc="left", fontsize=10.5)
        ax.set_xlabel("Coefficient on the real-time turbulence probability (95% Newey–West CI)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [LABEL[n] for n in INDICES])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, -0.36), fontsize=8.5, ncol=2)
    heading(fig, "What the real-time regime probability knows, once volatility is known",
            "Daily predictive regressions over the next 21 trading days, one market per row", y=1.04)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    save(fig, "fig8_beyond_volatility")


def fig9_belief():
    d = pd.read_csv(RES / "belief_vs_reality.csv")
    g = d[d.days == "real-time out"].set_index("index").loc[INDICES]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    y = np.arange(len(INDICES))
    ax.scatter(g.model_expected_ann * 100, y, marker="D", s=50, color=RED, zorder=3,
               label="What the real-time model expected on those days (its predicted mean)")
    ax.errorbar(g.realised_ann * 100, y, xerr=[(g.realised_ann - g.ci_lo) * 100, (g.ci_hi - g.realised_ann) * 100],
                fmt="o", color=BLUE, ms=7, mec=SURFACE, elinewidth=1.4, capsize=3, zorder=3,
                label="What the index actually earned on those days (95% Newey–West CI)")
    for yi in y:
        ax.plot([g.model_expected_ann.iloc[yi] * 100, g.realised_ann.iloc[yi] * 100], [yi, yi], color=AXIS, lw=1, zorder=1)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [f"{LABEL[n]}  ({int(g.n_days[n])} days)" for n in INDICES])
    ax.invert_yaxis()
    ax.set_xlabel("Annualised excess return on the days the real-time HMM rule was out of the market (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), fontsize=9)
    npos = int((g.realised_ann > 0).sum())
    nsig = int((g.ci_lo > 0).sum())
    heading(fig, "The bear regime's negative mean does not survive out of sample",
            f"The model stepped out because it expected losses. The index rose on those days in {npos} of 10 markets "
            f"(CI above zero in {nsig})", y=1.04)
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    save(fig, "fig9_belief_vs_reality")


if __name__ == "__main__":
    for f in [fig1_ladder, fig2_episodes, fig3_attribution, fig4_detection, fig5_model_world, fig6_stability,
              fig7_stress, fig8_regressions, fig9_belief]:
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
