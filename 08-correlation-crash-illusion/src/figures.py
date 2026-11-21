"""Publication figures for project 08. Run after conditioning.py and forecast.py."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from conditioning import WARM, market_of, masks  # noqa: E402
from core import corr, dcc_filter, dcc_fit, ewma_vol, q_to_r  # noqa: E402
from data import SECTOR, load  # noqa: E402
from forecast import BASELINE, EST_ORDER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens shared with project 01 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN, YELLOW, PINK, DKGREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f08a89", "#a51c1c"])
UNI = {"stocks": "19 US large caps, 1990–2022", "etfs": "6 cross-asset ETFs, 2007–2026"}
RULE_LABEL = {"crash_5": "Crash\nbottom 5%", "rally_5": "Rally\ntop 5%",
              "big_move_10": "Big move\n|r| top 10%", "highvol_ante_10": "High vol\nex ante",
              "calm_ante_50": "Calm\nex ante"}
RULES = list(RULE_LABEL)

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


def _matrix(pairs, cols, col):
    M = pd.DataFrame(np.full((len(cols), len(cols)), np.nan), index=cols, columns=cols)
    for _, p in pairs.iterrows():
        M.loc[p.a, p.b] = M.loc[p.b, p.a] = p[col]
    return M


def fig_heatmaps(pairs):
    p = pairs[pairs.universe == "stocks"]
    cols = list(load("stocks").columns)
    full = _matrix(p, cols, "rho_all").to_numpy(copy=True)
    np.fill_diagonal(full, 1)
    order = leaves_list(linkage(squareform(1 - full, checks=False), "average"))
    cols = [cols[i] for i in order]
    mats = [(_matrix(p, cols, "rho_all"), "A. All days", "Pearson correlation", (0, 0.8), "Blues"),
            (_matrix(p, cols, "rho_crash"), "B. Crash days, raw", "Pearson correlation", (0, 0.8), "Blues"),
            (_matrix(p, cols, "excess"), "C. Crash days minus constant-correlation null",
             "Excess correlation", (-0.25, 0.25), DIVERGING)]
    q = _matrix(p, cols, "q_bh")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    for ax, (M, t, lab, (lo, hi), cmap) in zip(axes, mats):
        M = M.loc[cols, cols]
        norm = TwoSlopeNorm(0, lo, hi) if lo < 0 else None
        im = ax.imshow(M.values, cmap=cmap, vmin=None if norm else lo, vmax=None if norm else hi, norm=norm)
        ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7.5)
        ax.set_yticks(range(len(cols)), [f"{c} · {SECTOR[c]}" for c in cols], fontsize=7.5)
        ax.grid(False)
        ax.set_title(t, loc="left")
        if "null" in t:
            Q = q.loc[cols, cols].values
            for i in range(len(cols)):
                for j in range(len(cols)):
                    if i != j and Q[i, j] < 0.05:
                        ax.text(j, i, "•", ha="center", va="center", fontsize=7, color=INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(lab, fontsize=8.5)
        cb.outline.set_visible(False)
    md = json.loads((RES / "conditioning_meta.json").read_text())[0]
    npos = int((p.excess > 0).sum())
    title(fig, "The crash-day heatmap shows less correlation than an average day, and both mislead",
          f"19 US large caps, {md['start'][:4]}–{md['end'][:4]}. Panel C subtracts what a world with constant latent "
          f"correlation shows on the same days: {npos} of {len(p)} pairs are red. No single pair survives "
          f"Benjamini–Hochberg at 5% (dots would mark them); the average does.", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig1_three_heatmaps")


PANELS = [("stocks", "all", "19 US large caps, 1990–2022: Stocks, average correlation"),
          ("etfs", "risk", "ETFs 2007–2026: Risk block (SPY, EEM, HYG, USO)"),
          ("etfs", "SPY-TLT", "ETFs 2007–2026: Stocks vs Treasuries (SPY–TLT)")]


def fig_decomposition(dec):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), sharey=True)
    comps = [("selection_artefact", MUTED, "1. Selection artefact: how the days were picked (Gaussian, constant correlation)"),
             ("tail_dependence", YELLOW, "2. Symmetric fat-tail dependence (Student-t null minus Gaussian null)"),
             ("regime_component", GREEN, "3. Slow regime shifts in latent correlation (DCC-path null minus t null)"),
             ("condition_specific", BLUE, "4. Left over: specific to these days (observed minus DCC-path null)")]
    for ax, (u, blk, lab) in zip(axes, PANELS):
        d = dec[(dec.universe == u) & (dec.block == blk)].set_index("rule").loc[RULES]
        x = np.arange(len(RULES))
        w = 0.19
        for k, (c, col, l) in enumerate(comps):
            ax.bar(x + (k - 1.5) * w, d[c], w, color=col, label=l)
        lo = d.gap_ci_lo - d.null_dcc_mean
        hi = d.gap_ci_hi - d.null_dcc_mean
        ax.errorbar(x + 1.5 * w, d.condition_specific, yerr=[d.condition_specific - lo, hi - d.condition_specific],
                    fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
        ax.scatter(x, d.gap_observed, marker="D", s=58, color=ORANGE, edgecolor=INK, linewidth=0.6, zorder=5,
                   label="Observed change vs all days (= sum of bars 1–4)")
        for xi, (_, row) in zip(x, d.iterrows()):
            if min(row.p_upper_vs_dcc, row.p_lower_vs_dcc) < 0.025:
                ax.text(xi + 1.5 * w, max(hi[row.name], 0) + 0.01, "*", ha="center", fontsize=13, color=INK)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x, [RULE_LABEL[r] for r in RULES], fontsize=8.5)
        ax.set_title(lab, loc="left", fontsize=10)
    axes[0].set_ylabel("Change in correlation vs all days")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.6, bbox_to_anchor=(0.5, -0.02))
    title(fig, "Most of the 'crisis correlation' a heatmap shows is how the days were chosen",
          "Each observed change (diamond) splits into four bars that add up to it. "
          "* bar 4 outside the 95% band of the DCC-path null; whiskers: 95% stationary-bootstrap CI", y=1.0)
    fig.text(0.01, -0.045, "Crash / Rally / Big move: days picked by that day's market return (ex post).  "
             "High vol / Calm: market volatility known the evening before (top 10% / below median).",
             fontsize=8.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    save(fig, "fig2_gap_decomposition")


def fig_null_hist(dec):
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.6))
    for row, (u, blk, plab) in enumerate(PANELS):
        dr = pd.read_csv(RES / f"null_draws_{u}.csv.gz")
        for ax, rule in zip(axes[row], ["crash_5", "big_move_10", "highvol_ante_10"]):
            key = rule if blk == "all" else f"{rule}@{blk}"
            obs = dec[(dec.universe == u) & (dec.block == blk) & (dec.rule == rule)].gap_observed.iloc[0]
            for null, c, lab in [("gauss", MUTED, "Gaussian null"), ("t", YELLOW, "Student-t null"),
                                 ("dcc", GREEN, "DCC-path null")]:
                v = dr.loc[dr.null == null, key]
                ax.hist(v, bins=40, color=c, alpha=0.6, label=lab, density=True)
            ax.axvline(obs, color=ORANGE, lw=2.2, label="Observed")
            ax.set_title(f"{plab.split(': ')[-1]} · {RULE_LABEL[rule].splitlines()[0]}", loc="left", fontsize=9)
            ax.set_yticks([])
            ax.set_xlabel("Gap in average correlation vs all days")
    axes[0, 0].legend(fontsize=8.3, loc="upper left")
    title(fig, "Each null removes one explanation; the orange line is what actually happened",
          "Simulated histories with the real volatility path: constant correlation (Gaussian, Student-t, 1,000 each) "
          "or correlation following the fitted DCC path (500)",
          y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig3_null_distributions")


def fig_exceedance(exc, meta):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, u, md in zip(axes, ["stocks", "etfs"], meta):
        e = exc[exc.universe == u]
        for side in [e.q < 0.5, e.q > 0.5]:
            s = e[side]
            ax.fill_between(s.q, s.t_p05, s.t_p95, color=YELLOW, alpha=0.3, lw=0)
            ax.plot(s.q, s.t_mean, color=YELLOW, lw=1.8)
            ax.plot(s.q, s.gauss_mean, color=MUTED, lw=1.4, ls=(0, (4, 3)))
            ax.plot(s.q, s.observed, color=ORANGE, lw=2.2, marker="o", ms=5, markeredgecolor=SURFACE)
        ax.axvline(0.5, color=AXIS, lw=0.8)
        ax.set_xlabel("Market-return quantile threshold (left: below it, right: above it)")
        ax.set_ylabel("Average pairwise correlation")
        ax.set_title(f"{UNI[u]}  ·  down-minus-up asymmetry p = {md['asym_p']:.2f}", loc="left", fontsize=10)
    axes[0].plot([], [], color=ORANGE, lw=2.2, marker="o", label="Observed")
    axes[0].plot([], [], color=YELLOW, lw=1.8, label="Student-t null (90% band shaded)")
    axes[0].plot([], [], color=MUTED, lw=1.4, ls=(0, (4, 3)), label="Gaussian null")
    axes[0].legend(fontsize=8.3, loc="center", bbox_to_anchor=(0.5, 0.52))
    title(fig, "Tail correlations rise as much in rallies as in crashes",
          "Longin–Solnik exceedance curves. In a constant-correlation world (yellow, grey) correlation falls the deeper "
          "into either tail you condition; the stocks (orange) stay flat, symmetrically")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig4_exceedance_curves")


def fig_paths():
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.6))
    for ax, u in zip(axes, ["stocks", "etfs"]):
        df = load(u)
        r = df.values
        sig = ewma_vol(r)
        z = (r / sig)[WARM:]
        dates = df.index[WARM:]
        (a, b), Qbar = dcc_fit(z)
        R = q_to_r(dcc_filter(z, a, b, Qbar)[1:])
        m = market_of(r, u, list(df.columns))
        msig = ewma_vol(m[:, None])[WARM:, 0]
        hv = msig >= np.quantile(msig, 0.9)
        ax.fill_between(dates, 0, 1, where=hv, transform=ax.get_xaxis_transform(), color=PINK, alpha=0.18,
                        lw=0, label="High-vol regime (ex-ante top decile)")
        if u == "stocks":
            iu = np.triu_indices(R.shape[1], 1)
            ax.plot(dates, np.tanh(np.arctanh(np.clip(R[:, iu[0], iu[1]], -.999, .999)).mean(1)), color=BLUE,
                    lw=1.2, label=f"DCC average correlation (a={a:.3f}, b={b:.3f})")
        else:
            c = list(df.columns)
            for (x, y), col in zip([("SPY", "TLT"), ("SPY", "HYG"), ("SPY", "GLD")], [BLUE, ORANGE, GREEN]):
                ax.plot(dates, R[:, c.index(x), c.index(y)], color=col, lw=1.1, label=f"{x}–{y}")
            ax.axhline(0, color=INK, lw=0.7)
            ax.set_ylim(-0.75, 1.2)
        ax.set_title(UNI[u], loc="left")
        ax.set_ylabel("Devolatised correlation")
        ax.legend(fontsize=8.3, loc="upper left", ncol=4)
        ax.margins(x=0)
    title(fig, "Latent correlation does move, slowly, and it moves with the volatility regime",
          "DCC(1,1) on EWMA-devolatised returns, full-sample parameters (descriptive)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig5_dcc_paths")


def fig_horse_race(s):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    ests = [e for e in EST_ORDER if e != BASELINE]
    for ax, u in zip(axes, ["stocks", "etfs"]):
        y = np.arange(len(ests))
        for k, (sub, col) in enumerate([("all", BLUE), ("stressed (top-20% vol origins)", ORANGE)]):
            d = s[(s.universe == u) & (s.subset == sub)].set_index("estimator").loc[ests]
            ax.barh(y + (k - 0.5) * 0.38, d.gmv_rel * 100, 0.38, color=col,
                    label="All origins" if sub == "all" else "Stressed origins (top 20% vol)")
            for yi, (_, row) in zip(y, d.iterrows()):
                if row.p_gmv_holm < 0.05:
                    xv = row.gmv_rel * 100
                    ax.text(xv + (0.3 if xv >= 0 else -0.3), yi + (k - 0.5) * 0.38, "*", va="center",
                            ha="left" if xv >= 0 else "right", fontsize=11)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(y, ests)
        ax.invert_yaxis()
        ax.set_xlabel(f"Realised GMV volatility vs '{BASELINE}' heatmap (%; negative is better)")
        ax.set_title(UNI[u], loc="left")
    axes[1].legend(fontsize=8.3, loc="lower right")
    title(fig, "Which heatmap should you trade on next month?",
          "Out-of-sample minimum-variance portfolios, monthly rebalancing, identical volatility forecasts. "
          "* Diebold–Mariano vs 1-year sample, Holm-adjusted p < 0.05")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig6_forecast_horse_race")


def main():
    dec = pd.read_csv(RES / "gap_decomposition.csv")
    pairs = pd.read_csv(RES / "pair_crash_excess.csv")
    exc = pd.read_csv(RES / "exceedance_curve.csv")
    meta = json.loads((RES / "conditioning_meta.json").read_text())
    fig_heatmaps(pairs)
    fig_decomposition(dec)
    fig_null_hist(dec)
    fig_exceedance(exc, meta)
    fig_paths()
    if (RES / "forecast_summary.csv").exists():
        fig_horse_race(pd.read_csv(RES / "forecast_summary.csv"))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
