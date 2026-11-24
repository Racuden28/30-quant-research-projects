"""Publication figures for project 24. Run after experiments.py, theory.py and evaluate.py."""
from __future__ import annotations
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_runs, common_start, RES, STRESS
from experiments import UNIV
import engine as en
import theory

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# ---- design tokens shared with projects 01-23 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
REAL, PLACEBO, NOISE = "#2a78d6", "#eb6834", "#8a8984"
GOOD, BAD = "#1baf7a", "#a51c1c"
UL = {"equity": "S&P 500 stocks, 2001–2026", "crypto": "Binance coins incl. delisted, 2019–2026"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def title(fig, t, sub, y=1.0):
    fig.text(0.01, y + 0.05, t, fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(0.01, y + 0.008, sub, fontsize=10, color=INK2, ha="left", va="bottom")


def ulabel(u, runs):
    t0 = common_start(runs, u)
    end = runs[(u, "main")]["pnl"]["ou"].index[-1]
    name = "S&P 500 stocks" if u == "equity" else "Binance coins incl. delisted"
    return f"{name}, {t0:%b %Y}–{end:%b %Y}"


# --------------------------------------------------------------------------------------
def fig1_star(runs):
    flt = pd.read_csv(os.path.join(RES, "theory_filter_null.csv"))
    noise = flt[(flt.L == 60) & (flt.design == "intercept + 15 factors")].pass_share.iloc[0]
    fr = pd.read_csv(os.path.join(RES, "filter_real_vs_placebo.csv"))
    mn = pd.read_csv(os.path.join(RES, "main_real_vs_placebo.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"wspace": 0.62, "width_ratios": [1, 1.15]})
    # A: who passes the "fast mean reversion" filter
    ax = axes[0]
    rows = [("Pure noise\n(simulated, no signal)", noise, NOISE)]
    for u in ("equity", "crypto"):
        f = fr[fr.universe == u]
        real = float(f[f.world == "real"].pass_share.iloc[0])
        pl = float(f[f.world.str.startswith("placebo (mean")].pass_share.iloc[0])
        nm = "S&P 500" if u == "equity" else "Crypto"
        rows.append((f"{nm}: placebo market", pl, PLACEBO))
        rows.append((f"{nm}: real market", real, REAL))
    for i, (lab, v, c) in enumerate(rows[::-1]):
        ax.barh(i, 100 * v, 0.62, color=c)
        ax.text(100 * v - 1.5, i, f"{100 * v:.1f}%", va="center", ha="right", color="white", fontweight="bold")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of asset-days that pass the\n'mean-reverts within 30 days' filter (%)")
    ax.set_title("A. The filter lets almost everything through", loc="left")
    ax.grid(axis="y", visible=False)
    # B: Sharpe ratio of the strategy, real vs 20 placebo markets
    ax = axes[1]
    cases = [("equity", 0, "gross"), ("equity", 1, "gross"), ("equity", 1, "net"),
             ("crypto", 0, "gross"), ("crypto", 1, "gross"), ("crypto", 1, "net")]
    labels = []
    rng = np.random.default_rng(0)
    for i, (u, d, kind) in enumerate(cases):
        y = len(cases) - 1 - i + (0.0 if u == "crypto" else 0.6)
        plac = [r for (uu, l), r in runs.items() if uu == u and l.startswith("placebo")]
        t0 = common_start(runs, u)
        from evaluate import metrics
        key = "sr_gross" if kind == "gross" else "sr_net"
        ps = [metrics(r["pnl"]["ou"].loc[t0:], u, d)[key] for r in plac]
        rs = metrics(runs[(u, "main")]["pnl"]["ou"].loc[t0:], u, d)[key]
        ax.scatter(ps, y + rng.uniform(-0.12, 0.12, len(ps)), s=16, color=PLACEBO, alpha=0.7, lw=0,
                   label="20 placebo markets" if i == 0 else None, zorder=2)
        ax.scatter([rs], [y], s=90, color=REAL, edgecolor="white", lw=1.2, zorder=3,
                   label="Real market" if i == 0 else None)
        ax.text(rs, y + 0.28, f"{rs:+.2f}", ha="center", fontsize=9, color=REAL, fontweight="bold")
        cost = UNIV[u]["cost"]
        lab = {("gross", 0): "trade at signal close\nno costs",
               ("gross", 1): "wait one day\nno costs",
               ("net", 1): f"wait one day\n{cost:.0f} bp + borrow"}[(kind, d)]
        labels.append((y, ("S&P 500: " if u == "equity" else "Crypto: ") + lab))
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks([l[0] for l in labels], [l[1] for l in labels])
    ax.set_xlabel("Annualised Sharpe ratio of the Avellaneda-Lee s-score strategy")
    ax.set_title("B. Stocks: real edge, gone after costs. Crypto: below noise", loc="left")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, frameon=True, edgecolor=GRID)
    ax.grid(axis="y", visible=False)
    title(fig, "A residual that 'mean-reverts' is not evidence of an edge",
          "PCA stat arb (Avellaneda-Lee 2010), walk-forward, no look-ahead. Placebo market = every day's returns "
          "multiplied by a random sign: same covariances and volatility, nothing predictable.")
    save(fig, "fig1_filter_and_edge")


def fig2_ic(runs):
    ic = pd.read_csv(os.path.join(RES, "ic_term_structure.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"wspace": 0.22})
    names = {"res1": ("yesterday's residual", GOOD), "res5": ("5-day residual", REAL), "ou": ("OU s-score", PLACEBO)}
    for ax, u in zip(axes, ("equity", "crypto")):
        d = ic[ic.universe == u]
        for n, (lab, c) in names.items():
            x = d[d.signal == n]
            ax.plot(x.horizon, 100 * x.ic_mean, "-o", color=c, label=lab, ms=4)
            ax.fill_between(x.horizon, 100 * (x.placebo_ic_mean - 2 * x.placebo_ic_sd),
                            100 * (x.placebo_ic_mean + 2 * x.placebo_ic_sd), color=c, alpha=0.12, lw=0)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10, 20], ["1", "2", "3", "5", "10", "20"])
        ax.set_xlabel("Day ahead j (residual return on day t+j only)")
        ax.set_title(ulabel(u, runs), loc="left")
    axes[0].set_ylabel("Mean rank IC (%)  [reversal > 0]")
    axes[0].legend(fontsize=9, loc="upper right")
    title(fig, "The predictability lives in the first day or two",
          "Daily cross-sectional rank correlation between the reversal signal at t and the residual return on day t+j. "
          "Shaded: placebo mean ± 2 sd.")
    save(fig, "fig2_ic_term_structure")


def fig3_equity_path(runs):
    u = "equity"; t0 = common_start(runs, u)
    p = runs[(u, "main")]["pnl"]["ou"].loc[t0:]
    plac = [r for (uu, l), r in runs.items() if uu == u and l.startswith("placebo")]
    c = UNIV[u]["cost"]
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    for i, r in enumerate(plac):
        ax.plot(r["pnl"]["ou"].loc[t0:, "gross_d0"].cumsum() * 100, color=PLACEBO, lw=0.6, alpha=0.35,
                label="Placebo markets, no costs" if i == 0 else None)
    ax.plot(p.gross_d0.cumsum() * 100, color=REAL, lw=1.6, label="Real, trade at signal close, no costs")
    ax.plot(p.gross_d1.cumsum() * 100, color=REAL, lw=1.2, ls="--", label="Real, wait one day, no costs")
    net = p.gross_d1 - p.turn_d1 * c * 1e-4 - p.short * UNIV[u]["borrow"] / 252
    ax.plot(net.cumsum() * 100, color=BAD, lw=1.4, label=f"Real, wait one day, {c:.0f} bp per trade + borrow")
    for lab, a, b in STRESS[u]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GRID, alpha=0.8, lw=0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("Cumulative return per $1 of positions (%)")
    ax.legend(fontsize=9, loc="upper left")
    title(fig, "S&P 500: a real gross edge that waiting one day halves and costs erase",
          "Grey bands: Aug 2007 quant quake, GFC, COVID, 2022 bear, SVB. Sum of daily returns (not compounded).")
    save(fig, "fig3_equity_cumulative")


def fig4_kernel():
    L = 60
    rng = np.random.default_rng(7)
    e = rng.standard_normal((L, 3000)); e -= e.mean(0)
    _, _, m, seq, _ = en.ou_fit(e)
    s_unc = -m / seq
    approx = (theory.kernel_weights(L)[:, None] * e).sum(0) / seq
    ok = np.isfinite(s_unc)
    k = pd.read_csv(os.path.join(RES, "theory_kernel.csv")).iloc[0]
    flt = pd.read_csv(os.path.join(RES, "theory_filter_null.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    ax.bar(np.arange(1, L + 1), theory.kernel_weights(L), color=REAL, width=0.8)
    ax.set_xlabel("Day in the 60-day window (60 = today)"); ax.set_ylabel("Weight on that day's residual")
    ax.set_title("A. The s-score's implied weights", loc="left")
    ax = axes[1]
    ax.scatter(approx[ok], s_unc[ok], s=4, color=REAL, alpha=0.4, lw=0)
    ax.set_xlabel("Linear-weighted residual sum / σ_eq"); ax.set_ylabel("OU s-score")
    ax.set_title(f"B. Same number (corr {k.corr_uncentred:.3f})", loc="left")
    ax = axes[2]
    for lab, c in (("random walk, no regression", NOISE), ("intercept + 15 factors", PLACEBO)):
        x = flt[flt.design == lab]
        ax.plot(x.L, 100 * x.pass_share, "-o", color=c, label="pure noise: " + ("raw random walk" if "random" in lab else "regression residual (bridge)"))
    ax.axvline(60, color=INK2, lw=0.8, ls=":"); ax.text(62, 8, "window used\nby the paper", fontsize=8, color=INK2)
    ax.set_xlabel("Residual window L (days)"); ax.set_ylabel("Noise paths passing the filter (%)")
    ax.set_ylim(0, 102); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("C. Why: short windows + a bridge", loc="left")
    title(fig, "The OU s-score is a linearly-weighted 60-day residual reversal",
          "Regression residuals sum to zero, so the cumulative residual ends at 0 and the AR(1) fit reduces to a recency kernel.")
    save(fig, "fig4_kernel_and_bridge")


def fig5_crypto(runs):
    u = "crypto"; t0 = common_start(runs, u)
    p = runs[(u, "main")]["pnl"]["ou"].loc[t0:]
    plac = [r for (uu, l), r in runs.items() if uu == u and l.startswith("placebo")]
    ba = pd.read_csv(os.path.join(RES, "crypto_contribution_by_asset.csv"), index_col=0).iloc[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"wspace": 0.3, "width_ratios": [1.5, 1]})
    ax = axes[0]
    for i, r in enumerate(plac):
        ax.plot(r["pnl"]["ou"].loc[t0:, "gross_d0"].cumsum() * 100, color=PLACEBO, lw=0.6, alpha=0.35,
                label="Placebo markets, no costs" if i == 0 else None)
    ax.plot(p.gross_d0.cumsum() * 100, color=REAL, lw=1.6, label="Real, trade at signal close, no costs")
    net = p.gross_d1 - p.turn_d1 * UNIV[u]["cost"] * 1e-4 - p.short * UNIV[u]["borrow"] / 365
    ax.plot(net.cumsum() * 100, color=BAD, lw=1.4, label="Real, wait one day, 10 bp + 10% borrow")
    for lab, a, b in STRESS[u]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GRID, alpha=0.8, lw=0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("Cumulative return per $1 of positions (%)")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title(ulabel(u, runs), loc="left")
    ax = axes[1]
    sel = pd.concat([ba.head(8), ba.tail(4)])
    cols = [BAD if v < 0 else GOOD for v in sel]
    ax.barh(range(len(sel)), 100 * sel.values, color=cols)
    ax.set_yticks(range(len(sel)), [s.replace("USDT", "") for s in sel.index])
    ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Contribution to cumulative gross return (%)")
    ax.set_title("Who paid for it", loc="left")
    ax.grid(axis="y", visible=False)
    title(fig, "Crypto: the same machine loses money before costs",
          "Fading a coin's residual works on ordinary days and loses on the 0.4% of held coin-days that move more than 30%. "
          "Grey bands: stress windows.")
    save(fig, "fig5_crypto")


def fig6_ablations(runs):
    ab = pd.read_csv(os.path.join(RES, "ablations.csv"))
    order = ["k0", "k1", "main", "k5", "k15", "L120", "pit_added", "survivors", "N50"]
    names = {"k0": "k=0 (equal-weight market)", "k1": "k=1", "main": "k by Marchenko-Pastur (main)", "k5": "k=5",
             "k15": "k=15 (paper)", "L120": "L=120 residual window", "pit_added": "only after S&P inclusion",
             "survivors": "survivors only (no delisted)", "N50": "top 50 coins"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"wspace": 0.55})
    for ax, u in zip(axes, ("equity", "crypto")):
        d = ab[(ab.universe == u) & (ab.signal == "ou")]
        specs_ = [s for s in order if s in set(d.spec)]
        y = np.arange(len(specs_))
        for dl, c, off in ((0, REAL, 0.18), (1, PLACEBO, -0.18)):
            v = [d[(d.spec == s) & (d.delay == dl)].sr_gross.iloc[0] for s in specs_]
            ax.barh(y + off, v, 0.34, color=c, label="trade at signal close" if dl == 0 else "wait one day")
        ax.set_yticks(y, [names[s] for s in specs_]); ax.invert_yaxis()
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel("Gross Sharpe ratio, s-score strategy")
        ax.set_title(ulabel(u, runs), loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    title(fig, "Ablations: factors lift the stock Sharpe; no variant rescues crypto",
          "Gross Sharpe ratio, no costs, same common sample per universe. Net-of-cost numbers in results/ablations.csv.")
    save(fig, "fig6_ablations")


def fig7_costs(runs):
    cs = pd.read_csv(os.path.join(RES, "cost_sweep.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.22})
    for ax, u in zip(axes, ("equity", "crypto")):
        for s, d, c, ls, lab in (("ou", 0, REAL, "-", "s-score, trade at close"), ("ou", 1, REAL, "--", "s-score, wait one day"),
                                 ("res5", 0, GOOD, "-", "5-day residual, trade at close"), ("res5", 1, GOOD, "--", "5-day residual, wait one day")):
            x = cs[(cs.universe == u) & (cs.signal == s) & (cs.delay == d)]
            ax.plot(x.cost_bps, x.sr_net, ls, color=c, marker="o", ms=3, label=lab)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(UNIV[u]["cost"], color=MUTED, ls=":", lw=1)
        ax.set_xlabel("One-way cost per unit of turnover (bp)"); ax.set_title(ulabel(u, runs), loc="left")
    axes[0].set_ylabel("Net Sharpe ratio (no borrow cost)")
    axes[0].legend(fontsize=8)
    title(fig, "Break-even costs are a few basis points", "Dotted line: base cost used in the paper (5 bp stocks, 10 bp crypto).")
    save(fig, "fig7_cost_sweep")


if __name__ == "__main__":
    runs = load_runs()
    fig1_star(runs); fig2_ic(runs); fig3_equity_path(runs); fig4_kernel(); fig5_crypto(runs)
    fig6_ablations(runs); fig7_costs(runs)
    print("figures written")
