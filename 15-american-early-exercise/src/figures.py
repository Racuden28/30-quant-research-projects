"""Publication figures for project 15. Run after accuracy.py, history.py and exercise.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- design tokens shared with projects 01 and 04 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7b5bd6"]
BLUE, ORANGE, GREEN, AMBER, PINK, DGREEN, PURPLE = PALETTE
PUT_C, CALL_C = "#a51c1c", "#184f95"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f08a89", "#a51c1c"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
})

STRESS = {"1929–32": ("1929-09-01", "1932-07-31"), "1937–38": ("1937-03-01", "1938-03-31"),
          "1973–74": ("1973-01-01", "1974-12-31"), "1987": ("1987-08-25", "1987-12-31"),
          "2008": ("2008-09-01", "2009-03-31"), "2020": ("2020-02-19", "2020-04-30"),
          "2022–23": ("2022-01-01", "2023-10-31")}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def title(fig, t, sub, y=1.0, gap=None):
    fig.text(0.01, y, t, fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(0.01, y - (gap if gap is not None else 0.2 / fig.get_figheight()), sub, fontsize=10, color=INK2, ha="left", va="bottom")


METH_STYLE = {
    "crr": ("CRR", MUTED, "o", "-"), "tian": ("Tian", AMBER, "v", "-"), "tri": ("Trinomial", PINK, "D", "-"),
    "bbs": ("BBS", GREEN, "s", "--"), "crr_cv": ("CRR + control variate", PURPLE, "P", "-"),
    "lr": ("Leisen-Reimer", ORANGE, "^", "-"), "lr_r2": ("LR + Richardson, 1/n² (textbook)", ORANGE, "x", ":"),
    "bbsr": ("BBS + Richardson", DGREEN, "s", "-"), "lr_r": ("LR + Richardson, 1/n (this paper)", BLUE, "o", "-"),
}


def fig_efficiency():
    e = pd.read_csv(RES / "accuracy_errors.csv")
    e = e[e.group == "put"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, col, lab in [(axes[0], "rmse_rel", "RMS relative error of the PRICE"),
                         (axes[1], "rmse_rel_eep", "RMS error as a share of the EARLY-EXERCISE PREMIUM")]:
        for m, (name, c, mk, ls) in METH_STYLE.items():
            d = e[e.method == m].sort_values("n")
            lw = 2.4 if m == "lr_r" else 1.4
            ax.plot(d.sec_per_option * 1e3, d[col], color=c, marker=mk, ls=ls, lw=lw, ms=4.5, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("CPU time per option, milliseconds (one core, numba)")
        ax.set_title(lab, loc="left")
    axes[0].axhline(1e-4, color=AXIS, lw=1, ls="--")
    axes[0].text(0.0021, 1.15e-4, "1 basis point of price", color=MUTED, fontsize=8)
    axes[1].axhline(0.01, color=AXIS, lw=1, ls="--")
    axes[1].text(0.0021, 0.0115, "1% of the premium", color=MUTED, fontsize=8)
    axes[1].legend(loc="upper right", fontsize=8.2, ncol=1)
    title(fig, "Part I. The same tree is 10–50× less accurate on the early-exercise premium than on the price",
          "1,000 random American puts (Broadie–Detemple design); each marker is n = 25, 50, …, 1600 steps. "
          "Down and to the left is better.", y=1.04)
    fig.tight_layout()
    save(fig, "fig1_tree_efficiency")


def fig_convergence():
    t = pd.read_csv(RES / "convergence_trace.csv")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for m in ["crr", "tian", "tri", "bbs", "lr", "bbsr", "lr_r"]:
        name, c, _, ls = METH_STYLE[m]
        err = (t[m] - t.ref).abs()
        ax.plot(t.n, err.clip(lower=1e-9), color=c, lw=0.9 if m in ("crr", "tian", "tri") else 1.6, ls=ls, label=name)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("number of time steps n")
    ax.set_ylabel("absolute pricing error ($)")
    ax.legend(ncol=4, fontsize=8.5, loc="lower left")
    title(fig, "Odd–even zig-zag and how to remove it",
          "Hull's American put (S = K = 50, r = 10%, σ = 40%, 5 months), every n from 10 to 1000. "
          "Reference 4.2842158 (LR at n = 20,003 and 40,007, extrapolated)", y=1.02)
    fig.tight_layout()
    save(fig, "fig2_convergence_hull")


ERAS = [("1928–63 (calls)", 1928, 1963, CALL_C, "o"), ("1964–2008 (puts)", 1964, 2008, "#ffffff", "o"),
        ("2009–22 (calls)", 2009, 2022, GREEN, "D"), ("2023–26 (puts)", 2023, 2026, "#f5d000", "*")]


def _label_points(Z, rx, qy):
    """One label per contour, where it crosses an empty horizontal band (q = 8.4% for calls, 7.0% for puts)."""
    pts = []
    for lev, qband in [(-4, 8.4), (-2, 8.4), (-1, 8.4), (1, 7.0), (2, 7.0), (4, 7.0)]:
        row = Z[np.argmin(np.abs(qy - qband))]
        k = np.argmin(np.abs(row - lev))
        pts.append((rx[k], qy[np.argmin(np.abs(qy - qband))]))
    return pts


def fig_rq_star():
    g = pd.read_csv(RES / "rq_grid.csv")
    h = pd.read_csv(RES / "history_daily.csv.gz", parse_dates=["date"], index_col="date")
    rg, qg = np.sort(g.r.unique()), np.sort(g.q.unique())
    Z = (g.pivot(index="q", columns="r", values="put_share") - g.pivot(index="q", columns="r", values="call_share"))
    Z = Z.reindex(index=qg, columns=rg).to_numpy() * 100
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    lim = 8.0
    im = ax.pcolormesh(rg * 100, qg * 100, np.clip(Z, -lim, lim), cmap=DIVERGING,
                       norm=TwoSlopeNorm(0, -lim, lim), shading="auto", rasterized=True)
    cs = ax.contour(rg * 100, qg * 100, Z, levels=[-4, -2, -1, 1, 2, 4], colors=[INK2], linewidths=0.5)
    # place contour labels in the empty top band, away from the monthly dots
    ax.clabel(cs, fmt=lambda v: f"{abs(v):.0f}%", fontsize=7.5, inline_spacing=3,
              manual=_label_points(Z, rg * 100, qg * 100))
    ax.plot([0, 9], [0, 9], color=INK, lw=0.8, ls="--")
    ax.text(6.4, 6.9, "r = q", rotation=29, fontsize=8, color=INK)
    m = h[["r", "q"]].resample("ME").mean()
    for lab, a, b, c, mk in ERAS:
        w = m[(m.index.year >= a) & (m.index.year <= b)]
        ax.scatter(w.r * 100, w.q * 100, s=26 if mk == "*" else 11, color=c, marker=mk, edgecolor=INK,
                   linewidth=0.35, label=lab, zorder=3)
    for txt, (r_, q_) in {"Nov 2026": (m.r.iloc[-1], m.q.iloc[-1]), "Jun 1942": (m.loc["1942-06"].r.iloc[0], m.loc["1942-06"].q.iloc[0]),
                          "Aug 1981": (m.loc["1981-08"].r.iloc[0], m.loc["1981-08"].q.iloc[0]),
                          "Dec 2015": (m.loc["2015-12"].r.iloc[0], m.loc["2015-12"].q.iloc[0])}.items():
        ax.annotate(txt, (r_ * 100, q_ * 100), xytext=(-70, 14) if r_ > 0.1 else (14, 10), textcoords="offset points", fontsize=8.5,
                    fontweight="bold", zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=AXIS, lw=0.5),
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_xlabel("short rate r, % per year (1-month T-bill)")
    ax.set_ylabel("dividend yield q, % per year (S&P 500)")
    ax.text(13.0, 1.2, "PUTS worth\nexercising early", color="white", fontsize=12, fontweight="bold", ha="center")
    ax.text(5.2, 7.75, "CALLS worth exercising early", color=CALL_C, fontsize=12, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))
    cb = fig.colorbar(im, ax=ax, pad=0.012, shrink=0.9, fraction=0.04)
    cb.set_label("early-exercise premium, % of the American price\n(red: put premium · blue: call premium)")
    cb.set_ticks([-8, -4, 0, 4, 8])
    cb.set_ticklabels(["8% call", "4% call", "0", "4% put", "8% put"])
    leg = ax.legend(title="monthly (r, q) of the S&P 500", loc="upper right", fontsize=8.5, markerscale=2.2,
                    title_fontsize=8.5, frameon=True, facecolor=SURFACE, edgecolor=AXIS)
    leg.get_frame().set_linewidth(0.6)
    title(fig, "Who should exercise early? Calls in 1930–63, puts in 1964–2008, calls in 2009–22, puts since 2023",
          "Background: premium of a 3-month at-the-money American option over its European twin at σ = 15%.\n"
          "Dots: where the S&P 500 actually was, month by month, 1928–2026 (6 months of 1932 with q > 9% lie above the frame).",
          y=1.075, gap=0.07)
    fig.tight_layout()
    save(fig, "fig3_rq_century")


def shade(ax):
    for k, (a, b) in STRESS.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GRID, alpha=0.7, lw=0)


def fig_history():
    h = pd.read_csv(RES / "history_daily.csv.gz", parse_dates=["date"], index_col="date")
    m = h.resample("ME").mean()
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.3, 1]))
    ax = axes[0]
    shade(ax)
    ax.plot(m.index, m.r * 100, color=PUT_C, lw=1.2, label="short rate r")
    ax.plot(m.index, m.q * 100, color=CALL_C, lw=1.2, label="dividend yield q")
    ax.set_ylabel("% per year")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(-4.5, 17)
    ax.set_title("Inputs: the carry that decides which side exercises", loc="left")
    ax = axes[1]
    shade(ax)
    ax.plot(m.index, m.put_atm_3m_share * 100, color=PUT_C, lw=1.3, label="3-month ATM put")
    ax.plot(m.index, -m.call_atm_3m_share * 100, color=CALL_C, lw=1.3, label="3-month ATM call (plotted downward)")
    ax.plot(m.index, m.put_atm_1y_share * 100, color=PUT_C, lw=0.7, alpha=0.55, label="1-year ATM put")
    ax.plot(m.index, -m.call_atm_1y_share * 100, color=CALL_C, lw=0.7, alpha=0.55, label="1-year ATM call (downward)")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_ylabel("premium, % of American price")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    ax.set_title("Early-exercise premium: puts above the line, calls below", loc="left")
    ax = axes[2]
    shade(ax)
    ax.plot(m.index, m.put_atm_3m_volpts, color=PUT_C, lw=1.2, label="3-month ATM put")
    ax.plot(m.index, m.call_atm_3m_volpts, color=CALL_C, lw=1.2, label="3-month ATM call")
    ax.set_ylabel("vol points")
    ax.set_title("The same premium in volatility points: what a European pricer would have to add", loc="left")
    ax.legend(loc="upper left", ncol=2)
    for k, (a, b) in STRESS.items():
        axes[0].text(pd.Timestamp(a), -1.2, k, fontsize=7, color=INK2, va="top", rotation=90)
    title(fig, "Ninety-eight years of the American premium on the S&P 500",
          "Daily, 1928-04 to 2026-11, monthly means; American by LR + Richardson (n = 400), European by Black–Scholes, "
          "trailing 63-day volatility. Grey: stress windows.", y=1.025)
    fig.tight_layout()
    save(fig, "fig4_eep_history")


CASE_STYLE = {"1981 put (r 14%, q 5%)": PUT_C, "2023 put (r 5%, q 1.5%)": ORANGE, "1942 call (r 0.3%, q 6%)": CALL_C}


def fig_misspec():
    s = pd.read_csv(RES / "misspec_sigma.csv")
    rr = pd.read_csv(RES / "misspec_rate.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for case, c in CASE_STYLE.items():
        for mny, lw, ls in [(1.0, 2.0, "-"), (0.9, 0.9, ":"), (1.1, 0.9, "--")]:
            d = s[(s.case == case) & (s.moneyness == mny)]
            ax.plot(d.sigma_mult, d.loss_share * 100, color=c, lw=lw, ls=ls,
                    label=case if mny == 1.0 else None)
    ax.axhline(100, color=INK, lw=0.8)
    ax.text(1.02, 104, "100% = no better than a European option", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.67, 0.8, 1, 1.25, 1.5, 2])
    ax.set_xticklabels(["½×", "⅔×", "0.8×", "true σ", "1.25×", "1.5×", "2×"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0, 200)
    ax.set_xlabel("volatility used to draw the exercise boundary, relative to the true one")
    ax.set_ylabel("early-exercise premium lost, %")
    ax.legend(loc="upper center", fontsize=8.5, title="1-year option, σ = 15% (solid ATM; dotted OTM, dashed ITM)",
              title_fontsize=8)
    ax.set_title("Wrong volatility", loc="left")
    ax = axes[1]
    for case, c in CASE_STYLE.items():
        d = rr[(rr.case == case) & (rr.moneyness == 1.0)]
        ax.plot(d.rate_mult, d.loss_share * 100, color=c, lw=2)
    ax.axhline(100, color=INK, lw=0.8)
    ax.set_ylim(0, 200)
    ax.set_xlabel("carry used to draw the boundary, relative to the true one\n(short rate r for the puts, dividend yield q for the call)")
    ax.set_ylabel("early-exercise premium lost, %")
    ax.set_title("Wrong carry (rate or dividend yield)", loc="left")
    title(fig, "Part III. The exercise rule needs to know volatility: too LOW is far worse than too high",
          "Exact policy values on the true CRR lattice (n = 2,000). A boundary drawn at ½ the true vol "
          "exercises so early that the American feature destroys value.", y=1.04)
    fig.tight_layout()
    save(fig, "fig5_boundary_misspec")


def fig_garch():
    g = pd.read_csv(RES / "garch_policies.csv")
    g = g[g.policy.isin(["static", "dynamic"])].copy()
    g = g[g.early_share < 0.999]      # drop cells where every rule exercises at once (deep ITM, low vol)
    g["cell"] = g.apply(lambda x: f"K/S {x.K_over_S:.1f} · r {x.r:.0%} · start vol {x.vol0:.0%}", axis=1)
    cells = g[g.policy == "static"].cell.tolist()
    y = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(11, 6))
    for k, (pol, c, lab, off) in enumerate([("static", ORANGE, "constant-vol tree boundary, drawn once at the start", -0.16),
                                            ("dynamic", BLUE, "tree boundary re-read daily at the current GARCH vol", 0.16)]):
        d = g[g.policy == pol].set_index("cell").reindex(cells)
        ax.errorbar(d.captured * 100, y + off, xerr=1.96 * d.se_captured * 100, fmt="o", color=c, ms=6,
                    capsize=2.5, lw=1.2, label=lab)
    ax.axvline(100, color=INK, lw=0.8)
    ax.text(101, -0.75, "100% = Longstaff–Schwartz\n(regression rule fitted on separate paths)", fontsize=8, color=INK, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(cells, fontsize=8.5)
    ax.set_ylim(len(cells) - 0.5, -1.1)
    ax.set_xlabel("share of the Longstaff–Schwartz early-exercise premium captured, % (95% CI)")
    ax.legend(loc="lower left", fontsize=9)
    title(fig, "With moving volatility, a frozen tree boundary throws away 17–95% of the premium",
          "3-month American puts, daily exercise, risk-neutral GJR-GARCH fitted to the S&P 500 1928–2026. 600,000 evaluation "
          "paths per cell,\nsame shocks for every rule. Two cells where every rule exercises at once are omitted.", y=1.0)
    fig.tight_layout()
    save(fig, "fig6_garch_policies")


if __name__ == "__main__":
    fig_efficiency()
    fig_convergence()
    fig_rq_star()
    fig_history()
    fig_misspec()
    if (RES / "garch_policies.csv").exists():
        fig_garch()
    print("figures written")
