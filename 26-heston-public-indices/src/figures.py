"""Publication figures for project 26. Run after calibrate.py, evaluate.py, ablations.py, chain.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import calibrate as C  # noqa: E402
import heston as H  # noqa: E402
from data import FIG, RES, STRESS  # noqa: E402

# ---- design tokens shared with projects 01, 04, 16 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN, GOLD, PINK, DKGREEN, PURPLE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                     "#008300", "#7b5bd6")
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def shade(ax):
    for _, (a, b) in STRESS.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GRID, alpha=0.6, lw=0)


def fig1_two_volvols(c, e5):
    ok = c.sigma_status == "ok"
    s_v = c.sigma.where(ok).rolling(21, min_periods=10).median()
    s_k = c.sigma_skew_min.rolling(21, min_periods=10).median()
    sP = float(e5.loc[(e5.param == "sigma") & e5.P_estimator.str.startswith("robust"), "P"].iloc[0])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    shade(ax)
    ax.fill_between(s_v.index, s_v, s_k, where=s_k > s_v, color=ORANGE, alpha=0.12, lw=0)
    ax.plot(s_k.index, s_k, color=ORANGE, lw=1.6, label="Vol of vol that SKEW needs (even with ρ = −1)")
    ax.plot(s_v.index, s_v, color=BLUE, lw=1.6, label="Vol of vol that VVIX implies")
    ax.axhline(sP, color=INK2, lw=1.0, ls="--")
    ax.text(pd.Timestamp("2010-07-01"), sP * 1.04, f"history: robust physical vol of variance = {sP:.2f}",
            color=INK2, fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=AXIS, lw=0.6, alpha=0.95))
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([0.2, 0.5, 1, 2, 5])
    ax.set_yticklabels(["0.2", "0.5", "1", "2", "5"])
    ax.set_ylabel("Heston σ (vol of variance), 21-day median, log scale")
    infe = (c.rho_status != "ok").mean()
    ratio = (c.sigma_skew_min / c.sigma)[ok].median()
    ax.set_title(f"SKEW needs {ratio:.1f}× the vol of vol that VVIX allows: no Heston fits both on "
                 f"{100 * infe:.0f}% of days (2008–2026)", loc="left", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=9)
    ax.title.set_y(1.0)
    ax.title.set_va("bottom")
    ax.text(1.0, -0.13, "Grey bands: 2008–09 crisis, Covid 2020, 2022 bear market. Daily calibration to "
            "VIX9D, VIX, VIX3M, VIX6M, VVIX and SKEW (CBOE).\nGaps in the blue line: days when the observed VVIX "
            "is above the highest VVIX Heston can produce, so no σ exists (4.6% of days).",
            transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    save(fig, "fig1_two_volvols.png")


def fig2_vvix_hump(c):
    d = pd.Timestamp("2017-05-01")
    r = c.loc[d]
    sig = np.exp(np.linspace(np.log(0.05), np.log(60), 140))
    vv = [100 * H.vvix_model(r.v0, r.kappa, r.theta, s) for s in sig]
    sk = [H.skew_index(r.v0, r.kappa, r.theta, s, C.RHO_MIN) for s in sig]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axs[0]
    ax.plot(sig, vv, color=BLUE, lw=1.8)
    ax.axhline(r.VVIX, color=INK2, ls="--", lw=1)
    ax.axvline(r.sigma, color=BLUE, lw=1, ls=":")
    if np.isfinite(r.get("sigma_hi", np.nan)):
        ax.axvline(r.sigma_hi, color=MUTED, lw=1, ls=":")
        ax.text(r.sigma_hi * 0.92, 205, f"degenerate high root\nσ = {r.sigma_hi:.0f}", color=MUTED, fontsize=8,
                ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("σ (vol of variance), log scale")
    ax.set_ylabel("Model VVIX")
    ax.set_title(f"VVIX pins σ = {r.sigma:.2f} (low root)", loc="left")
    ax.text(r.sigma * 1.1, r.VVIX + 8, f"observed VVIX {r.VVIX:.0f}", fontsize=8.5, color=INK2)
    ax = axs[1]
    ax.plot(sig, sk, color=ORANGE, lw=1.8, label="best case ρ = −1")
    ax.axhline(r.SKEW, color=INK2, ls="--", lw=1)
    ax.axvline(r.sigma, color=BLUE, lw=1, ls=":")
    ax.axvline(r.sigma_skew_min, color=ORANGE, lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(98, 200)
    ax.set_xlabel("σ (vol of variance), log scale")
    ax.set_ylabel("Model SKEW")
    ax.set_title(f"…but SKEW {r.SKEW:.0f} needs σ ≥ {r.sigma_skew_min:.2f}", loc="left")
    ax.text(r.sigma * 1.08, 102, "σ from VVIX", color=BLUE, fontsize=8.5)
    ax.text(r.sigma_skew_min * 1.08, 190, "σ SKEW needs", color=ORANGE, fontsize=8.5)
    fig.suptitle(f"How one day is calibrated ({d.date()}): term structure → (v0, κ, θ), VVIX → σ, SKEW → ρ",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_identification_one_day.png")


def fig3_share_by_year(yr, ci):
    fig, ax = plt.subplots(figsize=(9.5, 4))
    ax.bar(yr.index.astype(str), 100 * yr.nonheston_share, color=ORANGE, alpha=0.85, width=0.7)
    ax.axhline(100 * yr.nonheston_share.median(), color=INK2, lw=1, ls="--")
    ax.set_ylabel("% of 30-day risk-neutral skewness\nHeston cannot produce (median day)")
    ax.set_title("Half or more of the S&P 500's priced skew lies outside diffusive stochastic volatility",
                 loc="left")
    ax.text(0.0, 1.0, f"full-sample median {100 * ci[0]:.0f}–{100 * ci[1]:.0f}% (95% block-bootstrap CI); "
            "σ fixed by VVIX, ρ at its most favourable value −1, so this is a lower bound",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 100)
    save(fig, "fig3_nonheston_share_by_year.png")


def fig4_state_dependence(el, e3):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
    for ax, col, q, lim, color in [(axs[0], "dVVIX_dVIX", "dVVIX/dVIX", (-4, 5), BLUE),
                                   (axs[1], "dSKEW_dVIX", "dSKEW/dVIX", (-1.2, 0.4), ORANGE)]:
        x = el[col].clip(*lim)
        ax.hist(x, bins=60, color=color, alpha=0.75)
        row = e3[e3.quantity == q].iloc[0]
        ax.axvspan(row.data_ci_lo, row.data_ci_hi, color=INK, alpha=0.15)
        ax.axvline(row.data_slope, color=INK, lw=1.6)
        ax.axvline(row.heston_median, color=color, lw=1.2, ls="--")
        ax.set_xlim(*lim)
        ax.set_xlabel(f"{q}: index points per VIX point")
        ax.set_title(f"{q}: Heston median {row.heston_median:+.2f}, data {row.data_slope:+.2f}", loc="left")
    axs[0].set_ylabel("weeks (model comparative statics)")
    axs[0].text(0.02, 0.95, "bars: Heston at each week's parameters (clipped at the axis ends)\n"
                "black line: daily regression slope in the data (95% Newey-West band)",
                transform=axs[0].transAxes, fontsize=8, color=INK2, va="top")
    fig.suptitle("Heston gets the state dependence wrong: it predicts VVIX barely moves with VIX and SKEW falls "
                 "four times faster than it does", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_state_dependence.png")


def fig5_forecasts(t4):
    tt = t4[t4.target.isin(["VIX9D", "VIX3M", "VIX6M"])]
    fig, ax = plt.subplots(figsize=(9, 4))
    names = ["VIX9D", "VIX3M", "VIX6M"]
    x = np.arange(3)
    w = 0.2
    h = tt[tt.benchmark == "shift"].set_index("target").loc[names, "rmse_heston"]
    vals = {"Heston, structure frozen at t": h}
    for b, lab in [("shift", "parallel shift by ΔVIX"), ("prop", "proportional to VIX"), ("rw", "no change")]:
        vals[lab] = tt[tt.benchmark == b].set_index("target").loc[names, "rmse_bench"]
    for i, (lab, v) in enumerate(vals.items()):
        ax.bar(x + (i - 1.5) * w, v.values, width=w, label=lab, color=[BLUE, GOLD, GREEN, MUTED][i])
    for j, nm in enumerate(names):
        g = tt[(tt.target == nm) & (tt.benchmark != "rw")]
        best = g.loc[g.rmse_bench.idxmin()]
        ax.text(j - 1.5 * w, h[nm] + 0.05, f"{best.mse_gain_pct:+.0f}% vs best\nHolm p={best.p_holm:.3f}",
                ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} on day t+1" for n in names])
    ax.set_ylabel("RMSE, vol points (4,680 days)")
    ax.set_title("Out of sample, the one-factor term-structure dynamics are Heston's strongest part", loc="left")
    ax.legend(fontsize=8.5, ncol=2, loc="upper right")
    ax.text(0, -0.16, "All forecasts know VIX on day t+1; Heston maps it to the other tenors with (κ, θ) "
            "from day t. MSE gain vs the better simple benchmark; Holm over 11 tests.",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "fig5_term_structure_forecasts.png")


def fig6_p_vs_q(yr):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axs[0]
    ax.plot(yr.year, yr.sigma_Q_skew_min, "o-", color=ORANGE, label="Q: needed by SKEW (ρ = −1)")
    ax.plot(yr.year, yr.sigma_Q_vvix, "o-", color=BLUE, label="Q: implied by VVIX")
    ax.plot(yr.year, yr.sigma_P_robust, "s--", color=INK2, label="P: robust realised vol of variance")
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([0.3, 0.5, 1, 2, 3])
    ax.set_yticklabels(["0.3", "0.5", "1", "2", "3"])
    ax.set_ylabel("σ (yearly median), log scale")
    ax.set_title("VVIX agrees with history; SKEW does not", loc="left")
    ax.legend(fontsize=8)
    ax = axs[1]
    ax.plot(yr.year, yr.rho_P, "s--", color=INK2, label="P: corr(d ln S, dv), daily")
    ax.plot(yr.year, yr.rho_Q_joint, "o-", color=ORANGE, label="Q: joint six-index fit")
    ax.set_ylim(-1.05, -0.4)
    ax.set_ylabel("ρ")
    ax.set_title("Q wants ρ at the −1 wall; history says about −0.7", loc="left")
    ax.legend(fontsize=8, loc="upper right")
    for a in axs:
        a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(3))
        a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
    fig.suptitle("Physical vs risk-neutral: under Heston's standard risk premium σ and ρ must be equal",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig6_p_vs_q.png")


def fig7_chain(E, tab):
    exps = sorted(E.expiry.unique())
    Ts = E.groupby("expiry")["T"].first()
    picks = [Ts.sub(t).abs().idxmin() for t in (8 / 365, 30 / 365, 0.5)]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    labs = {"Heston fitted to the chain": BLUE, "Index route: joint six-index fit": ORANGE}
    for ax, e in zip(axs, picks):
        g = E[(E.expiry == e)]
        m = g[g.model == "Heston fitted to the chain"].sort_values("k")
        ax.fill_between(m.k, 100 * (m.iv - m.half_spread), 100 * (m.iv + m.half_spread), color=MUTED, alpha=0.4,
                        lw=0, label="market bid–ask")
        ax.plot(m.k, 100 * m.iv, ".", color=INK, ms=2.5, label="market mid")
        for lab, col in labs.items():
            mm = g[g.model == lab].sort_values("k")
            ax.plot(mm.k, 100 * mm.iv_model, color=col, lw=1.5, label=lab)
        ax.set_title(f"{365 * Ts[e]:.0f}-day options", loc="left")
        ax.set_xlabel("log-moneyness ln(K/F)")
    axs[0].set_ylabel("implied vol, %")
    axs[0].legend(fontsize=7.5, loc="upper right")
    r = tab.set_index("model")
    fig.suptitle(f"S&P 500 chain, 2026-11-23: best Heston misses by {r.loc['Heston fitted to the chain', 'iv_rmse_volpts']:.1f} "
                 f"vol pts (half-spread ≈ 0.07); the index-route Heston by {r.loc['Index route: joint six-index fit', 'iv_rmse_volpts']:.1f}",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig7_chain_snapshot.png")


def main():
    c = pd.read_csv(RES / "daily_calibration.csv", index_col=0, parse_dates=True)
    e5 = pd.read_csv(RES / "e5_p_vs_q.csv")
    fig1_two_volvols(c, e5)
    fig2_vvix_hump(c)
    yr = pd.read_csv(RES / "e2_by_year.csv", index_col=0)
    t2 = pd.read_csv(RES / "e2_feasibility_by_window.csv")
    ci = eval(t2.loc[0, "nonheston_share_ci"])
    fig3_share_by_year(yr, ci)
    fig4_state_dependence(pd.read_csv(RES / "e3_model_elasticities_weekly.csv", index_col=0),
                          pd.read_csv(RES / "e3_state_dependence.csv"))
    fig5_forecasts(pd.read_csv(RES / "e4_forecast_tests.csv"))
    fig6_p_vs_q(pd.read_csv(RES / "e5_p_vs_q_by_year.csv"))
    fig7_chain(pd.read_csv(RES / "e6_iv_errors.csv.gz"), pd.read_csv(RES / "e6_chain_vs_index_route.csv"))


if __name__ == "__main__":
    main()
