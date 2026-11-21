"""Statistics for project 05: horse race, information factorial, optimism gap, stress, robustness."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import BOOT_REPS, COV_LEVELS, MEAN_LEVELS, SEED, STRATEGIES, STRESS, UNIVERSES

RES = Path(__file__).resolve().parents[1] / "results"


def read(name):
    return pd.read_csv(RES / name, index_col=0, parse_dates=True)


def stationary_indices(t, block, reps, rng):
    """Politis-Romano stationary bootstrap index matrix (reps x t)."""
    p = 1.0 / block
    idx = np.empty((reps, t), dtype=np.int64)
    idx[:, 0] = rng.integers(0, t, reps)
    new = rng.random((reps, t)) < p
    jumps = rng.integers(0, t, (reps, t))
    for i in range(1, t):
        idx[:, i] = np.where(new[:, i], jumps[:, i], (idx[:, i - 1] + 1) % t)
    return idx


def sharpe(x, ppy):
    return x.mean(axis=-2) / x.std(axis=-2, ddof=1) * np.sqrt(ppy)


def boot_sharpes(df: pd.DataFrame, ppy, block, reps=BOOT_REPS, seed=SEED):
    """Bootstrap distribution of annualised Sharpe ratios, paired across columns."""
    x = df.to_numpy()
    rng = np.random.default_rng(seed)
    idx = stationary_indices(len(x), block, reps, rng)
    out = np.empty((reps, x.shape[1]))
    for a in range(0, reps, 100):
        out[a:a + 100] = sharpe(x[idx[a:a + 100]], ppy)
    return pd.DataFrame(out, columns=df.columns)


def diff_test(point: float, draws: np.ndarray):
    """Two-sided p-value from the centred bootstrap distribution, plus percentile 95% CI."""
    p = float(np.mean(np.abs(draws - draws.mean()) >= abs(point)))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return p, lo, hi


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values()
    m = len(order)
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(order.values)])
    return pd.Series(np.minimum(adj, 1), index=order.index).reindex(p.index)


def max_dd(r):
    w = (1 + r).cumprod()
    return float((w / w.cummax() - 1).min())


def horse_race(u, cfg, suffix=""):
    net, gross = read(f"{u}_returns_net{suffix}.csv.gz"), read(f"{u}_returns_gross{suffix}.csv.gz")
    net, gross = net[STRATEGIES].dropna(), gross[STRATEGIES].dropna()
    ppy, block = cfg["ppy"], cfg["block"]
    bs = boot_sharpes(net, ppy, block)
    pt = sharpe(net.to_numpy(), ppy)
    rows = []
    for j, s in enumerate(STRATEGIES):
        d = bs[s] - bs["1/N"]
        p, lo, hi = diff_test(pt[j] - pt[0], d.to_numpy()) if s != "1/N" else (np.nan, np.nan, np.nan)
        rows.append(dict(strategy=s, ann_return=net[s].mean() * ppy, ann_vol=net[s].std() * np.sqrt(ppy),
                         sharpe_net=pt[j], sharpe_gross=sharpe(gross[[s]].to_numpy(), ppy)[0],
                         sharpe_ci_lo=np.quantile(bs[s], 0.025), sharpe_ci_hi=np.quantile(bs[s], 0.975),
                         diff_vs_1N=pt[j] - pt[0], diff_ci_lo=lo, diff_ci_hi=hi, p_vs_1N=p,
                         max_drawdown=max_dd(net[s])))
    out = pd.DataFrame(rows).set_index("strategy")
    out["p_holm"] = holm(out["p_vs_1N"].dropna()).reindex(out.index)
    if not suffix:
        to = read(f"{u}_turnover.csv.gz")[STRATEGIES].iloc[1:]
        reb_per_year = ppy / cfg["step"]
        out["turnover_per_year"] = to.mean() * reb_per_year
        en = read(f"{u}_effective_n.csv.gz")
        out["effective_n"] = en[STRATEGIES].mean()
    return out


def factorial(u, cfg, suffix=""):
    g = read(f"{u}_returns_gross{suffix}.csv.gz")
    cells = [c for c in g.columns if c.startswith("cell|")]
    g = g[cells].dropna()
    ppy, block = cfg["ppy"], cfg["block"]
    bs = boot_sharpes(g, ppy, block)
    pt = pd.Series(sharpe(g.to_numpy(), ppy), index=cells)
    grid = []
    for m in MEAN_LEVELS:
        for c in COV_LEVELS:
            k = f"cell|{m}|{c}"
            grid.append(dict(mean=m, cov=c, sharpe=pt[k], ci_lo=np.quantile(bs[k], 0.025),
                             ci_hi=np.quantile(bs[k], 0.975)))
    grid = pd.DataFrame(grid)

    def eff(a_hi, a_lo):
        d = bs[a_hi] - bs[a_lo]
        p, lo, hi = diff_test(pt[a_hi] - pt[a_lo], d.to_numpy())
        return pt[a_hi] - pt[a_lo], lo, hi, p

    effects = []
    # value of covariance information at each level of mean information
    for m in MEAN_LEVELS:
        for c in ["Ledoit-Wolf", "Clairvoyant"]:
            v, lo, hi, p = eff(f"cell|{m}|{c}", f"cell|{m}|Sample")
            effects.append(dict(effect=f"cov: Sample -> {c}", given=f"mean={m}", value=v, ci_lo=lo, ci_hi=hi, p=p))
    # value of mean information at each level of covariance information
    for c in COV_LEVELS:
        for m in ["None", "Bayes-Stein", "Full-sample", "Clairvoyant"]:
            v, lo, hi, p = eff(f"cell|{m}|{c}", f"cell|Sample|{c}")
            effects.append(dict(effect=f"mean: Sample -> {m}", given=f"cov={c}", value=v, ci_lo=lo, ci_hi=hi, p=p))
    effects = pd.DataFrame(effects)
    effects["p_holm"] = holm(effects["p"])

    # Shapley split of the total value of perfect information (Sample,Sample) -> (Clairv,Clairv)
    SS, CS, SC, CC = (f"cell|{a}|{b}" for a, b in [("Sample", "Sample"), ("Clairvoyant", "Sample"),
                                                  ("Sample", "Clairvoyant"), ("Clairvoyant", "Clairvoyant")])

    def shap(s):
        phi_m = 0.5 * ((s[CS] - s[SS]) + (s[CC] - s[SC]))
        phi_c = 0.5 * ((s[SC] - s[SS]) + (s[CC] - s[CS]))
        return phi_m, phi_c

    pm, pc = shap(pt)
    bm, bc = shap(bs)
    share = bm / (bm + bc)
    sh = dict(phi_mean=pm, phi_mean_lo=np.quantile(bm, .025), phi_mean_hi=np.quantile(bm, .975),
              phi_cov=pc, phi_cov_lo=np.quantile(bc, .025), phi_cov_hi=np.quantile(bc, .975),
              mean_share=pm / (pm + pc), mean_share_lo=np.quantile(share, .025),
              mean_share_hi=np.quantile(share, .975), interaction=pt[CC] - pt[CS] - pt[SC] + pt[SS],
              interaction_lo=np.quantile(bs[CC] - bs[CS] - bs[SC] + bs[SS], .025),
              interaction_hi=np.quantile(bs[CC] - bs[CS] - bs[SC] + bs[SS], .975),
              interaction_p=diff_test(pt[CC] - pt[CS] - pt[SC] + pt[SS],
                                      (bs[CC] - bs[CS] - bs[SC] + bs[SS]).to_numpy())[0],
              start=g.index[0].date(), end=g.index[-1].date(), periods=len(g))
    return grid, effects, sh


def optimism(u, cfg):
    ppy = cfg["ppy"]
    prom = read(f"{u}_promised.csv").iloc[:, 0] * np.sqrt(ppy)
    g = read(f"{u}_returns_gross.csv.gz")
    real = {s: sharpe(g[[s]].dropna().to_numpy(), ppy)[0] for s in STRATEGIES}
    sim = pd.read_csv(RES / f"{u}_sim_paths.csv")
    gap_real = prom.mean() - real["Max-Sharpe (sample)"]
    gap_sim = sim["promised"] - sim["Max-Sharpe (sample)"]
    rel_real = real["Max-Sharpe (sample)"] - real["1/N"]
    rel_sim = sim["Max-Sharpe (sample)"] - sim["1/N"]
    cov_eff_sim = sim["cell|Sample|Clairvoyant"] - sim["cell|Sample|Sample"]
    return dict(promised=prom.mean(), realized=real["Max-Sharpe (sample)"], gap_real=gap_real,
                sim_promised=sim["promised"].mean(), sim_realized=sim["Max-Sharpe (sample)"].mean(),
                gap_sim_mean=gap_sim.mean(), gap_sim_p95=gap_sim.quantile(0.95),
                p_gap=(gap_sim >= gap_real).mean(), nonstationary_share=(gap_real - gap_sim.mean()) / gap_real,
                msr_minus_1N_real=rel_real, msr_minus_1N_sim_mean=rel_sim.mean(),
                p_msr_minus_1N=(rel_sim <= rel_real).mean(),
                sim_cov_info_effect_mean=cov_eff_sim.mean(), sim_cov_info_effect_lo=cov_eff_sim.quantile(.025),
                sim_cov_info_effect_hi=cov_eff_sim.quantile(.975), sim_cov_info_effect_share_neg=(cov_eff_sim < 0).mean(),
                n_paths=len(sim))


def cost_sensitivity(u, cfg):
    g = read(f"{u}_returns_gross.csv.gz")[STRATEGIES].dropna()
    to = read(f"{u}_turnover.csv.gz")[STRATEGIES]
    rows = []
    for bps in [0, 10, 25, 50]:
        net = g.copy()
        idx = to.index.intersection(net.index)
        net.loc[idx] = net.loc[idx] - to.loc[idx] * bps / 1e4
        s = pd.Series(sharpe(net.to_numpy(), cfg["ppy"]), index=STRATEGIES)
        rows.append({"cost_bps": bps, **(s - s["1/N"]).add_suffix(" minus 1/N").to_dict()})
    return pd.DataFrame(rows)


def stress(u):
    net = read(f"{u}_returns_net.csv.gz")[STRATEGIES]
    rows = []
    for name, (a, b) in STRESS[u].items():
        w = net.loc[a:b]
        for s in STRATEGIES:
            rows.append(dict(window=name, strategy=s, cum_return=(1 + w[s]).prod() - 1, max_drawdown=max_dd(w[s])))
    return pd.DataFrame(rows)


def robustness(u, cfg):
    rows = []
    variants = [("main", "", cfg["window"], cfg["horizon"])]
    variants += [(f"W={W}", f"_W{W}", W, cfg["horizon"]) for W in cfg["robust_windows"]]
    variants += [(f"H={H}", f"_H{H}", cfg["window"], H) for H in cfg["robust_horizons"]]
    variants += [("shorts allowed (w >= -0.3)", "_short", cfg["window"], cfg["horizon"])]
    for name, suf, W, H in variants:
        grid, eff, sh = factorial(u, cfg, suf)
        e = eff.set_index(["effect", "given"])
        row = dict(variant=name, window=W, horizon=H, mean_share=sh["mean_share"],
                   cov_clairv_given_sample_mean=e.loc[("cov: Sample -> Clairvoyant", "mean=Sample"), "value"],
                   cov_clairv_given_sample_mean_p=e.loc[("cov: Sample -> Clairvoyant", "mean=Sample"), "p"],
                   cov_clairv_given_no_mean=e.loc[("cov: Sample -> Clairvoyant", "mean=None"), "value"],
                   none_minus_sample_mean=e.loc[("mean: Sample -> None", "cov=Sample"), "value"])
        if suf.startswith("_W") or suf == "":
            hr = horse_race(u, cfg, suf)
            for s in ["Min-var (LW)", "Max-Sharpe (sample)", "Max-Sharpe (resampled)"]:
                row[f"{s} minus 1/N"] = hr.loc[s, "diff_vs_1N"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    summary = {}
    for u, cfg in UNIVERSES.items():
        hr = horse_race(u, cfg)
        hr.to_csv(RES / f"{u}_horse_race.csv", float_format="%.4f")
        grid, eff, sh = factorial(u, cfg)
        grid.to_csv(RES / f"{u}_factorial_grid.csv", index=False, float_format="%.4f")
        eff.to_csv(RES / f"{u}_factorial_effects.csv", index=False, float_format="%.4f")
        op = optimism(u, cfg)
        st = stress(u)
        st.to_csv(RES / f"{u}_stress.csv", index=False, float_format="%.4f")
        cost_sensitivity(u, cfg).to_csv(RES / f"{u}_cost_sensitivity.csv", index=False, float_format="%.4f")
        rb = robustness(u, cfg)
        rb.to_csv(RES / f"{u}_robustness.csv", index=False, float_format="%.4f")
        summary[u] = {**{f"shapley_{k}": v for k, v in sh.items()}, **{f"optimism_{k}": v for k, v in op.items()}}
        print(f"\n===== {u}: {cfg['name']}\n", hr.round(3).to_string(), "\n")
        print(grid.pivot(index="mean", columns="cov", values="sharpe").reindex(MEAN_LEVELS)[COV_LEVELS].round(3))
        print(eff.round(3).to_string())
        print(pd.Series(sh).to_string())
        print(pd.Series(op).round(3).to_string())
        print(rb.round(3).to_string())
    pd.DataFrame(summary).to_csv(RES / "summary.csv")


if __name__ == "__main__":
    main()
