"""
Part A. Do correlations really go to one in a crash?

For each universe and each conditioning rule, compare the observed rise in average pairwise correlation
against two counterfactual worlds in which the latent correlation never changes:

  * Gaussian null:  r_t = sigma_t * z_t,  z_t ~ N(0, R)          (only volatility moves)
  * Student-t null: r_t = sigma_t * z_t,  z_t ~ t_nu(0, R)        (plus symmetric tail dependence)

sigma_t is the *actual* ex-ante EWMA volatility path of every asset, R is the full-sample correlation of
devolatised returns, nu is fitted by profile likelihood. Each null is simulated B times, and every
conditioning rule is re-applied to the simulated market, exactly as on the real data.

Decomposition of the observed gap  G = rho(condition) - rho(all days):
    selection artefact    = E[G | Gaussian null]
    tail dependence       = E[G | t null] - E[G | Gaussian null]
    excess (genuine)      = G_observed - E[G | t null]

Outputs in results/: gap_decomposition.csv, null_draws.csv.gz, exceedance_curve.csv,
pair_excess_<universe>.csv, conditioning_meta.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import (corr, dcc_filter, dcc_fit, q_to_r, ewma_vol, fisher_mean, fit_mvt_nu, mvt_unit, offdiag, stationary_bootstrap_idx,
                  bh)
from data import load

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

B_NULL = 1000         # simulations per null
B_DCC = 500          # simulations of the regime (DCC) null
B_BOOT = 1000         # stationary-bootstrap replications of the observed statistic
BLOCK = 21
WARM = 63
SEED = 20261121
EXC_Q = np.round(np.r_[np.arange(0.05, 0.50, 0.05), np.arange(0.55, 0.96, 0.05)], 2)

# Conditioning rules (pre-registered). q is the market-return quantile; "ante" uses the EWMA vol of
# the market known the evening before.
RULES = {
    "crash_5": "Market return in its bottom 5%",
    "rally_5": "Market return in its top 5%",
    "big_move_10": "|Market return| in its top 10%",
    "highvol_ante_10": "Ex-ante market volatility in its top decile",
    "calm_ante_50": "Ex-ante market volatility below its median",
}


def market_of(r: np.ndarray, universe: str, cols: list[str]) -> np.ndarray:
    if universe == "etfs":
        return r[:, cols.index("SPY")]
    return r.mean(axis=1)  # equal-weight portfolio of the 19 stocks


def masks(m: np.ndarray, m_sig: np.ndarray) -> dict[str, np.ndarray]:
    q05, q95 = np.quantile(m, [0.05, 0.95])
    a = np.abs(m)
    return {
        "crash_5": m <= q05,
        "rally_5": m >= q95,
        "big_move_10": a >= np.quantile(a, 0.90),
        "highvol_ante_10": m_sig >= np.quantile(m_sig, 0.90),
        "calm_ante_50": m_sig <= np.quantile(m_sig, 0.50),
    }


def exceedance(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Average correlation conditional on the market below (q<.5) or above (q>.5) its q-quantile."""
    out = []
    for q in EXC_Q:
        thr = np.quantile(m, q)
        mk = m <= thr if q < 0.5 else m >= thr
        out.append(fisher_mean(corr(x[mk])))
    return np.array(out)


def blocks_of(universe: str, cols: list[str]) -> dict[str, list[int]]:
    """Which average correlations to test. For the ETFs, a single average mixes risk assets and hedges
    whose correlations move in opposite directions, so the risk block and the two hedge pairs are tested
    separately as well (pre-registered)."""
    if universe == "stocks":
        return {"all": list(range(len(cols)))}
    ix = {c: i for i, c in enumerate(cols)}
    return {"all": list(range(len(cols))), "risk": [ix[c] for c in ["SPY", "EEM", "HYG", "USO"]],
            "SPY-TLT": [ix["SPY"], ix["TLT"]], "SPY-GLD": [ix["SPY"], ix["GLD"]]}


def stats_for(r: np.ndarray, sig: np.ndarray, universe: str, cols: list[str], m_sig_fixed=None):
    """Gap statistics for every (block, rule), on raw returns. m_sig: ex-ante market vol."""
    m = market_of(r, universe, cols)
    m_sig = ewma_vol(m[:, None])[:, 0] if m_sig_fixed is None else m_sig_fixed
    mk = masks(m, m_sig)
    blocks = blocks_of(universe, cols)
    C0 = corr(r)
    Cs = {k: corr(r[v]) for k, v in mk.items()}
    base = fisher_mean(C0)
    gaps = {k: fisher_mean(Cs[k]) - base for k in mk}
    for bname, ix in blocks.items():
        if bname == "all":
            continue
        b0 = fisher_mean(C0, ix)
        for k in mk:
            gaps[f"{k}@{bname}"] = fisher_mean(Cs[k], ix) - b0
    return base, gaps, m, mk


def pair_gaps(r: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return offdiag(corr(r[mask])) - offdiag(corr(r))


def run_universe(universe: str, rng) -> tuple[list[dict], dict, pd.DataFrame, list[dict]]:
    df = load(universe)
    cols = list(df.columns)
    r_all = df.values
    sig_all = ewma_vol(r_all)
    r, sig = r_all[WARM:], sig_all[WARM:]
    dates = df.index[WARM:]
    z = r / sig
    R = corr(z)
    nu = fit_mvt_nu(z, R)
    T, N = r.shape

    base, gaps_obs, m, mk_obs = stats_for(r, sig, universe, cols)
    zbase = fisher_mean(corr(z))
    gaps_z = {k: fisher_mean(corr(z[v])) - zbase for k, v in mk_obs.items()}
    exc_obs = exceedance(r, m)
    pg_obs = pair_gaps(r, mk_obs["crash_5"])

    # ---- nulls
    draws, exc_null, pg_null = [], {"gauss": [], "t": []}, []
    for null, nn in [("gauss", None), ("t", nu)]:
        for b in range(B_NULL):
            zs = mvt_unit(rng, R, T, nn)
            rs = sig * zs
            _, g, ms, mks = stats_for(rs, sig, universe, cols)
            draws.append({"universe": universe, "null": null, "b": b, **g})
            if b < 400:
                exc_null[null].append(exceedance(rs, ms))
            if null == "t":
                pg_null.append(pair_gaps(rs, mks["crash_5"]))
    # ---- regime null: latent correlation follows the fitted DCC path (slow regime moves, no crash-day
    # specific coupling). Same t shocks, same vol path. Anything left over is crash-specific.
    (dcc_a, dcc_b), Qbar = dcc_fit(z)
    Lpath = np.linalg.cholesky(q_to_r(dcc_filter(z, dcc_a, dcc_b, Qbar)[:-1]))
    for b in range(B_DCC):
        g = np.einsum("tij,tj->ti", Lpath, rng.standard_normal((T, N)))
        w = rng.chisquare(nu, size=(T, 1)) / nu
        rs = sig * g / np.sqrt(w) * np.sqrt((nu - 2) / nu)
        _, gd, _, _ = stats_for(rs, sig, universe, cols)
        draws.append({"universe": universe, "null": "dcc", "b": b, **gd})
    draws = pd.DataFrame(draws)
    pg_null = np.array(pg_null)

    # ---- stationary bootstrap of the observed gaps (re-sampling days, rules recomputed)
    boot = {k: [] for k in gaps_obs}
    m_sig_obs = ewma_vol(market_of(r_all, universe, cols)[:, None])[WARM:, 0]
    for b in range(B_BOOT):
        idx = stationary_bootstrap_idx(rng, T, BLOCK)
        _, g, _, _ = stats_for(r[idx], sig[idx], universe, cols, m_sig_fixed=m_sig_obs[idx])
        for k in gaps_obs:
            boot[k].append(g[k])

    rows = []
    for key in gaps_obs:
        k, _, block = key.partition("@")
        block = block or "all"
        desc = RULES[k]
        gn = draws.loc[draws.null == "gauss", key].values
        gt = draws.loc[draws.null == "t", key].values
        gd_ = draws.loc[draws.null == "dcc", key].values
        lo, hi = np.quantile(boot[key], [0.025, 0.975])
        obs = gaps_obs[key]
        rows.append(dict(
            universe=universe, block=block, rule=k, description=desc, n_days=int(mk_obs[k].sum()),
            rho_all=base if block == "all" else np.nan, gap_observed=obs, gap_ci_lo=lo, gap_ci_hi=hi,
            gap_devolatised=gaps_z.get(key, np.nan),
            null_gauss_mean=gn.mean(), null_gauss_p05=np.quantile(gn, 0.05), null_gauss_p95=np.quantile(gn, 0.95),
            null_t_mean=gt.mean(), null_t_p025=np.quantile(gt, 0.025), null_t_p975=np.quantile(gt, 0.975),
            selection_artefact=gn.mean(), tail_dependence=gt.mean() - gn.mean(), excess=obs - gt.mean(),
            excess_ci_lo=lo - gt.mean(), excess_ci_hi=hi - gt.mean(),
            p_upper_vs_t=(1 + (gt >= obs).sum()) / (1 + len(gt)),
            p_lower_vs_t=(1 + (gt <= obs).sum()) / (1 + len(gt)),
            share_explained_by_nulls=gt.mean() / obs if abs(obs) > 1e-9 else np.nan,
            null_dcc_mean=gd_.mean(), null_dcc_p025=np.quantile(gd_, 0.025), null_dcc_p975=np.quantile(gd_, 0.975),
            regime_component=gd_.mean() - gt.mean(), condition_specific=obs - gd_.mean(),
            p_upper_vs_dcc=(1 + (gd_ >= obs).sum()) / (1 + len(gd_)),
            p_lower_vs_dcc=(1 + (gd_ <= obs).sum()) / (1 + len(gd_)),
        ))

    # asymmetry: down-10% minus up-10% exceedance correlation
    i_d, i_u = list(EXC_Q).index(0.10), list(EXC_Q).index(0.90)
    asym_obs = exc_obs[i_d] - exc_obs[i_u]
    asym_null = np.array(exc_null["t"])[:, i_d] - np.array(exc_null["t"])[:, i_u]
    exc_rows = []
    for j, q in enumerate(EXC_Q):
        en = np.array(exc_null["t"])[:, j]
        eg = np.array(exc_null["gauss"])[:, j]
        exc_rows.append(dict(universe=universe, q=q, observed=exc_obs[j], t_mean=en.mean(),
                             t_p05=np.quantile(en, 0.05), t_p95=np.quantile(en, 0.95), gauss_mean=eg.mean()))

    # pair-level crash-day excess, BH-controlled
    p_pair = (1 + (pg_null >= pg_obs).sum(0)) / (1 + len(pg_null))
    q_pair = bh(p_pair)
    iu = np.triu_indices(N, 1)
    pairs = pd.DataFrame(dict(universe=universe, a=np.array(cols)[iu[0]], b=np.array(cols)[iu[1]],
                              rho_all=offdiag(corr(r)), rho_crash=offdiag(corr(r[mk_obs["crash_5"]])),
                              gap=pg_obs, null_t_mean=pg_null.mean(0), excess=pg_obs - pg_null.mean(0),
                              p=p_pair, q_bh=q_pair))

    meta = dict(universe=universe, T=T, N=N, dcc_a=dcc_a, dcc_b=dcc_b, start=str(dates[0].date()), end=str(dates[-1].date()), nu=nu,
                rho_all=base, rho_all_devolatised=zbase,
                asym_down10_minus_up10=asym_obs, asym_null_mean=float(asym_null.mean()),
                asym_null_p95=float(np.quantile(asym_null, 0.95)),
                asym_p=float((1 + (asym_null >= asym_obs).sum()) / (1 + len(asym_null))),
                n_pairs=int(len(pairs)), n_pairs_bh05=int((q_pair < 0.05).sum()),
                n_pairs_excess_negative=int((pairs.excess < 0).sum()))
    draws.to_csv(RES / f"null_draws_{universe}.csv.gz", index=False, float_format="%.5f")
    return rows, meta, pairs, exc_rows


def subperiods(rng):
    """Robustness: the stock decomposition in 1990-2006 and 2007-2022 separately (t null, fewer draws)."""
    df = load("stocks")
    cols = list(df.columns)
    out = []
    sig_all = ewma_vol(df.values)
    for lab, a, b in [("1990-2006", "1990-01-01", "2006-12-31"), ("2007-2022", "2007-01-01", "2022-12-31")]:
        sel = (df.index >= a) & (df.index <= b)
        sel[:WARM] = False
        r, sig = df.values[sel], sig_all[sel]
        z = r / sig
        R = corr(z)
        nu = fit_mvt_nu(z, R)
        _, g_obs, _, _ = stats_for(r, sig, "stocks", cols)
        gt = {k: [] for k in RULES}
        gg = {k: [] for k in RULES}
        for bb in range(300):
            for null, nn, store in [("t", nu, gt), ("gauss", None, gg)]:
                _, g, _, _ = stats_for(sig * mvt_unit(rng, R, len(r), nn), sig, "stocks", cols)
                for k in RULES:
                    store[k].append(g[k])
        for k in RULES:
            t_arr = np.array(gt[k])
            out.append(dict(period=lab, rule=k, nu=nu, gap_observed=g_obs[k], null_gauss_mean=np.mean(gg[k]),
                            null_t_mean=t_arr.mean(), excess=g_obs[k] - t_arr.mean(),
                            p_upper_vs_t=(1 + (t_arr >= g_obs[k]).sum()) / (1 + len(t_arr))))
    return pd.DataFrame(out)


def main():
    rng = np.random.default_rng(SEED)
    rows, metas, pairs, excs = [], [], [], []
    for u in ["stocks", "etfs"]:
        a, b, c, d = run_universe(u, rng)
        rows += a
        metas.append(b)
        pairs.append(c)
        excs += d
        print(u, "done", b)
    pd.DataFrame(rows).to_csv(RES / "gap_decomposition.csv", index=False, float_format="%.5f")
    pd.DataFrame(excs).to_csv(RES / "exceedance_curve.csv", index=False, float_format="%.5f")
    pd.concat(pairs).to_csv(RES / "pair_crash_excess.csv", index=False, float_format="%.5f")
    subperiods(rng).to_csv(RES / "subperiod_decomposition.csv", index=False, float_format="%.5f")
    (RES / "conditioning_meta.json").write_text(json.dumps(metas, indent=2, default=float))
    print(pd.DataFrame(rows)[["universe", "block", "rule", "gap_observed", "selection_artefact", "tail_dependence",
                              "excess", "p_upper_vs_t", "regime_component", "condition_specific", "p_upper_vs_dcc", "p_lower_vs_dcc"]].round(3).to_string())


if __name__ == "__main__":
    main()
