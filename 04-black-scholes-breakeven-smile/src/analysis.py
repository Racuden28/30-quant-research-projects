"""Part II-III: break-even smiles, hypothesis tests, stress windows, VIX comparison, out-of-sample test.

Pre-registered design (fixed before looking at results; see README section 3):
  H1 skew      S = m*(-1.5) - m*(+1.5)                      != 0
  H2 convexity C = (m*(-1.5) + m*(+1.5)) / 2 - m*(0)         >  0
  H3 level     L = m*(0) - 1 (textbook BS with trailing vol is unbiased at the money)
  H4 VRP       SPY ATM options sold at VIX earn a positive hedged P&L, which turns negative in 2008/2020
  H5 OOS       an expanding-window break-even smile, applied to the ex-ante trailing vol, prices future
               options with lower absolute bias than flat trailing-vol Black-Scholes (2012-2026)
Inference: moving-block bootstrap over start dates (block 63 days, B = 1000); Holm across the 7 assets
within each hypothesis family; HAC (Newey-West, 21+ lags) for time-series means.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from breakeven import (ASSETS, M_GRID, RES, X_GRID, Spec, block_bootstrap_counts, build_paths, interp_pnl,
                       load_returns, load_vix, pnl_grid, root_on_grid)

RES.mkdir(exist_ok=True)
B, BLOCK, SEED = 1000, 63, 20261121
XW = 1.5
STRESS = {"GFC (Sep-Dec 2008)": ("2008-09-01", "2008-12-31"),
          "COVID (Feb 15-Apr 30 2020)": ("2020-02-15", "2020-04-30"),
          "2022 rate shock": ("2022-01-01", "2022-12-31")}
OOS_START = "2012-01-01"


def ix(x):
    return int(np.argmin(np.abs(X_GRID - x)))


def stats_from_curve(m):
    s = m[..., ix(-XW)] - m[..., ix(XW)]
    c = 0.5 * (m[..., ix(-XW)] + m[..., ix(XW)]) - m[..., ix(0)]
    lv = m[..., ix(0)] - 1.0
    return s, c, lv


def boot_p(stat_b, two_sided=True):
    """Bootstrap p-value for H0: stat = 0 from the share of replicates on the far side of zero."""
    lo, hi = np.mean(stat_b <= 0), np.mean(stat_b >= 0)
    return float(min(1.0, 2 * min(lo, hi))) if two_sided else float(lo)


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def nw_se(x, lags):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    g = [x @ x / n] + [x[k:] @ x[:-k] / n for k in range(1, lags + 1)]
    var = g[0] + 2 * sum((1 - k / (lags + 1)) * g[k] for k in range(1, lags + 1))
    return float(np.sqrt(var / n))


def pooled_curve(pnl, mask=None):
    p = pnl if mask is None else pnl[mask]
    return root_on_grid(p.mean(0))


def main():
    lr = load_returns()
    vix = load_vix()
    rng = np.random.default_rng(SEED)
    curves, tests, stress_rows, oos_rows, oos_tests, path_rows, ts_rows = [], [], [], [], [], [], []
    for asset in ASSETS:
        dates, sig, logS = build_paths(lr[asset], Spec())
        pnl = pnl_grid(sig, logS, Spec()).astype(np.float32)
        P = len(dates)
        # ---------------- pooled break-even smile + bootstrap ----------------
        m_hat = pooled_curve(pnl)
        C = block_bootstrap_counts(P, B, BLOCK, rng)
        sums = (C @ pnl.reshape(P, -1)).reshape(B, len(X_GRID), len(M_GRID))
        m_b = root_on_grid(sums)
        lo, hi = np.nanpercentile(m_b, [2.5, 97.5], axis=0)
        for k, x in enumerate(X_GRID):
            curves.append(dict(asset=asset, x=x, m_star=m_hat[k], ci_lo=lo[k], ci_hi=hi[k], n_paths=P,
                               start=str(dates[0].date()), end=str(dates[-1].date())))
        s, c, lv = stats_from_curve(m_hat)
        sb, cb, lb = stats_from_curve(m_b)
        for name, v, vb in [("skew", s, sb), ("convexity", c, cb), ("atm_level_minus_1", lv, lb)]:
            tests.append(dict(asset=asset, stat=name, value=float(v), ci_lo=float(np.percentile(vb, 2.5)),
                              ci_hi=float(np.percentile(vb, 97.5)), p_boot=boot_p(vb)))
        # ---------------- per-path break-even vols ----------------
        m_path = root_on_grid(pnl)
        path_df = pd.DataFrame({"date": dates, "asset": asset, "sigma_ref": sig,
                                "m_put15": m_path[:, ix(-XW)], "m_atm": m_path[:, ix(0)], "m_call15": m_path[:, ix(XW)]})
        path_rows.append(path_df)
        # rolling 1-year pooled smile statistics (calendar-year buckets)
        yrs = pd.DatetimeIndex(dates).year
        for y in np.unique(yrs):
            mk = yrs == y
            if mk.sum() < 120:
                continue
            my = pooled_curve(pnl, mk)
            sy, cy, ly = stats_from_curve(my)
            ts_rows.append(dict(asset=asset, year=int(y), m_atm=my[ix(0)], skew=sy, convexity=cy,
                                mean_sigma_ref=float(sig[mk].mean())))
        # ---------------- stress windows ----------------
        for w, (a, b) in STRESS.items():
            mk = (dates >= a) & (dates <= b)
            if mk.sum() < 10:
                continue
            mw = pooled_curve(pnl, mk)
            for k, x in enumerate(X_GRID):
                stress_rows.append(dict(asset=asset, window=w, x=x, m_star=mw[k], n_paths=int(mk.sum())))
        mk = np.ones(P, bool)
        for a, b in STRESS.values():
            mk &= ~((dates >= a) & (dates <= b))
        mw = pooled_curve(pnl, mk)
        for k, x in enumerate(X_GRID):
            stress_rows.append(dict(asset=asset, window="All other start dates", x=x, m_star=mw[k], n_paths=int(mk.sum())))
        # ---------------- out-of-sample smile forecast ----------------
        cum = np.cumsum(pnl.astype(np.float64), axis=0)         # cum[i] = sum of paths 0..i
        oos = np.where(dates >= pd.Timestamp(OOS_START))[0]
        month = pd.DatetimeIndex(dates).to_period("M")
        m_smile = np.full((P, len(X_GRID)), np.nan)
        last_key, cur = None, None
        for i in oos:
            key = month[i]
            if key != last_key:
                j = i - Spec().horizon - 1                         # last path fully observed before t0
                cur = root_on_grid(cum[j] / (j + 1))
                last_key = key
            m_smile[i] = cur
        sel = oos
        models = {
            "Textbook BS (flat trailing vol)": np.ones((len(sel), len(X_GRID))),
            "Flat, level-corrected": np.repeat(m_smile[sel][:, [ix(0)]], len(X_GRID), axis=1),
            "Break-even smile (expanding)": m_smile[sel],
        }
        if asset == "SPY":
            v = vix.reindex(dates[sel]).ffill().values
            okv = np.isfinite(v)
            vix_m = v / sig[sel]
            models["VIX flat"] = np.repeat(vix_m[:, None], len(X_GRID), axis=1)
            shape = m_smile[sel] / m_smile[sel][:, [ix(0)]]
            models["VIX x break-even shape"] = vix_m[:, None] * shape
        else:
            okv = np.ones(len(sel), bool)
        errs = {}
        for name, mq in models.items():
            e = interp_pnl(pnl[sel].astype(np.float64), mq)          # seller's P&L = pricing error
            e = e[okv]
            errs[name] = e
            for k, x in enumerate(X_GRID):
                se = nw_se(e[:, k], 42)
                oos_rows.append(dict(asset=asset, model=name, x=x, mean_pnl=float(e[:, k].mean()), hac_se=se,
                                     rmse=float(np.sqrt(np.mean(e[:, k] ** 2)))))
        # primary OOS metric: mean absolute bias across strikes, block bootstrap of the difference
        base, alt = errs["Textbook BS (flat trailing vol)"], errs["Break-even smile (expanding)"]
        lvl = errs["Flat, level-corrected"]
        n = len(base)
        Cb = block_bootstrap_counts(n, B, BLOCK, rng) / n
        def mab(e, w=None):
            mu = e.mean(0) if w is None else w @ e
            return np.abs(mu).mean(axis=-1)
        d_obs = mab(base) - mab(alt)
        d_b = mab(base, Cb) - mab(alt, Cb)
        d2_obs = mab(lvl) - mab(alt)
        d2_b = mab(lvl, Cb) - mab(alt, Cb)
        # secondary: Diebold-Mariano on squared error averaged over strikes
        dl = (base ** 2).mean(1) - (alt ** 2).mean(1)
        dm = dl.mean() / nw_se(dl, 42)
        oos_tests.append(dict(asset=asset, n_oos=n, mab_textbook=float(mab(base)), mab_level=float(mab(lvl)),
                              mab_smile=float(mab(alt)), gain_vs_textbook=float(d_obs),
                              gain_ci_lo=float(np.percentile(d_b, 2.5)), gain_ci_hi=float(np.percentile(d_b, 97.5)),
                              p_gain=float(np.mean(d_b <= 0)), gain_vs_level=float(d2_obs),
                              p_gain_vs_level=float(np.mean(d2_b <= 0)), dm_stat_mse=float(dm)))
        # ---------------- VIX vs SPY break-even ----------------
        if asset == "SPY":
            v = vix.reindex(dates).ffill()
            ok = v.notna().values
            atm_be = m_path[:, ix(0)] * sig
            pnl_vix_atm = interp_pnl(pnl[:, [ix(0)], :].astype(np.float64), (v.values / sig)[:, None])[:, 0]
            vdf = pd.DataFrame({"date": dates, "vix": v.values, "sigma_ref": sig, "atm_breakeven_vol": atm_be,
                                "pnl_sell_atm_at_vix": pnl_vix_atm})[ok]
            vdf.to_csv(RES / "spy_vix_vs_breakeven.csv", index=False)
            rows = []
            for w, (a, b) in {"Full sample": ("2007-01-01", "2026-12-31"), **STRESS}.items():
                g = vdf[(vdf.date >= a) & (vdf.date <= b)]
                rows.append(dict(window=w, n=len(g), mean_vix=g.vix.mean(), mean_atm_breakeven=g.atm_breakeven_vol.mean(),
                                 mean_gap_vol_pts=100 * (g.vix - g.atm_breakeven_vol).mean(),
                                 share_vix_above=(g.vix > g.atm_breakeven_vol).mean(),
                                 mean_pnl_sell_at_vix=g.pnl_sell_atm_at_vix.mean(),
                                 hac_se=nw_se(g.pnl_sell_atm_at_vix.values, 42) if len(g) > 50 else np.nan))
            pd.DataFrame(rows).to_csv(RES / "spy_vix_summary.csv", index=False)
        print(asset, "done", P, "paths; m*(-1.5,0,1.5) =", np.round(m_hat[[ix(-XW), ix(0), ix(XW)]], 3))
    curves = pd.DataFrame(curves)
    curves.to_csv(RES / "breakeven_smiles.csv", index=False)
    tests = pd.DataFrame(tests)
    tests["p_holm"] = tests.groupby("stat").p_boot.transform(lambda p: holm(p.values))
    tests.to_csv(RES / "hypothesis_tests.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(RES / "stress_smiles.csv", index=False)
    pd.concat(path_rows).to_csv(RES / "per_path_breakeven.csv.gz", index=False)
    pd.DataFrame(ts_rows).to_csv(RES / "yearly_smile_stats.csv", index=False)
    pd.DataFrame(oos_rows).to_csv(RES / "oos_bias_by_strike.csv", index=False)
    ot = pd.DataFrame(oos_tests)
    ot["p_holm"] = holm(ot.p_gain.values)
    ot.to_csv(RES / "oos_tests.csv", index=False)
    print(tests.to_string())
    print(ot.to_string())


if __name__ == "__main__":
    main()
