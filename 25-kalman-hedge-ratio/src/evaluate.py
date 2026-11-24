"""
Every number quoted in README.md: hedging tables, the project-09 comparison, tuning curves, stress windows,
the pre-registered tests, and the pairs-trading results against the placebo. Writes results/*.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from data import RES, p09_panel
from hedging import EST
from kalman import matched_halflife
from stats import block_boot_ci, holm, nw_t

LABEL = {"KF": "Kalman, noise ratio by likelihood", "EW-match": "EWMA, same memory as the Kalman filter",
         "EW-var": "EWMA, same memory, market-variance clock", "EW-vol": "EWMA, same memory, |market| clock",
         "KF-HR": "Kalman, time-varying noise (R_t)", "KF-SSE": "Kalman, memory chosen by past hedging loss",
         "EW-SSE": "EWMA, memory chosen by past hedging loss", "EWMA-126": "EWMA, half-life 126 days (project 09)",
         "OLS-252": "Rolling OLS, 252 days (project 09)", "OLS-63": "Rolling OLS, 63 days",
         "Static": "Expanding OLS (static hedge ratio)"}
STRESS = {"industries": {"1929-32 crash": ("1929-10", "1932-06"), "1987 crash": ("1987-10", "1987-12"),
                         "2000-02 dot-com": ("2000-03", "2002-10"), "2008-09 GFC": ("2008-09", "2009-03"),
                         "2020 Covid": ("2020-02", "2020-04"), "2022 inflation": ("2022-01", "2022-10")},
          "countries": {"2008-09 GFC": ("2008-09", "2009-03"), "2011 euro crisis": ("2011-07", "2011-12"),
                        "2020 Covid": ("2020-02", "2020-04"), "2022 inflation": ("2022-01", "2022-10")}}
BOOT = dict(mean_block=12, draws=2000, seed=25)


# ==========================================================================================================
# Hedging
# ==========================================================================================================
class Hedge:
    def __init__(self, u):
        z = np.load(RES / f"hedge_{u}.npz", allow_pickle=True)
        self.u = u
        self.z = z
        self.months = pd.PeriodIndex(z["months"], freq="M")
        self.assets = list(z["assets"])
        ev = pd.Period(str(z["eval_start"]), "M")
        n = z["n"]
        ok = (n >= 15) & (self.months >= ev)[:, None]
        if u == "industries":
            ok &= (self.months <= pd.Period("2026-09", "M"))[:, None]
        for p in "DM":
            for k in EST:
                ok &= ~np.isnan(z[f"{p}|{k}"])
        self.ok = ok
        self.n = np.where(ok, n, np.nan)
        orc = z["syy"] - z["sxy"] ** 2 / z["sxx"]
        self.oracle = self.cs(orc)
        self.mvar = pd.Series(np.nanmean(np.where(ok, z["sxx"] / n, np.nan), 1), index=self.months)
        keep = self.oracle.notna()
        self.keep = keep.values
        self.oracle = self.oracle[keep]
        self.mvar = self.mvar[keep]

    def cs(self, s):
        """Cross-sectional average of the per-asset mean squared error, one value per month."""
        return pd.Series(np.nanmean(np.where(self.ok, s / self.z["n"], np.nan), 1), index=self.months)

    def loss(self, key):
        return self.cs(self.z[key])[self.keep]

    def turnover(self, k):
        to = np.where(self.ok, self.z[f"TO|{k}"], 0.0)
        return to.sum() / self.ok.sum() * 12      # sum |d beta| per asset-year


def hedge_tables():
    out, tests, curves, stress, conc = [], [], [], [], []
    for u in ("industries", "countries"):
        H = Hedge(u)
        rh = pd.read_csv(RES / f"hedge_rhat_{u}.csv")
        hl_med = {"KF": rh.halflife_days.median(), "KF-SSE": rh.halflife_sse.median(),
                  "EW-SSE": rh.halflife_ew_sse.median(), "EW-match": rh.halflife_days.median(),
                  "EW-var": rh.halflife_days.median(), "EW-vol": rh.halflife_days.median(), "EWMA-126": 126.0}
        base = H.loss("D|EWMA-126")
        Lkf = H.loss("D|KF")
        for p in "DM":
            for k in EST:
                L = H.loss(f"{p}|{k}")
                d = L - base
                m, t = nw_t(d.values, 6)
                lo, hi = block_boot_ci(d.values / base.mean(), **BOOT)
                out.append(dict(universe=u, protocol=p, estimator=k, label=LABEL[k],
                                loss_pct2_per_day=L.mean() * 1e4,
                                vs_ewma126_pct=100 * (L.sum() / base.sum() - 1), ci_lo_pct=100 * lo, ci_hi_pct=100 * hi,
                                dm_t=t, vs_kalman_pct=100 * (L.sum() / Lkf.sum() - 1),
                                beta_turnover_per_year=H.turnover(k), median_halflife_days=hl_med.get(k, np.nan),
                                months=len(L), asset_months=int(H.ok[H.keep].sum())))
        # H1: equivalence of Kalman and matched EWMA (TOST at +-1 % of the loss)
        d = (H.loss("D|KF") - H.loss("D|EW-match")).values
        m, t = nw_t(d, 6)
        se = abs(m / t)
        margin = 0.01 * H.loss("D|EW-match").mean()
        p_tost = max(1 - norm.cdf((m + margin) / se), norm.cdf((m - margin) / se))
        tests.append(dict(test=f"H1 {u}: Kalman within 1% of matched EWMA (TOST)",
                          estimate_pct=100 * m / H.loss("D|EW-match").mean(),
                          ci_lo_pct=100 * (m - 1.96 * se) / H.loss("D|EW-match").mean(),
                          ci_hi_pct=100 * (m + 1.96 * se) / H.loss("D|EW-match").mean(), t=t, p=p_tost))
        # H3: daily updating beats month-end updating for the Kalman beta
        d = (H.loss("D|KF") - H.loss("M|KF")).values
        m, t = nw_t(d, 6)
        tests.append(dict(test=f"H3 {u}: daily-updated Kalman beats month-end Kalman",
                          estimate_pct=100 * m / H.loss("M|KF").mean(),
                          ci_lo_pct=100 * (m - 1.96 * abs(m / t)) / H.loss("M|KF").mean(),
                          ci_hi_pct=100 * (m + 1.96 * abs(m / t)) / H.loss("M|KF").mean(), t=t, p=norm.cdf(t)))
        # H2: concentration of the Kalman vs matched-EWMA difference and of Kalman's loss to EWMA-126
        top = H.mvar >= H.mvar.quantile(0.9)
        for a, b in (("KF", "EW-match"), ("KF", "EWMA-126"), ("KF-SSE", "EWMA-126")):
            dd = H.loss(f"D|{a}") - H.loss(f"D|{b}")
            conc.append(dict(universe=u, comparison=f"{a} - {b}", total=dd.sum(),
                             share_top10pct_var_months=dd[top].sum() / dd.sum(),
                             share_months_a_better=float((dd < 0).mean())))
        # tuning curves on the common sample
        okk = H.ok[H.keep]
        nn = np.nansum(np.where(okk, H.z["n"][H.keep], 0))
        s2 = np.nansum(np.where(okk, H.z["sxx"][H.keep], 0)) / nn
        hl = matched_halflife(H.z["rgrid"], s2)
        ck = np.nansum(np.where(okk[:, :, None], H.z["curve_kf"][H.keep], 0), (0, 1)) / nn
        ce = np.nansum(np.where(okk[:, :, None], H.z["curve_ew"][H.keep], 0), (0, 1)) / nn
        for g in range(len(hl)):
            curves.append(dict(universe=u, r=H.z["rgrid"][g], halflife_days=hl[g], loss_kalman=ck[g] * 1e4,
                               loss_ewma=ce[g] * 1e4))
        # stress windows
        for name, (a, b) in STRESS[u].items():
            sl = (H.months[H.keep] >= pd.Period(a, "M")) & (H.months[H.keep] <= pd.Period(b, "M"))
            row = dict(universe=u, window=name, months=int(sl.sum()))
            for k in ("KF", "EW-match", "KF-HR", "KF-SSE", "EW-SSE", "OLS-252", "OLS-63", "Static"):
                row[f"{k}_vs_EWMA126_pct"] = 100 * (H.loss(f"D|{k}")[sl].sum() / base[sl].sum() - 1)
            row["M_KF_vs_D_KF_pct"] = 100 * (H.loss("M|KF")[sl].sum() / H.loss("D|KF")[sl].sum() - 1)
            stress.append(row)
    return (pd.DataFrame(out), pd.DataFrame(tests), pd.DataFrame(curves), pd.DataFrame(stress), pd.DataFrame(conc))


def hedge_buckets():
    """Kalman against its matched EWMA (and EWMA-126) in the most volatile 10 % of months and in the rest."""
    rows = []
    for u in ("industries", "countries"):
        H = Hedge(u)
        top = (H.mvar >= H.mvar.quantile(0.9)).values
        for b, sl in (("most volatile 10% of months", top), ("other 90% of months", ~top)):
            kf, em, e126 = H.loss("D|KF")[sl], H.loss("D|EW-match")[sl], H.loss("D|EWMA-126")[sl]
            d = (kf - em).values
            m, t = nw_t(d, 6)
            rows.append(dict(universe=u, bucket=b, months=int(sl.sum()), kf_vs_ewmatch_pct=100 * (kf.sum() / em.sum() - 1),
                             t=t, share_months_kf_better=float((d < 0).mean()),
                             kf_vs_ewma126_pct=100 * (kf.sum() / e126.sum() - 1)))
    return pd.DataFrame(rows)


def halflife_by_decade():
    rows = []
    for u in ("industries", "countries"):
        r = pd.read_csv(RES / f"hedge_rhat_{u}.csv").replace(np.inf, np.nan)
        r["decade"] = pd.to_datetime(r.date).dt.year // 10 * 10
        g = r.groupby("decade")[["halflife_days", "halflife_sse", "halflife_ew_sse"]].median()
        g["universe"] = u
        rows.append(g.reset_index())
        a = r[["halflife_days", "halflife_sse", "halflife_ew_sse"]].median().to_dict()
        rows.append(pd.DataFrame([dict(universe=u, decade="all", **a)]))
    return pd.concat(rows, ignore_index=True)


def p09_comparison():
    """Project 09's table extended with the Kalman beta under project 09's own protocol and scoring."""
    H = Hedge("industries")
    p = p09_panel()
    p["eval"] = (pd.PeriodIndex(p["month"], freq="M") + 1).astype(str)
    idx = {m: i for i, m in enumerate(H.months.astype(str))}
    ai = {a: j for j, a in enumerate(H.assets)}
    ii = p["eval"].map(idx)
    jj = p["asset"].map(ai)
    good = ii.notna() & jj.notna() & p["d_n"].notna()
    p = p[good].copy()
    ii, jj = ii[good].astype(int).values, jj[good].astype(int).values
    for k in ("KF", "KF-SSE", "EWMA-126", "OLS-252"):
        p[f"mine_M|{k}"] = H.z[f"M|{k}"][ii, jj] / H.z["n"][ii, jj]
        p[f"mine_D|{k}"] = H.z[f"D|{k}"][ii, jj] / H.z["n"][ii, jj]
    check = np.nanmax(np.abs(H.z["sxx"][ii, jj] - p["d_sxx"].values))
    n = p["d_n"]
    est9 = ["OLS-5Y-M", "OLS-1Y-D", "EWMA", "Welch-SWA", "Vasicek", "Combination"]
    for k in est9:
        b = p[f"beta_{k}"]
        p[f"p09|{k}"] = (p["d_syy"] - 2 * b * p["d_sxy"] + b * b * p["d_sxx"]) / n
    p["oracle"] = (p["d_syy"] - p["d_sxy"] ** 2 / p["d_sxx"]) / n
    cols = [f"p09|{k}" for k in est9] + [c for c in p.columns if c.startswith("mine_")]
    p = p.dropna(subset=cols + ["oracle"])
    g = p.groupby("eval")[cols + ["oracle"]].mean()
    ex = g[cols].sub(g["oracle"], axis=0)
    text = ex["p09|OLS-5Y-M"]
    rows = []
    for c in cols:
        d = (g[c] - g["p09|OLS-5Y-M"]).values
        m, t = nw_t(d, 6)
        rows.append(dict(estimator=c, loss_pct2_per_day=g[c].mean() * 1e4, excess_loss_ratio=ex[c].sum() / text.sum(),
                         dm_t_vs_textbook=t))
    df = pd.DataFrame(rows)
    # stationary-bootstrap CI for the ratio of sums (months resampled in blocks, mean length 12)
    rng = np.random.default_rng(25)
    T = len(ex)
    ci = {c: [] for c in cols}
    for _ in range(1000):
        idx_ = np.empty(T, int)
        idx_[0] = rng.integers(T)
        new = rng.random(T) < 1 / 12
        st = rng.integers(0, T, T)
        for t in range(1, T):
            idx_[t] = st[t] if new[t] else (idx_[t - 1] + 1) % T
        den = text.values[idx_].sum()
        for c in cols:
            ci[c].append(ex[c].values[idx_].sum() / den)
    df["ci_lo"] = [np.quantile(ci[c], 0.025) for c in cols]
    df["ci_hi"] = [np.quantile(ci[c], 0.975) for c in cols]
    df["common_months"] = T
    df["industry_months"] = len(p)
    df["max_abs_diff_sxx_vs_p09"] = check
    return df


# ==========================================================================================================
# Pairs
# ==========================================================================================================
RULE_LABEL = {"S": "Static OLS hedge ratio (formation)", "R": "Rolling 63-day OLS hedge ratio",
              "K": "Kalman hedge ratio (likelihood on formation)", "K-rebal": "Kalman, legs rebalanced daily",
              "K-chan": "Kalman, Chan's fixed settings", "K-chan-1sd": "Kalman, Chan's settings and 1-sd entry",
              "R-rebal": "Rolling OLS, legs rebalanced daily", "K-wait5": "Kalman, traded 5 days after the signal",
              "REV1": "One-day reversal of the pair's relative return (benchmark)"}
PERIODS = {"countries": ("2002-01", "2026-11"), "industries": ("1927-07", "2026-09")}


class Pairs:
    def __init__(self, u):
        z = np.load(RES / f"pairs_{u}.npz")
        self.z = z
        self.dates = pd.to_datetime(z["dates"], unit="D")
        self.starts = z["starts"]
        self.valid = z["nassets"] >= 10

    def daily(self, src, sel, rule, cost_bp=0.0, rep=None):
        pnl = self.z[f"pnl|{src}|{sel}|{rule}"].astype(float)
        turn = self.z[f"turn|{src}|{sel}|{rule}"].astype(float)
        x = pnl - cost_bp * 1e-4 * turn
        x = np.nanmean(x, 1) if rep is None else x[:, rep]
        T = len(self.dates)
        tot, cnt = np.zeros(T), np.zeros(T)
        for k, s in enumerate(self.starts):
            if not self.valid[k] or np.isnan(x[k]).all():
                continue
            tot[s:s + len(x[k])] += np.nan_to_num(x[k])
            cnt[s:s + len(x[k])] += 1
        return pd.Series(np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), index=self.dates).dropna()

    def monthly(self, *a, **k):
        d = self.daily(*a, **k)
        return (1 + d).groupby(d.index.to_period("M")).prod() - 1

    def stats(self, src, sel, rule):
        s = self.z[f"stats|{src}|{sel}|{rule}"]
        s = s[self.valid]
        tot = np.nansum(s.reshape(-1, s.shape[-1]), 0)
        nwin = np.isfinite(s[..., 0]).sum()
        return dict(trades_per_pair_per_window=tot[0] / nwin / 20, convergence_share=tot[1] / tot[0],
                    price_share_of_closure=tot[6] / tot[5], days_open_share=tot[4] / nwin / 20 / 126,
                    skipped_negative_beta=tot[8] / nwin)


def pairs_tables():
    rows, tests = [], []
    for u in ("countries", "industries"):
        P = Pairs(u)
        a, b = PERIODS[u]
        for sel in ("dist", "eg"):
            for rule in RULE_LABEL:
                for cost in (0.0, 5.0):
                    real = P.monthly("real", sel, rule, cost)
                    pl = P.monthly("N1", sel, rule, cost)
                    real = real[(real.index >= pd.Period(a, "M")) & (real.index <= pd.Period(b, "M"))]
                    pl = pl.reindex(real.index)
                    edge = real - pl
                    m, t = nw_t(real.values, 6)
                    me, te = nw_t(edge.values, 6)
                    lo, hi = block_boot_ci(edge.values, **BOOT)
                    row = dict(universe=u, selection=sel, rule=rule, label=RULE_LABEL[rule], cost_bp=cost,
                               months=len(real), real_pct_yr=1200 * m, real_t=t, placebo_pct_yr=1200 * pl.mean(),
                               edge_pct_yr=1200 * me, edge_t=te, edge_ci_lo=1200 * lo, edge_ci_hi=1200 * hi)
                    if cost == 0:
                        sr, sp = P.stats("real", sel, rule), P.stats("N1", sel, rule)
                        row.update({f"real_{k}": v for k, v in sr.items()})
                        row.update({f"placebo_{k}": v for k, v in sp.items()})
                        turn = P.monthly("real", sel, rule, 0.0) - P.monthly("real", sel, rule, 1.0)
                        row["cost_per_bp_pct_yr"] = 1200 * turn.mean()
                    rows.append(row)
    df = pd.DataFrame(rows)
    # H6: Kalman vs static edge, countries, distance pairs, gross
    P = Pairs("countries")
    a, b = PERIODS["countries"]
    e = {}
    for rule in ("K", "S"):
        r_ = P.monthly("real", "dist", rule)
        p_ = P.monthly("N1", "dist", rule).reindex(r_.index)
        e[rule] = (r_ - p_)[(r_.index >= pd.Period(a, "M"))]
    d = (e["K"] - e["S"]).dropna()
    m, t = nw_t(d.values, 6)
    tests.append(dict(test="H6 countries: Kalman edge over placebo larger than static edge",
                      estimate_pct=1200 * m, ci_lo_pct=1200 * (m - 1.96 * abs(m / t)),
                      ci_hi_pct=1200 * (m + 1.96 * abs(m / t)), t=t, p=1 - norm.cdf(t)))
    adf = []
    for u in ("countries", "industries"):
        A = pd.read_csv(RES / f"pairs_adf_{u}.csv")
        for src, g in A.groupby("source"):
            adf.append(dict(universe=u, source=src, eg_residual_sig_share=g.eg_sig_share.mean(),
                            kalman_innovation_sig_share=g.kalman_innov_sig_share.mean(),
                            ra_median=g.ra_median.median(), rb_median=g.rb_median.median(),
                            share_pairs_ra_gt_0_1=g["share_ra_gt_0.1"].mean()))
    return df, pd.DataFrame(tests), pd.DataFrame(adf)


def subperiod_pairs():
    rows = []
    P = Pairs("countries")
    for name, (a, b) in {"2002-2009": ("2002-01", "2009-12"), "2010-2026": ("2010-01", "2026-11")}.items():
        for rule in ("S", "R", "K", "K-rebal"):
            for cost in (0.0, 5.0):
                r_ = P.monthly("real", "dist", rule, cost)
                p_ = P.monthly("N1", "dist", rule, cost).reindex(r_.index)
                sl = (r_.index >= pd.Period(a, "M")) & (r_.index <= pd.Period(b, "M"))
                m, t = nw_t(r_[sl].values, 6)
                me, te = nw_t((r_ - p_)[sl].values, 6)
                rows.append(dict(period=name, rule=rule, cost_bp=cost, real_pct_yr=1200 * m, real_t=t,
                                 placebo_pct_yr=1200 * p_[sl].mean(), edge_pct_yr=1200 * me, edge_t=te))
    return pd.DataFrame(rows)


def main():
    h, ht, cur, st, conc = hedge_tables()
    h.to_csv(RES / "hedge_summary.csv", index=False, float_format="%.5g")
    cur.to_csv(RES / "tuning_curves.csv", index=False, float_format="%.6g")
    st.to_csv(RES / "hedge_stress_windows.csv", index=False, float_format="%.4g")
    p9 = p09_comparison()
    p9.to_csv(RES / "p09_comparison.csv", index=False, float_format="%.5g")
    pt, ptt, adf = pairs_tables()
    pt.to_csv(RES / "pairs_summary.csv", index=False, float_format="%.5g")
    adf.to_csv(RES / "pairs_unit_root.csv", index=False, float_format="%.4g")
    hedge_buckets().to_csv(RES / "hedge_buckets.csv", index=False, float_format="%.4g")
    halflife_by_decade().to_csv(RES / "halflife_by_decade.csv", index=False, float_format="%.4g")
    cor = []
    for u in ("countries", "industries"):
        P = Pairs(u)
        k = P.monthly("real", "dist", "K")
        for other in ("REV1", "S", "R", "K-wait5"):
            o = P.monthly("real", "dist", other).reindex(k.index)
            cor.append(dict(universe=u, a="K", b=other, corr_monthly=k.corr(o)))
    pd.DataFrame(cor).to_csv(RES / "pairs_correlations.csv", index=False, float_format="%.3f")
    subperiod_pairs().to_csv(RES / "pairs_subperiods.csv", index=False, float_format="%.4g")
    tests = pd.concat([ht, ptt], ignore_index=True)
    tests["holm_p"] = holm(tests["p"].values)
    tests.to_csv(RES / "preregistered_tests.csv", index=False, float_format="%.4g")
    print(tests.to_string())


if __name__ == "__main__":
    main()
