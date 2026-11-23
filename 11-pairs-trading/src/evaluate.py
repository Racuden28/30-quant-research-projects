"""
Turn the raw window results into strategy return series, placebo edges, convergence statistics and the
pre-registered tests. Writes every number quoted in README.md to results/*.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from core import TRADE
from data import RES
from stats import block_boot_ci, holm, nw_t, sharpe

PERIODS = {
    "industries": {"Full 1927-2026": ("1927-07", "2026-09"), "1927-1961": ("1927-07", "1961-12"),
                   "1962-2002 (GGR sample)": ("1962-01", "2002-12"), "2003-2026": ("2003-01", "2026-09")},
    "countries": {"Full 2002-2026": ("2002-01", "2026-11"),
                  "2003-2026": ("2003-01", "2026-11"), "2010-2026": ("2010-01", "2026-11")},
}
STRESS = {"1929-32 crash": ("1929-10-01", "1932-06-30"), "1987 crash": ("1987-10-01", "1987-12-31"),
          "2000-02 dot-com": ("2000-03-01", "2002-10-31"), "2008-09 GFC": ("2008-09-01", "2009-03-31"),
          "2020 Covid": ("2020-02-20", "2020-04-30"), "2022 inflation": ("2022-01-01", "2022-10-31")}
MIN_ASSETS = 10
LABEL = {"D20_w1": "Distance, top 20, wait 1 day (primary)", "D20_w0": "Distance, top 20, no wait",
         "D20_w5": "Distance, top 20, wait 5 days", "D101_w1": "Distance, pairs 101-120, wait 1",
         "Dall_w1": "Distance, all pairs, wait 1", "D20c_w1": "Distance, top 20, no renormalisation at trading start", "C20_w1": "Cointegration, top 20, wait 1",
         "C20_w0": "Cointegration, top 20, no wait"}


class Raw:
    def __init__(self, universe: str, tag: str = ""):
        z = np.load(RES / f"raw_{universe}{tag}.npz")
        self.u = universe
        self.z = z
        self.dates = pd.to_datetime(z["dates"], unit="D")
        self.starts = z["starts"]
        self.nassets = z["nassets"]
        self.valid = self.nassets >= MIN_ASSETS      # a portfolio is formed only with >= 10 eligible assets

    def keys(self):
        return sorted({tuple(k.split("|")[1:]) for k in self.z.files if k.startswith("pnl|")})

    def daily(self, src: str, cfg: str, cost_bp: float = 0.0, rep: int | None = None) -> pd.Series:
        """Strategy daily return: average over the active overlapping portfolios (and over placebo reps)."""
        pnl = self.z[f"pnl|{src}|{cfg}"]
        turn = self.z[f"turn|{src}|{cfg}"]
        x = pnl - cost_bp * 1e-4 * turn
        x = x.mean(1) if rep is None else x[:, rep]
        T = len(self.dates)
        tot, cnt = np.zeros(T), np.zeros(T)
        for k, s in enumerate(self.starts):
            if not self.valid[k]:
                continue
            v = x[k]
            ok = ~np.isnan(v)
            idx = np.arange(s, s + TRADE)[ok]
            tot[idx] += v[ok]
            cnt[idx] += 1
        out = pd.Series(np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), index=self.dates)
        return out.dropna()

    def monthly(self, src: str, cfg: str, cost_bp: float = 0.0, rep: int | None = None) -> pd.Series:
        d = self.daily(src, cfg, cost_bp, rep)
        return d.groupby(d.index.to_period("M")).sum()

    def turnover_monthly(self, src: str, cfg: str) -> pd.Series:
        """Notional traded per $1 committed per month (both legs, open and close)."""
        T = len(self.dates)
        turn = self.z[f"turn|{src}|{cfg}"].mean(1)
        tot, cnt = np.zeros(T), np.zeros(T)
        for k, s in enumerate(self.starts):
            if not self.valid[k]:
                continue
            v = turn[k]
            ok = ~np.isnan(v)
            idx = np.arange(s, s + TRADE)[ok]
            tot[idx] += v[ok]
            cnt[idx] += 1
        d = pd.Series(np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), index=self.dates).dropna()
        return d.groupby(d.index.to_period("M")).sum()

    def trade_stats(self, src: str, cfg: str, mask=None) -> dict:
        st = self.z[f"stats|{src}|{cfg}"]                 # (K, reps, 6)
        mask = self.valid if mask is None else (mask & self.valid)
        st = st[mask]
        s = np.nansum(st.reshape(-1, 6), axis=0)
        reps = self.z[f"stats|{src}|{cfg}"].shape[1]
        ntr, nconv = s[0], s[1]
        return {"trades": ntr / reps, "converged_share": nconv / ntr if ntr else np.nan,
                "mean_ret_converged": s[2] / nconv if nconv else np.nan,
                "mean_ret_nonconverged": s[3] / (ntr - nconv) if ntr > nconv else np.nan,
                "mean_ret_trade": (s[2] + s[3]) / ntr if ntr else np.nan,
                "days_open_per_trade": s[4] / ntr if ntr else np.nan}

    def window_dates(self):
        return self.dates[self.starts]


def summarize(x: pd.Series, seed=0) -> dict:
    m, t = nw_t(x.values, 6)
    lo, hi = block_boot_ci(x.values, draws=2000, seed=seed)
    return {"months": len(x), "mean_pct_month": 100 * m, "ann_mean_pct": 1200 * m,
            "ann_vol_pct": 100 * x.std() * np.sqrt(12), "sharpe": sharpe(x.values),
            "nw_t": t, "p_one_sided": 1 - norm.cdf(t), "ci95_lo_pct": 100 * lo, "ci95_hi_pct": 100 * hi}


def sl(x: pd.Series, a: str, b: str) -> pd.Series:
    return x.loc[pd.Period(a, "M"):pd.Period(b, "M")]


def performance(raw: Raw) -> pd.DataFrame:
    rows = []
    real = [c for c in raw.keys() if c[0] == "real"]
    for _, cfg in real:
        m_real = raw.monthly("real", cfg)
        series = {"real": m_real}
        for src in ("N1", "ND"):
            if f"pnl|{src}|{cfg}" in raw.z.files:
                series[src] = raw.monthly(src, cfg)
                series[f"real-{src}"] = (m_real - series[src]).dropna()
        for pname, (a, b) in PERIODS[raw.u].items():
            for sname, x in series.items():
                rows.append({"universe": raw.u, "config": cfg, "series": sname, "period": pname,
                             **summarize(sl(x, a, b))})
    return pd.DataFrame(rows)


def convergence(raw: Raw) -> pd.DataFrame:
    wd = raw.window_dates()
    rows = []
    for pname, (a, b) in PERIODS[raw.u].items():
        mask = (wd >= pd.Timestamp(a)) & (wd <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))
        for src in ("real", "N1", "ND"):
            for cfg in ("D20_w1", "D20_w0", "C20_w1", "Dall_w1", "D20_w5", "D101_w1", "C20_w0", "D20c_w1"):
                if f"stats|{src}|{cfg}" not in raw.z.files:
                    continue
                rows.append({"universe": raw.u, "period": pname, "source": src, "config": cfg,
                             **raw.trade_stats(src, cfg, mask)})
    return pd.DataFrame(rows)


def by_decade(raw: Raw, cfg="D20_w1") -> pd.DataFrame:
    real, n1, n2 = raw.monthly("real", cfg), raw.monthly("N1", cfg), raw.monthly("ND", cfg)
    wd = raw.window_dates()
    rows = []
    for dec in range(1920, 2030, 10):
        r = real[(real.index.year >= dec) & (real.index.year < dec + 10)]
        if len(r) < 24:
            continue
        e1 = (r - n1.reindex(r.index)).dropna()
        e2 = (r - n2.reindex(r.index)).dropna()
        mask = (wd.year >= dec) & (wd.year < dec + 10)
        cs_r, cs_1 = raw.trade_stats("real", cfg, mask), raw.trade_stats("N1", cfg, mask)
        rows.append({"universe": raw.u, "decade": f"{dec}s", "months": len(r),
                     "real_ann_pct": 1200 * r.mean(), "N1_ann_pct": 1200 * n1.reindex(r.index).mean(),
                     "ND_ann_pct": 1200 * n2.reindex(r.index).mean(),
                     "edge_N1_ann_pct": 1200 * e1.mean(), "edge_N1_t": nw_t(e1.values)[1],
                     "edge_ND_ann_pct": 1200 * e2.mean(), "edge_ND_t": nw_t(e2.values)[1],
                     "conv_real": cs_r["converged_share"], "conv_N1": cs_1["converged_share"],
                     "mean_assets": raw.nassets[mask].mean()})
    return pd.DataFrame(rows)


def wait_effect(raw: Raw) -> pd.DataFrame:
    """Same-close execution minus one-day wait: the part of the profit that needs the closing print."""
    rows = []
    for pname, (a, b) in PERIODS[raw.u].items():
        for c0, c1 in (("D20_w0", "D20_w1"), ("D20_w1", "D20_w5"), ("C20_w0", "C20_w1")):
            d = sl(raw.monthly("real", c0) - raw.monthly("real", c1), a, b)
            m, t = nw_t(d.values)
            rows.append({"universe": raw.u, "period": pname, "comparison": f"{c0} - {c1}", "ann_pct": 1200 * m,
                         "nw_t": t})
    return pd.DataFrame(rows)


def nofilter() -> pd.DataFrame:
    """Country ETFs with and without the stale-price rule (ablation)."""
    f, nf = Raw("countries"), Raw("countries", tag="_nofilter")
    rows = []
    for pname, (a, b) in {"1997-2001": ("1997-03", "2001-12"), "2002-2026": ("2002-01", "2026-11")}.items():
        wd = nf.window_dates()
        mask = (wd >= pd.Timestamp(a)) & (wd <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))
        for name, raw in (("stale-price rule (main)", f), ("no rule (ablation)", nf)):
            x = sl(raw.monthly("real", "D20_w1"), a, b)
            e = sl((raw.monthly("real", "D20_w1") - raw.monthly("N1", "D20_w1")).dropna(), a, b)
            if len(x) < 12:
                rows.append({"period": pname, "version": name, "months": len(x)})
                continue
            cs = raw.trade_stats("real", "D20_w1", mask)
            rows.append({"period": pname, "version": name, "months": len(x), "mean_assets": raw.nassets[mask & raw.valid].mean(),
                         "real_ann_pct": 1200 * x.mean(), "real_t": nw_t(x.values)[1],
                         "edge_N1_ann_pct": 1200 * e.mean(), "edge_N1_t": nw_t(e.values)[1],
                         "converged_share": cs["converged_share"]})
    return pd.DataFrame(rows)


def variance_ratios(raw: Raw) -> pd.DataFrame:
    """Pooled variance ratios of traded pair spreads, real vs the drift-equalised IID placebo."""
    from core import VR_H
    wd = raw.window_dates()
    rows = []
    for pname, (a, b) in PERIODS[raw.u].items():
        mask = (wd >= pd.Timestamp(a)) & (wd <= pd.Timestamp(b) + pd.offsets.MonthEnd(0)) & raw.valid
        for cfg in ("D20_w1", "Dall_w1"):
            m = {src: np.nanmean(raw.z[f"vr|{src}|{cfg}"][mask], axis=(0, 1)) for src in ("real", "N1")}
            for src in ("real", "N1"):
                vr = m[src] / (np.array(VR_H) * m[src][0])
                for h, v, rv in zip(VR_H, vr, m["real"] / (np.array(VR_H) * m["real"][0])):
                    rows.append({"universe": raw.u, "period": pname, "config": cfg, "source": src, "h": h, "vr": v})
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["universe", "period", "config", "h"], columns="source", values="vr").reset_index()
    piv["real_over_placebo"] = piv["real"] / piv["N1"]
    return piv


def costs(raw: Raw, cfg="D20_w1") -> pd.DataFrame:
    rows = []
    for pname, (a, b) in PERIODS[raw.u].items():
        g = sl(raw.monthly("real", cfg), a, b)
        g1 = sl(raw.monthly("N1", cfg), a, b)
        tv = sl(raw.turnover_monthly("real", cfg), a, b)
        row = {"universe": raw.u, "config": cfg, "period": pname, "gross_ann_pct": 1200 * g.mean(),
               "turnover_per_month": tv.mean(),
               "breakeven_bp_per_leg": 1e4 * g.mean() / tv.mean(),
               "breakeven_edge_bp_per_leg": 1e4 * (g - g1).mean() / tv.mean()}
        for c in (5, 10, 20):
            n = sl(raw.monthly("real", cfg, cost_bp=c), a, b)
            m, t = nw_t(n.values)
            row[f"net{c}_ann_pct"] = 1200 * m
            row[f"net{c}_t"] = t
            row[f"net{c}_sharpe"] = sharpe(n.values)
        rows.append(row)
    return pd.DataFrame(rows)


def stress(raw: Raw, cfg="D20_w1") -> pd.DataFrame:
    real, n1 = raw.daily("real", cfg), raw.daily("N1", cfg)
    rows = []
    allmask = pd.Series(False, index=real.index)
    for name, (a, b) in STRESS.items():
        r = real.loc[a:b]
        if len(r) < 20:
            continue
        allmask.loc[a:b] = True
        e = r - n1.reindex(r.index)
        rows.append({"universe": raw.u, "window": name, "days": len(r), "real_cum_pct": 100 * r.sum(),
                     "N1_cum_pct": 100 * n1.reindex(r.index).sum(), "edge_cum_pct": 100 * e.sum(),
                     "edge_ann_pct": 252 * 100 * e.mean(), "real_ann_vol_pct": 100 * r.std() * np.sqrt(252)})
    e_all = (real - n1.reindex(real.index)).dropna()
    rows.append({"universe": raw.u, "window": "All stress days", "days": int(allmask.sum()),
                 "edge_ann_pct": 252 * 100 * e_all[allmask.reindex(e_all.index).fillna(False)].mean()})
    rows.append({"universe": raw.u, "window": "All other days", "days": int((~allmask).sum()),
                 "edge_ann_pct": 252 * 100 * e_all[~allmask.reindex(e_all.index).fillna(False)].mean()})
    return pd.DataFrame(rows)


def screen(universe: str) -> pd.DataFrame:
    s = pd.read_csv(RES / f"screen_{universe}.csv")
    rows = []
    raw = Raw(universe)
    wd = raw.window_dates()
    s["date"] = wd[s.k.values]
    # non-overlapping years only: one window every 12 months, so year-1 and year-2 blocks do not overlap
    s = s[(s.k % 12 == 0) & raw.valid[s.k.values]]
    for pname, (a, b) in PERIODS[universe].items():
        x = s[(s.date >= pd.Timestamp(a)) & (s.date <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))]
        for src in ("real", "N1"):
            y = x[x.source == src]
            if not len(y):
                continue
            lift = (y.sig2_given_sig1 - y.sig2_given_not1)
            rows.append({"universe": universe, "period": pname, "source": src, "windows": y.k.nunique(),
                         "mean_pairs": y.npairs.mean(), "share_sig_p05": y.sig1.mean(),
                         "bh_discoveries_mean": y.bh_discoveries.mean(),
                         "share_windows_any_bh": (y.bh_discoveries > 0).mean(),
                         "persist_if_sig": y.sig2_given_sig1.mean(), "persist_if_not": y.sig2_given_not1.mean(),
                         "persistence_lift": lift.mean()})
    return pd.DataFrame(rows)


def tests(perf: pd.DataFrame, conv: pd.DataFrame, scr: pd.DataFrame, cst: pd.DataFrame,
          raws: dict) -> pd.DataFrame:
    def get(u, cfg, series, period):
        r = perf[(perf.universe == u) & (perf.config == cfg) & (perf.series == series) & (perf.period == period)]
        return r.iloc[0]

    rows = []
    h1 = get("industries", "D20_w1", "real-N1", "Full 1927-2026")
    rows.append({"hypothesis": "H1", "test": "Industries, real - N1 edge > 0, full sample",
                 "estimate_ann_pct": h1.ann_mean_pct, "nw_t": h1.nw_t, "p": h1.p_one_sided, "confirmatory": True})
    for u in ("industries", "countries"):
        h2 = get(u, "D20_w1", "real-N1", "2003-2026")
        rows.append({"hypothesis": "H2", "test": f"{u}, real - N1 edge > 0 in 2003-2026 (predicted: not significant)",
                     "estimate_ann_pct": h2.ann_mean_pct, "nw_t": h2.nw_t, "p": h2.p_one_sided,
                     "confirmatory": True})
    # H5: countries, net of 5 bp/leg, 2010-2026
    raw = raws["countries"]
    n = sl(raw.monthly("real", "D20_w1", cost_bp=5), "2010-01", "2026-11")
    m, t = nw_t(n.values)
    rows.append({"hypothesis": "H5", "test": "Countries, net of 5 bp/leg, 2010-2026 mean > 0 (predicted: not significant)",
                 "estimate_ann_pct": 1200 * m, "nw_t": t, "p": 1 - norm.cdf(t), "confirmatory": True})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p.values)
    # descriptive / exploratory
    ex = []
    old = get("industries", "D20_w1", "real-N1", "1962-2002 (GGR sample)")
    for u in ("industries", "countries"):
        new = get(u, "D20_w1", "real-N1", "2003-2026")
        ex.append({"hypothesis": "H2", "test": f"{u} 2003-2026 edge smaller than industries 1962-2002",
                   "estimate_ann_pct": new.ann_mean_pct - old.ann_mean_pct, "confirmatory": False})
    for u in ("industries", "countries"):
        full = list(PERIODS[u])[0]
        c = conv[(conv.universe == u) & (conv.period == full) & (conv.config == "D20_w1")].set_index("source")
        for src in ("N1",):
            ex.append({"hypothesis": "H3", "test": f"{u}: {src} convergence share / real (>= 0.8 predicted)",
                       "ratio": c.loc[src, "converged_share"] / c.loc["real", "converged_share"],
                       "real": c.loc["real", "converged_share"], "placebo": c.loc[src, "converged_share"],
                       "confirmatory": False})
    for u in ("countries", "industries"):
        full = list(PERIODS[u])[0]
        x = scr[(scr.universe == u) & (scr.period == full)].set_index("source")
        for src in ("N1",):
            ex.append({"hypothesis": "H4", "test": f"{u}: share of pairs with EG p<0.05, real / {src} (< 2 predicted)",
                       "ratio": x.loc["real", "share_sig_p05"] / x.loc[src, "share_sig_p05"],
                       "real": x.loc["real", "share_sig_p05"], "placebo": x.loc[src, "share_sig_p05"],
                       "confirmatory": False})
            ex.append({"hypothesis": "H4", "test": f"{u}: persistence lift, real vs {src} (real <= placebo predicted)",
                       "ratio": np.nan, "real": x.loc["real", "persistence_lift"],
                       "placebo": x.loc[src, "persistence_lift"], "confirmatory": False})
    return pd.concat([df, pd.DataFrame(ex)], ignore_index=True)


if __name__ == "__main__":
    raws = {u: Raw(u) for u in ("industries", "countries")}
    perf = pd.concat([performance(r) for r in raws.values()])
    perf.to_csv(RES / "performance.csv", index=False, float_format="%.5g")
    conv = pd.concat([convergence(r) for r in raws.values()])
    conv.to_csv(RES / "convergence.csv", index=False, float_format="%.5g")
    dec = pd.concat([by_decade(r) for r in raws.values()])
    dec.to_csv(RES / "by_decade.csv", index=False, float_format="%.5g")
    cst = pd.concat([costs(r) for r in raws.values()])
    cst.to_csv(RES / "costs.csv", index=False, float_format="%.5g")
    strs = pd.concat([stress(r) for r in raws.values()])
    strs.to_csv(RES / "stress_windows.csv", index=False, float_format="%.5g")
    scr = pd.concat([screen(u) for u in raws])
    scr.to_csv(RES / "coint_screen.csv", index=False, float_format="%.5g")
    we = pd.concat([wait_effect(r) for r in raws.values()])
    we.to_csv(RES / "wait_effect.csv", index=False, float_format="%.5g")
    vrs = pd.concat([variance_ratios(r) for r in raws.values()])
    vrs.to_csv(RES / "variance_ratios.csv", index=False, float_format="%.5g")
    print(vrs.round(3).to_string())
    nfl = nofilter()
    nfl.to_csv(RES / "stale_price_ablation.csv", index=False, float_format="%.5g")
    print(we.round(3).to_string())
    print(nfl.round(3).to_string())
    tst = tests(perf, conv, scr, cst, raws)
    tst.to_csv(RES / "hypothesis_tests.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 250)
    cols = ["universe", "config", "series", "period", "ann_mean_pct", "sharpe", "nw_t", "ci95_lo_pct", "ci95_hi_pct"]
    print(perf[perf.config.isin(["D20_w1", "C20_w1", "Dall_w1"])][cols].round(3).to_string())
    print(conv[conv.config == "D20_w1"].round(3).to_string())
    print(dec.round(3).to_string())
    print(cst.round(3).to_string())
    print(strs.round(3).to_string())
    print(scr.round(4).to_string())
    print(tst.round(4).to_string())
