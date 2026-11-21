"""All experiments for project 03. Writes every number used in README.md to results/.

Pre-registered design (fixed before looking at any result, see README section 3):
  * headline rule: price above its 200-day simple moving average, decided at the close,
    held from the next close (no look-ahead), 5 bp one-way cost;
  * volatility state: trailing 21-day realised volatility;
  * matched placebo: invested iff vol is below the rolling-5y quantile that matches the trend
    rule's own rolling-5y exposure; it never sees prices, only volatility;
  * headline statistics: (H1) coefficient of the trend signal on next-month vol-standardised
    returns controlling for vol quintile; (H2) Sharpe(MA) - Sharpe(placebo), net of costs.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

from core import (INDICES, LABEL, PLACEBO_WIN, ROOT, TDAYS, crossover_signal, holm, load_closes,
                  load_index, ma_signal, matched_vol_placebo, max_drawdown, nw_se_mean, romano_wolf,
                  sharpe, sharpe_diff_hac, spell_shuffle, stationary_bootstrap_idx, strategy_returns,
                  trailing_vol)

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED = 20261121
B_BOOT = 999
N_SHUFFLE = 1000
BLOCK = 63

RULES = {"MA20": ("ma", 20), "MA50": ("ma", 50), "MA100": ("ma", 100), "MA150": ("ma", 150),
         "MA200": ("ma", 200), "MA250": ("ma", 250), "MA300": ("ma", 300),
         "X20/100": ("x", (20, 100)), "X50/200": ("x", (50, 200))}
STRESS = {"1937-38 recession": ("1937-03-01", "1938-04-30"), "1973-74 oil shock": ("1973-01-01", "1974-12-31"),
          "1987 crash": ("1987-08-01", "1987-12-31"), "2000-02 dot-com": ("2000-03-01", "2002-10-31"),
          "2008 GFC": ("2007-10-01", "2009-03-31"), "2020 Covid": ("2020-02-15", "2020-04-30"),
          "2022 inflation": ("2022-01-01", "2022-10-31")}
SUBPERIODS = {"1934-1961": ("1934-01-01", "1961-12-31"), "1962-1991": ("1962-01-01", "1991-12-31"),
              "1992-2026 (post-publication)": ("1992-01-01", "2026-12-31")}


def signal_for(rule, p):
    kind, arg = RULES[rule]
    return ma_signal(p, arg) if kind == "ma" else crossover_signal(p, *arg)


def vol_target_weight(vol: pd.Series) -> pd.Series:
    """Long-only volatility targeting baseline: w = min(1, rolling-5y median vol / current vol)."""
    tgt = vol.rolling(PLACEBO_WIN, min_periods=PLACEBO_WIN // 2).median()
    return (tgt / vol).clip(upper=1.0)


def build(name, closes, vol_win=21, placebo_win=PLACEBO_WIN, rules=("MA200",), extra_lag=0, div_yield=0.0):
    df = load_index(name, closes)
    if div_yield:
        # crude total-return proxy: a constant dividend yield accrued daily (signals still use the price index)
        df = df.assign(r=df.r + div_yield / TDAYS)
    vol = trailing_vol(df.r, vol_win)
    sig = {k: signal_for(k, df.p) for k in rules}
    plc = {k: matched_vol_placebo(sig[k], vol, placebo_win) for k in rules}
    vt = vol_target_weight(vol)
    valid = vol.notna() & vt.notna()
    for k in rules:
        valid &= sig[k].notna() & plc[k].notna()
    first = valid.idxmax()
    keep = df.index >= first
    df, vol, vt = df[keep], vol[keep], vt[keep]
    sig = {k: v[keep].shift(extra_lag) for k, v in sig.items()}
    plc = {k: v[keep].shift(extra_lag) for k, v in plc.items()}
    return df, vol, sig, plc, vt


def stats_row(x: pd.Series):
    x = x.dropna()
    return dict(ann_excess=x.mean() * TDAYS, ann_vol=x.std() * np.sqrt(TDAYS), sharpe=sharpe(x.values),
                max_dd=max_drawdown(x))


# ------------------------------------------------------------------ 1. headline performance
def headline(closes, rng):
    rows, tests, shuffle = [], [], []
    daily = {}
    for name in INDICES:
        df, vol, sig, plc, vt = build(name, closes)
        s, pl = sig["MA200"], plc["MA200"]
        strats = {"Buy & hold": strategy_returns(pd.Series(1.0, index=df.index), df.r, df.rf),
                  "MA200": strategy_returns(s, df.r, df.rf),
                  "Vol placebo": strategy_returns(pl, df.r, df.rf),
                  "Vol target": strategy_returns(vt, df.r, df.rf)}
        idx = strats["MA200"].net.index
        for k, st in strats.items():
            for kind in ("gross", "net"):
                x = getattr(st, kind).reindex(idx)
                rows.append(dict(index=name, strategy=k, returns=kind, exposure=st.exposure,
                                 turnover=st.turnover, start=idx[0].date(), end=idx[-1].date(),
                                 years=len(idx) / TDAYS, **stats_row(x)))
        ma, bh, pv = (strats[k].net.reindex(idx).fillna(0).values for k in ("MA200", "Buy & hold", "Vol placebo"))
        daily[name] = pd.DataFrame({"bh": bh, "ma": ma, "pl": pv}, index=idx)
        # bootstrap CI for the three Sharpe differences
        boots = []
        for _ in range(B_BOOT):
            ii = stationary_bootstrap_idx(len(ma), BLOCK, rng)
            sr = sharpe(np.column_stack([ma[ii], bh[ii], pv[ii]]))
            boots.append([sr[0] - sr[1], sr[2] - sr[1], sr[0] - sr[2]])
        boots = np.array(boots)
        sr_ma, sr_bh, sr_pl = sharpe(ma), sharpe(bh), sharpe(pv)
        for j, (lab, a, b) in enumerate([("MA200 - B&H", ma, bh), ("Placebo - B&H", pv, bh), ("MA200 - Placebo", ma, pv)]):
            d, p = sharpe_diff_hac(a, b)
            lo, hi = np.quantile(boots[:, j], [0.025, 0.975])
            tests.append(dict(index=name, comparison=lab, sharpe_diff=d, p_hac=p, ci_lo=lo, ci_hi=hi))
        gain = sr_ma - sr_bh
        tests.append(dict(index=name, comparison="share of MA gain matched by placebo",
                          sharpe_diff=(sr_pl - sr_bh) / gain if abs(gain) > 1e-9 else np.nan))
        # spell-shuffle null: same exposure and turnover, no information
        raw = s.values
        sh = np.empty(N_SHUFFLE)
        for b in range(N_SHUFFLE):
            pos = pd.Series(spell_shuffle(raw, rng), index=s.index)
            sh[b] = sharpe(strategy_returns(pos, df.r, df.rf).net.values)
        shuffle.append(dict(index=name, sharpe_ma=sr_ma, sharpe_placebo=sr_pl,
                            shuffle_mean=sh.mean(), shuffle_p95=np.quantile(sh, 0.95),
                            p_ma=(1 + (sh >= sr_ma).sum()) / (N_SHUFFLE + 1),
                            p_placebo=(1 + (sh >= sr_pl).sum()) / (N_SHUFFLE + 1)))
        print(f"  {name}: SR bh {sr_bh:.2f} ma {sr_ma:.2f} placebo {sr_pl:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "performance.csv", index=False)
    pd.DataFrame(tests).to_csv(RES / "sharpe_tests.csv", index=False)
    pd.DataFrame(shuffle).to_csv(RES / "shuffle_null.csv", index=False)
    pd.concat(daily, names=["index", "date"]).to_csv(RES / "daily_net_returns_ma200.csv.gz", float_format="%.6g")


# ------------------------------------------------------------------ 2. direction vs volatility regressions
def quintile_state(vol: pd.Series, win=PLACEBO_WIN) -> pd.Series:
    """Rolling-5y quintile (0..4) of today's vol among the past `win` days (no look-ahead)."""
    rank = vol.rolling(win, min_periods=win // 2).rank(pct=True)
    return np.minimum((rank * 5).apply(np.ceil) - 1, 4).clip(lower=0)


def forward_sum(x: pd.Series, h: int) -> pd.Series:
    return x[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)


def regressions(closes, rules=("MA200",), out="direction_regressions.csv"):
    rows, panel = [], []
    for name in INDICES:
        df, vol, sig, _, _ = build(name, closes, rules=rules)
        ex = df.r - df.rf
        q = quintile_state(vol)
        dvol = vol / np.sqrt(TDAYS)
        fut_rv = np.sqrt(forward_sum(df.r ** 2, 21) / 21) * np.sqrt(TDAYS)
        for rule in rules:
            s = sig[rule]
            for h in (1, 21):
                y_std = forward_sum(ex, h) / (dvol * np.sqrt(h))
                y_raw = forward_sum(ex, h) * TDAYS / h        # annualised
                dummies = pd.get_dummies(q, prefix="q", drop_first=True, dtype=float)
                base = pd.DataFrame({"S": s}).join(dummies)
                for yname, y, ctrl in [("vol-standardised", y_std, False), ("vol-standardised", y_std, True),
                                       ("raw (annualised)", y_raw, False), ("raw (annualised)", y_raw, True)]:
                    X = base[["S"]] if not ctrl else base
                    d = pd.concat([y.rename("y"), X], axis=1).dropna()
                    m = sm.OLS(d.y, sm.add_constant(d.drop(columns="y"))).fit(
                        cov_type="HAC", cov_kwds={"maxlags": max(10, 2 * h)})
                    rows.append(dict(index=name, rule=rule, h=h, target=yname, vol_controls=ctrl,
                                     b=m.params["S"], se=m.bse["S"], t=m.tvalues["S"], p=m.pvalues["S"], n=int(m.nobs)))
                    if rule == "MA200" and h == 21 and yname == "vol-standardised" and ctrl:
                        d2 = d.copy(); d2["index"] = name; d2["date"] = d2.index
                        panel.append(d2)
            # volatility prediction: does the signal forecast next-month realised vol beyond current vol?
            d = pd.DataFrame({"y": np.log(fut_rv), "S": s, "lv": np.log(vol),
                              "lv63": np.log(trailing_vol(df.r, 63))}).replace([np.inf, -np.inf], np.nan).dropna()
            m = sm.OLS(d.y, sm.add_constant(d[["S", "lv", "lv63"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 42})
            rows.append(dict(index=name, rule=rule, h=21, target="log future vol", vol_controls=True,
                             b=m.params["S"], se=m.bse["S"], t=m.tvalues["S"], p=m.pvalues["S"], n=int(m.nobs)))
            m0 = sm.OLS(d.y, sm.add_constant(d[["S"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 42})
            rows.append(dict(index=name, rule=rule, h=21, target="log future vol", vol_controls=False,
                             b=m0.params["S"], se=m0.bse["S"], t=m0.tvalues["S"], p=m0.pvalues["S"], n=int(m0.nobs)))
    res = pd.DataFrame(rows)
    res.to_csv(RES / out, index=False)
    if panel:
        P = pd.concat(panel, ignore_index=True)
        P["tcode"] = pd.factorize(P.date)[0]
        X = sm.add_constant(P[["S"] + [c for c in P.columns if c.startswith("q_")]].fillna(0))
        m = sm.OLS(P.y, X).fit(cov_type="hac-groupsum", cov_kwds={"time": P.tcode.values, "maxlags": 42})
        pd.DataFrame([dict(model="pooled panel, 10 indices, h=21, vol-standardised, vol-quintile controls,"
                                  " Driscoll-Kraay SE", b=m.params["S"], se=m.bse["S"], t=m.tvalues["S"],
                           p=m.pvalues["S"], n=int(m.nobs))]).to_csv(RES / "pooled_panel.csv", index=False)
    return res


# ------------------------------------------------------------------ 3. grid + multiple testing
def grid(closes, rng):
    rows = []
    for name in INDICES:
        t0 = time.time()
        df, vol, sig, plc, _ = build(name, closes, rules=tuple(RULES))
        R = {}
        for k in RULES:
            R[(k, "ma")] = strategy_returns(sig[k], df.r, df.rf).net
            R[(k, "pl")] = strategy_returns(plc[k], df.r, df.rf).net
        bh = strategy_returns(pd.Series(1.0, index=df.index), df.r, df.rf).net
        M = pd.DataFrame(R).fillna(0.0)
        M["bh"] = bh.reindex(M.index).fillna(0.0)
        A = M.values
        sr = sharpe(A)
        K = len(RULES)
        d_pl = sr[0:2 * K:2] - sr[1:2 * K:2]
        d_bh = sr[0:2 * K:2] - sr[-1]
        bd_pl, bd_bh = np.empty((B_BOOT, K)), np.empty((B_BOOT, K))
        for b in range(B_BOOT):
            ii = stationary_bootstrap_idx(len(A), BLOCK, rng)
            s_b = sharpe(A[ii])
            bd_pl[b] = s_b[0:2 * K:2] - s_b[1:2 * K:2] - d_pl
            bd_bh[b] = s_b[0:2 * K:2] - s_b[-1] - d_bh
        rw_pl, rw_bh = romano_wolf(d_pl, bd_pl), romano_wolf(d_bh, bd_bh)
        for j, k in enumerate(RULES):
            rows.append(dict(index=name, rule=k, sharpe_ma=sr[2 * j], sharpe_placebo=sr[2 * j + 1], sharpe_bh=sr[-1],
                             diff_vs_bh=d_bh[j], p_rw_vs_bh=rw_bh[j], diff_vs_placebo=d_pl[j], p_rw_vs_placebo=rw_pl[j]))
        print(f"  grid {name} {time.time() - t0:.0f}s", flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(RES / "grid_romano_wolf.csv", index=False)
    return g


# ------------------------------------------------------------------ 4. sub-periods, stress, robustness
def subperiods_and_stress(closes):
    df, vol, sig, plc, _ = build("GSPC", closes)
    st = {"Buy & hold": strategy_returns(pd.Series(1.0, index=df.index), df.r, df.rf).net,
          "MA200": strategy_returns(sig["MA200"], df.r, df.rf).net,
          "Vol placebo": strategy_returns(plc["MA200"], df.r, df.rf).net}
    rows = []
    for lab, (a, b) in SUBPERIODS.items():
        for k, x in st.items():
            rows.append(dict(period=lab, strategy=k, **stats_row(x[a:b])))
    pd.DataFrame(rows).to_csv(RES / "subperiods_gspc.csv", index=False)
    # decade table
    rows = []
    for k, x in st.items():
        for dec, g in x.groupby((x.index.year // 10) * 10):
            rows.append(dict(decade=f"{dec}s", strategy=k, sharpe=sharpe(g.values), n=len(g)))
    pd.DataFrame(rows).to_csv(RES / "decades_gspc.csv", index=False)
    rows = []
    for name in INDICES:
        d, vol, sig, plc, _ = build(name, closes)
        ss = {"Buy & hold": strategy_returns(pd.Series(1.0, index=d.index), d.r, d.rf).net,
              "MA200": strategy_returns(sig["MA200"], d.r, d.rf).net,
              "Vol placebo": strategy_returns(plc["MA200"], d.r, d.rf).net}
        for lab, (a, b) in STRESS.items():
            if d.index[0] > pd.Timestamp(a):
                continue
            for k, x in ss.items():
                xx = x[a:b]
                rows.append(dict(index=name, window=lab, strategy=k, cum_return=float((1 + xx).prod() - 1),
                                 max_dd=max_drawdown(xx)))
    pd.DataFrame(rows).to_csv(RES / "stress_windows.csv", index=False)


def robustness(closes):
    """Every variant is evaluated on the SAME dates per index (the latest common start), so that
    differences come from the design choice and not from a shorter sample."""
    variants = {"baseline": {}, "63-day vol state": dict(vol_win=63), "20-year placebo window": dict(placebo_win=5040),
                "1-year placebo window": dict(placebo_win=252), "extra 1-day execution lag": dict(extra_lag=1),
                "dividend proxy 3%/yr": dict(div_yield=0.03)}
    rows = []
    for name in INDICES:
        series = {}
        for vname, kw in variants.items():
            df, vol, sig, plc, _ = build(name, closes, **kw)
            for cost in ([0, 5, 10, 20] if vname == "baseline" else [5]):
                series[(vname, cost)] = (strategy_returns(sig["MA200"], df.r, df.rf, cost).net,
                                         strategy_returns(plc["MA200"], df.r, df.rf, cost).net,
                                         strategy_returns(pd.Series(1.0, index=df.index), df.r, df.rf, cost).net)
        start = max(v[0].index[0] for v in series.values())
        for (vname, cost), (ma, pl, bh) in series.items():
            ma, pl, bh = ma[start:], pl.reindex(ma[start:].index).fillna(0), bh.reindex(ma[start:].index)
            rows.append(dict(variant=vname, cost_bp=cost, index=name, start=start.date(), sharpe_ma=sharpe(ma.values),
                             sharpe_placebo=sharpe(pl.values), sharpe_bh=sharpe(bh.values)))
    pd.DataFrame(rows).to_csv(RES / "robustness.csv", index=False)


# ------------------------------------------------------------------ 5. mechanism: where does the signal sit?
def mechanism(closes):
    rows = []
    for name in INDICES:
        df, vol, sig, plc, _ = build(name, closes)
        q = quintile_state(vol)
        s = sig["MA200"]
        ex = df.r - df.rf
        y = forward_sum(ex, 21) / (vol / np.sqrt(TDAYS) * np.sqrt(21))
        raw = forward_sum(ex, 21) * TDAYS / 21
        fv = np.sqrt(forward_sum(df.r ** 2, 21) / 21) * np.sqrt(TDAYS)
        d = pd.DataFrame({"q": q, "S": s, "y": y, "raw": raw, "fv": fv, "pl": plc["MA200"]}).dropna()
        for qq, g in d.groupby("q"):
            for sv, gg in g.groupby("S"):
                rows.append(dict(index=name, vol_quintile=int(qq) + 1, signal=int(sv), share_days=len(gg) / len(d),
                                 frac_of_quintile=len(gg) / len(g), mean_std_ret=gg.y.mean(),
                                 se_std_ret=nw_se_mean(gg.y.values, 21) if len(gg) > 50 else np.nan,
                                 mean_raw_ret=gg.raw.mean(), mean_future_vol=gg.fv.mean()))
        # agreement between MA and placebo
        rows.append(dict(index=name, vol_quintile=0, signal=-1, share_days=float((d.S == d.pl).mean())))
    pd.DataFrame(rows).to_csv(RES / "mechanism.csv", index=False)


def main():
    rng = np.random.default_rng(SEED)
    closes = load_closes()
    t0 = time.time()
    print("1/5 headline", flush=True); headline(closes, rng)
    print("2/5 regressions", flush=True); regressions(closes)
    regressions(closes, rules=tuple(RULES), out="direction_regressions_grid.csv")
    print("3/5 grid", flush=True); grid(closes, rng)
    print("4/5 subperiods, stress, robustness", flush=True); subperiods_and_stress(closes); robustness(closes)
    print("5/5 mechanism", flush=True); mechanism(closes)
    # Holm across indices x rules for the direction coefficient with vol controls (h=21, one-sided b>0)
    g = pd.read_csv(RES / "direction_regressions_grid.csv")
    g = g[(g.h == 21) & (g.target == "vol-standardised") & g.vol_controls].copy()
    from scipy.stats import norm
    g["p_one_sided"] = 1 - norm.cdf(g.t)
    g["p_holm"] = holm(g.p_one_sided.values)
    g.to_csv(RES / "direction_grid_holm.csv", index=False)
    print("6/6 stratified decomposition", flush=True); stratified_decomposition(closes, np.random.default_rng(SEED + 6))
    strat_robustness(closes, np.random.default_rng(SEED + 7))
    print(f"done in {time.time() - t0:.0f}s")




# ------------------------------------------------------------------ 6. (added after step 1-5) volatility-stratified shuffle decomposition
def strat_robustness(closes, rng):
    """Decomposition under other volatility-state definitions, all on the 20y-window common sample."""
    starts = {}
    for name in INDICES:
        df, vol, *_ = build(name, closes)
        starts[name] = decile_state(vol, 5040).dropna().index[0]
    rows = []
    for lab, kw in {"5y deciles, 21d vol (baseline)": {}, "1y deciles": dict(dec_win=252),
                    "20y deciles": dict(dec_win=5040), "5y deciles, 63d vol": dict(vol_win=63)}.items():
        o = stratified_decomposition(closes, rng, n_rep=200, start=starts, out="_tmp.csv", boot=False, **kw)
        o["variant"] = lab
        rows.append(o)
    (RES / "_tmp.csv").unlink()
    pd.concat(rows).to_csv(RES / "stratified_robustness.csv", index=False)


def decile_state(vol: pd.Series, win=PLACEBO_WIN) -> pd.Series:
    rank = vol.rolling(win, min_periods=win // 2).rank(pct=True)
    return np.minimum(np.ceil(rank * 10) - 1, 9).clip(lower=0)


def stratified_decomposition(closes, rng, n_rep=N_SHUFFLE, dec_win=PLACEBO_WIN, vol_win=21, start=None, out="stratified_decomposition.csv", boot=True):
    """Split Sharpe(MA) - Sharpe(B&H) into
         vol-timing part  = E[Sharpe(stratified shuffle)] - Sharpe(B&H)
         direction part   = Sharpe(MA) - E[Sharpe(stratified shuffle)]
    The stratified shuffle permutes the MA signal across days *within the same rolling-5y
    volatility decile*: it keeps exactly how often the rule is out of the market in each
    volatility state, and destroys any information the rule has about direction beyond that.
    Gross of costs, because a day-level permutation has artificial turnover."""
    out_name = out
    rows = []
    for name in INDICES:
        df, vol, sig, _, _ = build(name, closes, vol_win=vol_win)
        dec = decile_state(vol, dec_win).where(lambda x: x.index >= (start or {}).get(name, x.index[0]))
        s = sig["MA200"]
        ex = (df.r - df.rf).shift(-1)          # return earned by holding position decided at t
        d = pd.DataFrame({"s": s, "dec": dec, "ex": ex}).dropna()
        sv, dv, xv = d.s.values, d.dec.values.astype(int), d.ex.values
        sr_ma, sr_bh = sharpe(sv * xv), sharpe(xv)
        groups = [np.flatnonzero(dv == k) for k in range(10)]
        sh, mu, sd = np.empty(n_rep), np.empty(n_rep), np.empty(n_rep)
        for b in range(n_rep):
            perm = sv.copy()
            for g in groups:
                perm[g] = sv[rng.permutation(g)]
            y = perm * xv
            sh[b], mu[b], sd[b] = sharpe(y), y.mean() * TDAYS, y.std(ddof=1) * np.sqrt(TDAYS)
        e = sh.mean()
        y_ma = sv * xv

        # Dependence-robust version: the permutation's expectation in closed form (exposure p_k per
        # decile applied as a weight), re-estimated inside a stationary block bootstrap of days.
        def parts(ii):
            s_, d_, x_ = sv[ii], dv[ii], xv[ii]
            pk = np.array([s_[d_ == k].mean() if (d_ == k).any() else 0.0 for k in range(10)])
            w = pk[d_]
            mu_b, m2_b = (w * x_).mean(), (w * x_ ** 2).mean()
            sr_b = mu_b / np.sqrt(m2_b - mu_b ** 2) * np.sqrt(TDAYS)
            y = s_ * x_
            return np.array([sharpe(y) - sr_b, (y.mean() - mu_b) * TDAYS,
                             (y.std(ddof=1) - np.sqrt(m2_b - mu_b ** 2)) * np.sqrt(TDAYS), sr_b - sharpe(x_),
                             sharpe(y) - sharpe(x_)])
        obs = parts(np.arange(len(sv)))
        bt = np.array([parts(stationary_bootstrap_idx(len(sv), BLOCK, rng)) for _ in range(B_BOOT if boot else 2)])
        ci = np.quantile(bt, [0.025, 0.975], axis=0)
        rows.append(dict(index=name, start=d.index[0].date(), sharpe_bh=sr_bh, sharpe_ma=sr_ma, sharpe_strat_shuffle=e,
                         strat_p05=np.quantile(sh, 0.05), strat_p95=np.quantile(sh, 0.95),
                         vol_timing_part=e - sr_bh, direction_part=sr_ma - e,
                         p_direction=(1 + (sh >= sr_ma).sum()) / (n_rep + 1),
                         mean_bh=xv.mean() * TDAYS, vol_bh=xv.std(ddof=1) * np.sqrt(TDAYS),
                         mean_shuffle=mu.mean(), vol_shuffle=sd.mean(),
                         mean_ma=y_ma.mean() * TDAYS, vol_ma=y_ma.std(ddof=1) * np.sqrt(TDAYS),
                         p_mean=(1 + (mu >= y_ma.mean() * TDAYS).sum()) / (n_rep + 1),
                         p_vol=(1 + (sd <= y_ma.std(ddof=1) * np.sqrt(TDAYS)).sum()) / (n_rep + 1),
                         analytic_direction_part=obs[0], dir_ci_lo=ci[0, 0], dir_ci_hi=ci[1, 0],
                         p_boot_direction=(1 + (bt[:, 0] <= 0).sum()) / (B_BOOT + 1),
                         analytic_mean_gap=obs[1], mean_gap_ci_lo=ci[0, 1], mean_gap_ci_hi=ci[1, 1],
                         p_boot_mean=(1 + (bt[:, 1] <= 0).sum()) / (B_BOOT + 1),
                         analytic_vol_gap=obs[2], vol_gap_ci_lo=ci[0, 2], vol_gap_ci_hi=ci[1, 2],
                         p_boot_vol=(1 + (bt[:, 2] >= 0).sum()) / (B_BOOT + 1),
                         analytic_vol_timing_part=obs[3], vt_ci_lo=ci[0, 3], vt_ci_hi=ci[1, 3],
                         p_boot_vol_timing=(1 + (bt[:, 3] >= 0).sum()) / (B_BOOT + 1),
                         analytic_total=obs[4], total_ci_lo=ci[0, 4], total_ci_hi=ci[1, 4],
                         exposure_by_decile=";".join(f"{sv[g].mean():.2f}" for g in groups)))
        print(f"  strat {name}: bh {sr_bh:.2f} shuffle {e:.2f} ma {sr_ma:.2f}", flush=True)
    out = pd.DataFrame(rows)
    out["p_direction_holm"] = holm(out.p_direction.values)
    out["p_boot_direction_holm"] = holm(out.p_boot_direction.values)
    out["p_boot_mean_holm"] = holm(out.p_boot_mean.values)
    out["p_boot_vol_holm"] = holm(out.p_boot_vol.values)
    out["p_boot_vol_timing_holm"] = holm(out.p_boot_vol_timing.values)
    out.to_csv(RES / out_name, index=False)
    return out


if __name__ == "__main__":
    if "--strat" in __import__("sys").argv:
        stratified_decomposition(load_closes(), np.random.default_rng(SEED + 6))
    elif "--robust" in __import__("sys").argv:
        robustness(load_closes()); strat_robustness(load_closes(), np.random.default_rng(SEED + 7))
    else:
        main()
