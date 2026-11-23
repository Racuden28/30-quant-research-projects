"""
Every number in README.md. Writes CSVs to ../results.

Sections (numbers match the README):
  4.1 validation.csv       the engine reproduces projects 03, 12 and 19 under their own convention
  4.2 audit.csv            causality audit: real strategies pass, two classic bugs are caught
  5   lattice.csv          Sharpe under each fill, per strategy and instrument; diff vs same-close
      delay.csv            delay cost per unit of turnover (bp): one close later, next open
      pooled.csv           pooled tests across the 10 indices / 10 ETFs (joint date bootstrap)
  6   profile.csv          the day-after profile: mean signed return h = 1..10 days after a trade
      mechanism.csv        cells instrument x era: autocorrelation, trigger return, predicted vs actual
      mechanism_fit.csv    weighted regression of actual on predicted delay cost
  7   claims.csv           the headline claims of 03, 19 and 12 under each fill
      momentum.csv         12-1 momentum (project 12) under each fill, universes A and B
  8   stress.csv           delay P&L in the 2008, 2020 and 2022 windows
      delay_curve.csv      Sharpe as a function of the fill lag k = 0..10 closes
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data
from audit import causality_audit
from data import ETF_OF, INDICES, RES, STRESS, TDAYS
from engine import Market, first_day_pnl, run, run_timing
from stats import boot_mean_ci, holm, panel_boot, sharpe_diff_hac, sign_test
from strategies import centred_ma_bug, hmm_realtime, hmm_smoothed_bug, ma200, momentum_targets

COST = 5.0                      # bp per unit one-way turnover, as in projects 03 and 19
FILLS = ["close", "close+1", "close+2"]
ERAS = [("1928-1989", None, "1989-12-31"), ("1990-2005", "1990-01-01", "2005-12-31"),
        ("2006-2026", "2006-01-01", None)]
B = 2000
RES.mkdir(exist_ok=True)
ETFS = list(ETF_OF.values())
TURN: dict = {}
ERA_P = [("1928-1989", "1900-01-01", "1989-12-31"), ("1990-2005", "1990-01-01", "2005-12-31"),
         ("2006-2026", "2006-01-01", "2026-12-31")]


def sr(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(TDAYS))


# ------------------------------------------------------------------------------------- inputs
def instrument(sym: str, opens: bool = False):
    m, g = data.single_market(sym, opens=opens)
    px = g.adjclose if sym in ETFS else g.close
    return m, g, px


def signals(sym: str, px: pd.Series) -> dict[str, pd.Series]:
    return {"MA200": ma200(px), "HMM": hmm_realtime(None, key=sym)}


def eval_start(sig: pd.Series, m: Market, start=None) -> pd.Timestamp:
    """First day a position is held: the day after the first valid decision."""
    first = sig.dropna().index[0]
    s = m.index[m.index.get_loc(first) + 1]
    return max(s, pd.Timestamp(start)) if start is not None else s


# ------------------------------------------------------------------------------------- 4.1
def validation() -> pd.DataFrame:
    """Published numbers are in data/published_results.csv (copied from the result CSVs of 03, 12, 19)."""
    pub = pd.read_csv(data.DATA / "published_results.csv")
    rows = []
    c = data.closes_p03()
    for n in INDICES:
        p = c[n].dropna()
        if n in data.START:
            p = p[p.index >= data.START[n]]
        if n in ("GSPC", "IXIC"):
            rf = c["IRX"].dropna().reindex(p.index, method="ffill").fillna(0.0) / 100 / TDAYS
            rf[p.index < "1960-01-04"] = 0.0
        else:
            rf = pd.Series(0.0, index=p.index)
        m = data.market_from_ohlc({n: pd.DataFrame({"close": p, "open": np.nan})}, rf, p.index)
        r = run_timing(m, ma200(p), "close", COST)
        for ret, ser in [("gross", r.excess_gross), ("net", r.excess_net)]:
            q = pub[(pub.project == 3) & (pub["index"] == n) & (pub.strategy == "MA200") & (pub.returns == ret)]
            rows.append(dict(project="03", item=f"{n} MA200 Sharpe ({ret})", engine=sr(ser[q.start.iloc[0]:q.end.iloc[0]]),
                             published=float(q.value.iloc[0])))
    # project 19: real-time HMM, net, on its own evaluation window (our price file runs 3 days longer,
    # and the HMM is refitted on it, so agreement is close but not exact)
    for n in INDICES:
        m, g, px = instrument(n)
        q = pub[(pub.project == 19) & (pub["index"] == n) & (pub.strategy == "R3 Real-time") & (pub.returns == "net")]
        r = run_timing(m, hmm_realtime(None, key=n), "close", COST)
        rows.append(dict(project="19", item=f"{n} real-time HMM Sharpe (net)",
                         engine=sr(r.excess_net[q.start.iloc[0]:q.end.iloc[0]]), published=float(q.value.iloc[0])))
    res = momentum_run("A", "close")
    q = pub[pub.project == 12].set_index("metric").value
    rows.append(dict(project="12", item="A 12-1 gross premium (%/yr)", engine=res["gross_pct_yr"], published=q["gross_pct_yr"]))
    rows.append(dict(project="12", item="A 12-1 gross Sharpe (monthly)", engine=res["sharpe_gross"], published=q["sharpe_gross"]))
    rows.append(dict(project="12", item="A 12-1 turnover (per month)", engine=res["turnover_mo"], published=q["turnover_mo"]))
    rows.append(dict(project="12", item="A 12-1 average break-even (bp)", engine=res["be_bps"], published=q["be_const_bps"]))
    df = pd.DataFrame(rows)
    df["diff"] = df.engine - df.published
    df.to_csv(RES / "validation.csv", index=False)
    return df


# ------------------------------------------------------------------------------------- 4.2
def audit() -> pd.DataFrame:
    rows = []
    m, g, px = instrument("GSPC")
    px = px["1995":]
    r = px.pct_change()
    cases = [("MA200 (project 03)", lambda x: ma200(x), px, True),
             ("Centred MA200 (bug)", lambda x: centred_ma_bug(x), px, False),
             ("Real-time HMM (project 19)", lambda x: hmm_realtime(x, key=None), r, True),
             ("Smoothed HMM (project 19, rung R1)", lambda x: hmm_smoothed_bug(x), r, False)]
    ex, _ = data.industries()
    ex = ex["1990":"2026-09"]
    cases.append(("12-1 momentum (project 12)", lambda x: momentum_targets(x), ex, True))
    from strategies import month_end_days
    rng = np.random.default_rng(1)
    for name, fn, x, causal in cases:
        if "momentum" in name:
            # a calendar strategy decides on the last trading day of each month, which the exchange
            # calendar fixes in advance; audit it at its own decision dates
            me = month_end_days(x.index)
            cuts = sorted(rng.choice(me[len(me) // 2:-1], size=3, replace=False))
        else:
            cuts = None
        a = causality_audit(fn, x, cuts=cuts, n_cuts=3, seed=1)
        rows.append(dict(strategy=name, expected_causal=causal, cuts=len(a),
                         decisions_checked=int(a.decisions_before_cut.sum()), changed=int(a.changed.sum()),
                         max_share_changed=float(a.share_changed.max()), passed=bool((a.changed == 0).all())))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "audit.csv", index=False)
    return df


# ------------------------------------------------------------------------------------- 5
def lattice_and_delay():
    lat, dl, daily = [], [], {}
    for sym in INDICES + ETFS:
        kind = "ETF" if sym in ETFS else "index"
        m, g, px = instrument(sym)
        mo = None
        if kind == "ETF":
            mo, go, _ = instrument(sym, opens=True)
        for strat, s in signals(sym, px).items():
            st = eval_start(s, m)
            res = {f: run_timing(m, s, f, COST) for f in FILLS}
            if mo is not None:
                st_o = max(st, mo.index[1])
                res["open"] = run_timing(mo, s.reindex(mo.index), "open", COST)
            base = res["close"]
            turn = float(base.turnover[st:].sum())
            years = len(base.excess_net[st:]) / TDAYS
            for f, r in res.items():
                s0 = st if f != "open" else st_o
                a, b = r.excess_net[s0:], base.excess_net[s0:]
                d, p = sharpe_diff_hac(a.values, b.values) if f != "close" else (0.0, np.nan)
                lat.append(dict(strategy=strat, kind=kind, sym=sym, fill=f, start=s0.date(),
                                end=a.index[-1].date(), sharpe_net=sr(a), sharpe_gross=sr(r.excess_gross[s0:]),
                                sharpe_close_same_sample=sr(b), d_sharpe=d, p_hac=p,
                                ann_turnover=float(r.turnover[s0:].sum() / (len(a) / TDAYS))))
            # delay cost: one close later. Daily series d_t = first-day P&L; its sum = gross_close - gross_close+1
            dd = first_day_pnl(s.reindex(m.index).fillna(0.0).values, m)[st:]
            daily[(strat, sym, "close+1")] = dd
            # |trade| dated like dd (the day after the trade), for per-trade averages over sub-periods
            TURN[(strat, sym)] = s.reindex(m.index).fillna(0.0).diff().abs().shift(1).reindex(dd.index).fillna(0.0)
            mean, lo, hi, p = boot_mean_ci(dd.values, B=B, seed=1)
            scale = len(dd) / turn * 1e4
            dl.append(dict(strategy=strat, kind=kind, sym=sym, fill="close+1", trades=turn, years=years,
                           bp_per_trade=mean * scale, lo=lo * scale, hi=hi * scale, p=p,
                           sharpe_effect=lat[-len(res) + 1]["d_sharpe"]))
            if mo is not None:
                o = (res["close"].excess_gross - res["open"].excess_gross)[st_o:]
                c0 = res["close"].excess_gross.reindex(mo.index)
                # same-sample comparison: rerun close on the open sample
                rc = run_timing(mo, s.reindex(mo.index), "close", COST)
                o = (rc.excess_gross - res["open"].excess_gross)[st_o:]
                daily[(strat, sym, "open")] = o
                t_o = float(rc.turnover[st_o:].sum())
                mean, lo, hi, p = boot_mean_ci(o.values, B=B, seed=2)
                scale = len(o) / t_o * 1e4
                dl.append(dict(strategy=strat, kind=kind, sym=sym, fill="open", trades=t_o, years=len(o) / TDAYS,
                               bp_per_trade=mean * scale, lo=lo * scale, hi=hi * scale, p=p,
                               sharpe_effect=sr(res["open"].excess_net[st_o:]) - sr(rc.excess_net[st_o:])))
    lat, dl = pd.DataFrame(lat), pd.DataFrame(dl)
    dl["p_holm"] = np.nan
    for (s, k, f), gidx in dl.groupby(["strategy", "kind", "fill"]).groups.items():
        dl.loc[gidx, "p_holm"] = holm(dl.loc[gidx, "p"].values)
    lat.to_csv(RES / "lattice.csv", index=False)
    dl.to_csv(RES / "delay.csv", index=False)
    return lat, dl, daily


def same_period_index(daily_unused=None) -> pd.DataFrame:
    """MA200 and HMM on each index restricted to the dates its ETF existed (ETF signal valid)."""
    rows, daily = [], {}
    for n, e in ETF_OF.items():
        me, ge, pxe = instrument(e)
        m, g, px = instrument(n)
        for strat in ["MA200", "HMM"]:
            se = signals(e, pxe)[strat]
            s = signals(n, px)[strat]
            st = max(eval_start(se, me), eval_start(s, m))
            dd = first_day_pnl(s.reindex(m.index).fillna(0.0).values, m)[st:]
            TURN[(strat, n, "etf dates")] = s.reindex(m.index).fillna(0.0).diff().abs().shift(1).reindex(dd.index).fillna(0.0)
            r0 = run_timing(m, s, "close", COST)
            turn = float(r0.turnover[st:].sum())
            daily[(strat, n)] = dd
            mean, lo, hi, p = boot_mean_ci(dd.values, B=B, seed=3)
            scale = len(dd) / turn * 1e4
            rows.append(dict(strategy=strat, kind="index, ETF dates", sym=n, etf=e, start=st.date(), trades=turn,
                             bp_per_trade=mean * scale, lo=lo * scale, hi=hi * scale, p=p))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "delay_index_etf_dates.csv", index=False)
    return df, daily


def pooled(daily: dict, daily_idx: dict, dl: pd.DataFrame) -> pd.DataFrame:
    """Average per-trade delay cost across the 10 instruments of a group, with a joint date bootstrap."""
    rows = []
    groups = []
    for strat in ["MA200", "HMM"]:
        groups.append((strat, "ETF", "close+1", {e: daily[(strat, e, "close+1")] for e in ETFS}))
        groups.append((strat, "ETF", "open", {e: daily[(strat, e, "open")] for e in ETFS}))
        groups.append((strat, "index", "close+1", {n: daily[(strat, n, "close+1")] for n in INDICES}))
        groups.append((strat, "index, ETF dates", "close+1", {n: daily_idx[(strat, n)] for n in INDICES}))
    for strat, kind, fill, d in groups:
        X = pd.DataFrame(d)
        # per-instrument per-trade scale: days / trades, computed on that instrument's own sample
        if kind == "index, ETF dates":
            tr = pd.read_csv(RES / "delay_index_etf_dates.csv").query("strategy == @strat").set_index("sym").trades
        else:
            tr = dl.query("strategy == @strat and kind == @kind and fill == @fill").set_index("sym").trades
        n_days = X.notna().sum()
        scale = (n_days / tr.reindex(X.columns)).values * 1e4
        mask = X.notna().values
        Xv = X.fillna(0.0).values

        def stat(A, M=None):
            return A.sum(0) / n_days.values * scale

        obs = stat(Xv)
        bs = panel_boot(Xv, stat, B=B, seed=7)
        mean_obs = obs.mean()
        bm = bs.mean(1)
        lo, hi = np.quantile(bm, [0.025, 0.975])
        p = float(np.mean(np.abs(bm - mean_obs) >= abs(mean_obs)))
        k, nn, ps = sign_test(obs)
        rows.append(dict(strategy=strat, kind=kind, fill=fill, mean_bp_per_trade=mean_obs, lo=lo, hi=hi,
                         p_boot=max(p, 1 / B), n_positive=k, n=nn, sign_p=ps,
                         median_bp_per_trade=float(np.median(obs))))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "pooled.csv", index=False)
    return df


def era_pooled(daily: dict, daily_idx: dict) -> pd.DataFrame:
    """Per-trade cost of filling one close later, by era, for every instrument, and the across-instrument
    mean with a joint date bootstrap (all instruments resampled on the same dates). An instrument enters
    an era only with at least 25 trades in it."""
    rows = []
    for strat in ["MA200", "HMM"]:
        for kind, syms in [("index", INDICES), ("ETF", ETFS)]:
            for era, a, b in ERA_P:
                D = pd.DataFrame({s: daily[(strat, s, "close+1")][a:b] for s in syms})
                Tn = pd.DataFrame({s: TURN[(strat, s)][a:b] for s in syms}).reindex(D.index)
                keep = [s for s in syms if Tn[s].sum() >= 25]
                if len(keep) < 3:
                    continue
                Dv, Tv = D[keep].fillna(0.0).values, Tn[keep].fillna(0.0).values
                X = np.hstack([Dv, Tv])
                k = len(keep)

                def stat(A):
                    t = A[:, k:].sum(0)
                    return A[:, :k].sum(0) / np.where(t > 0, t, np.nan) * 1e4

                obs = stat(X)
                bs = panel_boot(X, stat, B=B, seed=21)
                bm = np.nanmean(bs, 1)
                lo, hi = np.nanquantile(bm, [0.025, 0.975])
                p = float(np.mean(np.abs(bm - np.nanmean(obs)) >= abs(np.nanmean(obs))))
                kpos, nn, ps = sign_test(obs)
                for s_, v in zip(keep, obs):
                    rows.append(dict(strategy=strat, kind=kind, era=era, sym=s_, bp_per_trade=v,
                                     trades=float(Tn[s_].sum()), pooled=False))
                rows.append(dict(strategy=strat, kind=kind, era=era, sym="mean", bp_per_trade=float(np.nanmean(obs)),
                                 lo=lo, hi=hi, p_boot=max(p, 1 / B), n_positive=kpos, n=nn, sign_p=ps,
                                 trades=float(Tv.sum()), pooled=True))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "era_pooled.csv", index=False)
    return df


def paired_gap(daily: dict, daily_idx: dict) -> pd.DataFrame:
    """Index (on its ETF's dates) minus ETF: per-trade delay cost gap, pooled over the 10 pairs."""
    rows = []
    for strat in ["MA200", "HMM"]:
        cols, k = [], len(INDICES)
        D = {}
        for n in INDICES:
            D[f"i_{n}"] = daily_idx[(strat, n)]
        for n in INDICES:
            D[f"e_{n}"] = daily[(strat, ETF_OF[n], "close+1")]
        for n in INDICES:
            D[f"ti_{n}"] = TURN[(strat, n, "etf dates")]
        for n in INDICES:
            D[f"te_{n}"] = TURN[(strat, ETF_OF[n])]
        X = pd.DataFrame(D)
        # common dates only: both series defined
        X = X.fillna(0.0).values

        def stat(A):
            li = A[:, :k].sum(0) / A[:, 2 * k:3 * k].sum(0)
            le = A[:, k:2 * k].sum(0) / A[:, 3 * k:].sum(0)
            return (li - le) * 1e4

        obs = stat(X)
        bs = panel_boot(X, stat, B=B, seed=31)
        bm = bs.mean(1)
        lo, hi = np.quantile(bm, [0.025, 0.975])
        p = float(np.mean(np.abs(bm - obs.mean()) >= abs(obs.mean())))
        kpos, nn, ps = sign_test(obs)
        for n, v in zip(INDICES, obs):
            rows.append(dict(strategy=strat, pair=f"{n}-{ETF_OF[n]}", gap_bp=v))
        rows.append(dict(strategy=strat, pair="mean", gap_bp=float(obs.mean()), lo=lo, hi=hi, p_boot=max(p, 1 / B),
                         n_positive=kpos, n=nn, sign_p=ps))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "paired_gap.csv", index=False)
    return df


# ------------------------------------------------------------------------------------- 6
def profile_and_mechanism():
    prof, mech = [], []
    for sym in INDICES + ETFS:
        kind = "ETF" if sym in ETFS else "index"
        m, g, px = instrument(sym)
        ex = pd.Series(np.nan_to_num(m.close_ret[:, 0]) - m.rf, m.index)
        for strat, s in signals(sym, px).items():
            st = eval_start(s, m)
            d = s.reindex(m.index).fillna(0.0).diff()
            d = d[st:]
            flips = d[d != 0]
            for h in range(0, 11):
                exh = ex.shift(-h).reindex(flips.index)
                ok = exh.notna()
                prof.append(dict(strategy=strat, kind=kind, sym=sym, h=h,
                                 bp=float((flips[ok] * exh[ok]).sum() / flips[ok].abs().sum() * 1e4),
                                 n=int(ok.sum())))
            for era, a, b in ERAS:
                e = ex[st:]
                e = e[(e.index >= pd.Timestamp(a)) if a else slice(None)]
                if b:
                    e = e[e.index <= pd.Timestamp(b)]
                if len(e) < 750:
                    continue
                f = flips.reindex(e.index).dropna()
                f = f[f != 0]
                if len(f) < 15:
                    continue
                rho = float(e.autocorr(1))
                trig = float((f * ex.reindex(f.index)).sum() / f.abs().sum())
                nxt = ex.shift(-1).reindex(f.index)
                act = float((f * nxt).sum() / f.abs().sum())
                mech.append(dict(strategy=strat, kind=kind, sym=sym, era=era, days=len(e), trades=float(f.abs().sum()),
                                 rho1=rho, trigger_bp=trig * 1e4, predicted_bp=rho * trig * 1e4, actual_bp=act * 1e4,
                                 se_bp=float((f * nxt).std() / np.sqrt(len(f)) * 1e4)))
    prof, mech = pd.DataFrame(prof), pd.DataFrame(mech)
    prof.to_csv(RES / "profile.csv", index=False)
    mech.to_csv(RES / "mechanism.csv", index=False)
    # weighted least squares of actual on predicted, weights = trades (precision ~ number of trades)
    fits = []
    for lab, q in [("all cells", mech), ("MA200 cells", mech[mech.strategy == "MA200"]),
                   ("HMM cells", mech[mech.strategy == "HMM"])]:
        X = np.column_stack([np.ones(len(q)), q.predicted_bp.values])
        w = q.trades.values
        y = q.actual_bp.values
        Wm = np.diag(w)
        beta = np.linalg.solve(X.T @ Wm @ X, X.T @ Wm @ y)
        res = y - X @ beta
        # heteroskedasticity-robust (HC1) standard errors for WLS
        XtWX_inv = np.linalg.inv(X.T @ Wm @ X)
        meat = X.T @ np.diag((w * res) ** 2) @ X
        V = XtWX_inv @ meat @ XtWX_inv * len(y) / (len(y) - 2)
        se = np.sqrt(np.diag(V))
        r2 = 1 - np.sum(w * res ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)
        corr = float(np.corrcoef(q.predicted_bp, q.actual_bp)[0, 1])
        fits.append(dict(sample=lab, cells=len(q), intercept=beta[0], intercept_se=se[0], slope=beta[1],
                         slope_se=se[1], weighted_r2=r2, corr=corr))
    fits = pd.DataFrame(fits)
    fits.to_csv(RES / "mechanism_fit.csv", index=False)
    return prof, mech, fits


# ------------------------------------------------------------------------------------- 7
def momentum_market(univ: str, opens: bool = False):
    if univ == "A":
        ex, rf = data.industries()
        ex = ex[:"2026-09-30"]
        rf = rf.reindex(ex.index)
        m = Market(ex.index, list(ex.columns), ex.values + rf.values[:, None], np.full(ex.shape, np.nan),
                   rf.values, ex.notna().values)
        return m, ex
    d = data.country_etfs()
    d = d[(d.ticker != "SPY") & (d.ticker != "EWUS")]
    d = d[d.date <= "2026-10-31"]
    panels = {t: g.set_index("date").sort_index() for t, g in d.groupby("ticker")}
    if opens:
        panels = {t: g[data.real_open_mask(g)] for t, g in panels.items()}
        panels = {t: g for t, g in panels.items() if len(g) > 300}
    ix = pd.DatetimeIndex(sorted(set().union(*[g.index for g in panels.values()])))
    rf = data.cash_for("SPY", ix, True)
    m = data.market_from_ohlc(panels, rf, ix)
    ex = pd.DataFrame(m.close_ret - m.rf[:, None], index=ix, columns=m.assets)
    if opens:
        # where an asset has no real open on a rebalancing day, the open fill is impossible: treat the
        # missing overnight piece as 0 only for days the asset is not held (checked by the engine)
        pass
    return m, ex


_MOM = {}


def momentum_run(univ: str, fill: str, opens: bool = False, cost_bps: float = 0.0) -> dict:
    key = (univ, opens)
    if key not in _MOM:
        m, ex = momentum_market(univ, opens)
        tgt = momentum_targets(ex)
        _MOM[key] = (m, ex, tgt)
    m, ex, tgt = _MOM[key]
    if fill == "open":
        on = m.on_ret.copy()
        held_or_traded = np.abs(np.nan_to_num(tgt.values)) > 0
        m = Market(m.index, m.assets, m.close_ret, np.where(np.isnan(on), 0.0, on), m.rf, m.tradeable)
    r = run(m, tgt.values, fill, cost_bps)
    first = tgt.dropna(how="all").index[0]
    x = r.excess_gross[first:]
    mo = (1 + x).groupby(x.index.to_period("M")).prod() - 1
    mo = mo.iloc[1:]                                   # first month: book bought at its end
    turn = float(r.turnover[first:].sum() / len(mo))
    return dict(univ=univ, fill=fill, start=str(mo.index[0]), months=len(mo), gross_pct_yr=float(mo.mean() * 1200),
                sharpe_gross=float(mo.mean() / mo.std() * np.sqrt(12)), turnover_mo=turn,
                be_bps=float(mo.mean() / turn * 1e4), daily=x, r=r)


def momentum() -> pd.DataFrame:
    rows = []
    for univ, opens, fills in [("A", False, ["close", "close+1", "close+2", "close+5"]),
                               ("B", False, ["close", "close+1", "close+2"]),
                               ("B", True, ["close", "open", "close+1"])]:
        base = None
        for f in fills:
            res = momentum_run(univ, f, opens)
            if base is None:
                base = res
            dpy = (res["daily"] - base["daily"]).dropna()
            mean, lo, hi, p = boot_mean_ci(dpy.values, B=B, seed=11)
            per_trade = -(dpy.sum()) / (base["turnover_mo"] * base["months"]) * 1e4
            rows.append(dict(universe=univ, sample="real opens" if opens else "full", fill=f, start=res["start"],
                             months=res["months"], gross_pct_yr=res["gross_pct_yr"], sharpe_gross=res["sharpe_gross"],
                             turnover_mo=res["turnover_mo"], be_bps=res["be_bps"],
                             delay_cost_bp_per_trade=per_trade, d_pct_yr=mean * TDAYS * 100,
                             d_lo=lo * TDAYS * 100, d_hi=hi * TDAYS * 100, p=p))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "momentum.csv", index=False)
    return df


def claims(lat: pd.DataFrame) -> pd.DataFrame:
    """Project 03: does MA200 beat buy & hold (net Sharpe)? Project 19: does the real-time HMM trail it?"""
    rows = []
    for sym in INDICES + ETFS:
        kind = "ETF" if sym in ETFS else "index"
        m, g, px = instrument(sym)
        bh = pd.Series(1.0, index=m.index)
        for strat, s in signals(sym, px).items():
            st = eval_start(s, m)
            for f in FILLS + (["close (open sample)", "open"] if kind == "ETF" else []):
                if f in ("open", "close (open sample)"):
                    mo, _, _ = instrument(sym, opens=True)
                    st_o = max(st, mo.index[1])
                    a = run_timing(mo, s.reindex(mo.index), "open" if f == "open" else "close", COST).excess_net[st_o:]
                    b = run_timing(mo, bh.reindex(mo.index), "close", COST).excess_net[st_o:]
                else:
                    a = run_timing(m, s, f, COST).excess_net[st:]
                    b = run_timing(m, bh, "close", COST).excess_net[st:]
                b = b.iloc[1:]
                a = a.reindex(b.index)
                d, p = sharpe_diff_hac(a.values, b.values)
                rows.append(dict(strategy=strat, kind=kind, sym=sym, fill=f, sharpe_strategy=sr(a),
                                 sharpe_buy_hold=sr(b), diff=d, p=p))
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for _, gidx in df.groupby(["strategy", "kind", "fill"]).groups.items():
        df.loc[gidx, "p_holm"] = holm(df.loc[gidx, "p"].values)
    df.to_csv(RES / "claims.csv", index=False)
    summ = df.groupby(["strategy", "kind", "fill"]).agg(median_diff=("diff", "median"),
                                                          n_beats_bh=("diff", lambda v: int((v > 0).sum())),
                                                          n=("diff", "size"),
                                                          n_sig_holm=("p_holm", lambda v: int((v < 0.05).sum()))).reset_index()
    summ.to_csv(RES / "claims_summary.csv", index=False)
    return df


# ------------------------------------------------------------------------------------- 8
def stress(daily: dict) -> pd.DataFrame:
    rows = []
    for (strat, sym, fill), d in daily.items():
        for w, (a, b) in STRESS.items():
            x = d[a:b]
            if len(x) < 10:
                continue
            rows.append(dict(strategy=strat, sym=sym, kind="ETF" if sym in ETFS else "index", fill=fill, window=w,
                             delay_pnl_pct=float(x.sum() * 100), share_of_total=float(x.sum() / d.sum()) if d.sum() else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "stress.csv", index=False)
    return df


def delay_curve() -> pd.DataFrame:
    rows = []
    for sym in INDICES + ETFS:
        kind = "ETF" if sym in ETFS else "index"
        m, g, px = instrument(sym)
        for strat, s in signals(sym, px).items():
            st = eval_start(s, m)
            base = None
            for k in range(0, 11):
                f = "close" if k == 0 else f"close+{k}"
                r = run_timing(m, s, f, COST)
                v = sr(r.excess_net[st + pd.Timedelta(days=20):])
                base = v if base is None else base
                rows.append(dict(strategy=strat, kind=kind, sym=sym, k=k, sharpe_net=v, d_sharpe=v - base))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "delay_curve.csv", index=False)
    return df


if __name__ == "__main__":
    import time
    t0 = time.time()
    print(validation().round(4).to_string(), flush=True)
    print(audit().to_string(), flush=True)
    lat, dl, daily = lattice_and_delay()
    print(dl.round(2).to_string(), flush=True)
    idx_df, daily_idx = same_period_index()
    print(idx_df.round(2).to_string(), flush=True)
    print(pooled(daily, daily_idx, dl).round(3).to_string(), flush=True)
    print(era_pooled(daily, daily_idx).query("pooled").round(2).to_string(), flush=True)
    print(paired_gap(daily, daily_idx).round(2).to_string(), flush=True)
    prof, mech, fits = profile_and_mechanism()
    print(fits.round(3).to_string(), flush=True)
    print(momentum().round(3).to_string(), flush=True)
    claims(lat)
    print(pd.read_csv(RES / "claims_summary.csv").round(3).to_string(), flush=True)
    st = stress(daily)
    print(st.groupby(["strategy", "kind", "fill", "window"]).delay_pnl_pct.agg(["mean", "median"]).round(2), flush=True)
    delay_curve()
    print(f"done in {time.time() - t0:.0f}s")
