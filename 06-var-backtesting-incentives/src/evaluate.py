"""
Part B - the empirical evidence. Reads results/forecasts.csv.gz (from backtest.py) and writes:

  backtests.csv          coverage / independence / DQ / ES (Z2) tests per series x model, with Holm + BH
  scores.csv             mean tick and FZ0 loss, DM test against the best model per series
  capital_curves.csv     Basel 2.5 and FRTB charge vs reporting scale lambda, per series x model
  shading.csv            honest scale vs capital-optimal scale (ex post, red-zone constrained)
  adaptive.csv           an ex-ante "exception budget" shading rule, run out of sample
  model_choice.csv       which model a capital-minimising bank picks vs which model the FZ0 score picks
  stress.csv             exceptions and charge in 2008-09, 2020 and 2022
  tail_index.csv         Hill tail index of GARCH-standardised losses per series
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from backtests import (BASEL_PLUS, FRTB_PLUS, QUARTER, RED, acerbi_szekely_z2, capital_path,
                       christoffersen_ind, dm_test, dq_test, fz0_loss, kupiec, tick_loss,
                       trailing_exceptions)
from models import ALPHA_ES, ALPHA_VAR, MODEL_ORDER

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SERIES = ["SPY", "EEM", "TLT", "HYG", "GLD", "USO", "URA", "BOOK"]
LAMS = np.round(np.arange(0.50, 1.501, 0.01), 2)
STRESS = {"GFC 2008-09": ("2008-09-01", "2009-03-31"), "COVID 2020": ("2020-02-19", "2020-04-30"),
          "Rates 2022": ("2022-01-03", "2022-10-31")}
B_BOOT = 200


def load() -> pd.DataFrame:
    return pd.read_csv(RES / "forecasts.csv.gz", parse_dates=["date"])


def holm(p: np.ndarray) -> np.ndarray:
    o = np.argsort(p); m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[o])
    out = np.empty(m); out[o] = np.minimum(adj, 1)
    return out


def bh(p: np.ndarray) -> np.ndarray:
    o = np.argsort(p); m = len(p)
    adj = np.minimum.accumulate((m / np.arange(1, m + 1) * p[o])[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(adj, 1)
    return out


# ------------------------------------------------------------------ 1. classical backtests

def backtest_table(df):
    rows = []
    for (s, m), g in df.groupby(["series", "model"], sort=False):
        r, v, v2, e = g.ret.values, g.var99.values, g.var975.values, g.es975.values
        h = r < -v
        lr_uc, p_uc = kupiec(h, ALPHA_VAR)
        lr_ind, p_ind = christoffersen_ind(h)
        p_cc = float(stats.chi2.sf(lr_uc + lr_ind, 2))
        dq, p_dq = dq_test(h, v, ALPHA_VAR)
        z2, p_z2 = acerbi_szekely_z2(r, v2, e, ALPHA_ES)
        rows.append(dict(series=s, model=m, n=len(r), exceptions=int(h.sum()), exc_rate=h.mean(),
                         exc_rate_975=(r < -v2).mean(), p_kupiec=p_uc, p_christ=p_ind, p_cc=p_cc,
                         p_dq=p_dq, z2=z2, p_z2=p_z2))
    t = pd.DataFrame(rows)
    for c in ["p_kupiec", "p_cc", "p_dq", "p_z2"]:
        t[c + "_holm"] = holm(t[c].values)
        t[c + "_bh"] = bh(t[c].values)
    return t


# ------------------------------------------------------------------ 2. scoring

def score_table(df):
    df = df.assign(tick=tick_loss(df.ret.values, df.var99.values, ALPHA_VAR),
                   fz0=fz0_loss(df.ret.values, df.var975.values, df.es975.values, ALPHA_ES))
    rows = []
    for s, g in df.groupby("series", sort=False):
        piv = g.pivot(index="date", columns="model", values="fz0")[MODEL_ORDER]
        best = piv.mean().idxmin()
        for m in MODEL_ORDER:
            t, p = dm_test((piv[m] - piv[best]).values) if m != best else (0.0, 1.0)
            rows.append(dict(series=s, model=m, fz0=piv[m].mean(),
                             tick=g[g.model == m].tick.mean() * 1e4, best_fz0=best, dm_t=t, dm_p=p))
    return pd.DataFrame(rows)


def honest_scale(r, v, v2, e):
    """Scale factors a forecaster would choose to minimise strictly consistent losses."""
    lt = [tick_loss(r, lam * v, ALPHA_VAR).mean() for lam in LAMS]
    lf = [fz0_loss(r, lam * v2, lam * e, ALPHA_ES).mean() for lam in LAMS]
    return LAMS[int(np.argmin(lt))], LAMS[int(np.argmin(lf))], np.array(lt), np.array(lf)


# ------------------------------------------------------------------ 3. capital vs reporting scale

def regimes(v, e):
    return {"Basel 2.5": (np.sqrt(10) * v, 3.0, BASEL_PLUS), "FRTB": (np.sqrt(10) * e, 1.5, FRTB_PLUS)}


def constrained_optimum(out):
    """Cheapest reporting scale that spends no more quarters in the red zone than honest reporting."""
    c = out["charge"] / out["charge"][LAMS == 1.0][0]
    ok = out["red"] <= out["red"][LAMS == 1.0][0] + 1e-12
    j = np.where(ok)[0][np.argmin(c[ok])] if ok.any() else int(np.argmax(LAMS == 1.0))
    return LAMS[j], 1 - c[j]


def block_indices(T, rng, block=250):
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T - block, nb)
    return np.concatenate([np.arange(st, st + block) for st in starts])[:T]


def capital_cell(args):
    """Capital curves and shading statistics for one series x model (run in a worker process).
    The bootstrap seed depends only on the cell, so results do not depend on scheduling."""
    s, m, g = args
    rng = np.random.default_rng([20261121, SERIES.index(s), MODEL_ORDER.index(m)])
    curves, shade = [], []
    if True:
        r, v, v2, e = g.ret.values, g.var99.values, g.var975.values, g.es975.values
        lam_tick, lam_fz, lt, lf = honest_scale(r, v, v2, e)
        for reg, (risk, base, table) in regimes(v, e).items():
            out = capital_path(r, v, risk, LAMS, base, table)
            rel = out["charge"] / out["charge"][LAMS == 1.0][0]
            curves.append(pd.DataFrame(dict(series=s, model=m, regime=reg, lam=LAMS, rel_charge=rel,
                                            red=out["red"], yellow=out["yellow"], exc_rate=out["exc_rate"],
                                            tick=lt, fz0=lf)))
            lam_star, saving = constrained_optimum(out)
            # moving-block bootstrap (250-day blocks) of the capital-optimal scale
            boot = []
            for _ in range(B_BOOT if reg == "Basel 2.5" else 100):
                idx = block_indices(len(r), rng)
                ob = capital_path(r[idx], v[idx], risk[idx], LAMS, base, table)
                boot.append(constrained_optimum(ob))
            boot = np.array(boot)
            shade.append(dict(series=s, model=m, regime=reg, lam_honest_tick=lam_tick, lam_honest_fz0=lam_fz,
                              lam_capital=lam_star, saving=saving,
                              lam_capital_lo=np.quantile(boot[:, 0], 0.025),
                              lam_capital_hi=np.quantile(boot[:, 0], 0.975),
                              saving_lo=np.quantile(boot[:, 1], 0.025), saving_hi=np.quantile(boot[:, 1], 0.975),
                              gap=lam_tick - lam_star, red_share_honest=out["red"][LAMS == 1.0][0],
                              saving_vs_calibrated=1 - rel[LAMS == lam_star][0] / rel[LAMS == lam_tick][0]))
    print(s, m, "done", flush=True)
    return curves, shade


def capital_analysis(df):
    cells = [(s, m, g) for (s, m), g in df.groupby(["series", "model"], sort=False)]
    with Pool(4) as pool:
        res = pool.map(capital_cell, cells)
    curves = [c for cs, _ in res for c in cs]
    shade = [x for _, sh in res for x in sh]
    return pd.concat(curves, ignore_index=True), pd.DataFrame(shade)


# ------------------------------------------------------------------ 4. an ex-ante shading rule

def adaptive_shading(r, v, risk, base, table, budget=7, burn=250, floor=0.5):
    """Each quarter the bank picks the smallest lambda such that its *past* 250 days would have
    produced at most `budget` exceptions (i.e. it aims for the yellow zone, never the red one).
    Uses only information available at the reset date. Returns charge relative to honest reporting,
    red/yellow quarter shares, and the lambda path."""
    T = len(r)
    lam_t = np.ones(T)
    grid = np.round(np.arange(floor, 1.001, 0.01), 2)
    for q0 in range(burn, T, QUARTER):
        past = slice(q0 - 250, q0)
        exc = (r[None, past] < -grid[:, None] * v[None, past]).sum(axis=1)
        ok = grid[exc <= budget - 0]
        lam_t[q0:q0 + QUARTER] = ok.min() if len(ok) else 1.0
    hits = (r < -lam_t * v).astype(float)
    N = trailing_exceptions(hits[None, :])[0]
    resets = np.arange(burn, T, QUARTER)

    def charge(lam_path, hits_):
        Nn = trailing_exceptions(hits_[None, :])[0]
        qi = np.clip(np.arange(T) - (np.arange(T) - burn) % QUARTER, 0, T - 1)
        mult = base + table[Nn[qi].astype(int)]
        R = lam_path * risk
        cs = np.concatenate([[0.0], np.cumsum(R)])
        t = np.arange(T); lo = np.maximum(t - 60, 0)
        avg = (cs[t] - cs[lo]) / np.maximum(t - lo, 1)
        prev = np.concatenate([R[:1], R[:-1]])
        return np.maximum(prev, mult * avg)[burn:].mean()

    c_shade = charge(lam_t, hits)
    c_honest = charge(np.ones(T), (r < -v).astype(float))
    return dict(rel_charge=c_shade / c_honest, red=(N[resets] >= RED).mean(),
                yellow=((N[resets] >= 5) & (N[resets] < RED)).mean(), mean_lam=lam_t[burn:].mean(),
                exc_rate=hits[burn:].mean()), lam_t


def adaptive_table(df):
    rows, paths = [], []
    for (s, m), g in df.groupby(["series", "model"], sort=False):
        r, v, e = g.ret.values, g.var99.values, g.es975.values
        for reg, (risk, base, table) in regimes(v, e).items():
            for budget in (4, 7, 9):
                res, lam_t = adaptive_shading(r, v, risk, base, table, budget=budget)
                rows.append(dict(series=s, model=m, regime=reg, budget=budget, **res))
                if reg == "Basel 2.5" and budget == 7:
                    paths.append(pd.DataFrame(dict(date=g.date.values, series=s, model=m, lam=lam_t)))
    return pd.DataFrame(rows), pd.concat(paths)


# ------------------------------------------------------------------ 5. model choice

def model_choice(df, scores):
    rows = []
    for s, g in df.groupby("series", sort=False):
        for reg in ("Basel 2.5", "FRTB"):
            cand = []
            for m in MODEL_ORDER:
                gm = g[g.model == m]
                risk, base, table = regimes(gm.var99.values, gm.es975.values)[reg]
                out = capital_path(gm.ret.values, gm.var99.values, risk, np.array([1.0]), base, table)
                cand.append(dict(model=m, charge=out["charge"][0], red=out["red"][0], yellow=out["yellow"][0]))
            c = pd.DataFrame(cand)
            adm = c[c.red == 0]
            pick = (adm if len(adm) else c).sort_values("charge").iloc[0]
            sc = scores[scores.series == s].set_index("model")
            best = sc.fz0.idxmin()
            best_charge = c.set_index("model").charge[best]
            rows.append(dict(series=s, regime=reg, capital_pick=pick.model, fz0_best=best,
                             same=pick.model == best, charge_saving_vs_fz0_best=1 - pick.charge / best_charge,
                             fz0_penalty_of_pick=sc.fz0[pick.model] - sc.fz0[best],
                             dm_p_pick_vs_best=sc.dm_p[pick.model],
                             n_red_free=int(len(adm))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 6. stress and tails

def stress_table(df):
    rows = []
    for name, (a, b) in STRESS.items():
        w = df[(df.date >= a) & (df.date <= b)]
        for (s, m), g in w.groupby(["series", "model"], sort=False):
            rows.append(dict(window=name, series=s, model=m, days=len(g),
                             exceptions=int((g.ret < -g.var99).sum()),
                             expected=len(g) * ALPHA_VAR,
                             worst_ratio=float((-g.ret / g.var99).max())))
    return pd.DataFrame(rows)


def hill(x, k):
    xs = np.sort(x)[::-1]
    return 1.0 / np.mean(np.log(xs[:k] / xs[k]))


def tail_table(df):
    """Hill estimate of the tail index of losses standardised by the GARCH-t volatility (5% tail)."""
    rows = []
    for s, g in df[df.model == "GARCH-t"].groupby("series", sort=False):
        z = -g.ret.values / g.var99.values      # loss in units of VaR: scale-free
        k = int(0.05 * len(z))
        rows.append(dict(series=s, hill_alpha=hill(z, k), k=k))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load()
    df["series"] = pd.Categorical(df.series, SERIES, ordered=True)
    df["model"] = pd.Categorical(df.model, MODEL_ORDER, ordered=True)
    df = df.sort_values(["series", "model", "date"]).reset_index(drop=True)
    df["series"] = df.series.astype(str); df["model"] = df.model.astype(str)

    bt = backtest_table(df); bt.to_csv(RES / "backtests.csv", index=False, float_format="%.5f")
    sc = score_table(df); sc.to_csv(RES / "scores.csv", index=False, float_format="%.6f")
    print(bt[["series", "model", "exc_rate", "p_kupiec_holm", "p_dq_holm", "p_z2_holm"]].round(3).to_string())
    tail_table(df).to_csv(RES / "tail_index.csv", index=False, float_format="%.4f")
    stress_table(df).to_csv(RES / "stress.csv", index=False, float_format="%.4f")
    mc = model_choice(df, sc); mc.to_csv(RES / "model_choice.csv", index=False, float_format="%.5f")
    print(mc.to_string())
    ad, paths = adaptive_table(df)
    ad.to_csv(RES / "adaptive.csv", index=False, float_format="%.5f")
    paths.to_csv(RES / "adaptive_lambda_paths.csv.gz", index=False, float_format="%.3f", compression="gzip")
    print(ad.groupby(["regime", "budget"])[["rel_charge", "red", "yellow", "mean_lam"]].median())
    curves, shade = capital_analysis(df)
    curves.to_csv(RES / "capital_curves.csv.gz", index=False, float_format="%.6f", compression="gzip")
    shade.to_csv(RES / "shading.csv", index=False, float_format="%.5f")
    print(shade.round(3).to_string())
