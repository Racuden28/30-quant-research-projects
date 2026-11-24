"""
Pairs trading with three hedge ratios on the same pairs, on the real market and on project 11's placebo market.

Design (project 11, unchanged): a portfolio starts on the first trading day of every month; formation is the
previous 252 trading days, trading the next 126; an asset is eligible if it has a return on every day of the
window and fewer than 5 % zero returns in formation (industries also need five firms). The 20 pairs with the
smallest sum of squared differences of normalised prices are selected (distance rule) and are then traded with
three hedge ratios, so that only the hedge ratio differs:

  S   static: OLS of log price A on log price B over formation; z = residual / formation residual s.d.
  R   rolling: the same OLS on the last 63 days, re-estimated every day (coefficients from days < t);
      z = today's residual / the s.d. of those 63 residuals
  K   Kalman: intercept and slope follow independent random walks; their noise ratios are chosen by profile
      likelihood on formation; the state is carried into trading; z = innovation / its predicted s.d.

Ablations: K with Chan's (2013) fixed settings (delta = 1e-4, Ve = 1e-3), with Chan's 1-s.d. entry, and with the
legs rebalanced to the current hedge ratio every day; and every rule on the pairs selected by Engle-Granger instead.

Placebo N1 (project 11): every asset of the window is given the same drift, then whole days are resampled IID
with replacement (10 replications per window, seeded by window). It keeps volatilities, fat tails and the
correlation matrix and removes any memory in relative prices.

Output: results/pairs_<universe>.npz (daily P&L and turnover per window, rule, source and replication, trade
statistics and the spread decomposition) and results/pairs_adf_<universe>.csv.
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import norm
from statsmodels.tsa import adfvalues as _adf

from data import DATA, RES, pairs_panel
from kalman import BURN, kf2_loglik_grid, kf2_run, kf2_run_abs

FORM, TRADE, NPAIRS = 252, 126, 20
MAX_ZERO_SHARE = 0.05
OPEN_Z = 2.0
ROLL_W = 63
NREP = 10
RA_GRID = np.r_[0.0, np.logspace(-4, 1, 11)]
RB_GRID = np.r_[0.0, np.logspace(-4, 2, 13)]
CHAN = dict(qa=1e-4 / (1 - 1e-4), qb=1e-4 / (1 - 1e-4), Ve=1e-3)
UNIVERSE_ID = {"industries": 1, "countries": 2}
# rule name -> (spread, entry z, rebalance legs daily, wait in days)
RULES = {"S": ("S", OPEN_Z, False, 1), "R": ("R", OPEN_Z, False, 1), "K": ("K", OPEN_Z, False, 1),
         "K-rebal": ("K", OPEN_Z, True, 1), "K-chan": ("C", OPEN_Z, False, 1), "K-chan-1sd": ("C", 1.0, False, 1),
         "R-rebal": ("R", OPEN_Z, True, 1), "K-wait5": ("K", OPEN_Z, False, 5), "REV1": ("V", OPEN_Z, False, 1)}
NSTAT = 9   # opened, converged, pnl converged, pnl other, days open, sum closure, sum price part, n decomposed, n beta<=0 skips


# ----------------------------------------------------------------------------------------------------------
# Selection (project 11)
# ----------------------------------------------------------------------------------------------------------
def pair_index(n):
    ia, ib = np.triu_indices(n, k=1)
    return ia.astype(np.int64), ib.astype(np.int64)


def select_distance(rf, ia, ib):
    p = np.cumprod(1.0 + rf, axis=0)
    s = p[:, ia] - p[:, ib]
    return np.argsort((s * s).sum(axis=0), kind="stable")[:NPAIRS]


def eg_tstats(rf, ia, ib):
    lp = np.cumsum(np.log1p(rf), axis=0)
    y, x = lp[:, ia], lp[:, ib]
    xc, yc = x - x.mean(0), y - y.mean(0)
    beta = (xc * yc).sum(0) / (xc * xc).sum(0)
    e = yc - beta * xc
    return beta, adf_tau(e)


def adf_tau(e):
    """ADF t-statistic with one lagged difference and no constant, vectorised over columns (project 11)."""
    de = np.diff(e, axis=0)
    yv, x1, x2 = de[1:], e[1:-1], de[:-1]
    s11, s22, s12 = (x1 * x1).sum(0), (x2 * x2).sum(0), (x1 * x2).sum(0)
    s1y, s2y = (x1 * yv).sum(0), (x2 * yv).sum(0)
    det = s11 * s22 - s12 * s12
    rho = (s22 * s1y - s12 * s2y) / det
    g = (s11 * s2y - s12 * s1y) / det
    u = yv - rho * x1 - g * x2
    sig2 = (u * u).sum(0) / (yv.shape[0] - 2)
    return rho / np.sqrt(sig2 * s22 / det)


def adf_tau_c(e):
    """ADF t-statistic with a constant and one lagged difference (for a series that is not a regression
    residual, such as Kalman innovations), vectorised over columns."""
    de = np.diff(e, axis=0)
    yv, x1, x2 = de[1:], e[1:-1], de[:-1]
    n, P = yv.shape
    out = np.empty(P)
    for j in range(P):
        X = np.column_stack([np.ones(n), x1[:, j], x2[:, j]])
        XtX = X.T @ X
        b = np.linalg.solve(XtX, X.T @ yv[:, j])
        u = yv[:, j] - X @ b
        s2 = u @ u / (n - 3)
        out[j] = b[1] / np.sqrt(s2 * np.linalg.inv(XtX)[1, 1])
    return out


def mackinnon(t, N):
    """MacKinnon (1994) approximate p-value, constant case, N variables (statsmodels coefficients)."""
    t = np.asarray(t, float)
    smallp, largep = np.asarray(_adf.tau_c_smallp[N - 1]), np.asarray(_adf.tau_c_largep[N - 1])
    p = np.where(t <= _adf.tau_star_c[N - 1], norm.cdf(np.polyval(smallp[::-1], t)),
                 norm.cdf(np.polyval(largep[::-1], t)))
    p = np.where(t > _adf.tau_max_c[N - 1], 1.0, p)
    return np.where(t < _adf.tau_min_c[N - 1], 0.0, p)


# ----------------------------------------------------------------------------------------------------------
# Spreads: z (T,), the spread e (T,), the coefficients used for it (a, b), all known at the close of day t
# ----------------------------------------------------------------------------------------------------------
def spread_static(y, x):
    yf, xf = y[:FORM], x[:FORM]
    b = np.cov(xf, yf, ddof=1)[0, 1] / np.var(xf, ddof=1)
    a = yf.mean() - b * xf.mean()
    e = y - a - b * x
    sd = e[:FORM].std(ddof=1)
    T = len(y)
    return e / sd, e, np.full(T, a), np.full(T, b)


@njit(cache=True)
def spread_rolling(y, x, w):
    T = len(y)
    z = np.full(T, np.nan)
    e = np.full(T, np.nan)
    A = np.full(T, np.nan)
    Bc = np.full(T, np.nan)
    for t in range(w, T):
        xs, ys = x[t - w:t], y[t - w:t]
        mx, my = xs.mean(), ys.mean()
        cxx = ((xs - mx) ** 2).sum()
        b = ((xs - mx) * (ys - my)).sum() / cxx
        a = my - b * mx
        res = ys - a - b * xs
        sd = np.sqrt((res ** 2).sum() / (w - 2))
        e[t] = y[t] - a - b * x[t]
        z[t] = e[t] / sd
        A[t] = a
        Bc[t] = b
    return z, e, A, Bc


def spread_kalman(y, x):
    ll = kf2_loglik_grid(y[:FORM], x[:FORM], RA_GRID, RB_GRID)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    A, Bp, V, F, _, _ = kf2_run(y, x, RA_GRID[i], RB_GRID[j])
    Rh = np.mean(V[BURN:FORM] ** 2 / F[BURN:FORM])
    return V / np.sqrt(F * Rh), V, A, Bp, (RA_GRID[i], RB_GRID[j])


def spread_reversal(y, x):
    """Benchmark for what the Kalman rule turns into: today's relative log return of the pair in units of its
    formation s.d. (a one-day reversal signal), with a 1 : 1 hedge."""
    d = np.r_[0.0, np.diff(y - x)]
    sd = d[1:FORM].std(ddof=1)
    T = len(y)
    return d / sd, d, np.zeros(T), np.ones(T)


def spread_chan(y, x):
    A, Bp, V, F, _, _ = kf2_run_abs(y, x, CHAN["qa"], CHAN["qb"], CHAN["Ve"], 0.0, 0.0, 0.0)
    return V / np.sqrt(F), V, A, Bp


# ----------------------------------------------------------------------------------------------------------
# Trading simulator (project 11's, generalised to a hedge ratio that changes every day)
# ----------------------------------------------------------------------------------------------------------
@njit(cache=True)
def simulate(rt, z, bt, e, y, x, A, Bc, open_z, wait, rebal):
    """One pair over the trading period.

    rt (T, 2) simple returns of legs A and B; z, bt, e, y, x, A, Bc (T,) the signal, the hedge ratio, the spread,
    the log prices and the coefficients, all known at the close of day t. Decisions at the close of t are traded
    at the close of t + wait. A long-spread position is long A, short B, with legs 2/(1+b) and 2b/(1+b) per $1 of
    committed capital. No position is opened when b <= 0.
    Returns daily pnl (T,), turnover (T,), stats (NSTAT,)."""
    T = len(z)
    pnl = np.zeros(T)
    turn = np.zeros(T)
    st = np.zeros(9)
    want = np.zeros(T, np.int64)
    conv = np.zeros(T, np.int64)
    t_open = -1
    pos = 0
    for t in range(T):
        zt = z[t]
        if np.isnan(zt):
            want[t] = pos
            continue
        if pos == 0:
            if bt[t] <= 0.0:
                if abs(zt) > open_z:
                    st[8] += 1
            elif zt > open_z:
                pos = -1
                t_open = t
            elif zt < -open_z:
                pos = 1
                t_open = t
        elif (pos == 1 and zt >= 0.0) or (pos == -1 and zt <= 0.0):
            pos = 0
            conv[t] = 1
            # decomposition of the spread's move between the two decisions
            tot = e[t] - e[t_open]
            price = (y[t] - y[t_open]) - Bc[t_open] * (x[t] - x[t_open])
            st[5] += abs(tot)
            st[6] += price * np.sign(tot)
            st[7] += 1
        want[t] = pos
    ex = np.zeros(T, np.int64)
    exc = np.zeros(T, np.int64)
    wa_d = np.zeros(T)
    wb_d = np.zeros(T)
    for t in range(T):
        if t - wait >= 0:
            s = t - wait
            ex[t] = want[s]
            exc[t] = conv[s]
            b = max(bt[s], 0.0)
            wa_d[t] = 2.0 / (1.0 + b)
            wb_d[t] = 2.0 * b / (1.0 + b)
    ex[T - 1] = 0
    va = 0.0
    vb = 0.0
    cur = 0
    tp = 0.0
    for t in range(T):
        if cur != 0:
            ra, rb = rt[t, 0], rt[t, 1]
            d = cur * (va * ra - vb * rb)
            pnl[t] = d
            tp += d
            va *= 1.0 + ra
            vb *= 1.0 + rb
            st[4] += 1
        new = ex[t]
        if new != cur:
            if cur != 0:
                turn[t] += va + vb
                if exc[t] == 1:
                    st[1] += 1
                    st[2] += tp
                else:
                    st[3] += tp
            if new != 0:
                va, vb = wa_d[t], wb_d[t]
                turn[t] += va + vb
                st[0] += 1
                tp = 0.0
            cur = new
        elif cur != 0 and rebal:
            na, nb = wa_d[t], wb_d[t]
            turn[t] += abs(na - va) + abs(nb - vb)
            va, vb = na, nb
    return pnl, turn, st


# ----------------------------------------------------------------------------------------------------------
# One window
# ----------------------------------------------------------------------------------------------------------
def run_window(block):
    """block: (FORM + TRADE, N) simple returns of eligible assets. Returns {(selection, rule): result} and the
    ADF rows of the selected pairs."""
    rf, rt = block[:FORM], block[FORM:]
    n = block.shape[1]
    ia, ib = pair_index(n)
    lp = np.cumsum(np.log1p(block), axis=0)
    sels = {"dist": select_distance(rf, ia, ib)}
    beta, tau = eg_tstats(rf, ia, ib)
    order = np.argsort(np.where(beta > 0, tau, np.inf), kind="stable")
    sels["eg"] = order[beta[order] > 0][:NPAIRS]
    out, adf_rows = {}, []
    for sname, sel in sels.items():
        acc = {r: [np.zeros(TRADE), np.zeros(TRADE), np.zeros(NSTAT)] for r in RULES}
        ks, kv = [], []
        for p in sel:
            a, b = ia[p], ib[p]
            y, x = lp[:, a], lp[:, b]
            sp = {"S": spread_static(y, x), "R": spread_rolling(y, x, ROLL_W)}
            zk, ek, Ak, Bk, rr = spread_kalman(y, x)
            sp["K"] = (zk, ek, Ak, Bk)
            sp["C"] = spread_chan(y, x)
            sp["V"] = spread_reversal(y, x)
            r2 = np.ascontiguousarray(rt[:, [a, b]])
            for rule, (spn, oz, rebal, wait) in RULES.items():
                z, e, A, Bc = sp[spn]
                pnl, turn, st = simulate(r2, z[FORM:], Bc[FORM:], e[FORM:], y[FORM:], x[FORM:], A[FORM:], Bc[FORM:],
                                         oz, wait, rebal)
                acc[rule][0] += pnl
                acc[rule][1] += turn
                acc[rule][2] += st
            ks.append(rr)
            kv.append(ek[BURN:FORM])
        k = len(sel)
        for rule in RULES:
            out[(sname, rule)] = (acc[rule][0] / k, acc[rule][1] / k, acc[rule][2])
        if sname == "dist":
            # unit-root evidence in formation: static OLS residual (Engle-Granger, N = 2) vs Kalman innovations
            sel_a, sel_b = ia[sel], ib[sel]
            yy, xx = lp[:FORM, sel_a], lp[:FORM, sel_b]
            xc, yc = xx - xx.mean(0), yy - yy.mean(0)
            bb = (xc * yc).sum(0) / (xc * xc).sum(0)
            p_s = mackinnon(adf_tau(yc - bb * xc), 2)
            p_k = mackinnon(adf_tau_c(np.column_stack(kv)), 1)
            ra = np.array([r[0] for r in ks])
            rb = np.array([r[1] for r in ks])
            adf_rows = [(float((p_s < 0.05).mean()), float((p_k < 0.05).mean()), float(np.median(ra)),
                         float(np.median(rb)), float((ra > 0.1).mean()))]
    return out, adf_rows


def eligible(block, extra_ok=None):
    ok = ~np.isnan(block).any(axis=0)
    ok &= (block[:FORM] == 0).mean(axis=0) < MAX_ZERO_SHARE
    if extra_ok is not None:
        ok &= extra_ok
    return ok


def demean(b):
    mu = b.mean(axis=0)
    return b - mu + mu.mean()


def month_starts(index):
    m = index.to_period("M")
    first = np.r_[True, m[1:] != m[:-1]]
    s = np.nonzero(first)[0]
    return s[(s >= FORM) & (s + TRADE <= len(index))]


G = {}


def _init(universe):
    r = pairs_panel(universe)
    extra = None
    if universe == "industries":
        extra = pd.read_csv(DATA / "industries_nfirms_monthly.csv", index_col=0, parse_dates=True)[r.columns]
    G.update(R=r.values, dates=r.index, extra=extra, uid=UNIVERSE_ID[universe])


def _extra_ok(extra, dates, s):
    if extra is None:
        return None
    prev = (dates[s - FORM].to_period("M") - 1).to_timestamp()
    return extra.loc[prev].values >= 5


def work(k_s):
    k, s = k_s
    R, dates, extra, uid = G["R"], G["dates"], G["extra"], G["uid"]
    block = R[s - FORM:s + TRADE]
    ok = eligible(block, _extra_ok(extra, dates, s))
    b = np.ascontiguousarray(block[:, np.nonzero(ok)[0]])
    if b.shape[1] < 10:
        return k, b.shape[1], None, []
    res = {"real": run_window(b)}
    bd = demean(b)
    for rep in range(NREP):
        rng = np.random.default_rng([uid, k, rep, 1])          # project 11's N1 seeds
        res[f"N1_{rep}"] = run_window(np.ascontiguousarray(bd[rng.integers(0, len(b), len(b))]))
    return k, b.shape[1], res, None


def run(universe, procs=4):
    r = pairs_panel(universe)
    starts = month_starts(r.index)
    K = len(starts)
    keys = [(s, rule) for s in ("dist", "eg") for rule in RULES]
    pnl = {(src, *key): np.full((K, 1 if src == "real" else NREP, TRADE), np.nan) for src in ("real", "N1")
           for key in keys}
    turn = {kk: np.full_like(v, np.nan) for kk, v in pnl.items()}
    stats = {kk: np.full(v.shape[:2] + (NSTAT,), np.nan) for kk, v in pnl.items()}
    adf = []
    nassets = np.zeros(K, int)
    with Pool(procs, initializer=_init, initargs=(universe,)) as pool:
        for i, (k, na, res, _) in enumerate(pool.imap_unordered(work, list(enumerate(starts)), chunksize=2)):
            nassets[k] = na
            if res is None:
                continue
            for key, (out, arows) in res.items():
                src, rep = ("real", 0) if key == "real" else ("N1", int(key[3:]))
                for (sname, rule), (p_, t_, s_) in out.items():
                    pnl[(src, sname, rule)][k, rep] = p_
                    turn[(src, sname, rule)][k, rep] = t_
                    stats[(src, sname, rule)][k, rep] = s_
                for row in arows:
                    adf.append((k, src, rep) + row)
            if i % 100 == 0:
                print(universe, i, K, flush=True)
    arrays = {"starts": starts, "nassets": nassets, "dates": r.index.values.astype("datetime64[D]").astype("int64")}
    for kk in pnl:
        tag = "|".join(kk)
        arrays[f"pnl|{tag}"] = pnl[kk].astype(np.float32)
        arrays[f"turn|{tag}"] = turn[kk].astype(np.float32)
        arrays[f"stats|{tag}"] = stats[kk]
    np.savez_compressed(RES / f"pairs_{universe}.npz", **arrays)
    pd.DataFrame(adf, columns=["k", "source", "rep", "eg_sig_share", "kalman_innov_sig_share", "ra_median",
                               "rb_median", "share_ra_gt_0.1"]).sort_values(["k", "source", "rep"]).to_csv(
        RES / f"pairs_adf_{universe}.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for u in sys.argv[1:] or ["countries", "industries"]:
        run(u)
