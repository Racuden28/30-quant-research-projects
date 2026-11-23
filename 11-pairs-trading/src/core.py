"""
Core engine for project 11: pair selection (distance and Engle-Granger), the trading simulator, and the
placebo markets. Everything here works on one window of returns (formation + trading) at a time, so the
same code path runs on real and on placebo data.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.stats import norm

FORM, TRADE = 252, 126          # formation and trading lengths, trading days
NPAIRS = 20                     # pairs traded per portfolio
OPEN_Z = 2.0                    # open when |spread| > 2 formation standard deviations
MAX_ZERO_SHARE = 0.05           # eligibility: fewer than 5 % zero-return days in formation


# --------------------------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------------------------
def pair_index(n: int) -> tuple[np.ndarray, np.ndarray]:
    ia, ib = np.triu_indices(n, k=1)
    return ia.astype(np.int64), ib.astype(np.int64)


def distance_stats(rf: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SSD of normalised total-return indices and the formation std of their spread, for every pair.

    rf: (FORM, N) simple returns of the formation period. The index starts at 1 *before* day 0, so the
    first observation is 1 + r_0, as in GGR's normalisation at the start of formation."""
    p = np.cumprod(1.0 + rf, axis=0)
    s = p[:, ia] - p[:, ib]
    return (s * s).sum(axis=0), s.std(axis=0, ddof=1)


def eg_stats(rf: np.ndarray, ia: np.ndarray, ib: np.ndarray):
    """Engle-Granger on log prices for every pair (A = ia regressed on B = ib, with a constant).

    Returns alpha, beta, residual std, and the ADF t-statistic with one lagged difference and no constant
    on the residual. Fully vectorised over pairs."""
    lp = np.cumsum(np.log1p(rf), axis=0)
    y, x = lp[:, ia], lp[:, ib]
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    beta = (xc * yc).sum(0) / (xc * xc).sum(0)
    alpha = ym - beta * xm
    e = y - alpha - beta * x
    de = np.diff(e, axis=0)
    yv, x1, x2 = de[1:], e[1:-1], de[:-1]          # d e_t = rho e_{t-1} + g d e_{t-1} + u
    s11, s22, s12 = (x1 * x1).sum(0), (x2 * x2).sum(0), (x1 * x2).sum(0)
    s1y, s2y = (x1 * yv).sum(0), (x2 * yv).sum(0)
    det = s11 * s22 - s12 * s12
    rho = (s22 * s1y - s12 * s2y) / det
    g = (s11 * s2y - s12 * s1y) / det
    u = yv - rho * x1 - g * x2
    n = yv.shape[0]
    sig2 = (u * u).sum(0) / (n - 2)
    se = np.sqrt(sig2 * s22 / det)
    return alpha, beta, e.std(0, ddof=1), rho / se


# MacKinnon (1994) approximate p-value for the Engle-Granger tau with N = 2 variables and a constant, i.e.
# statsmodels.tsa.adfvalues.mackinnonp(t, "c", N=2), vectorised. The coefficients are read from statsmodels;
# test_core.py checks equality on a grid.
from statsmodels.tsa import adfvalues as _adf  # noqa: E402

_TAU_MAX, _TAU_MIN, _TAU_STAR = _adf.tau_max_c[1], _adf.tau_min_c[1], _adf.tau_star_c[1]
_SMALL, _LARGE = np.asarray(_adf.tau_c_smallp[1]), np.asarray(_adf.tau_c_largep[1])


def mackinnon_p(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    small = np.polyval(_SMALL[::-1], t)
    large = np.polyval(_LARGE[::-1], t)
    p = np.where(t <= _TAU_STAR, norm.cdf(small), norm.cdf(large))
    p = np.where(t > _TAU_MAX, 1.0, p)
    return np.where(t < _TAU_MIN, 0.0, p)


# --------------------------------------------------------------------------------------------------------
# Trading simulator
# --------------------------------------------------------------------------------------------------------
@njit(cache=True)
def simulate(rt, ia, ib, wa, wb, z, wait):
    """Trade every pair in (ia, ib) over the trading period.

    rt: (T, N) simple returns of the trading period; z: (T, P) signal (spread in formation-sd units) known at
    the close of each day; wa, wb: dollar size of leg A and leg B when a position opens; wait: days between the
    decision and the trade. A long-spread position (+1) is long A and short B.

    Returns
      pnl (T, P): daily P&L per $1 of committed capital, gross of costs,
      turn (T, P): notional traded that day (sum over both legs), to charge costs later,
      stats (P, 6): trades opened, trades converged, sum of converged trade returns, sum of non-converged trade
                    returns, days with an open position, trades closed by crossing but after a sign flip.
    """
    T, P = z.shape
    pnl = np.zeros((T, P))
    turn = np.zeros((T, P))
    stats = np.zeros((P, 6))
    for k in range(P):
        a, b = ia[k], ib[k]
        # 1) desired position after each close (decision path)
        want = np.zeros(T, np.int64)
        conv_flag = np.zeros(T, np.int64)   # 1 if the decision to close at t came from a crossing
        pos = 0
        for t in range(T):
            zt = z[t, k]
            if pos == 0:
                if zt > OPEN_Z:
                    pos = -1
                elif zt < -OPEN_Z:
                    pos = 1
            elif pos == 1 and zt >= 0.0:
                pos = 0
                conv_flag[t] = 1
            elif pos == -1 and zt <= 0.0:
                pos = 0
                conv_flag[t] = 1
            want[t] = pos
        # 2) executed position: the decision of day t is traded at the close of day t + wait;
        #    everything is closed at the last close of the trading period
        ex = np.zeros(T, np.int64)
        exconv = np.zeros(T, np.int64)
        for t in range(T):
            if t - wait >= 0:
                ex[t] = want[t - wait]
                exconv[t] = conv_flag[t - wait]
        ex[T - 1] = 0
        # 3) P&L with buy-and-hold legs inside each trade
        va = 0.0
        vb = 0.0
        cur = 0
        trade_pnl = 0.0
        for t in range(T):
            if cur != 0:
                ra, rb = rt[t, a], rt[t, b]
                d = cur * (va * ra - vb * rb)
                pnl[t, k] = d
                trade_pnl += d
                va *= 1.0 + ra
                vb *= 1.0 + rb
                stats[k, 4] += 1
            new = ex[t]
            if new != cur:
                if cur != 0:                       # close the open trade at today's close
                    turn[t, k] += va + vb
                    if exconv[t] == 1:
                        stats[k, 1] += 1
                        stats[k, 2] += trade_pnl
                    else:
                        stats[k, 3] += trade_pnl
                if new != 0:                       # open a new trade at today's close
                    va, vb = wa[k], wb[k]
                    turn[t, k] += va + vb
                    stats[k, 0] += 1
                    trade_pnl = 0.0
                cur = new
    return pnl, turn, stats


VR_H = (1, 2, 5, 10, 21, 42, 63)


def vr_moments(rt: np.ndarray, a, b) -> np.ndarray:
    """Mean squared h-day relative log return of each pair in the trading period, averaged over pairs.

    VR(h) = m_h / (h m_1) is a variance ratio of the pair spread. It is not demeaned, so a persistent drift
    between the legs (momentum) pushes it above 1 and fast mean reversion pushes it below 1. Its finite-sample
    bias is the same on real and placebo data, so only the comparison with the placebo is read."""
    lr = np.log1p(rt)
    c = np.vstack([np.zeros((1, len(a))), np.cumsum(lr[:, a] - lr[:, b], axis=0)])
    return np.array([((c[h:] - c[:-h]) ** 2).mean() for h in VR_H])


def distance_signal(rt: np.ndarray, ia, ib, sd: np.ndarray) -> np.ndarray:
    p = np.cumprod(1.0 + rt, axis=0)
    return (p[:, ia] - p[:, ib]) / sd


def distance_signal_cont(rf: np.ndarray, rt: np.ndarray, ia, ib, sd: np.ndarray) -> np.ndarray:
    """Ablation: no renormalisation at the start of trading; the indices keep running from formation day 0."""
    p = np.cumprod(1.0 + np.vstack([rf, rt]), axis=0)[len(rf):]
    return (p[:, ia] - p[:, ib]) / sd


def eg_signal(rf: np.ndarray, rt: np.ndarray, ia, ib, alpha, beta, sd) -> np.ndarray:
    lp0 = np.log1p(rf).sum(0)                       # log price at the end of formation
    lp = lp0 + np.cumsum(np.log1p(rt), axis=0)
    e = lp[:, ia] - alpha - beta * lp[:, ib]
    return e / sd


# --------------------------------------------------------------------------------------------------------
# Placebo markets
# --------------------------------------------------------------------------------------------------------
def iid_days(L: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, L, L)


def stationary_days(L: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap of row indices (wrapping at the end of the window)."""
    idx = np.empty(L, np.int64)
    idx[0] = rng.integers(0, L)
    new = rng.random(L) < 1.0 / mean_block
    starts = rng.integers(0, L, L)
    for t in range(1, L):
        idx[t] = starts[t] if new[t] else (idx[t - 1] + 1) % L
    return idx


# --------------------------------------------------------------------------------------------------------
# One window
# --------------------------------------------------------------------------------------------------------
def eligible(block: np.ndarray, extra_ok: np.ndarray | None = None) -> np.ndarray:
    ok = ~np.isnan(block).any(axis=0)
    ok &= (block[:FORM] == 0).mean(axis=0) < MAX_ZERO_SHARE
    if extra_ok is not None:
        ok &= extra_ok
    return ok


def run_window(block: np.ndarray, configs: list[dict]) -> dict:
    """Select and trade on one (FORM + TRADE, N) block of *eligible* assets.

    configs: list of dicts with keys name, method ('dist'|'coint'), ranks ('top'|'mid'|'all'), wait.
    Returns per config: pnl (TRADE,), turn (TRADE,), stats (6,) summed over selected pairs, npairs."""
    rf, rt = block[:FORM], block[FORM:]
    n = block.shape[1]
    ia, ib = pair_index(n)
    out = {}
    cache = {}
    for cfg in configs:
        m = "dist" if cfg["method"] == "distc" else cfg["method"]
        if m not in cache:
            if m == "dist":
                ssd, sd = distance_stats(rf, ia, ib)
                order = np.argsort(ssd, kind="stable")
                cache[m] = (order, sd)
            else:
                alpha, beta, sd, tstat = eg_stats(rf, ia, ib)
                # a long/short pair needs a positive hedge ratio; negative-beta pairs are never ranked
                order = np.argsort(np.where(beta > 0, tstat, np.inf), kind="stable")
                order = order[beta[order] > 0]
                cache[m] = (order, alpha, beta, sd, tstat)
        c = cache[m]
        order = c[0]
        if cfg["ranks"] == "top":
            sel = order[:NPAIRS]
        elif cfg["ranks"] == "mid":
            sel = order[100:100 + NPAIRS]
        else:
            sel = order
        if len(sel) == 0:
            continue
        a, b = ia[sel], ib[sel]
        if m == "dist":
            z = distance_signal(rt, a, b, c[1][sel]) if cfg["method"] == "dist" else \
                distance_signal_cont(rf, rt, a, b, c[1][sel])
            wa = np.ones(len(sel))
            wb = np.ones(len(sel))
        else:
            z = eg_signal(rf, rt, a, b, c[1][sel], c[2][sel], c[3][sel])
            bt = np.abs(c[2][sel])
            wa, wb = 2.0 / (1.0 + bt), 2.0 * bt / (1.0 + bt)
        pnl, turn, st = simulate(rt, a, b, wa, wb, np.ascontiguousarray(z), int(cfg["wait"]))
        out[cfg["name"]] = {"pnl": pnl.mean(1), "turn": turn.mean(1), "stats": st.sum(0), "npairs": len(sel),
                            "vr": vr_moments(rt, a, b)}
    return out


def coint_screen(block2: np.ndarray, alpha_level: float = 0.05) -> dict:
    """Formation-year Engle-Granger p-values and their persistence into the following 252 days.

    block2: (2 * FORM, N). Returns the share of pairs significant in year 1, in year 2, the conditional shares,
    and the number of Benjamini-Hochberg discoveries at q = 0.05 in year 1."""
    n = block2.shape[1]
    ia, ib = pair_index(n)
    p1 = mackinnon_p(eg_stats(block2[:FORM], ia, ib)[3])
    p2 = mackinnon_p(eg_stats(block2[FORM:], ia, ib)[3])
    s1, s2 = p1 < alpha_level, p2 < alpha_level
    m = len(p1)
    ps = np.sort(p1)
    below = ps <= alpha_level * np.arange(1, m + 1) / m
    bh = int(np.nonzero(below)[0].max() + 1) if below.any() else 0
    return {"npairs": m, "sig1": s1.mean(), "sig2": s2.mean(),
            "sig2_given_sig1": s2[s1].mean() if s1.any() else np.nan,
            "sig2_given_not1": s2[~s1].mean() if (~s1).any() else np.nan,
            "n_sig1": int(s1.sum()), "bh_discoveries": bh}
