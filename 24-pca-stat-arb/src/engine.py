"""Walk-forward PCA residual statistical arbitrage (Avellaneda & Lee, 2010) with a placebo null.

Everything that decides a position on day t uses returns up to and including day t only.
A weight vector fixed at the close of day t earns the return of day t+1 (delay 0, the
original paper's convention) or of day t+2 (delay 1, a one-day wait; see project #11).

Pipeline per day t
------------------
1. Universe (first day of each month): assets with a complete return history over the last
   W days, ranked by median dollar volume over the last 30 days, top N kept.
2. Eigenportfolios (every `pca_every` days): correlation matrix of the W-day standardised
   returns of the active assets, top-k eigenvectors v_j, weights Q_j = v_j / sigma.
   k is fixed or chosen as the number of eigenvalues above the Marchenko-Pastur edge.
3. Residuals (daily): over the last L days, regress each asset's return on the k factor
   returns F = R Q' with an intercept; X = cumulative residual.
4. OU fit (daily): AR(1) X_{n+1} = a + b X_n + z. kappa = -ln b per observation,
   m = a/(1-b), sigma_eq = sd(z)/sqrt(1-b^2), s = (X_L - (m - mean m)) / sigma_eq.
   The "fast mean reversion" filter keeps assets with 1/kappa < 30 observations.
5. Bands (Avellaneda-Lee): open short s > 1.25, open long s < -1.25, close short s < 0.75,
   close long s > -0.5. Each open position is $1/n_open of residual: +1 asset, -beta.Q in
   the eigenportfolios, so the book is factor-neutral by construction.

Signals run in parallel on the same residuals:
  "ou"     the paper's s-score with the kappa filter
  "ou_nf"  the s-score without the filter
  "res5"   z-score of the last 5 days of residual return (no OU fit at all)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

SIGNALS = ("ou", "ou_nf", "res5")
IC_H = (1, 2, 3, 5, 10, 20)


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


@dataclass
class Config:
    W: int = 252                 # correlation window (observations)
    L: int = 60                  # residual / OU window (observations)
    k: object = "mp"             # int, "mp" (Marchenko-Pastur), or 0 (equal-weight market, beta=1)
    k_max: int = 15
    N: int | None = None         # max assets by liquidity (None = all eligible)
    min_assets: int = 30
    pca_every: int = 5
    s_open: float = 1.25
    s_close_long: float = -0.5
    s_close_short: float = 0.75
    max_tau: float = 30.0        # filter: mean-reversion time 1/kappa < max_tau observations
    signals: tuple = SIGNALS
    placebo_seed: int | None = None
    placebo_block: str = "Q"     # blocks for the placebo market (calendar quarter)
    placebo_method: str = "signflip"
    start: str | None = None     # first signal date (None = as soon as W days exist)


# --------------------------------------------------------------------------------------
# placebo market
# --------------------------------------------------------------------------------------
def placebo(ret: pd.DataFrame, liq: pd.DataFrame, seed: int, block: str = "Q", method: str = "signflip"):
    """Placebo markets.

    method="signflip" (used for every placebo result in the README): every day's whole
    cross-section of returns is multiplied by an independent random sign. The covariance of
    every day (r r' is unchanged), the volatility path, volatility clustering, listing and
    delisting dates and liquidity are all exactly as in the real market; any linear serial
    dependence, and with it every way of predicting the sign of tomorrow's residual, is
    destroyed. Drift and skewness are lost (a market-neutral residual book does not use them).

    Two earlier versions are kept only to document why they were rejected (README, section 7):

    method="bootstrap": each day of a calendar block is an IID
    draw, with replacement, of one of the block's real days. A day drawn twice makes a residual
    spike recur, a manufactured momentum: S&P 500 placebo Sharpe -3.2 (results/placebo_versions.csv).

    method="permute" (the first version): a permutation of
    the block's days. It preserves each block's total return. Real residuals
    mean-revert, so real quarterly totals are small, and in the permuted series later days
    offset earlier ones: the real reversal leaks back, spread over the quarter. S&P 500
    placebo Sharpe +0.8 (results/placebo_versions.csv; unit test in tests/test_engine.py).

    An asset that is not present on every day of a block (listing or delisting block) keeps
    its own days in that block, so no NaN is moved into the middle of a history. Liquidity
    moves with its row.
    """
    rng = np.random.default_rng(seed)
    if method == "signflip":
        flip = rng.choice([-1.0, 1.0], size=len(ret))
        return ret.mul(flip, axis=0), liq.reindex_like(ret)
    R = ret.to_numpy().copy(); Q = liq.reindex_like(ret).to_numpy().copy()
    R0 = ret.to_numpy(); Q0 = liq.reindex_like(ret).to_numpy()
    keys = ret.index.to_period(block)
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        if len(idx) < 2:
            continue
        if method == "permute":
            draw = idx[rng.permutation(len(idx))]
        else:
            draw = idx[rng.integers(0, len(idx), len(idx))]
        full = np.isfinite(R0[idx]).all(axis=0)
        R[np.ix_(idx, full)] = R0[np.ix_(draw, full)]
        Q[np.ix_(idx, full)] = Q0[np.ix_(draw, full)]
    return (pd.DataFrame(R, index=ret.index, columns=ret.columns),
            pd.DataFrame(Q, index=ret.index, columns=ret.columns))


# --------------------------------------------------------------------------------------
# building blocks (pure functions, unit-tested)
# --------------------------------------------------------------------------------------
def mp_edge(n_assets: int, n_obs: int) -> float:
    q = n_assets / n_obs
    return (1 + np.sqrt(q)) ** 2


def eigenportfolios(Rw: np.ndarray, k, k_max: int = 15):
    """Rw: W x n returns. Returns Q (k x n), eigenvalues (all, descending), k used."""
    mu = Rw.mean(0); sd = Rw.std(0, ddof=1)
    Y = (Rw - mu) / sd
    C = (Y.T @ Y) / (len(Y) - 1)
    lam, V = np.linalg.eigh(C)
    lam = lam[::-1]; V = V[:, ::-1]
    if k == "mp":
        kk = int(np.clip((lam > mp_edge(Rw.shape[1], Rw.shape[0])).sum(), 1, k_max))
    else:
        kk = int(k)
    if kk == 0:
        Q = np.full((1, Rw.shape[1]), 1.0 / Rw.shape[1])
    else:
        Q = (V[:, :kk] / sd[:, None]).T
    return Q, lam, kk


def residuals(RL: np.ndarray, Q: np.ndarray, equal_weight: bool = False):
    """Regress L x n returns on the factor returns (with intercept). Returns beta (k x n),
    residuals (L x n). equal_weight=True: residual = r - EW market (beta fixed at 1)."""
    F = RL @ Q.T                                   # L x k
    if equal_weight:
        E = RL - F                                 # Q is the equal-weight row
        return np.ones((1, RL.shape[1])), E - E.mean(0)
    Xd = np.column_stack([np.ones(len(F)), F])
    coef = np.linalg.lstsq(Xd, RL, rcond=None)[0]  # (k+1) x n
    E = RL - Xd @ coef
    return coef[1:], E


def ou_fit(E: np.ndarray):
    """AR(1) fit of the cumulative residual X = cumsum(E), column by column.
    Returns b, kappa (per observation), m, sigma_eq, X_L."""
    X = np.cumsum(E, axis=0)
    x0, x1 = X[:-1], X[1:]
    m0, m1 = x0.mean(0), x1.mean(0)
    vx = ((x0 - m0) ** 2).sum(0)
    b = ((x0 - m0) * (x1 - m1)).sum(0) / np.where(vx > 0, vx, np.nan)
    a = m1 - b * m0
    z = x1 - a - b * x0
    vz = z.var(0, ddof=0)
    ok = (b > 0) & (b < 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = np.where(ok, -np.log(np.where(ok, b, 0.5)), np.nan)
        m = np.where(ok, a / (1 - b), np.nan)
        seq = np.where(ok, np.sqrt(vz / (1 - b ** 2)), np.nan)
    return b, kappa, m, seq, X[-1]


def s_scores(E: np.ndarray, max_tau: float):
    b, kappa, m, seq, XL = ou_fit(E)
    mbar = m - np.nanmean(m)
    s = (XL - mbar) / seq
    passed = np.isfinite(kappa) & (kappa > 1.0 / max_tau)
    return s, passed, kappa, b


def res_z(E: np.ndarray, h: int = 5):
    sd = E.std(0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return E[-h:].sum(0) / (sd * np.sqrt(h))


def step_positions(pos: np.ndarray, s: np.ndarray, allowed: np.ndarray, cfg: Config):
    """Avellaneda-Lee band rules. pos in {-1,0,1}; returns new pos (same assets)."""
    new = pos.copy()
    fin = np.isfinite(s)
    new[(pos == 1) & (~fin | (s > cfg.s_close_long))] = 0
    new[(pos == -1) & (~fin | (s < cfg.s_close_short))] = 0
    flat = (new == 0) & fin & allowed
    new[flat & (s < -cfg.s_open)] = 1
    new[flat & (s > cfg.s_open)] = -1
    return new


# --------------------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------------------
@dataclass
class Result:
    weights: dict = field(default_factory=dict)    # signal -> DataFrame (dates x assets), set at close t
    diag: pd.DataFrame | None = None
    ic: pd.DataFrame | None = None
    ou_cross: list = field(default_factory=list)   # sampled (kappa, b) for the filter study


def run(ret: pd.DataFrame, liq: pd.DataFrame, cfg: Config, keep_weights: bool = True) -> Result:
    if cfg.placebo_seed is not None:
        ret, liq = placebo(ret, liq, cfg.placebo_seed, cfg.placebo_block, cfg.placebo_method)
    R = ret.to_numpy(); LQ = liq.reindex_like(ret).to_numpy()
    T, N = R.shape
    dates = ret.index
    fin = np.isfinite(R)
    # trailing count of finite returns, for "complete history over W days"
    cs = np.vstack([np.zeros((1, N)), np.cumsum(fin, 0)])
    sigs = cfg.signals
    W = {s: np.zeros((T, N), dtype=np.float32) for s in sigs}
    pos = {s: np.zeros(N, dtype=np.int8) for s in sigs}
    diag = []; ic_rows = []
    rng = np.random.default_rng(12345)
    res = Result()
    t0 = cfg.W - 1 if cfg.start is None else max(cfg.W - 1, int(dates.searchsorted(pd.Timestamp(cfg.start))))
    month = None; U = np.array([], dtype=int); S = None; Q = None; kk = 0; last_pca = -10 ** 9
    for t in range(t0, T - 1):
        if dates[t].month != month:
            month = dates[t].month
            full = (cs[t + 1] - cs[t + 1 - cfg.W]) == cfg.W
            cand = np.where(full)[0]
            if len(cand):
                med = np.nanmedian(LQ[max(0, t - 29): t + 1, cand], axis=0)
                med = np.where(np.isfinite(med), med, -np.inf)
                order = cand[np.argsort(-med)]
                U = order[: cfg.N] if cfg.N else order
            else:
                U = cand
            last_pca = -10 ** 9
        # active = universe members still trading with a complete W window
        act = U[((cs[t + 1, U] - cs[t + 1 - cfg.W, U]) == cfg.W)] if len(U) else U
        if len(act) < cfg.min_assets:
            for s in sigs:
                pos[s][:] = 0
            diag.append((dates[t], len(act), 0, 0, np.nan, np.nan, np.nan) + tuple(0 for _ in sigs))
            continue
        if (S is None) or (t - last_pca >= cfg.pca_every) or (not np.array_equal(act, S)):
            S = act
            Q, lam, kk = eigenportfolios(R[t + 1 - cfg.W: t + 1, S], cfg.k, cfg.k_max)
            last_pca = t
            top_share = lam[: max(kk, 1)].sum() / lam.sum()
        RL = R[t + 1 - cfg.L: t + 1, S]
        beta, E = residuals(RL, Q, equal_weight=(kk == 0))
        s_ou, passed, kappa, b = s_scores(E, cfg.max_tau)
        scores = {"ou": s_ou, "ou_nf": s_ou, "res5": res_z(E, 5)}
        allowed = {"ou": passed, "ou_nf": np.ones(len(S), bool), "res5": np.ones(len(S), bool)}
        if t % 21 == 0:
            pick = rng.choice(len(S), size=min(20, len(S)), replace=False)
            res.ou_cross.extend((dates[t], kk, kappa[i], b[i]) for i in pick)
        # positions
        n_open = []
        H = None
        for s in sigs:
            p_old = pos[s][S]
            p_new = step_positions(p_old, scores[s], allowed[s], cfg)
            pos[s][:] = 0; pos[s][S] = p_new
            no = int((p_new != 0).sum()); n_open.append(no)
            if no:
                if H is None:
                    H = np.eye(len(S)) - (beta.T @ Q)       # row i: residual portfolio of asset i
                w = (p_new.astype(float) @ H) / no
                W[s][t, S] = w
        # information coefficient at horizons 1..H (diagnostic, no trading): rank correlation
        # between -score at t and the residual return on day t+j, with beta and Q frozen at t
        row = [dates[t]]
        for j in IC_H:
            if t + j >= T:
                row += [np.nan] * 3; continue
            rj = R[t + j, S]
            okn = np.isfinite(rj)
            if okn.sum() <= 10:
                row += [np.nan] * 3; continue
            Fj = np.where(okn, rj, 0.0) @ Q.T
            ej = rj - (Fj @ beta if kk else Fj)
            with np.errstate(invalid="ignore", divide="ignore"):
                r1z = E[-1] / E.std(0, ddof=1)
            for sc in (s_ou, scores["res5"], r1z):
                g = okn & np.isfinite(sc)
                row.append(_rank_corr(-sc[g], ej[g]) if g.sum() > 10 else np.nan)
        ic_rows.append(row)
        diag.append((dates[t], len(S), kk, top_share, float(np.mean(passed)),
                     float(np.nanmedian(kappa)), float(np.nanmedian(b))) + tuple(n_open))
    cols = ["date", "n_assets", "k", "top_share", "pass_filter", "kappa_med", "b_med"] + [f"open_{s}" for s in sigs]
    res.diag = pd.DataFrame(diag, columns=cols).set_index("date")
    res.ic = pd.DataFrame(ic_rows, columns=["date"] + [f"ic_{n}_h{j}" for j in IC_H for n in ("ou", "res5", "res1")]).set_index("date")
    if keep_weights:
        res.weights = {s: pd.DataFrame(W[s], index=dates, columns=ret.columns) for s in sigs}
    res.ret = ret
    return res


# --------------------------------------------------------------------------------------
# P&L
# --------------------------------------------------------------------------------------
def pnl(weights: pd.DataFrame, ret: pd.DataFrame, delay: int = 0, cost_bps: float = 0.0,
        borrow_apr: float = 0.0, periods: int = 252) -> pd.DataFrame:
    """Daily P&L of a weight path. weights row t is set at the close of t.
    delay=0 earns r_{t+1}; delay=1 earns r_{t+2}. Costs: cost_bps per unit of one-way
    turnover of the net asset weights; borrow: APR on the short side."""
    w = weights.shift(1 + delay).fillna(0.0)
    r = ret.fillna(0.0)
    gross = (w * r).sum(axis=1)
    turn = weights.diff().abs().sum(axis=1).shift(delay).fillna(0.0)   # traded when the weights are set
    short = (-w.clip(upper=0)).sum(axis=1)
    cost = turn.shift(1).fillna(0.0) * cost_bps * 1e-4
    borrow = short * borrow_apr / periods
    out = pd.DataFrame({"gross": gross, "turnover": turn.shift(1).fillna(0.0), "short": short,
                        "gross_exposure": w.abs().sum(axis=1)})
    out["net"] = gross - cost - borrow
    return out
