"""Part II engine: break-even (realised) Black-Scholes volatilities by strike.

For a path S_0..S_n with rebalancing every h days, a delta-hedged short Black-Scholes option sold at
volatility sigma earns, to second order (the gamma-theta identity),

    P&L(sigma) = 1/2 * sum_j  Gamma_j S_j^2 (sigma^2 dt - R_j^2),       R_j = S_{j+1}/S_j - 1,

and the drift of S drops out. Gamma is evaluated at sigma itself, so the break-even volatility is the
fixed point sigma* with P&L(sigma*) = 0: a gamma-weighted realised volatility (Bergomi 2016, ch. 1).

We price on a standardised strike grid. At each start date t0 the trailing 63-day volatility
sigma_ref is known; strikes are K = S0 exp(x sigma_ref sqrt(tau)) for x in X_GRID, and every option is
sold at sigma = m * sigma_ref. The output for each path, strike x and multiplier m is the hedged P&L
expressed in units of (inception vega * sigma_ref), so a pricing error of +1% of sigma_ref in the
quoted volatility shows up as about -0.01. Pooling paths and solving for the m that zeroes average P&L
gives the "break-even multiplier" m*(x): the smile history would have charged.

Rates are set to zero: the gamma-theta form is rate- and drift-free to second order, and for 1-month
options the financing term on the premium is below 0.5% of vega for any rate seen in 2007-2026.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"

ASSETS = ["SPY", "EEM", "HYG", "TLT", "GLD", "USO", "URA"]
X_GRID = np.round(np.arange(-2.5, 2.5001, 0.25), 2)
M_GRID = np.round(np.arange(0.20, 4.0001, 0.01), 2)
ANN = 252.0
START = "2007-01-01"


@dataclass(frozen=True)
class Spec:
    horizon: int = 21        # trading days to expiry
    step: int = 1            # rebalancing interval in trading days
    lookback: int = 63       # trailing window for sigma_ref
    simple: bool = True      # hedge P&L uses simple returns (True) or log returns (False)
    gamma_at_ref: bool = False  # ablation: evaluate Gamma at sigma_ref instead of at the quoted vol


def load_returns() -> pd.DataFrame:
    df = pd.read_csv(DATA / "etf_daily_log_returns.csv", parse_dates=["date"], index_col="date")
    return df[ASSETS]


def load_vix() -> pd.Series:
    v = pd.read_csv(DATA / "vix_daily.csv", parse_dates=["DATE"], index_col="DATE")["CLOSE"]
    v.index.name = "date"
    return v / 100.0


def build_paths(logret: pd.Series, spec: Spec, start: str = START):
    """Return (dates, sigma_ref, logS) with logS[:, j] = log S_{t0+j}/S_{t0}, j = 0..horizon."""
    r = logret.dropna()
    vals = r.values
    idx = r.index
    n = spec.horizon
    t0s = [i for i in range(spec.lookback - 1, len(vals) - n) if idx[i] >= pd.Timestamp(start)]
    t0s = np.array(t0s)
    sig = np.array([vals[i - spec.lookback + 1:i + 1].std(ddof=1) for i in t0s]) * np.sqrt(ANN)
    fwd = np.stack([vals[i + 1:i + n + 1] for i in t0s])  # (P, n) next-day log returns
    logS = np.concatenate([np.zeros((len(t0s), 1)), np.cumsum(fwd, axis=1)], axis=1)
    return idx[t0s], sig, logS


def _npdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)


def pnl_grid(sig_ref, logS, spec: Spec, x_grid=X_GRID, m_grid=M_GRID, dtype=np.float64):
    """Hedged P&L of a short option, normalised by inception vega * sigma_ref.

    Returns array (P, len(x_grid), len(m_grid)).
    """
    n, h = spec.horizon, spec.step
    P = len(sig_ref)
    reb = np.arange(0, n, h)                       # rebalancing indices
    tau = (n - reb) / ANN                          # time to expiry at each rebalance, (J,)
    dt = h / ANN
    lS = logS[:, reb]                              # (P, J)
    lS_next = logS[:, np.minimum(reb + h, n)]
    R = np.expm1(lS_next - lS) if spec.simple else (lS_next - lS)
    R2 = R * R                                     # (P, J)
    T = n / ANN
    lK = x_grid[None, :] * sig_ref[:, None] * np.sqrt(T)   # log strike relative to S0, (P, X)
    S = np.exp(lS)                                 # (P, J)
    out = np.empty((P, len(x_grid), len(m_grid)), dtype=dtype)
    logm = (lS[:, None, :] - lK[:, :, None])       # ln(S_j / K), (P, X, J)
    for a, m in enumerate(m_grid):
        sq = m * sig_ref                           # quoted vol (P,)
        sg = sig_ref if spec.gamma_at_ref else sq  # vol used inside Gamma
        v = sg[:, None, None] * np.sqrt(tau)[None, None, :]
        d1 = logm / v + 0.5 * v
        cash_gamma = S[:, None, :] * _npdf(d1) / v                  # Gamma S^2, (P, X, J)
        pnl = 0.5 * np.sum(cash_gamma * (sq[:, None, None] ** 2 * dt - R2[:, None, :]), axis=2)
        v0 = sq * np.sqrt(T)
        d10 = -lK / v0[:, None] + 0.5 * v0[:, None]
        vega0 = _npdf(d10) * np.sqrt(T)                              # S0 = 1
        out[:, :, a] = pnl / (vega0 * sig_ref[:, None])
    return out


def root_on_grid(f, m_grid=M_GRID):
    """First upward zero crossing of the seller's P&L f along the last axis (f rises with the quoted
    vol), linearly interpolated. Clipped to the grid ends when there is no crossing."""
    s = np.sign(f)
    cross = (s[..., :-1] < 0) & (s[..., 1:] >= 0)
    has = cross.any(axis=-1)
    k = np.argmax(cross, axis=-1)
    f0 = np.take_along_axis(f, k[..., None], -1)[..., 0]
    f1 = np.take_along_axis(f, (k + 1)[..., None], -1)[..., 0]
    m0, m1 = m_grid[k], m_grid[k + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        root = m0 + (m1 - m0) * (-f0) / (f1 - f0)
    root = np.where(has, root, np.nan)
    root = np.where(~has & (f[..., 0] >= 0), m_grid[0], root)   # seller already breaks even at min m
    root = np.where(~has & (f[..., -1] < 0), m_grid[-1], root)  # seller loses even at max m
    return root


def interp_pnl(pnl, m_query):
    """P&L at an arbitrary multiplier per path (and per strike if m_query is (P, X))."""
    m_query = np.asarray(m_query, float)
    if m_query.ndim == 1:
        m_query = np.repeat(m_query[:, None], pnl.shape[1], axis=1)
    mq = np.clip(m_query, M_GRID[0], M_GRID[-1])
    pos = (mq - M_GRID[0]) / (M_GRID[1] - M_GRID[0])
    k = np.clip(np.floor(pos).astype(int), 0, len(M_GRID) - 2)
    w = pos - k
    a = np.take_along_axis(pnl, k[..., None], -1)[..., 0]
    b = np.take_along_axis(pnl, (k + 1)[..., None], -1)[..., 0]
    return a * (1 - w) + b * w


def block_bootstrap_counts(P, B, block, rng):
    """Moving-block bootstrap: (B, P) matrix of how often each path index is drawn."""
    nb = int(np.ceil(P / block))
    C = np.zeros((B, P), dtype=np.float32)
    starts = rng.integers(0, P - block + 1, size=(B, nb))
    for b in range(B):
        idx = (starts[b][:, None] + np.arange(block)[None, :]).ravel()[:P]
        np.add.at(C[b], idx, 1.0)
    return C


def compute_asset(asset, spec=Spec(), start=START, logret=None):
    lr = load_returns() if logret is None else logret
    dates, sig, logS = build_paths(lr[asset], spec, start)
    pnl = pnl_grid(sig, logS, spec)
    return dates, sig, logS, pnl
