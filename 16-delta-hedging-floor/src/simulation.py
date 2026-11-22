"""Experiment 2: what is the floor made of? A factorial of simulated worlds.

Each world is a long daily return series cut into overlapping one-month windows, exactly like the real
data, and hedged with the same engine and the same quote rules. Worlds:

  GBM            constant volatility, Gaussian returns              (the textbook world)
  iid-t5         constant volatility, Student-t(5) returns          (fat tails, no clustering)
  Merton         constant diffusion + compound-Poisson jumps         (jumps, no clustering)
  GARCH-N        GARCH(1,1) with Gaussian innovations, fit to SPX    (clustering, thin tails)
  GJR-t          GJR-GARCH(1,1) with Student-t innovations, fit to SPX (clustering + leverage + tails)

Quotes:
  model-optimal  the model's own conditional expectation of the next month's average variance (the best
                 forecast anyone could make inside that world)
  EWMA           the RiskMetrics forecast a real hedger uses (the rule in experiment 1)
  true integrated vol  (GARCH worlds only) the average variance the model actually delivered over the
                 option's life: an ablation that removes the forecast error but keeps the vol path

Writes results/sim_factorial.csv and results/derman_check.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

from data import H_GRID, RES, T_DAYS, spx
from frontier import boot_floor, floor_stats
from hedgesim import DT, derman_std_volpts, every_h_days, hedge_short_call, windows

N_DAYS = 200_000
STEP = 4  # start a new option every 4 days (50k options per world)
SEED = 1616


def ewma_from_returns(r: np.ndarray, lam: float = 0.94) -> np.ndarray:
    v = np.empty_like(r)
    v0 = r[:250].var()
    for i, x in enumerate(r):
        v0 = lam * v0 + (1 - lam) * x * x
        v[i] = v0
    return np.sqrt(252 * v)


def fit_spx():
    d = spx()
    r = 100 * d.r.dropna().to_numpy()
    g = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    gj = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    return g.params, gj.params, r


def sim_garch(rng, omega, alpha, gamma, beta, nu, n):
    """Simulate (GJR-)GARCH returns in decimal units; omega in (percent)^2. Returns r, sigma2_next."""
    if nu is None:
        z = rng.standard_normal(n)
    else:
        z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    s2 = np.empty(n + 1)
    pers = alpha + gamma / 2 + beta
    s2[0] = omega / (1 - pers)
    r = np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(s2[t]) * z[t]
        s2[t + 1] = omega + (alpha + gamma * (r[t] < 0)) * r[t] ** 2 + beta * s2[t]
    return r / 100.0, s2 / 1e4, pers, omega / (1 - pers) / 1e4


def integrated_vol(s2, T):
    """Ex-post true average variance of the next T days (s2[i] is the variance of the move i -> i+1).

    In a simulation this is not look-ahead in the sense of the real-data oracle: it is the variance the
    model actually delivered, not the squared returns the hedge is scored on.
    """
    c = np.concatenate([[0.0], np.cumsum(s2[:-1])])
    m = (c[T:] - c[:-T]) / T
    return np.sqrt(252 * np.concatenate([m, np.full(T, np.nan)]))


def garch_month_forecast(s2_next, pers, vbar, T):
    """Expected average daily variance over the next T days given sigma^2_{t+1}; annualised vol."""
    k = np.arange(T)
    w = (pers ** k).mean()
    avg = vbar + (s2_next - vbar) * w
    return np.sqrt(252 * avg)


def world(name, rng, gp, gjp, sig):
    n = N_DAYS
    if name == "GBM":
        r = sig * np.sqrt(DT) * rng.standard_normal(n)
        opt = np.full(n + 1, sig)
        true = opt
    elif name == "iid-t5":
        nu = 5
        r = sig * np.sqrt(DT) * rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
        opt = np.full(n + 1, sig)
        true = opt
    elif name == "Merton":
        lam, mj, sj = 4.0, -0.03, 0.04  # 4 jumps a year, mean -3%, sd 4% (in the range used in project 01)
        jump_var = lam * (mj ** 2 + sj ** 2)
        sd = np.sqrt(max(sig ** 2 - jump_var, 1e-6))
        nj = rng.poisson(lam * DT, n)
        J = mj * nj + sj * np.sqrt(nj) * rng.standard_normal(n)
        r = sd * np.sqrt(DT) * rng.standard_normal(n) + J
        r = r - r.mean()
        opt = np.full(n + 1, np.sqrt(sd ** 2 + jump_var))
        true = opt
    elif name == "GARCH-N":
        r, s2, pers, vbar = sim_garch(rng, gp["omega"], gp["alpha[1]"], 0.0, gp["beta[1]"], None, n)
        opt = garch_month_forecast(s2, pers, vbar, T_DAYS)
        true = integrated_vol(s2, T_DAYS)
    elif name == "GJR-t":
        r, s2, pers, vbar = sim_garch(rng, gjp["omega"], gjp["alpha[1]"], gjp["gamma[1]"], gjp["beta[1]"],
                                      gjp["nu"], n)
        opt = garch_month_forecast(s2, pers, vbar, T_DAYS)
        true = integrated_vol(s2, T_DAYS)
    else:
        raise ValueError(name)
    close = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    # opt[i] is the forecast made at the close of day i for days i+1..i+T (s2[i] is the variance of r[i],
    # i.e. of the move from close i to close i+1)
    ew = np.concatenate([[np.nan], ewma_from_returns(r)])
    return close, opt, ew, r, true


def main():
    rng = np.random.default_rng(SEED)
    gp, gjp, r_spx = fit_spx()
    sig = float(np.std(r_spx) / 100 * np.sqrt(252))
    pd.DataFrame({"GARCH-N": gp, "GJR-t": gjp}).to_csv(RES / "sim_garch_params.csv")
    hcols = [f"h{h}" for h in H_GRID]
    rows = []
    for name in ["GBM", "iid-t5", "Merton", "GARCH-N", "GJR-t"]:
        close, opt, ew, r, true = world(name, rng, gp, gjp, sig)
        P, idx = windows(close, T_DAYS, step=STEP)
        burn = idx >= 1000
        P, idx = P[burn], idx[burn]
        z = r[: (len(r) // T_DAYS) * T_DAYS].reshape(-1, T_DAYS)
        zz = z / np.sqrt((z ** 2).mean(1, keepdims=True))
        ek_std = float((zz ** 4).mean() / (zz ** 2).mean() ** 2 - 3)
        quotes = [("model-optimal", opt[idx]), ("EWMA", ew[idx])]
        if name in ("GARCH-N", "GJR-t"):
            quotes.append(("true integrated vol", true[idx]))
        for qname, q in quotes:
            M = np.column_stack([hedge_short_call(P, 100.0, q, schedule=every_h_days(T_DAYS, h)).pnl
                                 for h in H_GRID])
            s = floor_stats(M)
            bt = boot_floor(M, rng, B=300)
            lo, hi = np.percentile(bt[:, 0], [2.5, 97.5])
            glo, ghi = np.percentile(bt[:, 2], [2.5, 97.5])
            rows.append(dict(world=name, quote=qname, n=len(M), std1=s["std1"], std5=s["std5"],
                             share=s["share"], share_lo=lo, share_hi=hi, gain_5_to_1=s["gain_5_to_1"],
                             gain_lo=glo, gain_hi=ghi, ek_std=ek_std, pred_share_iid=ek_std / (2 + ek_std),
                             **{f"std_{c}": v for c, v in zip(hcols, M.std(0, ddof=1))}))
            print(rows[-1]["world"], qname, round(s["share"], 3), round(s["gain_5_to_1"], 3), round(s["std1"], 2))
    pd.DataFrame(rows).to_csv(RES / "sim_factorial.csv", index=False, float_format="%.4f")

    # ---------- the textbook law, checked: GBM, ATM, exact quote ----------
    rows = []
    for s_ in [0.1, 0.2, 0.4]:
        r = s_ * np.sqrt(DT) * rng.standard_normal((100_000, T_DAYS)) - 0.5 * s_ ** 2 * DT
        P = 100 * np.exp(np.concatenate([np.zeros((len(r), 1)), np.cumsum(r, 1)], 1))
        for h in H_GRID:
            N = int(np.ceil(T_DAYS / h))
            res = hedge_short_call(P, 100.0, s_, schedule=every_h_days(T_DAYS, h))
            rows.append(dict(sigma=s_, h=h, n_rebalances=N, sim_std=res.pnl.std(ddof=1),
                             sim_mean=res.pnl.mean(), derman_std=derman_std_volpts(s_, N)))
    pd.DataFrame(rows).to_csv(RES / "derman_check.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
