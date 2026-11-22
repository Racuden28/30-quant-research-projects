"""Experiment 3: which volatility belongs inside the delta?

S&P 500, one-month ATM calls sold at the VIX close, hedged daily, 1992-2026 (the first two years of VIX
history are kept as the estimation window of the leverage slope). Delta rules:

  fixed     Black-Scholes delta at the volatility the option was sold at (VIX on day 0)
  floating  Black-Scholes delta at today's VIX (what most desks do)
  ewma      Black-Scholes delta at today's EWMA realised volatility
  mv        minimum-variance delta (Hull & White 2017 in spirit): floating + vega * E[d sigma | dS] / dS,
            where E[d sigma | dS] = beta_t * dS / S and beta_t is the OLS slope of daily VIX changes on
            S&P log returns, estimated on all data before the option's start date (expanding window)

Two scores:
  terminal  variance across options of the P&L at expiry (model-free: premium, payoff and hedge gains)
  daily     sum over days of squared mark-to-market hedge errors, with the option marked at Black-Scholes
            and today's VIX. This mark is a proxy (VIX is a 30-day variance-swap rate, not the vol of this
            option at this strike and maturity), so this score is secondary.

Writes results/which_delta.csv, results/which_delta_tests.csv, results/which_delta_paths.csv.gz,
results/leverage_slope.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import RES, T_DAYS, spx
from frontier import block_indices
from hedgesim import DT, call_delta, call_price, call_vega, hedge_short_call, windows

SEED = 1603
B = 2000
MIN_EST = 504  # two years of daily data before the first slope is used


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def leverage_slopes(d: pd.DataFrame) -> pd.Series:
    """Expanding OLS slope of dVIX (in vol units) on the log return, using data strictly before each day."""
    x = d.r.to_numpy()
    y = d.vix.diff().to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x0, y0 = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    n = np.cumsum(ok)
    sx, sy, sxx, sxy = np.cumsum(x0), np.cumsum(y0), np.cumsum(x0 * x0), np.cumsum(x0 * y0)
    beta = (sxy - sx * sy / np.maximum(n, 1)) / (sxx - sx * sx / np.maximum(n, 1))
    beta = pd.Series(beta, index=d.index).shift(1)  # known before today's close
    beta[pd.Series(n, index=d.index).shift(1).fillna(0) < MIN_EST] = np.nan
    return beta


def main():
    rng = np.random.default_rng(SEED)
    d = spx().loc["1990":]
    d = d[d.vix.notna()]
    d["beta"] = leverage_slopes(d)
    d[["beta"]].dropna().resample("YE").last().to_csv(RES / "leverage_slope.csv", float_format="%.4f")

    close = d.close.to_numpy()
    T = T_DAYS
    P, idx = windows(close, T)
    V, _ = windows(d.vix.to_numpy(), T)
    V = V * d.vix.to_numpy()[idx, None] / 100.0  # undo the normalisation of windows()
    E, _ = windows(d.ewma.to_numpy(), T)
    E = E * d.ewma.to_numpy()[idx, None] / 100.0
    beta = d.beta.to_numpy()[idx]
    ok = np.isfinite(beta)
    P, V, E, beta, idx = P[ok], V[ok], E[ok], beta[ok], idx[ok]
    dates = d.index[idx]
    sq = V[:, 0]
    K = 100.0

    def mv_delta(S_i, K_, tau, i):
        sig = V[:, i]
        return call_delta(S_i, K_, tau, sig) + call_vega(S_i, K_, tau, sig) * beta / S_i

    rules = {
        "fixed": dict(sigma_delta=sq),
        "floating": dict(sigma_delta=V[:, :T]),
        "ewma": dict(sigma_delta=E[:, :T]),
        "mv": dict(delta_fn=mv_delta),
    }
    term, daily, means = {}, {}, {}
    for name, kw in rules.items():
        res = hedge_short_call(P, K, sq, **kw)
        term[name] = res.pnl
        # daily mark-to-market hedge errors, option marked at today's VIX
        unit = res.vega0 / 100.0
        err2 = np.zeros(len(P))
        for i in range(T):
            tau, tau1 = (T - i) * DT, (T - i - 1) * DT
            if "delta_fn" in kw:
                pos = mv_delta(P[:, i], K, tau, i)
            else:
                sd = kw["sigma_delta"]
                pos = call_delta(P[:, i], K, tau, sd if np.ndim(sd) == 1 else sd[:, i])
            c0 = call_price(P[:, i], K, tau, V[:, i])
            c1 = call_price(P[:, i + 1], K, tau1, V[:, i + 1])
            e = (-(c1 - c0) + pos * (P[:, i + 1] - P[:, i])) / unit
            err2 += e * e
        daily[name] = err2
        means[name] = res.pnl.mean()

    out = pd.DataFrame({"date": dates, **{f"terminal_{k}": v for k, v in term.items()},
                        **{f"daily_sse_{k}": v for k, v in daily.items()}})
    out.to_csv(RES / "which_delta_paths.csv.gz", index=False, float_format="%.5g")

    n = len(P)
    boots = [block_indices(n, rng) for _ in range(B)]
    rows, tests = [], []
    for name in rules:
        rows.append(dict(rule=name, n=n, mean_pnl=means[name], std_terminal=term[name].std(ddof=1),
                         rmse_daily=np.sqrt(daily[name].mean() / T)))
    for base in ["fixed", "floating"]:
        for name in rules:
            if name == base:
                continue
            g_term = 1 - term[name].var() / term[base].var()
            g_day = 1 - daily[name].mean() / daily[base].mean()
            bt = np.array([[1 - term[name][b].var() / term[base][b].var(),
                            1 - daily[name][b].mean() / daily[base][b].mean()] for b in boots])
            # two-sided p-value of "no change in variance" from the bootstrap distribution (re-centred)
            p_term = np.mean(np.abs(bt[:, 0] - bt[:, 0].mean()) >= abs(g_term))
            p_day = np.mean(np.abs(bt[:, 1] - bt[:, 1].mean()) >= abs(g_day))
            lo, hi = np.percentile(bt, [2.5, 97.5], axis=0)
            tests.append(dict(rule=name, baseline=base, gain_terminal=g_term, gain_terminal_lo=lo[0],
                              gain_terminal_hi=hi[0], p_terminal=max(p_term, 1 / B), gain_daily=g_day,
                              gain_daily_lo=lo[1], gain_daily_hi=hi[1], p_daily=max(p_day, 1 / B)))
    tests = pd.DataFrame(tests)
    tests["p_terminal_holm"] = holm(tests.p_terminal)
    tests["p_daily_holm"] = holm(tests.p_daily)
    pd.DataFrame(rows).to_csv(RES / "which_delta.csv", index=False, float_format="%.4f")
    tests.to_csv(RES / "which_delta_tests.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(rows).round(3).to_string())
    print(tests.round(3).to_string())
    print("median beta", np.median(beta))


if __name__ == "__main__":
    main()
