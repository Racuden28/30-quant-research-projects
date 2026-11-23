"""Part II: ninety-eight years of the early-exercise premium on the S&P 500.

For every trading day from 1928-04 to 2026-11 price American and European puts and calls on the index with
the prevailing short rate, dividend yield and trailing 63-day volatility (data/sp500_option_inputs.csv).
The engine is the one Part I selects (LR + Richardson with the Hull-White control variate on top; see README 2.4).

Outputs
    results/history_daily.csv.gz    per day x contract: American, European, EEP, EEP share, EEP in vol points
    results/history_boundary.csv month-end critical price S*/K of the 3-month ATM put and call
    results/history_summary.csv  per decade and per stress window
    results/rq_grid.csv          EEP share of 3m ATM put and call on an (r, q) grid at sigma = 15%
    results/history_robustness.csv  EEP share under alternative volatility inputs
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

import trees as tr

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_ENGINE = 400
REUSE = "--reuse" in __import__("sys").argv

# (name, is_call, K/S, T in years)
CONTRACTS = [
    ("put_atm_1m", False, 1.00, 1 / 12), ("put_atm_3m", False, 1.00, 0.25), ("put_atm_1y", False, 1.00, 1.0),
    ("put_otm_3m", False, 0.90, 0.25), ("put_itm_3m", False, 1.10, 0.25),
    ("call_atm_1m", True, 1.00, 1 / 12), ("call_atm_3m", True, 1.00, 0.25), ("call_atm_1y", True, 1.00, 1.0),
    ("call_itm_3m", True, 0.90, 0.25),
]
STRESS = {
    "1929 crash": ("1929-09-01", "1932-07-31"), "1937-38": ("1937-03-01", "1938-03-31"),
    "1973-74": ("1973-01-01", "1974-12-31"), "1987 crash": ("1987-08-25", "1987-12-31"),
    "2008 GFC": ("2008-09-01", "2009-03-31"), "2020 Covid": ("2020-02-19", "2020-04-30"),
    "2022 hikes": ("2022-01-01", "2023-10-31"),
}


@njit(cache=True)
def engine(S, K, T, r, q, sigma, is_call, n):
    """LR + Richardson (O(1/n)) on the EEP, with the exact BS European added back (control variate)."""
    a = tr.lr_r(S, K, T, r, q, sigma, is_call, True, n)
    e = tr.lr_r(S, K, T, r, q, sigma, is_call, False, n)
    return a - e + tr.bs_price(S, K, T, r, q, sigma, is_call)


@njit(cache=True)
def bs_vega(S, K, T, r, q, sigma):
    sd = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sd
    return S * math.exp(-q * T) * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(T)


@njit(cache=True)
def price_panel(r, q, sig, is_call, kfac, T, n):
    m = r.shape[0]
    am = np.empty(m)
    eu = np.empty(m)
    vega = np.empty(m)
    for i in range(m):
        K = 100.0 * kfac
        eu[i] = tr.bs_price(100.0, K, T, r[i], q[i], sig[i], is_call)
        am[i] = max(engine(100.0, K, T, r[i], q[i], sig[i], is_call, n), eu[i], max((K - 100.0) if not is_call else (100.0 - K), 0.0))
        vega[i] = bs_vega(100.0, K, T, r[i], q[i], sig[i])
    return am, eu, vega


@njit(cache=True)
def _am(S, K, T, r, q, sigma, is_call, n):
    # BBS + Richardson: robust far from the strike, where the LR inversion can degenerate (p -> 0 or 1)
    return tr.bbsr(S, K, T, r, q, sigma, is_call, True, n)


@njit(cache=True)
def critical_price(K, T, r, q, sigma, is_call, n):
    """S* where the American option equals intrinsic value (bisection on S). NaN if never optimal."""
    if is_call:
        lo, hi = K, K * 5.0
        if _am(hi, K, T, r, q, sigma, True, n) > hi - K + 1e-9:
            return np.nan
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if _am(mid, K, T, r, q, sigma, True, n) > mid - K + 1e-9 * K:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    lo, hi = K * 0.2, K
    if _am(lo, K, T, r, q, sigma, False, n) > K - lo + 1e-9:
        return np.nan
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _am(mid, K, T, r, q, sigma, False, n) > K - mid + 1e-9 * K:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main():
    d = pd.read_csv(ROOT / "data" / "sp500_option_inputs.csv", parse_dates=["date"], index_col="date")
    r, q, s = d.r.to_numpy(), d.q.to_numpy(), d.sigma.to_numpy()
    if REUSE and (RES / "history_daily.csv.gz").exists():
        out = pd.read_csv(RES / "history_daily.csv.gz", parse_dates=["date"], index_col="date")
    else:
        out = price_all(d, r, q, s)
    rest(d, out, r, q, s)


def price_all(d, r, q, s):
    out = pd.DataFrame(index=d.index)
    out["r"], out["q"], out["sigma"] = r, q, s
    for name, call, kf, T in CONTRACTS:
        am, eu, vega = price_panel(r, q, s, call, kf, T, N_ENGINE)
        out[name + "_am"] = am
        out[name + "_eu"] = eu
        out[name + "_eep"] = am - eu
        out[name + "_share"] = (am - eu) / am
        out[name + "_volpts"] = (am - eu) / vega * 100.0   # vol points a European pricer would have to add
        print(name, "done")
    out.to_csv(RES / "history_daily.csv.gz", float_format="%.6g")
    return out


def rest(d, out, r, q, s):

    # month-end critical prices of the 3m ATM put and call
    me = d.resample("ME").last().dropna()
    bnd = pd.DataFrame(index=me.index)
    bnd["put_Sstar_over_K"] = [critical_price(100.0, 0.25, a.r, a.q, a.sigma, False, 200) / 100.0 for a in me.itertuples()]
    bnd["call_Sstar_over_K"] = [critical_price(100.0, 0.25, a.r, a.q, a.sigma, True, 200) / 100.0 for a in me.itertuples()]
    bnd.to_csv(RES / "history_boundary.csv")

    # summaries
    cols = [c for c in out.columns if c.endswith("_share") or c.endswith("_volpts")]
    dec = out.groupby(out.index.year // 10 * 10)[["r", "q", "sigma"] + cols].mean()
    dec.index = [f"{y}s" for y in dec.index]
    rows = [dec]
    for k, (a, b) in STRESS.items():
        w = out.loc[a:b, ["r", "q", "sigma"] + cols].mean().to_frame(k).T
        rows.append(w)
    full = out[["r", "q", "sigma"] + cols].mean().to_frame("full sample").T
    rows.append(full)
    summ = pd.concat(rows)
    # share of days on which the European price is more than 1% / 5% below the American
    for name, *_ in CONTRACTS:
        summ.loc["full sample", name + "_days_gt1pct"] = (out[name + "_share"] > 0.01).mean()
        summ.loc["full sample", name + "_days_gt5pct"] = (out[name + "_share"] > 0.05).mean()
    summ.to_csv(RES / "history_summary.csv")
    print(summ[["r", "q", "put_atm_3m_share", "call_atm_3m_share", "put_itm_3m_share"]].round(4).to_string())

    # (r, q) grid at sigma = 15%: the map behind the star chart
    rg = np.linspace(0.0, 0.16, 81)
    qg = np.linspace(0.0, 0.09, 46)
    R, Q = np.meshgrid(rg, qg)
    sig = np.full(R.size, 0.15)
    pam, peu, _ = price_panel(R.ravel(), Q.ravel(), sig, False, 1.0, 0.25, 200)
    cam, ceu, _ = price_panel(R.ravel(), Q.ravel(), sig, True, 1.0, 0.25, 200)
    pd.DataFrame(dict(r=R.ravel(), q=Q.ravel(), put_share=(pam - peu) / pam, call_share=(cam - ceu) / cam)
                 ).to_csv(RES / "rq_grid.csv", index=False)

    # robustness of the 3m ATM shares to the volatility input
    lr_vol = pd.Series(np.log(d.S)).diff().rolling(252 * 5, min_periods=252).std().to_numpy() * np.sqrt(252)
    ew = pd.Series(np.log(d.S)).diff().pow(2).ewm(halflife=21).mean().pow(0.5).to_numpy() * np.sqrt(252)
    rob = {}
    for lab, v in [("trailing 63d (main)", s), ("trailing 5y", lr_vol), ("EWMA halflife 21d", ew),
                   ("constant 15%", np.full_like(s, 0.15))]:
        ok = ~np.isnan(v)
        for name, call, kf, T in [CONTRACTS[1], CONTRACTS[6]]:
            am, eu, _ = price_panel(r[ok], q[ok], v[ok], call, kf, T, 200)
            ser = pd.Series((am - eu) / am, index=d.index[ok])
            rob[(lab, name)] = ser.groupby(ser.index.year // 10 * 10).mean()
    robd = pd.DataFrame(rob)
    robd.index = [f"{y}s" for y in robd.index]
    robd.to_csv(RES / "history_robustness.csv")
    print(robd.round(4).to_string())


if __name__ == "__main__":
    main()
