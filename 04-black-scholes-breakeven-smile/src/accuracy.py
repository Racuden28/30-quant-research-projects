"""Part I: how many correct digits does a Black-Scholes pricer have?

Compares the textbook formula and the stable pricer in bs.py with a 50-digit mpmath reference on a
grid of log-moneyness ln(K/F) in [-4, 4] and total volatility sigma*sqrt(T) in [0.002, 3], and
checks that implied_vol inverts every representable price. Writes results/accuracy_grid.csv and
results/accuracy_summary.csv.
"""
from __future__ import annotations

from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd

from bs import black_price, black_price_naive, implied_vol

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
mp.mp.dps = 60


def ref_price(F, K, v, call):
    F, K, v = mp.mpf(F), mp.mpf(K), mp.mpf(v)
    d1 = mp.log(F / K) / v + v / 2
    d2 = d1 - v
    if call:
        return F * mp.ncdf(d1) - K * mp.ncdf(d2)
    return K * mp.ncdf(-d2) - F * mp.ncdf(-d1)


def main():
    ks = np.linspace(-4, 4, 81)          # ln(K/F)
    vs = np.geomspace(0.002, 3.0, 61)    # total vol
    rows = []
    for k in ks:
        K = float(np.exp(k))
        for v in vs:
            call = k >= 0                 # value the OTM option: that is what carries information
            ref = ref_price(1.0, K, v, call)
            ref_f = float(ref)
            stable = black_price(1.0, K, v * v, 1.0, call=call)   # T = v^2, sigma = 1  -> total vol v
            naive = black_price_naive(1.0, K, v * v, 1.0, call=call)
            def rel(x):
                if ref_f == 0.0:
                    return np.nan
                return float(abs((mp.mpf(x) - ref) / ref))
            iv = implied_vol(stable, 1.0, K, v * v, call=call) if ref_f > 1e-300 else np.nan
            rows.append(dict(log_moneyness=k, total_vol=v, ref=ref_f, rel_err_stable=rel(stable),
                             rel_err_naive=rel(naive), representable=ref_f > 1e-300,
                             iv_rel_err=abs(iv - 1.0) if np.isfinite(iv) else np.nan))
    g = pd.DataFrame(rows)
    g.to_csv(RES / "accuracy_grid.csv", index=False)
    rep = g[g.representable]
    def digits(e):
        return -np.log10(np.maximum(e, 1e-17))
    summ = pd.DataFrame({
        "pricer": ["textbook F N(d1) - K N(d2)", "stable (OTM side + erfcx)"],
        "median_correct_digits": [np.nanmedian(digits(rep.rel_err_naive)), np.nanmedian(digits(rep.rel_err_stable))],
        "worst_correct_digits": [np.nanmin(digits(rep.rel_err_naive.fillna(1))), np.nanmin(digits(rep.rel_err_stable.fillna(1)))],
        "share_cells_rel_err_gt_1e-6": [(rep.rel_err_naive.fillna(1) > 1e-6).mean(), (rep.rel_err_stable.fillna(1) > 1e-6).mean()],
        "share_cells_no_correct_digit": [(rep.rel_err_naive.fillna(1) >= 1).mean(), (rep.rel_err_stable.fillna(1) >= 1).mean()],
    })
    summ["iv_roundtrip_max_rel_err"] = [np.nan, np.nanmax(rep.iv_rel_err)]
    summ["iv_roundtrip_failures"] = [np.nan, int(rep.iv_rel_err.isna().sum() + (rep.iv_rel_err > 1e-8).sum())]
    summ["cells"] = len(rep)
    summ.to_csv(RES / "accuracy_summary.csv", index=False)
    print(summ.to_string())


if __name__ == "__main__":
    main()
