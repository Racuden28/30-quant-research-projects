"""
Why the frontier bends: the event-time decay of the momentum premium.

alpha_k = mean return in month m+1 of the textbook long-short book *formed at the end of m-k*
(equal weights, rebalanced to that formation-date selection every month, so this is the premium of
a k-month-old signal). If the premium of a k-month-old signal were the same as a fresh one, slowing
down would cost nothing; if it vanished after one month, slowing down would cost the whole premium.

A slow design is a mixture of old signals. Ignoring weight drift, partial rebalancing at rate theta
holds signal ages with weights theta (1-theta)^k and a K-month overlapping hold holds ages 0..K-1
with weight 1/K. So the decay curve alone predicts the gross premium of every partial and hold
design; prediction vs backtest is in results/decay_prediction.csv.

Outputs: results/decay_alpha.csv (alpha_k with 95% CI, by era and universe),
         results/decay_prediction.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data
import engine as E
import stats as S
from backtest import universe_a, universe_b
from designs import ERAS, PRIMARY_SIGNAL, SIGNALS

K_MAX = 24


def textbook_weights(U: E.Universe, sig: pd.DataFrame, q: float = 0.3) -> pd.DataFrame:
    elig = sig.notna() & U.present
    if U.extra_eligible is not None:
        elig &= U.extra_eligible.reindex_like(elig).fillna(False)
    pct = sig.where(elig).rank(axis=1, pct=True)
    L = (pct > 1 - q).astype(float)
    Sh = (pct <= q).astype(float)
    return L.div(L.sum(axis=1), axis=0).fillna(0) - Sh.div(Sh.sum(axis=1), axis=0).fillna(0)


def event_time(U: E.Universe, sig: pd.DataFrame) -> pd.DataFrame:
    W = textbook_weights(U, sig)
    nxt = U.monthly.shift(-1).fillna(0.0)
    out = {}
    for k in range(K_MAX + 1):
        wk = W.shift(k)
        r = (wk * nxt).sum(axis=1).where(wk.abs().sum(axis=1) > 0)
        out[k] = r.shift(1)                     # index = month in which the return is earned
    return pd.DataFrame(out)


def main():
    runs = pd.read_csv(data.RES / "runs.csv.gz", parse_dates=["month"])
    rows, pred = [], []
    for uname, U in (("A", universe_a()), ("B", universe_b())):
        for sname, (lb, sk) in SIGNALS.items():
            sig = E.momentum_signal(U, lb, sk)
            D = event_time(U, sig)
            for era, (a, b) in ERAS[uname].items():
                x = D.loc[a:b].dropna()
                idx = S.stationary_indices(len(x))
                for k in range(K_MAX + 1):
                    v = x[k].values
                    bm = S.boot_means(v, idx)
                    lo, hi = S.ci(1200 * bm)
                    rows.append({"universe": uname, "signal": sname, "era": era, "age": k,
                                 "alpha_pct_yr": 1200 * v.mean(), "lo": lo, "hi": hi,
                                 "t": v.mean() / v.std(ddof=1) * np.sqrt(len(v))})
                # cumulative "retention": premium of ages 1-3 relative to age 0, with CI
                ret3 = x[[1, 2, 3]].mean(axis=1).values
                a0 = x[0].values
                b3, b0 = S.boot_means(ret3, idx), S.boot_means(a0, idx)
                r_lo, r_hi = S.ci(b3 / b0)
                alpha = x.mean().values
                sub = runs[(runs.universe == uname) & (runs.signal == sname)]
                for d, g in sub.groupby("design"):
                    if d.startswith("partial"):
                        th = int(d[7:]) / 100
                        w = th * (1 - th) ** np.arange(K_MAX + 1)
                    elif d.startswith("hold"):
                        K = int(d[4:])
                        w = np.where(np.arange(K_MAX + 1) < K, 1.0 / K, 0.0)
                    elif d == "base":
                        w = np.eye(K_MAX + 1)[0]
                    else:
                        continue
                    w = w / w.sum()
                    actual = g.set_index("month")["gross"].reindex(x.index).mean()   # same months as alpha
                    pred.append({"universe": uname, "signal": sname, "era": era, "design": d,
                                 "predicted_pct_yr": 1200 * float(w @ alpha),
                                 "actual_pct_yr": 1200 * actual,
                                 "retention_1to3": float(ret3.mean() / a0.mean()),
                                 "retention_lo": r_lo, "retention_hi": r_hi})
    pd.DataFrame(rows).to_csv(data.RES / "decay_alpha.csv", index=False, float_format="%.5g")
    p = pd.DataFrame(pred)
    p.to_csv(data.RES / "decay_prediction.csv", index=False, float_format="%.5g")
    q = p[(p.universe == "A") & (p.signal == PRIMARY_SIGNAL)]
    print(q.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
