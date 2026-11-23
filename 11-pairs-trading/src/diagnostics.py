"""
Where is the mean reversion, and why does the textbook rule not collect it?

For the 20 distance-selected pairs of every window, and for every trading day t, take the spread z_t (in
formation standard deviations, as the rule sees it) and measure how far the spread moves back toward zero over
the next h days, skipping one day as the rule does: R_h = -sign(z_t) * (z_{t+1+h} - z_{t+1}). Averages are
reported by bucket of |z_t|, for real data and for the drift-equalised IID placebo (3 replications per window),
whose average removes mechanical effects (for example the fact that a spread far from zero has more room to
fall back). Output: results/reversion_by_distance.csv.
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd

from backtest import UNIVERSE_ID, _extra_ok, demean, month_starts, setup
from core import FORM, NPAIRS, TRADE, distance_signal, distance_stats, eligible, iid_days, pair_index
from data import RES
from evaluate import MIN_ASSETS

BUCKETS = [0, 0.5, 1, 1.5, 2, 3, np.inf]
HS = (5, 21, 63)
NREP = 3
G = {}


def _init(u):
    r, extra = setup(u)
    G.update(R=r.values, dates=r.index, extra=extra, uid=UNIVERSE_ID[u])


def _moves(b):
    rf, rt = b[:FORM], b[FORM:]
    ia, ib = pair_index(b.shape[1])
    ssd, sd = distance_stats(rf, ia, ib)
    sel = np.argsort(ssd, kind="stable")[:NPAIRS]
    z = distance_signal(rt, ia[sel], ib[sel], sd[sel])            # (TRADE, 20)
    out = np.zeros((len(BUCKETS) - 1, len(HS), 2))                # sum, count
    for j, h in enumerate(HS):
        if 1 + h >= TRADE:
            continue
        z0, z1, zh = z[:TRADE - 1 - h], z[1:TRADE - h], z[1 + h:]
        mv = -np.sign(z0) * (zh - z1)
        bk = np.digitize(np.abs(z0), BUCKETS) - 1
        for q in range(len(BUCKETS) - 1):
            m = bk == q
            out[q, j, 0] += mv[m].sum()
            out[q, j, 1] += m.sum()
    return out


def work(k_s):
    k, s = k_s
    R, dates, extra, uid = G["R"], G["dates"], G["extra"], G["uid"]
    block = R[s - FORM:s + TRADE]
    ok = eligible(block, _extra_ok(extra, dates, s))
    if ok.sum() < MIN_ASSETS:
        return None
    b = np.ascontiguousarray(block[:, ok])
    real = _moves(b)
    bd = demean(b)
    pl = sum(_moves(np.ascontiguousarray(bd[iid_days(len(b), np.random.default_rng([uid, k, rep, 6]))]))
             for rep in range(NREP)) / NREP
    return real, pl


def run(u):
    r, _ = setup(u)
    starts = month_starts(r.index)
    with Pool(4, initializer=_init, initargs=(u,)) as p:
        res = [x for x in p.map(work, list(enumerate(starts)), chunksize=8) if x is not None]
    real = sum(x[0] for x in res)
    pl = sum(x[1] for x in res)
    rows = []
    for q in range(len(BUCKETS) - 1):
        for j, h in enumerate(HS):
            rows.append({"universe": u, "abs_z_from": BUCKETS[q], "abs_z_to": BUCKETS[q + 1], "h": h,
                         "real_move": real[q, j, 0] / real[q, j, 1], "placebo_move": pl[q, j, 0] / pl[q, j, 1],
                         "pair_days": int(real[q, j, 1])})
    df = pd.DataFrame(rows)
    df["excess_move"] = df.real_move - df.placebo_move
    return df


if __name__ == "__main__":
    out = pd.concat([run(u) for u in ("countries", "industries")])
    out.to_csv(RES / "reversion_by_distance.csv", index=False, float_format="%.5g")
    print(out.round(3).to_string())
