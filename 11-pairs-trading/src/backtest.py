"""
Run every window of both universes on real data and on the two placebo markets.

For each monthly start s (first trading day of a month), the block R[s-252 : s+126] of eligible assets is
traded with every configuration in REAL_CONFIGS; then 10 IID-day placebo blocks (N1, every asset shifted to the
same drift first) are built from that same block and traded with PLACEBO_CONFIGS, and 10 drift-keeping IID
blocks (ND, ablation) are traded with the primary configuration.
The cointegration screen (formation year vs the following year) is run on R[s-252 : s+252].

Output: results/raw_<universe>.npz (daily P&L and turnover per window, config, source and replication) and
results/screen_<universe>.csv. Takes about 3 minutes on 4 cores for both universes.
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from core import FORM, TRADE, VR_H, coint_screen, eligible, iid_days, run_window
from data import DATA, RES, load

NREP = 10
REAL_CONFIGS = [
    dict(name="D20_w1", method="dist", ranks="top", wait=1),   # primary (GGR)
    dict(name="D20_w0", method="dist", ranks="top", wait=0),
    dict(name="D20_w5", method="dist", ranks="top", wait=5),
    dict(name="D101_w1", method="dist", ranks="mid", wait=1),
    dict(name="Dall_w1", method="dist", ranks="all", wait=1),
    dict(name="D20c_w1", method="distc", ranks="top", wait=1),  # ablation: no renormalisation at trading start
    dict(name="C20_w1", method="coint", ranks="top", wait=1),
    dict(name="C20_w0", method="coint", ranks="top", wait=0),
]
PLACEBO_CONFIGS = [c for c in REAL_CONFIGS if c["name"] in ("D20_w1", "D20_w0", "C20_w1", "Dall_w1", "D20c_w1")]
DRIFT_CONFIGS = [c for c in REAL_CONFIGS if c["name"] == "D20_w1"]
UNIVERSE_ID = {"industries": 1, "countries": 2}


def demean(b: np.ndarray) -> np.ndarray:
    """Give every asset the same drift (the cross-sectional average of the window's mean returns).

    Placebo days are drawn from the whole window, trading period included. Without this step each asset keeps
    its realised drift, so the placebo 'knows' which asset will outperform in the trading period and
    punishes any contrarian rule; see README section 3.3."""
    mu = b.mean(axis=0)
    return b - mu + mu.mean()


def month_starts(index: pd.DatetimeIndex) -> np.ndarray:
    m = index.to_period("M")
    first = np.r_[True, m[1:] != m[:-1]]
    s = np.nonzero(first)[0]
    return s[(s >= FORM) & (s + TRADE <= len(index))]


def setup(universe: str):
    r = load(universe)
    extra = None
    if universe == "industries":
        nf = pd.read_csv(DATA / "industries_nfirms_monthly.csv", index_col=0, parse_dates=True)[r.columns]
        extra = nf
    return r, extra


def _extra_ok(extra, dates, s):
    if extra is None:
        return None
    start = dates[s - FORM]                       # month-end before formation starts
    prev = (start.to_period("M") - 1).to_timestamp()
    return (extra.loc[prev].values >= 5)


G = {}


def _init(universe):
    r, extra = setup(universe)
    G.update(R=r.values, dates=r.index, extra=extra, uid=UNIVERSE_ID[universe])


def work(k_s):
    k, s = k_s
    R, dates, extra, uid = G["R"], G["dates"], G["extra"], G["uid"]
    block = R[s - FORM:s + TRADE]
    ok = eligible(block, _extra_ok(extra, dates, s))
    cols = np.nonzero(ok)[0]
    b = np.ascontiguousarray(block[:, cols])
    res = {"real": run_window(b, REAL_CONFIGS)}
    bd = demean(b)
    for rep in range(NREP):
        rng = np.random.default_rng([uid, k, rep, 1])
        res[f"N1_{rep}"] = run_window(np.ascontiguousarray(bd[iid_days(len(b), rng)]), PLACEBO_CONFIGS)
        rng = np.random.default_rng([uid, k, rep, 4])          # ablation: keep each asset's realised drift
        res[f"ND_{rep}"] = run_window(np.ascontiguousarray(b[iid_days(len(b), rng)]), DRIFT_CONFIGS)
    screens = []
    if s + FORM <= len(R):
        block2 = R[s - FORM:s + FORM]
        ok2 = ~np.isnan(block2).any(0) & ((block2[:FORM] == 0).mean(0) < 0.05)
        ex = _extra_ok(extra, dates, s)
        if ex is not None:
            ok2 &= ex
        b2 = np.ascontiguousarray(block2[:, np.nonzero(ok2)[0]])
        if b2.shape[1] >= 2:
            screens.append({"k": k, "source": "real", "rep": -1, **coint_screen(b2)})
            for rep in range(NREP):
                rng = np.random.default_rng([uid, k, rep, 3])
                screens.append({"k": k, "source": "N1", "rep": rep,
                                **coint_screen(np.ascontiguousarray(demean(b2)[iid_days(len(b2), rng)]))})
    return k, s, len(cols), res, screens


def run(universe: str, procs: int = 4, tag: str = ""):
    r, _ = setup(universe)
    starts = month_starts(r.index)
    names_real = [c["name"] for c in REAL_CONFIGS]
    names_pl = [c["name"] for c in PLACEBO_CONFIGS]
    K = len(starts)
    pnl = {("real", n): np.full((K, 1, TRADE), np.nan) for n in names_real}
    turn = {("real", n): np.full((K, 1, TRADE), np.nan) for n in names_real}
    stats = {("real", n): np.full((K, 1, 6), np.nan) for n in names_real}
    vr = {("real", n): np.full((K, 1, len(VR_H)), np.nan) for n in names_real}
    for src, names in (("N1", names_pl), ("ND", [c["name"] for c in DRIFT_CONFIGS])):
        for n in names:
            pnl[(src, n)] = np.full((K, NREP, TRADE), np.nan)
            turn[(src, n)] = np.full((K, NREP, TRADE), np.nan)
            stats[(src, n)] = np.full((K, NREP, 6), np.nan)
            vr[(src, n)] = np.full((K, NREP, len(VR_H)), np.nan)
    nassets = np.zeros(K, int)
    screens = []
    with Pool(procs, initializer=_init, initargs=(universe,)) as pool:
        for i, (k, s, na, res, sc) in enumerate(pool.imap_unordered(work, list(enumerate(starts)), chunksize=4)):
            nassets[k] = na
            screens += sc
            for key, out in res.items():
                src, rep = ("real", 0) if key == "real" else (key[:2], int(key[3:]))
                for n, v in out.items():
                    pnl[(src, n)][k, rep] = v["pnl"]
                    turn[(src, n)][k, rep] = v["turn"]
                    stats[(src, n)][k, rep] = v["stats"]
                    vr[(src, n)][k, rep] = v["vr"]
            if i % 100 == 0:
                print(universe, i, K, flush=True)
    arrays = {"starts": starts, "nassets": nassets,
              "dates": r.index.values.astype("datetime64[D]").astype("int64")}
    for (src, n), v in pnl.items():
        arrays[f"pnl|{src}|{n}"] = v
        arrays[f"turn|{src}|{n}"] = turn[(src, n)]
        arrays[f"stats|{src}|{n}"] = stats[(src, n)]
        arrays[f"vr|{src}|{n}"] = vr[(src, n)]
    np.savez_compressed(RES / f"raw_{universe}{tag}.npz", **arrays)
    pd.DataFrame(screens).sort_values(["k", "source", "rep"]).to_csv(RES / f"screen_{universe}{tag}.csv", index=False,
                                                                      float_format="%.6g")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args = sys.argv[1:]
    if "--nofilter" in args:                      # ablation: no stale-price filter (countries only)
        import core
        core.MAX_ZERO_SHARE = 1.01
        run("countries", tag="_nofilter")
    else:
        for u in (args or ["countries", "industries"]):
            run(u)
