"""Part I: accuracy and speed of 13 lattice methods for American options.

Outputs
    results/accuracy_sample.csv       the random contracts and their reference values
    results/accuracy_errors.csv       per method x n: RMS relative price error, max abs error, EEP error, time
    results/convergence_trace.csv     price vs n for one contract, every n from 10 to 1000
    results/reference_check.csv       reference vs an independent extrapolated BBS at much larger n
"""
from __future__ import annotations

import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import trees as tr

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

N_PUT, N_CALL = 1000, 300
NS = [25, 50, 100, 200, 400, 800, 1600]
REF_N = 20002          # LR pair (10001, 20003) with O(1/n) Richardson
CHECK_N = 40000        # independent check: BBS + Richardson at (20000, 40000)
N_CHECK = 60
SEED = 15


def sample_contracts(rng):
    """Broadie-Detemple (1996) style random contracts, S = 100."""
    rows = []
    for k, is_call in [(N_PUT, False), (N_CALL, True)]:
        K = rng.uniform(70, 130, k)
        sig = rng.uniform(0.10, 0.60, k)
        long = rng.random(k) < 0.25
        T = np.where(long, rng.uniform(1.0, 5.0, k), rng.uniform(0.1, 1.0, k))
        r = np.where(rng.random(k) < 0.8, rng.uniform(0.0, 0.10, k), 0.0)
        q = rng.uniform(0.0, 0.10, k)
        if is_call:   # calls only carry an early-exercise premium when q > 0
            q = rng.uniform(0.01, 0.10, k)
        rows.append(pd.DataFrame(dict(S=100.0, K=K, T=T, r=r, q=q, sigma=sig, is_call=is_call)))
    return pd.concat(rows, ignore_index=True)


def _ref(row):
    S, K, T, r, q, s, c = row
    am = tr.lr_r(S, K, T, r, q, s, bool(c), True, REF_N)
    return am


def _check(row):
    S, K, T, r, q, s, c = row
    return tr.bbsr(S, K, T, r, q, s, bool(c), True, CHECK_N), tr.lr_r(S, K, T, r, q, s, bool(c), True, 2 * REF_N)


def main():
    rng = np.random.default_rng(SEED)
    df = sample_contracts(rng)
    rows = list(df[["S", "K", "T", "r", "q", "sigma", "is_call"]].itertuples(index=False, name=None))
    tr.lr_r(100.0, 100.0, 1.0, 0.05, 0.02, 0.2, False, True, 101)  # compile before forking
    tr.bbsr(100.0, 100.0, 1.0, 0.05, 0.02, 0.2, False, True, 100)
    t0 = time.time()
    with Pool(os.cpu_count()) as pool:
        df["ref"] = pool.map(_ref, rows, chunksize=8)
        chk = pool.map(_check, rows[:N_CHECK // 2] + rows[N_PUT:N_PUT + N_CHECK // 2], chunksize=2)
    print(f"reference values: {time.time() - t0:.0f}s")
    df["euro"] = [tr.bs_price(*r_[:6], bool(r_[6])) for r_ in rows]
    df["eep"] = df.ref - df.euro
    df.to_csv(RES / "accuracy_sample.csv", index=False)
    cidx = list(range(N_CHECK // 2)) + list(range(N_PUT, N_PUT + N_CHECK // 2))
    ck = df.loc[cidx, ["K", "T", "r", "q", "sigma", "is_call", "ref"]].copy()
    ck["bbsr_40000"] = [c[0] for c in chk]
    ck["lr_r_40004"] = [c[1] for c in chk]
    ck.to_csv(RES / "reference_check.csv", index=False)
    print("reference check: max |ref - BBSR(40000)| =", np.abs(ck.ref - ck.bbsr_40000).max(),
          " max |ref - LR_R(40004)| =", np.abs(ck.ref - ck.lr_r_40004).max())

    # ---- error grid ----
    arr = df[["S", "K", "T", "r", "q", "sigma"]].to_numpy()
    calls = df.is_call.to_numpy()
    out = []
    for name, f in tr.METHODS.items():
        f(100.0, 100.0, 1.0, 0.05, 0.02, 0.2, False, True, 51)
        f(100.0, 100.0, 1.0, 0.05, 0.02, 0.2, True, True, 51)
        for n in NS:
            t = time.perf_counter()
            v = np.array([f(*a, bool(c), True, n) for a, c in zip(arr, calls)])
            el = (time.perf_counter() - t) / len(arr)
            for grp, m in [("put", ~calls), ("call", calls), ("all", np.ones_like(calls))]:
                d = df[m]
                vv = v[m]
                big = d.ref.to_numpy() >= 0.5          # Broadie-Detemple: relative error on prices >= 0.5
                rel = (vv - d.ref.to_numpy()) / d.ref.to_numpy()
                eepm = d.eep.to_numpy() >= 0.05        # EEP error on contracts whose premium is >= 5 cents
                eep_rel = (vv - d.ref.to_numpy()) / d.eep.to_numpy()
                out.append(dict(method=name, label=tr.LABELS[name], n=n, group=grp, sec_per_option=el,
                                rmse_rel=np.sqrt(np.mean(rel[big] ** 2)),
                                max_abs=np.max(np.abs(vv - d.ref.to_numpy())),
                                rmse_abs=np.sqrt(np.mean((vv - d.ref.to_numpy()) ** 2)),
                                rmse_rel_eep=np.sqrt(np.mean(eep_rel[eepm] ** 2)),
                                med_abs_rel_eep=np.median(np.abs(eep_rel[eepm])),
                                n_price=int(big.sum()), n_eep=int(eepm.sum())))
        print(name, "done")
    pd.DataFrame(out).to_csv(RES / "accuracy_errors.csv", index=False)

    # ---- convergence trace for one contract (Hull's American put) ----
    c = (50.0, 50.0, 5 / 12, 0.10, 0.0, 0.40)
    refc = tr.lr_r(*c, False, True, 2 * REF_N)
    tr_rows = []
    for n in range(10, 1001):
        rec = dict(n=n)
        for name in ["crr", "tian", "lr", "bbs", "bbsr", "lr_r", "crr_cv", "tri"]:
            rec[name] = tr.METHODS[name](*c, False, True, n)
        tr_rows.append(rec)
    t = pd.DataFrame(tr_rows)
    t["ref"] = refc
    t.to_csv(RES / "convergence_trace.csv", index=False)
    print("Hull put reference:", refc)


if __name__ == "__main__":
    main()
