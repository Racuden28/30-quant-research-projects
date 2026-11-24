"""Run every specification of the study and save compact daily outputs to results/runs/.

Usage: python src/experiments.py [--quick]
Each run stores, per signal, the daily gross P&L for delay 0 and delay 1, turnover, short
exposure and gross exposure, plus the daily diagnostics and information coefficients.
"""
from __future__ import annotations
import os, sys, pickle, time
from multiprocessing import Pool
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
import engine as en

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "runs")
N_PLACEBO = 20

UNIV = {
    # universe: (periods per year, base one-way cost bps, short borrow APR)
    "equity": dict(periods=252, cost=5.0, borrow=0.005),
    "crypto": dict(periods=365, cost=10.0, borrow=0.10),
}


def specs(quick: bool = False, placebo_only: str | None = None):
    S = []
    if placebo_only:
        for seed in range(N_PLACEBO):
            S.append(("equity", f"placebo{seed:02d}", dict(k="mp", placebo_seed=seed, placebo_method=placebo_only)))
            S.append(("crypto", f"placebo{seed:02d}", dict(k="mp", N=100, placebo_seed=seed, placebo_method=placebo_only)))
        return S
    # main specification: k chosen by the Marchenko-Pastur edge (project #8), capped at 15
    S.append(("equity", "main", dict(k="mp")))
    S.append(("crypto", "main", dict(k="mp", N=100)))
    if not quick:
        for k in (0, 1, 5, 15):
            S.append(("equity", f"k{k}", dict(k=k)))
            S.append(("crypto", f"k{k}", dict(k=k, N=100)))
        S.append(("equity", "pit_added", dict(k="mp")))           # only after S&P inclusion date
        S.append(("crypto", "survivors", dict(k="mp", N=100)))     # today's survivors only
        S.append(("crypto", "N50", dict(k="mp", N=50)))
        S.append(("equity", "L120", dict(k="mp", L=120)))
        S.append(("crypto", "L120", dict(k="mp", N=100, L=120)))
        for seed in range(N_PLACEBO):
            S.append(("equity", f"placebo{seed:02d}", dict(k="mp", placebo_seed=seed)))
            S.append(("crypto", f"placebo{seed:02d}", dict(k="mp", N=100, placebo_seed=seed)))
    return S


_cache = {}


def _load(universe: str, label: str):
    key = (universe, label == "survivors", label == "pit_added")
    if key in _cache:
        return _cache[key]
    if universe == "crypto":
        ret, liq, _ = data.load_crypto(survivors_only=(label == "survivors"))
    else:
        ret, liq, meta, _ = data.load_equity()
        if label == "pit_added":
            added = meta["added"].reindex(ret.columns)
            mask = pd.DataFrame(ret.index.values[:, None] >= added.values[None, :],
                                index=ret.index, columns=ret.columns)
            ret = ret.where(mask)
    _cache[key] = (ret, liq)
    return ret, liq


def run_one(spec):
    universe, label, kw = spec
    out_dir = OUT if kw.get("placebo_method", "signflip") == "signflip" else OUT + "_" + kw["placebo_method"]
    path = os.path.join(out_dir, f"{universe}_{label}.pkl")
    if os.path.exists(path):
        return path
    t0 = time.time()
    ret, liq = _load(universe, label)
    cfg = en.Config(**kw)
    res = en.run(ret, liq, cfg)
    u = UNIV[universe]
    out = {"universe": universe, "label": label, "cfg": cfg.__dict__, "diag": res.diag, "ic": res.ic,
           "ou_cross": pd.DataFrame(res.ou_cross, columns=["date", "k", "kappa", "b"]), "pnl": {}}
    first = res.diag.index[(res.diag["n_assets"] >= cfg.min_assets).to_numpy()]
    start = first[0] if len(first) else res.diag.index[0]
    for s, w in res.weights.items():
        cols = {}
        for d in (0, 1):
            p = en.pnl(w, res.ret, delay=d, periods=u["periods"])
            cols[f"gross_d{d}"] = p["gross"]; cols[f"turn_d{d}"] = p["turnover"]
            if d == 0:
                cols["short"] = p["short"]; cols["gexp"] = p["gross_exposure"]
        out["pnl"][s] = pd.DataFrame(cols).loc[start:].astype("float32")
    # long and short legs of the main signal, for the crypto loss decomposition
    w = res.weights["ou"].shift(1).fillna(0.0); r = res.ret.fillna(0.0)
    out["by_asset"] = (w * r).loc[start:].sum().astype("float64")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"{universe:7s} {label:12s} {time.time() - t0:6.1f}s", flush=True)
    return path


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    pm = sys.argv[sys.argv.index("--placebo") + 1] if "--placebo" in sys.argv else None
    S = specs(quick, pm)
    # crypto runs are fast; run equity first so the pool stays busy
    S.sort(key=lambda s: s[0] != "equity")
    with Pool(int(os.environ.get("NPROC", "4"))) as pool:
        for _ in pool.imap_unordered(run_one, S):
            pass
