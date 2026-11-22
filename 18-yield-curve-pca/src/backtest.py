"""Walk-forward hedging experiment.

Every 21 trading days each model is re-estimated on the trailing W days of yield changes and its
hedge ratios are frozen for the next 21 days. Nothing after the rebalance date is used, except by the
labelled 'clairvoyant' reference, which fits OLS on the holding period itself (a lower bound for any
static hedge with the same instruments).

Output: results/cache/<panel>_W<W>_lam<lam>.npz with, for every (method, k), the residual design
E (days x |T|): the hedged book's daily exposure to the target-maturity moves. A book w has hedged
P&L  -E @ w.  Also per-rebalance factor diagnostics (explained shares, subspace angles).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import ALL_PANELS, experiment_data
from models import (explained_share, ewma_vol, hedge_matrix, nonparallel, pca,
                    principal_angles)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"

HOLD = 21
WARM = 63
# Hedge instruments by number of factors k. Targets are every other maturity in the panel.
HEDGES = {
    "A": {1: ["10Y"], 2: ["2Y", "10Y"], 3: ["2Y", "5Y", "10Y"]},
    "B": {1: ["10Y"], 2: ["2Y", "10Y"], 3: ["2Y", "10Y", "30Y"], 4: ["2Y", "5Y", "10Y", "30Y"]},
    "G": {1: ["10Y"], 2: ["2Y", "10Y"], 3: ["2Y", "5Y", "10Y"]},
    "E": {1: ["IEF"], 2: ["SHY", "TLT"], 3: ["SHY", "IEF", "TLT"]},
}
METHODS = ["interp", "ns", "pca", "pca_vol", "ols"]


def targets(p: str, cols) -> list[str]:
    allh = set(HEDGES[p][max(HEDGES[p])])
    return [m for m in cols if m not in allh]


def run(p: str, W: int = 504, lam: float = 0.97, verbose: bool = True) -> Path:
    t0 = time.time()
    dyf, mats, lvl = experiment_data(p)
    lv2 = lvl.to_numpy()  # 2Y level at the close before day t
    cols = list(dyf.columns)
    dy = dyf.to_numpy()
    tgt = targets(p, cols)
    iT = [cols.index(c) for c in tgt]
    sig = ewma_vol(dy, lam=lam, warm=WARM)

    first = WARM + W
    rebal = list(range(first, len(dy), HOLD))
    n_oos = len(dy) - first
    meths = METHODS + [f"fm{j}" for j in range(1, len(cols))] + [f"fmv{j}" for j in range(1, len(cols))]
    keys = [(m, k) for k in HEDGES[p] for m in meths] + [("clair", k) for k in HEDGES[p]]
    E = {key: np.zeros((n_oos, len(iT))) for key in keys}
    AM = {key: np.zeros((len(rebal), k, len(iT))) for key in keys for k in [key[1]]}
    kmax = max(HEDGES[p])
    diag = []
    for ir, r in enumerate(rebal):
        win = dy[r - W:r]
        sig_win = sig[r - W:r]
        sig_now = sig[r]           # EWMA known at the close of day r-1
        hold = slice(r, min(r + HOLD, len(dy)))
        o = slice(r - first, min(r + HOLD, len(dy)) - first)
        fut = dy[hold]
        for k, hcols in HEDGES[p].items():
            iH = [cols.index(c) for c in hcols]
            for m in meths:
                A = hedge_matrix(m, k, win, sig_win, sig_now, mats, iT, iH)
                E[(m, k)][o] = fut[:, iT] - fut[:, iH] @ A
                AM[(m, k)][ir] = A
            A = np.linalg.lstsq(fut[:, iH], fut[:, iT], rcond=None)[0]
            E[("clair", k)][o] = fut[:, iT] - fut[:, iH] @ A
            AM[("clair", k)][ir] = A
        # factor diagnostics: trailing window vs the next 21 and 252 days
        Lw, vals = pca(np.cov(win, rowvar=False), kmax)
        nxt = dy[r:min(r + 252, len(dy))]
        row = {"date": dyf.index[r], "ev1": vals[0] / vals.sum()}
        if len(nxt) >= 126:
            Ln, _ = pca(np.cov(nxt, rowvar=False), kmax)
        for k in range(1, kmax + 1):
            row[f"ins_share_k{k}"] = explained_share(win - win.mean(0), Lw[:, :k])
            row[f"oos21_share_k{k}"] = explained_share(fut, Lw[:, :k])
            q, _ = np.linalg.qr(Lw[:, :k])
            res_in = win - win @ q @ q.T
            res_out = fut - fut @ q @ q.T
            row[f"ins_curve_left_k{k}"] = np.sum(res_in ** 2) / np.sum(nonparallel(win) ** 2)
            row[f"oos21_curve_left_k{k}"] = np.sum(res_out ** 2) / np.sum(nonparallel(fut) ** 2)
            if len(nxt) >= 126:
                row[f"angle_max_k{k}"] = principal_angles(Lw[:, :k], Ln[:, :k]).max()
        if len(nxt) >= 126:
            for j in range(3):  # angle between individual PCs (sign-free)
                c = abs(Lw[:, j] @ Ln[:, j])
                row[f"angle_pc{j + 1}"] = np.degrees(np.arccos(min(c, 1.0)))
        row["level_2y"] = lv2[r]
        diag.append(row)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{p}_W{W}_lam{lam}.npz"
    np.savez_compressed(out, dates=dyf.index[first:].to_numpy().astype("datetime64[D]"),
                        targets=np.array(tgt), dyT=dy[first:, iT],
                        rebal=np.array(rebal) - first, hedges=np.array([",".join(HEDGES[p][k]) for k in HEDGES[p]]),
                        **{f"{m}_k{k}": v for (m, k), v in E.items()},
                        **{f"A_{m}_k{k}": v for (m, k), v in AM.items()})
    pd.DataFrame(diag).to_csv(CACHE / f"{p}_W{W}_lam{lam}_diag.csv", index=False)
    if verbose:
        print(f"panel {p} W={W} lam={lam}: {n_oos} OOS days, {len(rebal)} rebalances, "
              f"{time.time() - t0:.1f}s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robust", action="store_true", help="also run the window and EWMA grids")
    a = ap.parse_args()
    for p in ALL_PANELS:
        run(p)
    if a.robust:
        for p in ("A", "B"):
            for W in (126, 252, 1260):
                run(p, W=W)
            for lam in (0.94, 0.99):
                run(p, lam=lam)
