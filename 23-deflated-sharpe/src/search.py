"""Run the zoo's in-sample search on one world (real or sign-flipped) and record what DSR needs.

A 'search' is the set of trials a researcher compares at one split: all eligible zoo-A trials
('A-pooled'), the 114 trials of one index ('A-GSPC', ...), or all 176 industry trials ('B').
At every split the in-sample (IS) window is the 10 calendar years ending in December of Y and the
out-of-sample (OOS) window is the following 5 years (the last one runs to the end of the data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import dsr as D
from data import INDICES, load_indices, load_industries
from zoo import zoo_a, zoo_b

SPLIT_YEARS = list(range(1940, 2021, 5))
IS_YEARS, OOS_YEARS = 10, 5
PLANT_SR = [0.5, 1.0, 1.5]  # annualised true Sharpe of the planted strategy (power analysis)

FIELDS = ["T", "N", "sr", "skew", "kurt", "v_xs", "ny", "lj", "pr", "k", "v_clu", "sr_med"]


def split_rows(index: pd.DatetimeIndex, Y: int):
    is_m = (index.year >= Y - IS_YEARS + 1) & (index.year <= Y)
    oos_m = (index.year >= Y + 1) & (index.year <= Y + OOS_YEARS + (6 if Y == SPLIT_YEARS[-1] else 0))
    return np.flatnonzero(is_m), np.flatnonzero(oos_m)


def search_stats(R: np.ndarray, do_cluster: bool):
    """Statistics of one search on its IS return matrix R (T x N)."""
    T, N = R.shape
    m, sd, sk, ku = D.moments(R)
    sr = np.where(sd > 0, m / np.where(sd > 0, sd, 1), 0.0)
    w = int(np.argmax(sr))
    out = dict(T=T, N=N, sr=sr[w], skew=sk[w], kurt=ku[w], v_xs=np.var(sr, ddof=1),
               sr_med=np.median(sr), win=w)
    lam, C = D.corr_eigs(R)
    out.update(ny=D.neff_nyholt(lam), lj=D.neff_liji(lam), pr=D.neff_participation(lam))
    if do_cluster:
        k, lab = D.cluster_trials(C, kmax=min(60, N - 1))
        csr = D.cluster_sharpes(R, lab)
        out.update(k=k, v_clu=np.var(csr, ddof=1) if k > 1 else np.nan)
    else:
        out.update(k=np.nan, v_clu=np.nan)
    return out, sr


def planted_stats(R, base, sr_all, s_ann, ann):
    """Add one strategy with true annual Sharpe s_ann (base series + drift) and redo the selection."""
    sig = base.std(ddof=1)
    p = base + s_ann / np.sqrt(ann) * sig
    m, sd, sk, ku = D.moments(p[:, None])
    srp = float(m[0] / sd[0]) if np.ndim(m) else float(m / sd)
    sr2 = np.append(sr_all, srp)
    w = int(np.argmax(sr2))
    if w == sr2.size - 1:
        skw, kuw = float(np.squeeze(sk)), float(np.squeeze(ku))
    else:
        _, _, s_, k_ = D.moments(R[:, [w]])
        skw, kuw = float(s_[0]), float(k_[0])
    return dict(sr=sr2[w], skew=skw, kurt=kuw, v_xs=np.var(sr2, ddof=1), won=float(w == sr2.size - 1),
                srp=srp)


class World:
    """Holds the real panels once; builds zoo returns for a given sign-flip draw."""

    def __init__(self):
        self.XA = load_indices()
        self.XB = load_industries()
        self.dA = self.XA.index
        self.dB = self.XB.index
        self.ann_a = len(self.XA) / ((self.dA[-1] - self.dA[0]).days / 365.25)

    def flipped(self, seed: int | None):
        XA, XB = self.XA.copy(), self.XB.copy()
        if seed is not None:
            rng = np.random.default_rng(seed)
            sa = rng.choice([-1.0, 1.0], size=len(XA))
            sb = rng.choice([-1.0, 1.0], size=len(XB))
            XA = XA.mul(sa, axis=0)   # one common sign per day across all indices
            XB = XB.mul(sb, axis=0)   # one common sign per month across all industries
        return XA, XB


def run_world(world: World, seed: int | None, do_cluster: bool, with_oos: bool = False):
    """All searches at all splits for one world. Returns a list of row dicts."""
    XA, XB = world.flipped(seed)
    gA, nA, eA, oA = zoo_a(XA)
    gB, nB, eB = zoo_b(XB.to_numpy())
    eqB = XB.mean(1).fillna(0).to_numpy()
    gspc = XA["GSPC"].fillna(0).to_numpy()
    rows = []
    variants = [("gross", gA, gB)] + ([("net", nA, nB)] if with_oos else [])
    for basis, RA, RB in variants:
        for Y in SPLIT_YEARS:
            ia, oa = split_rows(world.dA, Y)
            ib, ob = split_rows(world.dB, Y)
            searches = []
            okA = eA <= ia[0]
            searches.append(("A-pooled", RA, ia, oa, np.flatnonzero(okA), gspc, world.ann_a))
            for j, c in enumerate(INDICES):
                ids = np.flatnonzero(okA & (oA == j))
                if ids.size:
                    searches.append((f"A-{c}", RA, ia, oa, ids, None, world.ann_a))
            idsB = np.flatnonzero(eB <= ib[0]) if ib.size else np.array([], int)
            if idsB.size:
                searches.append(("B", RB, ib, ob, idsB, eqB, 12.0))
            for name, R, i_is, i_oos, ids, base, ann in searches:
                Ris = R[i_is][:, ids].astype(np.float64)
                st, sr_all = search_stats(Ris, do_cluster and (name in ("A-pooled", "B") or name.startswith("A-")))
                row = dict(seed=-1 if seed is None else seed, basis=basis, split=Y, search=name, ann=ann, **st)
                row["win_id"] = int(ids[st["win"]])
                if base is not None and seed is not None:
                    for s in PLANT_SR:
                        ps = planted_stats(Ris, base[i_is], sr_all, s, ann)
                        for k_, v_ in ps.items():
                            row[f"p{s}_{k_}"] = v_
                if with_oos and i_oos.size:
                    Ro = R[i_oos][:, ids].astype(np.float64)
                    mo, so, _, _ = D.moments(Ro)
                    sro = np.where(so > 0, mo / np.where(so > 0, so, 1), 0.0)
                    row.update(oos_T=i_oos.size, oos_sr_win=sro[st["win"]], oos_sr_med=np.median(sro),
                               oos_sr_top10=np.mean(sro[sr_all >= np.quantile(sr_all, 0.9)]),
                               oos_rank_corr=pd.Series(sr_all).corr(pd.Series(sro), method="spearman"),
                               oos_win_pct=float(np.mean(sro <= sro[st["win"]])))
                rows.append(row)
    return rows
