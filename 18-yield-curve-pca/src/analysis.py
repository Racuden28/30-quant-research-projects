"""Turn the walk-forward residual designs into every table in the README.

Loss definition (primary). Each panel gets 500 random 'outright' books and 500 random 'curve'
books (DV01-neutral: exposures sum to zero), unit norm, seed 18. For a residual design row e_t the
mean squared hedged P&L across a family of books is e_t Q e_t' with Q = W W' / 500. That daily loss
is divided by the trailing 252-day mean of the *unhedged* loss, which is known ex ante, so that
1979-82 and the zero-lower-bound years carry comparable weight. Ratios of mean losses between two
methods get a moving-block bootstrap CI (63-day blocks) and a HAC Diebold-Mariano test; the
pre-registered comparisons are Holm-adjusted together.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import CACHE, HEDGES
from data import PANEL_DESC, PANELS, changes, experiment_data, gsw, panel
from models import explained_share, nonparallel, pca
from scipy.stats import binomtest

from stats import dm_test, holm, ratio_ci

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
NB = 500
COST_BP = 0.25  # one-way cost per unit of DV01 traded, in bp of yield (robustness only)

STRESS = [
    ("Volcker disinflation", "1979-10-06", "1982-10-31"),
    ("1987 crash", "1987-10-01", "1987-11-30"),
    ("1994 bond massacre", "1994-02-01", "1994-12-31"),
    ("1998 LTCM", "1998-08-01", "1998-11-30"),
    ("GFC", "2008-09-01", "2009-03-31"),
    ("Taper tantrum", "2013-05-01", "2013-09-30"),
    ("COVID dash for cash", "2020-02-20", "2020-04-30"),
    ("2022 hiking cycle", "2022-03-16", "2022-12-31"),
    ("SVB, March 2023", "2023-03-08", "2023-03-31"),
    ("2023 term-premium sell-off", "2023-08-01", "2023-10-31"),
]
REGIMES = [("Zero lower bound", [("2008-12-16", "2015-12-16"), ("2020-03-16", "2022-03-16")])]


def load(p: str, W: int = 504, lam: float = 0.97):
    z = np.load(CACHE / f"{p}_W{W}_lam{lam}.npz")
    return {k: z[k] for k in z.files}


def books(nT: int):
    rng = np.random.default_rng(18)
    out = rng.standard_normal((nT, NB))
    cur = rng.standard_normal((nT, NB))
    cur -= cur.mean(0)
    return {"outright": out / np.linalg.norm(out, axis=0), "curve": cur / np.linalg.norm(cur, axis=0)}


def qloss(E: np.ndarray, B: np.ndarray) -> np.ndarray:
    Q = B @ B.T / B.shape[1]
    return np.einsum("ti,ij,tj->t", E, Q, E)


def norm_losses(z, B, keys):
    """Daily losses divided by the ex-ante trailing unhedged loss. Returns (dates, dict, scale)."""
    un = qloss(z["dyT"], B)
    scale = pd.Series(un).rolling(252, min_periods=63).mean().shift(1).to_numpy()
    ok = np.isfinite(scale)
    out = {k: qloss(z[k], B)[ok] / scale[ok] for k in keys}
    out["unhedged"] = un[ok] / scale[ok]
    raw = {k: qloss(z[k], B)[ok] for k in keys}
    raw["unhedged"] = un[ok]
    return pd.to_datetime(z["dates"][ok]), out, raw


def method_keys(z, K):
    return [k for k in z if k.endswith(f"_k{K}") and not k.startswith("A_")]


def base_for(p):
    return "ols" if p == "E" else "interp"


def gross_turnover(z, key, B):
    """Mean gross hedge DV01 per unit of book DV01, and mean one-way turnover per rebalance."""
    A = z[f"A_{key}"]  # rebal x |H| x |T|
    H = np.einsum("rht,tb->rhb", A, B)
    gross = np.abs(H).sum(1).mean() / np.abs(B).sum(0).mean()
    turn = np.abs(np.diff(H, axis=0)).sum(1).mean() / np.abs(B).sum(0).mean()
    return float(gross), float(turn)


# ---------------------------------------------------------------- 1. where the 99% comes from
def variance_table() -> pd.DataFrame:
    rows = []
    sets = {"A": panel("A"), "B": panel("B")}
    g = gsw([1, 2, 3, 5, 7, 10]).loc["1976-06-01":].dropna()
    g.columns = PANELS["A"]["mats"]
    sets["G"] = g
    for p, lv in sets.items():
        for label, X in [("levels", lv.to_numpy() * 100),
                         ("monthly changes (21d)", (lv.iloc[::21].diff().iloc[1:] * 100).to_numpy()),
                         ("weekly changes (5d)", (lv.iloc[::5].diff().iloc[1:] * 100).to_numpy()),
                         ("daily changes", changes(lv).to_numpy())]:
            Xc = X - X.mean(0)
            L, vals = pca(np.cov(Xc, rowvar=False), 3)
            row = {"panel": p, "desc": PANEL_DESC[p], "data": label, "n": len(X)}
            for k in (1, 2, 3):
                row[f"share_k{k}"] = vals[:k].sum() / vals.sum()
            q, _ = np.linalg.qr(L)
            res = Xc - Xc @ q @ q.T
            row["curve_left_k3"] = np.sum(res ** 2) / np.sum(nonparallel(Xc) ** 2)
            row["eig4_to_eig3"] = vals[3] / vals[2]
            row["eigs"] = " ".join(f"{v / vals.sum():.2e}" for v in vals)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. horse race
def horse_race(p: str, W: int = 504, lam: float = 0.97, with_ci: bool = True) -> pd.DataFrame:
    z = load(p, W, lam)
    BB = books(len(z["targets"]))
    rows = []
    for K in HEDGES[p]:
        keys = method_keys(z, K)
        base = f"{base_for(p)}_k{K}"
        for fam, B in BB.items():
            d, L, raw = norm_losses(z, B, keys)
            for k in keys:
                m = k.rsplit("_k", 1)[0]
                if p == "E" and m in ("interp", "ns"):
                    continue
                r, lo, hi = ratio_ci(L[k], L[base]) if with_ci else (L[k].mean() / L[base].mean(), np.nan, np.nan)
                t, pv = dm_test(L[k], L[base]) if k != base else (0.0, 1.0)
                # month-by-month: ratio of holding-period mean losses
                mon = pd.DataFrame({"a": L[k], "b": L[base]}, index=d).groupby(d.to_period("M")).mean()
                gr, tu = gross_turnover(z, k, B) if m != "clair" else (np.nan, np.nan)
                rows.append({"panel": p, "K": K, "hedges": HEDGES[p][K], "family": fam, "method": m,
                             "ratio_vs_base": r, "ci_lo": lo, "ci_hi": hi, "dm_t": t, "dm_p": pv,
                             "raw_ratio_vs_base": raw[k].mean() / raw[base].mean(),
                             "hedge_eff": 1 - L[k].mean() / L["unhedged"].mean(),
                             "median_month_ratio": float(np.median(mon.a / mon.b)),
                             "share_months_better": float((mon.a < mon.b).mean()),
                             "gross_hedge": gr, "turnover": tu, "base": base_for(p), "W": W, "lam": lam})
    return pd.DataFrame(rows)


PLANNED = [  # (method a, method b): pre-registered pairwise tests, Holm-adjusted together
    ("pca", "interp"), ("ols", "interp"), ("fm3", "ols"), ("fmv3", "fm3"), ("pca_vol", "pca"),
    ("ns", "interp"),
]


def planned_tests(panels=("A", "B", "E")) -> pd.DataFrame:
    rows = []
    for p in panels:
        z = load(p)
        BB = books(len(z["targets"]))
        for K in HEDGES[p]:
            keys = method_keys(z, K)
            for fam, B in BB.items():
                d, L, _ = norm_losses(z, B, keys)
                for a, b in PLANNED:
                    ka, kb = f"{a}_k{K}", f"{b}_k{K}"
                    if ka not in L or kb not in L or (p == "E" and {"interp", "ns"} & {a, b}):
                        continue
                    r, lo, hi = ratio_ci(L[ka], L[kb])
                    t, pv = dm_test(L[ka], L[kb])
                    mon = pd.DataFrame({"a": L[ka], "b": L[kb]}, index=d).groupby(d.to_period("M")).mean()
                    nwin = int((mon.a < mon.b).sum())
                    sp = binomtest(nwin, len(mon)).pvalue
                    rows.append({"panel": p, "K": K, "family": fam, "a": a, "b": b, "ratio": r,
                                 "ci_lo": lo, "ci_hi": hi, "dm_t": t, "p": pv,
                                 "median_month_ratio": float(np.median(mon.a / mon.b)),
                                 "share_months_a_better": nwin / len(mon), "sign_p": sp})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p.to_numpy())
    df["sign_p_holm"] = holm(df.sign_p.to_numpy())
    return df


# ---------------------------------------------------------------- 3. how many factors to keep
def factor_count_curve(panels=("A", "B", "G", "E")) -> pd.DataFrame:
    rows = []
    for p in panels:
        z = load(p)
        BB = books(len(z["targets"]))
        n = len([k for k in z if re.fullmatch(r"fm\d+_k1", k)]) + 1
        for K in HEDGES[p]:
            keys = method_keys(z, K)
            for fam, B in BB.items():
                _, L, _ = norm_losses(z, B, keys)
                ref = L[f"ols_k{K}"].mean()
                for j in range(1, n):
                    for pre in ("fm", "fmv"):
                        rows.append({"panel": p, "K": K, "family": fam, "model": pre, "j": j,
                                     "ratio_vs_ols": L[f"{pre}{j}_k{K}"].mean() / ref})
                rows.append({"panel": p, "K": K, "family": fam, "model": "fm", "j": n, "ratio_vs_ols": 1.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. stress windows
def stress_table(panels=("A", "B", "E")) -> pd.DataFrame:
    rows = []
    show = ["interp", "ns", "pca", "pca_vol", "fm3", "fmv3", "ols", "clair"]
    for p in panels:
        z = load(p)
        BB = books(len(z["targets"]))
        for K in HEDGES[p]:
            keys = method_keys(z, K)
            for fam, B in BB.items():
                d, L, raw = norm_losses(z, B, keys)
                wins = [(n, [(a, b)]) for n, a, b in STRESS] + REGIMES + \
                       [("Outside the ZLB", [("1900-01-01", "2008-12-15"), ("2015-12-17", "2020-03-15"),
                                             ("2022-03-17", "2100-01-01")])]
                for name, spans in wins:
                    msk = np.zeros(len(d), bool)
                    for a, b in spans:
                        msk |= (d >= a) & (d <= b)
                    if msk.sum() < 10:
                        continue
                    row = {"panel": p, "K": K, "family": fam, "window": name, "days": int(msk.sum())}
                    un = raw["unhedged"][msk].mean()
                    for m in show:
                        k = f"{m}_k{K}"
                        if k in raw and not (p == "E" and m in ("interp", "ns")):
                            row[f"he_{m}"] = 1 - raw[k][msk].mean() / un
                    rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5. rotation and conditioning
def rotation_table(panels=("A", "B", "G")) -> pd.DataFrame:
    """Per rebalance: subspace angles, PCA-neutral hedge size, and that month's loss vs interp."""
    out = []
    for p in panels:
        z = load(p)
        diag = pd.read_csv(CACHE / f"{p}_W504_lam0.97_diag.csv", parse_dates=["date"])
        B = books(len(z["targets"]))["curve"]
        K = 3
        H = np.einsum("rht,tb->rhb", z[f"A_pca_k{K}"], B)
        diag["pca_gross_k3"] = np.abs(H).sum(1).mean(1) / np.abs(B).sum(0).mean()
        Hi = np.einsum("rht,tb->rhb", z[f"A_interp_k{K}"], B)
        diag["interp_gross_k3"] = np.abs(Hi).sum(1).mean(1) / np.abs(B).sum(0).mean()
        la, lb = qloss(z[f"pca_k{K}"], B), qloss(z[f"interp_k{K}"], B)
        reb = list(z["rebal"]) + [len(la)]
        diag["pca_vs_interp_k3"] = [la[a:b].mean() / lb[a:b].mean() for a, b in zip(reb[:-1], reb[1:])]
        diag["panel"] = p
        out.append(diag)
    return pd.concat(out, ignore_index=True)


def conditioning_rule(rot: pd.DataFrame) -> pd.DataFrame:
    """Does a simple ex-ante check (gross PCA hedge <= 2x the book's DV01) rescue the textbook hedge?"""
    rows = []
    for p, g in rot.groupby("panel"):
        for name, m in [("all months", np.ones(len(g), bool)), ("gross <= 2x", g.pca_gross_k3 <= 2),
                        ("gross > 2x", g.pca_gross_k3 > 2)]:
            x = g.pca_vs_interp_k3[m]
            rows.append({"panel": p, "months": name, "share_of_months": float(m.mean()),
                         "median_ratio_pca_vs_interp": float(np.median(x)) if len(x) else np.nan,
                         "share_pca_better": float((x < 1).mean()) if len(x) else np.nan})
    return pd.DataFrame(rows)


def named_trades() -> pd.DataFrame:
    trades = {"A": {"7Y bullet": {"7Y": 1}, "3s7s steepener": {"3Y": -1, "7Y": 1},
                    "1s3s7s butterfly": {"1Y": -0.5, "3Y": 1, "7Y": -0.5}},
              "B": {"20Y bullet": {"20Y": 1}, "7s20s steepener": {"7Y": -1, "20Y": 1},
                    "3s7s20s butterfly": {"3Y": -0.5, "7Y": 1, "20Y": -0.5}}}
    rows = []
    for p, tr in trades.items():
        z = load(p)
        tg = list(z["targets"])
        K = max(HEDGES[p])
        for name, wts in tr.items():
            w = np.zeros((len(tg), 1))
            for m, v in wts.items():
                w[tg.index(m), 0] = v
            un = ((z["dyT"] @ w) ** 2).mean()
            row = {"panel": p, "K": K, "trade": name, "unhedged_vol_bp": float(np.sqrt(un))}
            for m in ["interp", "ns", "pca", "pca_vol", "fm3", "fmv3", "ols", "clair"]:
                v = ((z[f"{m}_k{K}"] @ w) ** 2).mean()
                row[f"vol_{m}"] = float(np.sqrt(v))
            rows.append(row)
    return pd.DataFrame(rows)


def robustness() -> pd.DataFrame:
    rows = []
    for p in ("A", "B"):
        for W, lam in [(126, 0.97), (252, 0.97), (504, 0.97), (1260, 0.97), (504, 0.94), (504, 0.99)]:
            if not (CACHE / f"{p}_W{W}_lam{lam}.npz").exists():
                continue
            hr = horse_race(p, W, lam, with_ci=False)
            rows.append(hr[hr.method.isin(["pca", "pca_vol", "fm3", "fmv3", "ols", "ns"])])
    return pd.concat(rows, ignore_index=True)


def cost_check() -> pd.DataFrame:
    """Add a 0.25 bp one-way cost per unit of DV01 traded at each rebalance to the hedged P&L."""
    rows = []
    for p in ("A", "B"):
        z = load(p)
        B = books(len(z["targets"]))["curve"]
        K = max(HEDGES[p])
        reb = z["rebal"]
        for m in ["interp", "pca", "fm3", "fmv3", "ols"]:
            k = f"{m}_k{K}"
            H = np.einsum("rht,tb->rhb", z[f"A_{k}"], B)
            cost = np.zeros((len(z["dyT"]), B.shape[1]))
            dH = np.abs(np.diff(H, axis=0, prepend=np.zeros_like(H[:1]))).sum(1)
            cost[reb] = COST_BP * dH
            pnl = -(z[k] @ B) - cost
            rows.append({"panel": p, "K": K, "method": m, "rms_no_cost": float(np.sqrt(((z[k] @ B) ** 2).mean())),
                         "rms_with_cost": float(np.sqrt((pnl ** 2).mean())),
                         "mean_cost_per_day": float(cost.mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    vt = variance_table()
    vt.to_csv(RES / "variance_explained.csv", index=False)
    print(vt[["panel", "data", "share_k1", "share_k3", "curve_left_k3", "eig4_to_eig3"]].round(4).to_string())
    hr = pd.concat([horse_race(p) for p in ("A", "B", "G", "E")], ignore_index=True)
    hr.to_csv(RES / "horse_race.csv", index=False)
    pt = planned_tests()
    pt.to_csv(RES / "planned_tests.csv", index=False)
    print(pt.round(4).to_string())
    factor_count_curve().to_csv(RES / "factor_count.csv", index=False)
    stress_table().to_csv(RES / "stress_windows.csv", index=False)
    rot = rotation_table()
    rot.to_csv(RES / "rotation.csv", index=False)
    conditioning_rule(rot).to_csv(RES / "conditioning_rule.csv", index=False)
    named_trades().to_csv(RES / "named_trades.csv", index=False)
    robustness().to_csv(RES / "robustness.csv", index=False)
    cost_check().to_csv(RES / "costs.csv", index=False)
    print("analysis done")


def without_1y() -> pd.DataFrame:
    """Robustness: books with no 1Y exposure (the 1Y CMT point is a bill and may behave differently)."""
    rows = []
    for p in ("A", "B"):
        z = load(p)
        tg = list(z["targets"])
        BB = books(len(tg))
        K = max(HEDGES[p])
        for fam, B in BB.items():
            B = B.copy()
            B[tg.index("1Y")] = 0
            if fam == "curve":
                keep = [i for i in range(len(tg)) if tg[i] != "1Y"]
                B[keep] -= B[keep].mean(0)
            B /= np.linalg.norm(B, axis=0)
            keys = method_keys(z, K)
            d, L, _ = norm_losses(z, B, keys)
            mon = lambda k: pd.Series(L[k], index=d).groupby(d.to_period("M")).mean()  # noqa: E731
            for m in ["ns", "pca", "pca_vol", "fm3", "fmv3", "ols", "clair"]:
                k = f"{m}_k{K}"
                rows.append({"panel": p, "K": K, "family": fam, "method": m,
                             "ratio_vs_interp": L[k].mean() / L[f"interp_k{K}"].mean(),
                             "median_month_ratio": float(np.median(mon(k) / mon(f"interp_k{K}")))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    without_1y().to_csv(RES / "robustness_without_1y.csv", index=False)


def ridge_neutral_check(alphas=(0.001, 0.01, 0.1, 1.0)) -> pd.DataFrame:
    """Referee check: is the PCA-neutral blow-up just exact identification?

    Replaces the exact solve L_H' h = -L_T' w with a ridge solve
    h = -(L_H L_H' + alpha I)^-1 L_H L_T' w (loadings are unit vectors, so alpha is unit-free),
    and separately caps the gross hedge at 2x the book's DV01 by scaling it down. Same windows,
    same holding periods, curve books, max K.
    """
    from backtest import HOLD, WARM
    rows = []
    for p in ("A", "B"):
        z = load(p)
        dyf, mats, _ = experiment_data(p)
        cols = list(dyf.columns)
        dy = dyf.to_numpy()
        tg = list(z["targets"])
        iT = [cols.index(c) for c in tg]
        K = max(HEDGES[p])
        iH = [cols.index(c) for c in HEDGES[p][K]]
        B = books(len(tg))["curve"]
        first = WARM + 504
        n = len(dy) - first
        E = {a: np.zeros((n, len(iT))) for a in alphas}
        capped = np.zeros((n, B.shape[1]))
        for r in range(first, len(dy), HOLD):
            L, _ = pca(np.cov(dy[r - 504:r], rowvar=False), K)
            LH, LT = L[iH], L[iT]
            fut = dy[r:min(r + HOLD, len(dy))]
            o = slice(r - first, r - first + len(fut))
            for a in alphas:
                A = np.linalg.solve(LH @ LH.T + a * np.eye(K), LH @ LT.T)
                E[a][o] = fut[:, iT] - fut[:, iH] @ A
            A0 = np.linalg.solve(LH.T, LT.T)
            H = A0 @ B
            scale = np.minimum(1, 2 * np.abs(B).sum(0) / np.maximum(np.abs(H).sum(0), 1e-12))
            capped[o] = -(fut[:, iT] @ B - fut[:, iH] @ (H * scale))
        keys = {f"ridge_{a}": E[a] for a in alphas}
        zz = dict(z, **keys)
        d, Lz, _ = norm_losses(zz, B, [f"interp_k{K}", f"pca_k{K}", f"ols_k{K}"] + list(keys))
        un = qloss(z["dyT"], B)
        scale = pd.Series(un).rolling(252, min_periods=63).mean().shift(1).to_numpy()
        ok = np.isfinite(scale)
        Lz["cap2x"] = (capped ** 2).mean(1)[ok] / scale[ok]
        base = Lz[f"interp_k{K}"]
        for k in [f"pca_k{K}"] + list(keys) + ["cap2x", f"ols_k{K}"]:
            mon = pd.DataFrame({"a": Lz[k], "b": base}, index=d).groupby(d.to_period("M")).mean()
            rows.append({"panel": p, "K": K, "variant": k, "ratio_vs_interp": Lz[k].mean() / base.mean(),
                         "median_month_ratio": float(np.median(mon.a / mon.b)),
                         "share_months_better": float((mon.a < mon.b).mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ridge_neutral_check().to_csv(RES / "ridge_neutral_check.csv", index=False)
