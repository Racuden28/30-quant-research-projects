"""How long does a variance shock last? Local projections that test each model on its own terms.

For each origin t and lag k = 1..252, regress the realized squared return r^2_{t+k} on the model's own
one-day-ahead variance s_t (the variance it would quote for tomorrow), with an intercept:

    r^2_{t+k} = c_k + b_k * s_t + e         (empirical, Jorda 2005 local projection)
    F_t(k)    = c'_k + m_k * s_t + e'       (the same regression on the model's own k-step forecast)

If the model's dynamics are right, b_k = m_k at every k. b_k / b_1 is the share of today's variance news
still visible k days later; its 0.5-crossing is the empirical half-life of a shock. m_k / m_1 is the
half-life the model believes. Using the model's own forecast on the same regressor makes the
comparison fair even though parameters are re-estimated every year.

All series are scaled by each index's mean r^2 (slopes are scale-free anyway). Confidence bands:
calendar-year block bootstrap, resampling the same years for every index (keeps the 2008-type
cross-market dependence). Output: results/irf_curves.csv.gz, results/half_lives.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import INDICES, ROOT  # noqa: E402
from models import curve_half_life  # noqa: E402

K = np.arange(1, 253)
NBOOT = 1000
SEED = 13
RES = ROOT / "results"
SPECS = ["GARCH", "GARCH-R10", "GJR-t", "CGARCH"]


def model_path(d: pd.DataFrame, spec: str, k: int) -> np.ndarray:
    """Model's E_t[r^2_{t+k}] for every origin, using that origin's parameters."""
    if spec in ("GARCH", "GARCH-R10", "GJR-t"):
        p = {"GARCH": "GARCH", "GARCH-R10": "R10", "GJR-t": "GJR"}[spec]
        s, vb, ph = d[f"{p}_s"].values, d[f"{p}_vbar"].values, d[f"{p}_phi"].values
        w = vb * (1 - ph)
        return w * (1 - ph ** (k - 1)) / (1 - ph) + ph ** (k - 1) * s
    if spec == "CGARCH":
        s, q, w, rho, psi = (d[c].values for c in ["CG_s", "CG_q", "CG_w", "CG_rho", "CG_psi"])
        return w + rho ** (k - 1) * (q - w) + psi ** (k - 1) * (s - q)
    raise KeyError(spec)


def state(d, spec):
    return d[{"GARCH": "GARCH_s", "GARCH-R10": "R10_s", "GJR-t": "GJR_s", "CGARCH": "CG_s"}[spec]].values


def block_stats(x, y, yr, nyr):
    """Per-year sufficient statistics for an OLS slope with intercept."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, yr = x[ok], y[ok], yr[ok]
    f = lambda v: np.bincount(yr, weights=v, minlength=nyr)  # noqa: E731
    return np.stack([f(np.ones_like(x)), f(x), f(y), f(x * x), f(x * y)])  # (5, nyr)


def slope(S):
    n, sx, sy, sxx, sxy = S
    return (sxy - sx * sy / n) / (sxx - sx * sx / n)


def compute(winsor: float | None = None, drop_1987: bool = False, tag: str = "main", period=None, specs=SPECS):
    data = {}
    for c in INDICES:
        d = pd.read_pickle(RES / "forecasts" / f"{c}.pkl.gz")
        data[c] = d
    years = sorted({y for d in data.values() for y in d.index.year})
    ymap = {y: i for i, y in enumerate(years)}
    nyr = len(years)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, nyr, size=(NBOOT, nyr))
    counts = np.stack([np.bincount(dr, minlength=nyr) for dr in draws])  # (B, nyr)

    rows, hl_rows = [], []
    for spec in specs:
        # stats[i][kind] -> array (len(K), 5, nyr)
        pooled = {"emp": np.zeros((len(K), 5, nyr)), "mod": np.zeros((len(K), 5, nyr))}
        per_index_cov = {"emp": [], "mod": []}
        for c, d in data.items():
            r2 = d["r"].values ** 2
            scale = np.nanmean(r2)
            if winsor:
                r2 = np.minimum(r2, np.nanquantile(r2, winsor))
            x = state(d, spec) / scale
            yr = np.array([ymap[y] for y in d.index.year])
            keep = np.ones(len(x), bool)
            if drop_1987:
                keep &= ~((d.index >= "1987-10-01") & (d.index <= "1988-03-31"))
            if period:
                keep &= (d.index >= period[0]) & (d.index < period[1])
            if keep.sum() < 750:
                continue
            n = len(x)
            st = {"emp": np.zeros((len(K), 5, nyr)), "mod": np.zeros((len(K), 5, nyr))}
            for j, k in enumerate(K):
                y_emp = np.full(n, np.nan)
                y_emp[: n - k] = r2[k:] / scale
                y_mod = model_path(d, spec, k) / scale
                y_mod[n - k:] = np.nan
                st["emp"][j] = block_stats(np.where(keep, x, np.nan), y_emp, yr, nyr)
                st["mod"][j] = block_stats(np.where(keep, x, np.nan), y_mod, yr, nyr)
            for kind in ("emp", "mod"):
                b = np.array([slope(st[kind][j].sum(axis=1)) for j in range(len(K))])
                bb = np.stack([slope(np.einsum("sy,by->sb", st[kind][j], counts)) for j in range(len(K))])  # (K,B)
                per_index_cov[kind].append(st[kind])
                rows.append(pd.DataFrame(dict(tag=tag, spec=spec, index=c, kind=kind, k=K, b=b, norm=b / b[0],
                                              lo=np.nanquantile(bb / bb[:1], 0.05, axis=1),
                                              hi=np.nanquantile(bb / bb[:1], 0.95, axis=1))))
                hl_b = np.array([curve_half_life(bb[:, i] / bb[0, i], K) for i in range(NBOOT)])
                hl_rows.append(dict(tag=tag, spec=spec, index=c, kind=kind, b1=b[0],
                                    half_life=curve_half_life(b / b[0], K),
                                    hl_lo=np.nanquantile(hl_b, 0.05), hl_hi=np.nanquantile(hl_b, 0.95),
                                    share_left_22=b[21] / b[0], share_left_126=b[125] / b[0]))
        # pooled with index fixed effects: sum per-index centered moments
        for kind in ("emp", "mod"):
            def pooled_slope(weights):
                num = den = 0.0
                for S in per_index_cov[kind]:
                    T = np.einsum("ksy,y->ks", S, weights) if weights.ndim == 1 else np.einsum("ksy,by->bks", S, weights)
                    n, sx, sy, sxx, sxy = np.moveaxis(T, -1, 0)
                    num = num + (sxy - sx * sy / n)
                    den = den + (sxx - sx * sx / n)
                return num / den
            b = pooled_slope(np.ones(nyr))
            bb = pooled_slope(counts.astype(float))  # (B, K)
            nb = bb / bb[:, :1]
            rows.append(pd.DataFrame(dict(tag=tag, spec=spec, index="POOLED", kind=kind, k=K, b=b, norm=b / b[0],
                                          lo=np.nanquantile(nb, 0.05, axis=0), hi=np.nanquantile(nb, 0.95, axis=0))))
            hl_b = np.array([curve_half_life(nb[i], K) for i in range(NBOOT)])
            hl_rows.append(dict(tag=tag, spec=spec, index="POOLED", kind=kind, b1=b[0],
                                half_life=curve_half_life(b / b[0], K),
                                hl_lo=np.nanquantile(hl_b, 0.05), hl_hi=np.nanquantile(hl_b, 0.95),
                                share_left_22=b[21] / b[0], share_left_126=b[125] / b[0]))
            if kind == "emp":
                emp_bb = nb
            else:
                # bootstrap distribution of the gap in half-lives, same resampled years for both
                gap = np.array([curve_half_life(nb[i], K) for i in range(NBOOT)]) - \
                    np.array([curve_half_life(emp_bb[i], K) for i in range(NBOOT)])
                hl_rows.append(dict(tag=tag, spec=spec, index="POOLED", kind="gap_mod_minus_emp", b1=np.nan,
                                    half_life=np.nanmedian(gap), hl_lo=np.nanquantile(gap, 0.05),
                                    hl_hi=np.nanquantile(gap, 0.95), share_left_22=np.mean(gap <= 0),
                                    share_left_126=np.nan))
        print(f"[{tag}] {spec} done", flush=True)
    return pd.concat(rows), pd.DataFrame(hl_rows)


def main():
    curves, hls = [], []
    eras = [dict(tag=f"era{a[:4]}-{b[:4]}", period=(a, b), specs=["GARCH", "CGARCH"])
            for a, b in [("1939-01-01", "1970-01-01"), ("1970-01-01", "2000-01-01"), ("2000-01-01", "2027-01-01")]]
    for kw in [dict(tag="main"), dict(tag="winsor99.9", winsor=0.999), dict(tag="drop1987", drop_1987=True)] + eras:
        c, h = compute(**kw)
        curves.append(c)
        hls.append(h)
    pd.concat(curves).to_csv(RES / "irf_curves.csv.gz", index=False, float_format="%.5g")
    pd.concat(hls).to_csv(RES / "half_lives.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
