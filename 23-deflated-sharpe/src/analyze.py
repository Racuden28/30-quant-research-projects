"""Turn the raw search records into the paper's tables.

Every decision rule below answers 'is the in-sample winner of this search real?' at a nominal 5% level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import dsr as D
from data import INDICES, RES, load_indices, load_industries
from search import PLANT_SR, SPLIT_YEARS
from zoo import trial_names_a, trial_names_b, zoo_a, zoo_b

A = lambda d, c: d[c].to_numpy(dtype=float)  # noqa: E731


def methods(d, sr="sr", skew="skew", kurt="kurt", v="v_xs", extra=0):
    """Reject (True) / accept for each method. `extra` adds trials (the planted strategy)."""
    s, T, sk, ku = A(d, sr), A(d, "T"), A(d, skew), A(d, kurt)
    N = A(d, "N") + extra
    vx = A(d, v)
    p = D.psr(s, T, sk, ku)
    out = {
        "PSR, no deflation": p > 0.95,
        "DSR textbook (raw N, cross-sectional V)": D.dsr(s, T, sk, ku, N, vx)[0] > 0.95,
        "DSR, raw N, null V = 1/T": D.dsr(s, T, sk, ku, N, 1 / T)[0] > 0.95,
        "DSR, Li-Ji N, cross-sectional V": D.dsr(s, T, sk, ku, A(d, "lj") + extra, vx)[0] > 0.95,
        "DSR, clusters (K, cluster V)": D.dsr(s, T, sk, ku, A(d, "k"), A(d, "v_clu"))[0] > 0.95,
    }
    for lab, n in (("raw N", N), ("Li-Ji N", A(d, "lj") + extra), ("Nyholt N", A(d, "ny") + extra),
                   ("participation N", A(d, "pr") + extra), ("cluster K", A(d, "k"))):
        out[f"Sidak max-PSR, {lab}"] = p > 0.95 ** (1 / n)
    return out


def group(search):
    return np.where(search == "A-pooled", "A pooled (all indices)",
                    np.where(search == "B", "B industries", "A single index"))


def size_table(pl):
    g = pl[pl.basis == "gross"].copy()
    g["group"] = group(g.search.to_numpy())
    rows = []
    for gname, d in g.groupby("group"):
        dec = methods(d)
        for m, r in dec.items():
            # cluster by world for the CI: share of rejections per world, then bootstrap worlds
            per_w = pd.Series(r).groupby(d.seed.to_numpy()).mean()
            rng = np.random.default_rng(1)
            bs = [per_w.sample(frac=1, replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(500)]
            rows.append(dict(group=gname, method=m, size=r.mean(), lo=np.quantile(bs, 0.025),
                             hi=np.quantile(bs, 0.975), n_searches=len(r), worlds=per_w.size))
    return pd.DataFrame(rows)


def power_table(pl):
    g = pl[(pl.basis == "gross") & pl.search.isin(["A-pooled", "B"])].copy()
    g["group"] = group(g.search.to_numpy())
    rows = []
    for gname, d in g.groupby("group"):
        for s in PLANT_SR:
            dec = methods(d, sr=f"p{s}_sr", skew=f"p{s}_skew", kurt=f"p{s}_kurt", v=f"p{s}_v_xs", extra=1)
            won = A(d, f"p{s}_won") > 0.5
            for m, r in dec.items():
                rows.append(dict(group=gname, planted_sr=s, method=m, power=r.mean(),
                                 power_and_planted_won=(r & won).mean(), planted_won=won.mean()))
    return pd.DataFrame(rows)


def implied_n_table(pl):
    """N that would give DSR (cross-sectional V) and the Sidak max-PSR test an exact 5% size."""
    g = pl[pl.basis == "gross"].copy()
    g["group"] = group(g.search.to_numpy())
    rows = []
    for gname, d in g.groupby("group"):
        s, T, sk, ku, vx = A(d, "sr"), A(d, "T"), A(d, "skew"), A(d, "kurt"), A(d, "v_xs")
        p = D.psr(s, T, sk, ku)
        # Sidak: N* with mean(p > 0.95**(1/N*)) = 5%  <=>  0.95**(1/N*) = 95th pct of p
        q = np.quantile(p, 0.95)
        n_sidak = np.log(0.95) / np.log(q) if q < 1 else np.inf
        n_dsr = D.implied_n(s, T, sk, ku, vx)
        rows.append(dict(group=gname, mean_raw_N=A(d, "N").mean(), mean_liji=A(d, "lj").mean(),
                         mean_nyholt=A(d, "ny").mean(), mean_participation=A(d, "pr").mean(),
                         mean_clusters=np.nanmean(A(d, "k")), implied_N_sidak=n_sidak,
                         implied_N_dsr_textbook=n_dsr, mean_vxs_times_T=np.mean(vx * T)))
    return pd.DataFrame(rows)


def real_table(re, rc):
    names = {"A": trial_names_a(INDICES), "B": trial_names_b()}
    d = re[re.basis == "net"].copy()
    d["winner"] = [names["B" if s == "B" else "A"][i] for s, i in zip(d.search, d.win_id)]
    dec = methods(d)
    for m, r in dec.items():
        d[m] = r
    # reality-check p-value: share of drift-preserving sign-flip worlds with a best IS net SR >= real
    null = rc[rc.basis == "net"].groupby(["split", "search"]).max_sr.apply(np.array)
    pv = []
    for _, r in d.iterrows():
        z = null.get((r.split, r.search))
        pv.append(np.nan if z is None else (1 + np.sum(z >= r.sr)) / (1 + z.size))
    d["rc_p"] = pv
    d["Reality check (drift-kept sign flip)"] = d.rc_p < 0.05
    d["sr_ann"] = d.sr * np.sqrt(d.ann)
    d["oos_ann_win"] = d.oos_sr_win * np.sqrt(d.ann)
    d["oos_ann_med"] = d.oos_sr_med * np.sqrt(d.ann)
    d["oos_ann_top10"] = d.oos_sr_top10 * np.sqrt(d.ann)
    d["group"] = group(d.search.to_numpy())
    return d


def real_summary(d):
    rows = []
    meths = [c for c in d.columns if c.startswith(("PSR", "DSR", "Sidak", "Reality"))]
    rng = np.random.default_rng(23)
    for gname, g in list(d.groupby("group")) + [("All 99 searches", d)]:
        for m in meths:
            acc = g[m].astype(bool)
            rows.append(dict(group=gname, method=m, searches=len(g), flagged=int(acc.sum()),
                             oos_win_flagged=g.loc[acc, "oos_ann_win"].mean(),
                             oos_win_unflagged=g.loc[~acc, "oos_ann_win"].mean(),
                             oos_pct_flagged=g.loc[acc, "oos_win_pct"].mean(),
                             oos_pct_unflagged=g.loc[~acc, "oos_win_pct"].mean(),
                             **_diff_ci(g.oos_ann_win.to_numpy(), acc.to_numpy(), rng)))
    return pd.DataFrame(rows)


def _diff_ci(y, flag, rng, B=2000):
    """Bootstrap (over searches) CI of mean OOS Sharpe of flagged minus unflagged winners."""
    if flag.sum() < 2 or (~flag).sum() < 2:
        return dict(diff=np.nan, diff_lo=np.nan, diff_hi=np.nan)
    n = y.size
    out = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        f = flag[i]
        if f.sum() and (~f).sum():
            out.append(y[i][f].mean() - y[i][~f].mean())
    return dict(diff=y[flag].mean() - y[~flag].mean(), diff_lo=np.quantile(out, 0.025),
                diff_hi=np.quantile(out, 0.975))


def stress(re_net):
    """Calendar-year net return of each search's winner in 2008, 2020 and 2022 vs the zoo median."""
    XA, XB = load_indices(), load_industries()
    _, nA, _, _ = zoo_a(XA)
    _, nB, _ = zoo_b(XB.to_numpy())
    rows = []
    for yr, Y in ((2008, 2005), (2020, 2015), (2022, 2020)):
        for search in ["A-pooled", "A-GSPC", "B"]:
            r = re_net[(re_net.split == Y) & (re_net.search == search)].iloc[0]
            if search == "B":
                m = XB.index.year == yr
                R = nB[m]
                ids = np.arange(nB.shape[1])
                bh = (1 + XB[m].mean(1)).prod() - 1
            else:
                m = XA.index.year == yr
                R = nA[m].astype(float)
                if search == "A-GSPC":
                    ids = np.arange(0, 114)
                else:
                    ids = np.arange(nA.shape[1])
                bh = (1 + XA["GSPC"][m].fillna(0)).prod() - 1
            cum = np.prod(1 + R[:, ids], 0) - 1
            w = cum[list(ids).index(r.win_id)] if r.win_id in ids else np.nan
            rows.append(dict(year=yr, chosen_in=f"{Y - 9}-{Y}", search=search, winner=r.winner,
                             winner_return=w, zoo_median=np.median(cum), zoo_p10=np.quantile(cum, 0.1),
                             zoo_p90=np.quantile(cum, 0.9), benchmark=bh,
                             benchmark_name="EW industries (excess)" if search == "B" else "S&P 500 (excess)"))
    return pd.DataFrame(rows)


def main():
    pl = pd.read_csv(RES / "placebo_searches.csv.gz")
    re = pd.read_csv(RES / "real_searches.csv")
    f = RES / "reality_check_null.csv.gz"
    rc = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["basis", "split", "search", "max_sr"])
    st = size_table(pl)
    st.to_csv(RES / "size.csv", index=False, float_format="%.4g")
    pw = power_table(pl)
    pw.to_csv(RES / "power.csv", index=False, float_format="%.4g")
    im = implied_n_table(pl)
    im.to_csv(RES / "implied_n.csv", index=False, float_format="%.4g")
    d = real_table(re, rc)
    keep = ["split", "search", "group", "winner", "N", "T", "sr_ann", "skew", "kurt", "lj", "k", "rc_p",
            "oos_ann_win", "oos_ann_med", "oos_ann_top10", "oos_win_pct", "oos_rank_corr"]
    meths = [c for c in d.columns if c.startswith(("PSR", "DSR", "Sidak", "Reality"))]
    d[keep + meths].to_csv(RES / "real_decisions.csv", index=False, float_format="%.4g")
    rs = real_summary(d)
    rs.to_csv(RES / "real_summary.csv", index=False, float_format="%.4g")
    sw = stress(d)
    sw.to_csv(RES / "stress_windows.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(st.pivot_table(index="method", columns="group", values="size").round(4).to_string())
    print(pw.pivot_table(index="method", columns=["group", "planted_sr"], values="power").round(3).to_string())
    print(im.round(2).to_string())
    print(rs.round(3).to_string())
    print(sw.round(3).to_string())


if __name__ == "__main__":
    main()
