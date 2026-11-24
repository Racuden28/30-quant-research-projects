"""Part 3: deflate this portfolio's own published Sharpe claims.

For each claim: the series behind it, the variants that the project itself reports (a LOWER bound on
the trials actually run), and every deflation method studied in Parts 0-2. Inputs are read from the
sibling project folders, so this script needs the monorepo checkout (it is skipped with a note
otherwise). Only claims whose null is 'Sharpe = 0' are audited: market-neutral books and active
returns against the project's own benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import dsr as D
from data import INDICES, RES, ROOT, load_indices
from zoo import SIGNALS_A, zoo_a

MONO = ROOT.parent
GRID3 = [("SMA", p, 0) for p in (20, 50, 100, 150, 200, 250, 300)] + [("XMA", 20, 100), ("XMA", 50, 200)]


def _stats(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m, sd, sk, ku = D.moments(x[:, None])
    return dict(T=x.size, sr=float(m[0] / sd[0]), skew=float(sk[0]), kurt=float(ku[0]))


def _srs(M):
    out = []
    for c in M.T:
        c = c[~np.isnan(c)]
        out.append(c.mean() / c.std(ddof=1))
    return np.array(out)


def _lj(M):
    M = pd.DataFrame(M).dropna()
    lam, _ = D.corr_eigs(M.to_numpy())
    return D.neff_liji(lam), len(M)


def hedged(a, b):
    """Return of a minus its OLS beta on b. A positive mean means a improves b's Sharpe when added to it."""
    ok = ~(np.isnan(a) | np.isnan(b))
    out = np.full(a.shape, np.nan)
    A, B = a[ok], b[ok]
    beta = np.cov(A, B)[0, 1] / np.var(B, ddof=1)
    out[ok] = A - beta * B
    return out


def claim_03():
    """#3: does the 200-day rule on the S&P 500 add to buy & hold (daily, 1934-2026, net of costs)?

    DSR is defined for one Sharpe ratio, so a 'better than the benchmark' claim is audited through the
    rule's alpha on the benchmark: the series is the rule's return minus beta x buy & hold. A positive
    alpha means the rule improves the benchmark's Sharpe ratio when mixed with it, which is weaker than
    'the rule's own Sharpe is higher'.
    """
    s = pd.read_csv(MONO / "03-moving-average-volatility/results/daily_net_returns_ma200.csv.gz")
    g = s[s["index"] == "GSPC"]
    head = hedged(g["ma"].to_numpy(), g["bh"].to_numpy())
    X = load_indices()
    _, net, elig, owner = zoo_a(X)
    cols = []
    for j, c in enumerate(INDICES):
        x = X[c].to_numpy()
        for spec in GRID3:
            k = j * len(SIGNALS_A) * 2 + SIGNALS_A.index(spec) * 2  # LF mode
            a = net[:, k].astype(float)
            a[: elig[k]] = np.nan
            a[np.isnan(x)] = np.nan
            cols.append(hedged(a, x))
    V = np.column_stack(cols)
    return dict(project="#03", claim="S&P 500 200-day rule: positive alpha on buy & hold (net)",
                freq="daily", ann=252, series=head, variants_sr=_srs(V), lj=_lj(V[X.index >= "1996-01-01"])[0],
                n_raw=90, ledger="9 rules x 10 indices (Romano-Wolf grid of #3)")


def claim_03_ixic():
    """#3: Nasdaq MA20, one of the 2 of 90 grid cells that survived Romano-Wolf against buy & hold."""
    X = load_indices()
    _, net, elig, _ = zoo_a(X)
    j = INDICES.index("IXIC")
    k = j * len(SIGNALS_A) * 2 + SIGNALS_A.index(("SMA", 20, 0)) * 2
    x = X["IXIC"].to_numpy()
    a = net[:, k].astype(float)[elig[k]:]
    xx = x[elig[k]:]
    a = hedged(a[~np.isnan(xx)], xx[~np.isnan(xx)])
    c = claim_03()
    c.update(project="#03", claim="Nasdaq 20-day rule: positive alpha on buy & hold (net)", series=a)
    return c


def claim_11():
    """#11: country-ETF pairs (GGR top-20, one-day wait), gross, 2002-2026."""
    p = pd.read_csv(MONO / "11-pairs-trading/results/performance.csv")
    p = p[(p.series == "real") & p.period.str.startswith("Full")]
    head = p[(p.universe == "countries") & (p.config == "D20_w1")].iloc[0]
    sr = p.sharpe.to_numpy() / np.sqrt(12)
    return dict(project="#11", claim="Country-ETF pairs, GGR top 20, gross", freq="monthly", ann=12,
                summary=dict(T=int(head.months), sr=float(head.sharpe) / np.sqrt(12), skew=0.0, kurt=3.0),
                variants_sr=sr, lj=np.nan, n_raw=len(p),
                ledger=f"{len(p)} configurations x universes reported (full period); moments assumed normal")


def claim_12():
    """#12: textbook 12-1 industry momentum, gross long-short (monthly, 1928-2026)."""
    r = pd.read_csv(MONO / "12-momentum-trading-costs/results/runs.csv.gz")
    piv = r.pivot_table(index="month", columns=["universe", "signal", "design"], values="gross")
    head = piv[("A", "12-1", "base")].to_numpy()
    V = piv.to_numpy()
    return dict(project="#12", claim="Industry momentum 12-1 (textbook), gross", freq="monthly", ann=12,
                series=head, variants_sr=_srs(V), lj=_lj(V)[0], n_raw=V.shape[1],
                ledger="2 universes x 3 signals x 21 trading designs")


def claim_17():
    """#17: real-time factor-model selector, tangency excess return (monthly, 1984-2026)."""
    r = pd.read_csv(MONO / "17-five-factor-holdout/results/realtime_returns.csv")
    V = r[["CAPM", "FF3", "C4", "FF5", "FF5-HML", "FF6", "FF6-HML", "Selector"]].to_numpy()
    return dict(project="#17", claim="Real-time factor selector (tangency of chosen model)", freq="monthly",
                ann=12, series=r["Selector"].to_numpy(), variants_sr=_srs(V), lj=_lj(V)[0], n_raw=V.shape[1],
                ledger="7 models + the selector")


def claim_20():
    """#20: does levered risk parity add to 60/40 (monthly, 1929-2026)? RP minus beta x 60/40."""
    base = MONO / "20-risk-parity-bond-bull/results"
    files = ["bt_main.csv.gz"] + sorted(f.name for f in base.glob("bt_rob_*.csv.gz"))
    cols = []
    for f in files:
        b = pd.read_csv(base / f, index_col=0)
        cols.append(pd.Series(hedged(b["RP|xret"].to_numpy(), b["60/40|xret"].to_numpy()), index=b.index, name=f))
    V = pd.concat(cols, axis=1)
    return dict(project="#20", claim="Risk parity: positive alpha on 60/40", freq="monthly", ann=12,
                series=V.iloc[:, 0].to_numpy(), variants_sr=_srs(V.to_numpy()), lj=_lj(V.to_numpy())[0],
                n_raw=V.shape[1], ledger=f"main + {V.shape[1] - 1} robustness backtests")


def evaluate(c):
    st = c.get("summary") or _stats(c["series"])
    T, sr, sk, ku = st["T"], st["sr"], st["skew"], st["kurt"]
    n = c["n_raw"]
    lj = c["lj"] if np.isfinite(c["lj"]) else n
    v_xs = float(np.var(c["variants_sr"], ddof=1))
    p = D.psr(sr, T, sk, ku)
    row = dict(project=c["project"], claim=c["claim"], freq=c["freq"], T=T, years=T / c["ann"],
               sr_ann=sr * np.sqrt(c["ann"]), skew=sk, kurt=ku, n_raw=n, n_lj=lj, ledger=c["ledger"],
               v_xs_times_T=v_xs * T, psr=p,
               dsr_textbook=D.dsr(sr, T, sk, ku, n, v_xs)[0],
               dsr_nullV=D.dsr(sr, T, sk, ku, n, 1 / T)[0],
               sidak_raw_p=1 - (p ** n) if p < 1 else 0.0,
               sidak_lj_p=1 - (p ** lj) if p < 1 else 0.0)
    # minimum track record (years) for the Sidak-Li-Ji test to pass at this Sharpe
    z = D.stats.norm.ppf(0.95 ** (1 / lj))
    v = 1 - sk * sr + (ku - 1) / 4 * sr ** 2
    row["mintrl_years_sidak_lj"] = (1 + v * (z / sr) ** 2) / c["ann"] if sr > 0 else np.inf
    return row


def main():
    if not (MONO / "03-moving-average-volatility").exists():
        print("monorepo siblings not found; audit skipped (results/audit.csv is committed)")
        return
    rows = [evaluate(f()) for f in (claim_03, claim_03_ixic, claim_11, claim_12, claim_17, claim_20)]
    df = pd.DataFrame(rows)
    df.to_csv(RES / "audit.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 250)
    print(df.drop(columns=["ledger", "claim"]).round(3).to_string())


if __name__ == "__main__":
    main()
