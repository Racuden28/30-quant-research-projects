"""Why the Avellaneda-Lee mean-reversion filter passes almost everything.

1. The residuals come from an OLS regression with an intercept over the same L days, so they
   sum to zero and the cumulative residual X ends at exactly 0: X is a discrete Brownian
   bridge, which is mean-reverting by construction.
2. An AR(1) slope fitted to a short random walk is biased down (Dickey-Fuller), so even
   without the bridge most noise paths look mean-reverting.
3. With X_L = 0 the s-score reduces to a weighted sum of the window's residuals whose weight
   grows linearly with recency: s ~ sum_i (i-1)/L * eps_i / sigma_eq. It is a 60-day,
   linearly-weighted residual reversal signal, not a test of anything.

This module simulates (1)-(2) under the null (pure IID noise, no predictability) and checks (3).
Output: results/theory_filter_null.csv, results/theory_kernel.csv.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as en

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def null_pass_rate(L: int, k: int, intercept: bool, n: int = 20000, max_tau: float = 30.0,
                   t_df: float | None = None, seed: int = 0):
    """Share of pure-noise residual paths that pass the 1/kappa < max_tau filter."""
    rng = np.random.default_rng(seed)
    if t_df:
        e = rng.standard_t(t_df, size=(L, n))
    else:
        e = rng.standard_normal((L, n))
    if k or intercept:
        cols = []
        if intercept:
            cols.append(np.ones(L))
        if k:
            cols += list(rng.standard_normal((k, L)))
        Xd = np.column_stack(cols)
        e = e - Xd @ np.linalg.lstsq(Xd, e, rcond=None)[0]
    s, passed, kappa, b = en.s_scores(e, max_tau)
    tau = 1 / kappa
    return float(passed.mean()), float(np.nanmedian(tau)), float(np.nanmedian(b))


def kernel_weights(L: int):
    """Weights of the linear-recency kernel implied by the s-score when X_L = 0."""
    i = np.arange(1, L + 1)
    return (i - 1) / L


def kernel_check(n: int = 5000, L: int = 60, seed: int = 1):
    """Correlation between the exact s-score and the linear-kernel approximation on noise."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((L, n)); e -= e.mean(0)
    s, passed, kappa, b = en.s_scores(e, 30.0)
    _, _, m, seq, _ = en.ou_fit(e)
    approx = (kernel_weights(L)[:, None] * e).sum(0) / seq
    ok = np.isfinite(s)
    # the cross-sectional centring of m in s is a constant shift; drop it for the comparison
    s_unc = -m / seq
    return float(np.corrcoef(s_unc[ok], approx[ok])[0, 1]), float(np.corrcoef(s[ok], approx[ok])[0, 1])


def main():
    rows = []
    for L in (30, 60, 120, 252):
        for k, icpt, lab in ((0, False, "random walk, no regression"),
                             (0, True, "intercept only (bridge)"),
                             (15, True, "intercept + 15 factors")):
            p, tau, b = null_pass_rate(L, k, icpt)
            rows.append(dict(L=L, design=lab, pass_share=p, median_tau=tau, median_b=b))
    p, tau, b = null_pass_rate(60, 15, True, t_df=3)
    rows.append(dict(L=60, design="intercept + 15 factors, t(3) noise", pass_share=p, median_tau=tau, median_b=b))
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "theory_filter_null.csv"), index=False)
    c_unc, c = kernel_check()
    pd.DataFrame([dict(corr_uncentred=c_unc, corr_centred=c)]).to_csv(os.path.join(OUT, "theory_kernel.csv"), index=False)
    print(df.to_string(index=False)); print("kernel corr", c_unc, c)


if __name__ == "__main__":
    main()
