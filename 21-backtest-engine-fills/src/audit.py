"""
Causality audit: a mechanical look-ahead detector that any strategy can be run through.

Idea. A decision dated t may depend on data up to t only. So if the input is truncated after a cut date
tau, every decision dated <= tau must be unchanged. The audit recomputes the strategy on truncated
inputs at several cut dates and reports the share of decisions before the cut that moved. A causal
strategy scores exactly 0; any dependence on the future (a centred window, a full-sample normalisation,
a smoother, a full-sample fit) shows up as a non-zero share.

Truncation is stricter than perturbing the future: it also catches code that behaves differently when
the sample is shorter (e.g. a z-score over the whole series), which is itself look-ahead.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def causality_audit(fn: Callable, x: pd.Series | pd.DataFrame, cuts: list | None = None,
                    n_cuts: int = 4, seed: int = 0, tol: float = 1e-12) -> pd.DataFrame:
    """fn(x) -> Series/DataFrame of decisions indexed like x. Returns one row per cut date."""
    full = fn(x)
    idx = x.index
    if cuts is None:
        rng = np.random.default_rng(seed)
        lo, hi = int(len(idx) * 0.5), len(idx) - 30
        cuts = sorted(idx[rng.choice(np.arange(lo, hi), size=n_cuts, replace=False)])
    rows = []
    for tau in cuts:
        part = fn(x.loc[:tau])
        a = full.loc[:tau]
        b = part.reindex(a.index)
        A, Bv = np.asarray(a, float), np.asarray(b, float)
        both_nan = np.isnan(A) & np.isnan(Bv)
        diff = ~both_nan & ~(np.abs(A - Bv) <= tol)
        n = int(np.isfinite(A).sum())
        rows.append(dict(cut=pd.Timestamp(tau).date(), decisions_before_cut=n,
                         changed=int(diff.sum()), share_changed=float(diff.sum() / max(n, 1))))
    return pd.DataFrame(rows)
