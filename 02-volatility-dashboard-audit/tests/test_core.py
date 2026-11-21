"""Unit tests: estimator bias/efficiency on simulated paths, no look-ahead, and the inference code.

    python -m pytest -q tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from estimators import daily_proxies, variance_ratio_hat, weighted_variance  # noqa: E402
from evaluate import bootstrap_counts, holm, mcs, nw_tstat, select_walk_forward  # noqa: E402
from forecast import har_forecasts, realized  # noqa: E402


def simulate_days(n_days=4000, steps=390, sigma=0.01, overnight_share=0.2, seed=0):
    """Brownian log-price: overnight jump N(0, s*sigma^2) then an intraday path on a fine grid."""
    rng = np.random.default_rng(seed)
    o = rng.normal(0, sigma * np.sqrt(overnight_share), n_days)
    inc = rng.normal(0, sigma * np.sqrt((1 - overnight_share) / steps), (n_days, steps))
    path = np.concatenate([np.zeros((n_days, 1)), np.cumsum(inc, 1)], 1)
    u, d, c = path.max(1), path.min(1), path[:, -1]
    r = o + c
    idx = pd.bdate_range("2000-01-03", periods=n_days)
    return pd.DataFrame({"r": r, "o": o, "u": u, "d": d, "c": c}, index=idx), sigma ** 2


@pytest.fixture(scope="module")
def sim():
    # fine grid so the range is close to its continuous-time value
    return simulate_days(n_days=3000, steps=4000)


def test_range_estimators_unbiased_and_more_efficient(sim):
    df, s2 = sim
    px = daily_proxies(df.r, df.o, df.u, df.d, df.c)
    for k in ["C2C", "PK", "GK", "RS"]:
        assert abs(px[k].mean() / s2 - 1) < 0.05, k
    # intraday-only Parkinson misses the overnight share (20%)
    assert px["PK_noON"].mean() / s2 == pytest.approx(0.8, abs=0.05)
    # efficiency: variance of the daily proxy relative to C2C
    assert px["GK"].var() < 0.6 * px["C2C"].var()
    assert px["PK"].var() < 0.7 * px["C2C"].var()


def test_discrete_monitoring_biases_range_down():
    """With 390 one-minute prints the observed range understates the true range (known bias)."""
    df, s2 = simulate_days(n_days=3000, steps=390, seed=5)
    px = daily_proxies(df.r, df.o, df.u, df.d, df.c)
    fine, _ = simulate_days(n_days=3000, steps=4000, seed=5)
    assert px["PK"].mean() < daily_proxies(fine.r, fine.o, fine.u, fine.d, fine.c)["PK"].mean()


def test_yang_zhang_window_unbiased(sim):
    df, s2 = sim
    px = daily_proxies(df.r, df.o, df.u, df.d, df.c)
    yz = weighted_variance(px, df.o, df.c, "YZ", "R21").dropna()
    assert abs(yz.mean() / s2 - 1) < 0.06


def test_no_lookahead(sim):
    """Changing data after day t must not change any estimate dated t or earlier."""
    df, _ = sim
    t = 2000
    df2 = df.copy()
    df2.iloc[t + 1:] *= 3.0
    for est in ["C2C", "GK", "YZ", "GK_cal"]:
        for w in ["R21", "E10"]:
            a = weighted_variance(daily_proxies(*[df[k] for k in "roudc"]), df.o, df.c, est, w)
            b = weighted_variance(daily_proxies(*[df2[k] for k in "roudc"]), df2.o, df2.c, est, w)
            np.testing.assert_allclose(a.iloc[: t + 1], b.iloc[: t + 1], equal_nan=True)
    va, vb = variance_ratio_hat(df.r, 22), variance_ratio_hat(df2.r, 22)
    np.testing.assert_allclose(va.iloc[: t + 1], vb.iloc[: t + 1], equal_nan=True)
    r2a, r2b = (df.r ** 2).to_numpy(), (df2.r ** 2).to_numpy()
    ha = har_forecasts(r2a, r2a, 22)
    hb = har_forecasts(r2b, r2b, 22)
    np.testing.assert_allclose(ha[: t + 1], hb[: t + 1], equal_nan=True)


def test_realized_window():
    r = np.arange(1, 8, dtype=float)
    rv = realized(r, 3)
    assert rv[0] == 4 + 9 + 16 and rv[3] == 25 + 36 + 49 and np.isnan(rv[4:]).all()


def test_variance_ratio_iid_close_to_one(sim):
    df, _ = sim
    # short horizon: the trailing 3-year estimate is precise enough to be near 1 under iid returns
    vr = variance_ratio_hat(df.r, 5).dropna()
    assert abs(vr.mean() - 1) < 0.05
    assert (variance_ratio_hat(df.r, 22).dropna().between(0.75, 1.5)).all()   # shrinkage + clipping


def test_mcs_drops_dominated_keeps_equal():
    rng = np.random.default_rng(1)
    T = 2000
    base = rng.exponential(1, T)
    L = np.column_stack([base, base + rng.normal(0, 0.05, T), base + 0.5 + rng.normal(0, 0.05, T)])
    W = bootstrap_counts(T, 10, rng, b=300)
    p, inset = mcs(L, W)
    assert inset[0] and inset[1] and not inset[2]


def test_bootstrap_weights_sum_to_one():
    W = bootstrap_counts(500, 20, np.random.default_rng(0), b=50)
    np.testing.assert_allclose(W.sum(1), 1.0, rtol=1e-5)


def test_nw_and_holm():
    rng = np.random.default_rng(3)
    t, p = nw_tstat(rng.normal(0, 1, 5000), 10)
    assert p > 0.001
    t, p = nw_tstat(rng.normal(0.2, 1, 5000), 10)
    assert p < 1e-6
    adj = holm(pd.Series([0.01, 0.04, 0.03], index=list("abc")))
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.06) and adj["c"] == pytest.approx(0.06)


def test_walk_forward_uses_only_past():
    """A model that is perfect only in the future must never be chosen before that future arrives."""
    from estimators import DEFAULT
    dates = pd.bdate_range("2005-01-03", "2010-12-31")
    T = len(dates)
    models = ["|".join(DEFAULT), "B|x|y"]
    L = np.ones((T, 2))
    L[:, 1] = 2.0
    late = dates.year >= 2009
    L[late, 1] = 0.0          # B becomes perfect from 2009
    ch = select_walk_forward(dates, L, models, models, h=5)
    assert (ch[dates.year <= 2009] == 0).all()
