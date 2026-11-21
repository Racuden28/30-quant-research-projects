"""Unit tests on synthetic data: correctness, no look-ahead, and the loss identity. Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import estimators as est  # noqa: E402
from evaluate import holm, nw_tstat, stationary_bootstrap_idx  # noqa: E402


def synthetic(T=3000, N=6, seed=0, lagged=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.01, T)
    true = np.linspace(0.5, 1.5, N)
    xl = np.concatenate([[0.0], x[:-1]])
    Y = (true - lagged) * x[:, None] + lagged * xl[:, None] + rng.normal(0, 0.01, (T, N))
    rf = np.full(T, 1e-4)
    mid = np.arange(T) // 21
    E = np.array([np.flatnonzero(mid == k)[-1] for k in range(mid.max() + 1)])
    return Y, x, rf, mid, E, true


def test_ols_daily_recovers_beta():
    Y, x, rf, mid, E, true = synthetic(T=20000)
    b = est.ols_daily(Y, x, E)
    assert np.allclose(np.nanmean(b[20:], axis=0), true, atol=0.03)


def test_ols_matches_numpy_polyfit():
    Y, x, rf, mid, E, _ = synthetic(T=600)
    b = est.ols_daily(Y, x, E)
    k = 25
    e = E[k]
    ref = np.polyfit(x[e - 251:e + 1], Y[e - 251:e + 1, 2], 1)[0]
    assert b[k, 2] == pytest.approx(ref, rel=1e-9)


def test_dimson_corrects_lagged_response():
    Y, x, rf, mid, E, true = synthetic(lagged=0.4)
    plain = np.nanmean(est.ols_daily(Y, x, E)[20:], axis=0)
    dim = np.nanmean(est.dimson(Y, x, E)[20:], axis=0)
    assert np.all(plain < true - 0.3)
    assert np.allclose(dim, true, atol=0.05)


@pytest.mark.parametrize("fn", ["ols", "ewma", "dimson", "fp", "welch", "vasicek"])
def test_no_look_ahead(fn):
    """Changing every return after month-end k must not change the beta forecast made at k."""
    Y, x, rf, mid, E, _ = synthetic(T=2000)
    k = 70
    Y2, x2 = Y.copy(), x.copy()
    Y2[E[k] + 1:] *= -3.0
    x2[E[k] + 1:] += 0.05
    f = {"ols": lambda Y, x: est.ols_daily(Y, x, E),
         "ewma": lambda Y, x: est.ewma(Y, x, E),
         "dimson": lambda Y, x: est.dimson(Y, x, E),
         "fp": lambda Y, x: est.frazzini_pedersen(Y, x, E),
         "welch": lambda Y, x: est.welch_swa(Y, x, E),
         "vasicek": lambda Y, x: est.vasicek(*est.ols_daily(Y, x, E, return_se=True))}[fn]
    a, b = f(Y, x), f(Y2, x2)
    assert np.allclose(a[:k + 1], b[:k + 1], equal_nan=True)
    assert not np.allclose(a[k + 2:], b[k + 2:], equal_nan=True)


def test_vasicek_shrinks_toward_mean():
    Y, x, rf, mid, E, _ = synthetic()
    b, se2 = est.ols_daily(Y, x, E, return_se=True)
    v = est.vasicek(b, se2)
    k = 50
    assert np.all(np.abs(v[k] - b[k].mean()) <= np.abs(b[k] - b[k].mean()) + 1e-12)


def test_welch_winsor_limits_outliers():
    Y, x, rf, mid, E, true = synthetic()
    Y[1500, :] = 0.5                           # a huge one-day jump in every asset
    k = np.searchsorted(E, 1500) + 1
    plain = est.welch_swa(Y, x, E, bounds=(-1e9, 1e9))[k]
    wins = est.welch_swa(Y, x, E)[k]
    assert np.mean(np.abs(wins - true)) < np.mean(np.abs(plain - true))


def test_hedge_loss_identity():
    rng = np.random.default_rng(3)
    x, y, b = rng.normal(size=50), rng.normal(size=50), 0.8
    direct = np.mean((y - b * x) ** 2)
    sums = (y @ y - 2 * b * (x @ y) + b * b * (x @ x)) / 50
    bstar = (x @ y) / (x @ x)
    oracle = np.mean((y - bstar * x) ** 2)
    assert direct == pytest.approx(sums)
    assert direct - oracle == pytest.approx((b - bstar) ** 2 * (x @ x) / 50)


def test_monthly_excess_compounding():
    Y, x, rf, mid, E, _ = synthetic(T=63)
    Ym, xm = est.monthly_excess(Y, x, rf, mid)
    sl = mid == 1
    ref = np.prod(1 + Y[sl, 0] + rf[sl]) - np.prod(1 + rf[sl])
    assert Ym[1, 0] == pytest.approx(ref)


def test_holm_and_nw():
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03})
    h = holm(p)
    assert h["a"] == pytest.approx(0.03) and h["c"] == pytest.approx(0.06) and h["b"] == pytest.approx(0.06)
    rng = np.random.default_rng(1)
    t, pv = nw_tstat(rng.normal(size=5000), 6)
    assert abs(t) < 4
    idx = stationary_bootstrap_idx(100, 10, 12, rng)
    assert idx.shape == (10, 100) and idx.min() >= 0 and idx.max() < 100
