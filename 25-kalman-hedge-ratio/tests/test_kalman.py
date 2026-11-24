"""Unit tests for project 25: filters against brute-force references, no look-ahead, steady-state algebra,
and the trading simulator on a placebo."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kalman import (R_GRID, ew_beta, kf2_loglik_grid, kf2_run, kf_scalar_grid, matched_halflife,  # noqa: E402
                    matched_lambda, r_from_halflife, rolling_ols, steady_state_pm)


def _data(T=1500, seed=0, beta=1.0):
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.standard_normal(T)
    y = beta * x + 0.01 * rng.standard_normal(T)
    return y, x


def _naive_kf(y, x, r, b0=1.0, p0=1e4, R=1e-4):
    """Textbook scalar Kalman filter written out with absolute variances."""
    q = r * R
    b, P = b0, p0 * R
    out = []
    for yt, xt in zip(y, x):
        Pm = P + q
        out.append(b)
        F = xt * xt * Pm + R
        K = Pm * xt / F
        b = b + K * (yt - b * xt)
        P = Pm - K * xt * Pm
    return np.array(out)


def test_scalar_filter_matches_textbook():
    y, x = _data()
    rg = np.array([0.0, 1.0, 50.0])
    B, LL, PM = kf_scalar_grid(y, x, rg)
    for g, r in enumerate(rg):
        assert np.allclose(B[:, g], _naive_kf(y, x, r), atol=1e-10)


def test_only_the_noise_ratio_matters():
    """Scaling q and R together (q = r R) leaves every beta unchanged, which is why R can be concentrated out."""
    y, x = _data()
    for r in (0.5, 20.0):
        assert np.allclose(_naive_kf(y, x, r, R=1e-4), _naive_kf(y, x, r, R=7e-2), atol=1e-10)


def test_r_zero_is_recursive_ols_through_origin():
    y, x = _data()
    B, _, _ = kf_scalar_grid(y, x, np.array([0.0]), b0=0.0, p0=1e12)
    t = 800
    assert B[t, 0] == pytest.approx((x[:t] * y[:t]).sum() / (x[:t] ** 2).sum(), rel=1e-6)


def test_no_lookahead():
    """Changing the future never changes a past prediction (filters, EWMAs, rolling OLS)."""
    y, x = _data()
    y2 = y.copy()
    y2[1000:] += 0.5
    B1, L1, _ = kf_scalar_grid(y, x, R_GRID)
    B2, L2, _ = kf_scalar_grid(y2, x, R_GRID)
    assert np.array_equal(B1[:1001], B2[:1001])
    assert np.allclose(L1[:1001], L2[:1001], equal_nan=True)
    d = np.full(len(y), 0.99)
    assert np.array_equal(ew_beta(y, x, d)[:1001], ew_beta(y2, x, d)[:1001])
    assert np.array_equal(rolling_ols(y, x, 63, 50)[:1001], rolling_ols(y2, x, 63, 50)[:1001], equal_nan=True)


def test_likelihood_recovers_noise_ratio():
    """Positive control: with a true random-walk beta, the profile likelihood picks a nearby grid value."""
    rng = np.random.default_rng(3)
    T, s2 = 20000, 1e-4
    r_true = float(r_from_halflife(63.0, s2))
    x = 0.01 * rng.standard_normal(T)
    beta = 1 + np.cumsum(np.sqrt(r_true * 1e-4) * rng.standard_normal(T))
    y = beta * x + 0.01 * rng.standard_normal(T)
    _, LL, _ = kf_scalar_grid(y, x, R_GRID)
    r_hat = R_GRID[np.nanargmax(LL[-1])]
    assert 0.3 < matched_halflife(r_hat, s2) / 63.0 < 3.0


def test_constant_beta_likelihood_prefers_long_memory():
    y, x = _data(T=20000, seed=5)
    _, LL, _ = kf_scalar_grid(y, x, R_GRID)
    assert matched_halflife(R_GRID[np.nanargmax(LL[-1])], 1e-4) > 1000


def test_steady_state_and_matching():
    s2 = 1.3e-4
    for r in (1e-3, 1.0, 100.0):
        pm = steady_state_pm(r, s2)
        p = pm / (1 + s2 * pm)
        assert p + r == pytest.approx(pm, rel=1e-10)          # fixed point of the Riccati recursion
        lam = matched_lambda(r, s2)
        assert 1 - lam == pytest.approx(s2 * pm / (1 + s2 * pm), rel=1e-10)
    for h in (5.0, 63.0, 500.0):
        assert matched_halflife(r_from_halflife(h, s2), s2) == pytest.approx(h, rel=1e-8)


def test_matched_ewma_tracks_kalman():
    """In a stationary world the Kalman beta and its matched EWMA nearly coincide."""
    y, x = _data(T=6000, seed=7)
    r = float(r_from_halflife(40.0, 1e-4))
    B, _, _ = kf_scalar_grid(y, x, np.array([r]))
    lam = matched_lambda(r, (x ** 2).mean())
    E = ew_beta(y, x, np.full(len(y), lam))
    assert np.corrcoef(B[500:, 0], E[500:])[0, 1] > 0.98


def test_rolling_ols_matches_numpy():
    y, x = _data(T=400)
    b = rolling_ols(y, x, 63, 50)
    t = 300
    ref = np.polyfit(x[t - 63:t], y[t - 63:t], 1)[0]
    assert b[t] == pytest.approx(ref, rel=1e-8)


def test_two_state_filter_and_grid():
    rng = np.random.default_rng(1)
    x = np.cumsum(0.01 * rng.standard_normal(300))
    y = 0.2 + 0.8 * x + 0.01 * rng.standard_normal(300)
    ll = kf2_loglik_grid(y, x, np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert np.unravel_index(np.argmax(ll), ll.shape) == (0, 0)      # constant coefficients are preferred
    A, Bp, V, F, Af, Bf = kf2_run(y, x, 0.0, 0.0)
    assert Af[-1] == pytest.approx(0.2, abs=0.02) and Bf[-1] == pytest.approx(0.8, abs=0.05)
    assert np.allclose(V, y - A - Bp * x)


def test_placebo_has_zero_mean_pnl():
    """The pairs simulator earns nothing on a market of driftless IID relative returns (project 11's N1)."""
    from pairs import FORM, TRADE, run_window
    rng = np.random.default_rng(11)
    tot = {}
    for rep in range(30):
        f = 0.01 * rng.standard_normal((FORM + TRADE, 12)) + 0.006 * rng.standard_normal((FORM + TRADE, 1))
        out, _ = run_window(f)
        for k, (p, _, _) in out.items():
            tot.setdefault(k, []).append(p.sum())
    for k, v in tot.items():
        v = np.array(v)
        assert abs(v.mean()) < 4 * v.std() / np.sqrt(len(v)) + 1e-12, k


def test_kalman_spread_closure_is_mostly_the_filter():
    """On noise, a Kalman spread closes mainly because the coefficients move, not because prices converge."""
    from pairs import FORM, TRADE, run_window
    rng = np.random.default_rng(12)
    num = den = 0.0
    for rep in range(10):
        f = 0.01 * rng.standard_normal((FORM + TRADE, 12)) + 0.006 * rng.standard_normal((FORM + TRADE, 1))
        out, _ = run_window(f)
        st = out[("dist", "K")][2]
        num += st[6]
        den += st[5]
        s = out[("dist", "S")][2]
        assert s[5] == 0 or s[6] / s[5] == pytest.approx(1.0)       # static spread: all closure is price
    assert num / den < 0.9
