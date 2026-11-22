"""Tests for the hedging engine and the analysis helpers of project 16."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bs import bs_greeks, bs_price  # noqa: E402
from frontier import floor_stats  # noqa: E402
from hedgesim import (DT, call_delta, call_price, call_vega, derman_std_volpts, every_h_days,  # noqa: E402
                      floor_fit, hedge_short_call, windows)
from which_delta import holm  # noqa: E402


def gbm(n, T, sig, seed=0):
    rng = np.random.default_rng(seed)
    r = sig * np.sqrt(DT) * rng.standard_normal((n, T)) - 0.5 * sig ** 2 * DT
    return 100 * np.exp(np.concatenate([np.zeros((n, 1)), np.cumsum(r, 1)], 1))


def test_pricer_matches_project_04():
    S, K, tau, s = 100.0, np.array([80, 95, 100, 105, 130.0]), 0.1, 0.25
    assert np.allclose(call_price(S, K, tau, s), bs_price(S, K, tau, 0.0, s), rtol=1e-12, atol=1e-12)
    g = bs_greeks(S, K, tau, 0.0, s)
    assert np.allclose(call_delta(S, K, tau, s), g["delta"], rtol=1e-12)
    assert np.allclose(call_vega(S, K, tau, s), g["vega"], rtol=1e-12)


def test_expiry_values():
    assert call_price(110.0, 100.0, 0.0, 0.2) == pytest.approx(10.0)
    assert call_delta(90.0, 100.0, 0.0, 0.2) == 0.0


def test_zero_vol_path_pnl_is_premium_minus_theta_identity():
    # a flat path: hedge gains are zero, the seller keeps the premium; in vol points premium/vega*100 = sigma*100
    # for ATM to first order (C ~ S sigma sqrt(T)/sqrt(2 pi), vega ~ S sqrt(T)/sqrt(2 pi))
    P = np.full((1, 22), 100.0)
    res = hedge_short_call(P, 100.0, 0.2)
    assert res.hedge_gains[0] == 0.0
    assert res.pnl[0] == pytest.approx(20.0, rel=0.01)


def test_unbiased_and_derman_law_under_gbm():
    P = gbm(60_000, 21, 0.2, seed=1)
    for h, tol in [(1, 0.08), (5, 0.08)]:
        r = hedge_short_call(P, 100.0, 0.2, schedule=every_h_days(21, h))
        N = int(np.ceil(21 / h))
        assert abs(r.pnl.mean()) < 0.1
        assert r.pnl.std() == pytest.approx(derman_std_volpts(0.2, N), rel=tol)


def test_variance_falls_like_one_over_n_under_gbm():
    P = gbm(60_000, 21, 0.2, seed=2)
    v1 = hedge_short_call(P, 100.0, 0.2, schedule=every_h_days(21, 1)).pnl.var()
    v5 = hedge_short_call(P, 100.0, 0.2, schedule=every_h_days(21, 5)).pnl.var()
    assert v1 / v5 == pytest.approx(5 / 21, rel=0.1)


def test_floor_is_zero_under_gbm_and_positive_with_wrong_vol():
    P = gbm(60_000, 21, 0.2, seed=3)
    hg = [1, 2, 3, 4, 5, 10, 21]
    M = np.column_stack([hedge_short_call(P, 100.0, 0.2, schedule=every_h_days(21, h)).pnl for h in hg])
    assert abs(floor_stats(M)["share"]) < 0.12
    rng = np.random.default_rng(4)
    q = 0.2 * np.exp(0.3 * rng.standard_normal(len(P)))  # noisy quote: a level error the hedge can't fix
    M2 = np.column_stack([hedge_short_call(P, 100.0, q, schedule=every_h_days(21, h)).pnl for h in hg])
    assert floor_stats(M2)["share"] > 0.5


def test_put_hedge_equals_call_hedge():
    # short put + its delta hedge = short call + its delta hedge (put-call parity, r = q = 0)
    P = gbm(2000, 21, 0.3, seed=5)
    res = hedge_short_call(P, 100.0, 0.3)
    put_payoff = np.maximum(100.0 - P[:, -1], 0)
    put_prem = res.premium - (100.0 - 100.0)
    gains = 0.0
    for i in range(21):
        tau = (21 - i) * DT
        gains += (call_delta(P[:, i], 100.0, tau, 0.3) - 1) * (P[:, i + 1] - P[:, i])
    put_pnl = (put_prem - put_payoff + gains) / (res.vega0 / 100)
    assert np.allclose(put_pnl, res.pnl, atol=1e-9)


def test_band_zero_equals_daily_and_costs_are_positive():
    P = gbm(3000, 21, 0.2, seed=6)
    a = hedge_short_call(P, 100.0, 0.2, cost_bps=5)
    b = hedge_short_call(P, 100.0, 0.2, band=0.0, cost_bps=5)
    assert np.allclose(a.pnl, b.pnl)
    assert (a.cost > 0).all()
    wide = hedge_short_call(P, 100.0, 0.2, band=0.3, cost_bps=5)
    assert wide.cost.mean() < a.cost.mean()


def test_carry_is_negligible():
    # adding a 5%/yr drift to the forward changes the daily-hedge P&L std by well under 2%
    P = gbm(40_000, 21, 0.2, seed=7)
    drift = np.exp(0.05 * DT * np.arange(22))[None, :]
    s0 = hedge_short_call(P, 100.0, 0.2).pnl.std()
    s1 = hedge_short_call(P * drift, 100.0, 0.2).pnl.std()
    assert abs(s1 / s0 - 1) < 0.02


def test_windows_and_fit_helpers():
    W, idx = windows(np.arange(1, 11, dtype=float), 3)
    assert W.shape == (7, 4) and np.allclose(W[:, 0], 100) and idx[-1] == 6
    a, b = floor_fit(np.array([1, 2, 3.0]), np.array([3, 5, 7.0]))
    assert a == pytest.approx(1) and b == pytest.approx(2)


def test_holm():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_schedule_validation():
    P = gbm(10, 21, 0.2)
    with pytest.raises(ValueError):
        hedge_short_call(P, 100.0, 0.2, schedule=np.zeros(21, bool))
    with pytest.raises(ValueError):
        hedge_short_call(P, 100.0, 0.2, schedule=every_h_days(21, 2), band=0.1)
