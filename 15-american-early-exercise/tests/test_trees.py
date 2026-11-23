"""Unit tests for the lattice engines, the boundary/policy tools and the data panel. Run: pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import trees as tr  # noqa: E402

HULL = (50.0, 50.0, 5 / 12, 0.10, 0.0, 0.40)   # Hull's American put example
HULL_REF = 4.2842157                             # LR at n = 20,003 and 40,007 extrapolated; BBS + Richardson at 40,000 agrees (README 2.2)


@pytest.mark.parametrize("name", list(tr.METHODS))
def test_american_put_converges(name):
    v = tr.METHODS[name](*HULL, False, True, 1600)
    assert abs(v - HULL_REF) < 5e-3


@pytest.mark.parametrize("name", ["crr", "jr", "tian", "trig", "lr", "crr_avg", "bbs", "bbsr", "lr_r", "lr_r2", "tri"])
def test_european_matches_black_scholes(name):
    for call in (True, False):
        bs = tr.bs_price(100.0, 95.0, 0.7, 0.04, 0.02, 0.25, call)
        v = tr.METHODS[name](100.0, 95.0, 0.7, 0.04, 0.02, 0.25, call, False, 1600)
        assert abs(v - bs) < 5e-3


def test_put_call_parity_bs():
    S, K, T, r, q, s = 100.0, 110.0, 0.5, 0.03, 0.01, 0.2
    c, p = tr.bs_price(S, K, T, r, q, s, True), tr.bs_price(S, K, T, r, q, s, False)
    assert abs(c - p - (S * np.exp(-q * T) - K * np.exp(-r * T))) < 1e-12


def test_american_call_without_dividends_equals_european():
    v_am = tr.lr_r(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, True, True, 801)
    v_eu = tr.bs_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, True)
    assert abs(v_am - v_eu) < 2e-4


def test_american_put_zero_rate_has_no_premium():
    v_am = tr.lr_r(100.0, 100.0, 1.0, 0.0, 0.0, 0.2, False, True, 801)
    v_eu = tr.bs_price(100.0, 100.0, 1.0, 0.0, 0.0, 0.2, False)
    assert abs(v_am - v_eu) < 2e-4


def test_american_at_least_intrinsic_and_european():
    for K in (60.0, 100.0, 160.0):
        am = tr.lr_r(100.0, K, 1.0, 0.08, 0.01, 0.3, False, True, 401)
        assert am >= max(K - 100.0, 0.0) - 1e-12
        assert am >= tr.bs_price(100.0, K, 1.0, 0.08, 0.01, 0.3, False) - 1e-6


def test_put_call_symmetry_mcdonald_schroder():
    # American call(S, K, r, q) = American put(K, S, q, r)
    c = tr.crr(100.0, 90.0, 1.0, 0.02, 0.06, 0.25, True, True, 1000)
    p = tr.crr(90.0, 100.0, 1.0, 0.06, 0.02, 0.25, False, True, 1000)
    assert abs(c - p) < 1e-10


def test_lr_european_second_order():
    bs = tr.bs_price(100.0, 105.0, 1.0, 0.05, 0.0, 0.2, False)
    e1 = abs(tr.lr(100.0, 105.0, 1.0, 0.05, 0.0, 0.2, False, False, 101) - bs)
    e2 = abs(tr.lr(100.0, 105.0, 1.0, 0.05, 0.0, 0.2, False, False, 201) - bs)
    assert e2 < e1 / 3.0          # about 1/4 for an O(1/n^2) method


def test_boundary_policy_reproduces_american_value():
    S, K, T, r, q, s, n = 100.0, 100.0, 1.0, 0.05, 0.015, 0.15, 600
    _, b, v = tr.crr_boundary(S, K, T, r, q, s, False, n)
    assert abs(tr.crr_policy_value(S, K, T, r, q, s, False, n, b) - v) < 1e-12
    assert abs(v - tr.crr(S, K, T, r, q, s, False, True, n)) < 1e-12


def test_never_exercise_policy_is_european():
    S, K, T, r, q, s, n = 100.0, 100.0, 1.0, 0.05, 0.015, 0.15, 600
    b = np.full(n, np.nan)
    assert abs(tr.crr_policy_value(S, K, T, r, q, s, False, n, b) - tr.crr(S, K, T, r, q, s, False, False, n)) < 1e-12


def test_wrong_boundary_never_beats_optimal():
    S, K, T, r, q, s, n = 100.0, 100.0, 1.0, 0.05, 0.015, 0.15, 400
    _, _, v = tr.crr_boundary(S, K, T, r, q, s, False, n)
    for m in (0.6, 0.9, 1.1, 1.8):
        _, b, _ = tr.crr_boundary(S, K, T, r, q, s * m, False, n)
        assert tr.crr_policy_value(S, K, T, r, q, s, False, n, b) <= v + 1e-12


def test_put_boundary_below_strike_and_rising_to_strike():
    _, b, _ = tr.crr_boundary(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, False, 1000)
    ok = ~np.isnan(b)
    assert np.all(b[ok] < 100.0 * 1.01)
    assert b[ok][-1] > b[ok][len(b[ok]) // 2]


def test_delta_gamma_signs():
    d, g, v = tr.crr_delta(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, False, True, 500)
    assert -1.0 < d < 0.0 and g > 0.0


def test_input_panel_no_gaps():
    p = Path(__file__).resolve().parents[1] / "data" / "sp500_option_inputs.csv"
    d = pd.read_csv(p, parse_dates=["date"], index_col="date")
    assert d[["r", "q", "sigma"]].notna().all().all()
    assert d.index.is_monotonic_increasing and d.index[0].year == 1928 and d.index[-1].year == 2026
    assert (d.q > 0).all() and (d.sigma > 0).all()
