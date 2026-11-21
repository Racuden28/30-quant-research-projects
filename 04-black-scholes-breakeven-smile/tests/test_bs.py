import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bs import black_price, black_price_naive, bs_greeks, bs_price, implied_vol  # noqa: E402
from breakeven import M_GRID, X_GRID, Spec, build_paths, interp_pnl, pnl_grid, root_on_grid  # noqa: E402

mp.mp.dps = 50


def ref(F, K, v, call):
    F, K, v = mp.mpf(F), mp.mpf(K), mp.mpf(v)
    d1 = mp.log(F / K) / v + v / 2
    d2 = d1 - v
    return F * mp.ncdf(d1) - K * mp.ncdf(d2) if call else K * mp.ncdf(-d2) - F * mp.ncdf(-d1)


@pytest.mark.parametrize("k,v", [(0.0, 0.2), (-3.0, 0.1), (3.0, 0.1), (2.0, 0.1), (-0.5, 2.5), (0.1, 0.003)])
def test_stable_price_matches_50_digit_reference(k, v):
    K = float(np.exp(k))
    call = k >= 0
    r = ref(1.0, K, v, call)
    p = black_price(1.0, K, v * v, 1.0, call=call)
    assert abs((mp.mpf(p) - r) / r) < 1e-10


def test_textbook_formula_loses_all_digits_in_the_put_wing():
    K = float(np.exp(-1.0))
    r = float(ref(1.0, K, 0.1, False))          # OTM put, 10 sd below the forward, value ~1e-24
    naive = black_price_naive(1.0, K, 0.01, 1.0, call=False)
    assert abs(naive - r) / r > 0.5               # textbook: call minus parity -> no correct digit
    assert abs(black_price(1.0, K, 0.01, 1.0, call=False) - r) / r < 1e-12


def test_put_call_parity():
    F, K, T, s, D = 105.0, np.array([60, 90, 100, 110, 180.0]), 0.7, 0.3, 0.97
    c = black_price(F, K, T, s, D, True)
    p = black_price(F, K, T, s, D, False)
    np.testing.assert_allclose(c - p, D * (F - K), rtol=0, atol=1e-12)


def test_matches_textbook_near_the_money():
    np.testing.assert_allclose(bs_price(100, 95, 0.5, 0.03, 0.25, 0.01),
                               black_price_naive(100 * np.exp(0.02 * 0.5), 95, 0.5, 0.25, np.exp(-0.015)), rtol=1e-13)


def test_known_value():
    # Hull, Options Futures and Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=0.5 -> c = 4.76, p = 0.81
    assert bs_price(42, 40, 0.5, 0.10, 0.20) == pytest.approx(4.7594, abs=1e-4)
    assert bs_price(42, 40, 0.5, 0.10, 0.20, call=False) == pytest.approx(0.8086, abs=1e-4)


@pytest.mark.parametrize("call", [True, False])
def test_greeks_against_finite_differences(call):
    S, K, T, r, s, q = 100.0, 110.0, 0.75, 0.04, 0.3, 0.015
    g = bs_greeks(S, K, T, r, s, q, call)
    f = lambda **kw: bs_price(**{**dict(S=S, K=K, T=T, r=r, sigma=s, q=q, call=call), **kw})
    h = 1e-3
    assert g["delta"] == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-7)
    assert g["gamma"] == pytest.approx((f(S=S + h) - 2 * f() + f(S=S - h)) / h ** 2, rel=1e-4)
    assert g["vega"] == pytest.approx((f(sigma=s + 1e-5) - f(sigma=s - 1e-5)) / 2e-5, rel=1e-7)
    assert g["theta"] == pytest.approx(-(f(T=T + 1e-5) - f(T=T - 1e-5)) / 2e-5, rel=1e-6)
    assert g["rho"] == pytest.approx((f(r=r + 1e-6) - f(r=r - 1e-6)) / 2e-6, rel=1e-5)


def test_implied_vol_roundtrip_whole_plane():
    k = np.linspace(-4, 4, 41)[:, None]
    v = np.geomspace(0.005, 3, 41)[None, :]
    K, T = np.exp(k) * np.ones_like(v), (v ** 2) * np.ones_like(k)
    for call in (True, False):
        p = black_price(1.0, K, T, 1.0, call=call)
        iv = implied_vol(p, 1.0, K, T, call=call)
        otm = (k >= 0) == call
        ok = np.broadcast_to(otm, K.shape) & (black_price(1.0, K, T, 1.0, call=call) > 1e-290)
        np.testing.assert_allclose(iv[ok], 1.0, rtol=1e-9)


def test_implied_vol_rejects_arbitrage():
    assert np.isnan(implied_vol(-0.01, 100, 100, 1.0))
    assert np.isnan(implied_vol(100.5, 100, 100, 1.0))


def test_breakeven_is_flat_under_gbm():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.2 / np.sqrt(252), 8000)
    s = pd.Series(r, index=pd.bdate_range("2000-01-03", periods=len(r)))
    _, sig, logS = build_paths(s, Spec(), start="2000-01-01")
    m = root_on_grid(pnl_grid(sig, logS, Spec()).mean(0))
    mid = m[np.abs(X_GRID) <= 1.5]
    assert np.all(np.abs(mid - 1.0) < 0.06)
    assert abs(mid[0] - mid[-1]) < 0.06


def test_root_and_interp_consistent():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 600)
    s = pd.Series(r, index=pd.bdate_range("2000-01-03", periods=len(r)))
    _, sig, logS = build_paths(s, Spec(), start="2000-01-01")
    pnl = pnl_grid(sig, logS, Spec())
    m = root_on_grid(pnl)
    ok = (m > M_GRID[0]) & (m < M_GRID[-1])
    e = interp_pnl(pnl, np.where(ok, m, 1.0))
    assert np.nanmax(np.abs(e[ok])) < 1e-2
