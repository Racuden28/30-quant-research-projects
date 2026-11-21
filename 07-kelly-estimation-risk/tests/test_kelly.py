"""Unit tests for project 07: theory, no look-ahead, accounting. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backtest import DATA, build_inputs, load_rf, run_path  # noqa: E402
from kelly import (drawdown_law, expected_plugin_growth, holm, newey_west_t,  # noqa: E402
                   optimal_multiplier, shrinkage_from_t)


def test_optimal_multiplier_maximises_expected_growth():
    c = np.linspace(0, 2, 20001)
    for theta, T in [(0.2, 5), (0.5, 20), (1.0, 3)]:
        best = c[np.argmax(expected_plugin_growth(c, theta, T))]
        assert abs(best - optimal_multiplier(theta, T)) < 1e-3


def test_full_kelly_beats_cash_iff_theta2T_gt_1():
    assert expected_plugin_growth(1.0, 0.3, 10) < 0 < expected_plugin_growth(1.0, 0.3, 12)


def test_shrinkage_from_t():
    assert np.allclose(shrinkage_from_t([0.5, -1.0, 2.0, -4.0]), [0, 0, 0.75, 0.9375])


def test_drawdown_law_under_gbm():
    rng = np.random.default_rng(0)
    theta, n, years, steps = 0.5, 4000, 60, 52        # weekly GBM steps, full Kelly
    dt = 1 / steps
    z = rng.standard_normal((n, years * steps))
    logw = np.cumsum(0.5 * theta**2 * dt + theta * np.sqrt(dt) * z, axis=1)
    p = (logw.min(1) <= np.log(0.5)).mean()
    assert abs(p - drawdown_law(0.5, 1.0)) < 0.04     # discrete monitoring slightly under-counts


def test_no_lookahead_in_expanding_moments():
    inp = build_inputs("SPY")
    r = pd.read_csv(DATA / "etf_daily_log_returns.csv", index_col=0, parse_dates=True)["SPY"].dropna()
    x = np.expm1(r.values) - load_rf(r.index)
    for i in [0, 100, 500, len(inp) - 1]:
        t = int(inp.n_days.iloc[i])
        assert r.index[t] == inp.origin.iloc[i]
        assert np.isclose(inp.mu_hat_ann.iloc[i], x[:t].mean() * 252, rtol=1e-6)
        assert np.isclose(inp.sigma_hat_ann.iloc[i], x[:t].std(ddof=1) * np.sqrt(252), rtol=1e-6)


def test_buy_and_hold_accounting():
    R = np.array([0.1, -0.05, 0.02])
    _, lr = run_path(np.ones(3), R, np.zeros(3), spread=0.0, tc=0.0)
    assert np.allclose(lr, np.log1p(R))


def test_leverage_pays_spread_and_ruin_is_absorbing():
    _, lr = run_path(np.array([4.0, 4.0]), np.array([-0.3, 0.5]), np.zeros(2), spread=0.0, tc=0.0)
    assert np.isnan(lr[1])


def test_holm_and_nw():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    t, _ = newey_west_t(np.r_[np.ones(50), -np.ones(50)] + 0.0)
    assert abs(t) < 1e-9
