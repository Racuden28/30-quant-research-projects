"""Unit tests for the parts of project 03 where a silent bug would change the conclusions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import (holm, ma_signal, matched_vol_placebo, romano_wolf, sharpe, sharpe_diff_hac,  # noqa: E402
                  spell_shuffle, stationary_bootstrap_idx, strategy_returns, trailing_vol)


def gbm(n=4000, seed=0, sigma=0.01):
    rng = np.random.default_rng(seed)
    # GARCH-like volatility clustering so the vol state is informative
    v = np.exp(np.cumsum(rng.normal(0, 0.05, n)) * 0.3)
    r = rng.normal(0.0003, sigma, n) * v / v.mean()
    idx = pd.bdate_range("2000-01-03", periods=n)
    p = pd.Series(100 * np.cumprod(1 + r), index=idx)
    return p, p.pct_change()


def test_signal_has_no_look_ahead():
    p, r = gbm()
    t = 2500
    p2 = p.copy()
    p2.iloc[t + 1:] *= 3.0                       # change the future
    r2 = p2.pct_change()
    s1, s2 = ma_signal(p, 200), ma_signal(p2, 200)
    pd.testing.assert_series_equal(s1.iloc[:t + 1], s2.iloc[:t + 1])
    v1, v2 = trailing_vol(r), trailing_vol(r2)
    pd.testing.assert_series_equal(v1.iloc[:t + 1], v2.iloc[:t + 1])
    pl1 = matched_vol_placebo(s1, v1, 500)
    pl2 = matched_vol_placebo(s2, v2, 500)
    pd.testing.assert_series_equal(pl1.iloc[:t + 1], pl2.iloc[:t + 1])


def test_position_earns_next_day_return():
    idx = pd.bdate_range("2020-01-01", periods=5)
    r = pd.Series([0.0, 0.10, -0.20, 0.30, 0.05], index=idx)
    pos = pd.Series([1, 0, 1, 0, 1], index=idx, dtype=float)
    st = strategy_returns(pos, r, pd.Series(0.0, index=idx), cost_bp=0)
    # position decided at day 0 (=1) earns day-1 return, at day 1 (=0) earns nothing on day 2, ...
    assert st.gross.tolist() == pytest.approx([0.10, 0.0, 0.30, 0.0])


def test_costs_charged_on_turnover():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series(0.0, index=idx)
    pos = pd.Series([1, 0, 1, 1], index=idx, dtype=float)
    st = strategy_returns(pos, r, pd.Series(0.0, index=idx), cost_bp=10)
    assert st.net.sum() == pytest.approx(-2 * 10 / 1e4)


def test_spell_shuffle_preserves_exposure_and_spells():
    rng = np.random.default_rng(1)
    s = (rng.random(3000) < 0.02).cumsum() % 2      # long spells
    sh = spell_shuffle(s.astype(float), rng)
    assert len(sh) == len(s)
    assert sh.mean() == pytest.approx(s.mean(), abs=1e-12)

    def spells(x):
        x = x.astype(int)
        b = np.concatenate([[0], np.flatnonzero(np.diff(x)) + 1, [len(x)]])
        return sorted(zip(x[b[:-1]], np.diff(b)))
    assert spells(s) == spells(sh)


def test_placebo_matches_exposure_on_average():
    p, r = gbm(6000, seed=3)
    s = ma_signal(p, 200)
    v = trailing_vol(r)
    pl = matched_vol_placebo(s, v, 1260)
    ok = pl.notna() & s.notna()
    assert abs(pl[ok].mean() - s[ok].mean()) < 0.05
    # and it is out of the market only in high-vol states
    assert v[ok & (pl == 0)].mean() > v[ok & (pl == 1)].mean()


def test_stationary_bootstrap_indices():
    rng = np.random.default_rng(2)
    ii = stationary_bootstrap_idx(1000, 20, rng)
    assert ii.min() >= 0 and ii.max() < 1000 and len(ii) == 1000
    runs = np.mean(np.diff(ii) % 1000 == 1)
    assert 0.9 < runs < 0.99                          # mean block ~20 -> ~95% continuations


def test_sharpe_diff_hac():
    rng = np.random.default_rng(4)
    a = rng.normal(0.0005, 0.01, 20000)
    d, p = sharpe_diff_hac(a, a.copy() + 1e-12)
    assert abs(d) < 1e-6
    b = rng.normal(0.0, 0.01, 20000)
    d, p = sharpe_diff_hac(a + 0.001, b)
    assert d > 0 and p < 0.01
    assert d == pytest.approx(sharpe(a + 0.001) - sharpe(b), abs=0.01)


def test_holm_and_romano_wolf():
    assert holm(np.array([0.01, 0.04, 0.03])).tolist() == pytest.approx([0.03, 0.06, 0.06])
    rng = np.random.default_rng(5)
    boot = rng.normal(0, 1, (5000, 3))
    p = romano_wolf(np.array([4.0, 0.0, -1.0]), boot)
    assert p[0] < 0.01 and p[1] > 0.3 and p[2] >= p[1]
    assert np.all((p >= 0) & (p <= 1))
