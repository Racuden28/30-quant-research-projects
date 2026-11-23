"""Unit tests for the engine, the strategies and the causality audit (synthetic data, runs in seconds)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit import causality_audit                      # noqa: E402
from engine import Market, first_day_pnl, market_from_ohlc, run, run_timing   # noqa: E402
from strategies import centred_ma_bug, ma200, momentum_targets            # noqa: E402


def synth(T=900, N=1, seed=0, with_open=True, rf=0.0001):
    rng = np.random.default_rng(seed)
    ix = pd.bdate_range("2000-01-03", periods=T)
    panels = {}
    for j in range(N):
        on = rng.normal(0, 0.004, T)
        intra = rng.normal(0.0003, 0.008, T)
        c = 100 * np.cumprod((1 + on) * (1 + intra))
        o = np.r_[100, c[:-1]] * (1 + on)
        panels[f"A{j}"] = pd.DataFrame({"open": o if with_open else np.nan, "close": c}, index=ix)
    return market_from_ohlc(panels, pd.Series(rf, index=ix), ix), panels


def test_buy_and_hold_matches_asset():
    m, p = synth()
    r = run_timing(m, pd.Series(1.0, index=m.index), "close")
    asset = p["A0"].close.pct_change().fillna(0.0)
    # held from the close of day 0: every later day earns the asset return
    assert np.allclose(r.gross.values[1:], asset.values[1:], atol=1e-12)


def test_flat_earns_cash():
    m, _ = synth()
    r = run_timing(m, pd.Series(0.0, index=m.index), "close", cost_bps=10)
    assert np.allclose(r.gross.values, m.rf) and r.turnover.sum() == 0


def test_delay_identity_is_exact():
    m, p = synth(seed=3)
    s = ma200(p["A0"].close)
    g0 = run_timing(m, s, "close").excess_gross.sum()
    g1 = run_timing(m, s, "close+1").excess_gross.sum()
    d = first_day_pnl(s.fillna(0.0).values, m).sum()
    assert abs((g0 - g1) - d) < 1e-12


def test_open_fill_equals_close_fill_when_open_is_previous_close():
    m, p = synth(seed=4)
    m.on_ret[:] = 0.0                                # open = previous close: no overnight move
    s = ma200(p["A0"].close)
    a = run_timing(m, s, "open", cost_bps=5).net.values
    b = run_timing(m, s, "close", cost_bps=5).net.values
    # same trade price, booked one day later in the series: compare the totals
    assert abs(np.log1p(a).sum() - np.log1p(b).sum()) < 1e-10


def test_open_fill_splits_overnight_and_intraday():
    m, p = synth(seed=5)
    tgt = np.zeros(len(m.index))
    tgt[100] = 1.0                                   # decide at close 100 to buy, sell at close 101
    tgt[101:] = 0.0
    tgt[:100] = 0.0
    r = run(m, tgt[:, None], "open")
    # day 101: bought at the open, so it earns only the intraday piece (+ cash on the day)
    assert np.isclose(r.gross.values[101], m.rf[101] + (m.in_ret[101, 0] - m.rf[101]), atol=1e-12)
    # day 102: sold at the open, so it earns only the overnight piece
    assert np.isclose(r.gross.values[102], (1 + m.on_ret[102, 0]) * (1 + m.rf[102]) - 1, atol=1e-12)


def test_open_fill_refuses_missing_opens():
    m, p = synth(with_open=False)
    with pytest.raises(ValueError):
        run_timing(m, ma200(p["A0"].close), "open")


def test_drift_is_not_turnover_and_costs_are_exact():
    m, _ = synth(N=2, seed=6)
    T = len(m.index)
    tgt = np.full((T, 2), np.nan)
    tgt[10] = [0.5, 0.5]
    tgt[500] = [0.5, 0.5]
    r = run(m, tgt, "close", cost_bps=10)
    R = m.close_ret[500]
    w = r.weights[499] * (1 + R) / (1 + r.gross.iloc[500])    # weights drifted through day 500, before the trade
    assert np.isclose(r.turnover.iloc[10], 1.0)
    assert np.isclose(r.turnover.iloc[500], np.abs(np.array([0.5, 0.5]) - w).sum())
    assert r.turnover.drop(r.index[[10, 500]]).sum() == 0
    assert np.isclose((1 + r.net.iloc[500]) / (1 + r.gross.iloc[500]), 1 - 10e-4 * r.turnover.iloc[500])


def test_engine_has_no_look_ahead():
    m, p = synth(seed=7)
    s = ma200(p["A0"].close)
    a = run_timing(m, s, "close+1", 5).net
    m2, _ = synth(seed=7)
    m2.close_ret[600:] = 0.05                       # change the future
    b = run_timing(m2, s, "close+1", 5).net
    assert np.allclose(a.values[:600], b.values[:600])


def test_audit_passes_causal_and_catches_centred_window():
    _, p = synth(T=1500, seed=8)
    px = p["A0"].close
    ok = causality_audit(ma200, px, n_cuts=3)
    bad = causality_audit(centred_ma_bug, px, n_cuts=3)
    assert ok.changed.sum() == 0
    assert bad.changed.sum() > 0


def test_momentum_book_is_dollar_neutral():
    rng = np.random.default_rng(9)
    ix = pd.bdate_range("2000-01-03", periods=900)
    ex = pd.DataFrame(rng.normal(0, 0.01, (900, 20)), index=ix, columns=[f"X{i}" for i in range(20)])
    tgt = momentum_targets(ex).dropna(how="all")
    assert len(tgt) > 20
    assert np.allclose(tgt.sum(axis=1), 0.0) and np.allclose(tgt.abs().sum(axis=1), 2.0)
    # decisions only on month ends, using complete months
    assert set(tgt.index) <= set(ex.groupby(ex.index.to_period("M")).tail(1).index)
    a = causality_audit(momentum_targets, ex, cuts=list(tgt.index[-6:-3]))
    assert a.changed.sum() == 0
