import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import engine as E  # noqa: E402
import stats as S  # noqa: E402
from backtest import tick_size  # noqa: E402


def synthetic(n_assets=20, years=6, seed=0, vol=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=252 * years)
    r = rng.normal(0.0002, vol, (len(idx), n_assets))
    return pd.DataFrame(r, index=idx, columns=[f"X{i:02d}" for i in range(n_assets)])


def test_month_path_is_exact_buy_and_hold():
    w = np.array([0.5, -0.5])
    rd = np.array([[0.10, 0.00], [0.00, 0.10]])
    r_book, drift, r_m = E._month_daily_path(w, rd)
    # day 1: long +10% -> +0.05 on capital 1; day 2: short +10% on 0.5 -> -0.05 on capital 1.05
    assert r_book[0] == pytest.approx(0.05)
    assert r_book[1] == pytest.approx(-0.05 / 1.05)
    assert r_m == pytest.approx(0.0)
    assert drift == pytest.approx(np.array([0.55, -0.55]) / 1.0)


def test_no_trading_when_prices_do_not_move_and_ranks_are_fixed():
    d = synthetic() * 0.0
    U = E.Universe("t", d)
    sig = pd.DataFrame(np.tile(np.arange(20.0), (len(U.monthly), 1)), index=U.monthly.index,
                       columns=d.columns)
    out = E.run(U, sig, E.Design("base", "b"), min_assets=10)["monthly"]
    assert out["turnover"].iloc[0] == pytest.approx(2.0)          # initial purchase: $1 long + $1 short
    assert out["turnover"].iloc[1:].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_breakeven_cost_zeroes_mean_net_return():
    U = E.Universe("t", synthetic(seed=1))
    sig = E.momentum_signal(U, 12, 1)
    out = E.run(U, sig, E.Design("base", "b"), min_assets=10)["monthly"]
    ref = out["sigma_bar"].mean()
    for model in ("const", "vol"):
        c = E.breakeven(out, ref, model)
        assert E.add_costs(out, ref, c, model).mean() == pytest.approx(0.0, abs=1e-12)


def test_kappa_is_one_when_every_asset_always_has_the_same_vol():
    U = E.Universe("t", synthetic(seed=2))
    U.vol21 = U.vol21 * 0 + 0.2
    sig = E.momentum_signal(U, 12, 1)
    out = E.run(U, sig, E.Design("base", "b"), min_assets=10)["monthly"]
    k = E.kappa(out)
    assert k["kappa"] == pytest.approx(1.0)
    assert k["kappa_time"] == pytest.approx(1.0)


def test_no_look_ahead():
    d = synthetic(seed=3)
    U1 = E.Universe("t", d)
    cut = U1.monthly.index[40]
    d2 = d.copy()
    d2.loc[d2.index > cut] = d2.loc[d2.index > cut] * -3.0 + 0.01     # rewrite the future
    U2 = E.Universe("t", d2)
    D = E.Design("vol", "v", vol_target=0.1, q_exit=0.5)
    o1 = E.run(U1, E.momentum_signal(U1, 12, 1), D, min_assets=10)["monthly"]
    o2 = E.run(U2, E.momentum_signal(U2, 12, 1), D, min_assets=10)["monthly"]
    # trades decided at or before `cut` and the returns earned through `cut` must be identical
    pd.testing.assert_series_equal(o1.loc[:cut, "gross"], o2.loc[:cut, "gross"])
    nxt = U1.monthly.index[41]
    assert o1.loc[nxt, "turnover"] == pytest.approx(o2.loc[nxt, "turnover"])


def test_theta_one_equals_textbook_and_slow_designs_trade_less():
    U = E.Universe("t", synthetic(seed=4, n_assets=30))
    sig = E.momentum_signal(U, 12, 1)
    base = E.run(U, sig, E.Design("base", "b"), min_assets=10)["monthly"]
    same = E.run(U, sig, E.Design("p", "p", theta=1.0), min_assets=10)["monthly"]
    pd.testing.assert_frame_equal(base, same)
    for D in (E.Design("p", "p", theta=0.33), E.Design("b", "b", q_exit=0.5),
              E.Design("h", "h", hold=3), E.Design("s", "s", smooth=3)):
        slow = E.run(U, sig, D, min_assets=10)["monthly"]
        assert slow["turnover"].iloc[12:].mean() < base["turnover"].iloc[12:].mean()


def test_long_short_book_is_dollar_neutral_and_long_only_fully_invested():
    U = E.Universe("t", synthetic(seed=5))
    sig = E.momentum_signal(U, 12, 1)
    ls = E.run(U, sig, E.Design("base", "b"), min_assets=10)["monthly"]
    lo = E.run(U, sig, E.Design("lo", "l", long_only=True), min_assets=10)["monthly"]
    assert ls["gross_exposure"].iloc[1:].between(1.9, 2.1).all()
    assert lo["gross_exposure"].iloc[1:].between(0.999, 1.001).all()
    assert (lo["n_short"] == 0).all()


def test_stationary_bootstrap_is_reproducible_and_in_range():
    a = S.stationary_indices(100, n_boot=50)
    b = S.stationary_indices(100, n_boot=50)
    assert (a == b).all() and a.min() >= 0 and a.max() < 100
    # mean block length is about 12
    runs = (np.diff(a, axis=1) != 1).sum() / a.shape[0]
    assert 5 < 100 / (runs + 1) < 20


def test_holm():
    adj = S.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)


def test_tick_regimes():
    t = tick_size(pd.DatetimeIndex(["1996-12-31", "1999-12-31", "2001-04-30"]))
    assert list(t) == [1 / 8, 1 / 16, 0.01]
