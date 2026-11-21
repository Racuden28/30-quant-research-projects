import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backtest import hold, run  # noqa: E402
from evaluate import holm, stationary_indices  # noqa: E402
from config import STRATEGIES  # noqa: E402


def _data(t=400, n=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.005, 0.04, (t, n))
    return pd.DataFrame(x, index=pd.date_range("1990-01-31", periods=t, freq="ME"),
                        columns=[f"a{i}" for i in range(n)])


def test_hold_buy_and_hold_arithmetic():
    r = np.array([[0.1, -0.1], [0.0, 0.2]])
    pr, end = hold(r, np.array([0.5, 0.5]))
    assert np.isclose(pr[0], 0.0)
    v1 = 0.5 * 1.1 + 0.5 * 0.9 * 1.2
    assert np.isclose((1 + pr).prod(), v1)
    assert np.allclose(end, np.array([0.55, 0.54]) / 1.09)


def test_no_look_ahead_in_investable_strategies():
    x = _data()
    cut = 250
    y = x.copy()
    y.iloc[cut:] = y.iloc[cut:] * 3 + 0.05          # rewrite the future
    a = run("A", window=60, horizon=24, draws=20, x=x)["gross"]
    b = run("A", window=60, horizon=24, draws=20, x=y)["gross"]
    before = a.index < x.index[cut]
    pd.testing.assert_frame_equal(a.loc[before, STRATEGIES], b.loc[before, STRATEGIES])
    # the clairvoyant cells are allowed to (and must) see the future
    assert not np.allclose(a.loc[before, "cell|Clairvoyant|Sample"], b.loc[before, "cell|Clairvoyant|Sample"])


def test_costs_reduce_returns_by_turnover():
    out = run("A", window=60, horizon=24, draws=0, x=_data())
    diff = (out["gross"] - out["net"])["Max-Sharpe (sample)"].dropna()
    to = out["turnover"]["Max-Sharpe (sample)"]
    assert np.allclose(diff.loc[to.index].values, to.values * 10 / 1e4)


def test_holm_monotone_and_bounded():
    p = pd.Series([0.01, 0.04, 0.03, 0.5], index=list("abcd"))
    h = holm(p)
    assert h["a"] == 0.04 and h["c"] == 0.09 and h["b"] == 0.09 and h["d"] == 0.5


def test_stationary_bootstrap_block_length():
    idx = stationary_indices(5000, 20, 4, np.random.default_rng(0))
    runs = np.mean(np.diff(idx, axis=1) != 1)
    assert 0.03 < runs < 0.07
