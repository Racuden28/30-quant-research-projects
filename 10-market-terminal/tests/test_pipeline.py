"""Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stats import bh, cw_series, cw_test, expanding_oos, r2_os  # noqa: E402
from widgets import WIDGETS, compute_targets, compute_widgets, monthly_targets, monthly_widgets  # noqa: E402

rng = np.random.default_rng(0)
IDX = pd.bdate_range("2000-01-03", periods=800)


def test_widgets_have_no_lookahead():
    r = pd.Series(rng.normal(0, 0.01, len(IDX)), IDX)
    w0 = compute_widgets(r)
    r2 = r.copy()
    r2.iloc[500:] = rng.normal(0, 0.05, len(r2) - 500)  # rewrite the future
    w1 = compute_widgets(r2)
    pd.testing.assert_frame_equal(w0.iloc[:500], w1.iloc[:500])


def test_targets_use_only_the_future_window():
    r = pd.Series(np.arange(1, 31) / 1000.0, pd.bdate_range("2020-01-01", periods=30))
    y = compute_targets(r, h=5)
    assert np.isclose(y.ret.iloc[0], r.iloc[1:6].sum())
    assert y.ret.iloc[-5:].isna().all() and np.isfinite(y.ret.iloc[-6])
    assert (y.mdd.dropna() <= 0).all()


def test_monthly_targets_skip_one_month():
    r = pd.Series(np.arange(1, 41) / 100.0, pd.date_range("1900-01-01", periods=40, freq="MS"))
    y = monthly_targets(r, skip=1)
    assert np.isclose(y.ret1.iloc[0], r.iloc[2])  # r[t+1] is skipped (Working effect)
    w = monthly_widgets(r)
    assert list(w.columns)[0] == "chg1m"


def test_expanding_oos_respects_embargo():
    n, h = 400, 10
    y = np.full(n, 1.0)
    y[300:] = 100.0  # targets from row 300 on are only known from row 310
    f = expanding_oos(y, None, h, start=305, step=1)
    assert np.isclose(f[305], 1.0)  # row 300's target not yet observable at 305
    assert f[320] > 1.0


def test_r2os_and_cw_detect_real_signal_and_reject_noise():
    n = 3000
    x = rng.normal(size=n)
    y = 0.3 * x + rng.normal(size=n)
    fb = expanding_oos(y, None, 1, 500, 21)
    fm = expanding_oos(y, x, 1, 500, 21)
    assert r2_os(y, fm, fb) > 0.05
    m = np.isfinite(fm)
    assert cw_test(cw_series(y[m], fm[m], fb[m]), 2)[1] < 0.01
    z = rng.normal(size=n)
    fz = expanding_oos(y, z, 1, 500, 21)
    assert cw_test(cw_series(y[m], fz[m], fb[m]), 2)[1] > 0.01


def test_bh_matches_known_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    rej, adj = bh(p, 0.05)
    assert rej.sum() == 2
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)


def test_widget_list_is_complete():
    r = pd.Series(rng.normal(0, 0.01, len(IDX)), IDX)
    assert list(compute_widgets(r).columns) == WIDGETS and len(WIDGETS) == 14
