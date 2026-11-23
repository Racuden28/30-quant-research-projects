"""Unit tests for the project-11 engine. Run: python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core import (FORM, TRADE, coint_screen, distance_stats, eg_stats, eligible, iid_days,  # noqa: E402
                  mackinnon_p, pair_index, run_window, simulate, stationary_days)
from backtest import demean  # noqa: E402
from stats import block_boot_ci, holm, nw_t  # noqa: E402


def test_mackinnon_matches_statsmodels():
    from statsmodels.tsa.adfvalues import mackinnonp
    grid = np.linspace(-20, 3, 400)
    ours = mackinnon_p(grid)
    ref = np.array([mackinnonp(t, "c", N=2) for t in grid])
    assert np.allclose(ours, ref, atol=1e-12)


def test_eg_t_matches_statsmodels_coint():
    from statsmodels.tsa.stattools import coint
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, (FORM, 4))
    ia, ib = pair_index(4)
    _, _, _, t = eg_stats(r, ia, ib)
    lp = np.cumsum(np.log1p(r), 0)
    for k in range(len(ia)):
        ref = coint(lp[:, ia[k]], lp[:, ib[k]], trend="c", maxlag=1, autolag=None)[0]
        assert t[k] == pytest.approx(ref, rel=1e-8)


def test_distance_stats_by_hand():
    r = np.array([[0.1, 0.0], [0.0, 0.1]])
    ssd, sd = distance_stats(r, *pair_index(2))
    p = np.cumprod(1 + r, 0)
    s = p[:, 0] - p[:, 1]
    assert ssd[0] == pytest.approx((s ** 2).sum())
    assert sd[0] == pytest.approx(s.std(ddof=1))


def _one_pair(z, ra, rb, wait):
    rt = np.column_stack([ra, rb]).astype(float)
    z = np.asarray(z, float).reshape(-1, 1)
    return simulate(rt, np.array([0]), np.array([1]), np.ones(1), np.ones(1), z, wait)


def test_simulator_one_converged_trade_by_hand():
    # spread opens above +2 on day 1 (short A, long B), crosses zero on day 3
    z = [0.0, 2.5, 1.0, -0.1, 0.0, 0.0]
    ra = np.array([0.0, 0.0, -0.02, -0.03, 0.01, 0.0])
    rb = np.array([0.0, 0.0, 0.01, 0.02, 0.05, 0.0])
    pnl, turn, st = _one_pair(z, ra, rb, wait=0)
    # opened at the close of day 1, so it earns days 2 and 3; closed at the close of day 3
    va, vb = 1.0, 1.0
    exp2 = -1 * (va * ra[2] - vb * rb[2])
    va, vb = va * (1 + ra[2]), vb * (1 + rb[2])
    exp3 = -1 * (va * ra[3] - vb * rb[3])
    assert pnl[2, 0] == pytest.approx(exp2)
    assert pnl[3, 0] == pytest.approx(exp3)
    assert pnl[4, 0] == 0.0 and pnl[1, 0] == 0.0
    assert st[0, 0] == 1 and st[0, 1] == 1               # one trade, converged
    assert st[0, 2] == pytest.approx(exp2 + exp3)
    assert turn[1, 0] == 2.0                            # $1 + $1 opened


def test_wait_shifts_execution_by_one_day():
    z = [0.0, -2.5, -1.0, 0.5, 0.0, 0.0, 0.0]
    ra = np.array([0, 0, .01, .02, .03, .04, 0.0])
    rb = np.zeros(7)
    p0, _, _ = _one_pair(z, ra, rb, wait=0)
    p1, _, _ = _one_pair(z, ra, rb, wait=1)
    assert np.nonzero(p0[:, 0])[0].tolist() == [2, 3]
    assert np.nonzero(p1[:, 0])[0].tolist() == [3, 4]


def test_open_trade_is_closed_at_end_and_counted_unconverged():
    z = [0.0, 3.0, 3.0, 3.0]
    ra = np.array([0, 0, .01, .01])
    pnl, turn, st = _one_pair(z, ra, np.zeros(4), wait=0)
    assert st[0, 0] == 1 and st[0, 1] == 0
    assert st[0, 3] == pytest.approx(pnl[:, 0].sum())
    assert turn[-1, 0] > 0


def test_positive_control_cointegrated_pairs_make_money():
    """Synthetic market: pairs share a random walk plus a fast OU spread. The strategy must profit and EG
    must flag most pairs; on a demeaned IID placebo of the same block the profit must vanish."""
    rng = np.random.default_rng(7)
    L, npair = FORM + TRADE, 20
    cols = []
    for _ in range(npair):
        common = np.cumsum(rng.normal(0, 0.01, L))
        u = np.zeros(L)
        for t in range(1, L):
            u[t] = 0.8 * u[t - 1] + rng.normal(0, 0.004)
        cols += [common + u, common]
    lp = np.column_stack(cols)
    r = np.vstack([np.zeros(lp.shape[1]), np.expm1(np.diff(lp, axis=0))])
    cfg = [dict(name="d", method="dist", ranks="top", wait=1), dict(name="c", method="coint", ranks="top", wait=1)]
    real = run_window(r, cfg)
    pl = run_window(demean(r)[iid_days(L, rng)], cfg)
    assert real["d"]["pnl"].sum() > 0.02
    assert real["d"]["stats"][1] / real["d"]["stats"][0] > 0.8          # most trades converge
    assert pl["d"]["pnl"].sum() < real["d"]["pnl"].sum()
    # every true partner (columns 2i, 2i+1) is flagged by Engle-Granger at 5 %
    ia, ib = pair_index(r.shape[1])
    p = mackinnon_p(eg_stats(r[:FORM], ia, ib)[3])
    partner = (ia % 2 == 0) & (ib == ia + 1)
    assert (p[partner] < 0.05).mean() >= 0.9
    # ...but a stationary noise component on one leg also makes some *unrelated* pairs look cointegrated:
    # the EG/ADF test is oversized when prices carry fast mean-reverting noise (README section 6.3)
    assert (p[~partner] < 0.05).mean() > 0.05


def test_martingale_placebo_has_zero_mean_pnl():
    """On driftless IID returns a pairs rule cannot make money in expectation."""
    rng = np.random.default_rng(3)
    cfg = [dict(name="d", method="dist", ranks="top", wait=1)]
    tot = []
    for _ in range(150):
        r = rng.standard_t(4, (FORM + TRADE, 12)) * 0.007 + rng.normal(0, 0.01, (FORM + TRADE, 1))
        tot.append(run_window(demean(r)[iid_days(len(r), rng)], cfg)["d"]["pnl"].sum())
    tot = np.array(tot)
    assert abs(tot.mean()) < 3 * tot.std(ddof=1) / np.sqrt(len(tot))


def test_demean_equalises_drift_and_keeps_covariance():
    rng = np.random.default_rng(0)
    b = rng.normal([0.001, -0.002, 0.0005], 0.01, (300, 3))
    d = demean(b)
    m = d.mean(0)
    assert np.allclose(m, m[0]) and m[0] == pytest.approx(b.mean())
    assert np.allclose(np.cov(d.T), np.cov(b.T))


def test_bootstrap_indices():
    rng = np.random.default_rng(0)
    i = iid_days(378, rng)
    assert i.min() >= 0 and i.max() < 378
    s = stationary_days(20000, 21.0, rng)
    runs = 1 + np.sum(np.diff(s) != 1)
    assert 20000 / runs == pytest.approx(21, rel=0.15)


def test_eligibility_rules():
    b = np.random.default_rng(0).normal(0, .01, (FORM + TRADE, 3))
    b[5, 0] = np.nan
    b[:int(0.06 * FORM), 1] = 0.0
    assert eligible(b).tolist() == [False, False, True]


def test_stats_helpers():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1, 2000)
    m, t = nw_t(x, 0)
    assert t == pytest.approx(m / (x.std() / np.sqrt(len(x))), rel=1e-6)
    lo, hi = block_boot_ci(x, draws=500)
    assert lo < x.mean() < hi
    assert holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_block_placebo_of_demeaned_window_is_biased():
    """Why the registered N2 placebo was dropped: on IID data with no dynamics, a stationary block bootstrap
    of the drift-equalised window makes a pairs rule profitable, while the IID-day placebo does not."""
    rng = np.random.default_rng(5)
    cfg = [dict(name="d", method="dist", ranks="top", wait=1)]
    blk, iid = [], []
    for _ in range(200):
        r = rng.standard_t(4, (FORM + TRADE, 20)) * 0.008 + rng.normal(0, 0.01, (FORM + TRADE, 1))
        d = demean(r)
        blk.append(run_window(np.ascontiguousarray(d[stationary_days(len(d), 21.0, rng)]), cfg)["d"]["pnl"].sum())
        iid.append(run_window(np.ascontiguousarray(d[iid_days(len(d), rng)]), cfg)["d"]["pnl"].sum())
    blk, iid = np.array(blk), np.array(iid)
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))  # noqa: E731
    assert blk.mean() > 3 * se(blk)          # biased upwards
    assert abs(iid.mean()) < 3 * se(iid)     # unbiased
