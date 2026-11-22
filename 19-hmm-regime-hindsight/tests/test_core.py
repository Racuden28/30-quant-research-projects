"""Unit tests for the parts of project 19 where a silent bug would change the conclusions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hmm  # noqa: E402
from core import (holm, matched_vol_placebo, realtime, sharpe_diff_hac, spells,  # noqa: E402
                  stationary_bootstrap_idx, strategy_returns, ewma_vol)

TRUE = hmm.HMMParams(pi=np.array([0.5, 0.5]), A=np.array([[0.99, 0.01], [0.04, 0.96]]),
                     mu=np.array([0.06, -0.12]), var=np.array([0.7, 4.5]))


def sim(T=6000, seed=0):
    return hmm.simulate(TRUE, T, np.random.default_rng(seed))


def brute_force_loglik(x, p):
    """Log-likelihood by summing over every state path (tiny T only)."""
    import itertools
    from scipy.stats import norm
    K, T = p.K, len(x)
    tot = 0.0
    for path in itertools.product(range(K), repeat=T):
        pr = p.pi[path[0]] * norm.pdf(x[0], p.mu[path[0]], np.sqrt(p.var[path[0]]))
        for t in range(1, T):
            pr *= p.A[path[t - 1], path[t]] * norm.pdf(x[t], p.mu[path[t]], np.sqrt(p.var[path[t]]))
        tot += pr
    return np.log(tot)


def test_forward_matches_brute_force():
    x = np.array([0.3, -2.5, 1.1, -0.2, 3.0, -1.4, 0.1])
    _, ll = hmm.filtered(x, TRUE)
    assert ll == pytest.approx(brute_force_loglik(x, TRUE), rel=1e-10)


def test_smoothed_matches_brute_force_marginals():
    import itertools
    from scipy.stats import norm
    x = np.array([0.3, -2.5, 1.1, -0.2, 3.0, -1.4])
    T = len(x)
    marg = np.zeros((T, 2))
    for path in itertools.product(range(2), repeat=T):
        pr = TRUE.pi[path[0]] * norm.pdf(x[0], TRUE.mu[path[0]], np.sqrt(TRUE.var[path[0]]))
        for t in range(1, T):
            pr *= TRUE.A[path[t - 1], path[t]] * norm.pdf(x[t], TRUE.mu[path[t]], np.sqrt(TRUE.var[path[t]]))
        for t in range(T):
            marg[t, path[t]] += pr
    marg /= marg.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(hmm.smoothed(x, TRUE), marg, atol=1e-10)
    # last smoothed = last filtered
    f, _ = hmm.filtered(x, TRUE)
    np.testing.assert_allclose(f[-1], marg[-1], atol=1e-10)


def test_em_recovers_parameters_and_increases_likelihood():
    x, s = sim(20000, 1)
    p = hmm.fit(x, 2, n_starts=3)
    assert np.sqrt(p.var) == pytest.approx(np.sqrt(TRUE.var), rel=0.05)
    assert np.diag(p.A) == pytest.approx(np.diag(TRUE.A), abs=0.01)
    _, ll_true = hmm.filtered(x, TRUE)
    assert p.loglik >= ll_true - 1e-6                     # MLE beats the truth in-sample
    assert p.var[0] < p.var[1]                            # state 0 = calm after sorting


def test_states_sorted_by_variance_whatever_the_start():
    x, _ = sim(8000, 2)
    bad = hmm.HMMParams(np.array([.5, .5]), np.array([[.97, .03], [.03, .97]]), np.zeros(2), np.array([6.0, 0.5]))
    p = hmm.fit(x, 2, init=bad)
    assert p.var[0] < p.var[1]


def test_viterbi_agrees_with_smoothed_mostly():
    x, s = sim(8000, 3)
    g = hmm.smoothed(x, TRUE)
    v = hmm.viterbi(x, TRUE)
    assert ((g[:, 1] > 0.5) == (v == 1)).mean() > 0.97
    assert ((v == 1) == (s == 1)).mean() > 0.9


def test_filtered_has_no_look_ahead_but_smoothed_does():
    x, _ = sim(3000, 4)
    t = 1500
    x2 = x.copy()
    x2[t + 1:] = x2[t + 1:] * 4 - 1                      # change the future
    f1, _ = hmm.filtered(x, TRUE)
    f2, _ = hmm.filtered(x2, TRUE)
    np.testing.assert_allclose(f1[:t + 1], f2[:t + 1])
    assert not np.allclose(hmm.smoothed(x, TRUE)[:t + 1], hmm.smoothed(x2, TRUE)[:t + 1])


def test_realtime_has_no_look_ahead():
    """Changing returns after day t must not change anything the real-time engine reports up to t,
    including the parameters re-estimated along the way."""
    x, _ = sim(2200, 5)
    t = 1700
    x2 = x.copy()
    x2[t + 1:] = -x2[t + 1:] * 3
    a = realtime(x, 2, min_train=1000, refit=50)
    b = realtime(x2, 2, min_train=1000, refit=50)
    np.testing.assert_allclose(a.filt[:t + 1], b.filt[:t + 1], equal_nan=True)
    np.testing.assert_allclose(a.pred[:t + 1], b.pred[:t + 1], equal_nan=True)
    assert np.isnan(a.pred[:999]).all() and not np.isnan(a.pred[999:]).any()


def test_predicted_is_filtered_times_A():
    x, _ = sim(500, 6)
    f, _ = hmm.filtered(x, TRUE)
    np.testing.assert_allclose(hmm.predicted(f, TRUE), f @ TRUE.A)
    np.testing.assert_allclose(hmm.predicted(f, TRUE).sum(1), 1.0)


def test_simulate_stationary_share():
    x, s = hmm.simulate(TRUE, 200000, np.random.default_rng(7))
    assert (s == 1).mean() == pytest.approx(TRUE.stationary()[1], abs=0.02)
    assert TRUE.expected_duration() == pytest.approx([100, 25])


def test_position_earns_next_day_return():
    idx = pd.bdate_range("2020-01-01", periods=5)
    r = pd.Series([0.0, 0.10, -0.20, 0.30, 0.05], index=idx)
    pos = pd.Series([1, 0, 1, 0, 1], index=idx, dtype=float)
    st = strategy_returns(pos, r, pd.Series(0.0, index=idx), cost_bp=0)
    assert st.gross.tolist() == pytest.approx([0.10, 0.0, 0.30, 0.0])


def test_costs_charged_on_turnover():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series(0.0, index=idx)
    pos = pd.Series([1, 0, 0, 1], index=idx, dtype=float)
    st = strategy_returns(pos, r, pd.Series(0.0, index=idx), cost_bp=10)
    assert st.net.tolist() == pytest.approx([0.0, -0.001, 0.0])


def test_placebo_and_ewma_no_look_ahead():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2000-01-03", periods=2000)
    r = pd.Series(rng.normal(0, 0.01, 2000), index=idx)
    sig = pd.Series((rng.random(2000) < 0.8).astype(float), index=idx)
    t = 1200
    r2, sig2 = r.copy(), sig.copy()
    r2.iloc[t + 1:] *= 5
    sig2.iloc[t + 1:] = 0
    pd.testing.assert_series_equal(ewma_vol(r).iloc[:t + 1], ewma_vol(r2).iloc[:t + 1])
    a = matched_vol_placebo(sig, ewma_vol(r), 500, 252)
    b = matched_vol_placebo(sig2, ewma_vol(r2), 500, 252)
    pd.testing.assert_series_equal(a.iloc[:t + 1], b.iloc[:t + 1])
    assert a.iloc[600:].mean() == pytest.approx(sig.iloc[600:].mean(), abs=0.05)


def test_spells():
    f = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1], bool)
    assert spells(f) == [(1, 3), (4, 7), (8, 9)]
    assert spells(f, 3) == [(4, 7)]


def test_hac_sharpe_test_null_and_power():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0004, 0.01, 5000)
    _, p_same = sharpe_diff_hac(a, a * 2)
    assert p_same > 0.99
    b = rng.normal(0.0, 0.01, 5000)
    d, p = sharpe_diff_hac(a + 0.002, b)
    assert d > 0 and p < 0.01


def test_holm_and_bootstrap():
    assert holm(np.array([0.01, 0.04, 0.03])).tolist() == pytest.approx([0.03, 0.06, 0.06])
    idx = stationary_bootstrap_idx(1000, 20, np.random.default_rng(0))
    assert idx.min() >= 0 and idx.max() < 1000 and len(idx) == 1000
