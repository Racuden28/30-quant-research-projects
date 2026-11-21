import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import core  # noqa: E402


def _R(n=5, rho=0.4):
    R = np.full((n, n), rho)
    np.fill_diagonal(R, 1)
    return R


def test_ewma_vol_is_ex_ante():
    r = np.zeros((100, 1))
    r[80] = 1.0
    s = core.ewma_vol(r, warm=10, floor=0.0)
    assert s[80, 0] <= s[79, 0]                           # shock not yet seen on its own day
    assert s[81, 0] > 10 * s[80, 0]                       # seen the next day


def test_fisher_mean_constant_matrix():
    assert core.fisher_mean(_R(6, 0.3)) == pytest.approx(0.3)


def test_mvt_unit_has_target_correlation_and_unit_variance():
    rng = np.random.default_rng(0)
    for nu in [None, 6.0]:
        x = core.mvt_unit(rng, _R(4, 0.5), 200_000, nu)
        assert np.allclose(x.var(0), 1, atol=0.03)
        assert core.fisher_mean(core.corr(x)) == pytest.approx(0.5, abs=0.01)


def test_fit_nu_recovers_truth():
    rng = np.random.default_rng(1)
    R = _R(5, 0.3)
    x = core.mvt_unit(rng, R, 40_000, 5.0)
    assert core.fit_mvt_nu(x, R) == pytest.approx(5.0, abs=0.5)


def test_gaussian_tail_conditioning_lowers_correlation():
    """Forbes-Rigobon in reverse: truncating the market to one tail cuts conditional correlation."""
    rng = np.random.default_rng(2)
    x = core.mvt_unit(rng, _R(10, 0.4), 100_000, None)
    m = x.mean(1)
    crash = m <= np.quantile(m, 0.05)
    big = np.abs(m) >= np.quantile(np.abs(m), 0.9)
    assert core.fisher_mean(core.corr(x[crash])) < 0.4 - 0.1
    assert core.fisher_mean(core.corr(x[big])) > 0.4 + 0.1


def test_shrinkage_and_clipping_return_valid_correlations():
    rng = np.random.default_rng(3)
    x = core.mvt_unit(rng, _R(20, 0.3), 120, None)
    for C in [core.lw_constant_corr(x), core.rmt_clip(core.corr(x), 120)]:
        assert np.allclose(np.diag(C), 1)
        assert np.linalg.eigvalsh(C).min() > -1e-10
        assert np.allclose(C, C.T)


def test_lw_shrinks_toward_constant_correlation_in_small_samples():
    rng = np.random.default_rng(4)
    R = _R(20, 0.3)
    errs_s, errs_lw = [], []
    for _ in range(30):
        x = core.mvt_unit(rng, R, 60, None)
        errs_s.append(np.linalg.norm(core.corr(x) - R))
        errs_lw.append(np.linalg.norm(core.lw_constant_corr(x) - R))
    assert np.mean(errs_lw) < np.mean(errs_s)


def test_dcc_recovers_parameters():
    rng = np.random.default_rng(5)
    a, b, N, T = 0.04, 0.93, 3, 4000
    Qbar = _R(N, 0.4)
    Q = Qbar.copy()
    z = np.empty((T, N))
    for t in range(T):
        d = np.sqrt(np.diag(Q))
        Rt = Q / np.outer(d, d)
        z[t] = np.linalg.cholesky(Rt) @ rng.standard_normal(N)
        Q = (1 - a - b) * Qbar + a * np.outer(z[t], z[t]) + b * Q
    (ah, bh), _ = core.dcc_fit(z)
    assert ah == pytest.approx(a, abs=0.015)
    assert bh == pytest.approx(b, abs=0.03)


def test_multiple_testing_adjustments():
    p = np.array([0.001, 0.01, 0.03, 0.5])
    assert np.allclose(core.holm(p), [0.004, 0.03, 0.06, 0.5])
    q = core.bh(p)
    assert np.all(q >= p) and q[0] == pytest.approx(0.004)


def test_newey_west_zero_mean():
    rng = np.random.default_rng(6)
    t, p = core.newey_west_t(rng.standard_normal(5000))
    assert p > 0.01
