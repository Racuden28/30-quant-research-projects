"""Unit tests for project 18. Run from the project folder: python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import nss_yield  # noqa: E402
from models import (ewma_vol, factor_cov, interp_matrix, minvar_matrix, ns_loadings, pca,  # noqa: E402
                    principal_angles, solve_factor_hedge, explained_share, nonparallel)
from stats import dm_test, holm, ratio_ci  # noqa: E402

MATS = np.array([1, 2, 3, 5, 7, 10], float)
iT, iH = [0, 2, 4], [1, 3, 5]


def test_interp_matrix_preserves_dv01_and_is_linear():
    A = interp_matrix(MATS[iT], MATS[iH])
    assert np.allclose(A.sum(0), 1)                       # total DV01 is moved, not created
    assert np.allclose(A[:, 1], [2 / 3, 1 / 3, 0])        # 3Y is a third of the way from 2Y to 5Y
    assert np.allclose(A[:, 0], [1, 0, 0])                # 1Y is below the first hedge: flat
    assert np.allclose(A[:, 2], [0, 0.6, 0.4])            # 7Y: 60% to 5Y, 40% to 10Y


def test_factor_hedge_neutralises_the_factors_exactly():
    rng = np.random.default_rng(0)
    L = rng.standard_normal((6, 3))
    A = solve_factor_hedge(L, iT, iH)
    w = rng.standard_normal(3)
    h = -A @ w
    assert np.allclose(L[iT].T @ w + L[iH].T @ h, 0)


def test_minvar_equals_ols_on_the_sample_covariance():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 6)) @ rng.standard_normal((6, 6))
    S = X.T @ X / len(X)
    A = minvar_matrix(S, iT, iH)
    B = np.linalg.lstsq(X[:, iH], X[:, iT], rcond=None)[0]
    assert np.allclose(A, B, atol=1e-10)


def test_factor_cov_keeps_diagonal_and_full_rank():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((500, 6)) @ rng.standard_normal((6, 6))
    S = np.cov(X, rowvar=False)
    for j in range(1, 6):
        C = factor_cov(S, j)
        assert np.allclose(np.diag(C), np.diag(S))
        assert np.linalg.eigvalsh(C).min() > 0
    assert np.allclose(factor_cov(S, 6), S, atol=1e-8)


def test_pca_sign_convention():
    m = np.linspace(1, 10, 6)
    level, slope = np.ones(6), (m - m.mean()) / 10
    hump = -(m - 5) ** 2 / 20 + 1
    hump -= hump.mean()
    S = 9 * np.outer(level, level) + 4 * np.outer(slope, slope) + 1 * np.outer(hump, hump) + 0.01 * np.eye(6)
    L, vals = pca(S, 3)
    assert vals[0] > vals[1] > vals[2]
    assert L[:, 0].sum() > 0 and L[-1, 1] > L[0, 1] and L[2:4, 2].mean() > L[[0, 5], 2].mean()


def test_ewma_vol_has_no_look_ahead():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((300, 2))
    s1 = ewma_vol(x)
    y = x.copy()
    y[200:] *= 50                                         # change the future only
    s2 = ewma_vol(y)
    assert np.allclose(s1[:201], s2[:201])                # sigma_t uses data up to t-1 only
    assert not np.allclose(s1[201:], s2[201:])


def test_ns_loadings_shape_and_limits():
    L = ns_loadings(np.array([0.01, 1, 5, 30, 1000.0]), 3)
    assert np.allclose(L[:, 0], 1)
    assert L[0, 1] == pytest.approx(1, abs=0.01) and L[-1, 1] < 0.01   # slope: 1 at 0, 0 at infinity
    assert L[0, 2] < 0.01 and L[-1, 2] < 0.01 and L[2, 2] > 0.2        # curvature: hump in the middle


def test_nss_reduces_to_ns_when_beta3_is_zero():
    n = np.array([1, 2, 5, 10, 30.0])
    a = nss_yield(n, 5, -1, 2, 0, 1.5, -999.99)
    b = nss_yield(n, 5, -1, 2, 0, 1.5, 3.0)
    assert np.allclose(a, b)
    assert nss_yield(np.array([1e-6]), 5, -1, 2, 3, 1.5, 3.0)[0] == pytest.approx(4, abs=1e-4)


def test_principal_angles_and_shares():
    A = np.eye(6)[:, :2]
    assert np.allclose(principal_angles(A, A), 0, atol=1e-6)
    B = np.eye(6)[:, [0, 2]]
    assert np.allclose(sorted(principal_angles(A, B)), [0, 90], atol=1e-6)
    X = np.ones((10, 6)) * np.arange(10)[:, None]
    assert explained_share(X, np.ones((6, 1))) == pytest.approx(1)
    assert np.allclose(nonparallel(X), 0)


def test_stats_helpers():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(holm(p), [0.04, 0.09, 0.09, 0.5])
    rng = np.random.default_rng(4)
    a, b = rng.exponential(1, 3000), rng.exponential(1, 3000)
    r, lo, hi = ratio_ci(a, b)
    assert lo < 1 < hi
    t, pv = dm_test(a * 0.5, b)
    assert t < 0 and pv < 1e-6
