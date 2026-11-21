import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import portfolio as pf  # noqa: E402

rng = np.random.default_rng(0)


def _problem(n=8, t=200):
    a = rng.standard_normal((n, n))
    cov = a @ a.T / n + np.eye(n) * 0.1
    mu = rng.normal(0.05, 0.1, n)
    x = rng.multivariate_normal(mu, cov, size=t)
    return mu, cov, x


def test_ledoit_wolf_matches_sklearn():
    from sklearn.covariance import LedoitWolf
    _, _, x = _problem()
    assert np.allclose(pf.ledoit_wolf(x), LedoitWolf().fit(x).covariance_, atol=1e-12)


def test_max_sharpe_beats_random_long_only():
    mu, cov, _ = _problem()
    w = pf.max_sharpe(mu, cov)
    assert w.min() >= 0 and abs(w.sum() - 1) < 1e-10
    best = pf.sharpe(mu, cov, w)
    for _ in range(2000):
        v = rng.dirichlet(np.ones(len(mu)) * 0.3)
        assert pf.sharpe(mu, cov, v) <= best + 1e-9


def test_max_sharpe_matches_generic_solver():
    mu, cov, _ = _problem()
    neg = lambda w: -(w @ mu) / np.sqrt(w @ cov @ w)  # noqa: E731
    res = minimize(neg, np.full(len(mu), 1 / len(mu)), method="SLSQP", bounds=[(0, 1)] * len(mu),
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"ftol": 1e-12})
    assert pf.sharpe(mu, cov, pf.max_sharpe(mu, cov)) >= -res.fun - 1e-7


def test_unconstrained_tangency_when_interior():
    n = 5
    cov = np.diag(np.arange(1, n + 1) / 10.0)
    mu = np.full(n, 0.05)
    mu[0] = 0.06
    w = pf.max_sharpe(mu, cov)
    closed = np.linalg.solve(cov, mu)
    assert np.allclose(w, closed / closed.sum(), atol=1e-8)


def test_equal_means_give_min_variance():
    _, cov, _ = _problem()
    assert np.allclose(pf.max_sharpe(np.full(len(cov), 0.03), cov), pf.min_variance(cov), atol=1e-8)


def test_no_positive_mean_falls_back_to_min_variance():
    mu, cov, _ = _problem()
    assert np.allclose(pf.max_sharpe(-np.abs(mu) - 0.01, cov), pf.min_variance(cov))


def test_min_variance_is_minimal():
    _, cov, _ = _problem()
    w = pf.min_variance(cov)
    v0 = w @ cov @ w
    for _ in range(2000):
        v = rng.dirichlet(np.ones(len(cov)))
        assert v @ cov @ v >= v0 - 1e-12


def test_bayes_stein_shrinks_towards_common_value():
    _, _, x = _problem(t=60)
    m, bs = x.mean(0), pf.bayes_stein_mean(x)
    assert np.std(bs) < np.std(m)


def test_resampled_is_reproducible_and_valid():
    mu, cov, _ = _problem()
    w1 = pf.resampled_max_sharpe(mu, cov, 120, 50, np.random.default_rng(1))
    w2 = pf.resampled_max_sharpe(mu, cov, 120, 50, np.random.default_rng(1))
    assert np.allclose(w1, w2) and w1.min() >= 0 and abs(w1.sum() - 1) < 1e-10


@pytest.mark.parametrize("n", [3, 12, 19])
def test_inverse_vol(n):
    cov = np.diag(np.linspace(0.01, 0.09, n))
    w = pf.inverse_vol(cov)
    assert abs(w.sum() - 1) < 1e-12 and np.all(np.diff(w) < 0)


def test_max_sharpe_short_respects_bounds_and_dominates_long_only():
    mu, cov, _ = _problem()
    w = pf.max_sharpe_short(mu, cov, 0.3)
    assert abs(w.sum() - 1) < 1e-9 and w.min() >= -0.3 - 1e-8
    assert pf.sharpe(mu, cov, w) >= pf.sharpe(mu, cov, pf.max_sharpe(mu, cov)) - 1e-9
