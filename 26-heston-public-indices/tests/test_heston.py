"""Tests for the Heston engine and the calibration steps (run: python -m pytest -q tests)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import ncx2, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import calibrate as C  # noqa: E402
import heston as H  # noqa: E402
from bs import black_price  # noqa: E402

P = (0.03, 3.0, 0.04, 0.6, -0.7)


def test_cf_is_a_martingale_and_normalised():
    v0, k, th, s, r = P
    assert abs(np.exp(H.log_cf(0.0, 0.5, *P)) - 1) < 1e-14
    # E[e^X] = 1  <=>  phi(-i) = 1
    assert abs(np.exp(H.log_cf(-1j, 0.5, *P)) - 1) < 1e-12


def test_cumulants_match_closed_form_mean_and_radius_invariance():
    v0, k, th, s, r = P
    T = 0.25
    c = H.cumulants(T, *P)
    ev = th * T + (v0 - th) * (1 - np.exp(-k * T)) / k  # E[int v]
    assert abs(c[0] + 0.5 * ev) < 1e-12  # c1 = -E[int v]/2
    for rr in (0.2, 1.0):
        assert np.allclose(H._cumulants(T, *P, 4, rr, 64), c, rtol=1e-8, atol=1e-14)


def test_zero_volvol_limit_is_black_scholes():
    v0 = th = 0.04
    p = (v0, 2.0, th, 1e-3, 0.0)  # rho = 0: the gap to BS is O(sigma^2)
    k = np.array([-0.2, -0.05, 0.0, 0.1])
    T = 0.5
    got = H.otm_price_over_F(k, T, p)
    K = np.exp(k)
    bs = black_price(1.0, K, T, 0.2, 1.0, call=(k >= 0))
    assert np.allclose(got, bs, atol=1e-8)
    assert abs(H.skewness(T, *p)) < 1e-3
    assert abs(H.skew_index(*p) - 100) < 0.01


def test_cos_matches_monte_carlo():
    x, _ = H.simulate(P, H.T30, 200, 400_000, seed=7)
    k = np.array([-0.15, -0.05, 0.0, 0.04])
    mc = np.array([np.mean(np.maximum(np.exp(q) - np.exp(x), 0)) if q < 0 else
                   np.mean(np.maximum(np.exp(x) - np.exp(q), 0)) for q in k])
    se = np.array([np.std(np.maximum(np.exp(q) - np.exp(x), 0)) if q < 0 else
                   np.std(np.maximum(np.exp(x) - np.exp(q), 0)) for q in k]) / np.sqrt(len(x))
    got = H.otm_price_over_F(k, H.T30, P)
    assert np.all(np.abs(got - mc) < 4 * se + 2e-5)


def test_skewness_matches_monte_carlo():
    x, _ = H.simulate(P, H.T30, 300, 1_000_000, seed=11)
    m = x - x.mean()
    s_mc = np.mean(m**3) / np.mean(m**2) ** 1.5
    assert abs(s_mc - H.skewness(H.T30, *P)) < 0.02


def test_vix_formula_is_expected_average_variance():
    v0, k, th, s, r = P
    for tau in H.TENORS.values():
        ev = quad(lambda t: th + (v0 - th) * np.exp(-k * t), 0, tau)[0] / tau
        assert abs(H.vix_model(v0, k, th, tau) ** 2 - ev) < 1e-12


def _vvix_quadrature(v0, k, th, s, T=H.T30):
    b = H.term_b(k, T)
    a = th * (1 - b)
    c = s**2 * (1 - np.exp(-k * T)) / (4 * k)
    df = 4 * k * th / s**2
    nc = 4 * k * np.exp(-k * T) * v0 / (s**2 * (1 - np.exp(-k * T)))
    f = lambda y: ncx2.pdf(y, df, nc)
    es = quad(lambda y: np.sqrt(a + b * c * y) * f(y), 0, np.inf, limit=1000, epsabs=1e-13)[0]
    el = quad(lambda y: np.log(a + b * c * y) * f(y), 0, np.inf, limit=1000, epsabs=1e-13)[0]
    return np.sqrt(2 * (np.log(es) - 0.5 * el) / T)


@pytest.mark.parametrize("p", [(0.03, 3.0, 0.04, 0.6), (0.02, 1.5, 0.05, 0.3), (0.09, 8.0, 0.06, 1.2)])
def test_vvix_laplace_matches_noncentral_chi2_quadrature(p):
    assert abs(H.vvix_model(*p) - _vvix_quadrature(*p)) < 2e-4


def test_vvix_does_not_depend_on_rho_and_futures_are_martingale_consistent():
    v0, k, th, s, _ = P
    x, vT = H.simulate((v0, k, th, s, -0.9), H.T30, 300, 400_000, seed=5)
    b = H.term_b(k, H.T30)
    vixT = np.sqrt(th * (1 - b) + b * vT)
    assert abs(vixT.mean() - H.vix_futures(v0, k, th, s, H.T30)) < 3 * vixT.std() / np.sqrt(len(vixT)) + 1e-4


def test_term_fit_recovers_parameters_exactly():
    v0, k, th = 0.05, 2.5, 0.03
    taus = np.array(list(H.TENORS.values()))
    vols = H.vix_model(v0, k, th, taus)
    fv0, fk, fth, rmse = C.fit_term(vols, taus)
    assert rmse < 1e-6
    assert abs(fv0 - v0) < 1e-6 and abs(fk - k) / k < 1e-3 and abs(fth - th) < 1e-6


def test_triangular_calibration_round_trip():
    p = (0.04, 4.0, 0.05, 0.7, -0.6)
    row = {n: 100 * float(H.vix_model(p[0], p[1], p[2], t)) for n, t in H.TENORS.items()}
    row["SKEW"] = H.skew_index(*p)
    row["VVIX"] = 100 * H.vvix_model(*p[:4])
    out = C.calibrate_day(row)
    assert out["sigma_status"] == "ok" and out["rho_status"] == "ok"
    assert abs(out["sigma"] - p[3]) < 1e-4 and abs(out["rho"] - p[4]) < 1e-4
    assert out["sigma_skew_min"] < out["sigma"]  # rho = -1 needs less vol of vol than rho = -0.6


def test_infeasible_day_is_flagged():
    p = (0.04, 4.0, 0.05, 0.5, -0.7)
    row = {n: 100 * float(H.vix_model(p[0], p[1], p[2], t)) for n, t in H.TENORS.items()}
    row["VVIX"] = 100 * H.vvix_model(*p[:4])
    row["SKEW"] = 150.0  # far more skew than sigma = 0.5 can give
    out = C.calibrate_day(row)
    assert out["rho_status"] == "infeasible"
    assert out["sigma_skew_min"] > out["sigma"]
    assert 0 < out["nonheston_share"] < 1


def test_cboe_moment_formulas_on_lognormal_strip():
    """With Black-Scholes prices on a dense strip, the CBOE variance and skewness formulas must return
    sigma^2 and the lognormal skewness of ln(S_T/F) (which is 0)."""
    from chain import cboe_moments
    F, T, sig = 100.0, 30 / 365, 0.2
    K = np.arange(40.0, 200.0, 0.25)
    K0 = 100.0
    Q = np.where(K < K0, black_price(F, K, T, sig, 1.0, call=False), black_price(F, K, T, sig, 1.0, call=True))
    Q[K == K0] = 0.5 * (black_price(F, K0, T, sig, 1.0, False) + black_price(F, K0, T, sig, 1.0, True))
    var, S = cboe_moments(K, Q, K0, F, T)
    assert abs(np.sqrt(var) - sig) < 2e-4
    assert abs(S) < 0.01
