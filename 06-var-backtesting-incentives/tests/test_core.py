"""Unit tests: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtests import (BASEL_PLUS, FRTB_PLUS, acerbi_szekely_z2, capital_path, christoffersen_ind,  # noqa: E402
                       dq_test, fz0_loss, kupiec, tick_loss, trailing_exceptions)
from models import evt_tail, hs, rolling_forecasts, std_quantiles, tail_from_ppf  # noqa: E402
from theory import break_even_schedule, exceed_prob  # noqa: E402

RNG = np.random.default_rng(0)


def test_kupiec_zero_at_nominal_rate():
    h = np.zeros(1000, bool); h[::100] = True
    lr, p = kupiec(h, 0.01)
    assert lr == pytest.approx(0, abs=1e-9) and p == pytest.approx(1)


def test_kupiec_rejects_double_rate():
    h = np.zeros(2500, bool); h[::50] = True
    assert kupiec(h, 0.01)[1] < 1e-4


def test_christoffersen_detects_clusters():
    h = np.zeros(2000, bool)
    for s in range(100, 2000, 200):
        h[s:s + 3] = True
    assert christoffersen_ind(h)[1] < 1e-6


def test_dq_size_under_iid_hits():
    ps = [dq_test(RNG.random(1500) < 0.01, np.ones(1500) + RNG.random(1500), 0.01)[1] for _ in range(200)]
    assert 0.0 < np.mean(np.array(ps) < 0.05) < 0.15


def test_tables():
    assert BASEL_PLUS[:5].sum() == 0 and BASEL_PLUS[5] == 0.40 and BASEL_PLUS[9] == 0.85 and BASEL_PLUS[10] == 1.0
    assert FRTB_PLUS[5] == 0.20 and FRTB_PLUS[12] == 0.50


def test_basel_table_is_gaussian_break_even():
    """Headline fact: under normality the 1996 plus-factor table offsets the capital saved by shading."""
    be = break_even_schedule()
    g = be[np.isinf(be.nu) & be.n_exc.between(5, 10)]
    assert np.max(np.abs(g.plus_required - g.plus_basel)) < 0.04
    t3 = be[(be.nu == 3.0) & (be.n_exc == 10)].iloc[0]
    assert t3.plus_required > 2 * t3.plus_basel


def test_normal_es_quadrature():
    a = 0.025
    es = tail_from_ppf(stats.norm.ppf, a)
    assert es == pytest.approx(-stats.norm.pdf(stats.norm.ppf(a)) / a, rel=1e-3)


def test_std_t_quantiles():
    nu = 5.0
    q, q2, es = std_quantiles(("t", [nu]))
    sc = np.sqrt((nu - 2) / nu)
    assert q == pytest.approx(stats.t.ppf(0.01, nu) * sc, rel=1e-4)
    assert es < q2 < 0


def test_evt_recovers_t_tail():
    z = stats.t.rvs(4, size=200000, random_state=1) / np.sqrt(2)
    q, q2, es = evt_tail(z)
    assert q == pytest.approx(np.quantile(z, 0.01), rel=0.03)
    assert es == pytest.approx(z[z <= np.quantile(z, 0.025)].mean(), rel=0.05)


def test_hs_orders():
    v, v2, e = hs(RNG.standard_normal(500))
    assert e > v2 and v > v2


def test_fz0_and_tick_are_minimised_by_truth():
    y = RNG.standard_normal(400000)
    a = 0.025
    v, e = -stats.norm.ppf(a), stats.norm.pdf(stats.norm.ppf(a)) / a
    base = fz0_loss(y, v, e, a).mean()
    for lam in (0.85, 0.95, 1.05, 1.15):
        assert fz0_loss(y, lam * v, lam * e, a).mean() > base
        assert tick_loss(y, lam * v, a).mean() > tick_loss(y, v, a).mean()


def test_z2_centred_under_h0():
    y = RNG.standard_normal(200000)
    a = 0.025
    v, e = -stats.norm.ppf(a), stats.norm.pdf(stats.norm.ppf(a)) / a
    z2, p = acerbi_szekely_z2(y, np.full_like(y, v), np.full_like(y, e), a)
    assert abs(z2) < 0.05
    assert acerbi_szekely_z2(y, np.full_like(y, 0.8 * v), np.full_like(y, 0.8 * e), a)[1] < 1e-6


def test_trailing_exceptions_matches_loop():
    h = (RNG.random((3, 900)) < 0.02).astype(float)
    N = trailing_exceptions(h)
    for t in (0, 1, 249, 250, 251, 899):
        assert np.allclose(N[:, t], h[:, max(t - 250, 0):t].sum(axis=1))


def test_capital_honest_iid_normal_has_no_saving_from_mild_shading():
    """With Gaussian returns, shading by 10% saves under 5%: the table is (nearly) break-even."""
    r = RNG.standard_normal(60000) * 0.01
    v = np.full_like(r, 0.01 * 2.326)
    lams = np.array([0.9, 1.0])
    out = capital_path(r, v, np.sqrt(10) * v, lams, 3.0, BASEL_PLUS)
    assert out["charge"][0] / out["charge"][1] > 0.95


def test_capital_t3_shading_pays():
    r = stats.t.rvs(3, size=60000, random_state=2) * 0.01
    v = np.full_like(r, -stats.t.ppf(0.01, 3) * 0.01)
    out = capital_path(r, v, np.sqrt(10) * v, np.array([0.85, 1.0]), 3.0, BASEL_PLUS)
    assert out["charge"][0] / out["charge"][1] < 0.92


def test_exceed_prob_identity():
    assert exceed_prob(np.array([1.0]), 4.0)[0] == pytest.approx(0.01)


def test_no_look_ahead():
    r = RNG.standard_normal(700) * 0.01
    a = rolling_forecasts(r, 500)
    r2 = r.copy(); r2[650:] *= 5
    b = rolling_forecasts(r2, 500)
    for m in a:
        assert np.allclose(a[m][:151], b[m][:151])
