import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backtest import forward_sum, refit_dates, trailing_mean  # noqa: E402
from evaluate import holm, qlike  # noqa: E402
from models import (CGARCH, GARCH, GJRt, curve_half_life, half_life,  # noqa: E402
                    simulate_garch, start_var)


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(0)
    z = rng.standard_t(6, 12_500) / np.sqrt(6 / 4)
    return simulate_garch(0.02, 0.08, 0.90, z)


def test_garch_recovers_parameters(sim):
    p, _ = GARCH.fit(sim)
    assert abs(GARCH.persistence(p) - 0.98) < 0.01
    assert abs(p[1] - 0.08) < 0.03


def test_garch_matches_arch_package(sim):
    arch = pytest.importorskip("arch")
    a = arch.arch_model(sim, mean="Zero", vol="GARCH").fit(disp="off")
    p, ll = GARCH.fit(sim)
    assert np.allclose(p, a.params.values, rtol=0.05, atol=2e-3)
    assert abs(ll - a.loglikelihood) < 5


def test_gjr_matches_arch_package(sim):
    arch = pytest.importorskip("arch")
    a = arch.arch_model(sim, mean="Zero", vol="GARCH", o=1, dist="t").fit(disp="off")
    _, ll = GJRt.fit(sim)
    assert abs(ll - a.loglikelihood) < 5


def test_filter_has_no_look_ahead(sim):
    p = np.array([0.02, 0.08, 0.90])
    s0 = start_var(sim)
    full = GARCH.filter(sim, p, s0)["s"]
    cut = GARCH.filter(sim[:5000], p, s0)["s"]
    assert np.allclose(full[:5000], cut)
    pc = np.array([1.0, 0.995, 0.02, 0.07, 0.85])
    assert np.allclose(CGARCH.filter(sim, pc, s0)["s"][:5000], CGARCH.filter(sim[:5000], pc, s0)["s"])


@pytest.mark.parametrize("M,p", [(GARCH, np.array([0.02, 0.08, 0.9])),
                                 (CGARCH, np.array([1.0, 0.995, 0.02, 0.07, 0.85]))])
def test_cumulative_forecast_is_sum_of_path(sim, M, p):
    st = M.filter(sim, p, start_var(sim))
    k = np.arange(1, 67)
    assert np.allclose(M.path(st, p, k).sum(1), M.cum_forecast(st, p, 66))


def test_cgarch_path_matches_recursion():
    """Iterate Engle-Lee expectations by hand: E[r^2] = s, so q and s-q evolve deterministically."""
    w, rho, phi, a, b = 1.0, 0.99, 0.03, 0.08, 0.85
    s, q = 3.0, 1.5
    st = {"s": np.array([s]), "q": np.array([q])}
    path = CGARCH.path(st, np.array([w, rho, phi, a, b]), np.arange(1, 30))[0]
    hand = []
    for _ in range(29):
        hand.append(s)
        qn = w + rho * (q - w)            # phi * E[r^2 - s] = 0
        s = qn + (a + b) * (s - q)        # alpha * E[r^2 - q] + beta (s - q) = (a+b)(s-q)
        q = qn
    assert np.allclose(path, hand)


def test_half_life_helpers():
    assert half_life(0.5 ** (1 / 10)) == pytest.approx(10)
    assert np.isinf(half_life(1.0))
    k = np.arange(1, 101)
    assert curve_half_life(0.5 ** ((k - 1) / 20), k) == pytest.approx(21, abs=0.05)
    assert np.isinf(curve_half_life(np.ones(100), k))


def test_cgarch_decay_normalized_and_two_speed():
    p = np.array([1.0, 0.999, 0.02, 0.08, 0.85])
    d = CGARCH.decay(p, np.arange(1, 500))
    assert d[0] == pytest.approx(1)
    # fast drop first, then a slow tail sitting near the long-run share phi / (phi + alpha)
    assert d[30] < 0.4 and d[300] > 0.12


def test_window_helpers_alignment():
    x = np.arange(10.0)
    assert forward_sum(x, 3)[0] == 1 + 2 + 3   # strictly after t
    assert np.isnan(forward_sum(x, 3)[-3])
    assert trailing_mean(x, 3)[2] == 1.0       # includes t
    idx = pd.bdate_range("2000-01-03", periods=3000)
    for p in refit_dates(idx):
        assert idx[p].year != idx[p - 1].year and p >= 2520


def test_qlike_and_holm():
    assert qlike(np.array([2.0]), np.array([2.0]))[0] == pytest.approx(0)
    assert qlike(np.array([1.0]), np.array([2.0]))[0] > 0
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
