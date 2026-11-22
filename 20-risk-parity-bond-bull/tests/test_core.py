import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data as dt  # noqa: E402
import portfolio as pf  # noqa: E402
import stats as st  # noqa: E402
from backtest import era_of, load_panel, windfall_free  # noqa: E402
from config import ERAS  # noqa: E402

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- bond construction

def test_par_bond_prices_at_par():
    for y in (0.005, 0.03, 0.15):
        assert pf is not None
        assert dt.par_bond_price(y, y, 10.0) == pytest.approx(1.0, abs=1e-12)


def test_zero_yield_limit():
    assert dt.par_bond_price(0.0, 0.04, 10.0) == pytest.approx(1 + 0.04 * 10, rel=1e-9)


def test_unchanged_yield_return_is_carry():
    y = pd.Series([0.05] * 13, index=pd.period_range("2000-01", periods=13, freq="M"))
    r = dt.bond_returns_from_yields(y)
    assert np.allclose(r["income"], 0.05 / 12)
    assert np.allclose(r["price"], 0.0, atol=1e-4)  # only the tiny semi-annual compounding drift


def test_duration_sign_and_size():
    y = pd.Series([0.05, 0.06], index=pd.period_range("2000-01", periods=2, freq="M"))
    p = dt.bond_returns_from_yields(y)["price"].iloc[0]
    assert -0.08 < p < -0.07  # modified duration of a 10y par bond at 5-6% is about 7.6


def test_panel_has_no_gaps_and_sane_ranges():
    df = load_panel()
    assert str(df.index[0]) == "1926-07" and str(df.index[-1]) == "2026-09"
    assert df[["rf", "eq_x", "bond_x"]].notna().all().all()
    assert df["y10"].between(0.003, 0.17).all()
    assert df["bond_x"].abs().max() < 0.15
    # bond_x is carry + price change by construction
    assert np.allclose(df.bond_x, df.bond_carry + df.bond_price, atol=1e-7)


# ---------------------------------------------------------------- weights

def _cov(n=3):
    A = rng.normal(size=(n, n))
    return A @ A.T / n + np.eye(n) * 0.01


def test_inverse_vol_equalises_standalone_risk():
    S = _cov(4)
    w = pf.inverse_vol(S)
    assert w.sum() == pytest.approx(1)
    assert np.allclose(w * np.sqrt(np.diag(S)), (w * np.sqrt(np.diag(S)))[0])


def test_erc_equal_contributions():
    S = _cov(5)
    w = pf.erc(S)
    rc = pf.risk_contributions(w, S)
    assert np.all(w > 0) and w.sum() == pytest.approx(1)
    assert np.allclose(rc, 0.2, atol=1e-8)


def test_erc_two_assets_equals_inverse_vol():
    S = np.array([[0.04, 0.003], [0.003, 0.0049]])
    assert np.allclose(pf.erc(S), pf.inverse_vol(S), atol=1e-10)


def test_scale_to_vol():
    S = _cov(2)
    w = pf.scale_to_vol(np.array([0.5, 0.5]), S, 0.1)
    assert pf.port_vol(w, S) == pytest.approx(0.1)


def test_max_sharpe_long_only_matches_closed_form_when_interior():
    S = np.array([[0.04, 0.0], [0.0, 0.01]])
    mu = np.array([0.04, 0.02])
    w = pf.max_sharpe_long_only(mu, S)
    wc = np.linalg.solve(S, mu)
    assert np.allclose(w, wc / wc.sum(), atol=1e-5)


# ---------------------------------------------------------------- the hurdle

@pytest.mark.parametrize("rho", [-0.4, 0.0, 0.3, 0.6])
@pytest.mark.parametrize("a", [0.2, 0.29, 0.4])
def test_hurdle_equalises_sharpe(rho, a):
    r = pf.hurdle(rho, a)
    assert pf.sharpe_of_direction(1.0, r, rho) == pytest.approx(pf.sharpe_of_direction(a, r, rho), rel=1e-10)
    # above the hurdle risk parity wins, below it loses
    assert pf.sharpe_of_direction(1.0, r + 0.1, rho) > pf.sharpe_of_direction(a, r + 0.1, rho)
    assert pf.sharpe_of_direction(1.0, r - 0.1, rho) < pf.sharpe_of_direction(a, r - 0.1, rho)


def test_hurdle_rises_with_correlation():
    rs = [pf.hurdle(c, 0.29) for c in np.linspace(-0.5, 0.7, 13)]
    assert np.all(np.diff(rs) > 0)


def test_hurdle_against_simulated_portfolios():
    """The closed form agrees with Sharpe ratios computed from simulated returns."""
    n = 400_000
    rho, ss, sb = 0.2, 0.4, 0.3
    C = np.array([[1, rho], [rho, 1]])
    Z = rng.multivariate_normal([0, 0], C, size=n)
    x = np.column_stack([0.16 * Z[:, 0] + ss * 0.16, 0.06 * Z[:, 1] + sb * 0.06])
    sr = lambda w: (x @ w).mean() / (x @ w).std()  # noqa: E731
    a = 0.4 * 0.06 / (0.6 * 0.16)
    gap_sim = sr(np.array([1 / 0.16, 1 / 0.06])) - sr(np.array([0.6, 0.4]))
    A, B = np.sqrt(2 + 2 * rho), np.sqrt(1 + a * a + 2 * rho * a)
    gap_formula = (ss + sb) / A - (ss + a * sb) / B
    assert gap_sim == pytest.approx(gap_formula, abs=0.01)


# ---------------------------------------------------------------- backtest engine

def _toy(n=120):
    idx = pd.period_range("1990-01", periods=n, freq="M")
    X = pd.DataFrame(rng.normal(0.005, [0.04, 0.015], size=(n, 2)), index=idx, columns=["eq", "bond"])
    rf = pd.Series(0.002, index=idx)
    return X, rf


def test_no_look_ahead():
    X, rf = _toy()
    a = pf.backtest(X, rf, pf.strat_rp_matched, 36, 0.0005, 0.0025)
    X2 = X.copy()
    X2.iloc[80:] = rng.normal(size=X2.iloc[80:].shape)  # change the future only
    b = pf.backtest(X2, rf, pf.strat_rp_matched, 36, 0.0005, 0.0025)
    cols = ["w_eq", "w_bond"]
    assert np.allclose(a.loc[a.index[:80 - 36 + 1], cols], b.loc[b.index[:80 - 36 + 1], cols])


def test_6040_without_costs_is_exact():
    X, rf = _toy()
    r = pf.backtest(X, rf, pf.strat_6040, 36, 0.0, 0.0)
    exp = rf.iloc[36:] + 0.6 * X["eq"].iloc[36:] + 0.4 * X["bond"].iloc[36:]
    assert np.allclose(r.ret.values, exp.values)


def test_financing_charged_only_on_leverage():
    X, rf = _toy()
    r = pf.backtest(X, rf, pf.strat_rp_matched, 36, 0.0, 0.012)
    assert np.allclose(r.fund_cost, 0.001 * np.maximum(r.lev - 1, 0))
    assert (r.lev > 1).all()  # bonds are less volatile, so matching 60/40's vol needs leverage


def test_rp_matches_6040_ex_ante_vol():
    X, rf = _toy()
    a = pf.backtest(X, rf, pf.strat_rp_matched, 36, 0, 0)
    b = pf.backtest(X, rf, pf.strat_6040, 36, 0, 0)
    assert np.allclose(a.ex_ante_vol, b.ex_ante_vol)


def test_leverage_cap():
    X, rf = _toy()
    r = pf.backtest(X, rf, pf.make_rp_capped(1.2), 36, 0, 0)
    assert r.lev.max() <= 1.2 + 1e-12


# ---------------------------------------------------------------- windfall removal and eras

def test_windfall_free_removes_era_means():
    df = load_panel()
    wf = windfall_free(df, stocks=False)
    era = era_of(df.index)
    for k in ERAS:
        m = era == k
        carry_part = df.loc[m, "bond_carry"].mean()
        assert wf.loc[m, "bond_x"].mean() == pytest.approx(carry_part, abs=1e-9)
    assert np.allclose(wf.eq_x, df.eq_x)


def test_eras_partition_the_sample():
    df = load_panel()
    e = era_of(df.index)
    assert e.notna().all()


# ---------------------------------------------------------------- statistics

def test_stationary_bootstrap_block_length():
    idx = st.stationary_bootstrap_idx(2000, 200, 12, np.random.default_rng(1))
    jumps = (np.diff(idx, axis=1) != 1).mean()
    assert jumps == pytest.approx(1 / 12, abs=0.01)


def test_paired_test_detects_nothing_on_identical_series():
    x = rng.normal(0.01, 0.04, 300)
    t = st.paired_sharpe_test(x, x, 500, 12, 0)
    assert t["diff"] == 0 and t["ci_lo"] == 0 and t["ci_hi"] == 0


def test_holm():
    adj = st.holm(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_newey_west_beta_is_ols():
    x = rng.normal(size=500)
    y = 0.5 + 2 * x + rng.normal(size=500)
    r = st.newey_west_ols(y, x, 5)
    assert r["beta"] == pytest.approx(np.polyfit(x, y, 1)[0])
