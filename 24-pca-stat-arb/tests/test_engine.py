"""Unit tests for the PCA stat-arb engine. Run: python -m pytest -q tests"""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import engine as en
import theory
import data


def test_residuals_sum_to_zero_and_X_ends_at_zero():
    rng = np.random.default_rng(0)
    R = rng.standard_normal((60, 40)) * 0.02
    Q, lam, k = en.eigenportfolios(rng.standard_normal((252, 40)) * 0.02, 5)
    beta, E = en.residuals(R, Q)
    assert np.allclose(E.sum(0), 0, atol=1e-12)
    *_, XL = en.ou_fit(E)
    assert np.allclose(XL, 0, atol=1e-12)


def test_residuals_orthogonal_to_factors():
    rng = np.random.default_rng(1)
    R = rng.standard_normal((60, 30))
    Q, _, _ = en.eigenportfolios(rng.standard_normal((252, 30)), 3)
    beta, E = en.residuals(R, Q)
    F = R @ Q.T
    assert np.allclose(F.T @ E, 0, atol=1e-9)


def test_ou_fit_recovers_known_ar1():
    rng = np.random.default_rng(2)
    b_true = 0.8; n = 20000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.1 + b_true * x[t - 1] + rng.standard_normal()
    E = np.diff(np.concatenate([[0.0], x]))[:, None]
    b, kappa, m, seq, XL = en.ou_fit(E)
    assert abs(b[0] - b_true) < 0.01
    assert abs(m[0] - 0.1 / (1 - b_true)) < 0.05
    assert abs(seq[0] - 1 / np.sqrt(1 - b_true ** 2)) < 0.03


def test_filter_passes_noise():
    # the headline mechanism: >95% of pure-noise residual paths pass the 30-day filter at L=60
    p, tau, b = theory.null_pass_rate(60, 15, True, n=4000)
    assert p > 0.95
    p_rw, *_ = theory.null_pass_rate(252, 0, False, n=4000)
    assert p_rw < 0.4


def test_s_score_is_linear_kernel():
    c_unc, c = theory.kernel_check(n=2000)
    assert c_unc > 0.99


def test_mp_edge():
    assert en.mp_edge(100, 400) == pytest.approx(2.25)


def test_eigenportfolio_factor_returns_have_eigenvalue_variance():
    rng = np.random.default_rng(3)
    Rw = rng.standard_normal((500, 20)) @ np.diag(np.linspace(0.01, 0.05, 20))
    Q, lam, k = en.eigenportfolios(Rw, 4)
    F = Rw @ Q.T
    assert np.allclose(F.var(0, ddof=1), lam[:4], rtol=1e-8)


def test_band_rules():
    cfg = en.Config()
    pos = np.array([0, 0, 1, 1, -1, -1, 0], dtype=np.int8)
    s = np.array([-1.3, 1.3, -0.6, -0.4, 0.8, 0.7, np.nan])
    allowed = np.ones(7, bool)
    new = en.step_positions(pos, s, allowed, cfg)
    assert new.tolist() == [1, -1, 1, 0, -1, 0, 0]
    # filter blocks new entries but never forces an exit
    new2 = en.step_positions(pos, s, np.zeros(7, bool), cfg)
    assert new2.tolist() == [0, 0, 1, 0, -1, 0, 0]


def test_residual_portfolio_is_factor_neutral():
    rng = np.random.default_rng(4)
    Rw = rng.standard_normal((252, 25)) * 0.02
    Q, _, _ = en.eigenportfolios(Rw, 3)
    beta, E = en.residuals(Rw[-60:], Q)
    H = np.eye(25) - beta.T @ Q
    # each residual portfolio has zero in-window beta to every factor
    F = Rw[-60:] @ Q.T
    P = Rw[-60:] @ H.T
    Xd = np.column_stack([np.ones(60), F])
    coef = np.linalg.lstsq(Xd, P, rcond=None)[0]
    assert np.allclose(coef[1:], 0, atol=1e-8)


def test_pnl_timing_no_lookahead():
    idx = pd.date_range("2020-01-01", periods=5)
    ret = pd.DataFrame({"a": [0.0, 0.1, 0.2, 0.3, 0.4]}, index=idx)
    w = pd.DataFrame({"a": [0.0, 1.0, 0.0, 0.0, 0.0]}, index=idx)    # set at close of day 1
    p0 = en.pnl(w, ret, delay=0)
    p1 = en.pnl(w, ret, delay=1)
    assert p0["gross"].tolist() == pytest.approx([0, 0, 0.2, 0, 0])   # earns day 2
    assert p1["gross"].tolist() == pytest.approx([0, 0, 0, 0.3, 0])   # earns day 3
    # cost charged once on entry and once on exit
    pc = en.pnl(w, ret, delay=0, cost_bps=10)
    assert (p0["gross"] - pc["net"]).sum() == pytest.approx(2 * 10e-4)


def test_block_placebos_keep_cross_section_and_kill_serial_dependence():
    rng = np.random.default_rng(5)
    T = 2000
    idx = pd.bdate_range("2000-01-03", periods=T)
    f = rng.standard_normal(T)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.5 * x[t - 1] + rng.standard_normal()
    ret = pd.DataFrame({"a": f + 0.1 * x, "b": f}, index=idx)
    liq = ret.abs()
    pr, pl = en.placebo(ret, liq, seed=0, method="permute")
    # every placebo day is one real day: cross-sectional pairs are kept intact
    pairs_r = set(map(tuple, np.round(ret.to_numpy(), 12)))
    pairs_p = set(map(tuple, np.round(pr.to_numpy(), 12)))
    assert pairs_p <= pairs_r
    # serial correlation of the residual a-b vanishes (real 0.5)
    d = (pr["a"] - pr["b"]).to_numpy()
    assert abs(np.corrcoef(d[1:], d[:-1])[0, 1]) < 0.08
    # liquidity moved with its row
    assert np.allclose(pl.to_numpy(), pr.abs().to_numpy())


def test_placebo_leaves_partial_assets_in_place():
    idx = pd.bdate_range("2000-01-03", periods=130)
    a = np.arange(130, dtype=float); b = a.copy(); b[:70] = np.nan
    ret = pd.DataFrame({"a": a, "b": b}, index=idx)
    pr, _ = en.placebo(ret, ret.abs(), seed=1, method="bootstrap")
    assert np.isnan(pr["b"].to_numpy()[:70]).all()


def test_crypto_exclusions():
    assert data.crypto_exclusion("USDCUSDT", "2018-12-01") == "stablecoin/fiat/gold"
    assert data.crypto_exclusion("ETHBULLUSDT", "2020-01-01") == "leveraged token"
    assert data.crypto_exclusion("JUPUSDT", "2024-01-31") is None
    assert data.crypto_exclusion("NVDABUSDT", "2026-06-12") == "tokenised stock"
    assert data.crypto_exclusion("YBUSDT", "2025-10-15") is None



def test_permutation_placebo_leaks_real_reversal_bootstrap_does_not():
    """Unit test of the flaw found in the first placebo. A within-quarter permutation keeps each
    quarter's total. If the real series mean-reverts, quarterly totals are small, so in the
    permuted series the next day offsets the earlier days of the quarter: the reversal the
    placebo should destroy leaks back in, spread over the whole quarter. Drawing days with
    replacement keeps no total and leaks nothing."""
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2000-01-03", periods=63 * 400)
    e = rng.standard_normal(len(idx) + 1)
    x = pd.DataFrame({"a": e[1:] - 0.9 * e[:-1]}, index=idx)      # strongly mean-reverting MA(1)
    out = {}
    for method in ("permute", "bootstrap"):
        corrs = []
        for seed in range(3):
            p, _ = en.placebo(x, x.abs(), seed=seed, method=method)
            g = [q.to_numpy() for _, q in p["a"].groupby(p.index.to_period("Q")) if len(q) > 41]
            past = np.array([q[:40].sum() for q in g]); nxt = np.array([q[40] for q in g])
            corrs.append(np.corrcoef(past, nxt)[0, 1])
        out[method] = np.mean(corrs)
    assert out["permute"] < -0.08          # theory about -0.17 for a zero-sum quarter
    assert abs(out["bootstrap"]) < 0.08


def test_signflip_placebo_keeps_daily_covariance_and_kills_reversal():
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2000-01-03", periods=5000)
    e = rng.standard_normal((5001, 2))
    x = pd.DataFrame(e[1:] - 0.5 * e[:-1], index=idx, columns=["a", "b"])
    x.iloc[:100, 1] = np.nan                                     # a late listing stays in place
    p, l = en.placebo(x, x.abs(), seed=0, method="signflip")
    assert np.isnan(p["b"].to_numpy()[:100]).all()
    A, B = x.to_numpy(), p.to_numpy()
    ok = np.isfinite(A).all(1)
    assert np.allclose(np.einsum("ti,tj->tij", A[ok], A[ok]), np.einsum("ti,tj->tij", B[ok], B[ok]))
    a = p["a"].to_numpy()
    assert np.corrcoef(x["a"].to_numpy()[1:], x["a"].to_numpy()[:-1])[0, 1] < -0.3
    assert abs(np.corrcoef(a[1:], a[:-1])[0, 1]) < 0.04
    assert np.allclose(l.to_numpy(), x.abs().to_numpy(), equal_nan=True)
