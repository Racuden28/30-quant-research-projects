import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as D  # noqa: E402
import stats as S  # noqa: E402

rng = np.random.default_rng(0)


def test_ols_matches_lstsq_and_white_when_no_lags():
    X = rng.normal(size=(500, 3))
    y = 0.2 + X @ [1.0, -0.5, 0.3] + rng.normal(size=500)
    b, se, e = S.ols_nw(y, X, lags=0)
    Xc = np.column_stack([np.ones(500), X])
    assert np.allclose(b, np.linalg.lstsq(Xc, y, rcond=None)[0])
    inv = np.linalg.inv(Xc.T @ Xc)
    white = inv @ (Xc * e[:, None]).T @ (Xc * e[:, None]) @ inv
    assert np.allclose(se, np.sqrt(np.diag(white)))


def test_grs_zero_alpha_when_assets_are_factor_portfolios():
    F = rng.normal(0.005, 0.04, size=(300, 2))
    R = F @ rng.normal(size=(2, 6)) + rng.normal(0, 1e-4, size=(300, 6))
    stat, p, a, aa, _ = S.grs(R, F)
    assert aa < 1e-4 and p > 0.01


def test_grs_size_under_null():
    rej = 0
    for k in range(400):
        r = np.random.default_rng(k)
        F = r.normal(0.005, 0.04, size=(120, 3))
        R = F @ r.normal(size=(3, 10)) + r.normal(0, 0.02, size=(120, 10))
        rej += S.grs(R, F)[1] < 0.05
    assert 0.02 < rej / 400 < 0.09


def test_tangency_beats_random_portfolios():
    F = rng.normal([0.004, 0.003, 0.002], [0.04, 0.02, 0.03], size=(400, 3))
    mu, Sig = F.mean(0), np.cov(F, rowvar=False)
    w = S.tangency(mu, Sig)
    best = (w @ mu) / np.sqrt(w @ Sig @ w)
    assert np.isclose(best ** 2, S.sh2(F))
    for _ in range(500):
        v = rng.normal(size=3)
        assert (v @ mu) / np.sqrt(v @ Sig @ v) <= best + 1e-12
    wt = S.tangency(mu, Sig, target_vol=0.03)
    assert np.isclose(np.sqrt(wt @ Sig @ wt), 0.03)


def test_adjusted_squared_sharpe_is_unbiased():
    mu = np.array([0.1, 0.05, 0.0])
    true = float(mu @ mu)          # identity covariance
    est_raw, est_adj = [], []
    for k in range(3000):
        X = np.random.default_rng(k).normal(mu, 1.0, size=(60, 3))
        est_raw.append(S.sh2(X))
        est_adj.append(S.sh2_adjusted(X))
    assert np.mean(est_raw) > true + 0.03          # the naive estimate is biased up
    assert abs(np.mean(est_adj) - true) < 0.006


def test_stationary_bootstrap_shape_and_block_length():
    idx = S.stationary_bootstrap(200, 300, mean_block=6, seed=1)
    assert idx.shape == (300, 200) and idx.min() >= 0 and idx.max() < 200
    breaks = (np.diff(idx, axis=1) % 200 != 1).mean()
    assert 1 / 7.5 < breaks < 1 / 5.0


def test_holm_known_example():
    assert np.allclose(S.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])


def test_joint_wald_with_one_series_equals_squared_t():
    y = rng.normal(size=300)
    d = (np.arange(300) > 200).astype(float)
    y[d == 1] += 0.4
    b, se, _ = S.ols_nw(y, d[:, None])
    _, W, p = S.joint_shift_wald(y[:, None], d)
    assert np.isclose(W, (b[1] / se[1]) ** 2)


def test_data_sample_and_vintage():
    F = D.factors()
    assert str(F.index[0]) == "1963-07" and str(F.index[-1]) >= "2026-09"
    assert not F.isna().any().any()
    assert D.vintage("F-F_Research_Data_5_Factors_2x3.csv") == "202609"
    assert len(D.window(F, D.IS_START, D.IS_END)) == 606
    for k in D.PRIMARY_ASSETS:
        assert D.test_assets(k).shape[1] == 25


def test_oos_alphas_have_no_look_ahead():
    import analysis as A
    R = D.test_assets("25 Size-B/M").iloc[:200]
    a0 = A.oos_alphas(R, D.MODELS["FF5"])
    R2 = R.copy()
    R2.iloc[150:] = R2.iloc[150:] * 5 + 1      # change only the future
    a1 = A.oos_alphas(R2, D.MODELS["FF5"])
    pd.testing.assert_frame_equal(a0.iloc[:150], a1.iloc[:150])
    assert a0.iloc[:60].isna().all().all()     # no alpha before a full beta window exists
