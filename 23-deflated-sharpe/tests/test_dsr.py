import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dsr as D  # noqa: E402
from zoo import zoo_a, zoo_b  # noqa: E402


def test_expected_max_matches_simulation():
    rng = np.random.default_rng(0)
    for n in (10, 100, 1000):
        sim = rng.standard_normal((4000, n)).max(1).mean()
        assert abs(D.expected_max_sr(n, 1.0) - sim) / sim < 0.03


def test_sr_standard_error_normal_and_fat_tails():
    rng = np.random.default_rng(1)
    T, sr = 500, 0.1
    x = sr + rng.standard_normal((6000, T))
    est = x.mean(1) / x.std(1, ddof=1)
    assert abs(est.std() / D.sr_se(sr, T) - 1) < 0.05
    t = rng.standard_t(5, (6000, T)) / np.sqrt(5 / 3) + sr  # raw kurtosis 9
    est = t.mean(1) / t.std(1, ddof=1)
    assert abs(est.std() / D.sr_se(sr, T, 0.0, 9.0) - 1) < 0.06


def test_textbook_dsr_is_far_below_nominal_size_with_iid_trials():
    """The paper's first result in miniature: DSR > 0.95 is not a 5% test."""
    rng = np.random.default_rng(2)
    T, N = 1000, 100
    sr = rng.standard_normal((20000, N)) / np.sqrt(T)
    w = sr.max(1)
    d = D.dsr(w, T, 0, 3, N, sr.var(1, ddof=1))[0]
    assert np.mean(d > 0.95) < 0.01
    sidak = D.psr(w, T) > 0.95 ** (1 / N)
    assert abs(np.mean(sidak) - 0.05) < 0.01


def test_neff_limits():
    lam_id = np.ones(50)
    assert D.neff_nyholt(lam_id) == 50 and D.neff_liji(lam_id) == 50
    assert abs(D.neff_participation(lam_id) - 50) < 1e-9
    lam_one = np.r_[50.0, np.zeros(49)]
    assert D.neff_liji(lam_one) == 1 and D.neff_nyholt(lam_one) == 1
    assert abs(D.neff_participation(lam_one) - 1) < 1e-9


def test_clusters_recover_blocks():
    rng = np.random.default_rng(3)
    f = rng.standard_normal((2000, 3))
    R = np.repeat(f, 10, axis=1) + 0.5 * rng.standard_normal((2000, 30))
    k, lab = D.cluster_trials(np.corrcoef(R, rowvar=False))
    assert k == 3
    assert len({tuple(np.unique(lab[i * 10:(i + 1) * 10])) for i in range(3)}) == 3


def test_min_track_record_inverts_psr():
    sr, star = 0.08, 0.02
    n = float(D.min_track_record(sr, star))
    assert abs(D.psr(sr, n, sr_star=star) - 0.95) < 1e-3


def _toy_index(seed, T=1500):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=T)
    return pd.DataFrame({c: 0.01 * rng.standard_normal(T) + 0.0003 for c in
                         ["GSPC", "IXIC", "GSPTSE", "FTSE", "GDAXI", "N225", "HSI", "AXJO", "BVSP", "FCHI"]},
                        index=idx)


def test_zoo_a_has_no_look_ahead():
    X = _toy_index(4)
    g1, _, _, _ = zoo_a(X)
    X2 = X.copy()
    X2.iloc[1000:] = X2.iloc[1000:].to_numpy()[::-1]  # rewrite the future
    g2, _, _, _ = zoo_a(X2)
    assert np.allclose(g1[:1000], g2[:1000])


def test_zoo_b_has_no_look_ahead():
    rng = np.random.default_rng(5)
    R = 0.05 * rng.standard_normal((200, 20))
    g1, n1, _ = zoo_b(R)
    R2 = R.copy()
    R2[150:] = rng.standard_normal((50, 20))
    g2, n2, _ = zoo_b(R2)
    assert np.allclose(g1[:150], g2[:150]) and np.allclose(n1[:150], n2[:150])
    assert not np.allclose(g1[150], g2[150])  # the first changed month does move the P&L


def test_sign_flip_removes_all_predictability():
    """Averaged over sign draws, every rule's mean return is zero, even on a trending series."""
    X = _toy_index(6, T=800) + 0.002  # strong drift
    rng = np.random.default_rng(7)
    acc = 0
    for _ in range(300):
        s = rng.choice([-1.0, 1.0], size=len(X))
        g, _, _, _ = zoo_a(X.mul(s, axis=0))
        acc = acc + g[700:].mean(0)
    se = 0.01 / np.sqrt(100 * 300)
    assert np.max(np.abs(acc / 300)) < 6 * se
