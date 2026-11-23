"""Part III: how much does the exercise rule need to know?

(a) Lattice experiment (exact, no Monte Carlo). On the true CRR lattice (sigma, r, q) evaluate the policy
    'exercise when S crosses the boundary computed with the WRONG sigma' (or the wrong r), and two heuristic
    rules. Loss is reported as a share of the true early-exercise premium (EEP).
(b) GARCH world. Risk-neutral GJR-GARCH paths fitted to the S&P 500 1928-2026. Compare
    hold-to-expiry, a static constant-volatility tree boundary, a dynamic tree boundary that is re-read every
    day at the current GARCH term volatility, and Longstaff-Schwartz (out-of-sample, fitted on separate paths).

Outputs: results/misspec_sigma.csv, results/misspec_rate.csv, results/heuristics.csv,
         results/garch_params.csv, results/garch_policies.csv, results/lsm_gbm_validation.csv
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

import garch
import trees as tr

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_LAT = 2000

# Three historical "regimes" for the lattice experiment (typical values of r, q, sigma; see Part II)
CASES = {
    "1981 put (r 14%, q 5%)": dict(is_call=False, r=0.14, q=0.05, sigma=0.15),
    "2023 put (r 5%, q 1.5%)": dict(is_call=False, r=0.05, q=0.015, sigma=0.15),
    "1942 call (r 0.3%, q 6%)": dict(is_call=True, r=0.003, q=0.06, sigma=0.15),
}


@njit(cache=True)
def policy_naive_bs(S, K, T, r, q, sigma, is_call, n):
    """Heuristic: exercise as soon as intrinsic value exceeds the European BS value of what is left."""
    u, d, p = tr._crr_params(T, r, q, sigma, n)
    dt = T / n
    disc = math.exp(-r * dt)
    sgn = 1.0 if is_call else -1.0
    v = np.empty(n + 1)
    ratio = u / d
    s = S * d ** n
    for j in range(n + 1):
        v[j] = max(sgn * (s - K), 0.0)
        s *= ratio
    for i in range(n - 1, -1, -1):
        s = S * d ** i
        tau = (n - i) * dt
        for j in range(i + 1):
            c = disc * (p * v[j + 1] + (1.0 - p) * v[j])
            ex = sgn * (s - K)
            if ex > 0.0 and ex > tr.bs_price(s, K, tau, r, q, sigma, is_call):
                c = ex
            v[j] = c
            s *= ratio
    return v[0]


def lattice_experiment():
    m_grid = np.exp(np.linspace(np.log(0.5), np.log(2.0), 41))
    rows, rrows, hrows = [], [], []
    for name, c in CASES.items():
        for mny in [0.9, 1.0, 1.1]:   # K / S
            S, K, T = 100.0, 100.0 * mny, 1.0
            r, q, sig, call = c["r"], c["q"], c["sigma"], c["is_call"]
            if call:
                K = 100.0 / mny     # mirror: K/S = 1.11, 1.0, 0.91 -> OTM, ATM, ITM call
            _, b_true, v_am = tr.crr_boundary(S, K, T, r, q, sig, call, N_LAT)
            v_eu = tr.crr(S, K, T, r, q, sig, call, False, N_LAT)
            eep = v_am - v_eu
            for m in m_grid:
                _, b, _ = tr.crr_boundary(S, K, T, r, q, sig * m, call, N_LAT)
                v = tr.crr_policy_value(S, K, T, r, q, sig, call, N_LAT, b)
                rows.append(dict(case=name, moneyness=mny, sigma_mult=m, v_am=v_am, v_eu=v_eu, eep=eep,
                                 v_policy=v, loss_share=(v_am - v) / eep))
            for rm in np.linspace(0.0, 2.0, 41):
                # scale the carry term that drives exercise: r for puts, q for calls
                rr_, qq_ = (r, q * rm) if call else (r * rm, q)
                _, b, _ = tr.crr_boundary(S, K, T, rr_, qq_, sig, call, N_LAT)
                v = tr.crr_policy_value(S, K, T, r, q, sig, call, N_LAT, b)
                rrows.append(dict(case=name, moneyness=mny, rate_mult=rm, eep=eep, loss_share=(v_am - v) / eep))
            v_naive = policy_naive_bs(S, K, T, r, q, sig, call, N_LAT)
            # 'exercise as soon as in the money' and 'never' (European)
            b_itm = np.full(N_LAT, K)
            v_itm = tr.crr_policy_value(S, K, T, r, q, sig, call, N_LAT, b_itm)
            hrows.append(dict(case=name, moneyness=mny, v_am=v_am, v_eu=v_eu, eep=eep, eep_share=eep / v_am,
                              naive_bs_loss=(v_am - v_naive) / eep, itm_loss=(v_am - v_itm) / eep))
    pd.DataFrame(rows).to_csv(RES / "misspec_sigma.csv", index=False)
    pd.DataFrame(rrows).to_csv(RES / "misspec_rate.csv", index=False)
    h = pd.DataFrame(hrows)
    h.to_csv(RES / "heuristics.csv", index=False)
    print(h.round(4).to_string())


# ------------------------------------------------------------------------------------------------
# (b) GARCH world
# ------------------------------------------------------------------------------------------------
STEPS = 63          # 3-month option, daily exercise opportunities
SUB = 10            # tree steps per day for the boundary tables
N_TRAIN, N_EVAL = 200_000, 600_000
SIG_GRID = np.exp(np.linspace(np.log(0.03), np.log(1.5), 80))


def boundary_table(K, r, q, is_call):
    """b[k, tau]: critical price for sigma = SIG_GRID[k] and tau = 1..STEPS days to expiry."""
    T = STEPS / 252.0
    tab = np.full((len(SIG_GRID), STEPS + 1), np.nan)
    for k, s in enumerate(SIG_GRID):
        _, b, _ = tr.crr_boundary(100.0, K, T, r, q, s, is_call, STEPS * SUB)
        # b[i] at time i*dt; tau days remaining = STEPS - i/SUB
        for tau in range(1, STEPS + 1):
            i = (STEPS - tau) * SUB
            tab[k, tau] = b[i]
    return tab


def interp_boundary(tab, sig, tau):
    """Vectorised log-sigma interpolation of the boundary table."""
    x = np.log(np.clip(sig, SIG_GRID[0], SIG_GRID[-1]))
    g = np.log(SIG_GRID)
    k = np.clip(np.searchsorted(g, x) - 1, 0, len(g) - 2)
    w = (x - g[k]) / (g[k + 1] - g[k])
    b0, b1 = tab[k, tau], tab[k + 1, tau]
    out = (1 - w) * b0 + w * b1
    # where one side is NaN (no exercise region), treat as never-exercise
    return np.where(np.isnan(b0) | np.isnan(b1), np.nan, out)


def term_vol(h, tau, params):
    v, _, _ = garch.expected_avg_var(h, tau, params)
    return np.sqrt(252.0 * np.maximum(v, 1e-12))


def run_policies(S, H, K, r, q, is_call, params, tab, sig_static, lsm_betas):
    n, m = S.shape
    disc = np.exp(-r / 252.0)
    sgn = 1.0 if is_call else -1.0
    intr = np.maximum(sgn * (S - K), 0.0)
    hold = intr[:, -1] * disc ** STEPS

    diag = {}

    def first_exercise(mask, key=None):
        """mask (n, STEPS+1) of exercise decisions; cash flow at first True, else at expiry."""
        mask = mask.copy()
        mask[:, -1] = True
        t = mask.argmax(axis=1)
        if key is not None:
            early = t < STEPS
            diag[key + "_early_share"] = early.mean()
            if early.any():
                te = t[early]
                diag[key + "_mean_exercise_day"] = te.mean()
                diag[key + "_termvol_at_exercise"] = np.mean(term_vol(H[early, te], STEPS - te, params))
        return intr[np.arange(n), t] * disc ** t

    # static constant-vol boundary at the initial term vol
    ex_static = np.zeros((n, m), bool)
    for t in range(STEPS):
        tau = STEPS - t
        b = interp_boundary(tab, np.array([sig_static]), tau)[0]
        if not np.isnan(b):
            ex_static[:, t] = (S[:, t] >= b) if is_call else (S[:, t] <= b)
    ex_static &= intr > 0
    # dynamic: boundary re-read every day at the current GARCH term vol
    ex_dyn = np.zeros((n, m), bool)
    for t in range(STEPS):
        tau = STEPS - t
        sv = term_vol(H[:, t], tau, params)
        b = interp_boundary(tab, sv, tau)
        ok = ~np.isnan(b)
        ex_dyn[:, t] = ok & (((S[:, t] >= b) if is_call else (S[:, t] <= b)))
    ex_dyn &= intr > 0
    # LSM with betas fitted on training paths
    ex_lsm = np.zeros((n, m), bool)
    for t in range(1, STEPS):
        beta = lsm_betas[t]
        if beta is None:
            continue
        X = lsm_basis(S[:, t] / K, H[:, t])
        cont = X @ beta
        ex_lsm[:, t] = (intr[:, t] > 0) & (intr[:, t] > cont)
    out = dict(hold=hold, static=first_exercise(ex_static, "static"), dynamic=first_exercise(ex_dyn, "dynamic"),
               lsm=first_exercise(ex_lsm, "lsm"))
    # LSM time-0 decision: exercise now if intrinsic beats the mean LSM continuation value
    i0 = intr[0, 0]
    if i0 > out["lsm"].mean():
        out["lsm"] = np.full(n, i0)
        diag["lsm_early_share"] = 1.0
    return out, diag


def lsm_basis(mny, h):
    v = np.sqrt(252.0 * h)
    return np.column_stack([np.ones_like(mny), mny, mny ** 2, mny ** 3, v, v * mny, v * mny ** 2, v ** 2])


def lsm_fit(S, H, K, r, is_call):
    n, m = S.shape
    disc = np.exp(-r / 252.0)
    sgn = 1.0 if is_call else -1.0
    intr = np.maximum(sgn * (S - K), 0.0)
    cf = intr[:, -1].copy()
    tcf = np.full(n, STEPS)
    betas = [None] * (STEPS + 1)
    for t in range(STEPS - 1, 0, -1):
        itm = intr[:, t] > 0
        if itm.sum() < 200:
            continue
        y = cf[itm] * disc ** (tcf[itm] - t)
        X = lsm_basis(S[itm, t] / K, H[itm, t])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        betas[t] = beta
        ex = np.zeros(n, bool)
        ex[itm] = intr[itm, t] > X @ beta
        cf[ex] = intr[ex, t]
        tcf[ex] = t
    return betas


def garch_experiment():
    x = garch.load_returns()
    params, ll = garch.fit(x.to_numpy())
    _, pers_q, hbar_q = garch.expected_avg_var(np.array(1e-4), 1, params)
    pd.DataFrame([dict(omega=params[0], alpha=params[1], gamma=params[2], beta=params[3], lam=params[4],
                       loglik=ll, n_obs=len(x), q_persistence=pers_q, q_longrun_vol=np.sqrt(252 * hbar_q))]
                 ).to_csv(RES / "garch_params.csv", index=False)
    print("GJR-GARCH params", params, "Q persistence", pers_q, "Q long-run vol", np.sqrt(252 * hbar_q))
    rows = []
    q = 0.015
    cells = [(mny, r, v0) for mny in [1.0, 1.1] for r in [0.05, 0.10] for v0 in [0.10, 0.20, 0.40]]
    seed = 1500
    for mny, r, v0 in cells:
        K = 100.0 * mny
        h0 = v0 ** 2 / 252.0
        tab = boundary_table(K, r, q, False)
        sig_static = float(term_vol(np.array(h0), STEPS, params))
        St, Ht = garch.simulate_q(100.0, h0, r, q, N_TRAIN, STEPS, params, seed)
        betas = lsm_fit(St, Ht, K, r, False)
        Se, He = garch.simulate_q(100.0, h0, r, q, N_EVAL, STEPS, params, seed + 1)
        seed += 2
        cfs, diag = run_policies(Se, He, K, r, q, False, params, tab, sig_static, betas)
        del St, Ht, Se, He
        base = cfs["hold"]
        eep_lsm = cfs["lsm"] - base
        for k in ["static", "dynamic", "lsm"]:
            d = cfs[k] - base
            rows.append(dict(K_over_S=mny, r=r, vol0=v0, term_vol=sig_static, policy=k,
                             value=cfs[k].mean(), se_value=cfs[k].std(ddof=1) / np.sqrt(N_EVAL),
                             eep=d.mean(), se_eep=d.std(ddof=1) / np.sqrt(N_EVAL),
                             hold=base.mean(),
                             gap_vs_lsm=(cfs[k] - cfs["lsm"]).mean(),
                             se_gap_vs_lsm=(cfs[k] - cfs["lsm"]).std(ddof=1) / np.sqrt(N_EVAL),
                             tree_am_static=tr.lr_r(100.0, K, STEPS / 252, r, q, sig_static, False, True, 800),
                             tree_eu_static=tr.bs_price(100.0, K, STEPS / 252, r, q, sig_static, False),
                             early_share=diag.get(k + "_early_share", np.nan),
                             mean_exercise_day=diag.get(k + "_mean_exercise_day", np.nan),
                             termvol_at_exercise=diag.get(k + "_termvol_at_exercise", np.nan)))
        print(mny, r, v0, {k: round(cfs[k].mean() - base.mean(), 4) for k in cfs})
    g = pd.DataFrame(rows)
    # captured share of the LSM premium, with a delta-method SE from paired differences
    lsm = g[g.policy == "lsm"].set_index(["K_over_S", "r", "vol0"])
    g = g.join(lsm[["eep"]].rename(columns={"eep": "eep_lsm"}), on=["K_over_S", "r", "vol0"])
    g["captured"] = g.eep / g.eep_lsm
    g["se_captured"] = g.se_gap_vs_lsm / g.eep_lsm
    g.to_csv(RES / "garch_policies.csv", index=False)


@njit(cache=True)
def bermudan_crr(S, K, T, r, q, sigma, days, sub):
    """CRR put with days*sub steps where exercise is allowed only every `sub` steps (daily)."""
    n = days * sub
    u, d, p = tr._crr_params(T, r, q, sigma, n)
    disc = math.exp(-r * T / n)
    v = np.empty(n + 1)
    ratio = u / d
    s = S * d ** n
    for j in range(n + 1):
        v[j] = max(K - s, 0.0)
        s *= ratio
    for i in range(n - 1, -1, -1):
        s = S * d ** i
        for j in range(i + 1):
            c = disc * (p * v[j + 1] + (1.0 - p) * v[j])
            if i % sub == 0:
                c = max(c, K - s)
            v[j] = c
            s *= ratio
    return v[0]


def lsm_gbm_validation():
    """LSM in a constant-vol world, where the lattice gives the exact answer."""
    rows = []
    params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    for mny in [1.0, 1.1]:
        for r in [0.05, 0.10]:
            for sig in [0.10, 0.20, 0.40]:
                K = 100.0 * mny
                h0 = sig ** 2 / 252.0
                p = np.array([h0, 0.0, 0.0, 0.0, 0.0])   # omega = h0, no dynamics
                St, Ht = garch.simulate_q(100.0, h0, r, 0.015, N_TRAIN, STEPS, p, 7)
                betas = lsm_fit(St, Ht, K, r, False)
                Se, He = garch.simulate_q(100.0, h0, r, 0.015, N_EVAL, STEPS, p, 8)
                tab = boundary_table(K, r, 0.015, False)
                cfs, _ = run_policies(Se, He, K, r, 0.015, False, p, tab, sig, betas)
                # exact daily-exercise (Bermudan) premium from a lattice with SUB steps per day, exercise allowed daily
                exact_am = bermudan_crr(100.0, K, STEPS / 252, r, 0.015, sig, STEPS, SUB)
                exact_eu = tr.bs_price(100.0, K, STEPS / 252, r, 0.015, sig, False)
                d_l, d_t = cfs["lsm"] - cfs["hold"], cfs["static"] - cfs["hold"]
                rows.append(dict(K_over_S=mny, r=r, sigma=sig, exact_eep=exact_am - exact_eu,
                                 lsm_eep=d_l.mean(), se_lsm=d_l.std(ddof=1) / np.sqrt(N_EVAL),
                                 tree_eep=d_t.mean(), se_tree=d_t.std(ddof=1) / np.sqrt(N_EVAL)))
    v = pd.DataFrame(rows)
    v.to_csv(RES / "lsm_gbm_validation.csv", index=False)
    print(v.round(4).to_string())


if __name__ == "__main__":
    t = time.time()
    lattice_experiment()
    print(f"lattice: {time.time() - t:.0f}s")
    lsm_gbm_validation()
    garch_experiment()
    print(f"total: {time.time() - t:.0f}s")
