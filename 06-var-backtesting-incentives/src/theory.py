"""
Part A - when does understating VaR pay?  Closed form under i.i.d. returns, plus a Monte Carlo
check with GARCH dynamics.

A bank whose true daily loss distribution has 99% quantile q reports lam*q. With i.i.d. returns the
number of exceptions in a 250-day year is Binomial(250, p(lam)), p(lam) = P(L > lam*q), and in steady
state the capital charge is proportional to

        C(lam) = lam * (base + E[plus(N)]),         N ~ Binomial(250, p(lam)).

The multiplier table was calibrated in 1996 with Gaussian returns in mind. The question is how the
capital-minimising lam* moves with the tail index of the loss distribution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from backtests import BASEL_PLUS, FRTB_PLUS, capital_path

RES = Path(__file__).resolve().parents[1] / "results"
LAMS = np.round(np.arange(0.50, 1.501, 0.005), 3)
NUS = [2.5, 3, 3.5, 4, 5, 6, 8, 10, 15, 30, np.inf]


def exceed_prob(lam: np.ndarray, nu: float) -> np.ndarray:
    d = stats.norm if np.isinf(nu) else stats.t(nu)
    return d.cdf(lam * d.ppf(0.01))


def expected_charge(lam: np.ndarray, nu: float, base: float, table: np.ndarray, n: int = 250):
    p = exceed_prob(lam, nu)
    k = np.arange(n + 1)
    pmf = stats.binom.pmf(k[None, :], n, p[:, None])
    e_plus = pmf @ table[: n + 1]
    p_red = pmf[:, 10:].sum(axis=1)
    p_yel = pmf[:, 5:10].sum(axis=1)
    return lam * (base + e_plus) / (base + e_plus[np.argmin(abs(lam - 1))]), p_yel, p_red


def analytic() -> pd.DataFrame:
    rows = []
    for regime, base, table in (("Basel 2.5", 3.0, BASEL_PLUS), ("FRTB", 1.5, FRTB_PLUS)):
        for nu in NUS:
            c, py, pr = expected_charge(LAMS, nu, base, table)
            for lam, ci, pyi, pri in zip(LAMS, c, py, pr):
                rows.append(dict(regime=regime, nu=nu, lam=lam, rel_charge=ci, p_yellow=pyi, p_red=pri,
                                 exc_prob=float(exceed_prob(np.array([lam]), nu)[0])))
    return pd.DataFrame(rows)


def optimum_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (regime, nu), g in df.groupby(["regime", "nu"]):
        i = g.rel_charge.idxmin()
        g_safe = g[g.p_red <= 0.05]               # supervisor tolerance: <=5% chance of a red year
        j = g_safe.rel_charge.idxmin()
        out.append(dict(regime=regime, nu=nu, lam_star=g.lam[i], saving=1 - g.rel_charge[i],
                        p_red_at_star=g.p_red[i], lam_star_red5=g.lam[j], saving_red5=1 - g.rel_charge[j]))
    return pd.DataFrame(out)


def break_even_schedule() -> pd.DataFrame:
    """The plus factor that exactly offsets the capital saved by reporting lam*VaR, for a bank whose
    expected yearly exception count is N:  lam(N) solves 250 * p(lam) = N, and
    plus_required(N) = base * (1/lam(N) - 1).  Compared with the Basel table."""
    from scipy.optimize import brentq
    rows = []
    for nu in (3.0, 4.0, 5.0, 6.0, 10.0, np.inf):
        for n_exc in range(3, 16):
            lam = brentq(lambda x: 250 * exceed_prob(np.array([x]), nu)[0] - n_exc, 0.05, 1.5)
            rows.append(dict(nu=nu, n_exc=n_exc, lam=lam, plus_required=3 * (1 / lam - 1),
                             plus_basel=BASEL_PLUS[n_exc], frtb_required=1.5 * (1 / lam - 1),
                             plus_frtb=FRTB_PLUS[n_exc]))
    return pd.DataFrame(rows)


def simulate_garch(nu: float, T: int, rng: np.random.Generator, omega=0.02, a=0.08, b=0.90):
    """GARCH(1,1) with standardised t innovations, returns in percent, plus the TRUE conditional 99% VaR."""
    d = stats.t(nu)
    scale = np.sqrt((nu - 2) / nu)
    z = d.rvs(size=T, random_state=rng) * scale
    s2 = np.empty(T); r = np.empty(T)
    s2[0] = omega / (1 - a - b)
    for t in range(T):
        if t:
            s2[t] = omega + a * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    q = -d.ppf(0.01) * scale * np.sqrt(s2)
    return r, q


def monte_carlo(n_paths: int = 200, T: int = 4000, seed: int = 6) -> pd.DataFrame:
    """Same question with volatility clustering and the real rolling/quarterly supervisory mechanics."""
    rng = np.random.default_rng(seed)
    lams = np.round(np.arange(0.60, 1.301, 0.01), 2)
    rows = []
    for nu in (3.0, 4.0, 6.0, 10.0, 50.0):
        acc, red = [], []
        for _ in range(n_paths):
            r, q = simulate_garch(nu, T, rng)
            out = capital_path(r, q, np.sqrt(10) * q, lams, 3.0, BASEL_PLUS)
            acc.append(out["charge"] / out["charge"][lams == 1.0])
            red.append(out["red"])
        acc, red = np.array(acc), np.array(red)

        def best(idx):
            m, rd = acc[idx].mean(axis=0), red[idx].mean(axis=0)
            ok = rd <= 0.05                       # at most 5% of supervisory quarters in the red zone
            j = np.where(ok)[0][np.argmin(m[ok])]
            return lams[j], 1 - m[j], rd[lams == 1.0][0]

        star, saving, red1 = best(np.arange(n_paths))
        boot = np.array([best(rng.integers(0, n_paths, n_paths))[:2] for _ in range(300)])
        rows.append(dict(nu=nu, lam_star_red5=star, saving=saving, red_share_honest=red1,
                         lam_lo=np.quantile(boot[:, 0], 0.025), lam_hi=np.quantile(boot[:, 0], 0.975),
                         saving_lo=np.quantile(boot[:, 1], 0.025), saving_hi=np.quantile(boot[:, 1], 0.975)))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = analytic()
    df.to_csv(RES / "theory_curves.csv", index=False, float_format="%.6f")
    opt = optimum_table(df)
    opt.to_csv(RES / "theory_optimum.csv", index=False, float_format="%.4f")
    print(opt.to_string())
    be = break_even_schedule()
    be.to_csv(RES / "theory_break_even.csv", index=False, float_format="%.4f")
    print(be[be.n_exc.isin([5, 7, 9, 10])].round(3).to_string())
    mc = monte_carlo()
    mc.to_csv(RES / "theory_mc_garch.csv", index=False, float_format="%.4f")
