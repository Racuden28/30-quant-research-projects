"""
Five Monte Carlo return simulators (plus an equal-weight pool).

Every simulator exposes the same two-step interface:

    params = Model.fit(r)                      # r: 1-D array of daily log returns (estimation window)
    paths  = Model.simulate(params, r, H, n, rng)  # -> array (n, len(H)) of simulated cumulative
                                                   #    log returns at each horizon in H (sorted, in days)

`simulate` receives the full estimation window `r` so that conditional models
(GARCH, Markov switching) can filter their latent state up to the forecast origin,
while `fit` may be called less often (parameters are refreshed every few origins).

All simulators are written from scratch in NumPy/SciPy so every assumption is visible.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats

# --------------------------------------------------------------------------------------
# 1. Geometric Brownian motion (i.i.d. Gaussian log returns, static volatility)
# --------------------------------------------------------------------------------------


class GBM:
    name = "GBM"

    @staticmethod
    def fit(r):
        return {"mu": float(np.mean(r)), "sigma": float(np.std(r, ddof=1))}

    @staticmethod
    def simulate(p, r, H, n, rng):
        segs = np.diff([0, *H])
        inc = [p["mu"] * d + p["sigma"] * np.sqrt(d) * rng.standard_normal(n) for d in segs]
        return np.cumsum(np.column_stack(inc), axis=1)


# --------------------------------------------------------------------------------------
# 2. Merton (1976) jump-diffusion, estimated by maximum likelihood
#    r_t = mu + sigma*Z_t + sum_{k=1}^{N_t} J_k,   N_t ~ Poisson(lam),  J_k ~ N(muJ, sJ^2)
# --------------------------------------------------------------------------------------


LAM_MAX = 0.10  # jumps per trading day


def _merton_negll(theta, r, kmax=8):
    mu, log_sig, logit_lam, muJ, log_sJ = theta
    sig, sJ = np.exp(log_sig), np.exp(log_sJ)
    lam = LAM_MAX * special.expit(logit_lam)  # intensity capped at ~25 jumps/year so "jumps" stay rare events
    k = np.arange(kmax + 1)[:, None]
    logw = -lam + k * np.log(lam) - special.gammaln(k + 1)
    var = sig**2 + k * sJ**2
    logd = -0.5 * (np.log(2 * np.pi * var) + (r[None, :] - mu - k * muJ) ** 2 / var)
    ll = special.logsumexp(logw + logd, axis=0)
    return -np.sum(ll)


class Merton:
    name = "Merton jumps"

    @staticmethod
    def fit(r, x0=None):
        s = np.std(r)
        starts = [x0] if x0 is not None else []
        starts += [
            np.array([np.mean(r), np.log(0.7 * s), special.logit(0.2), -0.5 * s, np.log(2.5 * s)]),
            np.array([np.mean(r), np.log(0.5 * s), special.logit(0.5), 0.0, np.log(1.5 * s)]),
        ]
        best = None
        for x in starts:
            try:
                res = optimize.minimize(_merton_negll, x, args=(r,), method="L-BFGS-B")
                if best is None or res.fun < best.fun:
                    best = res
            except Exception:  # pragma: no cover - numerical failures fall back to next start
                continue
        mu, log_sig, logit_lam, muJ, log_sJ = best.x
        return {
            "mu": mu, "sigma": np.exp(log_sig), "lam": LAM_MAX * special.expit(logit_lam),
            "muJ": muJ, "sJ": np.exp(log_sJ), "x": best.x,
        }

    @staticmethod
    def simulate(p, r, H, n, rng):
        out = []
        for d in np.diff([0, *H]):
            nj = rng.poisson(p["lam"] * d, n)
            diff = p["mu"] * d + p["sigma"] * np.sqrt(d) * rng.standard_normal(n)
            out.append(diff + nj * p["muJ"] + np.sqrt(nj) * p["sJ"] * rng.standard_normal(n))
        return np.cumsum(np.column_stack(out), axis=1)


# --------------------------------------------------------------------------------------
# 3. GJR-GARCH(1,1) with Student-t innovations (Glosten, Jagannathan & Runkle, 1993)
#    sigma2_t = omega + (alpha + gamma*1[e_{t-1}<0]) e_{t-1}^2 + beta sigma2_{t-1}
# --------------------------------------------------------------------------------------


def _gjr_filter(e, omega, alpha, gamma, beta, s2_0):
    s2 = np.empty(len(e) + 1)
    s2[0] = s2_0
    for t in range(len(e)):
        s2[t + 1] = omega + (alpha + gamma * (e[t] < 0)) * e[t] ** 2 + beta * s2[t]
    return s2  # s2[-1] is the one-step-ahead variance after the last observation


class GARCH:
    name = "GJR-GARCH-t"
    dist = "t"

    @classmethod
    def fit(cls, r):
        from arch import arch_model

        am = arch_model(100 * r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist=cls.dist, rescale=False)
        res = am.fit(disp="off", show_warning=False, options={"maxiter": 500})
        pr = res.params
        return {
            "mu": pr["mu"] / 100, "omega": pr["omega"] / 1e4, "alpha": pr["alpha[1]"],
            "gamma": pr["gamma[1]"], "beta": pr["beta[1]"],
            "nu": max(pr["nu"], 2.2) if cls.dist == "t" else np.inf,
        }

    @staticmethod
    def simulate(p, r, H, n, rng):
        e = r - p["mu"]
        s2 = _gjr_filter(e, p["omega"], p["alpha"], p["gamma"], p["beta"], np.var(r))[-1]
        nu = p["nu"]
        scale = np.sqrt((nu - 2) / nu) if np.isfinite(nu) else 1.0
        s2 = np.full(n, s2)
        cum, out = np.zeros(n), []
        for step in range(1, max(H) + 1):
            z = rng.standard_t(nu, n) * scale if np.isfinite(nu) else rng.standard_normal(n)
            eps = np.sqrt(s2) * z
            cum += p["mu"] + eps
            s2 = p["omega"] + (p["alpha"] + p["gamma"] * (eps < 0)) * eps**2 + p["beta"] * s2
            if step in H:
                out.append(cum.copy())
        return np.column_stack(out)


# --------------------------------------------------------------------------------------
# 4. Two-state Markov regime-switching Gaussian model (Hamilton, 1989), fitted by EM
# --------------------------------------------------------------------------------------


def _hmm_forward(r, pi0, P, mu, sd):
    T, K = len(r), len(mu)
    dens = stats.norm.pdf(r[:, None], mu[None, :], sd[None, :]) + 1e-300
    alpha = np.empty((T, K))
    c = np.empty(T)
    a = pi0 * dens[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ P) * dens[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, c, dens


class RegimeSwitching:
    name = "Markov switching"

    @staticmethod
    def fit(r, n_iter=60, tol=1e-6):
        K = 2
        s = np.std(r)
        mu = np.array([np.mean(r), np.mean(r)])
        sd = np.array([0.6 * s, 1.8 * s])
        P = np.array([[0.98, 0.02], [0.05, 0.95]])
        pi0 = np.array([0.5, 0.5])
        last = -np.inf
        T = len(r)
        for _ in range(n_iter):
            alpha, c, dens = _hmm_forward(r, pi0, P, mu, sd)
            beta = np.empty((T, K))
            beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (P @ (dens[t + 1] * beta[t + 1])) / c[t + 1]
            g = alpha * beta
            g /= g.sum(1, keepdims=True)
            xi = (alpha[:-1, :, None] * P[None] * (dens[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
            P = xi.sum(0) / xi.sum((0, 2))[:, None]
            P /= P.sum(1, keepdims=True)
            pi0 = g[0]
            w = g.sum(0)
            mu = (g * r[:, None]).sum(0) / w
            sd = np.sqrt((g * (r[:, None] - mu) ** 2).sum(0) / w)
            sd = np.maximum(sd, 0.05 * s)
            ll = np.log(c).sum()
            if ll - last < tol:
                break
            last = ll
        o = np.argsort(sd)  # state 0 = calm, state 1 = turbulent
        return {"mu": mu[o], "sd": sd[o], "P": P[np.ix_(o, o)], "pi0": pi0[o]}

    @staticmethod
    def simulate(p, r, H, n, rng):
        alpha, _, _ = _hmm_forward(r, p["pi0"], p["P"], p["mu"], p["sd"])
        prob = alpha[-1] @ p["P"]  # predictive state distribution for t+1
        state = (rng.random(n) < prob[1]).astype(int)
        cum, out = np.zeros(n), []
        for step in range(1, max(H) + 1):
            if step > 1:
                stay = rng.random(n) < p["P"][state, state]
                state = np.where(stay, state, 1 - state)
            cum += p["mu"][state] + p["sd"][state] * rng.standard_normal(n)
            if step in H:
                out.append(cum.copy())
        return np.column_stack(out)


# --------------------------------------------------------------------------------------
# 5. Stationary block bootstrap (Politis & Romano, 1994) of the estimation window
# --------------------------------------------------------------------------------------


class BlockBootstrap:
    name = "Block bootstrap"
    mean_block = 10  # trading days

    @staticmethod
    def fit(r):
        return {}

    @staticmethod
    def simulate(p, r, H, n, rng):
        T = len(r)
        q = 1.0 / BlockBootstrap.mean_block
        idx = rng.integers(0, T, n)
        cum, out = r[idx].copy(), []
        for step in range(1, max(H) + 1):
            if step > 1:
                jump = rng.random(n) < q
                idx = np.where(jump, rng.integers(0, T, n), (idx + 1) % T)
                cum += r[idx]
            if step in H:
                out.append(cum.copy())
        return np.column_stack(out)


class GARCHNormal(GARCH):
    """Control arm: same GJR-GARCH dynamics with Gaussian innovations (dynamic, thin-tailed)."""

    name = "GJR-GARCH-N"
    dist = "normal"


MODELS = [GBM, Merton, GARCH, RegimeSwitching, BlockBootstrap]  # the five simulators in the pool
CONTROLS = [GARCHNormal]  # evaluated alongside, but not part of the pool
