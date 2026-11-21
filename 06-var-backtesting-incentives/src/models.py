"""
One-day-ahead 99% VaR and 97.5% ES forecasters.

Convention: r_t is a daily log return. VaR and ES are reported as POSITIVE loss numbers,
so an exception (a "breach") on day t is r_t < -VaR_t.

Every forecaster sees only data strictly before the forecast day. Parametric models are
re-estimated every REFIT days on a rolling window of WINDOW observations and filtered daily
with the frozen parameters in between (the usual production setup).
"""
from __future__ import annotations

import warnings

import numpy as np
from arch import arch_model
from arch.univariate import Normal, SkewStudent, StudentsT
from scipy import stats

ALPHA_VAR = 0.01      # Basel 99% VaR
ALPHA_ES = 0.025      # FRTB 97.5% ES
WINDOW = 500          # estimation window (trading days)
REFIT = 20            # re-estimate every 20 trading days
EWMA_LAMBDA = 0.94    # RiskMetrics

MODEL_ORDER = ["HS-250", "RiskMetrics", "FHS-EWMA", "GARCH-t", "GJR-skew-t", "GJR-EVT"]

_ZN99, _ZN975 = stats.norm.ppf(ALPHA_VAR), stats.norm.ppf(ALPHA_ES)
_ESN975 = stats.norm.pdf(_ZN975) / ALPHA_ES

# Gauss-Legendre nodes on (0, 1) to integrate a quantile function: ES_a = (1/a) * int_0^a q(u) du
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_GL_U = 0.5 * (_GL_X + 1.0)
_GL_W = 0.5 * _GL_W


def tail_from_ppf(ppf, alpha: float) -> float:
    """Lower-tail expectation E[Z | Z <= q_alpha] of a standardised distribution, via its quantile function."""
    u = alpha * _GL_U
    return float(np.sum(_GL_W * ppf(u)))


# ---------------------------------------------------------------- non-parametric

def hs(window_returns: np.ndarray, n: int = 250) -> tuple[float, float, float]:
    x = window_returns[-n:]
    q_v = np.quantile(x, ALPHA_VAR, method="linear")
    q_e = np.quantile(x, ALPHA_ES, method="linear")
    es = x[x <= q_e].mean()
    return -q_v, -q_e, -es


def ewma_var_path(r: np.ndarray, lam: float = EWMA_LAMBDA, seed_n: int = 60) -> np.ndarray:
    """sigma2[t] = forecast of var(r_t) using r_0..r_{t-1}. Length len(r)+1 (last = next-day forecast)."""
    s2 = np.empty(len(r) + 1)
    s2[0] = np.mean(r[:seed_n] ** 2)
    for t in range(len(r)):
        s2[t + 1] = lam * s2[t] + (1 - lam) * r[t] ** 2
    return s2


# ---------------------------------------------------------------- GARCH family

def garch_filter(r: np.ndarray, mu: float, omega: float, alpha: float, gamma: float, beta: float,
                 s2_0: float) -> np.ndarray:
    """GJR-GARCH(1,1) conditional variance path; gamma=0 gives GARCH(1,1). Length len(r)+1."""
    e = r - mu
    s2 = np.empty(len(r) + 1)
    s2[0] = s2_0
    for t in range(len(r)):
        s2[t + 1] = omega + (alpha + gamma * (e[t] < 0)) * e[t] ** 2 + beta * s2[t]
    return s2


def fit_garch(r_pct: np.ndarray, kind: str):
    """Fit on returns in percent (numerically friendlier). kind in {'garch-t','gjr-skt','gjr-n'}."""
    o = 0 if kind == "garch-t" else 1
    dist = {"garch-t": "t", "gjr-skt": "skewt", "gjr-n": "normal"}[kind]
    am = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist, rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", options={"maxiter": 500})
    p = res.params
    out = dict(mu=p["mu"], omega=p["omega"], alpha=p["alpha[1]"], gamma=p.get("gamma[1]", 0.0),
               beta=p["beta[1]"])
    out["s2_last"] = float(res.conditional_volatility[-1] ** 2)
    out["e_last"] = float(r_pct[-1] - p["mu"])
    out["resid_std"] = np.asarray(res.std_resid)
    if dist == "t":
        out["dist"] = ("t", [p["nu"]])
    elif dist == "skewt":
        out["dist"] = ("skewt", [p["eta"], p["lambda"]])
    else:
        out["dist"] = ("normal", [])
    return out


def std_quantiles(dist: tuple[str, list]) -> tuple[float, float, float]:
    """(q_0.01, q_0.025, E[Z|Z<=q_0.025]) of the standardised innovation distribution."""
    name, par = dist
    if name == "t":
        f = StudentsT()
    elif name == "skewt":
        f = SkewStudent()
    else:
        f = Normal()
    ppf = lambda u: f.ppf(np.atleast_1d(u), par)  # noqa: E731
    return float(ppf(ALPHA_VAR)[0]), float(ppf(ALPHA_ES)[0]), tail_from_ppf(ppf, ALPHA_ES)


def evt_tail(z: np.ndarray, thresh_q: float = 0.90) -> tuple[float, float, float]:
    """McNeil-Frey (2000): GPD on the losses of standardised residuals above their 90% quantile.
    Returns (q_0.01, q_0.025, E[Z|Z<=q_0.025]) in the residual (return) scale, i.e. all negative."""
    L = -z
    u = np.quantile(L, thresh_q)
    exc = L[L > u] - u
    xi, _, beta = stats.genpareto.fit(exc, floc=0.0)
    xi = float(np.clip(xi, -0.5, 0.9))
    n, nu = len(L), len(exc)

    def var_l(a):
        return u + beta / xi * ((n / nu * a) ** (-xi) - 1) if abs(xi) > 1e-6 else u - beta * np.log(n / nu * a)

    v99, v975 = var_l(ALPHA_VAR), var_l(ALPHA_ES)
    es975 = v975 / (1 - xi) + (beta - xi * u) / (1 - xi)
    return -v99, -v975, -es975


# ---------------------------------------------------------------- rolling driver

def rolling_forecasts(r: np.ndarray, start: int) -> dict[str, np.ndarray]:
    """VaR/ES forecasts for days start..len(r)-1 (each uses r[:t] only).
    Returns {model: array (n_oos, 3)} with columns VaR99, VaR975, ES975 as positive numbers (decimal)."""
    n = len(r)
    T = n - start
    out = {m: np.full((T, 3), np.nan) for m in MODEL_ORDER}
    rp = r * 100.0
    state = {}
    for i, t in enumerate(range(start, n)):
        w = r[t - WINDOW:t]
        # HS-250
        out["HS-250"][i] = hs(w, 250)
        # EWMA path on the window (re-seeded at each step is wasteful; compute once per refit block)
        if i % REFIT == 0:
            s2w = ewma_var_path(w)
            z = w / np.sqrt(s2w[:-1])
            q_fhs = np.quantile(z, ALPHA_VAR)
            q_e = np.quantile(z, ALPHA_ES)
            es_fhs = z[z <= q_e].mean()
            state["ewma"] = dict(s2=s2w[-1], q=q_fhs, q2=q_e, es=es_fhs)
            for kind in ("garch-t", "gjr-skt", "gjr-n"):
                f = fit_garch(rp[t - WINDOW:t], kind)
                if kind == "gjr-n":
                    q, q2, es = evt_tail(f["resid_std"])
                else:
                    q, q2, es = std_quantiles(f["dist"])
                state[kind] = dict(f=f, q=q, q2=q2, es=es, s2=None)
        else:
            s2p = state["ewma"]["s2"]
            state["ewma"]["s2"] = EWMA_LAMBDA * s2p + (1 - EWMA_LAMBDA) * r[t - 1] ** 2
        sig = np.sqrt(state["ewma"]["s2"])
        out["RiskMetrics"][i] = (-_ZN99 * sig, -_ZN975 * sig, _ESN975 * sig)
        ew = state["ewma"]
        out["FHS-EWMA"][i] = (-ew["q"] * sig, -ew["q2"] * sig, -ew["es"] * sig)
        for kind, name in (("garch-t", "GARCH-t"), ("gjr-skt", "GJR-skew-t"), ("gjr-n", "GJR-EVT")):
            s = state[kind]
            f = s["f"]
            if s["s2"] is None:      # first day after a refit: one-step forecast from the fitted path
                e, s2 = f["e_last"], f["s2_last"]
            else:
                e, s2 = rp[t - 1] - f["mu"], s["s2"]
            s["s2"] = f["omega"] + (f["alpha"] + f["gamma"] * (e < 0)) * e ** 2 + f["beta"] * s2
            sd = np.sqrt(s["s2"])
            out[name][i] = (-(f["mu"] + s["q"] * sd) / 100.0, -(f["mu"] + s["q2"] * sd) / 100.0,
                            -(f["mu"] + s["es"] * sd) / 100.0)
    return out
