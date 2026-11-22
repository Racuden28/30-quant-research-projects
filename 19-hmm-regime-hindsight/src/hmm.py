"""A small Gaussian hidden Markov model, written from scratch (numba), for project 19.

Model. A hidden state s_t in {0, ..., K-1} follows a Markov chain with transition matrix A
(A[i, j] = P(s_{t+1} = j | s_t = i)) and initial distribution pi. Given the state, the daily
return is Gaussian: r_t | s_t = k ~ N(mu_k, sigma_k^2). Parameters are fitted by Baum-Welch EM
(Baum et al. 1970; Hamilton 1989 for the econometric version).

Three different "regime probabilities" come out of one fitted model, and the whole project is
about the difference between them:

* filtered  P(s_t = k | r_1..r_t)          uses data up to day t only  -> tradeable at close t
* predicted P(s_{t+1} = k | r_1..r_t)      = filtered @ A               -> the position for t+1
* smoothed  P(s_t = k | r_1..r_T)          uses the whole sample        -> hindsight

States are always relabelled so that state 0 has the lowest variance ("calm") and state K-1 the
highest ("turbulent"). That fixes the label-switching problem between refits.

Everything works on returns in percent (r * 100) so that variances are O(1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

LOG2PI = np.log(2.0 * np.pi)
VAR_FLOOR = 1e-4          # in %^2: a 0.01% daily std; stops a state collapsing onto one point


@dataclass
class HMMParams:
    pi: np.ndarray        # (K,)
    A: np.ndarray         # (K, K)
    mu: np.ndarray        # (K,)  daily mean, %
    var: np.ndarray       # (K,)  daily variance, %^2
    loglik: float = np.nan
    n_iter: int = 0

    @property
    def K(self) -> int:
        return len(self.mu)

    def copy(self) -> "HMMParams":
        return HMMParams(self.pi.copy(), self.A.copy(), self.mu.copy(), self.var.copy(), self.loglik, self.n_iter)

    def expected_duration(self) -> np.ndarray:
        """Expected spell length of each state in days, 1 / (1 - A_kk)."""
        return 1.0 / np.maximum(1.0 - np.diag(self.A), 1e-12)

    def stationary(self) -> np.ndarray:
        w, v = np.linalg.eig(self.A.T)
        s = np.real(v[:, np.argmin(np.abs(w - 1.0))])
        return s / s.sum()


@njit(cache=True)
def _emission(x, mu, var):
    T, K = x.shape[0], mu.shape[0]
    B = np.empty((T, K))
    for k in range(K):
        c = -0.5 * (LOG2PI + np.log(var[k]))
        for t in range(T):
            d = x[t] - mu[k]
            B[t, k] = np.exp(c - 0.5 * d * d / var[k])
    return B


@njit(cache=True)
def _forward(B, pi, A):
    """Scaled forward pass. Returns filtered probabilities alpha (T, K) and log-likelihood."""
    T, K = B.shape
    alpha = np.empty((T, K))
    ll = 0.0
    prior = pi.copy()
    for t in range(T):
        s = 0.0
        for k in range(K):
            alpha[t, k] = prior[k] * B[t, k]
            s += alpha[t, k]
        if s <= 0.0 or not np.isfinite(s):      # numerically impossible observation: reset to prior
            s = 1e-300
            for k in range(K):
                alpha[t, k] = prior[k]
            ll += np.log(s)
        else:
            ll += np.log(s)
            for k in range(K):
                alpha[t, k] /= s
        for j in range(K):
            p = 0.0
            for i in range(K):
                p += alpha[t, i] * A[i, j]
            prior[j] = p
    return alpha, ll


@njit(cache=True)
def _backward_and_stats(B, A, alpha):
    """Scaled backward pass. Returns smoothed gamma (T, K) and expected transition counts xi (K, K)."""
    T, K = B.shape
    gamma = np.empty((T, K))
    xi = np.zeros((K, K))
    beta = np.ones(K)
    gamma[T - 1] = alpha[T - 1]
    pred = np.empty(K)
    for t in range(T - 2, -1, -1):
        # pred[j] = P(s_{t+1}=j | r_1..t) up to the scale
        for j in range(K):
            p = 0.0
            for i in range(K):
                p += alpha[t, i] * A[i, j]
            pred[j] = p
        # numerator for xi_t(i, j) = alpha_t(i) A_ij B_{t+1}(j) beta_{t+1}(j)
        tot = 0.0
        for i in range(K):
            for j in range(K):
                tot += alpha[t, i] * A[i, j] * B[t + 1, j] * beta[j]
        if tot <= 0.0:
            tot = 1e-300
        for i in range(K):
            for j in range(K):
                xi[i, j] += alpha[t, i] * A[i, j] * B[t + 1, j] * beta[j] / tot
        nb = np.zeros(K)
        for i in range(K):
            s = 0.0
            for j in range(K):
                s += A[i, j] * B[t + 1, j] * beta[j]
            nb[i] = s
        sc = 0.0
        for i in range(K):
            sc += alpha[t, i] * nb[i]
        if sc <= 0.0:
            sc = 1e-300
        g = 0.0
        for i in range(K):
            gamma[t, i] = alpha[t, i] * nb[i] / sc
            g += gamma[t, i]
        if g <= 0.0:
            g = 1e-300
        for i in range(K):
            gamma[t, i] /= g
        m = nb.max()
        if m <= 0.0:
            m = 1e-300
        for i in range(K):
            beta[i] = nb[i] / m
    return gamma, xi


@njit(cache=True)
def _em(x, pi, A, mu, var, max_iter, tol):
    T, K = x.shape[0], mu.shape[0]
    prev = -np.inf
    ll = -np.inf
    it = 0
    for it in range(max_iter):
        B = _emission(x, mu, var)
        alpha, ll = _forward(B, pi, A)
        gamma, xi = _backward_and_stats(B, A, alpha)
        # M step
        for k in range(K):
            pi[k] = gamma[0, k]
        for i in range(K):
            s = 0.0
            for j in range(K):
                s += xi[i, j]
            for j in range(K):
                A[i, j] = (xi[i, j] + 1e-8) / (s + K * 1e-8)
        for k in range(K):
            w = 0.0
            m = 0.0
            for t in range(T):
                w += gamma[t, k]
                m += gamma[t, k] * x[t]
            m /= max(w, 1e-12)
            v = 0.0
            for t in range(T):
                d = x[t] - m
                v += gamma[t, k] * d * d
            v /= max(w, 1e-12)
            mu[k] = m
            var[k] = max(v, VAR_FLOOR)
        if abs(ll - prev) < tol * (1.0 + abs(ll)):
            break
        prev = ll
    return pi, A, mu, var, ll, it + 1


def _sort(p: HMMParams) -> HMMParams:
    o = np.argsort(p.var)
    return HMMParams(p.pi[o], p.A[np.ix_(o, o)], p.mu[o], p.var[o], p.loglik, p.n_iter)


def default_init(x: np.ndarray, K: int) -> HMMParams:
    """Deterministic start: states split by variance quantiles, sticky transitions."""
    sd = x.std()
    scale = np.linspace(0.6, 2.0, K) if K > 1 else np.ones(1)
    A = np.full((K, K), 0.02 / max(K - 1, 1))
    np.fill_diagonal(A, 0.98)
    return HMMParams(np.full(K, 1.0 / K), A, np.full(K, x.mean()), (sd * scale) ** 2)


def random_init(x: np.ndarray, K: int, rng: np.random.Generator) -> HMMParams:
    sd = x.std()
    var = np.sort((sd * rng.uniform(0.3, 2.5, K)) ** 2)
    stay = rng.uniform(0.9, 0.999, K)
    A = np.empty((K, K))
    for i in range(K):
        off = rng.dirichlet(np.ones(K - 1)) * (1 - stay[i]) if K > 1 else np.array([])
        A[i] = np.insert(off, i, stay[i])
    mu = x.mean() + rng.normal(0, 0.05, K)
    return HMMParams(rng.dirichlet(np.ones(K)), A, mu, var)


def fit(x: np.ndarray, K: int = 2, init: HMMParams | None = None, n_starts: int = 1,
        rng: np.random.Generator | None = None, max_iter: int = 500, tol: float = 1e-8) -> HMMParams:
    """Baum-Welch. Tries `init` (or the default start) plus `n_starts - 1` random starts and keeps the
    highest likelihood. Returned states are sorted by variance (0 = calm)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    rng = rng or np.random.default_rng(0)
    starts = [init.copy() if init is not None else default_init(x, K)]
    starts += [random_init(x, K, rng) for _ in range(n_starts - 1)]
    best = None
    for s in starts:
        try:
            pi, A, mu, var, ll, it = _em(x, s.pi.copy(), s.A.copy(), s.mu.copy(), s.var.copy(), max_iter, tol)
        except ZeroDivisionError:      # a start that collapses a state numerically: discard it
            continue
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(A))):
            continue
        cand = HMMParams(pi, A, mu, var, ll, it)
        if best is None or (np.isfinite(ll) and ll > best.loglik):
            best = cand
    return _sort(best)


def filtered(x: np.ndarray, p: HMMParams) -> tuple[np.ndarray, float]:
    """P(s_t | r_1..t) for every t, and the log-likelihood."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    B = _emission(x, p.mu, p.var)
    return _forward(B, p.pi.astype(np.float64), p.A.astype(np.float64))


def predicted(filt: np.ndarray, p: HMMParams) -> np.ndarray:
    """P(s_{t+1} | r_1..t) = filtered_t @ A."""
    return filt @ p.A


def smoothed(x: np.ndarray, p: HMMParams) -> np.ndarray:
    """P(s_t | r_1..T): uses the whole sample (hindsight)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    B = _emission(x, p.mu, p.var)
    alpha, _ = _forward(B, p.pi.astype(np.float64), p.A.astype(np.float64))
    gamma, _ = _backward_and_stats(B, p.A.astype(np.float64), alpha)
    return gamma


def viterbi(x: np.ndarray, p: HMMParams) -> np.ndarray:
    """Most likely state path given the whole sample (hindsight)."""
    x = np.asarray(x, dtype=np.float64)
    T, K = len(x), p.K
    logB = -0.5 * (LOG2PI + np.log(p.var)[None, :] + (x[:, None] - p.mu[None, :]) ** 2 / p.var[None, :])
    logA = np.log(np.maximum(p.A, 1e-300))
    delta = np.log(np.maximum(p.pi, 1e-300)) + logB[0]
    back = np.zeros((T, K), dtype=np.int64)
    for t in range(1, T):
        cand = delta[:, None] + logA
        back[t] = cand.argmax(0)
        delta = cand.max(0) + logB[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = delta.argmax()
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def simulate(p: HMMParams, T: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Simulate T days from the model. Starts from the stationary distribution. Returns (x, states)."""
    K = p.K
    s = np.empty(T, dtype=np.int64)
    s[0] = rng.choice(K, p=p.stationary())
    u = rng.random(T)
    cum = np.cumsum(p.A, axis=1)
    for t in range(1, T):
        s[t] = min(int(np.searchsorted(cum[s[t - 1]], u[t])), K - 1)
    x = p.mu[s] + np.sqrt(p.var[s]) * rng.standard_normal(T)
    return x, s


def n_params(K: int) -> int:
    return (K - 1) + K * (K - 1) + 2 * K


def bic(p: HMMParams, T: int) -> float:
    return -2 * p.loglik + n_params(p.K) * np.log(T)
