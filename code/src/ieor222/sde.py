"""Stochastic differential equations (Chapter 9).

Samplers take a ``seed`` so that figures and tests are reproducible. As in
``ieor222.brownian``, paths are returned as ``(t, X)`` with ``t`` of shape
(n_steps + 1,) and ``X`` of shape (n_paths, n_steps + 1): one row per path.

Coefficient functions ``b(t, x)`` and ``sigma(t, x)`` are called with a scalar time
and an array of states, and must return arrays (numpy broadcasting does this for
expressions such as ``lambda t, x: mu * x``).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- Brownian input

def brownian_increments(T: float, n_steps: int, n_paths: int = 1, seed=None) -> np.ndarray:
    """Independent N(0, T / n_steps) increments on a uniform grid, shape (n_paths, n_steps)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, np.sqrt(T / n_steps), size=(n_paths, n_steps))


def coarsen(dW: np.ndarray, factor: int) -> np.ndarray:
    """Sum blocks of ``factor`` consecutive increments.

    The result is the same Brownian path seen on a grid ``factor`` times coarser, which is
    how a scheme with a large step is coupled with a fine one (or with an exact solution).
    """
    n_paths, n = dW.shape
    if n % factor:
        raise ValueError("the number of steps must be a multiple of factor")
    return dW.reshape(n_paths, n // factor, factor).sum(axis=2)


def path_from_increments(dW: np.ndarray) -> np.ndarray:
    """W on the grid, with W_0 = 0, from its increments: shape (n_paths, n_steps + 1)."""
    return np.concatenate([np.zeros((dW.shape[0], 1)), np.cumsum(dW, axis=1)], axis=1)


# ---------------------------------------------------------------- schemes

def euler_maruyama(b, sigma, x0, T: float, n_steps: int, n_paths: int = 1, seed=None, dW=None):
    """Euler--Maruyama: X_{k+1} = X_k + b(t_k, X_k) h + sigma(t_k, X_k) (W_{t_{k+1}} - W_{t_k}).

    ``x0`` is a number or an array of shape (n_paths,). If ``dW`` (shape (n_paths, n_steps))
    is given, those Brownian increments are used, so that the scheme runs on a prescribed path;
    otherwise fresh N(0, h) increments are drawn. Returns ``(t, X)``.
    """
    if dW is None:
        dW = brownian_increments(T, n_steps, n_paths, seed)
    n_paths, n_steps = dW.shape
    h = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    for k in range(n_steps):
        x = X[:, k]
        X[:, k + 1] = x + b(t[k], x) * h + sigma(t[k], x) * dW[:, k]
    return t, X


def picard_iterates(b, sigma, x0: float, t: np.ndarray, W: np.ndarray, n_iter: int) -> np.ndarray:
    """Picard iterates of the integral equation along one sampled Brownian path.

    Integrals are replaced by left-point sums on the grid ``t`` (W is the path on that grid):
    X^{n+1}_k = x0 + sum_{j<k} b(t_j, X^n_j) (t_{j+1} - t_j) + sum_{j<k} sigma(t_j, X^n_j) (W_{j+1} - W_j),
    starting from X^0 = x0. Returns an array of shape (n_iter + 1, len(t)). The fixed point of
    this discrete iteration is the Euler--Maruyama path, reached after at most len(t) - 1 steps.
    """
    W = np.asarray(W, dtype=float)
    dt, dW = np.diff(t), np.diff(W)
    out = np.empty((n_iter + 1, len(t)))
    out[0] = x0
    for n in range(n_iter):
        x = out[n, :-1]
        incr = b(t[:-1], x) * dt + sigma(t[:-1], x) * dW
        out[n + 1] = x0 + np.concatenate([[0.0], np.cumsum(incr)])
    return out


# ---------------------------------------------------------------- geometric Brownian motion

def gbm_exact(S0: float, mu: float, sigma: float, t: np.ndarray, W: np.ndarray) -> np.ndarray:
    """S_t = S0 exp(sigma W_t + (mu - sigma^2 / 2) t) along given Brownian paths (W_0 = 0)."""
    return S0 * np.exp(sigma * W + (mu - 0.5 * sigma**2) * t)


def gbm_euler_mean(S0: float, mu: float, T: float, n_steps: int) -> float:
    """Exact mean of the Euler approximation of GBM at time T: S0 (1 + mu h)^n with h = T / n."""
    return S0 * (1.0 + mu * T / n_steps) ** n_steps


def gbm_euler_errors(S0: float, mu: float, sigma: float, T: float, levels, n_paths: int,
                     seed=None, batch: int = 20_000) -> dict:
    """Strong and weak errors of Euler--Maruyama for GBM, against the exact solution.

    For each k in ``levels`` the step is h = T / 2^k. All schemes run on the same Brownian
    paths, sampled on the finest grid and coarsened, and the exact S_T uses the same W_T.
    Returns a dict of arrays: ``h``; ``strong`` = estimate of E|S_T - S^h_T| and its standard
    error ``strong_se``; ``weak`` = estimate of E[S^h_T] - E[S_T] (computed as the mean of the
    coupled differences, which has a far smaller variance than the difference of two
    independent means) and ``weak_se``; and ``weak_exact`` = S0 (1 + mu h)^n - S0 e^{mu T}.
    """
    levels = list(levels)
    K = max(levels)
    rng = np.random.default_rng(seed)
    sums = {k: np.zeros(3) for k in levels}  # sum of |e|, of e and of e^2, e = S^h_T - S_T
    done = 0
    while done < n_paths:
        m = min(batch, n_paths - done)
        dW = rng.normal(0.0, np.sqrt(T / 2**K), size=(m, 2**K))
        ST = gbm_exact(S0, mu, sigma, T, dW.sum(axis=1))
        for k in levels:
            inc = coarsen(dW, 2 ** (K - k))
            Sh = S0 * np.prod(1.0 + mu * T / 2**k + sigma * inc, axis=1)
            e = Sh - ST
            sums[k] += [np.abs(e).sum(), e.sum(), (e**2).sum()]
        done += m
    h = np.array([T / 2**k for k in levels])
    abs_mean, mean, sq_mean = (np.array([sums[k][i] / n_paths for k in levels]) for i in range(3))
    strong, weak = abs_mean, mean
    strong_se = np.sqrt(np.maximum(sq_mean - abs_mean**2, 0.0) / n_paths)
    weak_se = np.sqrt(np.maximum(sq_mean - mean**2, 0.0) / n_paths)
    weak_exact = np.array([gbm_euler_mean(S0, mu, T, 2**k) - S0 * np.exp(mu * T) for k in levels])
    return dict(h=h, strong=strong, strong_se=strong_se, weak=weak, weak_se=weak_se,
                weak_exact=weak_exact)


# ---------------------------------------------------------------- Ornstein--Uhlenbeck / Vasicek

def ou_mean(x0, kappa: float, theta: float, t):
    """E[X_t] = theta + (x0 - theta) e^{-kappa t} for dX = kappa (theta - X) dt + sigma dW."""
    return theta + (x0 - theta) * np.exp(-kappa * np.asarray(t, dtype=float))


def ou_var(kappa: float, sigma: float, t):
    """Var(X_t) = sigma^2 (1 - e^{-2 kappa t}) / (2 kappa), for a deterministic start X_0."""
    return sigma**2 * (1.0 - np.exp(-2.0 * kappa * np.asarray(t, dtype=float))) / (2.0 * kappa)


def ou_stationary(kappa: float, theta: float, sigma: float):
    """Mean and variance of the stationary law N(theta, sigma^2 / (2 kappa)) (kappa > 0)."""
    return theta, sigma**2 / (2.0 * kappa)


def ou_exact(x0, kappa: float, theta: float, sigma: float, T: float, n_steps: int,
             n_paths: int = 1, seed=None):
    """Exact samples of the Ornstein--Uhlenbeck process on the grid t_k = k T / n_steps.

    Uses the Gaussian transition of the notes: given X_{t_k}, X_{t_{k+1}} is normal with mean
    theta + (X_{t_k} - theta) e^{-kappa h} and variance sigma^2 (1 - e^{-2 kappa h}) / (2 kappa).
    On the grid this is the law of the process exactly, for every step size. ``x0`` is a number
    or an array of shape (n_paths,) (a random initial condition). Returns ``(t, X)``.
    """
    rng = np.random.default_rng(seed)
    h = T / n_steps
    a = np.exp(-kappa * h)
    s = np.sqrt(ou_var(kappa, sigma, h))
    t = np.linspace(0.0, T, n_steps + 1)
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    Z = rng.standard_normal((n_paths, n_steps))
    for k in range(n_steps):
        X[:, k + 1] = theta + (X[:, k] - theta) * a + s * Z[:, k]
    return t, X


# ---------------------------------------------------------------- CEV

def cev_paths(S0: float, mu: float, sigma: float, gamma: float, T: float, n_steps: int,
              n_paths: int = 1, seed=None, dW=None):
    """Euler scheme for the CEV model dS = mu S dt + sigma S^gamma dW, absorbed at 0.

    The diffusion coefficient is evaluated as sigma (S^+)^gamma. A step that lands at or below
    0 sets the path to 0, where it stays: absorption at 0 is part of the model (default of the
    issuer), not a property of the scheme. With gamma = 1 and no absorption this is the Euler
    scheme for geometric Brownian motion. Returns ``(t, S)``.
    """
    if dW is None:
        dW = brownian_increments(T, n_steps, n_paths, seed)
    n_paths, n_steps = dW.shape
    h = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    for k in range(n_steps):
        s = S[:, k]
        nxt = s + mu * s * h + sigma * np.maximum(s, 0.0) ** gamma * dW[:, k]
        S[:, k + 1] = np.where((s > 0) & (nxt > 0), nxt, 0.0)
    return t, S


def absorbed_fraction(S: np.ndarray) -> np.ndarray:
    """Fraction of paths that have reached 0 by each grid time (paths of ``cev_paths``)."""
    return np.mean(S <= 0.0, axis=0)
