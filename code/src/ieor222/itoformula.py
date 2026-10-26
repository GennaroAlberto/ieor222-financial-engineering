"""Ito's formula (Chapter 8): Riemann sums, covariations and path-by-path checks.

Paths are sampled on a uniform grid t_0 = 0 < t_1 < ... < t_n = T and stored with time on
the last axis, as in :mod:`ieor222.brownian`: shape (n_points,) for one path, or
(n_paths, n_points) for several. Every "running" quantity returned here has the same shape
as the paths and starts at 0 at t_0, so its value at index k is the sum over the grid
intervals [t_j, t_{j+1}] with j < k.
"""
from __future__ import annotations

import numpy as np


def _prepend_zero(increments: np.ndarray) -> np.ndarray:
    zero = np.zeros(increments.shape[:-1] + (1,))
    return np.concatenate([zero, np.cumsum(increments, axis=-1)], axis=-1)


def left_sum(H, X) -> np.ndarray:
    """Running left-point Riemann sums  sum_{j<k} H_{t_j} (X_{t_{j+1}} - X_{t_j}).

    With X a Brownian path this is the discrete Ito integral of H: the integrand is evaluated
    at the left end of each interval, as predictability requires (Lemma "Riemann sums").
    ``H`` and ``X`` must have the same shape, or broadcast to it.
    """
    H, X = np.broadcast_arrays(np.asarray(H, float), np.asarray(X, float))
    return _prepend_zero(H[..., :-1] * np.diff(X, axis=-1))


def trapezoid_sum(H, X) -> np.ndarray:
    """Running sums  sum_{j<k} (H_{t_j} + H_{t_{j+1}})/2 (X_{t_{j+1}} - X_{t_j})  (Stratonovich-type)."""
    H, X = np.broadcast_arrays(np.asarray(H, float), np.asarray(X, float))
    return _prepend_zero(0.5 * (H[..., :-1] + H[..., 1:]) * np.diff(X, axis=-1))


def time_integral(G, t) -> np.ndarray:
    """Running left-point Riemann sums of  int_0^t G_s ds  on the grid ``t``."""
    G = np.asarray(G, float)
    return left_sum(G, np.broadcast_to(np.asarray(t, float), G.shape))


def covariation(X, Y, every: int = 1) -> np.ndarray:
    """Running covariation sums  sum_j (X_{t_{j+1}} - X_{t_j})(Y_{t_{j+1}} - Y_{t_j}).

    Uses every ``every``-th grid point, so that coarser partitions of the same path can be
    compared. With Y = X this is the running quadratic variation sum.
    """
    X, Y = np.broadcast_arrays(np.asarray(X, float)[..., ::every], np.asarray(Y, float)[..., ::every])
    return _prepend_zero(np.diff(X, axis=-1) * np.diff(Y, axis=-1))


def ito_check(t, W, f, f_x, f_xx, f_t=None) -> dict:
    """Both sides of Ito's formula for f(t, W_t), along sampled Brownian paths.

    ``f, f_x, f_xx, f_t`` are vectorised functions of (t, x); ``f_t`` defaults to zero (f
    depends on x only). Returns a dict of running arrays, all starting at 0:

    - ``lhs``: f(t, W_t) - f(0, W_0);
    - ``stochastic``: left-point sums of f_x(t_j, W_{t_j}) against W, the Ito integral;
    - ``second_order``: (1/2) int f_xx(s, W_s) ds, by left-point sums;
    - ``rhs``: the full right side, stochastic + int f_t ds + second_order;
    - ``first_order``: the chain rule of ordinary calculus, stochastic + int f_t ds.
    """
    t = np.asarray(t, float)
    W = np.asarray(W, float)
    tt = np.broadcast_to(t, W.shape)
    stochastic = left_sum(f_x(tt, W), W)
    drift_t = time_integral(f_t(tt, W), t) if f_t is not None else np.zeros_like(W)
    second = 0.5 * time_integral(f_xx(tt, W), t)
    return {
        "lhs": f(tt, W) - f(tt[..., :1], W[..., :1]),
        "stochastic": stochastic,
        "second_order": second,
        "rhs": stochastic + drift_t + second,
        "first_order": stochastic + drift_t,
    }


def correlated_brownian(T: float = 1.0, n_steps: int = 1000, corr=0.0, n_paths: int = 1, seed=None):
    """Brownian motions B = L W with prescribed correlation, on the grid t_j = j T / n_steps.

    ``corr`` is a correlation matrix C (d x d), or a number rho for d = 2. W is a vector of
    d independent Brownian motions and L L^T = C (the Cholesky factor, or L = V sqrt(Lambda) from
    the eigendecomposition C = V Lambda V^T when C is singular, e.g. rho = +-1). Each B^i is a Brownian motion and
    [B^i, B^j]_t = C_ij t. Returns ``(t, B)`` with B of shape (n_paths, d, n_steps + 1).
    """
    C = np.array([[1.0, corr], [corr, 1.0]]) if np.ndim(corr) == 0 else np.asarray(corr, float)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        lam, V = np.linalg.eigh(C)
        L = V * np.sqrt(np.clip(lam, 0.0, None))
    d = C.shape[0]
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    dW = rng.normal(0.0, np.sqrt(dt), size=(n_paths, d, n_steps))
    dB = np.einsum("ik,pkn->pin", L, dW)
    B = np.concatenate([np.zeros((n_paths, d, 1)), np.cumsum(dB, axis=-1)], axis=-1)
    return np.linspace(0.0, T, n_steps + 1), B


def sgn(x) -> np.ndarray:
    """sgn(x) = 1 for x > 0 and -1 for x <= 0, so that |sgn| = 1 everywhere."""
    return np.where(np.asarray(x) > 0, 1.0, -1.0)


def levy_brownian(W) -> np.ndarray:
    """B_t = int_0^t sgn(W_s) dW_s by left-point sums: a Brownian motion by Levy's theorem.

    On the grid the increments sgn(W_{t_j}) (W_{t_{j+1}} - W_{t_j}) are again independent
    N(0, h), so the discrete B is exactly a Gaussian random walk.
    """
    W = np.asarray(W, float)
    return left_sum(sgn(W), W)
