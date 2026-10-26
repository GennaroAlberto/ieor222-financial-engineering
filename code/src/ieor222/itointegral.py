"""The Ito integral by left-point sums (Chapter 7).

Paths are sampled on a uniform grid t_j = j T / n of [0, T] and stored as arrays of shape
``(n_paths, n + 1)``, one row per path, as in :mod:`ieor222.brownian`. An integrand is an
array of the same shape holding H at the grid times; only its values at the left end of
each step are used, which is what makes the sums Ito sums. Every sampler takes a ``seed``.
"""
from __future__ import annotations

import numpy as np

from .brownian import brownian_paths


def _cumulative(increments: np.ndarray) -> np.ndarray:
    """Prepend a zero and take cumulative sums along the last axis."""
    zero = np.zeros(increments.shape[:-1] + (1,))
    return np.concatenate([zero, np.cumsum(increments, axis=-1)], axis=-1)


def ito_integral(H, W) -> np.ndarray:
    """Cumulative left-point sums I_{t_k} = sum_{j<k} H_{t_j} (W_{t_{j+1}} - W_{t_j}).

    ``H`` and ``W`` have the same shape (..., n + 1); the last column of ``H`` is not used.
    The result has the same shape, with I_0 = 0. For a simple integrand whose jump times
    lie on the grid, this is its Ito integral exactly (the definition of simple integrands
    in Chapter 7).
    """
    H = np.asarray(H, dtype=float)
    W = np.asarray(W, dtype=float)
    return _cumulative(H[..., :-1] * np.diff(W, axis=-1))


def lambda_sum(W, lam: float = 0.0, every: int = 1) -> np.ndarray:
    """S_lam = sum_j ((1 - lam) W_{t_j} + lam W_{t_{j+1}}) (W_{t_{j+1}} - W_{t_j}).

    The t_j are every ``every``-th grid point (so ``every`` must divide n). lam = 0 gives
    the Ito (left-point) sum, lam = 1/2 the trapezoid sum, lam = 1 the right-point sum.
    Works on the last axis.
    """
    sub = np.asarray(W, dtype=float)[..., ::every]
    left, right = sub[..., :-1], sub[..., 1:]
    return np.sum(((1.0 - lam) * left + lam * right) * (right - left), axis=-1)


def midpoint_sum(W, every: int = 2) -> np.ndarray:
    """sum_j W_{m_j} (W_{t_{j+1}} - W_{t_j}), with m_j the grid point halfway between t_j and t_{j+1}.

    The t_j are every ``every``-th grid point; ``every`` must be even, so that the
    midpoints are grid points too.
    """
    if every % 2:
        raise ValueError("every must be even: the midpoints must lie on the grid")
    W = np.asarray(W, dtype=float)
    sub = W[..., ::every]
    mid = W[..., every // 2::every][..., : sub.shape[-1] - 1]
    return np.sum(mid * np.diff(sub, axis=-1), axis=-1)


def wdw_limit(W_T, T: float, lam: float = 0.0):
    """The L^2 limit 1/2 W_T^2 + (lam - 1/2) T of the lam-sums (Chapter 7, int W dW).

    lam = 0 is the Ito integral (W_T^2 - T) / 2.
    """
    return 0.5 * np.asarray(W_T) ** 2 + (lam - 0.5) * T


def quadratic_variation(X, every: int = 1) -> np.ndarray:
    """Cumulative sums of squared increments of X along every ``every``-th grid point.

    Returns an array whose k-th entry (last axis) is sum_{j<k} (X_{t_{j+1}} - X_{t_j})^2,
    where t_j runs over the sub-grid; the first entry is 0.
    """
    sub = np.asarray(X, dtype=float)[..., ::every]
    return _cumulative(np.diff(sub, axis=-1) ** 2)


def covariation(X, Y, every: int = 1) -> np.ndarray:
    """Cumulative sums of products of increments of X and Y along the sub-grid."""
    sx = np.asarray(X, dtype=float)[..., ::every]
    sy = np.asarray(Y, dtype=float)[..., ::every]
    return _cumulative(np.diff(sx, axis=-1) * np.diff(sy, axis=-1))


def time_integral(F, t) -> np.ndarray:
    """Cumulative left-point Riemann sums of int_0^t F_s ds on the grid ``t`` (last axis)."""
    F = np.asarray(F, dtype=float)
    return _cumulative(F[..., :-1] * np.diff(np.asarray(t, dtype=float)))


def simulate(integrand, T: float = 1.0, n_steps: int = 1000, n_paths: int = 1, seed=None):
    """Brownian paths, an integrand evaluated on them, and its Ito integral by left-point sums.

    ``integrand(t, W)`` receives the grid of shape (n + 1,) and the paths of shape
    (n_paths, n + 1) and returns H at the grid times (any array broadcastable to the shape
    of W). Returns ``(t, W, H, I)``.
    """
    t, W = brownian_paths(T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    H = np.broadcast_to(np.asarray(integrand(t, W), dtype=float), W.shape)
    return t, W, H, ito_integral(H, W)


def isometry_check(integrand, T: float = 1.0, n_steps: int = 500, n_paths: int = 20_000, seed=None):
    """Monte Carlo estimates of both sides of the Ito isometry E[I_T^2] = E int_0^T H_t^2 dt.

    Returns a dict with the estimates ``lhs`` and ``rhs``, the standard error of their
    difference ``se`` (both sides are computed on the same paths), and the mean of I_T.
    For left-point sums the identity holds exactly in expectation on the grid, so the gap
    is pure Monte Carlo noise.
    """
    t, W, H, I = simulate(integrand, T, n_steps, n_paths, seed)
    lhs_samples = I[:, -1] ** 2
    rhs_samples = time_integral(H**2, t)[:, -1]
    diff = lhs_samples - rhs_samples
    return {
        "lhs": float(lhs_samples.mean()),
        "rhs": float(rhs_samples.mean()),
        "se": float(diff.std(ddof=1) / np.sqrt(n_paths)),
        "mean": float(I[:, -1].mean()),
        "mean_se": float(I[:, -1].std(ddof=1) / np.sqrt(n_paths)),
    }
