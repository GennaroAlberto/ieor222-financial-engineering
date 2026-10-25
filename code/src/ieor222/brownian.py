"""Random walks and Brownian motion (Chapter 5).

All samplers take a ``seed`` so that figures and tests are reproducible, and
return arrays of shape ``(n_paths, n_points)``: one row per path.
"""
from __future__ import annotations

import numpy as np


def symmetric_random_walk(n_steps: int, n_paths: int = 1, p: float = 0.5, seed=None) -> np.ndarray:
    """Paths of the walk M_k = X_1 + ... + X_k, with X_j = +1 w.p. p and -1 otherwise.

    Returns an integer array of shape (n_paths, n_steps + 1) whose first column is M_0 = 0.
    """
    rng = np.random.default_rng(seed)
    steps = np.where(rng.random((n_paths, n_steps)) < p, 1, -1)
    start = np.zeros((n_paths, 1), dtype=steps.dtype)
    return np.concatenate([start, np.cumsum(steps, axis=1)], axis=1)


def scaled_random_walk(n: int, T: float = 1.0, n_paths: int = 1, seed=None):
    """The scaled walk W^(n)(t) = M_{nt} / sqrt(n) on the grid t = k/n, 0 <= k <= nT.

    Returns ``(t, W)`` with ``t`` of shape (m,) and ``W`` of shape (n_paths, m).
    """
    k = int(np.floor(n * T + 1e-12))
    M = symmetric_random_walk(k, n_paths, seed=seed)
    t = np.arange(k + 1) / n
    return t, M / np.sqrt(n)


def brownian_paths(T: float = 1.0, n_steps: int = 1000, n_paths: int = 1, seed=None):
    """Exact samples of a Brownian motion on the uniform grid t_j = j T / n_steps.

    The increments W_{t_{j+1}} - W_{t_j} are drawn as independent N(0, T / n_steps):
    on the grid this is the law of Brownian motion exactly, not an approximation.
    Returns ``(t, W)``.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    dW = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)
    return np.linspace(0.0, T, n_steps + 1), W


def p_variation(path: np.ndarray, p: float, every: int = 1) -> np.ndarray:
    """Sum of |increment|^p of a sampled path along the sub-grid of every ``every``-th point.

    With n_steps a power of two and ``every`` a power of two, the sub-grids are the
    dyadic partitions of [0, T] and always contain both endpoints.
    Works on the last axis, so a 2-D array of paths returns one value per path.
    """
    sub = np.asarray(path)[..., ::every]
    return np.sum(np.abs(np.diff(sub, axis=-1)) ** p, axis=-1)


def exponential_martingale(t: np.ndarray, W: np.ndarray, sigma: float) -> np.ndarray:
    """Z_t = exp(sigma W_t - sigma^2 t / 2), evaluated along sampled paths."""
    return np.exp(sigma * W - 0.5 * sigma**2 * t)
