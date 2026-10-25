"""Martingales and random walks in discrete time (Chapter 2).

Random walks with steps +1 (probability p) and -1 (probability q = 1 - p), their exit
times from an interval, the closed-form gambler's-ruin formulas proved in the notes, and
the gains of betting strategies (martingale transforms). All samplers take a ``seed``
and return one row per path.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- random walks

def random_walk(n_steps: int, n_paths: int = 1, p: float = 0.5, x0: int = 0, seed=None) -> np.ndarray:
    """Paths of S_n = x0 + X_1 + ... + X_n with independent X_k = +1 w.p. p and -1 otherwise.

    Returns an integer array of shape (n_paths, n_steps + 1) whose first column is x0.
    """
    rng = np.random.default_rng(seed)
    steps = np.where(rng.random((n_paths, n_steps)) < p, 1, -1)
    start = np.full((n_paths, 1), x0, dtype=steps.dtype)
    return np.concatenate([start, x0 + np.cumsum(steps, axis=1)], axis=1)


def first_exit(paths: np.ndarray, a: float, b: float):
    """First index n at which a sampled path leaves the open interval (a, b), and the value there.

    Works row by row on an array of paths. Paths that stay in (a, b) over the whole sample get
    ``tau = -1`` and ``S_tau = nan``: their exit time is beyond the simulated horizon.
    """
    paths = np.atleast_2d(paths)
    outside = (paths <= a) | (paths >= b)
    exited = outside.any(axis=1)
    tau = np.where(exited, outside.argmax(axis=1), -1)
    S_tau = np.where(exited, paths[np.arange(len(paths)), np.maximum(tau, 0)], np.nan)
    return tau, S_tau


def simulate_exit(x: int, a: int, b: int, p: float = 0.5, n_paths: int = 10_000, seed=None):
    """Simulate the walk started at x until it leaves (a, b), for every path.

    Returns ``(tau, S_tau)``: the exit times and the exit levels (a or b), as integer arrays.
    Exit happens almost surely and its time has a geometric tail (Lemma in Section 2.4),
    so the loop terminates.
    """
    if not (a <= x <= b and a < b):
        raise ValueError("need integers a <= x <= b with a < b")
    rng = np.random.default_rng(seed)
    pos = np.full(n_paths, x, dtype=np.int64)
    tau = np.zeros(n_paths, dtype=np.int64)
    active = np.flatnonzero((pos > a) & (pos < b))
    while active.size:
        pos[active] += np.where(rng.random(active.size) < p, 1, -1)
        tau[active] += 1
        active = active[(pos[active] > a) & (pos[active] < b)]
    return tau, pos


# ------------------------------------------------- gambler's ruin, in closed form

def prob_upper(x, a: int, b: int, p: float = 0.5):
    """P(S_tau = b) for the walk started at x, tau the exit time from (a, b).

    (x - a) / (b - a) if p = 1/2, and (r^x - r^a) / (r^b - r^a) with r = q/p otherwise.
    Computed as (1 - r^(x-a)) / (1 - r^(b-a)), which is the same number and does not overflow.
    """
    x = np.asarray(x, dtype=float)
    if p == 0.5:
        return (x - a) / (b - a)
    r = (1 - p) / p
    return np.expm1((x - a) * np.log(r)) / np.expm1((b - a) * np.log(r))


def expected_exit_time(x, a: int, b: int, p: float = 0.5):
    """E[tau] for the walk started at x, tau the exit time from (a, b).

    (x - a)(b - x) if p = 1/2, and ((b - a) P(S_tau = b) - (x - a)) / (p - q) otherwise.
    """
    x = np.asarray(x, dtype=float)
    if p == 0.5:
        return (x - a) * (b - x)
    return ((b - a) * prob_upper(x, a, b, p) - (x - a)) / (2 * p - 1)


# ------------------------------------------------ strategies and their gains

def martingale_transform(H: np.ndarray, X: np.ndarray) -> np.ndarray:
    """(H . X)_n = sum_{k=1}^n H_k (X_k - X_{k-1}), with (H . X)_0 = 0.

    ``X`` has shape (n_paths, n + 1) and ``H`` shape (n_paths, n), column k - 1 holding H_k.
    """
    X = np.atleast_2d(X)
    H = np.atleast_2d(H)
    gains = np.cumsum(H * np.diff(X, axis=1), axis=1)
    return np.concatenate([np.zeros((X.shape[0], 1)), gains], axis=1)


def play(strategy, n_rounds: int, n_paths: int = 1, p: float = 0.5, seed=None):
    """Play ``n_rounds`` of a game that pays +1 (w.p. p) or -1 per unit staked.

    ``strategy(k, past, gains)`` returns the stakes H_k of round k, an array of shape (n_paths,).
    It receives only the outcomes of rounds 1, ..., k-1 (``past``, shape (n_paths, k-1)) and the
    gains up to time k-1 (``gains``, shape (n_paths, k)), so the stakes are predictable by
    construction. Returns ``(outcomes, H, G)`` with G the gains (H . S)_n, shape (n_paths, n_rounds + 1).
    """
    rng = np.random.default_rng(seed)
    outcomes = np.where(rng.random((n_paths, n_rounds)) < p, 1, -1)
    H = np.zeros((n_paths, n_rounds))
    G = np.zeros((n_paths, n_rounds + 1))
    for k in range(1, n_rounds + 1):
        H[:, k - 1] = strategy(k, outcomes[:, : k - 1], G[:, :k])
        G[:, k] = G[:, k - 1] + H[:, k - 1] * outcomes[:, k - 1]
    return outcomes, H, G


def doubling(bankroll: float = np.inf):
    """The doubling strategy: stake 1, 2, 4, ... until the first win, then stop.

    With a finite ``bankroll`` B the gambler only plays round k if he can still cover its stake,
    that is if 2^k - 1 <= B; with B = 2^m - 1 he can afford exactly m losses in a row.
    """
    def strategy(k, past, gains):
        no_win_yet = np.all(past == -1, axis=1)
        affordable = 2.0**k - 1 <= bankroll
        return np.where(no_win_yet & affordable, 2.0 ** (k - 1), 0.0)
    return strategy


def mean_one_products(n_steps: int, n_paths: int = 1, a: float = 0.5, seed=None) -> np.ndarray:
    """Paths of M_n = xi_1 ... xi_n with independent xi_k = 1 + a or 1 - a, each w.p. 1/2.

    E[xi_k] = 1, so M is a martingale with E[M_n] = 1; for 0 < a < 1 its typical paths decay
    like (1 - a^2)^(n/2). Returns an array of shape (n_paths, n_steps + 1) starting at M_0 = 1.
    """
    rng = np.random.default_rng(seed)
    xi = np.where(rng.random((n_paths, n_steps)) < 0.5, 1 + a, 1 - a)
    return np.concatenate([np.ones((n_paths, 1)), np.cumprod(xi, axis=1)], axis=1)
