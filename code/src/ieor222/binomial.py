"""Binomial and trinomial models (Chapter 4).

Conventions follow the notes: ``u`` and ``d`` are the up and down factors per period,
``r`` is the interest rate per period, and ``N`` is the number of periods.

Recombining trees are lists of arrays: ``tree[n][j]`` is the value at time ``n`` after
``j`` up-moves, for ``j = 0, ..., n``. Path-dependent claims use the full (non-recombining)
tree: at time ``n`` the ``2**n`` nodes are indexed by the integer whose binary digits are
the first ``n`` tosses (first toss most significant, 1 = up), so that the children of node
``i`` are ``2 i`` (down) and ``2 i + 1`` (up).
"""
from __future__ import annotations

from math import comb

import numpy as np


# ---------------------------------------------------------------- binomial model

def risk_neutral_probability(u: float, d: float, r: float) -> float:
    """The risk-neutral probability of an up-move, (1 + r - d) / (u - d).

    Raises ``ValueError`` unless 0 < d < u and d < 1 + r < u, the no-arbitrage
    condition ``prop:bin-no-arbitrage`` of the notes.
    """
    if not (0 < d < u):
        raise ValueError(f"need 0 < d < u, got d={d}, u={u}")
    if not (d < 1 + r < u):
        raise ValueError(f"arbitrage: need d < 1 + r < u, got d={d}, 1+r={1 + r}, u={u}")
    return (1 + r - d) / (u - d)


def stock_tree(S0: float, u: float, d: float, N: int) -> list[np.ndarray]:
    """Stock prices on the recombining tree: ``tree[n][j] = S0 * u**j * d**(n - j)``."""
    return [S0 * u ** np.arange(n + 1) * d ** (n - np.arange(n + 1)) for n in range(N + 1)]


def european(payoff, S0: float, u: float, d: float, r: float, N: int):
    """Price and hedge a claim ``payoff(S_N)`` by backward induction.

    ``payoff`` acts elementwise on an array of terminal prices. Returns ``(V, Delta)``:
    ``V[n][j]`` is the price at node ``(n, j)`` and ``Delta[n][j]`` (for ``n < N``) the
    number of shares held from ``n`` to ``n + 1`` in the replicating portfolio.
    """
    p = risk_neutral_probability(u, d, r)
    S = stock_tree(S0, u, d, N)
    V = [None] * (N + 1)
    Delta = [None] * N
    V[N] = np.asarray(payoff(S[N]), dtype=float)
    for n in range(N - 1, -1, -1):
        up, down = V[n + 1][1:], V[n + 1][:-1]
        V[n] = (p * up + (1 - p) * down) / (1 + r)
        Delta[n] = (up - down) / (S[n + 1][1:] - S[n + 1][:-1])
    return V, Delta


def american(payoff, S0: float, u: float, d: float, r: float, N: int):
    """Price an American claim that pays ``payoff(S_n)`` if exercised at time ``n``.

    Backward induction U_N = G_N, U_n = max(G_n, E^Q[U_{n+1} | F_n] / (1 + r)).
    Returns ``(U, Delta, exercise)``. ``Delta`` is the seller's hedge of ``thm:bin-american`` (iii),
    ``exercise[n][j]`` is True where exercising is optimal and pays something
    (G_n > 0 and G_n >= continuation value, up to rounding).
    """
    p = risk_neutral_probability(u, d, r)
    S = stock_tree(S0, u, d, N)
    U = [None] * (N + 1)
    Delta = [None] * N
    exercise = [None] * (N + 1)
    G = np.asarray(payoff(S[N]), dtype=float)
    U[N] = G
    exercise[N] = G > 0
    for n in range(N - 1, -1, -1):
        up, down = U[n + 1][1:], U[n + 1][:-1]
        cont = (p * up + (1 - p) * down) / (1 + r)
        G = np.asarray(payoff(S[n]), dtype=float)
        U[n] = np.maximum(G, cont)
        exercise[n] = (G > 0) & (G >= cont - 1e-12 * (1 + np.abs(cont)))
        Delta[n] = (up - down) / (S[n + 1][1:] - S[n + 1][:-1])
    return U, Delta, exercise


def price_by_expectation(payoff, S0: float, u: float, d: float, r: float, N: int) -> float:
    """(1 + r)^(-N) E^Q[payoff(S_N)], with the number of up-moves Binomial(N, p~) under Q."""
    p = risk_neutral_probability(u, d, r)
    j = np.arange(N + 1)
    weights = np.array([comb(N, k) for k in j], dtype=float) * p**j * (1 - p) ** (N - j)
    return float(np.sum(weights * payoff(S0 * u**j * d ** (N - j))) / (1 + r) ** N)


def early_exercise_boundary(K: float, S0: float, u: float, d: float, r: float, N: int) -> np.ndarray:
    """For the American put, the largest stock price at each date at which exercise is optimal.

    Entry ``n`` is NaN when no node at time ``n`` is in the exercise region.
    """
    S = stock_tree(S0, u, d, N)
    _, _, ex = american(lambda s: np.maximum(K - s, 0.0), S0, u, d, r, N)
    return np.array([S[n][ex[n]].max() if ex[n].any() else np.nan for n in range(N + 1)])


# ---------------------------------------------------------------- the full tree

def tosses(N: int) -> np.ndarray:
    """All 2^N toss sequences as rows of 0 (down) and 1 (up), in node-index order."""
    i = np.arange(2**N)[:, None]
    return (i >> (N - 1 - np.arange(N))) & 1


def stock_paths(S0: float, u: float, d: float, N: int) -> np.ndarray:
    """Array of shape (2^N, N + 1): row ``i`` is the path S_0, ..., S_N for toss sequence ``i``."""
    steps = np.where(tosses(N) == 1, u, d)
    return S0 * np.concatenate([np.ones((2**N, 1)), np.cumprod(steps, axis=1)], axis=1)


def path_dependent(path_payoff, S0: float, u: float, d: float, r: float, N: int):
    """Price and hedge a path-dependent claim V_N = path_payoff(S_0, ..., S_N) on the full tree.

    ``path_payoff`` maps an array of paths, shape (m, N + 1), to an array of shape (m,).
    Returns ``(S, V, Delta)``, lists indexed by time ``n`` of arrays of length ``2**n``.
    """
    p = risk_neutral_probability(u, d, r)
    paths = stock_paths(S0, u, d, N)
    S = [paths[:: 2 ** (N - n), n].copy() for n in range(N + 1)]
    V = [None] * (N + 1)
    Delta = [None] * N
    V[N] = np.asarray(path_payoff(paths), dtype=float)
    for n in range(N - 1, -1, -1):
        up, down = V[n + 1][1::2], V[n + 1][0::2]
        V[n] = (p * up + (1 - p) * down) / (1 + r)
        Delta[n] = (up - down) / (S[n + 1][1::2] - S[n + 1][0::2])
    return S, V, Delta


def self_financing_wealth(X0: float, Delta, S, r: float):
    """Run X_{n+1} = Delta_n S_{n+1} + (1 + r)(X_n - Delta_n S_n) on the full tree.

    ``Delta`` and ``S`` are as returned by :func:`path_dependent`. Returns the list of
    wealth arrays X[n] of length 2^n.
    """
    X = [np.array([float(X0)])]
    for n in range(len(Delta)):
        parent_X = np.repeat(X[n], 2)
        parent_D = np.repeat(Delta[n], 2)
        parent_S = np.repeat(S[n], 2)
        X.append(parent_D * S[n + 1] + (1 + r) * (parent_X - parent_D * parent_S))
    return X


def replication_error(path_payoff, S0: float, u: float, d: float, r: float, N: int) -> float:
    """Largest |X_N - V_N| over all 2^N paths for the replicating portfolio of ``thm:bin-multi-period``."""
    S, V, Delta = path_dependent(path_payoff, S0, u, d, r, N)
    X = self_financing_wealth(V[0][0], Delta, S, r)
    return float(np.max(np.abs(X[N] - V[N])))


# ---------------------------------------------------------------- trinomial model

def trinomial_theta_max(u: float, m: float, d: float, r: float) -> float:
    """Upper end of the range 0 < q_m < theta_max of the risk-neutral measures (one period)."""
    if not (0 < d < m < u):
        raise ValueError(f"need 0 < d < m < u, got d={d}, m={m}, u={u}")
    R = 1 + r
    if not (d < R < u):
        raise ValueError(f"arbitrage: need d < 1 + r < u, got d={d}, 1+r={R}, u={u}")
    return min((R - d) / (m - d), (u - R) / (u - m))


def trinomial_risk_neutral(u: float, m: float, d: float, r: float, theta: float):
    """The risk-neutral measure (q_u, q_m, q_d) with q_m = theta (one period).

    For 0 < theta < theta_max all three weights are positive; at the end points one of them
    vanishes and the measure is no longer equivalent to P.
    """
    R = 1 + r
    q_u = (R - d - theta * (m - d)) / (u - d)
    q_d = (u - R - theta * (u - m)) / (u - d)
    return q_u, theta, q_d


def trinomial_price_interval(payoff, S0: float, u: float, m: float, d: float, r: float):
    """End points (low, high) of {E^Q[payoff(S_1)] / (1 + r) : Q risk-neutral}, one period.

    The price is affine in theta = q_m, so the set is the open interval between the values at
    theta = 0 and theta = theta_max (a single point when the payoff is replicable).
    """
    R = 1 + r
    h = np.array([payoff(np.array([x * S0]))[0] for x in (u, m, d)], dtype=float)
    ends = []
    for theta in (0.0, trinomial_theta_max(u, m, d, r)):
        q = np.array(trinomial_risk_neutral(u, m, d, r, theta))
        ends.append(float(q @ h / R))
    return min(ends), max(ends)


def trinomial_moment_matching(sigma: float, nu: float, dt: float, lam: float = np.sqrt(3.0)):
    """Branch probabilities of a trinomial tree for the log-price, by moment matching.

    The log-price moves by +dx, 0 or -dx with dx = lam * sigma * sqrt(dt), and the
    probabilities match E[dX] = nu dt and E[dX^2] = sigma^2 dt + nu^2 dt^2.
    Returns ``(dx, p_u, p_m, p_d)``; nothing guarantees positivity: with lam = 1 the middle
    probability is -nu^2 dt / sigma^2 < 0 whenever nu != 0. With lam = sqrt(3) all three are
    positive as long as dt < 2 sigma^2 / nu^2.
    """
    dx = lam * sigma * np.sqrt(dt)
    second = (sigma**2 * dt + nu**2 * dt**2) / dx**2
    first = nu * dt / dx
    return dx, 0.5 * (second + first), 1.0 - second, 0.5 * (second - first)
