"""Exotic options in the Black--Scholes model (Chapter 13).

Conventions follow the notes. Under the risk-neutral measure the stock is
``S_t = S0 exp(sigma W_t + (r - sigma^2 / 2) t)``, that is ``S_t = S0 exp(sigma X_t)`` with
``X_t = W_t + nu t`` a Brownian motion with drift ``nu = r / sigma - sigma / 2``. ``K`` is a
strike, ``B`` a barrier, ``T`` the maturity, ``r`` the interest rate and ``sigma > 0`` the
volatility. Monte Carlo functions take a ``seed`` and return ``(estimate, standard error)``.

Sections: the reflection principle (``thm:exo-reflection``), barrier options
(``prop:exo-barrier``), the floating-strike lookback (``prop:exo-lookback``), Asian options
(``prop:exo-asian-geometric``, ``thm:exo-vecer``) and the perpetual American put
(``prop:exo-perpetual-put``).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import log_ndtr

from .binomial import american
from .blackscholes import call, crr_parameters, d_plus_minus, digital_call, norm_cdf, norm_pdf

# beta = -zeta(1/2) / sqrt(2 pi), the Broadie--Glasserman--Kou constant (zeta(1/2) = -1.4603545088...).
BGK_BETA = 1.4603545088095868 / np.sqrt(2.0 * np.pi)


def drift(r: float, sigma: float) -> float:
    """nu = r / sigma - sigma / 2, the drift of X = log(S / S0) / sigma under Q."""
    return r / sigma - 0.5 * sigma


# ---------------------------------------------------------------- the reflection principle

def reflect(W: np.ndarray, m: float) -> np.ndarray:
    """Reflect sampled paths after their first grid time at or above ``m``: 2 W_{t ^ tau} - W_t.

    ``W`` has shape (n_points,) or (n_paths, n_points). On a grid, tau is the first sampled time
    with W >= m, so this is the discrete version of the reflected path of ``thm:exo-reflection`` (i).
    """
    W = np.asarray(W, dtype=float)
    hit = W >= m
    first = np.where(hit.any(axis=-1), hit.argmax(axis=-1), W.shape[-1])
    idx = np.arange(W.shape[-1])
    after = idx >= first[..., None]
    W_tau = np.take_along_axis(W, np.minimum(first, W.shape[-1] - 1)[..., None], axis=-1)
    return np.where(after, 2.0 * W_tau - W, W)


def max_tail(m, T: float, nu: float = 0.0):
    """P(max_{t <= T} (W_t + nu t) >= m) for m >= 0: N((nu T - m)/sqrt T) + e^{2 nu m} N((-m - nu T)/sqrt T)."""
    m = np.asarray(m, dtype=float)
    s = np.sqrt(T)
    return (norm_cdf((nu * T - m) / s) + _image(m, T, nu))[()]


def _image(m, T, nu):
    """e^{2 nu m} N((-m - nu T)/sqrt T), computed in log scale to avoid overflow for large m."""
    return np.exp(2.0 * nu * m + log_ndtr((-m - nu * T) / np.sqrt(T)))


def max_density(m, T: float, nu: float = 0.0):
    """Density of the maximum of W_t + nu t on [0, T], for m > 0 (minus the derivative of :func:`max_tail`)."""
    m = np.asarray(m, dtype=float)
    s = np.sqrt(T)
    dens = (2.0 / s) * norm_pdf((m - nu * T) / s) - 2.0 * nu * _image(m, T, nu)
    return np.where(m > 0, dens, 0.0)[()]


def joint_tail(m, w, T: float, nu: float = 0.0):
    """P(max X >= m, X_T <= w) = e^{2 nu m} N((w - 2m - nu T)/sqrt T), for m >= 0 and w <= m."""
    m, w = np.broadcast_arrays(np.asarray(m, dtype=float), np.asarray(w, dtype=float))
    if np.any(w > m) or np.any(m < 0):
        raise ValueError("the formula holds for m >= 0 and w <= m")
    return (np.exp(2.0 * nu * m) * norm_cdf((w - 2.0 * m - nu * T) / np.sqrt(T)))[()]


def joint_density(a, w, T: float, nu: float = 0.0):
    """Density of (max_{t<=T} X_t, X_T) at (a, w): e^{nu w - nu^2 T/2} 2(2a - w)/(T sqrt(2 pi T)) e^{-(2a-w)^2/(2T)}."""
    a, w = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(w, dtype=float))
    base = 2.0 * (2.0 * a - w) / (T * np.sqrt(2.0 * np.pi * T)) * np.exp(-(2.0 * a - w) ** 2 / (2.0 * T))
    return np.where((a > 0) & (w < a), np.exp(nu * w - 0.5 * nu * nu * T) * base, 0.0)[()]


def bridge_crossing(level, x0, x1, h: float):
    """P(the bridge from x0 to x1 over a time h reaches ``level``), for a level above both ends.

    exp(-2 (level - x0)(level - x1) / h), the same with or without drift (``rem:exo-bridge``);
    1 if an end point is already at or beyond the level. For a lower level, pass all three negated.
    """
    level, x0, x1 = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (level, x0, x1)))
    gap = (level - x0) * (level - x1)
    above = (x0 >= level) | (x1 >= level)
    return np.where(above, 1.0, np.exp(-2.0 * np.maximum(gap, 0.0) / h))[()]


def bridge_max_sample(x0, x1, h: float, rng) -> np.ndarray:
    """Exact samples of the maximum of the bridge from x0 to x1 over a time h (inverse of :func:`bridge_crossing`)."""
    x0, x1 = np.broadcast_arrays(np.asarray(x0, dtype=float), np.asarray(x1, dtype=float))
    U = rng.random(x0.shape)
    return 0.5 * (x0 + x1 + np.sqrt((x1 - x0) ** 2 - 2.0 * h * np.log(U)))


def log_paths(T: float, n_steps: int, n_paths: int, nu: float, seed=None) -> np.ndarray:
    """Exact samples of X_t = W_t + nu t on the grid t_j = j T / n_steps; shape (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    h = T / n_steps
    dX = nu * h + np.sqrt(h) * rng.standard_normal((n_paths, n_steps))
    return np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dX, axis=1)], axis=1)


# ---------------------------------------------------------------- barrier options

def _capped_call(x, K, B, T, r, sigma):
    """u(x): price of (S_T - K)^+ 1{S_T < B} for 0 < K < B, = c(x;K) - c(x;B) - (B - K) delta(x;B).

    Evaluated as x [N(-d+(B)) - N(-d+(K))] - K e^{-rT} [N(-d-(B)) - N(-d-(K))], the same number
    written with upper tails, which keeps its accuracy for large x (the image point B^2 / S0).
    """
    dpK, dmK = d_plus_minus(x, K, T, r, sigma)
    dpB, dmB = d_plus_minus(x, B, T, r, sigma)
    return x * (norm_cdf(-dpB) - norm_cdf(-dpK)) - K * np.exp(-r * T) * (norm_cdf(-dmB) - norm_cdf(-dmK))


def _floored_call(x, K, B, T, r, sigma):
    """l(x): price of (S_T - K)^+ 1{S_T > B}: c(x;K) if B <= K, else c(x;B) + (B - K) delta(x;B)."""
    if B <= K:
        return call(x, K, T, r, sigma)
    return call(x, B, T, r, sigma) + (B - K) * digital_call(x, B, T, r, sigma)


def up_and_out_call(S0, K: float, B: float, T: float, r: float, sigma: float):
    """Price of (S_T - K)^+ 1{max_{t<=T} S_t < B} (``prop:exo-barrier`` (i)); zero if S0 >= B or K >= B."""
    S0 = np.asarray(S0, dtype=float)
    if K >= B:
        return np.zeros_like(S0)[()]
    x = np.minimum(S0, B)                        # the formula is only used where S0 < B
    image = (B / x) ** (2.0 * r / sigma**2 - 1.0) * _capped_call(B * B / x, K, B, T, r, sigma)
    # max(., 0): near S0 = 0 the two terms agree to round-off and their difference can be -1e-100
    return np.where(S0 < B, np.maximum(_capped_call(x, K, B, T, r, sigma) - image, 0.0), 0.0)[()]


def down_and_out_call(S0, K: float, B: float, T: float, r: float, sigma: float):
    """Price of (S_T - K)^+ 1{min_{t<=T} S_t > B} (``prop:exo-barrier`` (ii)); zero if S0 <= B."""
    S0 = np.asarray(S0, dtype=float)
    x = np.maximum(S0, B)
    image = (B / x) ** (2.0 * r / sigma**2 - 1.0) * _floored_call(B * B / x, K, B, T, r, sigma)
    return np.where(S0 > B, np.maximum(_floored_call(x, K, B, T, r, sigma) - image, 0.0), 0.0)[()]


def up_and_in_call(S0, K, B, T, r, sigma):
    """In--out parity: up-and-in = call - up-and-out."""
    return (call(S0, K, T, r, sigma) - up_and_out_call(S0, K, B, T, r, sigma))[()]


def down_and_in_call(S0, K, B, T, r, sigma):
    """In--out parity: down-and-in = call - down-and-out."""
    return (call(S0, K, T, r, sigma) - down_and_out_call(S0, K, B, T, r, sigma))[()]


def bgk_barrier(B: float, sigma: float, T: float, n_monitor: int, up: bool = True) -> float:
    """Broadie--Glasserman--Kou shift: B e^{+- beta sigma sqrt(T/m)}, away from the starting price."""
    shift = BGK_BETA * sigma * np.sqrt(T / n_monitor)
    return float(B * np.exp(shift if up else -shift))


def barrier_call_mc(S0: float, K: float, B: float, T: float, r: float, sigma: float, up: bool = True,
                    n_monitor: int = 52, n_paths: int = 20_000, continuous: bool = False, seed=None):
    """Monte Carlo price of the up-and-out (``up``) or down-and-out call.

    The log-price is sampled exactly at the ``n_monitor`` equally spaced dates. With
    ``continuous=False`` the barrier is checked at those dates only (a discretely monitored
    option). With ``continuous=True`` each path is weighted by its probability of not crossing
    the barrier between dates (``rem:exo-bridge``): an unbiased estimator of the continuously
    monitored price. Returns ``(price, standard error)``.
    """
    nu = drift(r, sigma)
    X = log_paths(T, n_monitor, n_paths, nu, seed)
    b = np.log(B / S0) / sigma
    Y = X if up else -X                          # a down barrier for X is an up barrier for -X
    level = b if up else -b
    alive = np.all(Y[:, 1:] < level, axis=1) & (Y[:, 0] < level)
    weight = alive.astype(float)
    if continuous:
        h = T / n_monitor
        cross = bridge_crossing(level, Y[:, :-1], Y[:, 1:], h)
        weight = weight * np.prod(1.0 - cross, axis=1)
    payoff = np.exp(-r * T) * np.maximum(S0 * np.exp(sigma * X[:, -1]) - K, 0.0) * weight
    return float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(n_paths))


# ---------------------------------------------------------------- lookback options

def lookback_put(S0, T: float, r: float, sigma: float):
    """Time-0 price of the floating-strike lookback put paying max_{t<=T} S_t - S_T (``prop:exo-lookback``)."""
    S0 = np.asarray(S0, dtype=float)
    s = sigma * np.sqrt(T)
    if r == 0.0:
        a = 0.5 * s
        return (S0 * (norm_cdf(a) - norm_cdf(-a) + s * norm_pdf(a) + 0.5 * s * s * norm_cdf(a)))[()]
    dp = (r + 0.5 * sigma**2) * np.sqrt(T) / sigma
    dm = dp - s
    disc = np.exp(-r * T)
    european_atm_put = disc * norm_cdf(-dm) - norm_cdf(-dp)
    return (S0 * (european_atm_put + sigma**2 / (2.0 * r) * (norm_cdf(dp) - disc * norm_cdf(-dm))))[()]


def _lookback_put_mc(S0: float, T: float, r: float, sigma: float, n_monitor: int = 52, n_paths: int = 20_000,
                     continuous: bool = False, seed=None):
    """Monte Carlo price of the lookback put, with the maximum over the monitoring dates or over [0, T].

    Private: used by the tests only (the discretely monitored case is an exercise).

    With ``continuous=True`` the maximum between dates is sampled exactly (:func:`bridge_max_sample`).
    """
    rng = np.random.default_rng(seed)
    nu = drift(r, sigma)
    h = T / n_monitor
    X = np.concatenate([np.zeros((n_paths, 1)),
                        np.cumsum(nu * h + np.sqrt(h) * rng.standard_normal((n_paths, n_monitor)), axis=1)], axis=1)
    if continuous:
        top = bridge_max_sample(X[:, :-1], X[:, 1:], h, rng).max(axis=1)
    else:
        top = X.max(axis=1)
    payoff = np.exp(-r * T) * S0 * (np.exp(sigma * top) - np.exp(sigma * X[:, -1]))
    return float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(n_paths))


# ---------------------------------------------------------------- Asian options

def black_call(F, K, s, discount):
    """discount * (F N(d+) - K N(d-)), d+- = (log(F/K) +- s^2/2)/s: the call on a lognormal G with E G = F, Var log G = s^2."""
    F, K, s = (np.asarray(a, dtype=float) for a in (F, K, s))
    dp = (np.log(F / K) + 0.5 * s * s) / s
    return (discount * (F * norm_cdf(dp) - K * norm_cdf(dp - s)))[()]


def geometric_law(T: float, r: float, sigma: float, n: int | None = None):
    """Mean and variance of log(G / S0) for the geometric average of ``prop:exo-asian-geometric``.

    ``n=None``: continuous average over [0, T]; otherwise the average of S at t_i = i T / n, i = 1..n.
    """
    if n is None:
        return (r - 0.5 * sigma**2) * T / 2.0, sigma**2 * T / 3.0
    return (r - 0.5 * sigma**2) * T * (n + 1) / (2.0 * n), sigma**2 * T * (n + 1) * (2 * n + 1) / (6.0 * n * n)


def weighted_geometric_law(times, weights, r: float, sigma: float):
    """Mean and variance of sum_i w_i log(S_{t_i} / S0), for any dates and weights (exact, Gaussian)."""
    t = np.asarray(times, dtype=float)
    w = np.asarray(weights, dtype=float)
    return (r - 0.5 * sigma**2) * float(w @ t), sigma**2 * float(w @ np.minimum.outer(t, t) @ w)


def geometric_asian_call(S0, K, T: float, r: float, sigma: float, n: int | None = None):
    """Price of (G - K)^+ with G the geometric average (continuous if ``n`` is None)."""
    mean, var = geometric_law(T, r, sigma, n)
    S0 = np.asarray(S0, dtype=float)
    return black_call(S0 * np.exp(mean + 0.5 * var), K, np.sqrt(var), np.exp(-r * T))


def asian_call_mc(S0: float, K: float, T: float, r: float, sigma: float, n_steps: int = 200,
                  n_paths: int = 20_000, seed=None) -> dict:
    """Monte Carlo price of the arithmetic Asian call (1/T int_0^T S dt - K)^+, with a control variate.

    The time average is the trapezoidal rule on ``n_steps`` steps. The control variate is the
    geometric average with the same trapezoidal weights, whose price is known exactly
    (:func:`weighted_geometric_law`); its coefficient is estimated from the same paths.
    Returns the plain and the controlled estimates with their standard errors.
    """
    nu = drift(r, sigma)
    X = log_paths(T, n_steps, n_paths, nu, seed)
    S = S0 * np.exp(sigma * X)
    w = np.full(n_steps + 1, 1.0 / n_steps)
    w[0] = w[-1] = 0.5 / n_steps
    disc = np.exp(-r * T)
    arith = disc * np.maximum(S @ w - K, 0.0)
    geo = disc * np.maximum(S0 * np.exp(sigma * X @ w) - K, 0.0)
    mean, var = weighted_geometric_law(np.linspace(0.0, T, n_steps + 1), w, r, sigma)
    geo_exact = float(black_call(S0 * np.exp(mean + 0.5 * var), K, np.sqrt(var), disc))
    c = np.cov(arith, geo)
    b = c[0, 1] / c[1, 1] if c[1, 1] > 0 else 0.0
    controlled = arith - b * (geo - geo_exact)
    se = lambda x: float(x.std(ddof=1) / np.sqrt(n_paths))
    return dict(plain=float(arith.mean()), plain_se=se(arith), price=float(controlled.mean()),
                se=se(controlled), geometric_exact=geo_exact, beta=float(b))


def vecer_gamma(t, T: float, r: float):
    """gamma(t) = (1 - e^{-r (T - t)}) / (r T), the number of shares held by Vecer's portfolio ((T - t)/T if r = 0)."""
    t = np.asarray(t, dtype=float)
    if r == 0.0:
        return ((T - t) / T)[()]
    return (-np.expm1(-r * (T - t)) / (r * T))[()]          # expm1: accurate for tiny r


def vecer_solve(T: float, r: float, sigma: float, z_min: float = -3.0, n_z: int = 1601, n_t: int = 800,
                times=None):
    """Solve d_t g + (sigma^2/2)(gamma(t) - z)^2 d_zz g = 0, g(T, z) = z^+ (``thm:exo-vecer``).

    Crank--Nicolson in time (after four half-steps of implicit Euler that damp the kink of z^+),
    centred differences on a uniform grid of [z_min, z_max] with z_max = max(gamma) + 0.5, and the
    boundary values g = 0 at z_min and g(t, z) = z at z_max (exact for z >= gamma(t)).
    Returns ``(z, g0)``, or ``(z, {t: g(t, .)})`` for the requested ``times``.
    """
    z_max = float(max(vecer_gamma(0.0, T, r), vecer_gamma(T, T, r))) + 0.5
    z = np.linspace(z_min, z_max, n_z)
    dz = z[1] - z[0]
    g = np.maximum(z, 0.0)
    want = sorted(set(times or [])) if times is not None else []
    saved = {}
    if T in want:
        saved[T] = g.copy()

    def step(g, t_new, t_old, theta):
        # one step from t_old down to t_new < t_old: (I - theta k A_new) g_new = (I + (1 - theta) k A_old) g_old
        k = t_old - t_new
        a_new = 0.5 * sigma**2 * (vecer_gamma(t_new, T, r) - z[1:-1]) ** 2 / dz**2
        a_old = 0.5 * sigma**2 * (vecer_gamma(t_old, T, r) - z[1:-1]) ** 2 / dz**2
        rhs = g[1:-1] + (1 - theta) * k * a_old * (g[:-2] - 2 * g[1:-1] + g[2:])
        lo, hi = 0.0, z_max                      # boundary values at t_new
        rhs[0] += theta * k * a_new[0] * lo
        rhs[-1] += theta * k * a_new[-1] * hi
        ab = np.zeros((3, n_z - 2))
        ab[0, 1:] = -theta * k * a_new[:-1]
        ab[1, :] = 1 + 2 * theta * k * a_new
        ab[2, :-1] = -theta * k * a_new[1:]
        out = np.empty_like(g)
        out[1:-1] = solve_banded((1, 1), ab, rhs)
        out[0], out[-1] = lo, hi
        return out

    grid = np.linspace(T, 0.0, n_t + 1)
    # Rannacher start: the first interval in four implicit half-steps
    sub = np.linspace(grid[0], grid[1], 5)
    for t_old, t_new in zip(sub[:-1], sub[1:]):
        g = step(g, t_new, t_old, 1.0)
    for t_old, t_new in zip(grid[1:-1], grid[2:]):
        g = step(g, t_new, t_old, 0.5)
        for s in want:
            if t_new <= s < t_old:
                saved[s] = g.copy()
    if times is not None:
        saved.setdefault(0.0, g.copy())
        return z, saved
    return z, g


def asian_call_vecer(S0, K, T: float, r: float, sigma: float, n_z: int = 1601, n_t: int = 800,
                     z_min: float = -3.0):
    """Price S0 g(0, z0) of the continuously averaged Asian call, z0 = gamma(0) - e^{-rT} K / S0."""
    z, g0 = vecer_solve(T, r, sigma, z_min=z_min, n_z=n_z, n_t=n_t)
    S0, K = np.broadcast_arrays(np.asarray(S0, dtype=float), np.asarray(K, dtype=float))
    z0 = vecer_gamma(0.0, T, r) - np.exp(-r * T) * K / S0
    return (S0 * np.interp(z0, z, g0))[()]


# ---------------------------------------------------------------- the perpetual American put

def perpetual_put_boundary(K: float, r: float, sigma: float) -> float:
    """Optimal exercise level L* = 2 r K / (2 r + sigma^2) (needs r > 0)."""
    if r <= 0:
        raise ValueError("the perpetual put has an exercise boundary only for r > 0")
    return 2.0 * r * K / (2.0 * r + sigma**2)


def perpetual_put(x, K: float, r: float, sigma: float):
    """Value of the perpetual American put (``prop:exo-perpetual-put``): K - x below L*, (K - L*)(x/L*)^{-2r/sigma^2} above."""
    x = np.asarray(x, dtype=float)
    L = perpetual_put_boundary(K, r, sigma)
    beta = 2.0 * r / sigma**2
    return np.where(x <= L, K - x, (K - L) * (np.maximum(x, L) / L) ** (-beta))[()]


def american_put_binomial(S0: float, K: float, T: float, r: float, sigma: float, n: int = 500) -> float:
    """American put with maturity T in the n-period tree of ``thm:bs-limit`` (backward induction of ``thm:bin-american``)."""
    u, d, rn, _ = crr_parameters(T, n, r, sigma)
    U, _, _ = american(lambda s: np.maximum(K - s, 0.0), S0, u, d, rn, n)
    return float(U[0][0])
