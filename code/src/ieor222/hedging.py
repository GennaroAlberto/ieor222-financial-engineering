"""Delta hedging in the Black--Scholes model (Chapter 10).

Conventions follow :mod:`ieor222.blackscholes`: ``x`` is the stock price, ``K`` the strike,
``tau = T - t`` the time to maturity, ``r`` the interest rate and ``sigma > 0`` the volatility.
``mu`` is the drift of the stock under the real-world measure: it is used to simulate paths,
and never by a price or a hedge. Paths are arrays of shape ``(n_paths, n + 1)`` on a grid
``t_0 = 0 < t_1 < ... < t_n = T``, one row per path, as in :mod:`ieor222.brownian`.

A claim is described by two vectorised functions of the calendar time ``t`` and the stock
price ``x``: its price ``value(t, x)`` (equal to the payoff at ``t = T``) and its delta
``delta(t, x)``, used only at ``t < T``.
"""
from __future__ import annotations

import numpy as np

from ieor222.blackscholes import call, d_plus_minus, gbm_paths, norm_cdf, norm_pdf

# ---------------------------------------------------------------- claims

def call_delta(x, K, tau, r, sigma):
    """Delta N(d_+) of the call (``prop:bs-greeks``); needs tau > 0."""
    d_plus, _ = d_plus_minus(x, K, tau, r, sigma)
    return norm_cdf(d_plus)


def call_gamma(x, K, tau, r, sigma):
    """Gamma N'(d_+) / (x sigma sqrt(tau)) of the call; needs tau > 0."""
    d_plus, _ = d_plus_minus(x, K, tau, r, sigma)
    return norm_pdf(d_plus) / (np.asarray(x, float) * sigma * np.sqrt(tau))


def digital_call_delta(x, K, tau, r, sigma):
    """Delta e^{-r tau} N'(d_-) / (sigma x sqrt(tau)) of the claim paying 1 if S_T > K; tau > 0."""
    _, d_minus = d_plus_minus(x, K, tau, r, sigma)
    return np.exp(-r * tau) * norm_pdf(d_minus) / (sigma * np.asarray(x, float) * np.sqrt(tau))


def power_claim(x, tau, a, r, sigma) -> dict:
    """Price, delta and gamma of the claim paying S_T^a (the example ``exa:dh-power``).

    v(t, x) = e^{-alpha tau} x^a with alpha = (1 - a)(r + a sigma^2 / 2), for every real a.
    """
    x = np.asarray(x, float)
    alpha = (1.0 - a) * (r + 0.5 * a * sigma**2)
    v = np.exp(-alpha * np.asarray(tau, float)) * x**a
    return dict(value=v, delta=a * v / x, gamma=a * (a - 1.0) * v / x**2)


def bs_residual(value, t, x, r, sigma, dt=1e-5, dx=1e-3):
    """Finite-difference value of d_t v + r x d_x v + sigma^2 x^2 d_xx v / 2 - r v at (t, x).

    Central differences with steps ``dt`` in time and ``dx * x`` in space; zero (up to
    discretisation error) when ``value(t, x)`` solves the Black--Scholes equation.
    """
    t, x = np.asarray(t, float), np.asarray(x, float)
    hx = dx * x
    v = value(t, x)
    v_t = (value(t + dt, x) - value(t - dt, x)) / (2 * dt)
    v_x = (value(t, x + hx) - value(t, x - hx)) / (2 * hx)
    v_xx = (value(t, x + hx) - 2 * v + value(t, x - hx)) / hx**2
    return v_t + r * x * v_x + 0.5 * sigma**2 * x**2 * v_xx - r * v


# ---------------------------------------------------------------- discrete hedging

def rebalanced_wealth(t, S, shares, X0, r) -> np.ndarray:
    """Value at the grid times of the portfolio holding ``shares[:, k]`` shares over [t_k, t_{k+1}).

    The rest of the wealth is in the money market, so the portfolio is self-financing:
    X_{k+1} = shares_k S_{k+1} + e^{r (t_{k+1} - t_k)} (X_k - shares_k S_k), or, after
    discounting, e^{-r t_{k+1}} X_{k+1} - e^{-r t_k} X_k = shares_k (e^{-r t_{k+1}} S_{k+1} - e^{-r t_k} S_k).
    ``shares`` has shape (n_paths, n); the result has the shape of ``S``.
    """
    t = np.asarray(t, float)
    Y = np.asarray(S, float) * np.exp(-r * t)
    gains = np.cumsum(np.asarray(shares, float) * np.diff(Y, axis=-1), axis=-1)
    X0 = np.asarray(X0, float)[..., None]           # one initial capital per path, or a scalar
    disc = X0 + np.concatenate([np.zeros(gains.shape[:-1] + (1,)), gains], axis=-1)
    return disc * np.exp(r * t)


def delta_hedge(t, S, value, delta, r) -> np.ndarray:
    """Wealth of the discrete delta hedge: start from value(0, S_0), hold delta(t_k, S_{t_k}) over [t_k, t_{k+1})."""
    t = np.asarray(t, float)
    S = np.asarray(S, float)
    shares = delta(t[:-1], S[..., :-1])
    return rebalanced_wealth(t, S, shares, value(t[0], S[..., 0]), r)


def hedging_error(t, S, value, delta, payoff, r) -> np.ndarray:
    """Terminal hedging error X_T - payoff(S_T) of the discrete delta hedge, one value per path."""
    X = delta_hedge(t, S, value, delta, r)
    return X[..., -1] - payoff(np.asarray(S, float)[..., -1])


def hedged_pnl(t, S, value, delta, r) -> np.ndarray:
    """Discounted P&L over each period of a long claim hedged with delta(t_k, S_{t_k}) shares.

    P_k = e^{-r t_{k+1}} v(t_{k+1}, S_{t_{k+1}}) - e^{-r t_k} v(t_k, S_{t_k}) - delta_k (Y_{k+1} - Y_k),
    with Y = e^{-rt} S. The discounted hedging error of the short claim is -sum_k P_k.
    Shape (n_paths, n).
    """
    t = np.asarray(t, float)
    S = np.asarray(S, float)
    Y = S * np.exp(-r * t)
    V = value(t, S) * np.exp(-r * t)
    return np.diff(V, axis=-1) - delta(t[:-1], S[..., :-1]) * np.diff(Y, axis=-1)


def gamma_theta(t, S, gamma, r, sigma) -> np.ndarray:
    """Leading term (1/2) e^{-r t_k} Gamma_k S_{t_k}^2 (R_k^2 - sigma^2 h_k) of each period's P&L.

    R_k = Y_{k+1} / Y_k - 1 is the discounted return over [t_k, t_{k+1}] and h_k its length
    (``prop:dh-hedging-error``). Shape (n_paths, n).
    """
    t = np.asarray(t, float)
    S = np.asarray(S, float)
    Y = S * np.exp(-r * t)
    R = Y[..., 1:] / Y[..., :-1] - 1.0
    G = np.exp(-r * t[:-1]) * gamma(t[:-1], S[..., :-1]) * S[..., :-1] ** 2
    return 0.5 * G * (R**2 - sigma**2 * np.diff(t))


def call_hedge(S0, K, T, r, sigma, mu, n, n_paths, seed=None) -> dict:
    """Simulate GBM with drift ``mu`` on n equal steps and delta-hedge a short call along each path.

    Returns a dict with the grid ``t``, the paths ``S``, the hedge ``wealth`` and the terminal
    ``error`` X_T - (S_T - K)^+.
    """
    t, S = gbm_paths(S0, T, mu, sigma, n, n_paths, seed=seed)
    value = lambda s, x: call(x, K, T - s, r, sigma)
    delta = lambda s, x: call_delta(x, K, T - s, r, sigma)
    X = delta_hedge(t, S, value, delta, r)
    return dict(t=t, S=S, wealth=X, error=X[:, -1] - np.maximum(S[:, -1] - K, 0.0))


# ---------------------------------------------------------------- the leading term

def gamma_moment(gamma, u, S0, r, sigma, mu, n_nodes: int = 80):
    """E[e^{-2ru} Gamma(u, S_u)^2 S_u^4] under the real-world law of S, by Gauss--Hermite quadrature.

    ``gamma(u, x)`` is vectorised; ``u`` may be an array. Accurate when Gamma is smooth in x.
    """
    z, w = np.polynomial.hermite.hermgauss(n_nodes)
    u = np.asarray(u, float)[..., None]
    S = S0 * np.exp(sigma * np.sqrt(2.0 * u) * z + (mu - 0.5 * sigma**2) * u)
    vals = np.exp(-2 * r * u) * gamma(u, S) ** 2 * S**4
    return (vals @ w) / np.sqrt(np.pi)


def gamma_moment_call(u, S0, K, T, r, sigma, mu):
    """E[e^{-2ru} Gamma(u, S_u)^2 S_u^4] for the call, in closed form (a Gaussian integral), u < T.

    Gamma^2 S^4 = S^2 e^{-d_+^2} / (2 pi sigma^2 tau), and d_+ and log S_u are affine in one normal Z.
    """
    u = np.asarray(u, float)
    tau = T - u
    m = np.log(S0) + (mu - 0.5 * sigma**2) * u        # log S_u = m + s Z
    s = sigma * np.sqrt(u)
    a = (m - np.log(K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))   # d_+ = a + b Z
    b = np.sqrt(u / tau)
    lin = 2 * s - 2 * a * b
    # E[exp(-(a + bZ)^2 + 2 s Z)] = (1 + 2 b^2)^{-1/2} exp(-a^2 + lin^2 / (2 (1 + 2 b^2)))
    expect = np.exp(-a**2 + lin**2 / (2 * (1 + 2 * b**2))) / np.sqrt(1 + 2 * b**2)
    return np.exp(-2 * r * u + 2 * m) * expect / (2 * np.pi * sigma**2 * tau)


def asymptotic_rms(moment, T, sigma, n, n_nodes: int = 200) -> float:
    """sqrt((sigma^4 h / 2) int_0^T moment(u) du) with h = T / n: the leading term of ``prop:dh-hedging-error``(iii).

    This is the root-mean-square of the *discounted* hedging error to first order; multiply by
    e^{rT} for the error in time-T dollars. The substitution u = T - w^2 removes the
    integrable singularity (T - u)^{-1/2} of the call's moment at maturity.
    """
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    root = np.sqrt(T)
    wv = 0.5 * root * (x + 1.0)
    integral = np.sum(0.5 * root * w * moment(T - wv**2) * 2.0 * wv)
    return float(np.sqrt(0.5 * sigma**4 * (T / n) * integral))
