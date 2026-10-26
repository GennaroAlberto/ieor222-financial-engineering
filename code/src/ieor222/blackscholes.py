"""The Black--Scholes formula, digital options, the Greeks and the binomial limit (Chapter 6).

Conventions follow the notes: ``x`` is the stock price now, ``K`` the strike, ``tau = T - t``
the time to maturity, ``r`` the continuously compounded interest rate (any real number) and
``sigma > 0`` the volatility. The functions accept numpy arrays and broadcast.

Prices at ``tau = 0`` are the payoffs, and a call with ``K = 0`` is worth ``x``: the
formulas of ``thm:bs-formula`` are used only where ``tau > 0`` and ``K > 0``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.stats import binom

SQRT_2PI = np.sqrt(2.0 * np.pi)


def norm_cdf(y):
    """The standard normal distribution function N."""
    return ndtr(y)


def norm_pdf(y):
    """The standard normal density N'."""
    y = np.asarray(y, dtype=float)
    return np.exp(-0.5 * y * y) / SQRT_2PI


def d_plus_minus(x, K, tau, r, sigma):
    """d_+ and d_- = (log(x / K) + (r +- sigma^2 / 2) tau) / (sigma sqrt(tau)); needs x, K, tau > 0."""
    x, K, tau = (np.asarray(a, dtype=float) for a in (x, K, tau))
    v = sigma * np.sqrt(tau)
    d_plus = (np.log(x / K) + (r + 0.5 * sigma**2) * tau) / v
    return d_plus, d_plus - v


def _regular(x, K, tau):
    """Broadcast the arguments; mask where the closed forms apply (tau > 0 and K > 0).

    Returns ``(ok, x, K, tau, Ks, ts)``: ``Ks`` and ``ts`` equal ``K`` and ``tau`` on the mask
    and 1 elsewhere, so that the formulas can be evaluated everywhere without warnings.
    """
    x, K, tau = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (x, K, tau)))
    ok = (tau > 0) & (K > 0)
    return ok, x, K, tau, np.where(ok, K, 1.0), np.where(ok, tau, 1.0)


def call(x, K, tau, r, sigma):
    """Black--Scholes price x N(d_+) - K e^{-r tau} N(d_-) of the European call.

    At tau <= 0 this is the payoff (x - K)^+; with K = 0 the call is the stock, worth x.
    """
    ok, x, K, tau, Ks, ts = _regular(x, K, tau)
    dp, dm = d_plus_minus(x, Ks, ts, r, sigma)
    formula = x * norm_cdf(dp) - Ks * np.exp(-r * ts) * norm_cdf(dm)
    edge = np.where(tau > 0, x, np.maximum(x - K, 0.0))
    return np.where(ok, formula, edge)[()]


def put(x, K, tau, r, sigma):
    """Black--Scholes price K e^{-r tau} N(-d_-) - x N(-d_+) of the European put.

    At tau <= 0 this is the payoff (K - x)^+; with K = 0 the put is worthless.
    """
    ok, x, K, tau, Ks, ts = _regular(x, K, tau)
    dp, dm = d_plus_minus(x, Ks, ts, r, sigma)
    formula = Ks * np.exp(-r * ts) * norm_cdf(-dm) - x * norm_cdf(-dp)
    return np.where(ok, formula, np.maximum(K - x, 0.0))[()]


def digital_call(x, K, tau, r, sigma):
    """Price e^{-r tau} N(d_-) of the claim paying 1 if S_T > K (``prop:bs-digital``)."""
    ok, x, K, tau, Ks, ts = _regular(x, K, tau)
    _, dm = d_plus_minus(x, Ks, ts, r, sigma)
    edge = np.where(x > K, np.exp(-r * np.maximum(tau, 0.0)), 0.0)
    return np.where(ok, np.exp(-r * ts) * norm_cdf(dm), edge)[()]


def digital_put(x, K, tau, r, sigma):
    """Price e^{-r tau} N(-d_-) of the claim paying 1 if S_T <= K."""
    tau_ = np.maximum(np.asarray(tau, dtype=float), 0.0)
    return (np.exp(-r * tau_) - digital_call(x, K, tau, r, sigma))[()]


def asset_or_nothing_call(x, K, tau, r, sigma):
    """Price x N(d_+) of the claim paying S_T if S_T > K."""
    ok, x, K, tau, Ks, ts = _regular(x, K, tau)
    dp, _ = d_plus_minus(x, Ks, ts, r, sigma)
    return np.where(ok, x * norm_cdf(dp), np.where(x > K, x, 0.0))[()]


def greeks(x, K, tau, r, sigma, kind: str = "call") -> dict:
    """Delta, gamma, vega, theta (= d/dt, t the calendar time) and rho, from ``prop:bs-greeks``.

    Needs ``tau > 0`` and ``K > 0``; ``kind`` is ``"call"`` or ``"put"``.
    """
    x, K, tau = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (x, K, tau)))
    if np.any(tau <= 0) or np.any(K <= 0):
        raise ValueError("the Greeks are given by the formulas only for tau > 0 and K > 0")
    dp, dm = d_plus_minus(x, K, tau, r, sigma)
    disc = K * np.exp(-r * tau)
    gamma = norm_pdf(dp) / (x * sigma * np.sqrt(tau))
    vega = x * norm_pdf(dp) * np.sqrt(tau)
    decay = -x * norm_pdf(dp) * sigma / (2.0 * np.sqrt(tau))
    if kind == "call":
        out = dict(delta=norm_cdf(dp), theta=decay - r * disc * norm_cdf(dm), rho=tau * disc * norm_cdf(dm))
    elif kind == "put":
        out = dict(delta=norm_cdf(dp) - 1.0, theta=decay + r * disc * norm_cdf(-dm),
                   rho=-tau * disc * norm_cdf(-dm))
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    out.update(gamma=gamma, vega=vega)
    return {k: v[()] for k, v in out.items()}


# ---------------------------------------------------------------- the binomial limit

def crr_parameters(T: float, n: int, r: float, sigma: float):
    """The n-period tree of ``thm:bs-limit``: u = e^{sigma sqrt(T/n)}, d = 1/u, 1 + r_n = e^{r T/n}.

    Returns ``(u, d, r_n, q)`` with q the risk-neutral probability of an up-move. Raises
    ``ValueError`` when d < 1 + r_n < u fails, that is, when n <= r^2 T / sigma^2.
    """
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    growth = np.exp(r * dt)
    if not (d < growth < u):
        raise ValueError(f"arbitrage in the tree: need n > r^2 T / sigma^2 = {r * r * T / sigma**2:.3g}")
    return u, d, growth - 1.0, (growth - d) / (u - d)


def log_return_law(T: float, n: int, r: float, sigma: float):
    """Support and Q_n-probabilities of log(S_n / S_0) in the tree of :func:`crr_parameters`."""
    u, _, _, q = crr_parameters(T, n, r, sigma)
    j = np.arange(n + 1)
    return (2 * j - n) * np.log(u), binom.pmf(j, n, q)


def binomial_price(payoff, S0: float, T: float, r: float, sigma: float, n: int) -> float:
    """Time-0 price (1 + r_n)^{-n} E^{Q_n}[payoff(S_n)] in the n-period tree (formula ``eq:bin-crr``).

    The binomial weights come from ``scipy.stats.binom``, so n in the tens of thousands is fine.
    """
    support, weights = log_return_law(T, n, r, sigma)
    return float(np.exp(-r * T) * np.sum(weights * payoff(S0 * np.exp(support))))


def binomial_call(S0: float, K: float, T: float, r: float, sigma: float, n: int) -> float:
    """The n-period price of the call with strike K and maturity T."""
    return binomial_price(lambda s: np.maximum(s - K, 0.0), S0, T, r, sigma, n)


def binomial_put(S0: float, K: float, T: float, r: float, sigma: float, n: int) -> float:
    """The n-period price of the put with strike K and maturity T."""
    return binomial_price(lambda s: np.maximum(K - s, 0.0), S0, T, r, sigma, n)


# ---------------------------------------------------------------- at the money

def brenner_subrahmanyam(S0, T, sigma) -> dict:
    """The approximations of ``prop:bs-atm`` at the at-the-money-forward strike K = S0 e^{rT}.

    Price and vega of the call (or put) approx. S0 v / sqrt(2 pi) and S0 sqrt(T) / sqrt(2 pi),
    call delta approx. 1/2 + v / (2 sqrt(2 pi)), with v = sigma sqrt(T). The rounded rule of thumb
    replaces 1 / sqrt(2 pi) = 0.3989... by 0.4.
    """
    v = sigma * np.sqrt(np.asarray(T, dtype=float))
    return dict(price=S0 * v / SQRT_2PI, delta=0.5 + v / (2 * SQRT_2PI), vega=S0 * np.sqrt(T) / SQRT_2PI)


def atm_forward_call(S0, T, sigma):
    """Exact Black--Scholes price S0 (2 N(v / 2) - 1) of the call with K = S0 e^{rT}, v = sigma sqrt(T)."""
    v = sigma * np.sqrt(np.asarray(T, dtype=float))
    return S0 * (2.0 * norm_cdf(v / 2.0) - 1.0)


# ---------------------------------------------------------------- simulation

def gbm_terminal(S0: float, T: float, drift: float, sigma: float, n_paths: int, seed=None) -> np.ndarray:
    """Exact samples of S_T = S0 exp(sigma W_T + (drift - sigma^2 / 2) T).

    With ``drift = r`` these are samples under the risk-neutral measure of ``sec:bs-pricing``.
    """
    rng = np.random.default_rng(seed)
    return S0 * np.exp(sigma * np.sqrt(T) * rng.standard_normal(n_paths) + (drift - 0.5 * sigma**2) * T)


def gbm_paths(S0: float, T: float, drift: float, sigma: float, n_steps: int, n_paths: int = 1, seed=None):
    """Exact samples of geometric Brownian motion (``def:bs-gbm``) on a uniform grid of [0, T].

    Returns ``(t, S)`` with ``S`` of shape (n_paths, n_steps + 1).
    """
    from ieor222.brownian import brownian_paths

    t, W = brownian_paths(T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    return t, S0 * np.exp(sigma * W + (drift - 0.5 * sigma**2) * t)
