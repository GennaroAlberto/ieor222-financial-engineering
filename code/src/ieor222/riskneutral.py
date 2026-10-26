"""Change of measure, risk-neutral pricing, numeraires and dividends (Chapter 11).

Conventions follow the notes: ``x`` is the price now, ``K`` the strike, ``tau = T - t`` the
time to maturity, ``r`` the interest rate, ``sigma > 0`` the volatility and ``q`` a continuous
dividend yield (for an exchange rate, ``q`` is the foreign interest rate). The market price of
risk is ``theta = (mu - r) / sigma`` and the Girsanov density is
``Z_t = exp(-theta W_t - theta^2 t / 2)``, under which ``W_t + theta t`` is a Brownian motion.

Samplers take a ``seed`` and return one row per path. Monte Carlo helpers return an estimate
together with the half-width of a 95% confidence interval from the central limit theorem.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

# ---------------------------------------------------------------- Girsanov densities


def girsanov_density(t, W, theta: float):
    """Z_t = exp(-theta W_t - theta^2 t / 2) along sampled Brownian paths (constant theta).

    ``t`` has shape (m,) and ``W`` shape (n_paths, m), as returned by
    ``ieor222.brownian.brownian_paths``. Z is the density on F_t of the measure Q of
    Girsanov's theorem, under which W_t + theta t is a Brownian motion.
    """
    return np.exp(-theta * np.asarray(W) - 0.5 * theta**2 * np.asarray(t))


def density_from_drift(Theta, dW, dt: float):
    """Discretised Girsanov density for a random market price of risk.

    ``Theta[:, j]`` is the value of Theta at the left end t_j of step j and ``dW[:, j]`` the
    Brownian increment over [t_j, t_{j+1}], both of shape (n_paths, n_steps). Returns
    Z_{t_k} = exp(-sum_{j<k} Theta_j dW_j - dt/2 sum_{j<k} Theta_j^2), shape
    (n_paths, n_steps + 1), first column 1. For constant Theta this is exact on the grid.
    """
    Theta = np.asarray(Theta, dtype=float)
    dW = np.asarray(dW, dtype=float)
    Theta = np.broadcast_to(Theta, dW.shape)
    log_increments = -Theta * dW - 0.5 * Theta**2 * dt
    zeros = np.zeros(dW.shape[:-1] + (1,))
    return np.exp(np.concatenate([zeros, np.cumsum(log_increments, axis=-1)], axis=-1))


def density_second_moment(theta: float, T: float) -> float:
    """E[Z_T^2] = exp(theta^2 T): the cost of reweighting, as in importance sampling."""
    return float(np.exp(theta**2 * T))


def effective_sample_size(weights) -> float:
    """(sum w)^2 / sum w^2: the number of equally weighted samples the weights are worth."""
    w = np.asarray(weights, dtype=float).ravel()
    return float(w.sum() ** 2 / np.sum(w * w))


def gbm_from_brownian(t, W, S0: float, drift: float, sigma: float):
    """S_t = S0 exp(sigma W_t + (drift - sigma^2 / 2) t) along given Brownian paths."""
    return S0 * np.exp(sigma * np.asarray(W) + (drift - 0.5 * sigma**2) * np.asarray(t))


def weighted_estimate(values, weights=None):
    """Estimate of E[w X] from samples X_i (and weights w_i, default 1), with a 95% half-width.

    The half-width is 1.96 times the sample standard deviation of w_i X_i over sqrt(n):
    the weighted estimator is an ordinary sample mean of the products w_i X_i.
    """
    v = np.asarray(values, dtype=float).ravel()
    if weights is not None:
        v = v * np.asarray(weights, dtype=float).ravel()
    return float(v.mean()), float(1.959963984540054 * v.std(ddof=1) / np.sqrt(v.size))


# ---------------------------------------------------------------- Black--Scholes with a yield


def d_plus_minus(x, K, tau, r, sigma, q=0.0):
    """d_+- = (log(x / K) + (r - q +- sigma^2 / 2) tau) / (sigma sqrt(tau)); needs x, K, tau > 0."""
    x, K, tau = (np.asarray(a, dtype=float) for a in (x, K, tau))
    v = sigma * np.sqrt(tau)
    d_plus = (np.log(x / K) + (r - q + 0.5 * sigma**2) * tau) / v
    return d_plus, d_plus - v


def bs_price(x, K, tau, r, sigma, q=0.0, kind: str = "call"):
    """Black--Scholes prices with a continuous dividend yield q (Garman--Kohlhagen if q = r_f).

    ``kind`` is one of ``"call"``, ``"put"``, ``"digital_call"`` (pays 1 if S_T > K),
    ``"digital_put"`` (pays 1 if S_T <= K) or ``"asset_call"`` (pays S_T if S_T > K):
        call          x e^{-q tau} N(d_+) - K e^{-r tau} N(d_-)
        put           K e^{-r tau} N(-d_-) - x e^{-q tau} N(-d_+)
        digital_call  e^{-r tau} N(d_-)
        digital_put   e^{-r tau} N(-d_-)
        asset_call    x e^{-q tau} N(d_+)
    Requires x, K, tau > 0 and sigma > 0. Broadcasts over numpy arrays.
    """
    d_p, d_m = d_plus_minus(x, K, tau, r, sigma, q)
    x, K, tau = (np.asarray(a, dtype=float) for a in (x, K, tau))
    disc, carry = np.exp(-r * tau), x * np.exp(-q * tau)
    prices = {
        "call": lambda: carry * ndtr(d_p) - K * disc * ndtr(d_m),
        "put": lambda: K * disc * ndtr(-d_m) - carry * ndtr(-d_p),
        "digital_call": lambda: disc * ndtr(d_m),
        "digital_put": lambda: disc * ndtr(-d_m),
        "asset_call": lambda: carry * ndtr(d_p),
    }
    if kind not in prices:
        raise ValueError(f"unknown kind {kind!r}")
    return prices[kind]()[()]


def forward_price(x, tau, r, q=0.0):
    """Forward price x e^{(r - q) tau}: of a stock with yield q, or of a currency with q = r_f."""
    return np.asarray(x, dtype=float) * np.exp((r - q) * np.asarray(tau, dtype=float))


def exercise_probabilities(x, K, tau, r, sigma, q=0.0):
    """(Q(S_T > K), Q^S(S_T > K)) = (N(d_-), N(d_+)): the same event under the money-market
    and the stock-numeraire measures (Proposition rn-numeraire)."""
    d_p, d_m = d_plus_minus(x, K, tau, r, sigma, q)
    return ndtr(d_m)[()], ndtr(d_p)[()]


# ---------------------------------------------------------------- Monte Carlo under two numeraires


def call_samples_two_numeraires(S0: float, K: float, T: float, r: float, sigma: float,
                                n_paths: int, seed=None) -> dict:
    """Samples of two unbiased estimators of the call price, from the same normal draws Y.

    ``"Q"``:   e^{-rT} (S_T - K)^+   with S_T = S0 exp(sigma sqrt(T) Y + (r - sigma^2/2) T),
               the money market as numeraire;
    ``"QS"``:  S0 (1 - K / S_T)^+   with S_T = S0 exp(sigma sqrt(T) Y + (r + sigma^2/2) T),
               the stock as numeraire, under which S has drift r + sigma^2.
    Both have mean x N(d_+) - K e^{-rT} N(d_-); the second is bounded by S0.
    """
    Y = np.random.default_rng(seed).standard_normal(n_paths)
    v = sigma * np.sqrt(T)
    ST_Q = S0 * np.exp(v * Y + (r - 0.5 * sigma**2) * T)
    ST_S = S0 * np.exp(v * Y + (r + 0.5 * sigma**2) * T)
    return {"Q": np.exp(-r * T) * np.maximum(ST_Q - K, 0.0),
            "QS": S0 * np.maximum(1.0 - K / ST_S, 0.0)}


def _gauss_grid(n: int = 20001, width: float = 12.0):
    y = np.linspace(-width, width, n)
    w = np.exp(-0.5 * y * y) / np.sqrt(2.0 * np.pi) * (y[1] - y[0])
    return y, w


def call_estimator_std(S0: float, K, T: float, r: float, sigma: float) -> dict:
    """Exact standard deviations (per path) of the two estimators of call_samples_two_numeraires.

    Computed by quadrature against the normal density on a fine grid (error below 1e-9 for
    the parameters used in the notes). ``K`` may be an array.
    """
    y, w = _gauss_grid()
    v = sigma * np.sqrt(T)
    K = np.atleast_1d(np.asarray(K, dtype=float))[:, None]
    ST_Q = S0 * np.exp(v * y + (r - 0.5 * sigma**2) * T)[None, :]
    ST_S = S0 * np.exp(v * y + (r + 0.5 * sigma**2) * T)[None, :]
    out = {}
    for name, X in (("Q", np.exp(-r * T) * np.maximum(ST_Q - K, 0.0)),
                    ("QS", S0 * np.maximum(1.0 - K / ST_S, 0.0))):
        m1, m2 = X @ w, (X * X) @ w
        out[name] = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    return out


def normal_quantile(p):
    """The inverse N^{-1} of the standard normal distribution function."""
    return ndtri(p)
