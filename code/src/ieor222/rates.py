"""Short-rate models of the term structure (Chapter 14).

Conventions. ``tau = T - t`` is the time to maturity; ``P`` is the price of a zero-coupon bond
paying 1 at its maturity; yields and forward rates are continuously compounded. All models are
specified under the risk-neutral measure Q, with dr = b(t, r) dt + sigma(t, r) dW under Q.
Samplers take a ``seed`` and return paths as ``(t, X)`` with ``t`` of shape (n_steps + 1,) and
``X`` of shape (n_paths, n_steps + 1), one row per path, as in ``ieor222.sde``.

Contents: curves (yields, forward rates, bootstrapping); the affine machinery (Riccati
equations); the Gaussian models (Ho--Lee, Vasicek, Hull--White); the CIR model; Monte Carlo bond
prices; the forward measure and bond options (Jamshidian's formula and decomposition).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from scipy.stats import norm


# ---------------------------------------------------------------- curves

def yields(P, tau):
    """Continuously compounded yields Y = -log(P) / tau."""
    return -np.log(np.asarray(P, dtype=float)) / np.asarray(tau, dtype=float)


def forward_rates(P, T):
    """Forward rates F(0; T_i, T_{i+1}) = log(P_i / P_{i+1}) / (T_{i+1} - T_i) between grid points."""
    P, T = np.asarray(P, dtype=float), np.asarray(T, dtype=float)
    return -np.diff(np.log(P)) / np.diff(T)


def bootstrap_zeros(coupons, prices, face: float = 100.0) -> np.ndarray:
    """Zero-coupon prices P(0,1), ..., P(0,n) from n bonds with annual coupons.

    Bond k (k = 1..n) matures in k years and pays ``coupons[k-1] * face`` at the end of each
    year and ``face`` at maturity, so its price is
    face * (c_k * (P_1 + ... + P_k) + P_k). The system is triangular and solved forwards.
    """
    P = []
    for c, p in zip(coupons, prices):
        P.append((p / face - c * sum(P)) / (1.0 + c))
    return np.array(P)


# ---------------------------------------------------------------- affine models

def affine_B(tau, beta1: float, alpha1: float):
    """Solution of B' = 1 + beta1 B - alpha1 B^2 / 2, B(0) = 0 (alpha1 >= 0), in closed form.

    With gamma = sqrt(beta1^2 + 2 alpha1) > 0:
    B = 2 (e^{gamma tau} - 1) / ((gamma - beta1)(e^{gamma tau} - 1) + 2 gamma);  B = tau if gamma = 0.
    """
    tau = np.asarray(tau, dtype=float)
    g = np.sqrt(beta1**2 + 2.0 * alpha1)
    if g == 0.0:
        return tau.copy()
    e = -np.expm1(-g * tau)                      # 1 - e^{-g tau}: the same formula divided by e^{g tau}
    return 2.0 * e / ((g - beta1) * e + 2.0 * g * (1.0 - e))


def affine_riccati(tau, beta0: float, beta1: float, alpha0: float, alpha1: float):
    """Solve the Riccati system of the affine theorem numerically (time-homogeneous case).

    B' = 1 + beta1 B - alpha1 B^2 / 2,  A' = -beta0 B + alpha0 B^2 / 2,  A(0) = B(0) = 0.
    Returns arrays (A, B) at the times ``tau`` (any order, all >= 0). Used to check the closed
    forms; the bond price is exp(A(tau) - B(tau) r).
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    order = np.argsort(tau)

    def rhs(_, y):
        a, b = y
        return [-beta0 * b + 0.5 * alpha0 * b * b, 1.0 + beta1 * b - 0.5 * alpha1 * b * b]

    sol = solve_ivp(rhs, (0.0, float(tau.max())), [0.0, 0.0], t_eval=tau[order],
                    rtol=1e-11, atol=1e-13, method="DOP853")
    A, B = np.empty_like(tau), np.empty_like(tau)
    A[order], B[order] = sol.y
    return A, B


# ---------------------------------------------------------------- Gaussian models

def gauss_B(tau, kappa: float):
    """B(tau) = (1 - e^{-kappa tau}) / kappa, and B(tau) = tau for kappa = 0."""
    tau = np.asarray(tau, dtype=float)
    return tau.copy() if kappa == 0 else -np.expm1(-kappa * tau) / kappa


def gauss_B2_integral(tau, kappa: float):
    """int_0^tau B(u)^2 du = (tau - B - kappa B^2 / 2) / kappa^2, and tau^3 / 3 for kappa = 0."""
    tau = np.asarray(tau, dtype=float)
    if kappa == 0:
        return tau**3 / 3.0
    B = gauss_B(tau, kappa)
    return (tau - B - 0.5 * kappa * B**2) / kappa**2


def gaussian_bond(mean_integral, var_integral):
    """E[exp(-I)] = exp(-m + v / 2) for a normal I with mean m and variance v."""
    return np.exp(-np.asarray(mean_integral) + 0.5 * np.asarray(var_integral))


def hull_white_bond(r, t: float, T, beta, kappa: float, sigma: float):
    """Bond price in dr = (beta(t) - kappa r) dt + sigma dW (Ho--Lee for kappa = 0).

    P(t,T) = exp(A(t,T) - B(T-t) r) with
    A(t,T) = -int_t^T beta(s) B(T-s) ds + (sigma^2 / 2) int_0^{T-t} B(u)^2 du (affine theorem).
    ``beta`` is a function of time; the first integral is computed by quadrature.
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    A = np.array([-quad(lambda s: beta(s) * gauss_B(Ti - s, kappa), t, Ti, epsabs=1e-13,
                        epsrel=1e-11, limit=200)[0] for Ti in T])
    A += 0.5 * sigma**2 * gauss_B2_integral(T - t, kappa)
    return np.exp(A - gauss_B(T - t, kappa) * r)


def ho_lee_beta(t, dfdT, sigma: float):
    """Drift that fits the Ho--Lee model to an initial forward curve: beta(t) = d_T f(0,t) + sigma^2 t."""
    return dfdT(t) + sigma**2 * np.asarray(t, dtype=float)


def hull_white_beta(t, f0, dfdT, kappa: float, sigma: float):
    """Hull--White drift fitting the initial forward curve f0 (solution of the exercise):
    beta(t) = d_T f(0,t) + kappa f(0,t) + sigma^2 (1 - e^{-2 kappa t}) / (2 kappa)."""
    t = np.asarray(t, dtype=float)
    if kappa == 0:
        return ho_lee_beta(t, dfdT, sigma)
    return dfdT(t) + kappa * f0(t) - sigma**2 * np.expm1(-2.0 * kappa * t) / (2.0 * kappa)


# ---------------------------------------------------------------- Vasicek

def vasicek_long_yield(kappa: float, theta: float, sigma: float) -> float:
    """R_infinity = theta - sigma^2 / (2 kappa^2), the limit of the yields as tau -> infinity."""
    return theta - sigma**2 / (2.0 * kappa**2)


def vasicek_A(tau, kappa: float, theta: float, sigma: float):
    """A(tau) = R_inf (B(tau) - tau) - sigma^2 B(tau)^2 / (4 kappa)."""
    tau = np.asarray(tau, dtype=float)
    B = gauss_B(tau, kappa)
    return vasicek_long_yield(kappa, theta, sigma) * (B - tau) - sigma**2 * B**2 / (4.0 * kappa)


def vasicek_bond(r, tau, kappa: float, theta: float, sigma: float):
    """Vasicek bond price P = exp(A(tau) - B(tau) r)."""
    return np.exp(vasicek_A(tau, kappa, theta, sigma) - gauss_B(tau, kappa) * np.asarray(r))


def vasicek_yield(r, tau, kappa: float, theta: float, sigma: float):
    """Vasicek yield (B(tau) r - A(tau)) / tau, equal to r at tau = 0."""
    tau = np.asarray(tau, dtype=float)
    safe = np.where(tau > 0, tau, 1.0)
    y = (gauss_B(safe, kappa) * r - vasicek_A(safe, kappa, theta, sigma)) / safe
    return np.where(tau > 0, y, r)


def vasicek_forward(r, tau, kappa: float, theta: float, sigma: float):
    """Instantaneous forward rate f = theta + e^{-kappa tau}(r - theta) - sigma^2 B(tau)^2 / 2."""
    tau = np.asarray(tau, dtype=float)
    return theta + np.exp(-kappa * tau) * (r - theta) - 0.5 * sigma**2 * gauss_B(tau, kappa)**2


def vasicek_shape_thresholds(kappa: float, theta: float, sigma: float):
    """Vasicek (1977): the yield curve is increasing if r <= lo, decreasing if r >= hi, humped
    in between, with lo = R_inf - sigma^2/(4 kappa^2) and hi = R_inf + sigma^2/(2 kappa^2)."""
    R, s = vasicek_long_yield(kappa, theta, sigma), sigma**2 / (4.0 * kappa**2)
    return R - s, R + 2.0 * s


def vasicek_mean(t, r0, kappa: float, theta: float):
    """E[r_t] = theta + (r0 - theta) e^{-kappa t}."""
    return theta + (r0 - theta) * np.exp(-kappa * np.asarray(t, dtype=float))


def vasicek_var(t, kappa: float, sigma: float):
    """Var(r_t) = sigma^2 (1 - e^{-2 kappa t}) / (2 kappa)."""
    return -sigma**2 * np.expm1(-2.0 * kappa * np.asarray(t, dtype=float)) / (2.0 * kappa)


def vasicek_negative_prob(t, r0, kappa: float, theta: float, sigma: float):
    """Q(r_t < 0) = N(-E[r_t] / sd(r_t)) for t > 0."""
    t = np.asarray(t, dtype=float)
    return norm.cdf(-vasicek_mean(t, r0, kappa, theta) / np.sqrt(vasicek_var(t, kappa, sigma)))


def vasicek_integral_moments(r, tau, kappa: float, theta: float, sigma: float):
    """Mean and variance of int_t^T r_u du given r_t = r (Gaussian):
    theta tau + (r - theta) B(tau) and sigma^2 int_0^tau B^2."""
    tau = np.asarray(tau, dtype=float)
    return (theta * tau + (r - theta) * gauss_B(tau, kappa),
            sigma**2 * gauss_B2_integral(tau, kappa))


def vasicek_step_moments(h: float, kappa: float, sigma: float):
    """Conditional covariance of (r_{t+h}, int_t^{t+h} r) given r_t, for the exact sampler.

    Var r_{t+h} = sigma^2 (1 - e^{-2 kappa h}) / (2 kappa); Var I = sigma^2 int_0^h B^2;
    Cov = sigma^2 int_0^h e^{-kappa u} B(u) du = sigma^2 B(h)^2 / 2.
    """
    B = gauss_B(h, kappa)
    return vasicek_var(h, kappa, sigma), sigma**2 * gauss_B2_integral(h, kappa), 0.5 * sigma**2 * B**2


def vasicek_exact(r0, kappa: float, theta: float, sigma: float, T: float, n_steps: int,
                  n_paths: int = 1, seed=None):
    """Exact joint samples of the Vasicek rate and its integral on the grid t_k = k T / n_steps.

    Given r_{t_k}, the pair (r_{t_{k+1}}, int_{t_k}^{t_{k+1}} r) is Gaussian with means
    theta + (r - theta) e^{-kappa h} and theta h + (r - theta) B(h) and the covariance of
    ``vasicek_step_moments``; it is independent of the past given r_{t_k}. So the samples have
    the exact law of the process on the grid, with no discretisation error.
    Returns ``(t, r, I)`` with I[:, k] = int_0^{t_k} r_u du.
    """
    rng = np.random.default_rng(seed)
    h = T / n_steps
    vx, vi, c = vasicek_step_moments(h, kappa, sigma)
    a, B = np.exp(-kappa * h), gauss_B(h, kappa)
    sx = np.sqrt(vx)
    l21, l22 = c / sx, np.sqrt(max(vi - c * c / vx, 0.0))
    t = np.linspace(0.0, T, n_steps + 1)
    r = np.empty((n_paths, n_steps + 1))
    I = np.zeros((n_paths, n_steps + 1))
    r[:, 0] = r0
    for k in range(n_steps):
        z1, z2 = rng.standard_normal(n_paths), rng.standard_normal(n_paths)
        x = r[:, k] - theta
        r[:, k + 1] = theta + x * a + sx * z1
        I[:, k + 1] = I[:, k] + theta * h + x * B + l21 * z1 + l22 * z2
    return t, r, I


def vasicek_forward_measure_rT(r0, T: float, kappa: float, theta: float, sigma: float,
                               n_paths: int, seed=None):
    """Samples of r_T under the T-forward measure: normal with mean f(0,T) and variance v(T)."""
    rng = np.random.default_rng(seed)
    m = vasicek_forward(r0, T, kappa, theta, sigma)
    return m + np.sqrt(vasicek_var(T, kappa, sigma)) * rng.standard_normal(n_paths)


def vasicek_bond_option_vol(T: float, S: float, kappa: float, sigma: float) -> float:
    """Sigma = sigma B(S - T) sqrt((1 - e^{-2 kappa T}) / (2 kappa)), the log-volatility of P(T,S)
    under the T-forward measure (total, over [0, T])."""
    return float(gauss_B(S - T, kappa) * np.sqrt(vasicek_var(T, kappa, sigma)))


def vasicek_bond_option(K, T: float, S: float, r0, kappa: float, theta: float, sigma: float,
                        call: bool = True):
    """Jamshidian's formula: price at 0 of the option on the S-bond with strike K, expiry T < S.

    Call: P(0,S) N(d+) - K P(0,T) N(d-);  put: K P(0,T) N(-d-) - P(0,S) N(-d+),
    d+- = (log(P(0,S) / (K P(0,T))) +- Sigma^2 / 2) / Sigma.
    """
    K = np.asarray(K, dtype=float)
    PT, PS = vasicek_bond(r0, T, kappa, theta, sigma), vasicek_bond(r0, S, kappa, theta, sigma)
    s = vasicek_bond_option_vol(T, S, kappa, sigma)
    dp = (np.log(PS / (K * PT)) + 0.5 * s**2) / s
    dm = dp - s
    if call:
        return PS * norm.cdf(dp) - K * PT * norm.cdf(dm)
    return K * PT * norm.cdf(-dm) - PS * norm.cdf(-dp)


def jamshidian_coupon_option(K: float, T: float, pay_times, cash_flows, r0, kappa: float,
                             theta: float, sigma: float, call: bool = True):
    """Option with expiry T and strike K on a coupon bond paying cash_flows[i] at pay_times[i] > T.

    Jamshidian's decomposition: find r* with sum_i c_i P(T, T_i; r*) = K; then the option is a
    portfolio of zero-coupon options with strikes K_i = P(T, T_i; r*). Returns
    (price, r_star, strikes).
    """
    Ti, c = np.asarray(pay_times, dtype=float), np.asarray(cash_flows, dtype=float)

    def value(x):
        return np.sum(c * vasicek_bond(x, Ti - T, kappa, theta, sigma)) - K

    lo, hi = -1.0, 1.0
    while value(lo) < 0:
        lo *= 2
    while value(hi) > 0:
        hi *= 2
    rs = brentq(value, lo, hi, xtol=1e-14)
    Ks = vasicek_bond(rs, Ti - T, kappa, theta, sigma)
    price = sum(ci * vasicek_bond_option(k, T, s, r0, kappa, theta, sigma, call)
                for ci, k, s in zip(c, Ks, Ti))
    return float(price), float(rs), Ks


def simple_rate_caplet(k: float, T: float, delta: float, r0, kappa: float, theta: float,
                       sigma: float) -> float:
    """Caplet paying delta (L(T, T+delta) - k)^+ at T + delta, with L = (1/P(T,T+delta) - 1)/delta:
    (1 + delta k) puts on P(T, T+delta) with strike 1 / (1 + delta k) (exercise)."""
    return float((1 + delta * k) * vasicek_bond_option(1 / (1 + delta * k), T, T + delta, r0,
                                                       kappa, theta, sigma, call=False))


# ---------------------------------------------------------------- CIR

def feller_ratio(kappa: float, theta: float, sigma: float) -> float:
    """nu = 2 kappa theta / sigma^2: the Feller condition is nu >= 1."""
    return 2.0 * kappa * theta / sigma**2


def cir_gamma(kappa: float, sigma: float) -> float:
    return float(np.sqrt(kappa**2 + 2.0 * sigma**2))


def cir_B(tau, kappa: float, sigma: float):
    """B(tau) = 2 (e^{g tau} - 1) / ((g + kappa)(e^{g tau} - 1) + 2 g), g = sqrt(kappa^2 + 2 sigma^2)."""
    return affine_B(tau, -kappa, sigma**2)


def cir_A(tau, kappa: float, theta: float, sigma: float):
    """A(tau) = (2 kappa theta / sigma^2) log(2 g e^{(g + kappa) tau / 2} / ((g + kappa)(e^{g tau} - 1) + 2 g))."""
    tau = np.asarray(tau, dtype=float)
    g = cir_gamma(kappa, sigma)
    e = -np.expm1(-g * tau)                      # 1 - e^{-g tau}; den = e^{g tau} ((g + kappa) e + 2 g (1 - e))
    return 2.0 * kappa * theta / sigma**2 * (np.log(2.0 * g / ((g + kappa) * e + 2.0 * g * (1.0 - e)))
                                             + 0.5 * (kappa - g) * tau)


def cir_bond(r, tau, kappa: float, theta: float, sigma: float):
    return np.exp(cir_A(tau, kappa, theta, sigma) - cir_B(tau, kappa, sigma) * np.asarray(r))


def cir_yield(r, tau, kappa: float, theta: float, sigma: float):
    tau = np.asarray(tau, dtype=float)
    safe = np.where(tau > 0, tau, 1.0)
    y = (cir_B(safe, kappa, sigma) * r - cir_A(safe, kappa, theta, sigma)) / safe
    return np.where(tau > 0, y, r)


def cir_long_yield(kappa: float, theta: float, sigma: float) -> float:
    """Limit of the CIR yields as tau -> infinity: 2 kappa theta / (gamma + kappa)."""
    return 2.0 * kappa * theta / (cir_gamma(kappa, sigma) + kappa)


def cir_mean(t, r0, kappa: float, theta: float):
    """E[r_t] = theta + (r0 - theta) e^{-kappa t} (same as Vasicek)."""
    return vasicek_mean(t, r0, kappa, theta)


def cir_var(t, r0, kappa: float, theta: float, sigma: float):
    """Var(r_t) = sigma^2 r0 (e^{-kappa t} - e^{-2 kappa t}) / kappa + sigma^2 theta (1 - e^{-kappa t})^2 / (2 kappa)."""
    e = np.exp(-kappa * np.asarray(t, dtype=float))
    return sigma**2 * r0 * (e - e * e) / kappa + sigma**2 * theta * (1.0 - e)**2 / (2.0 * kappa)


def cir_stationary(kappa: float, theta: float, sigma: float):
    """Shape nu = 2 kappa theta / sigma^2 and scale sigma^2 / (2 kappa) of the stationary Gamma law."""
    return feller_ratio(kappa, theta, sigma), sigma**2 / (2.0 * kappa)


def cir_exact(r0, kappa: float, theta: float, sigma: float, T: float, n_steps: int,
              n_paths: int = 1, seed=None):
    """Exact samples of the CIR rate on the grid t_k = k T / n_steps.

    The transition law is a scaled noncentral chi-square: given r_t, r_{t+h} = c X with
    c = sigma^2 (1 - e^{-kappa h}) / (4 kappa) and X noncentral chi-square with 4 kappa theta /
    sigma^2 degrees of freedom and noncentrality r_t e^{-kappa h} / c (Cox, Ingersoll and Ross,
    1985). Returns ``(t, r)``; every value is >= 0.
    """
    rng = np.random.default_rng(seed)
    h = T / n_steps
    c = -sigma**2 * np.expm1(-kappa * h) / (4.0 * kappa)
    df, a = 4.0 * kappa * theta / sigma**2, np.exp(-kappa * h)
    t = np.linspace(0.0, T, n_steps + 1)
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = r0
    for k in range(n_steps):
        r[:, k + 1] = c * rng.noncentral_chisquare(df, r[:, k] * a / c)
    return t, r


def cir_euler(r0, kappa: float, theta: float, sigma: float, T: float, n_steps: int,
              n_paths: int = 1, seed=None, dW=None):
    """Euler scheme with full truncation for CIR: the drift and diffusion use max(r, 0).

    The scheme can produce negative values (the exact process cannot); full truncation keeps
    them from entering the square root. With sigma sqrt(r) replaced by a constant it would be the
    Euler scheme for Vasicek. Returns ``(t, r)``.
    """
    rng = np.random.default_rng(seed)
    h = T / n_steps
    if dW is None:
        dW = rng.normal(0.0, np.sqrt(h), size=(n_paths, n_steps))
    n_paths = dW.shape[0]
    t = np.linspace(0.0, T, n_steps + 1)
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = r0
    for k in range(n_steps):
        x = np.maximum(r[:, k], 0.0)
        r[:, k + 1] = r[:, k] + kappa * (theta - x) * h + sigma * np.sqrt(x) * dW[:, k]
    return t, r


def vasicek_euler(r0, kappa: float, theta: float, sigma: float, T: float, n_steps: int,
                  n_paths: int = 1, seed=None, dW=None):
    """Euler scheme for Vasicek, r_{k+1} = r_k + kappa (theta - r_k) h + sigma dW_k. Returns ``(t, r)``."""
    rng = np.random.default_rng(seed)
    h = T / n_steps
    if dW is None:
        dW = rng.normal(0.0, np.sqrt(h), size=(n_paths, n_steps))
    n_paths = dW.shape[0]
    t = np.linspace(0.0, T, n_steps + 1)
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = r0
    for k in range(n_steps):
        r[:, k + 1] = r[:, k] + kappa * (theta - r[:, k]) * h + sigma * dW[:, k]
    return t, r


# ---------------------------------------------------------------- Monte Carlo bond prices

def integrated_rate(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Trapezoidal approximation of int_0^{t_k} r_u du along each path, shape of ``r``."""
    incr = 0.5 * (r[:, 1:] + r[:, :-1]) * np.diff(t)
    return np.concatenate([np.zeros((r.shape[0], 1)), np.cumsum(incr, axis=1)], axis=1)


def mc_bond_prices(I: np.ndarray):
    """Monte Carlo estimates of P(0, t_k) = E[exp(-int_0^{t_k} r)] and their standard errors."""
    D = np.exp(-I)
    return D.mean(axis=0), D.std(axis=0, ddof=1) / np.sqrt(D.shape[0])
