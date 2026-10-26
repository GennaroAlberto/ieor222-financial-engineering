"""The Feynman--Kac formula, pricing PDEs and Kolmogorov's equations (Chapter 12).

Conventions follow :mod:`ieor222.sde` and :mod:`ieor222.blackscholes`: ``x`` is the current
state (or stock price), ``K`` a strike, ``tau = T - t`` the time to maturity, ``r`` the interest
rate. Samplers take a ``seed``. Coefficient functions ``b(t, x)`` and ``sigma(t, x)`` take a
scalar time and an array of states. In the general functions ``sigma`` is the diffusion
coefficient of ``dX = b dt + sigma dW``; in the local volatility functions ``vol(t, x)`` is the
volatility of returns, as in ``def:sde-local-vol``, so that the diffusion coefficient of the
stock is ``x vol(t, x)``.

The finite-difference solvers are tools for the demos of Chapter 12; the schemes themselves
(Crank--Nicolson with a few implicit start-up steps) are analysed in Chapter 16.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded

from ieor222.blackscholes import call, d_plus_minus, digital_call, norm_cdf, norm_pdf

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 has only trapz

# ---------------------------------------------------------------- transition densities


def heat_kernel(s, z):
    """G(s, z) = exp(-z^2 / (2 s)) / sqrt(2 pi s): the N(0, s) density, for s > 0."""
    s, z = np.asarray(s, dtype=float), np.asarray(z, dtype=float)
    return np.exp(-0.5 * z * z / s) / np.sqrt(2.0 * np.pi * s)


def bm_density(t, x, s, y):
    """Transition density p(t, x; s, y) = G(s - t, y - x) of Brownian motion, for t < s."""
    return heat_kernel(np.asarray(s, float) - np.asarray(t, float), np.asarray(y, float) - np.asarray(x, float))


def gbm_density(t, x, s, y, mu, sigma):
    """Transition density of dS = mu S dt + sigma S dW from (t, x) to (s, y), for t < s, y > 0.

    log S_s given S_t = x is normal with mean log x + (mu - sigma^2/2)(s - t) and variance
    sigma^2 (s - t), so p(t, x; s, y) = G(sigma^2 (s - t), log(y / x) - (mu - sigma^2/2)(s - t)) / y.
    """
    tau = np.asarray(s, float) - np.asarray(t, float)
    x, y = np.asarray(x, float), np.asarray(y, float)
    z = np.log(y / x) - (mu - 0.5 * sigma**2) * tau
    return heat_kernel(sigma**2 * tau, z) / y


def ou_density(t, x, s, y, kappa, theta, sigma):
    """Transition density of dX = kappa (theta - X) dt + sigma dW (``prop:sde-ou``), for t < s."""
    tau = np.asarray(s, float) - np.asarray(t, float)
    m = theta + (np.asarray(x, float) - theta) * np.exp(-kappa * tau)
    v = sigma**2 * (1.0 - np.exp(-2.0 * kappa * tau)) / (2.0 * kappa)
    return heat_kernel(v, np.asarray(y, float) - m)


def kolmogorov_residuals(density, b, sigma, t, x, s, y, h_time=1e-4, h_space=None):
    """Finite-difference residuals of the backward and forward equations at one point.

    ``density(t, x, s, y)`` is a transition density, and ``b``, ``sigma`` the coefficients of
    dX = b(t, X) dt + sigma(t, X) dW. Returns ``(backward, forward)`` with
    backward = d_t p + b(t, x) d_x p + sigma(t, x)^2 / 2 d_xx p, in the variables (t, x), and
    forward = d_s p - [ d_yy(sigma(s, y)^2 p) / 2 - d_y(b(s, y) p) ], in the variables (s, y),
    computed by central differences. Both should vanish (``thm:fk-kolmogorov``), up to the
    discretisation error, which is of order h^2.
    """
    hx = h_space if h_space is not None else 1e-3 * max(1.0, abs(x))
    hy = h_space if h_space is not None else 1e-3 * max(1.0, abs(y))
    p = lambda tt, xx: density(tt, xx, s, y)
    back = ((p(t + h_time, x) - p(t - h_time, x)) / (2 * h_time)
            + b(t, x) * (p(t, x + hx) - p(t, x - hx)) / (2 * hx)
            + 0.5 * sigma(t, x) ** 2 * (p(t, x + hx) - 2 * p(t, x) + p(t, x - hx)) / hx**2)
    q = lambda ss, yy: density(t, x, ss, yy)
    A = lambda yy: sigma(s, yy) ** 2 * q(s, yy)
    B = lambda yy: b(s, yy) * q(s, yy)
    fwd = ((q(s + h_time, y) - q(s - h_time, y)) / (2 * h_time)
           - 0.5 * (A(y + hy) - 2 * A(y) + A(y - hy)) / hy**2
           + (B(y + hy) - B(y - hy)) / (2 * hy))
    return float(back), float(fwd)


# ---------------------------------------------------------------- the heat equation


def heat_solution(phi, s, z, n_nodes: int = 8001, width: float = 12.0):
    """u(s, z) = E[phi(z + sqrt(s) Y)], Y ~ N(0, 1): the solution of d_s u = d_zz u / 2, u(0, .) = phi.

    Computed as the convolution of phi with the heat kernel (``prop:fk-heat``), by the
    trapezoidal rule on y in [-width, width] with ``n_nodes`` nodes. ``phi`` must accept arrays.
    ``s`` and ``z`` are scalars.
    """
    y = np.linspace(-width, width, n_nodes)
    w = norm_pdf(y)
    return float(_trapezoid(phi(z + np.sqrt(s) * y) * w, y))


def bs_price_by_heat(g, t, x, T, r, sigma, **kw):
    """Black--Scholes price of the claim g(S_T) at (t, x), through the heat equation.

    v(t, x) = e^{-r tau} u(sigma^2 tau, log x + (r - sigma^2 / 2) tau) with u the solution of the
    heat equation with initial condition phi(z) = g(e^z) (``prop:fk-heat``).
    """
    tau = T - t
    if tau <= 0:
        return float(g(np.asarray(x, dtype=float)))
    phi = lambda z: g(np.exp(z))
    return np.exp(-r * tau) * heat_solution(phi, sigma**2 * tau, np.log(x) + (r - 0.5 * sigma**2) * tau, **kw)


# ---------------------------------------------------------------- Monte Carlo for Feynman--Kac


def fk_monte_carlo(b, sigma, h, x0, T: float, n_steps: int, n_paths: int, k=None, t0: float = 0.0,
                   seed=None, batch: int = 50_000) -> dict:
    """Monte Carlo estimate of E[exp(-int_{t0}^T k(u, X_u) du) h(X_T)] with X_{t0} = x0.

    X is simulated by the Euler--Maruyama scheme (``thm:sde-euler``) on a uniform grid of
    ``n_steps`` steps, and the time integral of k by the trapezoidal rule on the same grid.
    Returns a dict with the estimate ``mean``, its standard error ``se``, and ``n_paths``.
    The estimate has a discretisation bias of order T / n_steps besides its statistical error.
    """
    rng = np.random.default_rng(seed)
    dt = (T - t0) / n_steps
    total = total_sq = 0.0
    done = 0
    while done < n_paths:
        m = min(batch, n_paths - done)
        x = np.full(m, float(x0))
        integral = np.zeros(m)
        t = t0
        k_prev = k(t, x) if k is not None else None
        for _ in range(n_steps):
            dW = rng.normal(0.0, np.sqrt(dt), size=m)
            x = x + b(t, x) * dt + sigma(t, x) * dW
            t += dt
            if k is not None:
                k_new = k(t, x)
                integral += 0.5 * (k_prev + k_new) * dt
                k_prev = k_new
        val = h(x) * np.exp(-integral)
        total += val.sum()
        total_sq += (val**2).sum()
        done += m
    mean = total / n_paths
    var = max(total_sq / n_paths - mean**2, 0.0)
    return dict(mean=mean, se=np.sqrt(var / n_paths), n_paths=n_paths)


def local_vol_mc(g, S0: float, T: float, r: float, vol, n_steps: int, n_paths: int, seed=None) -> dict:
    """Monte Carlo price e^{-r T} E^Q[g(S_T)] in the local volatility model dS = r S dt + vol(t, S) S dW.

    The scheme is Euler--Maruyama for log S, which keeps the price positive:
    Y_{k+1} = Y_k + (r - vol^2 / 2) h + vol (W_{t_{k+1}} - W_{t_k}), with vol = vol(t_k, e^{Y_k}).
    Returns a dict with ``mean`` and ``se``.
    """
    b = lambda t, y: r - 0.5 * vol(t, np.exp(y)) ** 2
    s = lambda t, y: vol(t, np.exp(y))
    res = fk_monte_carlo(b, s, lambda y: g(np.exp(y)), np.log(S0), T, n_steps, n_paths, seed=seed)
    disc = np.exp(-r * T)
    return dict(mean=disc * res["mean"], se=disc * res["se"], n_paths=n_paths)


def skew_vol(t, x, level: float = 0.25, skew: float = 0.12, slope: float = 2.0, x_ref: float = 100.0):
    """A local volatility with a downward skew: level - skew * tanh(slope * log(x / x_ref)).

    It is smooth, takes values in (level - skew, level + skew), and is Lipschitz in log x with
    constant skew * slope, so the hypotheses of ``cor:fk-pricing-pde`` hold when level > skew.
    """
    return level - skew * np.tanh(slope * np.log(np.asarray(x, dtype=float) / x_ref))


# ---------------------------------------------------------------- finite differences


def _theta_step(u, dtau, theta, coef_new, coef_old, bc_new):
    """One theta-scheme step for d_tau u = alpha u_{j-1} + beta u_j + gamma u_{j+1} (interior rows).

    ``coef_*`` are (alpha, beta, gamma) on the interior nodes at the new and old times; the
    boundary values ``bc_new`` = (u_0, u_J) at the new time are prescribed (Dirichlet).
    """
    a1, b1, c1 = coef_new
    a0, b0, c0 = coef_old
    rhs = u[1:-1] + (1 - theta) * dtau * (a0 * u[:-2] + b0 * u[1:-1] + c0 * u[2:])
    rhs[0] += theta * dtau * a1[0] * bc_new[0]
    rhs[-1] += theta * dtau * c1[-1] * bc_new[1]
    ab = np.zeros((3, len(rhs)))
    ab[0, 1:] = -theta * dtau * c1[:-1]
    ab[1] = 1.0 - theta * dtau * b1
    ab[2, :-1] = -theta * dtau * a1[1:]
    new = np.empty_like(u)
    new[1:-1] = solve_banded((1, 1), ab, rhs)
    new[0], new[-1] = bc_new
    return new


def _march(u0, taus, coef, bc, rannacher: int, keep: bool):
    """Crank--Nicolson in tau with ``rannacher`` initial steps each split into two implicit half-steps."""
    out = [u0.copy()] if keep else None
    u = u0.copy()
    for n in range(len(taus) - 1):
        t0, t1 = taus[n], taus[n + 1]
        if n < rannacher:
            mid = 0.5 * (t0 + t1)
            u = _theta_step(u, mid - t0, 1.0, coef(mid), coef(t0), bc(mid))
            u = _theta_step(u, t1 - mid, 1.0, coef(t1), coef(mid), bc(t1))
        else:
            u = _theta_step(u, t1 - t0, 0.5, coef(t1), coef(t0), bc(t1))
        if keep:
            out.append(u.copy())
    return (u, np.array(out)) if keep else (u, None)


def solve_pricing_pde(g, T: float, r: float, vol, x_min: float, x_max: float, n_x: int = 400,
                      n_t: int = 200, boundary=None, rannacher: int = 2, keep: bool = False) -> dict:
    """Solve d_t v + r x d_x v + vol(t, x)^2 x^2 d_xx v / 2 = r v on [0, T) x (x_min, x_max), v(T, .) = g.

    This is the pricing equation of ``cor:fk-pricing-pde``. The equation is written in time to
    maturity tau = T - t and log-price y = log x, where it reads
    d_tau u = (vol^2 / 2) u_yy + (r - vol^2 / 2) u_y - r u, and is discretised on a uniform grid
    of ``n_x + 1`` points in y by central differences and Crank--Nicolson in tau (``n_t`` steps,
    the first ``rannacher`` of them replaced by two implicit half-steps, which damps the
    oscillations that a non-smooth payoff causes). The values at x_min and x_max are prescribed:
    by default the zero-volatility price e^{-r tau} g(x e^{r tau}), which is accurate far from
    where g bends; ``boundary(tau, x)`` overrides it.

    Returns a dict with ``x`` (the grid, increasing) and ``v`` (prices at t = 0 on the grid);
    with ``keep=True`` also ``t`` (increasing calendar times) and ``surface`` (shape
    (n_t + 1, n_x + 1), one row per time in ``t``).
    """
    y = np.linspace(np.log(x_min), np.log(x_max), n_x + 1)
    x = np.exp(y)
    dy = y[1] - y[0]
    if boundary is None:
        boundary = lambda tau, xx: np.exp(-r * tau) * g(xx * np.exp(r * tau))
    xi = x[1:-1]

    def coef(tau):
        s2 = vol(T - tau, xi) ** 2
        a, c = 0.5 * s2, r - 0.5 * s2
        return a / dy**2 - c / (2 * dy), -2 * a / dy**2 - r, a / dy**2 + c / (2 * dy)

    def bc(tau):
        return float(boundary(tau, x[0])), float(boundary(tau, x[-1]))

    taus = np.linspace(0.0, T, n_t + 1)
    u0 = np.asarray(g(x), dtype=float)
    u, hist = _march(u0, taus, coef, bc, rannacher, keep)
    out = dict(x=x, v=u)
    if keep:
        out.update(t=T - taus[::-1], surface=hist[::-1])
    return out


def price_at(res: dict, x0) -> np.ndarray:
    """Interpolate a finite-difference solution (``solve_pricing_pde``) at the price(s) x0, cubically in log x."""
    return CubicSpline(np.log(res["x"]), res["v"])(np.log(np.asarray(x0, dtype=float)))


def solve_forward_pde(S0: float, T: float, r: float, vol, x_min: float, x_max: float,
                      n_x: int = 1200, n_t: int = 400, s0: float = 0.01, rannacher: int = 2) -> dict:
    """Solve Kolmogorov's forward equation for the log-price in the local volatility model.

    Y = log S solves dY = (r - vol^2 / 2) dt + vol dW under Q, so its density q(s, y) solves
    d_s q = d_yy(a q) - d_y(c q), with a = vol(s, e^y)^2 / 2 and c = r - a (``thm:fk-kolmogorov``).
    The equation is discretised as the transpose of the backward operator (so that total mass is
    conserved up to the boundary), with q = 0 at both ends, from the starting time ``s0`` > 0,
    where q is taken to be the normal density of the model with the volatility frozen at
    vol(0, S0); the error of this start is of order s0.

    Returns a dict with ``x`` (price grid), ``y`` (log grid), ``q`` (density of log S_T on the
    grid) and ``density`` (density of S_T on the price grid, q(y) / x).
    """
    y = np.linspace(np.log(x_min), np.log(x_max), n_x + 1)
    x = np.exp(y)
    dy = y[1] - y[0]
    v0 = vol(0.0, S0)
    q = heat_kernel(v0**2 * s0, y - np.log(S0) - (r - 0.5 * v0**2) * s0)

    def coef(s):
        a = 0.5 * vol(s, x) ** 2
        c = r - a
        # row j of the transpose: q_{j-1} * (a_{j-1}/dy^2 + c_{j-1}/(2dy)) - 2 a_j/dy^2 q_j
        #                        + q_{j+1} * (a_{j+1}/dy^2 - c_{j+1}/(2dy))
        lo = a[:-2] / dy**2 + c[:-2] / (2 * dy)
        mid = -2 * a[1:-1] / dy**2
        hi = a[2:] / dy**2 - c[2:] / (2 * dy)
        return lo, mid, hi

    times = np.linspace(s0, T, n_t + 1)
    q, _ = _march(q, times, coef, lambda s: (0.0, 0.0), rannacher, False)
    return dict(x=x, y=y, q=q, density=q / x)


def density_at(res: dict, x) -> np.ndarray:
    """Density of S_T at the price(s) x from ``solve_forward_pde``, interpolated cubically in log x."""
    x = np.asarray(x, dtype=float)
    return CubicSpline(res["y"], res["q"])(np.log(x)) / x


def prices_from_density(res: dict, strikes, T: float, r: float) -> np.ndarray:
    """Call prices e^{-r T} int (x - K)^+ p_T(x) dx from a density of log S_T on a grid (trapezoid rule)."""
    y, q = res["y"], res["q"]
    x = np.exp(y)
    K = np.atleast_1d(np.asarray(strikes, dtype=float))
    vals = np.maximum(x[None, :] - K[:, None], 0.0) * q[None, :]
    return np.exp(-r * T) * _trapezoid(vals, y, axis=1)


# ---------------------------------------------------------------- digital and pay-later options


def digital_delta(x, K, tau, r, sigma):
    """Delta of the digital call, as e^{-r tau} (K / x) p(t, x; T, K) with p the GBM density under Q.

    This equals e^{-r tau} N'(d_-) / (sigma x sqrt(tau)), the derivative in x of e^{-r tau} N(d_-);
    needs tau > 0.
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-r * tau) * (K / x) * gbm_density(0.0, x, tau, K, r, sigma)


def pay_later_fee(S0, K, T, r, sigma):
    """Fee P = C_0 / D_0 of the pay-later call: the premium paid at T if and only if S_T > K.

    Equivalently P = E^Q[S_T - K | S_T > K] = S0 e^{rT} N(d_+) / N(d_-) - K.
    """
    return call(S0, K, T, r, sigma) / digital_call(S0, K, T, r, sigma)


def pay_later_value(x, K, tau, r, sigma, P):
    """Value c(t, x) - P D(t, x) at time t of the holder's position (call minus P digitals)."""
    return call(x, K, tau, r, sigma) - P * digital_call(x, K, tau, r, sigma)


def pay_later_delta(x, K, tau, r, sigma, P):
    """Delta N(d_+) - P * (digital delta) of the holder's position; needs tau > 0."""
    d_plus, _ = d_plus_minus(x, K, tau, r, sigma)
    return norm_cdf(d_plus) - P * digital_delta(x, K, tau, r, sigma)


def digital_hedge_variance(S0, K, T, r, sigma):
    """E^Q int_0^T (e^{-rt} sigma S_t Delta_t)^2 dt for the digital call: e^{-2rT} p (1 - p), p = N(d_-)."""
    _, d_minus = d_plus_minus(S0, K, T, r, sigma)
    p = norm_cdf(d_minus)
    return np.exp(-2 * r * T) * p * (1 - p)


def digital_hedge_integral(d) -> float:
    """(1 / 2 pi) int_0^{pi/2} exp(-d^2 / (1 + sin u)) du, computed by adaptive quadrature.

    With d = d_-(T, S0), e^{-2rT} times this number is E^Q int_0^T (e^{-rt} sigma S_t Delta_t)^2 dt for
    the digital call, written as a deterministic integral (exercises of Chapter 12); it equals
    N(d) (1 - N(d)).
    """
    from scipy.integrate import quad
    val, _ = quad(lambda u: np.exp(-d * d / (1.0 + np.sin(u))), 0.0, np.pi / 2, epsabs=1e-13, epsrel=1e-12)
    return val / (2.0 * np.pi)


# ---------------------------------------------------------------- a quadratic potential


def cameron_martin(lam: float, T: float, x: float = 0.0) -> float:
    """E[exp(-lam int_0^T (x + W_s)^2 ds)] = cosh(cT)^{-1/2} exp(-c tanh(cT) x^2 / 2), c = sqrt(2 lam).

    For lam < 0 the right side is continued as cos(wT)^{-1/2} exp(w tan(wT) x^2 / 2), w = sqrt(-2 lam),
    which is finite only for w T < pi / 2; beyond, ``inf`` is returned.
    """
    if lam >= 0:
        c = np.sqrt(2.0 * lam)
        return float(np.cosh(c * T) ** -0.5 * np.exp(-0.5 * c * np.tanh(c * T) * x * x))
    w = np.sqrt(-2.0 * lam)
    if w * T >= np.pi / 2:
        return float("inf")
    return float(np.cos(w * T) ** -0.5 * np.exp(0.5 * w * np.tan(w * T) * x * x))
