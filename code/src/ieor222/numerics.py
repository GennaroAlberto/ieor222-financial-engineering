"""Monte Carlo, finite differences, the American put on a grid, and a small PINN (Chapter 16).

Conventions follow :mod:`ieor222.blackscholes`: ``S0`` or ``x`` is a stock price, ``K`` a strike,
``T`` a maturity, ``r`` the interest rate and ``sigma`` the volatility. Samplers take a ``seed``.

The finite-difference functions solve, in time to maturity ``tau = T - t`` and log-price
``y = log x``, the constant-coefficient equation

    d_tau u = a u_yy + b u_y - c u,   with a = sigma^2 / 2, b = r - sigma^2 / 2, c = r

for the Black--Scholes model (``thm:num-schemes``), on a uniform grid of ``n_x + 1`` points with
Dirichlet values at both ends, by the theta-scheme: theta = 0 is the explicit scheme, 1 the
implicit one and 1/2 Crank--Nicolson. A price in the notes' sense is then
``v(t, x) = u(T - t, log x)``.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import ndtri

from ieor222.blackscholes import call, put

# ---------------------------------------------------------------- Monte Carlo


def mc_summary(samples, level: float = 0.95) -> dict:
    """Sample mean, standard error and the asymptotic confidence interval of ``thm:num-mc``.

    With M samples, mean m and sample standard deviation s (divisor M - 1), the interval is
    m +- z s / sqrt(M), where z = N^{-1}((1 + level) / 2) (z = 1.96 for level 0.95).
    """
    x = np.asarray(samples, dtype=float).ravel()
    M = x.size
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(M)
    z = float(ndtri(0.5 + 0.5 * level))
    return dict(mean=mean, sd=sd, se=se, half_width=z * se, low=mean - z * se, high=mean + z * se, n=M)


def bs_terminal(S0: float, T: float, r: float, sigma: float, n_paths: int, seed=None) -> np.ndarray:
    """Exact samples of S_T = S0 exp(sigma sqrt(T) Y + (r - sigma^2 / 2) T) under Q."""
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal(n_paths)
    return S0 * np.exp(sigma * np.sqrt(T) * Y + (r - 0.5 * sigma**2) * T)


def mc_european(payoff, S0: float, T: float, r: float, sigma: float, n_paths: int, seed=None,
                level: float = 0.95) -> dict:
    """Monte Carlo price e^{-rT} E^Q[payoff(S_T)] with exact sampling, and its confidence interval."""
    S = bs_terminal(S0, T, r, sigma, n_paths, seed)
    return mc_summary(np.exp(-r * T) * payoff(S), level)


def call_payoff_sd(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Standard deviation under Q of the discounted call payoff e^{-rT} (S_T - K)^+.

    E[((S_T - K)^+)^2] = S0^2 e^{(2r + sigma^2) T} N(d_+ + sigma sqrt(T)) - 2 K S0 e^{rT} N(d_+) + K^2 N(d_-).
    """
    from ieor222.blackscholes import d_plus_minus, norm_cdf
    dp, dm = d_plus_minus(S0, K, T, r, sigma)
    second = (S0**2 * np.exp((2 * r + sigma**2) * T) * norm_cdf(dp + sigma * np.sqrt(T))
              - 2 * K * S0 * np.exp(r * T) * norm_cdf(dp) + K * K * norm_cdf(dm))
    mean = np.exp(r * T) * call(S0, K, T, r, sigma)
    return float(np.exp(-r * T) * np.sqrt(second - mean**2))


def euler_terminal(S0: float, T: float, r: float, sigma: float, n_steps: int, n_paths: int,
                   seed=None) -> np.ndarray:
    """S^h_T from the Euler--Maruyama scheme for dS = r S dt + sigma S dW on S itself (``thm:sde-euler``).

    Each step multiplies the price by 1 + r h + sigma (W_{t+h} - W_t), h = T / n_steps.
    """
    rng = np.random.default_rng(seed)
    h = T / n_steps
    S = np.full(n_paths, float(S0))
    for _ in range(n_steps):
        S *= 1.0 + r * h + sigma * np.sqrt(h) * rng.standard_normal(n_paths)
    return S


def power_claim_price(S0: float, T: float, r: float, sigma: float) -> float:
    """e^{-rT} E^Q[S_T^2] = S0^2 e^{(r + sigma^2) T}, the price of the claim S_T^2."""
    return float(S0**2 * np.exp((r + sigma**2) * T))


def power_claim_variance(S0: float, T: float, r: float, sigma: float) -> float:
    """Var(e^{-rT} S_T^2) under Q: S0^4 e^{-2rT} (e^{(4r + 6 sigma^2) T} - e^{(4r + 2 sigma^2) T})."""
    return float(S0**4 * np.exp(-2 * r * T) * (np.exp((4 * r + 6 * sigma**2) * T) - np.exp((4 * r + 2 * sigma**2) * T)))


def euler_power_moments(S0: float, T: float, r: float, sigma: float, n_steps) -> tuple:
    """Exact mean and variance of Y_h = e^{-rT} (S^h_T)^2 for the Euler scheme on S.

    The factors 1 + r h + sigma sqrt(h) Z are independent, with second moment
    m2 = (1 + r h)^2 + sigma^2 h and fourth moment m4 = a^4 + 6 a^2 sigma^2 h + 3 sigma^4 h^2,
    a = 1 + r h; so E[Y_h] = e^{-rT} S0^2 m2^n and Var(Y_h) = e^{-2rT} S0^4 (m4^n - m2^(2n)).
    ``n_steps`` may be an array.
    """
    n = np.asarray(n_steps, dtype=float)
    lm2, lm4 = _log_factor_moments(T, r, sigma, n)
    mean = np.exp(-r * T) * S0**2 * np.exp(n * lm2)
    var = np.exp(-2 * r * T) * S0**4 * np.exp(2 * n * lm2) * np.expm1(n * (lm4 - 2 * lm2))
    return mean, var


def _log_factor_moments(T, r, sigma, n):
    """log m2 and log m4 for the Euler factor 1 + r h + sigma sqrt(h) Z, computed with log1p."""
    h = T / n
    a = 1.0 + r * h
    lm2 = np.log1p(2 * r * h + (r * h) ** 2 + sigma**2 * h)
    lm4 = np.log1p((a**4 - 1.0) + 6.0 * a * a * sigma**2 * h + 3.0 * sigma**4 * h * h)
    return lm2, lm4


def euler_power_bias(S0: float, T: float, r: float, sigma: float, n_steps):
    """Bias E[Y_h] - S0^2 e^{(r + sigma^2) T} of the Euler price of S_T^2 (``exa:num-power``), without cancellation."""
    n = np.asarray(n_steps, dtype=float)
    lm2, _ = _log_factor_moments(T, r, sigma, n)
    return power_claim_price(S0, T, r, sigma) * np.expm1(n * lm2 - (2 * r + sigma**2) * T)


def power_claim_rmse(S0: float, T: float, r: float, sigma: float, n_steps, n_paths):
    """Root mean square error sqrt(bias^2 + Var / M) of the Euler Monte Carlo price of S_T^2.

    ``n_steps = 0`` means exact sampling of S_T (no bias). Arrays broadcast.
    """
    n = np.asarray(n_steps)
    M = np.asarray(n_paths, dtype=float)
    var_exact = power_claim_variance(S0, T, r, sigma)
    nn = np.where(n > 0, n, 1)
    _, var = euler_power_moments(S0, T, r, sigma, nn)
    bias = np.where(n > 0, euler_power_bias(S0, T, r, sigma, nn), 0.0)
    var = np.where(n > 0, var, var_exact)
    return np.sqrt(bias**2 + var / M)


def balanced_allocation(eps: float, c: float, v: float, T: float = 1.0) -> tuple:
    """Steps and paths of ``prop:num-mc-cost``: h = eps / (sqrt(2) c), M = ceil(2 v / eps^2).

    With |bias| <= c h and variance <= v, the root mean square error is at most eps. Returns
    ``(n_steps, n_paths, cost)`` with n_steps = ceil(T / h) and cost = n_steps * n_paths.
    """
    h = eps / (np.sqrt(2.0) * c)
    n = int(np.ceil(T / h))
    M = int(np.ceil(2.0 * v / eps**2))
    return n, M, n * M


# ---------------------------------------------------------------- the theta-scheme


def bs_log_coefficients(r: float, sigma: float) -> tuple:
    """(a, b, c) = (sigma^2 / 2, r - sigma^2 / 2, r): the Black--Scholes equation in log-price."""
    return 0.5 * sigma**2, r - 0.5 * sigma**2, r


def _bands(n_int: int, a: float, b: float, c: float, h: float):
    """Sub-, main and super-diagonal coefficients (p_-, p_0, p_+) of the central-difference operator."""
    pm = a / h**2 - b / (2 * h)
    p0 = -2 * a / h**2 - c
    pp = a / h**2 + b / (2 * h)
    return pm, p0, pp


def apply_operator(u: np.ndarray, a: float, b: float, c: float, h: float) -> np.ndarray:
    """(L_h u)_j = a (u_{j+1} - 2u_j + u_{j-1}) / h^2 + b (u_{j+1} - u_{j-1}) / (2h) - c u_j, interior j."""
    pm, p0, pp = _bands(len(u) - 2, a, b, c, h)
    return pm * u[:-2] + p0 * u[1:-1] + pp * u[2:]


def psor(ab: np.ndarray, rhs: np.ndarray, obstacle: np.ndarray, x0: np.ndarray, omega: float = 1.2,
         tol: float = 1e-10, max_iter: int = 10_000) -> tuple:
    """Projected SOR for the complementarity problem z >= g, Bz - q >= 0, (z - g).(Bz - q) = 0.

    ``ab`` holds the tridiagonal matrix B in the banded form of ``scipy.linalg.solve_banded``
    (row 0: super-diagonal, row 1: diagonal, row 2: sub-diagonal); ``rhs`` is q and ``obstacle``
    is g. The nodes are swept in red-black order (all odd, then all even indices), which is
    Gauss--Seidel for a reordering of the unknowns, so Cryer's theorem applies to it as to the
    natural order (``prop:num-american-grid``). Returns ``(z, iterations)``; stops when the
    largest change in a sweep is below ``tol`` (relative to max(1, |z|)).
    """
    up, dg, lo = ab[0, 1:], ab[1], ab[2, :-1]   # B[j, j+1], B[j, j], B[j+1, j]
    z = np.maximum(np.array(x0, dtype=float), obstacle)
    m = len(z)
    idx = (np.arange(1, m, 2), np.arange(0, m, 2))
    for it in range(1, max_iter + 1):
        z_old = z.copy()
        for J in idx:
            s = rhs[J].copy()
            left = J[J > 0]
            s[J > 0] -= lo[left - 1] * z[left - 1]
            right = J[J < m - 1]
            s[J < m - 1] -= up[right] * z[right + 1]
            gs = s / dg[J]
            z[J] = np.maximum(obstacle[J], z[J] + omega * (gs - z[J]))
        if np.max(np.abs(z - z_old)) <= tol * max(1.0, np.max(np.abs(z))):
            return z, it
    return z, max_iter


def theta_scheme(u0: np.ndarray, y: np.ndarray, tau_max: float, n_t: int, a: float, b: float, c: float,
                 theta: float, boundary, rannacher: int = 0, obstacle=None, omega: float = 1.2,
                 keep: bool = False, solver: str = "psor") -> dict:
    """Solve d_tau u = a u_yy + b u_y - c u on the grid ``y`` for tau in [0, tau_max].

    ``u0`` is u(0, .) on the grid, ``boundary(tau)`` returns the Dirichlet values (left, right).
    The theta-scheme with ``n_t`` steps of size k = tau_max / n_t reads
    (I - theta k A) U^{n+1} = (I + (1 - theta) k A) U^n + boundary terms. With ``rannacher = m``
    the first m steps are each replaced by two implicit half-steps (Rannacher's start-up; m = 2
    gives the four half-steps of Giles and Carter). With ``obstacle`` (the payoff on the grid) every
    step solves the complementarity problem of the American option instead of a linear system,
    by projected SOR (``solver="psor"``, the iteration counts are returned) or by the direct
    algorithm of Brennan and Schwartz (``solver="brennan-schwartz"``).

    The explicit part of every step uses the boundary values ``boundary(t0)`` at the old time
    level, whatever the end values of ``u0`` are: the scheme of ``def:num-theta``.

    Returns a dict with ``u`` (values at tau_max) and ``iterations``; with ``keep=True`` also
    ``taus`` and ``history`` (one row per time level).
    """
    h = y[1] - y[0]
    n_int = len(y) - 2
    pm, p0, pp = _bands(n_int, a, b, c, h)
    u = np.array(u0, dtype=float)
    k = tau_max / n_t
    plan = []
    tau = 0.0
    for n in range(n_t):
        if n < rannacher:
            plan += [(tau, k / 2, 1.0), (tau + k / 2, k / 2, 1.0)]
        else:
            plan.append((tau, k, theta))
        tau += k
    taus, hist, iters = [0.0], [u.copy()] if keep else None, []
    for t0, dt, th in plan:
        t1 = t0 + dt
        left, right = boundary(t1)
        u[0], u[-1] = boundary(t0)          # old boundary values, also at the first step
        Au = pm * u[:-2] + p0 * u[1:-1] + pp * u[2:]
        rhs = u[1:-1] + (1 - th) * dt * Au
        rhs[0] += th * dt * pm * left
        rhs[-1] += th * dt * pp * right
        ab = np.zeros((3, n_int))
        ab[0, 1:] = -th * dt * pp
        ab[1] = 1.0 - th * dt * p0
        ab[2, :-1] = -th * dt * pm
        new = np.empty_like(u)
        new[0], new[-1] = left, right
        if obstacle is None:
            if th == 0.0:
                new[1:-1] = rhs
            else:
                new[1:-1] = solve_banded((1, 1), ab, rhs)
        elif solver == "brennan-schwartz":
            new[1:-1] = brennan_schwartz(ab, rhs, obstacle[1:-1])
        else:
            z, it = psor(ab, rhs, obstacle[1:-1], u[1:-1], omega=omega)
            new[1:-1] = z
            iters.append(it)
        u = new
        if keep:
            taus.append(t1)
            hist.append(u.copy())
    out = dict(u=u, iterations=np.array(iters if iters else [0], dtype=int))
    if keep:
        out.update(taus=np.array(taus), history=np.array(hist))
    return out


def log_grid(S0: float, T: float, sigma: float, n_x: int, width: float = 6.0, center: float | None = None):
    """Uniform grid in y = log x with n_x intervals, symmetric around log(center) (default S0).

    The half-width is ``width`` standard deviations sigma sqrt(T) of log S_T, and the grid is
    shifted so that log S0 is a node (n_x even).
    """
    if n_x % 2:
        raise ValueError("n_x must be even so that log S0 is a node")
    c = np.log(S0 if center is None else center)
    half = width * sigma * np.sqrt(T)
    return c + np.linspace(-half, half, n_x + 1)


def european_fd(kind: str, S0: float, K: float, T: float, r: float, sigma: float, n_x: int, n_t: int,
                theta: float = 0.5, rannacher: int = 0, width: float = 6.0, keep: bool = False) -> dict:
    """European call or put by the theta-scheme in log-price, on a grid centred at S0.

    Dirichlet values: call 0 at the bottom and x - K e^{-r tau} at the top; put
    K e^{-r tau} - x at the bottom and 0 at the top (the discounted payoff at zero volatility,
    which the solution approaches far from the strike).
    Returns a dict with ``x`` (price grid), ``v`` (prices at t = 0), ``price`` (at S0, a node),
    ``delta`` and ``gamma`` (central differences in x at S0), and the solver output.
    """
    a, b, c = bs_log_coefficients(r, sigma)
    y = log_grid(S0, T, sigma, n_x, width)
    x = np.exp(y)
    if kind == "call":
        g = np.maximum(x - K, 0.0)
        bc = lambda tau: (0.0, x[-1] - K * np.exp(-r * tau))
    elif kind == "put":
        g = np.maximum(K - x, 0.0)
        bc = lambda tau: (K * np.exp(-r * tau) - x[0], 0.0)
    else:
        raise ValueError("kind must be 'call' or 'put'")
    res = theta_scheme(g, y, T, n_t, a, b, c, theta, bc, rannacher=rannacher, keep=keep)
    v = res["u"]
    j = n_x // 2
    out = dict(x=x, y=y, v=v, price=float(v[j]), **_greeks_at(x, v, j))
    if keep:
        out.update(taus=res["taus"], history=res["history"])
    return out


def _greeks_at(x, v, j):
    """Delta and gamma at node j of a non-uniform price grid, by the three-point formulas."""
    h1, h2 = x[j] - x[j - 1], x[j + 1] - x[j]
    delta = (h1**2 * v[j + 1] - h2**2 * v[j - 1] + (h2**2 - h1**2) * v[j]) / (h1 * h2 * (h1 + h2))
    gamma = 2.0 * (h1 * v[j + 1] - (h1 + h2) * v[j] + h2 * v[j - 1]) / (h1 * h2 * (h1 + h2))
    return dict(delta=float(delta), gamma=float(gamma))


def grid_greeks(x, v):
    """Delta and gamma at every interior node of a (non-uniform) price grid."""
    h1, h2 = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    vm, v0, vp = v[:-2], v[1:-1], v[2:]
    delta = (h1**2 * vp - h2**2 * vm + (h2**2 - h1**2) * v0) / (h1 * h2 * (h1 + h2))
    gamma = 2.0 * (h1 * vp - (h1 + h2) * v0 + h2 * vm) / (h1 * h2 * (h1 + h2))
    return delta, gamma


def convergence_table(kind: str, S0: float, K: float, T: float, r: float, sigma: float, levels,
                      scheme: str, lam: float = 0.4, ratio: float = 1.0, width: float = 6.0) -> dict:
    """Errors at S0 of the three schemes on grids with n_x = levels[i] intervals.

    ``scheme`` is 'explicit' (k = lam h^2 / a), 'implicit', 'cn' or 'cn-rannacher' (k = ratio h /
    sigma, so that k / h is fixed). Returns arrays ``n_x``, ``n_t``, ``h``, ``error`` (price minus
    Black--Scholes), ``delta_error`` and ``gamma_error``.
    """
    from ieor222.blackscholes import greeks
    exact = (call if kind == "call" else put)(S0, K, T, r, sigma)
    gk = greeks(S0, K, T, r, sigma, kind)
    a = 0.5 * sigma**2
    rows = []
    for n_x in levels:
        h = 2 * width * sigma * np.sqrt(T) / n_x
        if scheme == "explicit":
            n_t = int(np.ceil(T * a / (lam * h * h)))
            th, ran = 0.0, 0
        else:
            n_t = int(np.ceil(T * sigma / (ratio * h)))
            th, ran = (1.0, 0) if scheme == "implicit" else (0.5, 2 if scheme == "cn-rannacher" else 0)
        res = european_fd(kind, S0, K, T, r, sigma, n_x, n_t, theta=th, rannacher=ran, width=width)
        rows.append((n_x, n_t, h, res["price"] - exact, res["delta"] - gk["delta"], res["gamma"] - gk["gamma"]))
    rows = np.array(rows)
    return dict(n_x=rows[:, 0].astype(int), n_t=rows[:, 1].astype(int), h=rows[:, 2], error=rows[:, 3],
                delta_error=rows[:, 4], gamma_error=rows[:, 5])


# ---------------------------------------------------------------- stability


def explicit_heat(u0: np.ndarray, lam: float, n_steps: int, keep_every: int = 0):
    """Explicit scheme U_j <- U_j + lam (U_{j+1} - 2U_j + U_{j-1}) for the heat equation, zero boundary values.

    ``u0`` includes the two boundary nodes. Returns the final vector (and the history every
    ``keep_every`` steps if requested).
    """
    u = np.array(u0, dtype=float)
    hist = [u.copy()]
    for n in range(1, n_steps + 1):
        u[1:-1] = u[1:-1] + lam * (u[2:] - 2 * u[1:-1] + u[:-2])
        if keep_every and n % keep_every == 0:
            hist.append(u.copy())
    return (u, np.array(hist)) if keep_every else u


def heat_eigenvalues(n_x: int, lam: float, theta: float) -> np.ndarray:
    """Eigenvalues g_m = (1 - 4(1-theta) lam s_m) / (1 + 4 theta lam s_m), s_m = sin^2(m pi / (2 n_x)), of the theta-scheme.

    These are the amplification factors of the scheme for the heat equation on an interval
    with zero boundary values and n_x intervals, lam = a k / h^2, m = 1, ..., n_x - 1.
    """
    m = np.arange(1, n_x)
    s = np.sin(m * np.pi / (2 * n_x)) ** 2
    return (1 - 4 * (1 - theta) * lam * s) / (1 + 4 * theta * lam * s)


def explicit_s_grid(K: float, r: float, sigma: float, T: float, S_max: float, M: int, N: int,
                    discounted_boundary: bool = True) -> dict:
    """The explicit scheme of the course slides for a call, on the grid S_j = j dS, j = 0..M.

    Backwards in time, f_j <- a_j f_{j-1} + b_j f_j + c_j f_{j+1} with
    a_j = dt (sigma^2 j^2 - r j) / 2, b_j = 1 - dt (sigma^2 j^2 + r), c_j = dt (sigma^2 j^2 + r j) / 2,
    f_0 = 0 and f_M = S_max - K e^{-r(T-t)} (or S_max - K if ``discounted_boundary`` is False).
    Every b_j is nonnegative if and only if dt (sigma^2 (M-1)^2 + r) <= 1 for the interior nodes.
    Returns ``S``, ``f`` (prices at t = 0), and ``max_abs`` (sup norm of f at t = 0).
    """
    dt = T / N
    S = np.linspace(0.0, S_max, M + 1)
    j = np.arange(1, M)
    aj = 0.5 * dt * (sigma**2 * j**2 - r * j)
    bj = 1.0 - dt * (sigma**2 * j**2 + r)
    cj = 0.5 * dt * (sigma**2 * j**2 + r * j)
    f = np.maximum(S - K, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(N, 0, -1):
            tau = T - (n - 1) * dt
            new = np.empty_like(f)
            new[1:-1] = aj * f[:-2] + bj * f[1:-1] + cj * f[2:]
            new[0] = 0.0
            new[-1] = S_max - K * (np.exp(-r * tau) if discounted_boundary else 1.0)
            f = new
    return dict(S=S, f=f, max_abs=float(np.max(np.abs(f))), positive=bool(dt * (sigma**2 * (M - 1) ** 2 + r) <= 1))


# ---------------------------------------------------------------- the American put


def american_put_fd(S0: float, K: float, T: float, r: float, sigma: float, n_x: int = 400, n_t: int = 200,
                    theta: float = 0.5, rannacher: int = 2, width: float = 6.0, omega: float = 1.2,
                    keep: bool = False, solver: str = "psor") -> dict:
    """American put by the projected theta-scheme in log-price (``prop:num-american-grid``).

    The grid is centred at K (so that the strike is a node) and must contain S0. Dirichlet values:
    K - x at the bottom (deep in the money the put is exercised) and 0 at the top. Each time step
    solves the discrete complementarity problem by projected SOR (or ``solver="brennan-schwartz"``).
    Returns ``x``, ``v`` (prices at t = 0), ``price`` (at S0, by cubic interpolation in log x), the mean number of PSOR iterations,
    and, with ``keep=True``, ``t`` (calendar times) and ``boundary``: at each time level, the
    largest grid price at which the computed value equals the payoff (NaN if none).
    """
    from scipy.interpolate import CubicSpline
    a, b, c = bs_log_coefficients(r, sigma)
    y = log_grid(K, T, sigma, n_x, width)
    x = np.exp(y)
    if not (x[0] < S0 < x[-1]):
        raise ValueError("S0 is outside the grid; increase width")
    g = np.maximum(K - x, 0.0)
    bc = lambda tau: (K - x[0], 0.0)
    res = theta_scheme(g, y, T, n_t, a, b, c, theta, bc, rannacher=rannacher, obstacle=g, omega=omega,
                       keep=keep, solver=solver)
    v = res["u"]
    out = dict(x=x, v=v, price=float(CubicSpline(y, v)(np.log(S0))),
               mean_iterations=float(res["iterations"].mean()))
    if keep:
        taus, hist = res["taus"], res["history"]
        bnd = []
        for u in hist:
            contact = (u - g <= 1e-9 * K) & (g > 0)
            bnd.append(x[contact].max() if contact.any() else np.nan)
        out.update(t=T - taus, boundary=np.array(bnd), history=hist)
    return out


def brennan_schwartz(ab: np.ndarray, rhs: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    """Direct solution of the put's complementarity problem (Brennan and Schwartz; Jaillet, Lamberton, Lapeyre).

    Eliminates the super-diagonal from the top of the grid down, then substitutes upwards from
    the bottom, projecting on the obstacle at each node. It returns the solution of the
    complementarity problem when the contact set is a set of bottom nodes, as for the put.
    """
    up, dg, lo = ab[0, 1:].copy(), ab[1].copy(), ab[2, :-1].copy()
    q = rhs.astype(float).copy()
    m = len(q)
    for j in range(m - 2, -1, -1):         # eliminate B[j, j+1] using row j + 1
        f = up[j] / dg[j + 1]
        dg[j] -= f * lo[j]
        q[j] -= f * q[j + 1]
    z = np.empty(m)
    z[0] = max(obstacle[0], q[0] / dg[0])
    for j in range(1, m):
        z[j] = max(obstacle[j], (q[j] - lo[j - 1] * z[j - 1]) / dg[j])
    return z


# ---------------------------------------------------------------- a physics-informed network


def pinn_init(n_hidden: int, seed=None) -> dict:
    """Parameters of u(tau, y) = c0 + sum_i c_i tanh(w_i y + s_i tau + b_i), a one-hidden-layer network."""
    rng = np.random.default_rng(seed)
    return dict(w=rng.normal(0.0, 1.5, n_hidden), s=rng.normal(0.0, 1.0, n_hidden),
                b=rng.normal(0.0, 1.0, n_hidden), c=rng.normal(0.0, 0.1, n_hidden), c0=np.zeros(1))


def pinn_eval(p: dict, tau, y) -> dict:
    """Network value and the derivatives u_tau, u_y, u_yy at the points (tau, y)."""
    z = np.outer(y, p["w"]) + np.outer(tau, p["s"]) + p["b"]
    th = np.tanh(z)
    d1 = 1.0 - th**2
    d2 = -2.0 * th * d1
    u = th @ p["c"] + p["c0"][0]
    return dict(u=u, u_tau=d1 @ (p["c"] * p["s"]), u_y=d1 @ (p["c"] * p["w"]), u_yy=d2 @ (p["c"] * p["w"] ** 2),
                z=z, th=th, d1=d1, d2=d2)


def pinn_loss_and_grad(p: dict, pts: dict, a: float, b: float, c: float) -> tuple:
    """Mean squared residual of u_tau - a u_yy - b u_y + c u at interior points, plus initial and boundary losses.

    ``pts`` has arrays ``tau_r, y_r`` (collocation points), ``y_0, g_0`` (initial values u(0, y) = g),
    ``tau_b, y_b, g_b`` (boundary values). The gradient with respect to every parameter is
    computed by hand from the derivatives of tanh. Returns ``(loss, grads, parts)``.
    """
    w, s, cc = p["w"], p["s"], p["c"]
    # residual part
    e = pinn_eval(p, pts["tau_r"], pts["y_r"])
    th, d1, d2 = e["th"], e["d1"], e["d2"]
    d3 = -2.0 * (d1 * d1 + th * d2)          # third derivative of tanh
    R = e["u_tau"] - a * e["u_yy"] - b * e["u_y"] + c * e["u"]
    nR = len(R)
    # dR/dz_i coefficient multiplying each parameter's dz/dparam
    # R = sum_i c_i [ s_i d1 - a w_i^2 d2 - b w_i d1 + c th ]  (+ c * c0)
    G = (s * d1 - a * w**2 * d2 - b * w * d1 + c * th)               # dR/dc_i
    dRdz = cc * (s * d2 - a * w**2 * d3 - b * w * d2 + c * d1)       # derivative through z_i
    gR = 2.0 * R / nR
    grads = dict(c=gR @ G, c0=np.array([np.sum(gR) * c]),
                 w=gR @ (dRdz * pts["y_r"][:, None] + cc * (-2 * a * w * d2 - b * d1)),
                 s=gR @ (dRdz * pts["tau_r"][:, None] + cc * d1),
                 b=gR @ dRdz)
    loss_r = float(np.mean(R**2))
    parts = dict(residual=loss_r)
    loss = loss_r
    for key, (tt, yy, target) in dict(initial=(np.zeros_like(pts["y_0"]), pts["y_0"], pts["g_0"]),
                                       boundary=(pts["tau_b"], pts["y_b"], pts["g_b"])).items():
        ev = pinn_eval(p, tt, yy)
        D = ev["u"] - target
        gD = 2.0 * D / len(D)
        grads["c"] += gD @ ev["th"]
        grads["c0"] += np.sum(gD)
        dz = ev["d1"] * cc
        grads["w"] += gD @ (dz * yy[:, None])
        grads["s"] += gD @ (dz * tt[:, None])
        grads["b"] += gD @ dz
        parts[key] = float(np.mean(D**2))
        loss += parts[key]
    return loss, grads, parts


def pinn_train(pts: dict, a: float, b: float, c: float, n_hidden: int = 24, n_iter: int = 3000,
               lr: float = 0.01, seed=None, record_every: int = 50) -> tuple:
    """Train the network by Adam on the loss of :func:`pinn_loss_and_grad`. Returns ``(params, history)``."""
    p = pinn_init(n_hidden, seed)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for it in range(1, n_iter + 1):
        loss, g, parts = pinn_loss_and_grad(p, pts, a, b, c)
        for k in p:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v2[k] = b2 * v2[k] + (1 - b2) * g[k] ** 2
            p[k] = p[k] - lr * (m[k] / (1 - b1**it)) / (np.sqrt(v2[k] / (1 - b2**it)) + eps)
        if it % record_every == 0 or it == 1:
            hist.append((it, loss, parts["residual"], parts["initial"], parts["boundary"]))
    return p, np.array(hist)
