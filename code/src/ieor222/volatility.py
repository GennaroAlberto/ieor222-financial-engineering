"""Implied volatility, the smile, local and stochastic volatility, variance swaps (Chapter 15).

Conventions follow the notes. Prices are at time 0 under a pricing measure, with constant
interest rate ``r`` and dividend yield ``q``; ``F = S0 e^{(r-q)T}`` is the forward price and
``k = log(K / F)`` the log-moneyness. Most functions work with the *normalized* call price
``c = C e^{rT} / F = E[(S_T / F - e^k)^+]`` and the *total volatility* ``s = sigma sqrt(T)``
(so that ``w = s^2`` is the total implied variance). The functions accept numpy arrays and
broadcast where this is natural; simulations take a ``seed``.

The Heston characteristic function is the one of ``prop:iv-heston``, in the form of Albrecher,
Mayer, Schoutens and Tistaert (2007); Fourier prices use the formula of ``lem:iv-lewis``,
evaluated by the trapezoidal rule, which converges exponentially fast for this integrand.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

SQRT_2PI = np.sqrt(2.0 * np.pi)


def _npdf(y):
    return np.exp(-0.5 * np.asarray(y, dtype=float) ** 2) / SQRT_2PI


# ---------------------------------------------------------------- Black--Scholes in forward form


def black_normalized(k, s):
    """b(k, s) = N(-k/s + s/2) - e^k N(-k/s - s/2), the call price in units of the prepaid forward.

    ``s >= 0`` is the total volatility; at ``s = 0`` the value is the intrinsic value (1 - e^k)^+.
    For k < 0 the out-of-the-money put is computed and parity added, for accuracy.
    """
    k, s = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(s, dtype=float))
    pos = s > 0
    ss = np.where(pos, s, 1.0)
    ka = np.abs(k)
    dp, dm = -ka / ss + 0.5 * ss, -ka / ss - 0.5 * ss
    otm = ndtr(dp) - np.exp(ka) * ndtr(dm)            # b(|k|, s): an out-of-the-money call
    otm = np.maximum(otm, 0.0)
    # b(k, s) = (1 - e^k) + e^k b(-k, s)  (put--call symmetry), used for k < 0
    val = np.where(k >= 0, otm, (1.0 - np.exp(k)) + np.exp(k) * otm)
    return np.where(pos, val, np.maximum(1.0 - np.exp(k), 0.0))[()]


def black_vega_normalized(k, s):
    """Derivative of :func:`black_normalized` in s: N'(d_+) with d_+ = -k/s + s/2 (s > 0)."""
    k, s = np.asarray(k, dtype=float), np.asarray(s, dtype=float)
    return _npdf(-k / s + 0.5 * s)


def bs_call(S0, K, T, r, sigma, q=0.0):
    """Black--Scholes call with dividend yield q (``prop:rn-dividends``): S0 e^{-qT} b(k, sigma sqrt T)."""
    F = S0 * np.exp((r - q) * T)
    return S0 * np.exp(-q * T) * black_normalized(np.log(K / F), sigma * np.sqrt(T))


def bs_put(S0, K, T, r, sigma, q=0.0):
    """Black--Scholes put with dividend yield q, by parity C - P = S0 e^{-qT} - K e^{-rT}."""
    return bs_call(S0, K, T, r, sigma, q) - S0 * np.exp(-q * T) + K * np.exp(-r * T)


def tehranchi_upper(c, k):
    """The upper bound s <= -2 N^{-1}((1 - c) / (1 + e^k)) of ``prop:iv-existence`` (equality at k = 0)."""
    c, k = np.asarray(c, dtype=float), np.asarray(k, dtype=float)
    return -2.0 * ndtri((1.0 - c) / (1.0 + np.exp(k)))


def _otm_total_vol(c, k, tol=1e-13, max_iter=300):
    """Solve b(k, s) = c for s when k >= 0 and 0 < c < 1 (an out-of-the-money call)."""
    logc = np.log(c)
    lo = np.zeros_like(c)
    hi = np.minimum(tehranchi_upper(c, k), 50.0) * (1 + 1e-12) + 1e-300
    s = np.clip(np.sqrt(2.0 * k), 0.0, hi)        # the inflection point of s -> b(k, s)
    s = np.where(s <= 0, 0.5 * hi, s)
    step_old = hi.copy()
    for _ in range(max_iter):
        b = black_normalized(k, s)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.log(b) - logc
            lo = np.where(f < 0, s, lo)
            hi = np.where(f >= 0, s, hi)
            newton = s - f * b / black_vega_normalized(k, s)
        good = np.isfinite(newton) & (newton > lo) & (newton < hi) & (np.abs(newton - s) <= 0.5 * step_old)
        s_new = np.where(good, newton, 0.5 * (lo + hi))
        step_old = np.abs(s_new - s)
        s = s_new
        if np.all((step_old <= tol * s) | (hi - lo <= tol * s)):
            break
    return s


def implied_total_vol(price, k, kind="call"):
    """Total implied volatility s = sigma sqrt(T) of a normalized call or put price at log-moneyness k.

    ``price`` is the option price divided by the prepaid forward S0 e^{-qT}. Returns nan outside
    the arbitrage bounds of ``prop:iv-existence`` ((1 - e^k)^+ < c < 1 for a call). The problem is
    reduced to an out-of-the-money call: a put with k <= 0 is, by the put--call symmetry
    p(k, s) = e^k b(-k, s), a call with log-moneyness -k, and an in-the-money option is first
    converted by parity (which loses the digits of a tiny out-of-the-money value: quote
    out-of-the-money options when the price is small). The solver is Newton's method for
    log b(k, s) = log c on the bracket [0, s_up] of the proposition; a Newton step is taken only if
    it stays in the current bracket and at least halves the previous step, and a bisection step
    otherwise, so the iteration converges for every admissible price.
    """
    p, k = np.broadcast_arrays(np.asarray(price, dtype=float), np.asarray(k, dtype=float))
    ek = np.exp(k)
    if kind == "call":
        ok = (p > np.maximum(1.0 - ek, 0.0)) & (p < 1.0)
        otm = np.where(k >= 0, p, (p - 1.0 + ek) / ek)          # OTM-call equivalent at |k|
    elif kind == "put":
        ok = (p > np.maximum(ek - 1.0, 0.0)) & (p < ek)
        otm = np.where(k <= 0, p / ek, p + 1.0 - ek)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    ok &= otm > 0
    ka = np.where(ok, np.abs(k), 0.0)
    otm = np.where(ok, otm, 0.5)
    return np.where(ok, _otm_total_vol(otm, ka), np.nan)[()]


def implied_vol(price, S0, K, T, r=0.0, q=0.0, kind="call"):
    """Black--Scholes implied volatility (``def:iv-implied``) of a call or put price.

    Calls and puts with the same strike and maturity whose prices satisfy parity have the same
    implied volatility (``prop:iv-existence``).
    """
    price, K, T = (np.asarray(a, dtype=float) for a in (price, K, T))
    pf = S0 * np.exp(-q * T)                      # prepaid forward
    F = S0 * np.exp((r - q) * T)
    return (implied_total_vol(price / pf, np.log(K / F), kind) / np.sqrt(T))[()]


def bachelier_call(F, K, T, sigma_n, r=0.0):
    """Bachelier (normal) call price e^{-rT} [(F - K) N(d) + sigma_n sqrt(T) N'(d)], d = (F-K)/(sigma_n sqrt T)."""
    F, K = np.asarray(F, dtype=float), np.asarray(K, dtype=float)
    sd = sigma_n * np.sqrt(T)
    d = (F - K) / sd
    return (np.exp(-r * T) * ((F - K) * ndtr(d) + sd * _npdf(d)))[()]


# ---------------------------------------------------------------- stylised facts


def kurtosis(x):
    """Sample kurtosis E[(x - m)^4] / Var(x)^2 (3 for a normal sample)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    return float(np.mean(x**4) / np.mean(x**2) ** 2)


def autocorrelation(x, max_lag: int):
    """Sample autocorrelations of x at lags 1, ..., max_lag."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    v = np.dot(x, x)
    return np.array([np.dot(x[:-j], x[j:]) / v for j in range(1, max_lag + 1)])


def cross_correlation(x, y, max_lag: int):
    """Sample correlations corr(x_t, y_{t+j}) for j = 1, ..., max_lag."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)
    sx, sy = np.sqrt(np.mean(x**2)), np.sqrt(np.mean(y**2))
    return np.array([np.mean(x[:-j] * y[j:]) / (sx * sy) for j in range(1, max_lag + 1)])


def mixture_kurtosis(v_values, v_probs):
    """Kurtosis 3 E[V^2] / E[V]^2 of sqrt(V) Z, Z ~ N(0, 1) independent of V (``prop:iv-mixture``)."""
    v, p = np.asarray(v_values, dtype=float), np.asarray(v_probs, dtype=float)
    return float(3.0 * np.sum(p * v**2) / np.sum(p * v) ** 2)


# ---------------------------------------------------------------- the Heston model


def heston_cf(u, T, v0, kappa, theta, xi, rho):
    """E[exp(i u X_T)] for X_T = log(S_T / F_T) in the Heston model (``prop:iv-heston``).

    The form of Albrecher et al. (2007): with b = kappa - rho xi i u,
    d = sqrt(b^2 + xi^2 (i u + u^2)) (principal root), g = (b - d) / (b + d),
    D = (b - d) (1 - e^{-dT}) / (xi^2 (1 - g e^{-dT})) and
    C = kappa theta / xi^2 [(b - d) T - 2 log((1 - g e^{-dT}) / (1 - g))],
    the characteristic function is exp(C + D v0). ``u`` may be complex and ``T`` an array
    (broadcast against u).
    """
    u = np.asarray(u, dtype=complex)
    T = np.asarray(T, dtype=float)
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi**2 * (iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    D = (b - d) / xi**2 * (1.0 - e) / (1.0 - g * e)
    C = kappa * theta / xi**2 * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    return np.exp(C + D * v0), C, D


def heston_cf_dT(u, T, v0, kappa, theta, xi, rho):
    """Derivative in T of :func:`heston_cf`, from the Riccati equations C' = kappa theta D and
    D' = xi^2 D^2 / 2 - (kappa - rho xi i u) D - (u^2 + i u) / 2."""
    phi, C, D = heston_cf(u, T, v0, kappa, theta, xi, rho)
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    dD = 0.5 * xi**2 * D * D - (kappa - rho * xi * iu) * D - 0.5 * (u * u + iu)
    return phi * (kappa * theta * D + v0 * dD)


def _lewis_nodes(h=0.1, u_max=400.0):
    n = int(np.ceil(u_max / h))
    u = h * np.arange(n + 1)
    w = np.full(n + 1, h)
    w[0] = 0.5 * h                              # trapezoid on [0, inf) for an even integrand
    return u, w


def lewis_transforms(cf_shifted, k, u, w, order=0):
    """The integrals of ``lem:iv-lewis`` for each log-moneyness k.

    ``cf_shifted`` holds phi(u - i/2) on the nodes ``u`` (any leading shape, nodes last).
    order 0: I0(k) = (1/pi) int_0^inf Re[e^{-iuk} phi(u - i/2)] / (u^2 + 1/4) du;
    order 1: I1(k) = (1/pi) int_0^inf Re[e^{-iuk} phi(u - i/2) / (1/2 + i u)] du;
    order 2: I2(k) = (1/pi) int_0^inf Re[e^{-iuk} phi(u - i/2)] du.
    Then E[(e^X - e^k)^+] = 1 - e^{k/2} I0, its k-derivative is -e^{k/2} I1, and the density of X at k
    is e^{-k/2} I2.
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    if order == 0:
        g = cf_shifted / (u * u + 0.25)
    elif order == 1:
        g = cf_shifted / (0.5 + 1j * u)
    else:
        g = cf_shifted
    ph = np.exp(-1j * np.outer(u, k))           # nodes x strikes
    return np.real((g * w) @ ph) / np.pi


def call_normalized_from_cf(cf, k, h=0.1, u_max=400.0):
    """E[(e^X - e^k)^+] = 1 - e^{k/2} I0(k) for a law with E[e^X] = 1 and characteristic function cf."""
    u, w = _lewis_nodes(h, u_max)
    I0 = lewis_transforms(cf(u - 0.5j), k, u, w, order=0)
    k = np.atleast_1d(np.asarray(k, dtype=float))
    return 1.0 - np.exp(0.5 * k) * I0


def heston_call(S0, K, T, r, v0, kappa, theta, xi, rho, q=0.0, h=0.1, u_max=None):
    """Heston call price S0 e^{-qT} [1 - e^{k/2} I0(k)], k = log(K / F), by ``lem:iv-lewis``.

    ``K`` may be an array (one maturity T). ``u_max`` defaults to a value at which the integrand
    is negligible: the characteristic function decays at least like exp(-v_eff T u^2 / 2) for
    moderate u, with v_eff the smaller of v0 and theta.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    F = S0 * np.exp((r - q) * T)
    if u_max is None:
        veff = max(min(v0, theta), 1e-4)
        u_max = min(max(60.0, np.sqrt(2.0 * 60.0 / (veff * T))), 3000.0)
    u, w = _lewis_nodes(h, u_max)
    phi = heston_cf(u - 0.5j, T, v0, kappa, theta, xi, rho)[0]
    k = np.log(K / F)
    c = 1.0 - np.exp(0.5 * k) * lewis_transforms(phi, k, u, w, 0)
    return S0 * np.exp(-q * T) * c


def heston_surface(k, T, v0, kappa, theta, xi, rho, h=0.1, u_max=None):
    """Normalized call prices c, dc/dk, d2c/dk2, dc/dT and the density of X_T on a (T, k) grid.

    c(T, k) = E[(S_T / F_T - e^k)^+]. Returns a dict of arrays of shape (len(T), len(k)), all from
    Fourier integrals (no finite differences). Used for Dupire's formula in forward variables,
    sigma_loc^2(T, F_T e^k) = 2 c_T / (c_kk - c_k) (``thm:iv-dupire``).
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    T = np.atleast_1d(np.asarray(T, dtype=float))
    if u_max is None:
        veff = max(min(v0, theta), 1e-4)
        u_max = min(max(60.0, np.sqrt(2.0 * 60.0 / (veff * T.min()))), 3000.0)
    u, w = _lewis_nodes(h, u_max)
    z = u - 0.5j
    phi = heston_cf(z[None, :], T[:, None], v0, kappa, theta, xi, rho)[0]
    dphi = heston_cf_dT(z[None, :], T[:, None], v0, kappa, theta, xi, rho)
    ek = np.exp(0.5 * k)[None, :]
    I0 = lewis_transforms(phi, k, u, w, 0)
    I1 = lewis_transforms(phi, k, u, w, 1)
    I2 = lewis_transforms(phi, k, u, w, 2)
    c = 1.0 - ek * I0
    c_k = -ek * I1
    dens_k = I2 / ek                     # density of X_T = log(S_T / F_T) at k
    c_kk = c_k + ek * ek * dens_k        # since c_kk - c_k = e^k q(k)
    c_T = -ek * lewis_transforms(dphi, k, u, w, 0)
    return dict(c=c, c_k=c_k, c_kk=c_kk, c_T=c_T, density=dens_k)


def heston_expected_variance(T, v0, kappa, theta):
    """(1/T) E int_0^T v_t dt = theta + (v0 - theta)(1 - e^{-kappa T}) / (kappa T): the variance-swap strike."""
    T = np.asarray(T, dtype=float)
    return (theta + (v0 - theta) * (1.0 - np.exp(-kappa * T)) / (kappa * T))[()]


def heston_paths(S0, T, mu, v0, kappa, theta, xi, rho, n_steps, n_paths, seed=None, q=0.0,
                 keep=False, running_max=False):
    """Full-truncation Euler scheme for the Heston model (Lord, Koekkoek and van Dijk, 2010).

    v_{j+1} = v_j + kappa (theta - v_j^+) h + xi sqrt(v_j^+ h) Z2,
    log S_{j+1} = log S_j + (mu - q - v_j^+ / 2) h + sqrt(v_j^+ h) (rho Z2 + sqrt(1 - rho^2) Z1).
    Under the pricing measure take mu = r. Returns a dict with the terminal ``S``, and, on request,
    the whole paths (``keep``) of S and v^+, or the running maximum of S on the grid (``running_max``).
    """
    rng = np.random.default_rng(seed)
    hstep = T / n_steps
    x = np.full(n_paths, np.log(S0))
    v = np.full(n_paths, float(v0))
    mx = np.full(n_paths, float(S0)) if running_max else None
    path_S, path_v = ([np.exp(x)], [v.copy()]) if keep else (None, None)
    a = np.sqrt(1.0 - rho * rho)
    for _ in range(n_steps):
        z2 = rng.standard_normal(n_paths)
        z1 = rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        sq = np.sqrt(vp * hstep)
        x = x + (mu - q - 0.5 * vp) * hstep + sq * (rho * z2 + a * z1)
        v = v + kappa * (theta - vp) * hstep + xi * sq * z2
        if running_max:
            np.maximum(mx, np.exp(x), out=mx)
        if keep:
            path_S.append(np.exp(x))
            path_v.append(np.maximum(v, 0.0))
    out = dict(S=np.exp(x))
    if keep:
        out.update(S_path=np.array(path_S).T, v_path=np.array(path_v).T)
    if running_max:
        out["max"] = mx
    return out


def mc_call_prices(S_T, K, F, disc):
    """Monte Carlo call prices disc E[(S_T - K)^+] with S_T as control variate (E[S_T] = F).

    Returns (prices, standard errors), one per strike.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    pay = np.maximum(S_T[None, :] - K[:, None], 0.0)
    cs = S_T - S_T.mean()
    beta = (pay - pay.mean(axis=1, keepdims=True)) @ cs / np.dot(cs, cs)
    adj = pay - beta[:, None] * (S_T[None, :] - F)
    n = S_T.size
    return disc * adj.mean(axis=1), disc * adj.std(axis=1, ddof=1) / np.sqrt(n)


# ---------------------------------------------------------------- static arbitrage in total variance


def butterfly_g(k, w, wk, wkk):
    """g(k) = (1 - k w'/(2w))^2 - (w')^2/4 (1/w + 1/4) + w''/2 (``prop:iv-static-arbitrage``)."""
    return (1.0 - k * wk / (2.0 * w)) ** 2 - wk**2 / 4.0 * (1.0 / w + 0.25) + wkk / 2.0


def density_from_w(k, w, wk, wkk):
    """Density of X = log(S_T / F) at k implied by a smooth slice: g(k) N'(d_-) / sqrt(w), d_- = -k/sqrt(w) - sqrt(w)/2.

    (The density of S_T at K = F e^k is this divided by K.)
    """
    sw = np.sqrt(w)
    return butterfly_g(k, w, wk, wkk) * _npdf(-k / sw - 0.5 * sw) / sw


def svi_raw(k, a, b, rho, m, sig):
    """Raw SVI total variance w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sig^2)), with w' and w''."""
    k = np.asarray(k, dtype=float)
    y = k - m
    r = np.sqrt(y * y + sig * sig)
    w = a + b * (rho * y + r)
    wk = b * (rho + y / r)
    wkk = b * sig * sig / r**3
    return w, wk, wkk


def ssvi(k, theta, rho, eta, gamma):
    """SSVI slice w = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)), phi = eta theta^{-gamma}.

    Returns w, dw/dk, d2w/dk2 and dw/dtheta (the power-law phi of Gatheral and Jacquier, 2014).
    """
    k = np.asarray(k, dtype=float)
    phi = eta * theta ** (-gamma)
    dphi = -gamma * phi / theta
    y = phi * k + rho
    r = np.sqrt(y * y + 1.0 - rho * rho)
    w = 0.5 * theta * (1.0 + rho * phi * k + r)
    wk = 0.5 * theta * phi * (rho + y / r)
    wkk = 0.5 * theta * phi**2 * (1.0 - rho * rho) / r**3
    # d/dtheta at fixed k
    dw = 0.5 * (1.0 + rho * phi * k + r) + 0.5 * theta * (rho * dphi * k + y * dphi * k / r)
    return w, wk, wkk, dw


def local_variance_from_w(k, w, wk, wkk, wT):
    """Dupire's local variance in total implied variance: w_T / g(k) (``eq:iv-dupire-w``)."""
    return wT / butterfly_g(k, w, wk, wkk)


# ---------------------------------------------------------------- local volatility


def ssvi_local_vol_grid(t_grid, k_grid, atm_var_rate, rho, eta, gamma, vol_bounds=(0.01, 3.0)):
    """Dupire local volatility of the SSVI surface with theta_t = atm_var_rate * t (``eq:iv-dupire-w``).

    sigma_loc^2(t, F_t e^k) = d_T w / g(k), with d_T w = atm_var_rate * dw/dtheta. Returns an array
    (len(t_grid), len(k_grid)); t must be positive. Values are clipped to ``vol_bounds`` (they are
    finite and positive when the surface is free of static arbitrage).
    """
    out = np.empty((len(t_grid), len(k_grid)))
    for i, t in enumerate(t_grid):
        w, wk, wkk, dw = ssvi(k_grid, atm_var_rate * t, rho, eta, gamma)
        out[i] = local_variance_from_w(k_grid, w, wk, wkk, atm_var_rate * dw)
    return np.clip(np.sqrt(np.maximum(out, 0.0)), *vol_bounds)




def dupire_heston_grid(t_grid, k_grid, v0, kappa, theta, xi, rho, floor=1e-6, vol_bounds=(0.02, 2.0)):
    """Dupire local volatility of the Heston surface on a (t, k) grid, k = log(S / F_t).

    sigma_loc^2 = 2 c_T / (c_kk - c_k) with the Fourier derivatives of :func:`heston_surface`.
    Where the density of log(S_t / F_t) is below ``floor`` times its maximum, the value at the
    nearest reliable k is used (flat extrapolation). Returns an array (len(t), len(k)).
    """
    srf = heston_surface(k_grid, t_grid, v0, kappa, theta, xi, rho)
    den = srf["c_kk"] - srf["c_k"]
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = 2.0 * srf["c_T"] / den
    rel = srf["density"] / srf["density"].max(axis=1, keepdims=True)
    good = (rel > floor) & np.isfinite(lv) & (lv > 0)
    out = np.empty_like(lv)
    for i in range(len(t_grid)):
        idx = np.flatnonzero(good[i])
        out[i] = np.interp(k_grid, k_grid[idx], lv[i, idx])
    return np.clip(np.sqrt(out), *vol_bounds)


def local_vol_paths(S0, T, r, q, t_grid, k_grid, lv_grid, n_paths, seed=None, running_max=False):
    """Euler scheme for log(S_t / F_t) in the local volatility model, with sigma = lv_grid[j](k).

    The time grid is ``t_grid`` (starting at 0); at step j the volatility is interpolated in k
    from row j of ``lv_grid``. Returns a dict with the terminal ``S`` and, on request, the
    running maximum of S on the grid.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n_paths)                         # log(S_t / F_t)
    mx = np.full(n_paths, float(S0)) if running_max else None
    for j in range(len(t_grid) - 1):
        hstep = t_grid[j + 1] - t_grid[j]
        sig = np.interp(x, k_grid, lv_grid[j])
        x = x - 0.5 * sig * sig * hstep + sig * np.sqrt(hstep) * rng.standard_normal(n_paths)
        if running_max:
            np.maximum(mx, S0 * np.exp((r - q) * t_grid[j + 1] + x), out=mx)
    out = dict(S=S0 * np.exp((r - q) * T + x))
    if running_max:
        out["max"] = mx
    return out


def up_and_out_call(S_T, running_max, K, B, disc):
    """Monte Carlo price and standard error of the discretely monitored up-and-out call."""
    pay = np.where(running_max < B, np.maximum(S_T - K, 0.0), 0.0)
    return disc * pay.mean(), disc * pay.std(ddof=1) / np.sqrt(pay.size)


def digital_from_smile(S0, K, T, r, sigma, dsigma_dK, q=0.0):
    """Digital call price -dC/dK = e^{-rT} N(d_-) - vega dsigma/dK under a smile (``prop:iv-digital``)."""
    F = S0 * np.exp((r - q) * T)
    s = sigma * np.sqrt(T)
    k = np.log(K / F)
    dm = -k / s - 0.5 * s
    vega = S0 * np.exp(-q * T) * _npdf(dm + s) * np.sqrt(T)
    return np.exp(-r * T) * ndtr(dm) - vega * dsigma_dK


# ---------------------------------------------------------------- variance swaps and the VIX


def strip_variance(K, otm, F, T, r):
    """(2 e^{rT} / T) int OTM(K) / K^2 dK by the trapezoidal rule on the strike grid K.

    ``otm`` holds out-of-the-money prices (puts below F, calls above). By ``prop:iv-variance-swap``
    this is the variance-swap strike, up to the discretization and truncation of the strike range.
    """
    K, otm = np.asarray(K, dtype=float), np.asarray(otm, dtype=float)
    f = otm / K**2
    return float(2.0 * np.exp(r * T) / T * np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(K)))


def vix_sigma2(K, calls, puts, F, T, r):
    """The CBOE estimate (2/T) sum dK_i / K_i^2 e^{rT} Q(K_i) - (1/T)(F / K_0 - 1)^2 (``def:iv-vix``).

    K_0 is the largest strike at or below F; Q is the put below K_0, the call above, and the
    average of the two at K_0; dK_i is half the distance between the neighbouring strikes (the
    distance to the single neighbour at the ends of the grid).
    """
    K, calls, puts = (np.asarray(a, dtype=float) for a in (K, calls, puts))
    i0 = int(np.searchsorted(K, F, side="right") - 1)
    if i0 < 0:
        raise ValueError("no strike at or below the forward")
    Q = np.where(np.arange(K.size) < i0, puts, calls)
    Q[i0] = 0.5 * (calls[i0] + puts[i0])
    dK = np.empty_like(K)
    dK[1:-1] = 0.5 * (K[2:] - K[:-2])
    dK[0], dK[-1] = K[1] - K[0], K[-1] - K[-2]
    K0 = K[i0]
    return float(2.0 / T * np.sum(dK / K**2 * np.exp(r * T) * Q) - (F / K0 - 1.0) ** 2 / T)
