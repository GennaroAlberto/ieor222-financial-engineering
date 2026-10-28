"""Rough volatility: fractional Brownian motion, the Gatheral--Jaisson--Rosenbaum estimator,
rough Bergomi Monte Carlo and the rough Heston characteristic function (Chapter 19).

Conventions follow the notes. ``H`` is the Hurst index and ``alpha = H - 1/2`` the exponent of
the Riemann--Liouville kernel. Samplers take a ``seed`` and return arrays of shape
``(n_paths, n_points)``, one row per path, with the time grid separately. Option prices are
*normalized* as in Chapter 15: with zero rates, ``c(k) = E[(S_T / S_0 - e^k)^+]`` at log-moneyness
``k = log(K / S_0)``, and the Black--Scholes value ``b(k, s)`` of :mod:`ieor222.volatility` has
total volatility ``s = sigma sqrt(T)``.

Simulation methods:

* exact Cholesky sampling of any Gaussian vector (fBM, or the Riemann--Liouville process jointly
  with its Brownian driver), cost O(n^3) once and O(n^2) per path: small grids;
* Davies--Harte circulant embedding for fBM on a uniform grid, exact, O(n log n) per path;
* the hybrid scheme of Bennedsen, Lunde and Pakkanen (2017) for the Riemann--Liouville process,
  O(n log n) per path by FFT convolution.
"""
from __future__ import annotations

import gzip
import struct

import numpy as np
from scipy.signal import fftconvolve, lfilter
from scipy.special import gamma as gamma_fn
from scipy.special import hyp2f1, ndtr

from ieor222.volatility import black_normalized, implied_total_vol, lewis_transforms

SQRT_2PI = np.sqrt(2.0 * np.pi)


# ---------------------------------------------------------------- Gaussian facts


def abs_moment(q):
    """K_q = E|N|^q = 2^{q/2} Gamma((q + 1) / 2) / sqrt(pi) for N ~ N(0, 1) and q > -1."""
    q = np.asarray(q, dtype=float)
    return 2.0 ** (q / 2) * gamma_fn((q + 1) / 2) / np.sqrt(np.pi)


def fbm_cov(s, t, H):
    """R_H(s, t) = (s^{2H} + t^{2H} - |t - s|^{2H}) / 2, the covariance of fBM (``def:rv-fbm``)."""
    s, t = np.asarray(s, dtype=float), np.asarray(t, dtype=float)
    return 0.5 * (s ** (2 * H) + t ** (2 * H) - np.abs(t - s) ** (2 * H))


def fgn_autocov(n, H):
    """rho_H(n) = (|n+1|^{2H} - 2|n|^{2H} + |n-1|^{2H}) / 2: covariance of unit-step fBM increments."""
    n = np.abs(np.asarray(n, dtype=float))
    return 0.5 * ((n + 1) ** (2 * H) - 2 * n ** (2 * H) + np.abs(n - 1) ** (2 * H))


def mvn_constant(H):
    """c_H^2 = int_R k_1(s)^2 ds for the Mandelbrot--Van Ness kernel k_1(s) = (1-s)_+^a - (-s)_+^a, a = H - 1/2.

    Closed form c_H^2 = Gamma(H + 1/2) Gamma(2 - 2H) / (2H Gamma(3/2 - H)), the inverse square of the
    constant C_H of Bayer, Friz and Gatheral (2016); checked against quadrature in the tests.
    """
    return gamma_fn(H + 0.5) * gamma_fn(2 - 2 * H) / (2 * H * gamma_fn(1.5 - H))


def rl_cov(s, t, H):
    """E[Y_s Y_t] for the Riemann--Liouville process Y_t = sqrt(2H) int_0^t (t-u)^{H-1/2} dW_u.

    For 0 < s <= t it equals s^{2H} G(t/s), G(x) = 2H int_0^1 (1-u)^a (x-u)^a du, a = H - 1/2,
    and Euler's integral for the hypergeometric function gives
    G(x) = (2H / (H + 1/2)) x^{H-1/2} 2F1(1, 1/2 - H; H + 3/2; 1/x)  (``prop:rv-rl``).
    """
    s, t = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(t, dtype=float))
    lo, hi = np.minimum(s, t), np.maximum(s, t)
    out = np.zeros(lo.shape)
    pos = lo > 0
    x = hi[pos] / lo[pos]
    a = H - 0.5
    G = 2 * H / (a + 1) * x ** a * hyp2f1(1.0, -a, a + 2.0, 1.0 / x)
    out[pos] = lo[pos] ** (2 * H) * G
    return out[()] if out.ndim == 0 else out


def rl_brownian_cov(t, s, H):
    """E[Y_t W_s] = sqrt(2H) / (H + 1/2) (t^{H+1/2} - (t - min(s, t))^{H+1/2})."""
    t, s = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(s, dtype=float))
    m = np.minimum(s, t)
    return np.sqrt(2 * H) / (H + 0.5) * (t ** (H + 0.5) - (t - m) ** (H + 0.5))


# ---------------------------------------------------------------- exact simulation


def gaussian_cholesky(cov, n_paths, seed=None):
    """Samples of the centered Gaussian vector with covariance ``cov`` (shape (n, n)): rows L G, L L^T = cov."""
    L = np.linalg.cholesky(np.asarray(cov, dtype=float))
    G = np.random.default_rng(seed).standard_normal((n_paths, L.shape[0]))
    return G @ L.T


def fbm_cholesky(H, T=1.0, n_steps=100, n_paths=1, seed=None, normals=None):
    """Exact fBM on the grid t_j = j T / n_steps by Cholesky factorization of R_H.

    Returns ``(t, X)`` with X[:, 0] = 0. ``normals`` (shape (n_paths, n_steps)) may be passed to
    use the same standard normal draws for several values of H.
    """
    t = np.linspace(0.0, T, n_steps + 1)
    tt = t[1:]
    L = np.linalg.cholesky(fbm_cov(tt[:, None], tt[None, :], H))
    G = np.random.default_rng(seed).standard_normal((n_paths, n_steps)) if normals is None else normals
    return t, np.concatenate([np.zeros((G.shape[0], 1)), G @ L.T], axis=1)


def fbm_davies_harte(H, T=1.0, n_steps=1024, n_paths=1, seed=None):
    """Exact fBM on a uniform grid by circulant embedding of fractional Gaussian noise (Davies--Harte).

    The covariance of the n increments is embedded in a circulant matrix of size 2n, diagonalized by
    the FFT; the method is exact when its eigenvalues are nonnegative, which is checked (it holds
    for H <= 1/2, where the noise has negative autocovariances: Craigmile, 2003).
    Returns ``(t, X)``.
    """
    n = int(n_steps)
    k = np.arange(n + 1)
    r = fgn_autocov(k, H)
    row = np.concatenate([r, r[-2:0:-1]])           # first row of the 2n circulant
    lam = np.fft.fft(row).real
    if lam.min() < -1e-10 * lam.max():
        raise ValueError("circulant embedding is not nonnegative definite for this H and n")
    lam = np.clip(lam, 0.0, None)
    m = row.size
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, m)) + 1j * rng.standard_normal((n_paths, m))
    noise = np.fft.fft(np.sqrt(lam / m) * Z, axis=1)[:, :n].real   # real part has the right law
    dt = T / n
    X = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(noise, axis=1)], axis=1) * dt ** H
    return np.linspace(0.0, T, n + 1), X


def rl_cholesky(H, T=1.0, n_steps=50, n_paths=1, seed=None):
    """Exact joint samples of (Y_{t_j}, W_{t_j}), t_j = j T / n_steps, j = 1..n_steps, by Cholesky.

    The covariance blocks are :func:`rl_cov`, :func:`rl_brownian_cov` and min(s, t), as in Bayer,
    Friz and Gatheral (2016), Section 4. Returns ``(t, Y, dW)`` with Y[:, 0] = 0 and the Brownian
    increments dW of shape (n_paths, n_steps).
    """
    t = np.linspace(0.0, T, n_steps + 1)
    tt = t[1:]
    Cyy = rl_cov(tt[:, None], tt[None, :], H)
    Cyw = rl_brownian_cov(tt[:, None], tt[None, :], H)
    Cww = np.minimum(tt[:, None], tt[None, :])
    cov = np.block([[Cyy, Cyw], [Cyw.T, Cww]])
    X = gaussian_cholesky(cov, n_paths, seed)
    Y = np.concatenate([np.zeros((n_paths, 1)), X[:, :n_steps]], axis=1)
    W = np.concatenate([np.zeros((n_paths, 1)), X[:, n_steps:]], axis=1)
    return t, Y, np.diff(W, axis=1)


# ---------------------------------------------------------------- the hybrid scheme


def optimal_points(k, alpha):
    """b_k^* = ((k^{a+1} - (k-1)^{a+1}) / (a + 1))^{1/a}: the evaluation points of Bennedsen et al., Prop. 2.8.

    Defined for a = H - 1/2 != 0; at H = 1/2 the kernel is constant and every b_k gives the exact value (b_k = k is
    returned)."""
    k = np.asarray(k, dtype=float)
    if alpha == 0:
        return k
    return ((k ** (alpha + 1) - (k - 1) ** (alpha + 1)) / (alpha + 1)) ** (1.0 / alpha)


def _cell_integrals(k, alpha):
    """A1 = int_{k-1}^k y^a dy and A2 = int_{k-1}^k y^{2a} dy."""
    k = np.asarray(k, dtype=float)
    A1 = (k ** (alpha + 1) - (k - 1) ** (alpha + 1)) / (alpha + 1)
    A2 = (k ** (2 * alpha + 1) - (k - 1) ** (2 * alpha + 1)) / (2 * alpha + 1)
    return A1, A2


def hybrid_points(n_cells, alpha, ell=1, points="optimal"):
    """Evaluation points b_k for the cells k = ell+1, ..., n_cells: optimal b_k^*, or b_k = k ("left")."""
    k = np.arange(ell + 1, n_cells + 1, dtype=float)
    return k, (optimal_points(k, alpha) if points == "optimal" else k)


def hybrid_mse(H, n_per_unit, i, ell=1, points="optimal"):
    """Exact mean-square error E[(Y_t - Yhat_t)^2] of the hybrid scheme at t = i / n_per_unit.

    Both are Wiener integrals against the same W, so by the isometry the error is
    2H n^{-2H} sum_{k=ell+1}^{i} int_{k-1}^{k} (y^a - b_k^a)^2 dy  (``prop:rv-hybrid``).
    """
    a = H - 0.5
    k, b = hybrid_points(i, a, ell, points)
    A1, A2 = _cell_integrals(k, a)
    return 2 * H * n_per_unit ** (-2 * H) * np.sum(A2 - 2 * b ** a * A1 + b ** (2 * a))


def hybrid_J(H, ell=1, points="optimal", n_terms=10**6):
    """J(a, ell, b) = sum_{k > ell} int_{k-1}^k (y^a - b_k^a)^2 dy, summed to n_terms plus a tail bound.

    For large k the summand is about a^2 k^{2a-2} / 12 (optimal points) or a^2 k^{2a-2} / 3 (b_k = k);
    the tail is added in that form.
    """
    a = H - 0.5
    k, b = hybrid_points(n_terms, a, ell, points)
    A1, A2 = _cell_integrals(k, a)
    s = np.sum(A2 - 2 * b ** a * A1 + b ** (2 * a))
    c = a * a / (12.0 if points == "optimal" else 3.0)
    tail = c * n_terms ** (2 * a - 1) / (1 - 2 * a)
    return s + tail


def hybrid_rl(H, T=1.0, n_steps=256, n_paths=1, seed=None, ell=1, points="optimal", antithetic=False):
    """The hybrid scheme for Y_t = sqrt(2H) int_0^t (t-s)^a dW_s on t_i = i T / n_steps.

    ell = 1: Yhat_i = sqrt(2H) [ int over the last cell of (t_i - s)^a dW_s
                                  + sum_{k=2}^{i} (b_k dt)^a (W_{t_{i-k+1}} - W_{t_{i-k}}) ];
    ell = 0: only the Riemann sum, from k = 1. The pair (Brownian increment, Wiener integral of the
    power kernel over the same cell) is drawn exactly. Returns ``(t, Y, dW)``; with ``antithetic`` the
    paths come in pairs (G, -G), so ``n_paths`` must be even.
    """
    if ell not in (0, 1):
        raise ValueError("ell (the number of cells integrated exactly) must be 0 or 1")
    a = H - 0.5
    n = int(n_steps)
    dt = T / n
    rng = np.random.default_rng(seed)
    m = n_paths // 2 if antithetic else n_paths
    G1 = rng.standard_normal((m, n))
    G2 = rng.standard_normal((m, n))
    if antithetic:
        G1, G2 = np.concatenate([G1, -G1]), np.concatenate([G2, -G2])
    v11, v12, v22 = dt, dt ** (a + 1) / (a + 1), dt ** (2 * a + 1) / (2 * a + 1)
    dW = np.sqrt(v11) * G1
    W1 = (v12 / np.sqrt(v11)) * G1 + np.sqrt(v22 - v12 ** 2 / v11) * G2   # int_cell (t_{j+1} - s)^a dW_s
    k, b = hybrid_points(n, a, ell, points)
    kernel = np.zeros(n + 1)
    kernel[ell + 1:] = (b * dt) ** a
    conv = fftconvolve(dW, kernel[None, :], axes=1)[:, 1:n + 1]          # sum_k kernel_k dW_{i-k}
    Y = conv + (W1 if ell == 1 else 0.0)
    Y = np.sqrt(2 * H) * np.concatenate([np.zeros((dW.shape[0], 1)), Y], axis=1)
    return np.linspace(0.0, T, n + 1), Y, dW


# ---------------------------------------------------------------- RFSV and the GJR estimator


def rfsv_logvol(H=0.14, nu=0.3, lam=5e-4, m=-5.0, n_days=2000, steps_per_day=192, seed=None):
    """Log-volatility of the RFSV model, X = m + nu (W^H - lam int_0^t e^{-lam(t-s)} W^H_s ds), time in days.

    ``lam`` is the mean-reversion speed (per day; called alpha by Gatheral, Jaisson and Rosenbaum).

    fBM by :func:`fbm_davies_harte`; the integral equation X_t = m - lam int_0^t (X_s - m) ds + nu W^H_t
    (``def:rv-rfsv``) is solved exactly for the piecewise-linear interpolation of W^H on the grid. Defaults are those
    of Gatheral, Jaisson and Rosenbaum (2018), Section 3.4. Returns ``(t, X)`` (one path).
    """
    n = n_days * steps_per_day
    dt = 1.0 / steps_per_day
    t, B = fbm_davies_harte(H, T=float(n_days), n_steps=n, n_paths=1, seed=seed)
    dB = np.diff(B[0])
    e = np.exp(-lam * dt)
    g = nu * (1.0 - e) / (lam * dt) if lam > 0 else nu     # exact for W^H linear on each step
    X = np.concatenate([[0.0], lfilter([g], [1.0, -e], dB)]) + m  # Y_{j+1} = e Y_j + g dB_j
    return t, X


def daily_proxies(t, X, steps_per_day, window=1.0 / 3.0):
    """Daily log-volatility: the spot value at the start of each day, and half the log of the realized-variance
    proxy, the average of sigma^2 = e^{2X} over the grid points of the first ``window`` (in days) of each day.

    The average is a Riemann sum for (1/delta) int_d^{d+delta} sigma_s^2 ds with round(window * steps_per_day) points:
    use a fine grid (192 steps per day give 8 points per hour), or the proxy is an average of very few values."""
    n_days = (len(t) - 1) // steps_per_day
    w = max(1, int(round(window * steps_per_day)))
    var = np.exp(2.0 * X[:-1]).reshape(n_days, steps_per_day)
    spot = X[:-1:steps_per_day][:n_days]
    proxy = 0.5 * np.log(var[:, :w].mean(axis=1))
    return spot, proxy


def gjr_m(x, q, lag):
    """m(q, Delta) of Gatheral, Jaisson and Rosenbaum: the mean of |x_{k Delta} - x_{(k-1) Delta}|^q over the
    subsampled series, averaged over the Delta possible starting points."""
    x = np.asarray(x, dtype=float)
    return np.mean([np.mean(np.abs(np.diff(x[s::lag])) ** q) for s in range(lag)])


def gjr_estimate(x, qs=(0.5, 1.0, 1.5, 2.0, 3.0), lags=range(1, 31)):
    """The GJR procedure: regress log m(q, Delta) on log Delta for each q (slope zeta_q, intercept),
    then fit zeta_q = H q through the origin. Returns a dict with qs, lags, logm, zeta, intercept, H."""
    qs = np.asarray(qs, dtype=float)
    lags = np.asarray(list(lags))
    logm = np.array([[np.log(gjr_m(x, q, int(d))) for d in lags] for q in qs])
    A = np.vstack([np.log(lags), np.ones(lags.size)]).T
    coef = np.linalg.lstsq(A, logm.T, rcond=None)[0]
    zeta, icpt = coef[0], coef[1]
    H = float(qs @ zeta / (qs @ qs))
    return dict(qs=qs, lags=lags, logm=logm, zeta=zeta, intercept=icpt, H=H)


def smoothing_factor(theta, H):
    """f(theta) with E[D^2] = Delta^{2H} f(delta / Delta) for the increment D over a lag Delta of the delta-moving
    average of a standard fBM: f = [(1+th)^{p} + |1-th|^{p} - 2 - 2 th^{p}] / (p (p-1) th^2), p = 2H + 2."""
    th = np.asarray(theta, dtype=float)
    p = 2 * H + 2
    return ((1 + th) ** p + np.abs(1 - th) ** p - 2 - 2 * th ** p) / (p * (p - 1) * th ** 2)


# ---------------------------------------------------------------- rough Bergomi


def rbergomi_first_order_skew(T, H, eta, rho):
    """First-order (in eta) ATM skew (rho eta / 2) E_H T^{H-1/2}, E_H = sqrt(2H) / ((H+1/2)(H+3/2)):
    the Bergomi--Guyon term of Bayer, Friz and Gatheral (2016), Section 7.1, and the short-maturity limit of
    El Euch, Fukasawa, Gatheral and Rosenbaum (2019) (``prop:rv-skew``)."""
    T = np.asarray(T, dtype=float)
    return rho * eta * np.sqrt(2 * H) / (2 * (H + 0.5) * (H + 1.5)) * T ** (H - 0.5)


def rbergomi_integrals(Y, dW, t, xi, eta, rho, H):
    """Given (Y, dW) on the grid t, the variance v_i = xi exp(eta Y_i - eta^2 t_i^{2H} / 2) and the left-point sums
    A = rho sum sqrt(v_i) dW_i - rho^2 sum v_i dt_i / 2 and V = sum v_i dt_i (``def:rv-rbergomi``)."""
    dt = np.diff(t)
    v = xi * np.exp(eta * Y[:, :-1] - 0.5 * eta ** 2 * t[:-1] ** (2 * H))
    V = v @ dt
    A = rho * np.sum(np.sqrt(v) * dW, axis=1) - 0.5 * rho ** 2 * V
    return A, V


def conditional_calls(k, A, V, rho):
    """Per-path conditional call values E[(S_T/S_0 - e^k)^+ | W] = e^A b(k - A, sqrt((1 - rho^2) V)) and their
    k-derivatives -e^k N(d_-), d_- = -(k - A)/s - s/2 (Romano--Touzi conditioning). Shapes (len(k), n_paths)."""
    k = np.atleast_1d(np.asarray(k, dtype=float))[:, None]
    s = np.sqrt(max(1.0 - rho ** 2, 0.0) * V)[None, :]
    x = k - A[None, :]
    c = np.exp(A)[None, :] * black_normalized(x, s)
    ss = np.where(s > 0, s, 1.0)
    dm = np.where(s > 0, -x / ss - 0.5 * ss, np.where(x < 0, np.inf, -np.inf))
    ck = -np.exp(k) * ndtr(dm)
    return c, ck


def iv_and_skew(c, ck, k, T):
    """Implied vol at log-moneyness k from the normalized call price c(k), and the k-derivative of the implied vol
    from c'(k): with c(k) = b(k, s(k)), s' = (c' - d_k b) / d_s b, d_k b = -e^k N(d_-), d_s b = N'(d_+)."""
    s = implied_total_vol(c, k)
    dp, dm = -k / s + 0.5 * s, -k / s - 0.5 * s
    bk = -np.exp(k) * ndtr(dm)                      # d b / d k at fixed s
    bs = np.exp(-0.5 * dp * dp) / SQRT_2PI          # d b / d s = N'(d_+)
    return s / np.sqrt(T), (ck - bk) / (bs * np.sqrt(T))


def rbergomi_smiles(k, maturities, H=0.07, eta=1.9, rho=-0.9, xi=0.235 ** 2, n_steps=256, n_paths=2**15,
                    seed=None, scheme="hybrid", n_batches=20):
    """Implied volatilities and ATM skews of the rough Bergomi model with flat forward variance xi, by conditional
    Monte Carlo with antithetic paths.

    One simulation of (Y, W) on [0, 1] serves every maturity T, through the scaling of ``prop:rv-rbergomi``(iii):
    maturity T with (xi, eta) has the law of maturity 1 with (T xi, T^H eta). ``scheme`` is "hybrid" (ell = 1) or
    "cholesky" (exact on the grid; keep n_steps small). Returns a dict with ``iv`` (len(T), len(k)), ``skew`` (ATM
    derivative of the implied vol in k), and batch standard errors ``iv_se`` and ``skew_se``.
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    maturities = np.atleast_1d(np.asarray(maturities, dtype=float))
    if scheme == "hybrid":
        t, Y, dW = hybrid_rl(H, 1.0, n_steps, n_paths, seed, ell=1, antithetic=True)
    else:
        t, Y, dW = rl_cholesky(H, 1.0, n_steps, n_paths // 2, seed)
        Y, dW = np.concatenate([Y, -Y]), np.concatenate([dW, -dW])
    npaths = Y.shape[0]
    half = npaths // 2
    pair = np.arange(half)
    batches = np.array_split(pair, n_batches)
    kk = np.concatenate([k, [0.0]])
    iv = np.empty((maturities.size, k.size)); iv_se = np.empty_like(iv)
    skew = np.empty(maturities.size); skew_se = np.empty(maturities.size)
    for j, T in enumerate(maturities):
        A, V = rbergomi_integrals(Y, dW, t, T * xi, T ** H * eta, rho, H)
        c, ck = conditional_calls(kk, A, V, rho)
        c = 0.5 * (c[:, :half] + c[:, half:])       # average antithetic pairs
        ck = 0.5 * (ck[:, :half] + ck[:, half:])
        ivs, sks = [], []
        for idx in [pair] + batches:
            cm, ckm = c[:, idx].mean(axis=1), ck[:, idx].mean(axis=1)
            sig = implied_total_vol(cm[:-1], k)      # T = 1 after scaling: total vol of maturity T
            _, sk = iv_and_skew(cm[-1], ckm[-1], 0.0, 1.0)
            ivs.append(sig / np.sqrt(T)); sks.append(sk / np.sqrt(T))
        ivs, sks = np.array(ivs), np.array(sks)
        iv[j], skew[j] = ivs[0], sks[0]
        iv_se[j] = ivs[1:].std(axis=0, ddof=1) / np.sqrt(n_batches)
        skew_se[j] = sks[1:].std(ddof=1) / np.sqrt(n_batches)
    return dict(k=k, T=maturities, iv=iv, iv_se=iv_se, skew=skew, skew_se=skew_se)


def powerlaw_fit(T, y):
    """Least-squares fit of log|y| = log A + beta log T; returns (A, beta)."""
    beta, logA = np.polyfit(np.log(T), np.log(np.abs(y)), 1)
    return np.exp(logA), beta


# ---------------------------------------------------------------- rough Heston


def adams_weights(n_steps, dt, order):
    """Weights of the fractional Adams predictor--corrector (Diethelm, Ford and Freed, 2002) for
    y = I^order f(y): predictor b[k, j] and corrector a[k, j], for the step from t_k to t_{k+1}."""
    N = n_steps
    j = np.arange(N + 1, dtype=float)
    ga1, ga2 = gamma_fn(order + 1), gamma_fn(order + 2)
    b = np.zeros((N, N + 1))
    a = np.zeros((N, N + 2))
    for k in range(N):
        jj = j[:k + 1]
        b[k, :k + 1] = dt ** order / ga1 * ((k + 1 - jj) ** order - (k - jj) ** order)
        a[k, 0] = dt ** order / ga2 * (k ** (order + 1) - (k - order) * (k + 1) ** order)
        if k >= 1:
            jm = j[1:k + 1]
            a[k, 1:k + 1] = dt ** order / ga2 * ((k - jm + 2) ** (order + 1) + (k - jm) ** (order + 1)
                                                 - 2 * (k - jm + 1) ** (order + 1))
        a[k, k + 1] = dt ** order / ga2
    return b, a


def rough_heston_cf(z, T, V0, theta, lam, nu, rho, H, n_steps=200):
    """E[exp(i z log(S_T / S_0))] in the rough Heston model (``thm:rv-rough-heston``), for complex z with
    -1 <= Im z <= 0 (an array).

    h(z, .) solves the Volterra form h = I^order F(z, h), order = H + 1/2, of the fractional Riccati equation
    F(z, x) = -(z^2 + i z)/2 + lam (i z rho nu - 1) x + (lam nu)^2 x^2 / 2, by the fractional Adams scheme on n_steps
    steps; then log cf = theta lam int_0^T h + V0 int_0^T F(z, h) (since I^{1-order} h = I^1 F), the time integrals
    by the trapezoidal rule. At H = 1/2 this is the Heston characteristic function with kappa = lam, xi = lam nu.
    """
    z = np.asarray(z, dtype=complex)
    order = H + 0.5
    dt = T / n_steps
    c0 = -0.5 * (z * z + 1j * z)
    c1 = lam * (1j * z * rho * nu - 1.0)
    c2 = 0.5 * (lam * nu) ** 2

    def F(x):
        return c0 + c1 * x + c2 * x * x

    b, a = adams_weights(n_steps, dt, order)
    h = np.zeros((n_steps + 1,) + z.shape, dtype=complex)
    Fh = np.zeros_like(h)
    Fh[0] = F(h[0])
    for k in range(n_steps):
        pred = np.tensordot(b[k, :k + 1], Fh[:k + 1], axes=1)
        corr = np.tensordot(a[k, :k + 1], Fh[:k + 1], axes=1)
        h[k + 1] = corr + a[k, k + 1] * F(pred)
        Fh[k + 1] = F(h[k + 1])
    w = np.full(n_steps + 1, dt)
    w[0] = w[-1] = 0.5 * dt
    I_h = np.tensordot(w, h, axes=1)
    I_F = np.tensordot(w, Fh, axes=1)
    return np.exp(theta * lam * I_h + V0 * I_F)


def fourier_nodes(T, v_eff, h=0.1, u_min=60.0, u_max_cap=1500.0):
    """Trapezoidal nodes and weights on [0, u_max] for the Lewis integrals of ``lem:iv-lewis``."""
    u_max = min(max(u_min, np.sqrt(2.0 * 60.0 / (max(v_eff, 1e-4) * T))), u_max_cap)
    n = int(np.ceil(u_max / h))
    u = h * np.arange(n + 1)
    w = np.full(n + 1, h)
    w[0] = 0.5 * h
    return u, w


def smile_from_cf(cf_shifted, u, w, k, T):
    """Implied vols at log-moneyness k and the ATM skew d sigma / d k at k = 0, from phi(u - i/2) on the nodes u."""
    k = np.atleast_1d(np.asarray(k, dtype=float))
    c = 1.0 - np.exp(0.5 * k) * lewis_transforms(cf_shifted, k, u, w, order=0)
    iv = implied_total_vol(c, k) / np.sqrt(T)
    c0 = 1.0 - lewis_transforms(cf_shifted, 0.0, u, w, order=0)[0]
    ck0 = -lewis_transforms(cf_shifted, 0.0, u, w, order=1)[0]
    _, skew = iv_and_skew(c0, ck0, 0.0, T)
    return iv, skew


def rough_heston_smile(k, T, V0, theta, lam, nu, rho, H, n_steps=200, h=0.1, v_eff=None):
    """Implied vols and ATM skew of the rough Heston model at maturity T (Lewis's formula, ``lem:iv-lewis``).

    ``v_eff`` (default min(V0, theta)) is a lower estimate of the variance over [0, T], used to truncate the
    Fourier integral."""
    u, w = fourier_nodes(T, min(V0, theta) if v_eff is None else v_eff, h)
    phi = rough_heston_cf(u - 0.5j, T, V0, theta, lam, nu, rho, H, n_steps)
    return smile_from_cf(phi, u, w, k, T)


# ---------------------------------------------------------------- reading the public realized-variance file


SPX_RV_URL = "https://raw.githubusercontent.com/mpiktas/midasr/v0.9/data/rvsp500.RData"
SPX_RV_SHA256 = "cc3ea729a7d0f2e2ae7ff8befab0f0d27b91c72dbe4dfd88d6798e4d958a85e1"


def parse_rdata(raw: bytes):
    """Minimal reader for R's XDR serialization (gzip-compressed .RData, format RDX2/RDX3), enough for a data.frame
    of numeric and integer columns. Returns {name: object}; a data.frame comes back as {column name: numpy array}.

    Only the types needed here are understood (NULL, symbols, pairlists, character, logical, integer, double and
    list vectors); anything else raises ValueError. The input is treated as untrusted data.
    """
    buf = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    if buf[:5] not in (b"RDX2\n", b"RDX3\n") or buf[5:7] != b"X\n":
        raise ValueError("not an XDR .RData file")
    pos = 7
    refs = []

    def rint():
        nonlocal pos
        if pos + 4 > len(buf):
            raise ValueError("truncated file")
        v = struct.unpack(">i", buf[pos:pos + 4])[0]
        pos += 4
        return v

    def item():
        nonlocal pos
        flags = rint()
        typ, has_attr, has_tag = flags & 0xFF, bool(flags & 512), bool(flags & 1024)
        if typ in (254, 242):                       # NILVALUE_SXP, EMPTYENV
            return None
        if typ == 255:                               # REFSXP
            return refs[(flags >> 8) - 1]
        if typ == 1:                                 # SYMSXP
            name = item()
            refs.append(name)
            return name
        if typ == 9:                                 # CHARSXP
            n = rint()
            if n == -1:
                return None
            s = buf[pos:pos + n].decode("utf-8", "replace")
            pos += n
            return s
        if typ == 2:                                 # LISTSXP (pairlist)
            out = {}
            while True:
                attr = item() if has_attr else None  # noqa: F841 (attributes of pairlist cells unused)
                tag = item() if has_tag else None
                out[tag] = item()
                flags = rint()
                typ, has_attr, has_tag = flags & 0xFF, bool(flags & 512), bool(flags & 1024)
                if typ == 254:
                    return out
                if typ != 2:
                    raise ValueError(f"unexpected type {typ} in pairlist")
        if typ in (10, 13, 14):                      # LGLSXP, INTSXP, REALSXP
            n = rint()
            size = 8 if typ == 14 else 4
            if n < 0 or pos + size * n > len(buf):
                raise ValueError("bad vector length")
            v = np.frombuffer(buf, ">f8" if typ == 14 else ">i4", n, pos).astype(float if typ == 14 else np.int64)
            pos += size * n
        elif typ in (16, 19):                        # STRSXP, VECSXP
            n = rint()
            if n < 0 or n > len(buf):
                raise ValueError("bad vector length")
            v = [item() for _ in range(n)]
        else:
            raise ValueError(f"unsupported R type {typ}")
        attrs = item() if has_attr else None
        if typ == 19 and isinstance(attrs, dict) and attrs.get("names") is not None:
            return {name: col for name, col in zip(attrs["names"], v)}
        return v

    rint(); rint(); rint()                           # format version, R version, minimal version
    if buf[3:4] == b"3":                             # RDX3 adds the native encoding
        n = rint()
        pos += n
    top = item()
    return top if isinstance(top, dict) else {"value": top}


def load_spx_realized_variance(url=SPX_RV_URL, timeout=20, sha256=SPX_RV_SHA256):
    """Daily 5-minute realized variance of the S&P 500 (column SPX2.rv, January 2000 to November 2013) from the
    Oxford-Man Institute's realized library (Heber, Lunde, Shephard and Sheppard, 2009), downloaded at run time from the
    copy in the R package midasr (tag v0.9). The data are the Oxford-Man Institute's and subject to its terms of use;
    nothing is stored in this repository.

    Returns (dates as YYYYMMDD integers, daily realized variance), or None if the environment variable IEOR222_OFFLINE
    is "1", if the file cannot be reached, or if its SHA-256 differs from the pinned one.
    """
    import hashlib
    import os
    import urllib.request
    if os.environ.get("IEOR222_OFFLINE") == "1":
        return None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read(5_000_000)
        if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
            return None
        df = parse_rdata(raw)["rvsp500"]
        dates = np.asarray(df["DateID"], dtype=np.int64)
        rv = np.asarray(df["SPX2.rv"], dtype=float)
        ok = np.isfinite(rv) & (rv > 0)
        return dates[ok], rv[ok]
    except Exception:
        return None
