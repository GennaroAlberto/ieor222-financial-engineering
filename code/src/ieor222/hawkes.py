"""Hawkes processes: simulation, likelihood, goodness of fit, a toy limit order book (Chapter 18).

Conventions follow the notes. A d-variate Hawkes process with exponential kernels has intensities

    lambda^i(t) = mu_i + sum_j sum_{t^j_k < t} alpha_ij beta_ij exp(-beta_ij (t - t^j_k)),

so that phi_ij(t) = alpha_ij beta_ij e^{-beta_ij t} has integral alpha_ij and the branching matrix
is G = (alpha_ij) (``def:hk-hawkes``); G[i, j] is the mean number of type-i children of a type-j
event. Every process starts from an empty history at time 0. A path on [0, T] is returned as two
arrays: increasing event times and integer types in {0, ..., d-1}. Samplers take a ``seed``.

Sums of exponentials over past events, sum_{s_l < q} e^{-beta (q - s_l)}, are evaluated in the log
domain with ``np.logaddexp.accumulate``, so that likelihoods cost O(n) vectorised operations and
never overflow.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats
from scipy.linalg import expm

# ------------------------------------------------------------------ parameters and moments


def as_params(mu, alpha, beta):
    """Return (mu, alpha, beta) as float arrays of shapes (d,), (d, d), (d, d)."""
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    d = mu.size
    alpha = np.asarray(alpha, dtype=float).reshape(d, d)
    beta = np.array(np.broadcast_to(np.asarray(beta, dtype=float), (d, d)), dtype=float)
    if np.any(mu < 0) or np.any(alpha < 0) or np.any(beta <= 0):
        raise ValueError("need mu >= 0, alpha >= 0 and beta > 0")
    return mu, alpha, beta


def spectral_radius(G) -> float:
    """Largest modulus of an eigenvalue of the square matrix G."""
    return float(np.max(np.abs(np.linalg.eigvals(np.atleast_2d(np.asarray(G, dtype=float))))))


def mean_intensity(mu, G) -> np.ndarray:
    """Lambda = (I - G)^{-1} mu, the limit of E[lambda(t)] when rho(G) < 1 (``prop:hk-stationary``)."""
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    G = np.asarray(G, dtype=float).reshape(mu.size, mu.size)
    if spectral_radius(G) >= 1:
        raise ValueError("the spectral radius of G must be < 1")
    return np.linalg.solve(np.eye(mu.size) - G, mu)


def exp_hawkes_mean(t, mu, alpha, beta):
    """E[lambda(t)] and E[N(t)] for exponential kernels from an empty history, exactly.

    With R_ij(t) = sum_{t^j_k < t} beta_ij e^{-beta_ij (t - t^j_k)}, lambda^i = mu_i + sum_j alpha_ij R_ij
    and dR_ij = -beta_ij R_ij dt + beta_ij dN^j. Taking expectations (``lem:hk-compensator``),
    r_ij = E R_ij solves r_ij' = beta_ij (m_j - r_ij), with m = mu + (alpha * r).sum(1), and E N' = m:
    an affine ODE, solved here by a matrix exponential. Returns two arrays of shape (len(t), d).
    """
    mu, alpha, beta = as_params(mu, alpha, beta)
    d = mu.size
    n = d * d + d                       # state: r (row-major), then E N
    M = np.zeros((n + 1, n + 1))
    for i in range(d):
        for j in range(d):
            row = i * d + j
            M[row, row] -= beta[i, j]
            for l in range(d):          # beta_ij m_j = beta_ij (mu_j + sum_l alpha_jl r_jl)
                M[row, j * d + l] += beta[i, j] * alpha[j, l]
            M[row, n] += beta[i, j] * mu[j]
    for i in range(d):                  # d E N_i / dt = mu_i + sum_l alpha_il r_il
        for l in range(d):
            M[d * d + i, i * d + l] += alpha[i, l]
        M[d * d + i, n] += mu[i]
    z0 = np.zeros(n + 1)
    z0[n] = 1.0
    t = np.atleast_1d(np.asarray(t, dtype=float))
    Z = np.array([expm(M * s) @ z0 for s in t])
    r = Z[:, : d * d].reshape(-1, d, d)
    m = mu + np.einsum("ij,tij->ti", alpha, r)
    return m, Z[:, d * d: d * d + d]


def exp_hawkes_intensity_moments(t, mu, a, beta):
    """Mean and variance of lambda(t) for d = 1, kernel a beta e^{-beta t}, from an empty history.

    Between events lambda decays towards mu at rate beta and it jumps by a beta at each event, so
    m = E lambda and s = E lambda^2 solve m' = beta (mu - (1 - a) m) and
    s' = -2 beta (1 - a) s + 2 beta mu m + a^2 beta^2 m, with m(0) = mu, s(0) = mu^2.
    """
    M = np.array([[-beta * (1 - a), 0.0, beta * mu],
                  [2 * beta * mu + a * a * beta * beta, -2 * beta * (1 - a), 0.0],
                  [0.0, 0.0, 0.0]])
    z0 = np.array([mu, mu * mu, 1.0])
    t = np.atleast_1d(np.asarray(t, dtype=float))
    Z = np.array([expm(M * s) @ z0 for s in t])
    return Z[:, 0], Z[:, 1] - Z[:, 0] ** 2


def cir_from_zero_gamma(mu, lam, m, t=1.0):
    """Law at time t of the CIR limit of ``thm:hk-scaling-limit`` started at 0, as (shape, scale).

    dX = (lam / m)(mu - X) dt + (sqrt(lam) / m) sqrt(X) dB, X_0 = 0, has a Gamma law at time t with
    shape 2 kappa mu / sigma^2 = 2 mu m and scale sigma^2 (1 - e^{-kappa t}) / (2 kappa) = (1 - e^{-lam t / m}) / (2 m).
    """
    return 2 * mu * m, (1 - math.exp(-lam * t / m)) / (2 * m)


# ------------------------------------------------------------------ sums of exponentials


def _exp_sums(sources, queries, beta, weights=None):
    """S(q) = sum_{s_l < q} w_l e^{-beta (q - s_l)} for sorted ``sources`` and any ``queries``.

    Computed as exp(log-cumsum-exp(beta s_l + log w_l) - beta q); weights must be >= 0.
    """
    sources = np.asarray(sources, dtype=float)
    queries = np.asarray(queries, dtype=float)
    if sources.size == 0:
        return np.zeros(queries.shape)
    with np.errstate(divide="ignore"):
        logw = 0.0 if weights is None else np.log(np.asarray(weights, dtype=float))
    cum = np.logaddexp.accumulate(beta * sources + logw)
    idx = np.searchsorted(sources, queries, side="left") - 1   # last source strictly before q
    out = np.zeros(queries.shape)
    ok = idx >= 0
    out[ok] = np.exp(cum[idx[ok]] - beta * queries[ok])
    return out


def _linear_recursion(t, b, beta):
    """x_1 = 0, x_i = e^{-beta (t_i - t_{i-1})} (x_{i-1} + b_i): x_i = sum_{2<=j<=i} e^{-beta (t_i - t_{j-1})} b_j."""
    t = np.asarray(t, dtype=float)
    x = np.zeros(t.size)
    if t.size < 2:
        return x
    with np.errstate(divide="ignore"):
        u = beta * t[:-1] + np.log(np.asarray(b, dtype=float)[1:])
    x[1:] = np.exp(np.logaddexp.accumulate(u) - beta * t[1:])
    return x


# ------------------------------------------------------------------ intensities and compensators


def intensity_on_grid(grid, times, types, mu, alpha, beta):
    """lambda^i(t) at the points of ``grid`` (left limits at event times); shape (len(grid), d)."""
    mu, alpha, beta = as_params(mu, alpha, beta)
    times, types = np.asarray(times, dtype=float), np.asarray(types, dtype=int)
    grid = np.asarray(grid, dtype=float)
    lam = np.tile(mu, (grid.size, 1))
    for j in range(mu.size):
        s = times[types == j]
        for i in range(mu.size):
            if alpha[i, j] > 0:
                lam[:, i] += alpha[i, j] * beta[i, j] * _exp_sums(s, grid, beta[i, j])
    return lam


def compensators(queries, times, types, mu, alpha, beta):
    """Lambda^i(q) = int_0^q lambda^i(s) ds for every type i; shape (len(queries), d).

    For exponential kernels Lambda^i(q) = mu_i q + sum_j alpha_ij sum_{t^j_k < q} (1 - e^{-beta_ij (q - t^j_k)}).
    """
    mu, alpha, beta = as_params(mu, alpha, beta)
    times, types = np.asarray(times, dtype=float), np.asarray(types, dtype=int)
    q = np.asarray(queries, dtype=float)
    out = np.outer(q, mu)
    for j in range(mu.size):
        s = times[types == j]
        count = np.searchsorted(s, q, side="left")
        for i in range(mu.size):
            if alpha[i, j] > 0:
                out[:, i] += alpha[i, j] * (count - _exp_sums(s, q, beta[i, j]))
    return out


def time_changed_gaps(times, types, mu, alpha, beta):
    """Transformed inter-arrival times Lambda^i(t^i_k) - Lambda^i(t^i_{k-1}), one array per type.

    Under the model they are i.i.d. Exp(1) (``thm:hk-time-change``); the first gap is Lambda^i(t^i_1).
    """
    times, types = np.asarray(times, dtype=float), np.asarray(types, dtype=int)
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    gaps = []
    for i in range(mu.size):
        ti = times[types == i]
        Li = compensators(ti, times, types, mu, alpha, beta)[:, i]
        gaps.append(np.diff(np.concatenate([[0.0], Li])))
    return gaps


def ks_exponential(gaps):
    """Kolmogorov--Smirnov test of i.i.d. Exp(1); returns (statistic, p-value)."""
    res = stats.kstest(np.asarray(gaps, dtype=float), "expon")
    return float(res.statistic), float(res.pvalue)


# ------------------------------------------------------------------ simulation


def simulate_ogata(T, mu, alpha, beta, seed=None, max_events=5_000_000):
    """Ogata's thinning algorithm (``prop:hk-ogata``) for exponential kernels on [0, T].

    Between events every lambda^i decreases, so the total intensity just after the current time
    bounds the total intensity until the next accepted event. A candidate at time s is accepted
    as an event of type i when the uniform U on [0, M] falls in the i-th piece of
    [0, lambda^1(s)] u ... u [.., lambda^1(s) + ... + lambda^d(s)].
    """
    rng = np.random.default_rng(seed)
    mu, alpha, beta = as_params(mu, alpha, beta)
    if mu.size == 1:
        return _ogata_univariate(T, mu[0], alpha[0, 0], beta[0, 0], rng, max_events)
    return _ogata_multivariate(T, mu, alpha, beta, rng, max_events)


def _ogata_multivariate(T, mu, alpha, beta, rng, max_events):
    d = mu.size
    ab = alpha * beta
    R = np.zeros((d, d))                # R[i, j] = sum over past type-j events of e^{-beta_ij (t - t_k)}
    t = 0.0
    lam = mu.copy()
    times, types = [], []
    while len(times) < max_events:
        M = lam.sum()
        if M <= 0.0:
            break
        w = rng.exponential(1.0 / M)
        t += w
        if t > T:
            break
        R *= np.exp(-beta * w)
        lam = mu + (ab * R).sum(axis=1)
        cum = np.cumsum(lam)
        u = rng.uniform(0.0, M)
        if u <= cum[-1]:                # accept; the type is the piece of [0, M] containing u
            i = int(np.searchsorted(cum, u, side="left"))
            times.append(t)
            types.append(i)
            R[:, i] += 1.0
            lam = mu + (ab * R).sum(axis=1)
    return np.array(times), np.array(types, dtype=int)


def _ogata_univariate(T, mu, alpha, beta, rng, max_events):
    """Scalar version of the thinning loop, with random numbers drawn in blocks."""
    times = []
    t, S = 0.0, 0.0                     # S = sum_{t_k < t} e^{-beta (t - t_k)}
    lam = mu
    block = 4096
    E = rng.exponential(1.0, block)
    U = rng.random(block)
    p = 0
    ab = alpha * beta
    while len(times) < max_events:
        if p == block:
            E, U, p = rng.exponential(1.0, block), rng.random(block), 0
        M = lam
        if M <= 0.0:
            break
        w = E[p] / M
        t += w
        if t > T:
            break
        S *= math.exp(-beta * w)
        lam = mu + ab * S
        if U[p] * M <= lam:
            times.append(t)
            S += 1.0
            lam += ab
        p += 1
    return np.array(times), np.zeros(len(times), dtype=int)


def exponential_lags(beta):
    """Lag sampler for ``simulate_branching``: lags of a type-j parent's type-i children ~ Exp(beta_ij)."""
    beta = np.atleast_2d(np.asarray(beta, dtype=float))

    def sample(rng, i, j, n):
        b = beta[i, j] if beta.size > 1 else beta[0, 0]
        return rng.exponential(1.0 / b, n)
    return sample


def lomax_lags(tail, c=1.0):
    """Lag sampler for the heavy-tailed density tail c^tail / (c + x)^{1 + tail} (inversion)."""
    def sample(rng, i, j, n):
        return c * (rng.random(n) ** (-1.0 / tail) - 1.0)
    return sample


def simulate_branching(T, mu, G, lags, seed=None, n_paths=1, return_generation=False):
    """Cluster construction of ``prop:hk-branching`` on [0, T], generation by generation.

    Immigrants of type i form a Poisson process of rate mu_i; an event of type j has a
    Poisson(G[i, j]) number of type-i children, at lags drawn by ``lags(rng, i, j, n)`` (the kernel
    phi_ij / G_ij as a density). Children after T are dropped (their descendants come later still).
    With ``n_paths > 1`` independent paths are simulated together and a path index is returned.
    Returns (times, types) or (times, types, path) sorted by path then time, plus the generation
    of each event if ``return_generation``.
    """
    rng = np.random.default_rng(seed)
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    d = mu.size
    G = np.asarray(G, dtype=float).reshape(d, d)
    t_list, k_list, p_list, g_list = [], [], [], []
    cur_t, cur_k, cur_p = [], [], []
    for i in range(d):
        n = rng.poisson(mu[i] * T, n_paths)
        cur_t.append(rng.uniform(0.0, T, n.sum()))
        cur_k.append(np.full(n.sum(), i))
        cur_p.append(np.repeat(np.arange(n_paths), n))
    cur_t, cur_k, cur_p = np.concatenate(cur_t), np.concatenate(cur_k), np.concatenate(cur_p)
    g = 0
    while cur_t.size:
        t_list.append(cur_t), k_list.append(cur_k), p_list.append(cur_p)
        g_list.append(np.full(cur_t.size, g))
        g += 1
        nt, nk, npth = [], [], []
        for j in range(d):
            sel = cur_k == j
            if not sel.any():
                continue
            par_t, par_p = cur_t[sel], cur_p[sel]
            for i in range(d):
                if G[i, j] <= 0:
                    continue
                nc = rng.poisson(G[i, j], par_t.size)
                tot = int(nc.sum())
                if tot == 0:
                    continue
                birth = np.repeat(par_t, nc) + lags(rng, i, j, tot)
                keep = birth <= T
                nt.append(birth[keep])
                nk.append(np.full(int(keep.sum()), i))
                npth.append(np.repeat(par_p, nc)[keep])
        if nt:
            cur_t, cur_k, cur_p = np.concatenate(nt), np.concatenate(nk), np.concatenate(npth)
        else:
            cur_t = np.zeros(0)
    times = np.concatenate(t_list) if t_list else np.zeros(0)
    types = np.concatenate(k_list).astype(int) if k_list else np.zeros(0, dtype=int)
    path = np.concatenate(p_list) if p_list else np.zeros(0, dtype=int)
    gen = np.concatenate(g_list) if g_list else np.zeros(0, dtype=int)
    order = np.lexsort((times, path))
    out = [times[order], types[order]]
    if n_paths > 1:
        out.append(path[order])
    if return_generation:
        out.append(gen[order])
    return tuple(out)


# ------------------------------------------------------------------ likelihood: d = 1, (mu, alpha, beta)


def _univariate_pieces(times, T, beta):
    """A_i, B_i, C_i = sum_{j<i} (t_i - t_j)^p e^{-beta (t_i - t_j)} for p = 0, 1, 2, and the end terms."""
    t = np.asarray(times, dtype=float)
    dt = np.diff(t, prepend=0.0)
    A = _linear_recursion(t, np.ones(t.size), beta)
    A_prev = np.concatenate([[0.0], A[:-1]])
    B = _linear_recursion(t, dt * (1.0 + A_prev), beta)
    B_prev = np.concatenate([[0.0], B[:-1]])
    C = _linear_recursion(t, 2 * dt * B_prev + dt * dt * (1.0 + A_prev), beta)
    tau = T - t
    e = np.exp(-beta * tau)
    return A, B, C, tau, e


def exp_loglik(theta, times, T, derivatives=False):
    """Log-likelihood of ``prop:hk-mle`` for d = 1 and theta = (mu, alpha, beta), with its gradient and Hessian.

    log L = sum_i log(mu + alpha beta A_i) - mu T - alpha sum_i (1 - e^{-beta (T - t_i)}),
    A_i = sum_{j<i} e^{-beta (t_i - t_j)} computed by the O(n) recursion of ``prop:hk-mle``(ii).
    """
    mu, alpha, beta = (float(x) for x in theta)
    A, B, C, tau, e = _univariate_pieces(times, T, beta)
    lam = mu + alpha * beta * A
    if np.any(lam <= 0):
        return (-np.inf, None, None) if derivatives else -np.inf
    ll = np.sum(np.log(lam)) - mu * T - alpha * np.sum(1.0 - e)
    if not derivatives:
        return ll
    dl = np.stack([np.ones_like(A), beta * A, alpha * (A - beta * B)])          # d lambda_i / d theta
    dLam = np.array([T, np.sum(1.0 - e), alpha * np.sum(tau * e)])
    grad = (dl / lam).sum(axis=1) - dLam
    H = -(dl / lam) @ (dl / lam).T
    H[1, 2] += np.sum((A - beta * B) / lam)
    H[2, 1] = H[1, 2]
    H[2, 2] += np.sum(alpha * (beta * C - 2.0 * B) / lam)
    H[1, 2] -= np.sum(tau * e)
    H[2, 1] -= np.sum(tau * e)
    H[2, 2] += alpha * np.sum(tau * tau * e)
    return ll, grad, H


def fit_exp_hawkes(times, T, x0=None):
    """Maximum likelihood for d = 1: returns dict with 'theta' = (mu, alpha, beta), 'se', 'cov', 'loglik'.

    Standard errors are square roots of the diagonal of the inverse observed information
    -Hessian at the maximum (``prop:hk-mle``(iv)). Several starting decays are tried.
    """
    times = np.asarray(times, dtype=float)
    n = times.size
    rate = max(n / T, 1e-12)
    starts = [x0] if x0 is not None else [(0.5 * rate, 0.5, b * rate) for b in (0.3, 3.0, 30.0)]

    def f(x):
        ll, g, _ = exp_loglik(x, times, T, derivatives=True)
        if not np.isfinite(ll):
            return 1e300, np.zeros(3)
        return -ll, -g

    best = None
    for s in starts:
        res = optimize.minimize(f, np.asarray(s, dtype=float), jac=True, method="L-BFGS-B",
                                bounds=[(1e-10, None), (0.0, None), (1e-8, None)],
                                options=dict(maxiter=500, ftol=1e-13, gtol=1e-9))
        if best is None or res.fun < best.fun:
            best = res
    ll, g, H = exp_loglik(best.x, times, T, derivatives=True)
    cov = np.linalg.inv(-H)
    return dict(theta=best.x, se=np.sqrt(np.diag(cov)), cov=cov, loglik=ll, success=bool(best.success))


# ------------------------------------------------------------------ likelihood: d >= 1, fixed decays


def design_fixed_beta(times, types, T, beta, d):
    """For fixed decays: regressors x_k = (1, R_i1(t_k), ..., R_id(t_k)) at the events of each type i
    and the compensator vector c_i = (T, Lambda_i1, ..., Lambda_id), so lambda^i(t_k) = theta_i . x_k and
    Lambda^i(T) = theta_i . c_i with theta_i = (mu_i, alpha_i1, ..., alpha_id) (``prop:hk-mle``(iii)).
    """
    times, types = np.asarray(times, dtype=float), np.asarray(types, dtype=int)
    beta = np.array(np.broadcast_to(np.asarray(beta, dtype=float), (d, d)))
    X, c = [], []
    for i in range(d):
        ti = times[types == i]
        Xi = np.ones((ti.size, d + 1))
        ci = np.zeros(d + 1)
        ci[0] = T
        for j in range(d):
            s = times[types == j]
            Xi[:, 1 + j] = beta[i, j] * _exp_sums(s, ti, beta[i, j])
            ci[1 + j] = np.sum(1.0 - np.exp(-beta[i, j] * (T - s)))
        X.append(Xi)
        c.append(ci)
    return X, c


def fit_fixed_beta(times, types, T, beta, d):
    """MLE of mu and alpha for known decays: d separate concave problems (``prop:hk-mle``(iii)).

    Returns dict with 'mu', 'alpha', 'se_mu', 'se_alpha', 'loglik'. Standard errors come from the
    inverse of sum_k x_k x_k^T / lambda_k^2 (the observed information) and are meaningful for
    parameters away from the boundary 0.
    """
    X, c = design_fixed_beta(times, types, T, beta, d)
    mu, alpha = np.zeros(d), np.zeros((d, d))
    se_mu, se_alpha = np.zeros(d), np.zeros((d, d))
    ll_tot = 0.0
    for i in range(d):
        Xi, ci = X[i], c[i]
        ni = Xi.shape[0]

        def f(th, Xi=Xi, ci=ci):
            lam = Xi @ th
            if np.any(lam <= 0):
                return 1e300, np.zeros_like(th)
            return -(np.sum(np.log(lam)) - th @ ci), -(Xi.T @ (1.0 / lam) - ci)

        x0 = np.full(d + 1, 0.1)
        x0[0] = max(ni / T, 1e-6) * 0.5
        res = optimize.minimize(f, x0, jac=True, method="L-BFGS-B",
                                bounds=[(1e-10, None)] + [(0.0, None)] * d,
                                options=dict(maxiter=2000, ftol=1e-14, gtol=1e-10))
        th = res.x
        lam = Xi @ th
        info = (Xi / lam[:, None]).T @ (Xi / lam[:, None])
        cov = np.linalg.pinv(info)
        mu[i], alpha[i] = th[0], th[1:]
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        se_mu[i], se_alpha[i] = se[0], se[1:]
        ll_tot += -res.fun
    return dict(mu=mu, alpha=alpha, se_mu=se_mu, se_alpha=se_alpha, loglik=ll_tot)


def multivariate_loglik(times, types, T, mu, alpha, beta):
    """sum_i [ sum_{t^i_k} log lambda^i(t^i_k) - Lambda^i(T) ] (``prop:hk-mle``(i)), any exponential kernels."""
    mu, alpha, beta = as_params(mu, alpha, beta)
    times, types = np.asarray(times, dtype=float), np.asarray(types, dtype=int)
    ll = 0.0
    for i in range(mu.size):
        ti = times[types == i]
        lam = intensity_on_grid(ti, times, types, mu, alpha, beta)[:, i]
        ll += np.sum(np.log(lam))
    return ll - compensators([T], times, types, mu, alpha, beta)[0].sum()


# ------------------------------------------------------------------ a toy limit order book

LOB_TYPES = ("MB", "MS", "LB", "LA", "CB", "CA")
LOB_NAMES = ("market buy", "market sell", "limit bid", "limit ask", "cancel bid", "cancel ask")


def lob_toy_model():
    """An illustrative 6-dimensional order-flow model (not estimated from data).

    Types: 0 market buy, 1 market sell, 2 limit order at the bid, 3 at the ask, 4 cancellation at
    the bid, 5 at the ask. The branching matrix (rows: excited type, columns: exciting type) puts
    self-excitation on market orders (splitting of large orders), refills the side a market order
    has just hit (resilience), cancels recent limit orders, and is symmetric between the two sides.
    Times are in seconds. Returns (mu, G, beta).
    """
    MB, MS, LB, LA, CB, CA = range(6)
    G = np.zeros((6, 6))
    for buy, sell, lb, la, cb, ca in ((MB, MS, LB, LA, CB, CA), (MS, MB, LA, LB, CA, CB)):
        G[buy, buy] = 0.65      # market orders cluster (order splitting)
        G[la, buy] = 0.30       # a market buy is followed by new limit orders at the ask
        G[ca, buy] = 0.05
        G[lb, lb] = 0.30        # limit orders cluster
        G[cb, lb] = 0.60        # ... and are often cancelled soon after
        G[lb, cb] = 0.25        # a cancellation is often replaced
        G[sell, cb] = 0.10      # a thinning bid queue attracts market sells
    mu = np.array([0.03, 0.03, 0.10, 0.10, 0.04, 0.04])
    beta = np.full((6, 6), 4.0)
    return mu, G, beta


def lob_mid_price(times, types, q0=4, tick=0.01):
    """Mid-price path of a toy queue mechanism driven by the order flow of ``lob_toy_model``.

    The best bid and ask queues hold q0 lots each. Limit orders add one lot to their queue; market
    orders and cancellations remove one (a cancellation at an empty queue is ignored). When a
    queue is depleted the mid-price moves one tick away from it and both queues restart at q0.
    Returns the mid-price after each event (starting from 0).
    """
    qb, qa, mid = q0, q0, 0.0
    out = np.empty(len(times))
    for k, e in enumerate(types):
        if e == 0 or e == 5:
            qa -= 1
        elif e == 1 or e == 4:
            qb -= 1
        elif e == 2:
            qb += 1
        elif e == 3:
            qa += 1
        if qa <= 0:
            mid += tick
            qa = qb = q0
        elif qb <= 0:
            mid -= tick
            qa = qb = q0
        out[k] = mid
    return out


def lobster_to_events(message):
    """Map a LOBSTER message array to (times, types) in the coding of ``lob_toy_model``.

    A LOBSTER message file has columns (time in seconds after midnight, event type, order id,
    size, price, direction). Event types: 1 new limit order, 2 partial cancellation, 3 deletion,
    4 execution of a visible limit order, 5 execution of a hidden one, 6 cross trade, 7 halt.
    Direction is that of the limit order: +1 buy (bid side), -1 sell (ask side); the execution of a
    sell limit order is a buyer-initiated trade. Only types 1-4 are kept; messages with the same time
    stamp and the same coded type (one market order filling several limit orders) are merged.
    """
    m = np.asarray(message, dtype=float)
    t, ev, dirn = m[:, 0], m[:, 1].astype(int), m[:, 5].astype(int)
    code = np.full(t.size, -1)
    code[(ev == 1) & (dirn == 1)] = 2
    code[(ev == 1) & (dirn == -1)] = 3
    code[np.isin(ev, (2, 3)) & (dirn == 1)] = 4
    code[np.isin(ev, (2, 3)) & (dirn == -1)] = 5
    code[(ev == 4) & (dirn == -1)] = 0
    code[(ev == 4) & (dirn == 1)] = 1
    keep = code >= 0
    t, code = t[keep], code[keep]
    if t.size:
        new = np.ones(t.size, dtype=bool)
        new[1:] = (np.diff(t) > 0) | (np.diff(code) != 0)
        t, code = t[new], code[new]
    return t - (t[0] if t.size else 0.0), code


# ------------------------------------------------------------------ studies used in the notes


def mle_study(n_rep, T, mu, alpha, beta, seed=0):
    """Fit (mu, alpha, beta) by maximum likelihood on ``n_rep`` independent paths on [0, T].

    Paths are drawn with the cluster construction (``prop:hk-branching``), which has the law of
    the Ogata sampler and is vectorised. Returns (estimates, standard errors), each (n_rep, 3).
    """
    rng = np.random.default_rng(seed)
    est, se = np.zeros((n_rep, 3)), np.zeros((n_rep, 3))
    for r in range(n_rep):
        t, _ = simulate_branching(T, [mu], [[alpha]], exponential_lags(beta), seed=rng.integers(2**63))
        f = fit_exp_hawkes(t, T)
        est[r], se[r] = f["theta"], f["se"]
    return est, se


def nearly_unstable_light(T, mu, lam, beta, n_paths, seed=0):
    """Samples of C^T_1 = (1 - a_T) lambda^T_T with a_T = 1 - lam / T and kernel a_T beta e^{-beta t}.

    By ``thm:hk-scaling-limit``(i) (m = 1 / beta) the law converges to that of the CIR limit at
    time 1, the Gamma law of :func:`cir_from_zero_gamma`.
    """
    a = 1.0 - lam / T
    t, _, p = simulate_branching(T, [mu], [[a]], exponential_lags(beta), seed=seed, n_paths=n_paths)
    excite = np.bincount(p, weights=a * beta * np.exp(-beta * (T - t)), minlength=n_paths)
    return (1.0 - a) * (mu + excite)


def heavy_tail_scaling(T, tail, lam=1.0, mu_star=1.0, c=1.0):
    """Parameters of ``thm:hk-scaling-limit``(ii) for the kernel a_T tail c^tail (c + t)^{-1-tail}.

    The tail constant is K = tail c^tail, delta = K Gamma(1 - tail) / tail = c^tail Gamma(1 - tail),
    1 - a_T = lam delta T^{-tail} and mu_T = (mu_star / delta) T^{tail - 1}. Returns
    (a_T, mu_T, scale) with X^T_t = scale * N^T_{tT}, scale = (1 - a_T) delta / (T^tail mu_star).
    """
    from scipy.special import gamma as gamma_fn
    delta = c**tail * gamma_fn(1.0 - tail)
    one_minus_a = lam * delta * T ** (-tail)
    if one_minus_a >= 1:
        raise ValueError("T too small for this scaling")
    mu_T = mu_star / delta * T ** (tail - 1.0)
    return 1.0 - one_minus_a, mu_T, one_minus_a * delta / (T**tail * mu_star)


def binned_rates(times, T, scale, n_bins):
    """Rescaled event rate on [0, 1]: (X^T((b+1)/n) - X^T(b/n)) n for X^T_t = scale N_{tT}, and the
    Poisson-noise variance of one bin, estimated by mean count * (scale n)^2."""
    cnt = np.bincount(np.minimum((np.asarray(times) / T * n_bins).astype(int), n_bins - 1),
                      minlength=n_bins)
    return cnt * scale * n_bins, np.mean(cnt) * (scale * n_bins) ** 2


def binned_variogram(rates, lags, noise_var=0.0):
    """Mean of (Y_{b+k} - Y_b)^2 over bins, for k in ``lags``, minus twice the noise variance."""
    Y = np.asarray(rates, dtype=float)
    return np.array([np.mean((Y[k:] - Y[:-k]) ** 2) for k in lags]) - 2.0 * noise_var


def binned_fbm_variogram(lags, H):
    """E[(Ybar_{b+k} - Ybar_b)^2] / h^{2H} for bin averages Ybar of a process with variogram |r|^{2H}.

    Equals ((k+1)^{2H+2} - 2 k^{2H+2} + (k-1)^{2H+2} - 2) / ((2H+1)(2H+2)) for k >= 1; for H = 1/2 it
    is k - 1/3. Averaging over bins flattens the variogram at small lags, so log-log slopes of
    binned data overstate small H.
    """
    k = np.asarray(lags, dtype=float)
    p = 2.0 * H + 2.0
    return ((k + 1) ** p - 2 * k**p + (k - 1) ** p - 2.0) / ((2 * H + 1) * (2 * H + 2))


def heavy_tail_mean_limit(t, tail, lam=1.0, n_terms=200):
    """E[X_t] = int_0^t F^{tail,lam}(s) ds for the limit of ``thm:hk-scaling-limit``(ii).

    E[Y_s] = F(s) = 1 - E_tail(-lam s^tail) (Mittag-Leffler function), so E[X_t] =
    sum_{k>=1} (-1)^{k+1} lam^k t^{tail k + 1} / Gamma(tail k + 2); the series is used for lam t^tail <= 5.
    """
    from scipy.special import gammaln
    if lam * t**tail > 5:
        raise ValueError("series used only for lam t^tail <= 5")
    k = np.arange(1, n_terms + 1)
    terms = np.exp(k * np.log(lam) + (tail * k + 1) * np.log(t) - gammaln(tail * k + 2))
    return float(np.sum(np.where(k % 2 == 1, terms, -terms)))


def heavy_tail_mean_prelimit(T, tail, lam=1.0, mu_star=1.0, c=1.0, h=0.25):
    """E[X^T_1] before the limit, for the scaling of :func:`heavy_tail_scaling`, by the renewal equation.

    m = mu_T + phi^T * m (``prop:hk-stationary``) is solved by product integration with m piecewise
    linear on the grid t_i = i h: the kernel a_T tail c^tail (c + u)^{-1-tail} is integrated exactly
    against the hat functions, which gives a second-order scheme; E N_T is the trapezoidal integral of m.
    """
    a, mu_T, scale = heavy_tail_scaling(T, tail, lam, mu_star, c)
    n = int(round(T / h))
    x = np.arange(n + 1) * h
    F = 1.0 - (1.0 + x / c) ** (-tail)                                        # int_0^x phi / a_T
    Gm = c * (tail * ((1.0 + x / c) ** (1.0 - tail) - 1.0) / (1.0 - tail) - F)  # int_0^x u phi(u) du / a_T
    dF, dG = np.diff(F), np.diff(Gm)
    k = np.arange(n)
    up = (dG - k * h * dF) / h            # int over [kh, (k+1)h] of phi(u) (u - kh) / h
    down = dF - up                        # int over [kh, (k+1)h] of phi(u) ((k+1)h - u) / h
    W = np.zeros(n + 1)                   # weight of m_{i-k} in m_i, for 1 <= k <= i - 1
    W[0] = a * down[0]
    W[1:n] = a * (up[: n - 1] + down[1:n])
    m = np.zeros(n + 1)
    m[0] = mu_T
    for i in range(1, n + 1):
        acc = mu_T + a * up[i - 1] * m[0]                     # half hat at s = 0, lag i
        if i > 1:
            acc += W[i - 1:0:-1] @ m[1:i]                    # lags i-1, ..., 1 for nodes 1, ..., i-1
        m[i] = acc / (1.0 - W[0])                             # the node s = t_i itself, lag 0
    return scale * h * (m.sum() - 0.5 * (m[0] + m[-1]))
