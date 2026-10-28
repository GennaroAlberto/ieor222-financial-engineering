"""Arbitrage-free option surfaces: static-arbitrage checks, SANOS and a DYSANOS-type generator (Chapter 17).

Conventions follow the notes. Prices are *normalized*: for a strike K and maturity T,

    x = K / F_T,        c(T, x) = e^{rT} C(x F_T, T) / F_T = E[(Z_T - x)^+],   Z_T = S_T / F_T,

so that the "stock" is the call with strike 0 (c(T, 0) = 1) and no discounting is left. In these
units a surface is free of static arbitrage when it is the call surface of a nonnegative martingale
Z with Z_0 = 1 (``def:afs-static-arbitrage``).

* ``bs_call(s, x, v)`` is the Black--Scholes call Call(s, x, v) = s N(d_+) - x N(d_-) of SANOS
  (Buehler, Horvath, Kratsios, Limmer and Saqur, arXiv:2601.11209, eq. (3)), with v a total variance.
* Grid checks: :func:`grid_arbitrage` checks normalized quotes on a common strike grid against the
  conditions of ``thm:afs-grid`` (butterflies, slope bounds, calendar spreads).
* SANOS: :func:`sanos_prices`, :func:`sanos_surface` (the surface of ``thm:afs-sanos``) and
  :func:`sanos_calibrate`, the linear program of ``prop:afs-lp`` solved with
  ``scipy.optimize.linprog`` (HiGHS).
* Discrete local volatility: :func:`dlv_matrix`, :func:`dlv_densities`, :func:`implied_dlv`
  (``prop:afs-dlv``; SANOS Theorems 3.4 and 3.7).
* Naive interpolation of implied volatility, for comparison (``prop:afs-kink``).
* A DYSANOS-type generator (Buehler, Horvath, Kratsios and Wiese, arXiv:2608.12587): a DLV-SANOS
  decoder with a low-dimensional parameter, an affine compression of the daily fitted parameters,
  and a Gaussian AR(1) for the compressed state (``def:afs-dysanos``).
* Unconditional drift test and minimum-entropy reweighting (DYSANOS Section 4, eqs. (31)-(32), (39)).

All functions are vectorized with numpy; random functions take a ``seed``.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.linalg import solve_banded
from scipy.optimize import least_squares, linprog, minimize
from scipy.special import ndtr

from .volatility import black_normalized, heston_call, implied_total_vol

SQRT_2PI = np.sqrt(2.0 * np.pi)


# ---------------------------------------------------------------- Black--Scholes in normalized units


def bs_call(s, x, v):
    """Call(s, x, v) = s N(d_+) - x N(d_-), d_pm = (log(s/x) pm v/2)/sqrt(v): E[(s Y - x)^+] for Y
    lognormal with E[Y] = 1 and Var(log Y) = v. At v = 0 it is (s - x)^+. Arrays broadcast."""
    s, x, v = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(x, dtype=float),
                                  np.asarray(v, dtype=float))
    pos = (s > 0) & (x > 0)
    ss = np.where(pos, s, 1.0)
    xx = np.where(pos, x, 1.0)
    val = ss * black_normalized(np.log(xx / ss), np.sqrt(np.maximum(v, 0.0)))
    return np.where(pos, val, np.maximum(s - x, 0.0))[()]


def bs_call_dxx(s, x, v):
    """Second strike derivative of :func:`bs_call`: N'(d_-) / (x sqrt(v)) for v > 0 (s, x > 0)."""
    s, x, v = (np.asarray(a, dtype=float) for a in (s, x, v))
    sv = np.sqrt(v)
    dm = (np.log(s / x) - 0.5 * v) / sv
    return np.exp(-0.5 * dm * dm) / (SQRT_2PI * x * sv)


def implied_total_variance(c, x):
    """Total implied variance w = sigma^2 T of normalized call prices c at normalized strikes x.

    Out-of-the-money options are used (the put c - 1 + x below the forward), as in
    :func:`ieor222.volatility.implied_total_vol`; nan outside the bounds of ``prop:iv-existence``.
    """
    c, x = np.broadcast_arrays(np.asarray(c, dtype=float), np.asarray(x, dtype=float))
    k = np.log(x)
    put = c - 1.0 + x
    s = np.where(x >= 1.0, implied_total_vol(c, k, "call"), implied_total_vol(put, k, "put"))
    return (s * s)[()]


def black_from_w(x, w):
    """Normalized call price b(log x, sqrt(w)) for a total implied variance w (``eq:iv-b``)."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    return black_normalized(np.log(x), np.sqrt(np.maximum(w, 0.0)))


# ---------------------------------------------------------------- static arbitrage on a grid


def slopes(x, c):
    """Slopes s^1, ..., s^n of the piecewise-linear interpolation of (0, 1), (x_1, c_1), ..., (x_n, c_n).

    ``c`` may have leading dimensions (one row per maturity); the strike 0 with price 1 (the stock)
    is prepended, as in ``thm:afs-grid``.
    """
    x = np.asarray(x, dtype=float)
    c = np.asarray(c, dtype=float)
    xx = np.concatenate([[0.0], x])
    cc = np.concatenate([np.ones(c.shape[:-1] + (1,)), c], axis=-1)
    return np.diff(cc, axis=-1) / np.diff(xx)


def grid_arbitrage(x, C, tol=0.0):
    """Static-arbitrage diagnostics for normalized call prices C (M x n) on a common strike grid x.

    The conditions of ``thm:afs-grid`` (with the stock as strike 0) are
      slope bounds   -1 <= s^1 and s^n <= 0,
      butterflies    s^1 <= s^2 <= ... <= s^n (nondecreasing slopes),
      calendars      C[j+1, i] >= C[j, i].
    Returns a dict with the most negative value of each (>= 0 when the condition holds), and the
    numbers of violations below ``-tol``.
    """
    C = np.atleast_2d(np.asarray(C, dtype=float))
    s = slopes(x, C)
    fly = np.diff(s, axis=-1)                        # discrete density masses at the inner strikes
    lower = s[:, 0] + 1.0
    upper = -s[:, -1]
    cal = np.diff(C, axis=0) if C.shape[0] > 1 else np.zeros((0, C.shape[1]))
    out = dict(butterfly=float(fly.min()) if fly.size else 0.0,
               slope_lower=float(lower.min()), slope_upper=float(upper.min()),
               calendar=float(cal.min()) if cal.size else 0.0)
    out.update(n_butterfly=int(np.sum(fly < -tol)), n_slope=int(np.sum(lower < -tol) + np.sum(upper < -tol)),
               n_calendar=int(np.sum(cal < -tol)))
    return out


def discrete_density(x, c):
    """Weights p^0, ..., p^n at the strikes 0, x_1, ..., x_n of the measure whose call function is the
    linear interpolation of (0, 1), (x_1, c_1), ..., (x_n, c_n), with c_n = 0 (``thm:afs-grid``; SANOS
    Remark 2.10): p^i = s^{i+1} - s^i with s^0 = -1 and s^{n+1} = 0. Leading dimensions of ``c`` are kept.
    """
    s = slopes(x, c)
    lead = s.shape[:-1]
    ext = np.concatenate([-np.ones(lead + (1,)), s, np.zeros(lead + (1,))], axis=-1)
    return np.diff(ext, axis=-1)


def linear_interpolation(x_nodes, c_nodes, x):
    """Linear interpolation of the call prices (0, 1), (x_i, c_i), and 0 beyond the last strike
    (SANOS Definition 2.11). This is the call function of :func:`discrete_density`."""
    xx = np.concatenate([[0.0], np.asarray(x_nodes, dtype=float)])
    cc = np.concatenate([[1.0], np.asarray(c_nodes, dtype=float)])
    return np.interp(x, xx, cc, right=0.0)


def call_function(nodes, weights, x):
    """Call prices sum_i weights_i (nodes_i - x)^+ of a discrete measure (SANOS Proposition 2.13)."""
    nodes, weights = np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.maximum(nodes[None, :] - x.reshape(-1, 1), 0.0) @ weights


def martingale_coupling(xa, pa, xb, pb):
    """A martingale coupling of two discrete laws, found by linear programming (or None).

    Looks for pi >= 0 with row sums pa, column sums pb and sum_b pi[a, b] x_b = x_a pa[a]: a two-period
    martingale with the given marginals. By Strassen's theorem it exists if and only if the laws have
    the same mean and increasing call functions (``lem:afs-strassen``).
    """
    xa, pa, xb, pb = (np.asarray(a, dtype=float) for a in (xa, pa, xb, pb))
    na, nb = xa.size, xb.size
    rows = []
    for a in range(na):                               # row sums
        r = np.zeros((na, nb)); r[a, :] = 1.0; rows.append(r.ravel())
    for b in range(nb):                               # column sums
        r = np.zeros((na, nb)); r[:, b] = 1.0; rows.append(r.ravel())
    for a in range(na):                               # martingale condition
        r = np.zeros((na, nb)); r[a, :] = xb - xa[a]; rows.append(r.ravel())
    A = np.array(rows)
    b = np.concatenate([pa, pb, np.zeros(na)])
    res = linprog(np.zeros(na * nb), A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    return res.x.reshape(na, nb) if res.status == 0 else None


# ---------------------------------------------------------------- naive interpolation of implied volatility


def linear_w_interpolation(k_nodes, w_nodes, k):
    """Total implied variance interpolated linearly in log-moneyness, flat beyond the nodes."""
    return np.interp(k, k_nodes, w_nodes)


def concave_kinks(k_nodes, w_nodes):
    """Indices of the inner nodes at which the piecewise-linear interpolation of w has a concave kink
    (its slope decreases); each one is a butterfly arbitrage (``prop:afs-kink``)."""
    s = np.diff(w_nodes) / np.diff(k_nodes)
    return np.flatnonzero(np.diff(s) < 0) + 1


def kink_slope_jump(k0, w0, dw):
    """Jump of the strike derivative of the normalized call price at x0 = e^{k0} when w' jumps by dw:
    N'(d_+) dw / (2 sqrt(w0) x0) (``prop:afs-kink``)."""
    sw = np.sqrt(w0)
    dp = -k0 / sw + 0.5 * sw
    return np.exp(-0.5 * dp * dp) / SQRT_2PI * dw / (2.0 * sw * np.exp(k0))


# ---------------------------------------------------------------- SANOS


def sanos_prices(q, K, x, v):
    """SANOS prices sum_i q^i Call(K^i, x, v) at strikes x for one expiry (SANOS eq. (17)).

    ``q`` has shape (..., N) on the grid K (N,), ``x`` shape (L,), ``v`` a scalar or shape (...,).
    Returns shape (..., L).
    """
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    A = bs_call(np.asarray(K, dtype=float)[None, :], np.asarray(x, dtype=float)[:, None],
                v[..., None, None] if v.ndim else v)
    return np.einsum("...ln,...n->...l", A, q)


def sanos_density(q, K, x, v):
    """Density of Z_T at x for the SANOS slice: sum_i q^i d^2/dx^2 Call(K^i, x, v) (v > 0)."""
    A = bs_call_dxx(np.asarray(K, dtype=float)[None, :], np.asarray(x, dtype=float)[:, None], v)
    return A @ np.asarray(q, dtype=float)


def interpolation_weights(T, T_model):
    """For a maturity T, the index j with T in (T_{j-1}, T_j] (T_0 = 0) and alpha = (T - T_{j-1})/(T_j - T_{j-1}).

    Beyond the last model expiry, j = M + 1 is returned with alpha = 0 (the last density is kept).
    """
    Tm = np.concatenate([[0.0], np.asarray(T_model, dtype=float)])
    if T > Tm[-1]:
        return len(Tm), 0.0
    j = int(np.searchsorted(Tm, T, side="left"))
    j = max(j, 1)
    return j, float((T - Tm[j - 1]) / (Tm[j] - Tm[j - 1]))


def smoothing_variance(T, T_model, v_model):
    """v(T): linear interpolation in T of v_0 = 0, v_1, ..., v_M at 0, T_1, ..., T_M, and linear growth
    v_M T / T_M beyond (nondecreasing whenever the v_j are)."""
    Tm = np.concatenate([[0.0], np.asarray(T_model, dtype=float)])
    vm = np.concatenate([[0.0], np.asarray(v_model, dtype=float)])
    T = np.asarray(T, dtype=float)
    return np.where(T <= Tm[-1], np.interp(T, Tm, vm), vm[-1] * T / Tm[-1])[()]


def sanos_surface(q, K, T_model, v_model, T, x):
    """Normalized call prices c(T, x) of the SANOS surface of ``thm:afs-sanos`` at one maturity T > 0.

    For T in (T_{j-1}, T_j]: c = sum_i [alpha q_j^i + (1 - alpha) q_{j-1}^i] Call(K^i, x, v(T)), with
    q_0 the unit mass at 1 (SANOS eq. (14)), and c = sum_i q_M^i Call(K^i, x, v(T)) beyond T_M (eq. (15)).
    ``q`` has shape (M, N).
    """
    q = np.asarray(q, dtype=float)
    K = np.asarray(K, dtype=float)
    j, a = interpolation_weights(T, T_model)
    v = smoothing_variance(T, T_model, v_model)
    nodes = np.concatenate([[1.0], K])
    M = q.shape[0]
    qn = np.zeros((M + 1, nodes.size))
    qn[0, 0] = 1.0
    qn[1:, 1:] = q
    if j > M:
        mix = qn[M]
    else:
        mix = a * qn[j] + (1.0 - a) * qn[j - 1]
    return sanos_prices(mix, nodes, x, v)


def strike_grid(T_model, z_min=-4.0, z_max=3.0, n_z=41, sigma_ref=0.2, extra=(0.5, 1.5)):
    """A common model strike grid, in the spirit of DYSANOS eq. (15): exp(z sigma_ref sqrt(T_1)) and
    exp(z sigma_ref sqrt(T_M)) for z on a uniform grid, merged and sorted, with the forward 1 added and
    boundary strikes extra[0] * min and extra[1] * max."""
    z = np.linspace(z_min, z_max, n_z)
    T_model = np.asarray(T_model, dtype=float)
    pts = np.sort(np.concatenate([np.exp(z * sigma_ref * np.sqrt(T_model[0])),
                                  np.exp(z * sigma_ref * np.sqrt(T_model[-1]))]))
    keep = [pts[0]]
    for p in pts[1:]:                                 # drop points within 0.2 % of the previous one
        if p / keep[-1] - 1.0 > 2e-3:
            keep.append(p)
    pts = np.array(keep)
    pts[np.argmin(np.abs(np.log(pts)))] = 1.0         # the forward is a grid strike
    return np.concatenate([[extra[0] * pts[0]], pts, [extra[1] * pts[-1]]])


def atm_total_variance(x, w):
    """Total implied variance at x = 1 by linear interpolation of w in log-moneyness (nan entries ignored)."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    ok = np.isfinite(w)
    return float(np.interp(0.0, np.log(x[ok]), w[ok]))


def sanos_calibrate(K, quotes, v, eps=1e-8, hard=False, weights=None):
    """SANOS calibration by linear programming (``prop:afs-lp``; SANOS Section 4.2 and Remark 4.1).

    Parameters
    ----------
    K : (N,) model strike grid (common to all expiries), with K[0] < 1 < K[-1].
    quotes : list over the model expiries j = 1..M of dicts with arrays ``x`` (normalized strikes),
        ``bid``, ``ask`` and ``mid`` (normalized call prices).
    v : (M,) smoothing variances v_j = eta V_j, nondecreasing.
    eps : weight of the distance to the mid (SANOS recommends 1e-8).
    hard : if True, impose bid <= c <= ask as constraints (the problem may be infeasible);
        otherwise penalize the distance to the bid--ask interval.
    weights : list of arrays w_l > 0; default 1 / (ask - bid) (SANOS eq. (34)).

    The decision variables are the densities q_j on K (q_j >= 0, sum q_j = 1, K' q_j = 1) with the
    calendar constraints sum_i q_j^i (K^i - K^l)^+ >= sum_i q_{j-1}^i (K^i - K^l)^+ for every grid
    strike K^l (the case omega = 0 of SANOS eq. (33), which gives convex order). Model prices are
    linear in q: c = A_j q_j with A_j[l, i] = Call(K^i, x_l, v_j). The objective
    sum_l w_l [(c_l - ask_l)^+ + (bid_l - c_l)^+ + eps |c_l - mid_l|] is linearized with slack variables.

    Returns a dict with ``q`` (M, N), fitted prices ``c`` (list), the optimal value and the LP status.
    """
    K = np.asarray(K, dtype=float)
    N = K.size
    M = len(quotes)
    v = np.asarray(v, dtype=float)
    A = [bs_call(K[None, :], np.asarray(qt["x"], dtype=float)[:, None], v[j]) for j, qt in enumerate(quotes)]
    L = [a.shape[0] for a in A]
    Ltot = int(np.sum(L))
    off = np.concatenate([[0], np.cumsum(L)])
    nq = M * N
    n_slack = 0 if hard else 3 * Ltot
    nvar = nq + (Ltot if hard else n_slack)          # with hard constraints only the |c - mid| slacks remain
    if weights is None:
        weights = [1.0 / (np.asarray(qt["ask"]) - np.asarray(qt["bid"])) for qt in quotes]
    wvec = np.concatenate([np.asarray(w, dtype=float) for w in weights])

    cost = np.zeros(nvar)
    if hard:
        e0 = nq
        cost[e0:e0 + Ltot] = eps * wvec if eps > 0 else 0.0
    else:
        u0, d0, e0 = nq, nq + Ltot, nq + 2 * Ltot
        cost[u0:u0 + Ltot] = wvec
        cost[d0:d0 + Ltot] = wvec
        cost[e0:e0 + Ltot] = eps * wvec

    # equality constraints: mass 1 and mean 1 for each expiry
    Aeq = sparse.lil_matrix((2 * M, nvar))
    beq = np.ones(2 * M)
    for j in range(M):
        Aeq[2 * j, j * N:(j + 1) * N] = 1.0
        Aeq[2 * j + 1, j * N:(j + 1) * N] = K
    rows, rhs = [], []
    # calendar constraints, at the inner grid strikes (at K[0] and K[-1] they hold with equality)
    U = np.maximum(K[None, :] - K[1:-1, None], 0.0)                  # (N-2, N)
    for j in range(1, M):
        blk = sparse.lil_matrix((N - 2, nvar))
        blk[:, (j - 1) * N:j * N] = U
        blk[:, j * N:(j + 1) * N] = -U
        rows.append(blk.tocsr())
        rhs.append(np.zeros(N - 2))
    # price constraints
    for j in range(M):
        qt = quotes[j]
        a = A[j]
        n = L[j]
        idx = np.arange(off[j], off[j + 1])
        def blk_with(sign, slack_start, slack_sign):
            b = sparse.lil_matrix((n, nvar))
            b[:, j * N:(j + 1) * N] = sign * a
            if slack_start is not None:
                b[np.arange(n), slack_start + idx] = slack_sign
            return b.tocsr()
        if hard:
            rows += [blk_with(1.0, None, 0), blk_with(-1.0, None, 0)]
            rhs += [np.asarray(qt["ask"], dtype=float), -np.asarray(qt["bid"], dtype=float)]
        else:
            rows += [blk_with(1.0, u0, -1.0), blk_with(-1.0, d0, -1.0)]
            rhs += [np.asarray(qt["ask"], dtype=float), -np.asarray(qt["bid"], dtype=float)]
        rows += [blk_with(1.0, e0, -1.0), blk_with(-1.0, e0, -1.0)]
        rhs += [np.asarray(qt["mid"], dtype=float), -np.asarray(qt["mid"], dtype=float)]
    Aub = sparse.vstack(rows).tocsr()
    bub = np.concatenate(rhs)
    res = linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq, bounds=(0, None), method="highs")
    out = dict(status=res.status, message=res.message, value=res.fun if res.status == 0 else np.nan)
    if res.status == 0:
        q = np.maximum(res.x[:nq].reshape(M, N), 0.0)
        out["q"] = q
        out["c"] = [A[j] @ q[j] for j in range(M)]
    return out


def within_bid_ask(c, quotes, tol=1e-6):
    """Fraction of prices inside their spreads, and the excess outside the spread in units of the half-spread
    (as reported in SANOS Section 5 and DYSANOS Table 1). A price counts as inside when its excess is at most
    ``tol`` half-spreads (the linear-programming solver satisfies constraints up to a small tolerance)."""
    excess = []
    for cj, qt in zip(c, quotes):
        b, a = np.asarray(qt["bid"]), np.asarray(qt["ask"])
        e = np.maximum(cj - a, 0.0) + np.maximum(b - cj, 0.0)
        excess.append(e / (0.5 * (a - b)))
    excess = np.concatenate(excess)
    return float(np.mean(excess <= tol)), excess


def surface_checks(price_fn, T_grid, x_grid):
    """Numerical static-arbitrage checks of a surface c(T, x) on a fine grid.

    ``price_fn(T, x)`` returns normalized call prices. Returns the most negative butterfly (second
    difference of c in x, divided by the squared spacing: a density), the most negative calendar
    difference c(T_{k+1}, x) - c(T_k, x), and the largest violation of the bounds (1 - x)^+ <= c <= 1.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    C = np.array([price_fn(T, x_grid) for T in T_grid])
    s = np.diff(C, axis=-1) / np.diff(x_grid)
    dens = np.diff(s, axis=-1) / (0.5 * (x_grid[2:] - x_grid[:-2]))
    cal = np.diff(C, axis=0)
    bnd = np.maximum(np.maximum(1.0 - x_grid, 0.0)[None, :] - C, C - 1.0)
    return dict(butterfly=float(dens.min()), calendar=float(cal.min()) if cal.size else 0.0,
                bounds=float(bnd.max()), slope=float(max(s.max(), -1.0 - s.min())))


# ---------------------------------------------------------------- discrete local volatility


def dlv_weights(K, sigma, dt):
    """w^{i+-} = (1/2) (Sigma^i K^i)^2 dt gamma^{i+-} of SANOS eq. (19), with
    gamma^{i+} = 1 / ((K^{i+1} - K^{i-1})/2 (K^{i+1} - K^i)) and gamma^{i-} likewise with K^i - K^{i-1};
    zero at the two boundary strikes. ``sigma`` has shape (..., N) (its boundary entries are ignored)."""
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    half = 0.5 * (K[2:] - K[:-2])
    gp = np.zeros_like(K); gm = np.zeros_like(K)
    gp[1:-1] = 1.0 / (half * (K[2:] - K[1:-1]))
    gm[1:-1] = 1.0 / (half * (K[1:-1] - K[:-2]))
    a = 0.5 * (sigma * K) ** 2 * dt
    wp, wm = a * gp, a * gm
    wp[..., 0] = wp[..., -1] = 0.0
    wm[..., 0] = wm[..., -1] = 0.0
    return wp, wm


def dlv_bands(K, sigma, dt):
    """The three diagonals (sub, main, super) of the tri-band matrix Q^{-1} of SANOS eq. (20).

    Row i reads -w^{(i-1)+} p^{i-1} + (1 + w^{i-} + w^{i+}) p^i - w^{(i+1)-} p^{i+1}.
    Returns arrays of shape (..., N); sub[..., 0] and sup[..., -1] are unused (zero).
    """
    wp, wm = dlv_weights(K, sigma, dt)
    main = 1.0 + wp + wm
    sub = np.zeros_like(main); sup = np.zeros_like(main)
    sub[..., 1:] = -wp[..., :-1]
    sup[..., :-1] = -wm[..., 1:]
    return sub, main, sup


def dlv_matrix(K, sigma, dt):
    """The dense matrix Q^{-1} (one Sigma vector); its inverse Q is a martingale transition operator."""
    sub, main, sup = dlv_bands(K, sigma, dt)
    return np.diag(main) + np.diag(sub[1:], -1) + np.diag(sup[:-1], 1)


def thomas(sub, main, sup, rhs):
    """Solve tridiagonal systems (vectorized over leading dimensions) by Gaussian elimination without
    pivoting (the Thomas algorithm), which is stable for column diagonally dominant matrices such as Q^{-1}."""
    n = main.shape[-1]
    cp = np.empty(np.broadcast_shapes(main.shape, rhs.shape))
    dp = np.empty_like(cp)
    cp[..., 0] = sup[..., 0] / main[..., 0]
    dp[..., 0] = rhs[..., 0] / main[..., 0]
    for i in range(1, n):
        den = main[..., i] - sub[..., i] * cp[..., i - 1]
        cp[..., i] = sup[..., i] / den if i < n - 1 else 0.0
        dp[..., i] = (rhs[..., i] - sub[..., i] * dp[..., i - 1]) / den
    out = np.empty_like(dp)
    out[..., -1] = dp[..., -1]
    for i in range(n - 2, -1, -1):
        out[..., i] = dp[..., i] - cp[..., i] * out[..., i + 1]
    return out


def dlv_densities(K, Sigma, T_model):
    """Martingale densities q_j = Q_j q_{j-1} on the grid K from discrete local volatilities
    (SANOS Theorem 3.4), starting from the unit mass at the grid strike 1.

    ``Sigma`` has shape (..., M, N). Returns shape (..., M, N).
    """
    K = np.asarray(K, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    i1 = int(np.flatnonzero(np.isclose(K, 1.0))[0])
    T = np.concatenate([[0.0], np.asarray(T_model, dtype=float)])
    q = np.zeros(Sigma.shape[:-2] + (K.size,))
    q[..., i1] = 1.0
    out = np.empty_like(Sigma)
    for j in range(Sigma.shape[-2]):
        sub, main, sup = dlv_bands(K, Sigma[..., j, :], T[j + 1] - T[j])
        q = thomas(sub, main, sup, q)
        out[..., j, :] = q
    return out


def implied_dlv(K, C, T_model, C0=None, T0=0.0):
    """Implied discrete local volatilities of call prices on the grid (SANOS eq. (23)).

    ``C`` (M, N) are normalized call prices at the grid strikes for the expiries ``T_model``; the slice
    before the first one is ``C0`` at time ``T0`` (default (1 - K)^+ at 0). Sigma_j^i =
    sqrt(2 Theta / (K_i^2 Gamma)) with Theta = (C_j^i - C_{j-1}^i)/dT and
    Gamma = (dC^i - dC^{i-1}) / ((K^{i+1} - K^{i-1})/2). Where Gamma = 0 = Theta the value is 0 (0/0 := 0); where
    Gamma = 0 < Theta no finite discrete local volatility exists and the value is nan (``prop:afs-dlv``);
    boundary entries are 0.
    """
    K = np.asarray(K, dtype=float)
    C = np.atleast_2d(np.asarray(C, dtype=float))
    T = np.concatenate([[T0], np.asarray(T_model, dtype=float)])
    C0 = np.maximum(1.0 - K, 0.0) if C0 is None else np.asarray(C0, dtype=float)
    Cp = np.vstack([C0, C])
    theta = np.diff(Cp, axis=0) / np.diff(T)[:, None]
    dC = np.diff(C, axis=1) / np.diff(K)
    gamma = np.zeros_like(C)
    gamma[:, 1:-1] = np.diff(dC, axis=1) / (0.5 * (K[2:] - K[:-2]))
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = np.where(gamma > 0, 2.0 * theta / (K[None, :] ** 2 * gamma),
                      np.where(theta == 0, 0.0, np.nan))       # 0/0 := 0; no finite value if Gamma = 0 < Theta
    s2[:, 0] = s2[:, -1] = 0.0
    return np.sqrt(np.where(np.isnan(s2), np.nan, np.maximum(s2, 0.0)))


def dupire_reference(T, k, heston=None, h=0.05, u_max=400.0):
    """Dupire local volatility sigma(T, F_T e^k) of the Heston surface, sqrt(2 c_T / (c_kk - c_k)) in forward
    variables (``thm:iv-dupire``), from the Fourier derivatives of :func:`ieor222.volatility.heston_surface` with a
    finer quadrature (step ``h``, cutoff ``u_max``) than its default: converged to about 1e-6 here."""
    from .volatility import heston_surface
    heston = HESTON if heston is None else heston
    srf = heston_surface(np.atleast_1d(k), [T], h=h, u_max=u_max, **heston)
    return np.sqrt(2.0 * srf["c_T"][0] / (srf["c_kk"][0] - srf["c_k"][0]))


def heston_implied_dlv(T, dT, h, heston=None, y_range=(-1.5, 1.0), u_max=400.0):
    """Discrete local volatilities implied, by SANOS eq. (23), by Heston prices at maturities T - dT and T on the
    grid exp(y), y uniform with spacing h on ``y_range`` (``prop:afs-dlv``). The prices use a Fourier cutoff
    ``u_max`` large enough for second differences. Returns (y, Sigma)."""
    heston = HESTON if heston is None else heston
    y = np.arange(y_range[0], y_range[1] + 1e-9, h)
    K = np.exp(y)
    C = np.array([heston_call(1.0, K, t, 0.0, u_max=u_max, **heston) for t in (T - dT, T)])
    return y, implied_dlv(K, C[1:], [T], C0=C[0], T0=T - dT)[0]


def dlv_dupire_gap(T, dT, h, k_range=(-0.4, 0.3), heston=None):
    """Largest |discrete local volatility - Dupire local volatility| over grid points with log x in ``k_range``."""
    y, sig = heston_implied_dlv(T, dT, h, heston)
    keep = (y >= k_range[0] - 1e-9) & (y <= k_range[1] + 1e-9)
    return float(np.nanmax(np.abs(sig[keep] - dupire_reference(T, y[keep], heston))))


# ---------------------------------------------------------------- synthetic market data


def quote_strikes(T, z=np.arange(-3.0, 2.01, 0.25), sigma_ref=0.2):
    """Normalized strikes x = exp(z sigma_ref sqrt(T)) of the synthetic quotes at maturity T."""
    return np.exp(np.asarray(z, dtype=float) * sigma_ref * np.sqrt(T))


def make_quotes(T, x, true, half_spread_vol=0.005, min_half_spread=1e-4, noise=0.8, rng=None, w=None):
    """Bid, ask and mid quotes around arbitrage-free normalized prices ``true`` at strikes x, maturity T.

    The half-spread is h = max(min_half_spread, half_spread_vol * vega), with the normalized vega
    N'(d_+) sqrt(T) at the true implied volatility (so about half_spread_vol in volatility near the money);
    the mid is true + noise * h * U with U uniform on [-1, 1], and bid, ask = mid -+ h, so that the true
    price lies inside every spread when noise < 1. ``w`` (the true total implied variances) is computed
    from ``true`` if not given.
    """
    rng = np.random.default_rng(rng)
    if w is None:
        w = implied_total_variance(true, x)
    sw = np.sqrt(w)
    dp = -np.log(x) / sw + 0.5 * sw
    vega = np.exp(-0.5 * dp * dp) / SQRT_2PI * np.sqrt(T)
    h = np.maximum(min_half_spread, half_spread_vol * vega)
    mid = true + noise * h * rng.uniform(-1.0, 1.0, size=np.shape(x))
    mid = np.maximum(mid, 0.5 * true)
    return dict(T=float(T), x=np.asarray(x, dtype=float), true=np.asarray(true, dtype=float), mid=mid,
                bid=np.maximum(mid - h, 0.0), ask=mid + h)


def heston_quotes(T_model, heston, seed=None, **kw):
    """Synthetic quotes (:func:`make_quotes`) from a Heston model (``prop:iv-heston``), one dict per expiry,
    at the strikes of :func:`quote_strikes`; keyword arguments go to :func:`make_quotes`."""
    rng = np.random.default_rng(seed)
    out = []
    for T in np.atleast_1d(T_model):
        x = quote_strikes(T)
        out.append(make_quotes(T, x, heston_call(1.0, x, T, 0.0, **heston), rng=rng, **kw))
    return out


def heston_price_table(T_model, heston, v0_grid):
    """Normalized Heston prices and their total implied variances at the strikes of :func:`quote_strikes`,
    for each initial variance in ``v0_grid``: two arrays (len(v0_grid), M, L). Used to produce many days of
    quotes quickly."""
    tab, tw = [], []
    for v0 in v0_grid:
        row = [heston_call(1.0, quote_strikes(T), T, 0.0, **dict(heston, v0=v0)) for T in T_model]
        tab.append(row)
        tw.append([implied_total_variance(c, quote_strikes(T)) for c, T in zip(row, T_model)])
    return np.array(tab), np.array(tw)


def heston_quotes_from_table(table, v0_grid, v0, T_model, rng=None, **kw):
    """Quotes for initial variance v0, with true prices and implied variances interpolated in sqrt(v0) from
    :func:`heston_price_table` by cubic splines (accurate to far below the half-spread on a grid of a few
    dozen points)."""
    from scipy.interpolate import CubicSpline
    rng = np.random.default_rng(rng)
    prices, tw = table
    true = CubicSpline(np.sqrt(v0_grid), prices, axis=0)(np.sqrt(v0))
    w = CubicSpline(np.sqrt(v0_grid), tw, axis=0)(np.sqrt(v0))
    return [make_quotes(T, quote_strikes(T), true[j], rng=rng, w=w[j], **kw) for j, T in enumerate(T_model)]


# ---------------------------------------------------------------- a DYSANOS-type generator


def dlv_basis(K, T_model, sigma_ref=0.2, z_range=(-3.0, 2.0), degree=4):
    """Legendre basis for log-DLVs: P_b(y), b = 0..degree, with y the image in [-1, 1] of
    z = log(K) / (sigma_ref sqrt(T_j)) clipped to ``z_range`` (the range of the quotes, so that the log-DLV
    is flat beyond it). Returns shape (M, N, degree + 1)."""
    from numpy.polynomial import legendre
    K = np.asarray(K, dtype=float)
    T_model = np.asarray(T_model, dtype=float)
    lo, hi = z_range
    z = np.clip(np.log(K)[None, :] / (sigma_ref * np.sqrt(T_model)[:, None]), lo, hi)
    y = (2.0 * z - lo - hi) / (hi - lo)
    return np.stack([legendre.legval(y, np.eye(degree + 1)[b]) for b in range(degree + 1)], axis=-1)


def decode(u, K, T_model, basis, eta=0.25, sigma_floor=0.01, sigma_cap=3.0, n_expiries=None):
    """The DLV-SANOS decoder of a parameter vector u (..., M (degree + 1) + M).

    The first M (degree + 1) entries are coefficients a_{j,b} of log(Sigma_j - floor) in ``basis``, so
    that Sigma_j^i = floor + min(cap, exp(sum_b a_{j,b} phi_b(z_j^i))) > 0 (compare DYSANOS eq. (16)); the last
    M entries g_j are log forward ATM variances, and the smoothing variances are
    v_j = eta sum_{l <= j} e^{g_l} (T_l - T_{l-1}), nondecreasing. Returns (q, v) with q (..., M, N)
    from :func:`dlv_densities` and v (..., M). Every u gives a surface free of static arbitrage
    (``def:afs-dysanos``).
    """
    u = np.asarray(u, dtype=float)
    M, N, nb = basis.shape
    a = u[..., :M * nb].reshape(u.shape[:-1] + (M, nb))
    g = u[..., M * nb:M * nb + M]
    logs = np.einsum("...jb,jnb->...jn", a, basis)
    Sigma = sigma_floor + np.exp(np.minimum(logs, np.log(sigma_cap)))
    q = dlv_densities(K, Sigma, T_model)
    dT = np.diff(np.concatenate([[0.0], np.asarray(T_model, dtype=float)]))
    v = eta * np.cumsum(np.exp(g) * dT, axis=-1)
    return q, v


def atm_variance_from_quotes(qt):
    """At-the-money total implied variance from the mid quotes: the mid price at x = 1, linearly
    interpolated in log-moneyness between the two quotes around the forward, inverted exactly by
    s = 2 N^{-1}((1 + c)/2) (``rem:iv-solver``)."""
    from scipy.special import ndtri
    x = np.asarray(qt["x"], dtype=float)
    c1 = float(np.interp(0.0, np.log(x), np.asarray(qt["mid"], dtype=float)))
    return float((2.0 * ndtri(0.5 * (1.0 + c1))) ** 2)


def forward_atm_log_variances(quotes, T_model, floor=1e-4):
    """g_j = log of the forward ATM variance (V_j - V_{j-1}) / (T_j - T_{j-1}) from mid quotes, floored."""
    V = np.array([atm_variance_from_quotes(qt) for qt in quotes])
    dT = np.diff(np.concatenate([[0.0], np.asarray(T_model, dtype=float)]))
    fwd = np.diff(np.concatenate([[0.0], V])) / dT
    return np.log(np.maximum(fwd, floor))


def fit_dlv_sanos(quotes, K, T_model, basis, eta=0.25, u0=None, sigma_floor=0.01, sigma_cap=3.0, penalty=0.0):
    """Fit the DLV-SANOS decoder to one day of quotes, expiry by expiry (a bootstrap).

    The smoothing variances come from the mid ATM forward variances (:func:`forward_atm_log_variances`);
    the coefficients of expiry j minimize sum_l ((c_l - mid_l) / (ask_l - bid_l))^2 over the quotes of
    expiry j, given the densities of the earlier expiries (``scipy.optimize.least_squares``), plus
    ``penalty`` times the squared distance of the coefficients to those of expiry j - 1 (to the level-only
    coefficients for j = 1): a Tikhonov penalty that keeps the forward DLVs smooth in maturity.
    Returns (u, prices) with u as in :func:`decode`.
    """
    K = np.asarray(K, dtype=float)
    M, N, nb = basis.shape
    g = forward_atm_log_variances(quotes, T_model)
    dT = np.diff(np.concatenate([[0.0], np.asarray(T_model, dtype=float)]))
    v = eta * np.cumsum(np.exp(g) * dT)
    if u0 is None:
        a0 = np.zeros((M, nb))
        a0[:, 0] = np.log(np.maximum(np.sqrt(np.exp(g)) - sigma_floor, 1e-3))
    else:
        a0 = np.asarray(u0[:M * nb], dtype=float).reshape(M, nb)
    i1 = int(np.flatnonzero(np.isclose(K, 1.0))[0])
    q = np.zeros(N); q[i1] = 1.0
    a_fit = np.empty((M, nb))
    prices = []
    for j in range(M):
        qt = quotes[j]
        A = bs_call(K[None, :], np.asarray(qt["x"])[:, None], v[j])
        wgt = 1.0 / (np.asarray(qt["ask"]) - np.asarray(qt["bid"]))
        mid = np.asarray(qt["mid"])

        def solve(coef, q_prev=q, j=j):
            lg = basis[j] @ coef
            e = np.exp(np.minimum(lg, np.log(sigma_cap)))
            sig = sigma_floor + e
            sub, main, sup = dlv_bands(K, sig, dT[j])
            ab = np.vstack([np.concatenate([[0.0], sup[:-1]]), main, np.concatenate([sub[1:], [0.0]])])
            return solve_banded((1, 1), ab, q_prev), ab, sig, e * (lg < np.log(sigma_cap))

        prior = a_fit[j - 1] if j > 0 else np.zeros(nb)
        if j == 0:
            prior[0] = np.nan
        pen = np.sqrt(penalty) * np.isfinite(prior)
        prior = np.nan_to_num(prior)

        def resid(coef):
            return np.concatenate([(A @ solve(coef)[0] - mid) * wgt, pen * (coef - prior)])

        def jac(coef, j=j):
            # implicit differentiation of M(Sigma) q = q_prev: dq = -M^{-1} (dM q)
            qn, ab, sig, e = solve(coef)
            wp, wm = dlv_weights(K, sig, dT[j])
            dsig = e[:, None] * basis[j]                       # dSigma^i / dcoef_b, shape (N, nb)
            g = (2.0 * qn / sig)[:, None] * dsig                # (N, nb): column i of dM/dSigma^i times q
            rhs = (wm + wp)[:, None] * g
            rhs[:-1] -= wm[1:, None] * g[1:]                    # -w^{(i+1)-} q^{i+1} terms in row i
            rhs[1:] -= wp[:-1, None] * g[:-1]                   # -w^{(i-1)+} q^{i-1} terms in row i
            dq = -solve_banded((1, 1), ab, rhs)
            return np.vstack([(A @ dq) * wgt[:, None], np.diag(pen)])

        sol = least_squares(resid, a0[j], jac=jac, method="lm", xtol=1e-10, ftol=1e-10)
        a_fit[j] = sol.x
        q = solve(sol.x)[0]
        prices.append(A @ q)
    return np.concatenate([a_fit.ravel(), g]), prices


def compress(U, d):
    """Affine compression u ~ b + B h of daily parameter vectors U (D x p), rank d, in the gauge where the
    states h_t have sample mean 0 and sample covariance I (as in DYSANOS eqs. (21) and (53)).

    Returns (b, B, H) with H (D x d) and b + H B' the rank-d least-squares approximation of U (an SVD).
    """
    U = np.asarray(U, dtype=float)
    b = U.mean(axis=0)
    X = U - b
    Uu, s, Vt = np.linalg.svd(X, full_matrices=False)
    D = U.shape[0]
    H = Uu[:, :d] * np.sqrt(D)                 # unit sample variance, uncorrelated
    B = Vt[:d].T * (s[:d] / np.sqrt(D))
    return b, B, H


def fit_ar1(H, lag=1):
    """Gaussian AR(1) h_{t+lag} = m + A (h_t - m) + eps by least squares (the initialization step of
    DYSANOS Section 3.2). Returns (m, A, Omega) with Omega the residual covariance, and the spectral
    radius of A."""
    H = np.asarray(H, dtype=float)
    X, Y = H[:-lag], H[lag:]
    Xc = np.hstack([np.ones((X.shape[0], 1)), X])
    coef, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
    c, A = coef[0], coef[1:].T
    m = np.linalg.solve(np.eye(A.shape[0]) - A, c)
    E = Y - Xc @ coef
    Omega = E.T @ E / E.shape[0]
    return m, A, Omega, float(np.max(np.abs(np.linalg.eigvals(A))))


def ar1_invariant(A, Omega, tol=1e-14, max_iter=100000):
    """Covariance P of the invariant law N(m, P) of the AR(1): the solution of P = A P A' + Omega
    (spectral radius of A below 1), by doubling iteration."""
    P = Omega.copy()
    Ak = A.copy()
    for _ in range(max_iter):
        P_new = P + Ak @ P @ Ak.T
        Ak = Ak @ Ak
        if np.max(np.abs(P_new - P)) <= tol * max(1.0, np.max(np.abs(P_new))):
            return P_new
        P = P_new
    return P


def simulate_ar1(m, A, Omega, h0, n_steps, n_paths, seed=None):
    """Paths h_0, ..., h_{n_steps} of h_{t+1} = m + A (h_t - m) + Omega^{1/2} z_t, z_t ~ N(0, I) i.i.d.
    Returns shape (n_paths, n_steps + 1, d)."""
    rng = np.random.default_rng(seed)
    d = len(m)
    Lc = np.linalg.cholesky(Omega + 1e-14 * np.eye(d))
    h = np.empty((n_paths, n_steps + 1, d))
    h[:, 0] = h0
    for t in range(n_steps):
        z = rng.standard_normal((n_paths, d))
        h[:, t + 1] = m + (h[:, t] - m) @ A.T + z @ Lc.T
    return h


def cir_path(v0, kappa, theta, xi, dt, n, seed=None, substeps=10):
    """A path of the square-root process dv = kappa (theta - v) dt + xi sqrt(v) dB at n + 1 dates spaced dt,
    by the full-truncation Euler scheme with ``substeps`` steps per date (``prop:iv-heston``)."""
    rng = np.random.default_rng(seed)
    h = dt / substeps
    out = np.empty(n + 1)
    v = out[0] = v0
    for t in range(n):
        for _ in range(substeps):
            vp = max(v, 0.0)
            v = v + kappa * (theta - vp) * h + xi * np.sqrt(vp * h) * rng.standard_normal()
        out[t + 1] = max(v, 0.0)
    return out


# ---------------------------------------------------------------- dynamic arbitrage diagnostics (DYSANOS Section 4)


def atm_vol(q, v, K, T, j):
    """At-the-money implied volatility of expiry j of decoded SANOS surfaces (q (..., M, N), v (..., M)),
    by the exact inversion s = 2 N^{-1}((1 + c)/2) at x = 1 (``rem:iv-solver``)."""
    from scipy.special import ndtri
    c = sanos_prices(q[..., j, :], K, np.array([1.0]), v[..., j])[..., 0]
    return 2.0 * ndtri(0.5 * (1.0 + c)) / np.sqrt(T[j])


def spot_loadings(H, m, A, Omega, log_returns, front_vol, dt):
    """Regression of the spot's Gaussian scores on the state innovations (a simplified DYSANOS eq. (28)).

    With standardized innovations z_t = L^{-1}(h_{t+1} - m - A (h_t - m)), Omega = L L', and scores
    s_t = (log(S_{t+1}/S_t) + front_vol_t^2 dt / 2) / (front_vol_t sqrt(dt)), returns (beta, sigma_perp)
    from the least-squares regression s_t = beta' z_t + residual (without intercept: the spot is driftless).
    """
    Lc = np.linalg.cholesky(Omega)
    E = H[1:] - m - (H[:-1] - m) @ A.T
    Z = np.linalg.solve(Lc, E.T).T
    s = (log_returns + 0.5 * front_vol**2 * dt) / (front_vol * np.sqrt(dt))
    beta, *_ = np.linalg.lstsq(Z, s, rcond=None)
    resid = s - Z @ beta
    return beta, float(np.sqrt(np.mean(resid**2)))


def generate_market(b, B, m, A, Omega, beta, sigma_perp, h0, K, T, basis, n_steps, n_paths, dt,
                    instruments, seed=None, antithetic=True):
    """Simulate the DYSANOS-type market: latent AR(1) states, decoded SANOS surfaces, a driftless spot, and
    the one-step returns of floating instruments.

    At each step t the state moves by h_{t+1} = m + A (h_t - m) + L z_t, and the spot by
    S_{t+1} = S_t exp(sigma_t sqrt(dt) e_t - sigma_t^2 dt / 2) with sigma_t the front ATM volatility of the
    surface at t and e_t = (beta' z_t + sigma_perp e'_t) / sqrt(beta' beta + sigma_perp^2) (DYSANOS eq. (29),
    with a lognormal in place of the front SANOS marginal). ``instruments`` is a list of (T, x): at each t one
    buys the call with maturity T and strike x S_t (normalized strike x) and sells it at t + 1. Returns a dict
    with R (n_paths, n_steps, 1 + len(instruments)) (the spot first), the features (1, h_t, log S_t) used by
    the conditional test, and the spot paths. ``h0`` is one state, or one state per path (for instance
    draws from the invariant law, :func:`ar1_invariant`).
    """
    rng = np.random.default_rng(seed)
    d = len(m)
    Lc = np.linalg.cholesky(Omega)
    half = n_paths // 2 if antithetic else n_paths
    jmax = 1 + max(int(np.searchsorted(T, Ti - 1e-12)) for Ti, _ in instruments)
    jmax = max(jmax, 1)
    Tm = np.asarray(T, dtype=float)[:jmax]
    norm = np.sqrt(beta @ beta + sigma_perp**2)
    h0 = np.asarray(h0, dtype=float)
    h = np.tile(h0, (n_paths, 1)) if h0.ndim == 1 else h0.copy()
    S = np.ones(n_paths)
    nI = len(instruments)
    R = np.empty((n_paths, n_steps, 1 + nI))
    feats = np.empty((n_paths, n_steps, d + 2))
    Kn = np.concatenate([[1.0], K])

    def surface(hh):
        return decode(b + hh @ B.T, K, T, basis, n_expiries=jmax)

    def prices_at(q, v, Ti, x):
        # c(Ti, x) for each path, x of shape (n_paths,)
        j, a = interpolation_weights(Ti, Tm)
        vt = a * v[:, j - 1] + (1.0 - a) * (v[:, j - 2] if j >= 2 else 0.0)
        qn = np.zeros((q.shape[0], Kn.size))
        if j == 1:
            qn[:, 0] = 1.0 - a
            qn[:, 1:] = a * q[:, 0]
        else:
            qn[:, 1:] = a * q[:, j - 1] + (1.0 - a) * q[:, j - 2]
        sv = np.sqrt(vt)[:, None]
        dp = (np.log(Kn[None, :] / x[:, None]) + 0.5 * sv * sv) / sv
        calls = Kn[None, :] * ndtr(dp) - x[:, None] * ndtr(dp - sv)
        return np.einsum("pn,pn->p", calls, qn)

    q, v = surface(h)
    for t in range(n_steps):
        z = rng.standard_normal((half, d))
        e2 = rng.standard_normal(half)
        if antithetic:
            z, e2 = np.vstack([z, -z]), np.concatenate([e2, -e2])
        sig = atm_vol(q, v, K, Tm, 0)
        feats[:, t] = np.hstack([np.ones((n_paths, 1)), h, np.log(S)[:, None]])
        p0 = np.stack([prices_at(q, v, Ti, np.full(n_paths, xi)) for Ti, xi in instruments], axis=1)
        eps = (z @ beta + sigma_perp * e2) / norm
        S_new = S * np.exp(sig * np.sqrt(dt) * eps - 0.5 * sig**2 * dt)
        h = m + (h - m) @ A.T + z @ Lc.T
        q, v = surface(h)
        ratio = S_new / S
        p1 = np.stack([prices_at(q, v, Ti - dt, xi / ratio) for Ti, xi in instruments], axis=1)
        R[:, t, 0] = ratio - 1.0
        R[:, t, 1:] = p1 * ratio[:, None] / p0 - 1.0
        S = S_new
    return dict(R=R, features=feats, S=S)


def _ratio_se(values, w, antithetic):
    """Self-normalized mean sum w v / sum w and its delta-method standard error, with the antithetic pairs
    (path i with path i + n/2, the layout of :func:`generate_market`) as independent units when ``antithetic``."""
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    wr = w.reshape((-1,) + (1,) * (values.ndim - 1))
    u, d = wr * values, w
    if antithetic:
        h = n // 2
        u = u[:h] + u[h:2 * h]
        d = d[:h] + d[h:2 * h]
    D = d.sum()
    m = u.sum(axis=0) / D
    se = np.sqrt(np.sum((u - m * d.reshape((-1,) + (1,) * (values.ndim - 1))) ** 2, axis=0)) / D
    return m, se


def drift_zscores(R, weights=None, antithetic=False):
    """Self-normalized weighted means of the returns and their z-scores (DYSANOS eq. (39)):
    m = sum_p w_p R_p / sum_p w_p and z = m / se, with the delta-method standard error of the ratio. With
    ``antithetic`` the standard error treats the antithetic pairs (path i, path i + n/2), not the paths, as
    independent: their returns are strongly negatively correlated."""
    R = np.asarray(R, dtype=float)
    w = np.ones(R.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    m, se = _ratio_se(R, w, antithetic)
    return m, m / se


def entropy_weights(R):
    """Minimum relative entropy reweighting with zero mean returns (DYSANOS eqs. (31)-(32)).

    Minimizes E[D log D] subject to D > 0, E[D] = 1 and E[D R] = 0 over the empirical measure of the paths;
    by convex duality D = exp(-R a*) / mean(exp(-R a*)) where a* minimizes log mean exp(-R a) (the dual
    solved in DYSANOS Section 4.2). ``R`` has shape (n_paths, ...); returns (D, a*).
    """
    X = np.asarray(R, dtype=float).reshape(R.shape[0], -1)
    n, p = X.shape

    def soft(a):
        y = -X @ a
        e = np.exp(y - y.max())
        return y, e / e.sum()

    def f(a):
        y, pi = soft(a)
        mx = y.max()
        return mx + np.log(np.sum(np.exp(y - mx)) / n), -(X.T @ pi)

    def hess(a):
        _, pi = soft(a)
        mu = X.T @ pi
        return (X * pi[:, None]).T @ X - np.outer(mu, mu)

    res = minimize(f, np.zeros(p), jac=True, hess=hess, method="trust-exact", options=dict(gtol=1e-12, maxiter=500))
    _, pi = soft(res.x)
    D = pi * n
    return D, res.x


def effective_sample_size(w):
    """(sum w)^2 / sum w^2."""
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.sum(w**2))


def fit_actions(R, F, weights=None):
    """Minimize the weighted empirical mean of exp(-a' R + |a|^2/2) over actions a = Theta phi linear in the
    features phi (DYSANOS eq. (41) for one date; convex in Theta). ``R`` is (n, k), ``F`` is (n, p), and the
    result Theta is (k, p). With the constant feature, a* = (I + Gamma)^{-1} mu for Gaussian returns
    (``ex:afs-gaussian-test``)."""
    R, F = np.asarray(R, dtype=float), np.asarray(F, dtype=float)
    n, k = R.shape
    p = F.shape[1]
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)

    def f(theta):
        Th = theta.reshape(k, p)
        act = F @ Th.T
        e = np.exp(-np.sum(act * R, axis=1) + 0.5 * np.sum(act * act, axis=1))
        return np.sum(w * e), (((w * e)[:, None] * (act - R)).T @ F).ravel()

    res = minimize(f, np.zeros(k * p), jac=True, method="L-BFGS-B", options=dict(maxiter=2000, gtol=1e-10))
    return res.x.reshape(k, p)


def conditional_test(R, features=None, weights=None, seed=0, antithetic=True):
    """Cross-fitted conditional no-arbitrage test of DYSANOS Theorem 4.3, with actions linear in features.

    For each date t the action is a_t = Theta_t phi_t (phi_t = ``features[:, t]``, or the constant 1), with Theta_t
    from :func:`fit_actions` on one half of the paths; the captured returns sum_t a_t' R_t are evaluated on the
    other half, the halves are swapped, and the results pooled. With ``antithetic`` (the layout of
    :func:`generate_market`, path i paired with path i + n/2) the split keeps pairs together, as in DYSANOS
    Section 4.3.2: otherwise the mirrored noise of the two halves biases the out-of-fold returns; the standard
    error is then computed on the pairs. Returns
    (T_act, two-sided p-value, mean captured return): by ``prop:afs-conditional-test``, the population captured
    return vanishes if and only if every conditional mean E_t[R_t] is zero, so a large |T_act| detects drift.
    """
    from scipy.special import ndtr as Phi
    R = np.asarray(R, dtype=float)
    n, nt, k = R.shape
    F = np.ones((n, nt, 1)) if features is None else np.asarray(features, dtype=float)
    p = F.shape[-1]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)
    if antithetic:
        half = n // 2
        perm = rng.permutation(half)
        folds = [np.concatenate([perm[: half // 2], perm[: half // 2] + half]),
                 np.concatenate([perm[half // 2:], perm[half // 2:] + half])]
    else:
        perm = rng.permutation(n)
        folds = [perm[: n // 2], perm[n // 2:]]
    # standardize the non-constant features on the whole sample (an affine change of the linear class)
    mu = F.reshape(-1, p).mean(axis=0)
    sd = F.reshape(-1, p).std(axis=0)
    const = sd < 1e-12
    Fs = np.where(const, F, (F - mu) / np.where(const, 1.0, sd))
    captured = np.zeros(n)
    for a_idx, b_idx in ((folds[0], folds[1]), (folds[1], folds[0])):
        for t in range(nt):
            Th = fit_actions(R[a_idx, t], Fs[a_idx, t], w[a_idx])
            captured[b_idx] += np.sum((Fs[b_idx, t] @ Th.T) * R[b_idx, t], axis=1)
    m, se = _ratio_se(captured, w, antithetic)
    Tstat = m / se
    return float(Tstat), float(2.0 * Phi(-abs(Tstat))), float(m)


# ---------------------------------------------------------------- the experiments of the chapter

#: the "market" of Chapters 15 and 17: Heston under Q (``fig:iv-heston``), r = q = 0 in normalized units
HESTON = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
#: model (and market) expiries: 1, 2, 3, 6, 9 and 12 months
T_MODEL = np.array([1, 2, 3, 6, 9, 12]) / 12.0
#: SANOS smoothing parameter eta (SANOS Section 3.1, recommended default 0.25)
ETA = 0.25


def sanos_example(seed=17, eta=ETA, eps=1e-8, heston=HESTON, T_model=T_MODEL, **quote_kw):
    """The single-day SANOS experiment: Heston quotes with spreads, the model grid, the LP fit.

    Returns a dict with ``quotes``, ``K``, the smoothing variances ``v`` = eta V_j (V_j the ATM total
    variance of the mids, made nondecreasing), the LP result ``fit`` and the fraction of fitted prices
    inside the bid--ask spreads.
    """
    quotes = heston_quotes(T_model, heston, seed=seed, **quote_kw)
    K = strike_grid(T_model)
    V = np.maximum.accumulate([atm_variance_from_quotes(qt) for qt in quotes])
    v = eta * V
    fit = sanos_calibrate(K, quotes, v, eps=eps)
    frac, excess = within_bid_ask(fit["c"], quotes) if fit["status"] == 0 else (np.nan, None)
    return dict(quotes=quotes, K=K, v=v, V=V, fit=fit, inside=frac, excess=excess, T=np.asarray(T_model))


def naive_vol(quotes, T_model, kind="linear"):
    """Naive implied-volatility surface from mid quotes: per expiry, total variance interpolated linearly in
    log-moneyness (``kind="linear"``) or implied volatility by a natural cubic spline (``kind="spline"``),
    flat beyond the quotes; between expiries, implied volatility linear in maturity at fixed log-moneyness
    (flat before the first and after the last expiry). Returns a function (T, x) -> normalized call prices."""
    from scipy.interpolate import CubicSpline
    T_model = np.asarray(T_model, dtype=float)
    nodes = []
    for qt in quotes:
        k = np.log(qt["x"])
        w = implied_total_variance(qt["mid"], qt["x"])
        nodes.append((k, w, np.sqrt(w / qt["T"])))

    def sigma_at(j, k):
        kq, wq, sq = nodes[j]
        if kind == "linear":
            return np.sqrt(np.interp(k, kq, wq) / T_model[j])
        cs = CubicSpline(kq, sq, bc_type="natural")
        return np.where(k < kq[0], sq[0], np.where(k > kq[-1], sq[-1], cs(np.clip(k, kq[0], kq[-1]))))

    def price(T, x):
        k = np.log(np.asarray(x, dtype=float))
        if T <= T_model[0]:
            s = sigma_at(0, k)
        elif T >= T_model[-1]:
            s = sigma_at(len(T_model) - 1, k)
        else:
            j = int(np.searchsorted(T_model, T))
            a = (T - T_model[j - 1]) / (T_model[j] - T_model[j - 1])
            s = (1.0 - a) * sigma_at(j - 1, k) + a * sigma_at(j, k)
        return black_from_w(x, s * s * T)

    return price


def synthetic_history(n_days=250, seed=4, heston=HESTON, T_model=T_MODEL, kappa_p=8.0, theta_p=0.045,
                      xi_p=0.4, rho_p=-0.7, degree=4, penalty=1.0, eta=ETA):
    """A synthetic history of option surfaces and its daily DLV-SANOS fits (for the DYSANOS-type generator).

    A path (v_t, S_t) of the Heston equations with parameters (kappa_p, theta_p, xi_p, rho_p) and zero drift
    is simulated at daily dates (five Euler steps per day, :func:`ieor222.volatility.heston_paths`); on day t
    the market quotes the Heston surface of ``heston`` with v_0 = v_t (:func:`heston_quotes_from_table`).
    Each day is fitted by :func:`fit_dlv_sanos` (warm-started from the previous day). Returns a dict with the
    parameters U (n_days x p), the paths, the fraction of fitted prices inside the spreads per day, and the
    grid and basis.
    """
    from .volatility import heston_paths
    dt = 1.0 / 252
    T_model = np.asarray(T_model, dtype=float)
    P = heston_paths(1.0, (n_days - 1) * dt, 0.0, theta_p, kappa_p, theta_p, xi_p, rho_p, 5 * (n_days - 1), 1,
                     seed=seed, keep=True)
    S_path, v_path = P["S_path"][0, ::5], P["v_path"][0, ::5]
    K = strike_grid(T_model)
    basis = dlv_basis(K, T_model, degree=degree)
    v_grid = np.linspace(np.sqrt(0.002), np.sqrt(0.2), 40) ** 2
    table = heston_price_table(T_model, heston, v_grid)
    U, inside, u = [], [], None
    for t in range(n_days):
        qts = heston_quotes_from_table(table, v_grid, max(v_path[t], v_grid[0]), T_model, rng=1000 + t)
        u, pr = fit_dlv_sanos(qts, K, T_model, basis, eta=eta, u0=u, penalty=penalty)
        U.append(u)
        inside.append(within_bid_ask(pr, qts)[0])
    return dict(U=np.array(U), S=S_path, v=v_path, inside=np.array(inside), K=K, basis=basis, T=T_model,
                table=table, v_grid=v_grid, eta=eta)


def fit_generator(hist, d=6):
    """The AR(1) baseline on a history: rank-d affine compression of the daily parameters, Gaussian AR(1)
    for the states (least squares), and the spot loadings. Returns a dict (u0, Lam, H, m, A, Omega, rho,
    beta, sigma_perp, P) with u = u0 + Lam h, and P the invariant covariance."""
    u0, Lam, H = compress(hist["U"], d)
    m, A, Omega, rho = fit_ar1(H)
    q, v = decode(hist["U"], hist["K"], hist["T"], hist["basis"], eta=hist["eta"])
    front = atm_vol(q, v, hist["K"], hist["T"], 0)
    beta, sp = spot_loadings(H, m, A, Omega, np.diff(np.log(hist["S"])), front[:-1], 1.0 / 252)
    return dict(u0=u0, Lam=Lam, H=H, m=m, A=A, Omega=Omega, rho=rho, beta=beta, sigma_perp=sp,
                P=ar1_invariant(A, Omega))


def compressed_inside(hist, gen, every=1):
    """Fraction of the history's quotes priced inside their spreads by the compressed states u0 + Lam h_t."""
    out = []
    for t in range(0, len(hist["U"]), every):
        qts = heston_quotes_from_table(hist["table"], hist["v_grid"], max(hist["v"][t], hist["v_grid"][0]),
                                       hist["T"], rng=1000 + t)
        q, v = decode(gen["u0"] + gen["Lam"] @ gen["H"][t], hist["K"], hist["T"], hist["basis"], eta=hist["eta"])
        out.append(within_bid_ask([sanos_prices(q[j], hist["K"], qt["x"], v[j]) for j, qt in enumerate(qts)], qts)[0])
    return float(np.mean(out))


#: floating instruments of the dynamic diagnostics: (maturity, normalized strike)
INSTRUMENTS = [(1 / 12, 0.95), (1 / 12, 1.0), (3 / 12, 0.9), (3 / 12, 1.0)]


def dynamic_diagnostics(hist, gen, n_paths=4000, n_steps=25, seed=11, instruments=INSTRUMENTS):
    """DYSANOS Section 4 on the generator: simulate n_steps daily returns of the spot and the floating
    ``instruments`` under the generating measure P (antithetic pairs, initial states from the invariant
    law), and compute (i) the unconditional z-scores (eq. (39)), (ii) the cross-fitted conditional test of
    Theorem 4.3 with constant and with linear actions, and (iii) the minimum-entropy measure Q_1 of
    eqs. (31)-(32) with its effective sample size and relative entropy."""
    rng = np.random.default_rng(seed)
    d = len(gen["m"])
    half = n_paths // 2
    h0 = rng.multivariate_normal(np.zeros(d), gen["P"], size=half)
    h0 = gen["m"] + np.vstack([h0, -h0])
    mk = generate_market(gen["u0"], gen["Lam"], gen["m"], gen["A"], gen["Omega"], gen["beta"], gen["sigma_perp"],
                         h0, hist["K"], hist["T"], hist["basis"], n_steps, n_paths, 1.0 / 252, instruments,
                         seed=seed + 1)
    R = mk["R"]
    _, z = drift_zscores(R, antithetic=True)
    D, _ = entropy_weights(R)
    _, z1 = drift_zscores(R, D, antithetic=True)
    from scipy.special import xlogy
    return dict(R=R, features=mk["features"], z=z, max_z=float(np.abs(z).max()),
                n_z_large=int(np.sum(np.abs(z) > 2.576)),
                test_const=conditional_test(R), test_linear=conditional_test(R, mk["features"]),
                test_single=[conditional_test(R[:, :, [i]]) for i in range(R.shape[2])],
                D=D, max_z_q1=float(np.abs(z1).max()), ess_q1=effective_sample_size(D) / n_paths,
                entropy_q1=float(np.mean(xlogy(D, D))))


def frozen_surface_drift(hist, gen, n_paths=20000, seed=3, instruments=INSTRUMENTS + [(1 / 12, 1.05), (3 / 12, 1.1)]):
    """Mean one-day returns (and z-scores) of the instruments when the latent state is frozen at its mean
    (no surface noise) and the spot moves lognormally at the front ATM volatility: the drift that the spot
    model alone creates against a fixed surface. The spot comes first, then the instruments."""
    d = len(gen["m"])
    mk = generate_market(gen["u0"], gen["Lam"], gen["m"], np.zeros((d, d)), 1e-16 * np.eye(d), np.zeros(d), 1.0,
                         gen["m"], hist["K"], hist["T"], hist["basis"], 1, n_paths, 1.0 / 252, instruments, seed=seed)
    mean, z = drift_zscores(mk["R"][:, :1, :], antithetic=True)
    return mean[0], z[0]
