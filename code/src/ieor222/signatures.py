"""Signatures of paths and signature-based models (Chapter 20).

Conventions follow the notes.

* A path in R^d is an array of shape ``(..., m + 1, d)``: ``m + 1`` points, joined by straight
  segments. Leading axes are batch axes (one path per entry).
* Words over the alphabet {0, ..., d - 1} index the coordinates of tensors. The word
  ``I = (i_1, ..., i_k)`` has flat index ``i_1 d^(k-1) + ... + i_k`` in level k, which is the layout
  of ``a ⊗ b`` produced by an outer product followed by a reshape.
* A truncated tensor series of depth N is a list ``[a_0, a_1, ..., a_N]``: ``a_0`` has the batch
  shape and ``a_k`` has shape ``batch + (d**k,)``.
* A linear functional ``ℓ`` on the tensor algebra (an element of T(R^d), paired with signatures by
  ``<e_I, S> = S^I``) is a dict ``{word: coefficient}``; the empty word ``()`` is the constant.
* For time-augmented paths ``(t, X_t)``, letter 0 is time and letters 1, ..., d - 1 are the
  coordinates of X.

The signature of a piecewise-linear path is computed with Chen's identity (``thm:sig-chen``),
one segment at a time: S <- S ⊗ exp(v), evaluated by a Horner scheme. Nothing here depends on
iisignature or esig.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------- words and the tensor algebra


def words(d: int, depth: int, min_len: int = 0):
    """All words over {0, ..., d-1} of length min_len, ..., depth, shortest first, then lexicographic."""
    out = []
    for k in range(min_len, depth + 1):
        out += list(itertools.product(range(d), repeat=k))
    return out


def word_index(word, d: int) -> int:
    """Flat index of a word inside its level (first letter most significant)."""
    i = 0
    for a in word:
        i = i * d + a
    return i


def _level(a, n):
    """Level n of a series, with a trailing axis even at level 0."""
    return a[n] if n > 0 else a[0][..., None]


def _outer(x, y):
    shape = np.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    return (x[..., :, None] * y[..., None, :]).reshape(shape + (-1,))


def zero_series(d: int, depth: int, batch=()):
    """The unit 1 = (1, 0, 0, ...) of the truncated tensor algebra, with the given batch shape."""
    return [np.ones(batch)] + [np.zeros(tuple(batch) + (d**k,)) for k in range(1, depth + 1)]


def tensor_mul(a, b):
    """Truncated product (a ⊗ b)_n = sum_k a_k ⊗ b_{n-k}, up to the smaller of the two depths."""
    depth = min(len(a), len(b)) - 1
    out = []
    for n in range(depth + 1):
        s = sum(_outer(_level(a, k), _level(b, n - k)) for k in range(n + 1))
        out.append(s[..., 0] if n == 0 else s)
    return out


def tensor_exp(v, depth: int):
    """exp(v) = sum_n v^{⊗n} / n! truncated at depth: the signature of one straight segment."""
    v = np.asarray(v, dtype=float)
    out = [np.ones(v.shape[:-1])]
    cur = np.ones(v.shape[:-1] + (1,))
    for n in range(1, depth + 1):
        cur = _outer(cur, v) / n
        out.append(cur)
    return out


def _mul_exp(S, v):
    """S ⊗ exp(v) by Horner's scheme: level n is (((S_0 v/n + S_1) v/(n-1) + S_2) ...) v/1 + S_n."""
    depth = len(S) - 1
    out = [S[0]]
    for n in range(1, depth + 1):
        t = _level(S, 0)
        for j in range(1, n + 1):
            t = _outer(t, v / (n - j + 1)) + _level(S, j)
        out.append(t)
    return out


def tensor_inverse(a):
    """Inverse of a series with a_0 = 1: sum_k (1 - a)^{⊗k}, truncated (finite at each level)."""
    depth = len(a) - 1
    x = [np.zeros_like(a[0])] + [-ak for ak in a[1:]]          # 1 - a, without constant term
    out = zero_series(a[1].shape[-1], depth, a[0].shape)
    power = zero_series(a[1].shape[-1], depth, a[0].shape)
    for _ in range(depth):
        power = tensor_mul(power, x)
        out = [o + p for o, p in zip(out, power)]
    return out


def flatten(S, start: int = 0):
    """Concatenate levels start, ..., N of a series into one feature vector per batch entry."""
    return np.concatenate([_level(S, k) for k in range(start, len(S))], axis=-1)


# ---------------------------------------------------------------- signatures of paths


def signature(path, depth: int, at=None):
    """Truncated signature S(X)_{0,t} of the piecewise-linear path through the given points.

    ``path`` has shape (..., m + 1, d). Returns the series at the last point, or, if ``at`` is a
    list of point indices, a list of series at those points (index 0 gives the unit).
    """
    path = np.asarray(path, dtype=float)
    inc = np.diff(path, axis=-2)
    batch, d = path.shape[:-2], path.shape[-1]
    S = zero_series(d, depth, batch)
    if at is None:
        for j in range(inc.shape[-2]):
            S = _mul_exp(S, inc[..., j, :])
        return S
    want = {int(i) for i in at}
    keep = {0: S} if 0 in want else {}
    for j in range(inc.shape[-2]):
        S = _mul_exp(S, inc[..., j, :])
        if j + 1 in want:
            keep[j + 1] = S
    return [keep[int(i)] for i in at]


def level2(path):
    """Level 2 of the signature of a piecewise-linear path in closed form, vectorized over the steps.

    S^{(2)}_{0,T} = sum_k (x_{t_k} - x_0) ⊗ v_k + (1/2) sum_k v_k ⊗ v_k with v_k = x_{t_{k+1}} - x_{t_k}
    (Chen's identity for the segments; ``prop:sig-brownian-level2``). Returns shape (..., d, d).
    """
    path = np.asarray(path, dtype=float)
    v = np.diff(path, axis=-2)
    x = path[..., :-1, :] - path[..., :1, :]
    return np.einsum("...ki,...kj->...ij", x + 0.5 * v, v)


def coefficient(S, word, d: int):
    """The coordinate <e_I, S> = S^I of a series."""
    return S[0] if len(word) == 0 else S[len(word)][..., word_index(word, d)]


def pair(ell: dict, S, d: int):
    """<ℓ, S> = sum_I ℓ_I S^I for a linear functional given as {word: coefficient}."""
    return sum(c * coefficient(S, w, d) for w, c in ell.items())


def time_augment(t, x):
    """The path (t, x_t): time becomes letter 0. ``x`` has shape (..., m + 1) or (..., m + 1, k)."""
    x = np.asarray(x, dtype=float)
    if x.shape[-1] == np.size(t):                    # (..., m + 1): one coordinate
        x = x[..., None]
    tt = np.broadcast_to(np.asarray(t, dtype=float)[:, None], x.shape[:-1] + (1,))
    return np.concatenate([tt, x], axis=-1)


def path_length(path):
    """Length sum_j |x_{j+1} - x_j| of a piecewise-linear path (its 1-variation)."""
    return np.linalg.norm(np.diff(np.asarray(path, dtype=float), axis=-2), axis=-1).sum(axis=-1)


def levy_area(path):
    """Lévy area A = (S^{12} - S^{21}) / 2 of a planar piecewise-linear path (letters 0 and 1)."""
    S = signature(path, 2)
    return 0.5 * (S[2][..., 1] - S[2][..., 2])


# ---------------------------------------------------------------- shuffles and functionals


@lru_cache(maxsize=None)
def _shuffle_words(u: tuple, v: tuple):
    if not u:
        return ((v, 1),)
    if not v:
        return ((u, 1),)
    out = {}
    for w, c in _shuffle_words(u[:-1], v):
        out[w + (u[-1],)] = out.get(w + (u[-1],), 0) + c
    for w, c in _shuffle_words(u, v[:-1]):
        out[w + (v[-1],)] = out.get(w + (v[-1],), 0) + c
    return tuple(out.items())


def shuffle(u, v) -> dict:
    """The shuffle product e_u ⧢ e_v as {word: multiplicity} (``prop:sig-shuffle``)."""
    return dict(_shuffle_words(tuple(u), tuple(v)))


def shuffle_functionals(ell: dict, m: dict) -> dict:
    """ℓ ⧢ m, so that <ℓ, S><m, S> = <ℓ ⧢ m, S> for every signature S."""
    out = {}
    for u, a in ell.items():
        for v, b in m.items():
            for w, c in shuffle(u, v).items():
                out[w] = out.get(w, 0.0) + a * b * c
    return out


def append_letter(ell: dict, letter: int) -> dict:
    """ℓ ⊗ e_letter: <ℓ ⊗ e_i, S(X)_{0,t}> = ∫_0^t <ℓ, S(X)_{0,u}> dX^i_u."""
    return {w + (letter,): c for w, c in ell.items()}


def ito_integral_functional(ell: dict, k: int, corr) -> dict:
    """The functional of ∫ <ℓ, X̂_{0,u}> dW^k_u (Itô) for time-augmented correlated Brownian motion.

    ``prop:sig-brownian``(iii): ∫ <e_I, X̂> dW^k = <e_I ⊗ e_k - (ρ_{i,k}/2) e_{I'} ⊗ e_0, X̂>, where
    I = I' i and ρ_{0,k} = 0. Letters 1, ..., d of the primary process have correlation matrix
    ``corr`` (letter j is row j - 1).
    """
    corr = np.atleast_2d(np.asarray(corr, dtype=float))
    out = {}
    for w, c in ell.items():
        out[w + (k,)] = out.get(w + (k,), 0.0) + c
        if w and w[-1] != 0:
            rho = corr[w[-1] - 1, k - 1]
            key = w[:-1] + (0,)
            out[key] = out.get(key, 0.0) - 0.5 * rho * c
    return out


def iterated_ito_integrals(n: int) -> list:
    """J_0 = 1, J_k = ∫ J_{k-1} dW for one Brownian motion, as functionals of (t, W) (letters 0, 1)."""
    J = [{(): 1.0}]
    for _ in range(n):
        J.append(ito_integral_functional(J[-1], 1, [[1.0]]))
    return J


# ---------------------------------------------------------------- Brownian signatures


def brownian_generator(corr):
    """A = e_0 + (1/2) sum_{i,j} ρ_ij e_i ⊗ e_j for time-augmented correlated Brownian motion."""
    corr = np.atleast_2d(np.asarray(corr, dtype=float))
    d = corr.shape[0] + 1
    A = {(0,): 1.0}
    for i in range(1, d):
        for j in range(1, d):
            if corr[i - 1, j - 1] != 0:
                A[(i, j)] = 0.5 * corr[i - 1, j - 1]
    return A


def functional_to_series(ell: dict, d: int, depth: int):
    """The element sum_I ℓ_I e_I of the truncated tensor algebra (words longer than depth dropped)."""
    S = [np.zeros(())] + [np.zeros(d**k) for k in range(1, depth + 1)]
    for w, c in ell.items():
        if len(w) <= depth:
            if len(w) == 0:
                S[0] = S[0] + c
            else:
                S[len(w)][word_index(w, d)] += c
    return S


def series_exp(a, depth: int | None = None):
    """exp(a) = sum_k a^{⊗k}/k! for a series with a_0 = 0 (finite sum at each level)."""
    depth = len(a) - 1 if depth is None else depth
    d = a[1].shape[-1]
    out = zero_series(d, depth, a[0].shape)
    term = zero_series(d, depth, a[0].shape)
    for k in range(1, depth + 1):
        term = [t / k for t in tensor_mul(term, a)]
        out = [o + t for o, t in zip(out, term)]
    return out


def brownian_expected_signature(t: float, corr, depth: int):
    """E[S(X̂)_{0,t}] = exp(t A) for X̂ = (t, W), W Brownian with correlation matrix ``corr``."""
    corr = np.atleast_2d(np.asarray(corr, dtype=float))
    d = corr.shape[0] + 1
    A = functional_to_series(brownian_generator(corr), d, depth)
    return series_exp([a * t for a in A], depth)


def expected_signature_random_walk(T: float, n_steps: int, corr, depth: int, nodes: int = 6):
    """E[S(X̂^Π)_{0,T}] for the piecewise-linear interpolation of X̂ = (t, W) on a uniform grid.

    By Chen's identity and the independence of the increments, it is (E[exp(h e_0 + ΔW)])^{⊗m}
    with h = T/m. Each level of exp(h e_0 + ΔW) is a polynomial in the Gaussian increment, of
    degree at most ``depth``, integrated exactly by a tensor Gauss--Hermite rule with ``nodes``
    points per dimension (exact up to degree 2 nodes - 1). Compare with
    :func:`brownian_expected_signature`: the difference is of order h.
    """
    corr = np.atleast_2d(np.asarray(corr, dtype=float))
    k = corr.shape[0]
    assert 2 * nodes - 1 >= depth
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    Z = np.array(list(itertools.product(x, repeat=k)))
    weights = np.prod(np.array(list(itertools.product(w, repeat=k))), axis=1)
    h = T / n_steps
    v = np.concatenate([np.full((len(Z), 1), h), Z @ np.linalg.cholesky(corr).T * np.sqrt(h)], axis=1)
    step = [np.tensordot(weights, e, axes=(0, 0)) for e in tensor_exp(v, depth)]
    out = zero_series(k + 1, depth)
    for _ in range(n_steps):
        out = tensor_mul(out, step)
    return out


def correlated_brownian(T: float, n_steps: int, corr, n_paths: int, seed=None, antithetic=False):
    """Brownian motions with correlation matrix ``corr`` on the grid t_j = jT/n_steps.

    Returns ``(t, W)`` with W of shape (n_paths, n_steps + 1, k). With ``antithetic=True`` the second
    half of the paths are the negatives of the first half (n_paths must be even).
    """
    corr = np.atleast_2d(np.asarray(corr, dtype=float))
    L = np.linalg.cholesky(corr)
    rng = np.random.default_rng(seed)
    m = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((m, n_steps, corr.shape[0])) @ L.T * np.sqrt(T / n_steps)
    if antithetic:
        z = np.concatenate([z, -z], axis=0)
    W = np.concatenate([np.zeros((z.shape[0], 1, z.shape[2])), np.cumsum(z, axis=1)], axis=1)
    return np.linspace(0.0, T, n_steps + 1), W


# ---------------------------------------------------------------- signature-based models


def linear_model_words(d_primary: int, n: int):
    """Words I with |I| <= n - 1 over {0, ..., d_primary}: the parameters ℓ_I of the martingale form."""
    return words(d_primary + 1, n - 1)


def linear_model_features(sigs, n: int, corr, k: int = 1):
    """Features F_I = <ẽ^k_I, X̂_{0,T}> of the martingale linear signature model.

    In ``def:sig-model``(i) written in martingale form, S_T = S_0 + sum_I ℓ_I ∫_0^T <e_I, X̂_u> dW^k_u
    = S_0 + sum_I ℓ_I F_I (``prop:sig-brownian``). ``sigs`` is a signature (depth >= n) of the
    time-augmented primary process; returns ``(words, F)`` with F of shape batch + (len(words),).
    """
    corr = np.atleast_2d(np.asarray(corr, dtype=float))
    d = corr.shape[0] + 1
    W = linear_model_words(d - 1, n)
    F = np.stack([pair(ito_integral_functional({w: 1.0}, k, corr), sigs, d) for w in W], axis=-1)
    return W, F


def gbm_initial_guess(sigma: float, S0: float, model_words, k: int = 1) -> np.ndarray:
    """Coefficients of the truncated Black--Scholes model in the martingale linear form.

    dS = σ S dW^k has integrand σ S_u = σ S_0 sum_j σ^j J_j(u) (``ex:sig-gbm``); the words of
    J_j are in letters 0 and k. Words of length > n - 1 are dropped.
    """
    n1 = max(len(w) for w in model_words)
    J = iterated_ito_integrals(n1)
    ell = {}
    for j, Jj in enumerate(J):
        for w, c in Jj.items():
            ww = tuple(k if a == 1 else a for a in w)
            ell[ww] = ell.get(ww, 0.0) + sigma * S0 * sigma**j * c
    return np.array([ell.get(w, 0.0) for w in model_words])


def call_prices(S_T, K, S0=None):
    """Monte Carlo call prices E[(S_T - K)^+], one per strike (r = 0).

    Plain sample means if ``S0`` is None. Otherwise strikes below S0 are priced through the
    out-of-the-money put and parity, mean((K - S_T)^+) + S0 - K: this is the call estimator with
    the control variate S_T - S0, whose expectation is 0 because the price is a martingale (the
    simplest sig-payoff), and it is much less noisy in the money.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    calls = np.maximum(S_T[:, None] - K[None, :], 0.0).mean(axis=0)
    if S0 is None:
        return calls
    return np.where(K < S0, calls - (S_T.mean() - S0), calls)


def _otm_payoffs(S_T, K, S0):
    """Payoffs whose sample means are :func:`call_prices` with ``S0`` (up to the constant S0 - K)."""
    return np.where(K[None, :] < S0, np.maximum(K[None, :] - S_T[:, None], 0.0),
                    np.maximum(S_T[:, None] - K[None, :], 0.0))


def calibrate_linear_model(S0, quotes, features, ell0):
    """Vega-weighted least squares fit of the linear signature model to call prices (r = 0).

    ``quotes`` maps each maturity T to ``(K, price, vega)`` arrays, ``features`` maps T to the
    feature matrix F (n_paths, p) of :func:`linear_model_features`; the model price of the call
    (T, K) is the Monte Carlo estimate of :func:`call_prices` (with S0) at S_T = S_0 + F ℓ over the
    stored paths, so the signature samples are computed once and only ℓ changes during the
    optimization. Minimizes
    sum ((model - market) / vega)^2 with ``scipy.optimize.least_squares`` and the exact gradient
    of the sample prices. Returns the optimizer result (``.x`` is ℓ).
    """
    from scipy.optimize import least_squares

    Ts = sorted(quotes)

    def resid(ell):
        out = []
        for T in Ts:
            K, price, vega = quotes[T]
            out.append((call_prices(S0 + features[T] @ ell, K, S0) - price) / vega)
        return np.concatenate(out)

    def jac(ell):
        out = []
        for T in Ts:
            K, _, vega = quotes[T]
            F = features[T]
            itm = ((S0 + F @ ell)[:, None] > K[None, :]).astype(float)
            grad = itm.T @ F / F.shape[0] - (K < S0)[:, None] * F.mean(axis=0)[None, :]
            out.append(grad / vega[:, None])
        return np.concatenate(out)

    return least_squares(resid, np.asarray(ell0, dtype=float), jac=jac, x_scale="jac", max_nfev=2000)


def expected_product_matrix(funcs, Esig, d: int) -> np.ndarray:
    """M_ij = <f_i ⧢ f_j, E S>: then E[<a, S><b, S>] = aᵀ M b for a = sum_i a_i f_i (shuffle + linearity)."""
    p = len(funcs)
    M = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            M[i, j] = M[j, i] = pair(shuffle_functionals(funcs[i], funcs[j]), Esig, d)
    return M


def vix2_quadratic_form(model_words, sig_T, corr, Delta: float) -> np.ndarray:
    """Q(T) with E[∫_T^{T+Δ} σ_u^2 du | F_T] = ℓᵀ Q(T) ℓ for σ_u = <ℓ, X̂_{0,u}> (``prop:sig-calibration``).

    Q_IJ = <(e_I ⧢ e_J) ⊗ e_0, X̂_{0,T} ⊗ (exp(Δ A) - 1)>, with A of :func:`brownian_generator`.
    ``sig_T`` is the signature of the time-augmented primary process at T, of depth at least
    2 max|I| + 1 (one path, or a batch: then Q has the batch shape in front).
    """
    corr = np.atleast_2d(np.asarray(corr, dtype=float))
    d = corr.shape[0] + 1
    depth = len(sig_T) - 1
    E = brownian_expected_signature(Delta, corr, depth)
    E[0] = E[0] - 1.0
    Y = tensor_mul(sig_T, [np.broadcast_to(e, sig_T[0].shape + np.shape(e)) for e in E])
    p = len(model_words)
    Q = np.zeros(sig_T[0].shape + (p, p))
    for a in range(p):
        for b in range(a, p):
            f = append_letter(shuffle(model_words[a], model_words[b]), 0)
            Q[..., a, b] = Q[..., b, a] = pair(f, Y, d)
    return Q


# ---------------------------------------------------------------- experiments of the notes


def regression_experiment(depths=(1, 2, 3, 4, 5, 6), n_paths=8000, n_steps=100, sigma=0.3, strike=1.0,
                          seed=3):
    """Linear regression of path-dependent payoffs on the truncated signature of (t, W_t), t <= 1.

    Payoffs: the Asian call (A - K)^+ with A the time average of S_t = exp(σ W_t - σ^2 t / 2)
    (trapezoidal rule on the grid) and the running maximum of W. Half the paths fit the
    coefficients by least squares, the other half measure the out-of-sample R^2. Returns
    ``{payoff: (r2 per depth, test targets, test predictions at the largest depth)}``.
    """
    t, W = correlated_brownian(1.0, n_steps, [[1.0]], n_paths, seed=seed)
    w = W[:, :, 0]
    S = signature(time_augment(t, W), max(depths))
    St = np.exp(sigma * w - 0.5 * sigma**2 * t)
    payoffs = {"asian": np.maximum(0.5 * (St[:, 1:] + St[:, :-1]).mean(axis=1) - strike, 0.0),
               "max": w.max(axis=1)}
    half = n_paths // 2
    out = {}
    for name, y in payoffs.items():
        r2, pred = [], None
        for depth in depths:
            Phi = flatten(S[: depth + 1])
            coef = np.linalg.lstsq(Phi[:half], y[:half], rcond=None)[0]
            pred = Phi[half:] @ coef
            r2.append(1.0 - np.var(y[half:] - pred) / np.var(y[half:]))
        out[name] = (np.array(r2), y[half:], pred)
    return out


def calibration_experiment(n=3, maturities=(1 / 12, 0.25, 0.5, 1.0), x=np.linspace(-2.0, 2.0, 9),
                           heston=None, rho12=-0.5, n_paths=50_000, n_check=100_000, steps_per_year=120,
                           seed=7):
    """Calibrate the martingale linear signature model to Heston call prices (r = q = 0, S_0 = 1).

    The primary process is (W^1, W^2) with correlation ``rho12``, the price is driven by W^1
    (``def:sig-model``, ``prop:sig-martingale``), and the strikes are K = exp(0.2 sqrt(T) x).
    Steps: simulate ``n_paths`` antithetic paths once and store the features at each maturity;
    fit ℓ by :func:`calibrate_linear_model`; then reprice with ``n_check`` independent paths,
    with the same estimator, to separate the fit from the sample.
    Returns a dict with the targets, the model implied volatilities on the calibration paths
    (``iv_in``) and on the independent paths (``iv_out``), and the Monte Carlo standard errors of
    both, in volatility units (price standard error divided by the vega), and the exact
    discretization bias E[S_T] - S_0 of the grid model (:func:`expected_signature_random_walk`).
    """
    from time import perf_counter

    from .volatility import black_vega_normalized, heston_call, implied_vol

    heston = heston or dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    maturities = list(maturities)
    corr = [[1.0, rho12], [rho12, 1.0]]
    T_max = max(maturities)
    steps = int(round(steps_per_year * T_max))
    idx = [int(round(T * steps_per_year)) for T in maturities]
    assert np.allclose(np.array(idx) / steps_per_year, maturities), "maturities must lie on the grid"

    def features(N, s):
        t, W = correlated_brownian(T_max, steps, corr, N, seed=s, antithetic=True)
        sigs = signature(time_augment(t, W), n, at=idx)
        out = {}
        for T, S in zip(maturities, sigs):
            mw, out[T] = linear_model_features(S, n, corr)
        return mw, out

    quotes, iv_market = {}, {}
    for T in maturities:
        K = np.exp(0.2 * np.sqrt(T) * np.asarray(x))
        p = heston_call(1.0, K, T, 0.0, **heston)
        iv_market[T] = implied_vol(p, 1.0, K, T)
        quotes[T] = (K, p, black_vega_normalized(np.log(K), iv_market[T] * np.sqrt(T)) * np.sqrt(T))
    t0 = perf_counter()
    mw, F = features(n_paths, seed)
    t1 = perf_counter()
    res = calibrate_linear_model(1.0, quotes, F, gbm_initial_guess(0.2, 1.0, mw))
    t2 = perf_counter()
    _, F2 = features(n_check, seed + 1)
    iv_in, iv_out, iv_se, iv_se_fit, neg, bias = {}, {}, {}, {}, {}, {}
    for j, T in enumerate(maturities):
        K, _, vega = quotes[T]
        for S_T, iv, ivse in ((1.0 + F[T] @ res.x, iv_in, iv_se_fit), (1.0 + F2[T] @ res.x, iv_out, iv_se)):
            iv[T] = implied_vol(call_prices(S_T, K, 1.0), 1.0, K, T)
            pay = _otm_payoffs(S_T, K, 1.0)
            half = pay.shape[0] // 2                   # antithetic pairs are independent of each other
            ivse[T] = (0.5 * (pay[:half] + pay[half:])).std(axis=0, ddof=1) / np.sqrt(half) / vega
        neg[T] = float(np.mean(1.0 + F2[T] @ res.x <= 0.0))
        Epl = expected_signature_random_walk(T, idx[j], corr, n)
        bias[T] = sum(c * pair(ito_integral_functional({w: 1.0}, 1, corr), Epl, len(corr) + 1)
                      for c, w in zip(res.x, mw))
    return dict(words=mw, ell=res.x, quotes=quotes, iv_market=iv_market, iv_in=iv_in, iv_out=iv_out,
                iv_se=iv_se, iv_se_fit=iv_se_fit, prob_negative=neg, mean_bias=bias, x=np.asarray(x),
                times=(t1 - t0, t2 - t1), nfev=res.nfev)
