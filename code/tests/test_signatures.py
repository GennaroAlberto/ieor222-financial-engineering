"""Checks of the Chapter 20 code against the results proved in the notes."""
from math import comb, factorial

import numpy as np
import pytest

from ieor222 import signatures as sg


def random_path(m, d, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([np.zeros((1, d)), rng.normal(size=(m, d)).cumsum(axis=0)])


def max_diff(a, b):
    return max(np.max(np.abs(np.asarray(x) - np.asarray(y))) for x, y in zip(a, b))


def test_segment_signature_is_tensor_exponential():
    # prop:sig-segment: one straight segment with increment v has signature exp(v).
    v = np.array([0.3, -1.2, 0.7])
    S = sg.signature(np.stack([np.ones(3), np.ones(3) + v]), 5)
    assert max_diff(S, sg.tensor_exp(v, 5)) < 1e-14
    # Level k, word (i_1, ..., i_k): v_{i_1} ... v_{i_k} / k!
    assert sg.coefficient(S, (0, 2, 1, 1), 3) == pytest.approx(v[0] * v[2] * v[1] ** 2 / 24)


def test_chen_identity():
    # thm:sig-chen: S(X)_{0,T} = S(X)_{0,u} ⊗ S(X)_{u,T}, for any split point u.
    p = random_path(9, 3, seed=1)
    S = sg.signature(p, 5)
    for j in (1, 4, 8):
        assert max_diff(S, sg.tensor_mul(sg.signature(p[: j + 1], 5), sg.signature(p[j:], 5))) < 1e-12
    # cor:sig-piecewise-linear: the product of the exponentials of the increments.
    prod = sg.zero_series(3, 5)
    for v in np.diff(p, axis=0):
        prod = sg.tensor_mul(prod, sg.tensor_exp(v, 5))
    assert max_diff(S, prod) < 1e-12


def test_shuffle_identity_and_counts():
    # prop:sig-shuffle: <e_u, S><e_v, S> = <e_u ⧢ e_v, S>; the shuffle has C(|u|+|v|, |u|) terms.
    p = random_path(6, 3, seed=2)
    S = sg.signature(p, 5)
    for u in sg.words(3, 2, 1):
        for v in sg.words(3, 3, 1):
            lhs = sg.coefficient(S, u, 3) * sg.coefficient(S, v, 3)
            assert sg.pair(sg.shuffle(u, v), S, 3) == pytest.approx(lhs, rel=1e-12, abs=1e-12)
            assert sum(sg.shuffle(u, v).values()) == comb(len(u) + len(v), len(u))
    assert sg.shuffle((1,), (2,)) == {(1, 2): 1, (2, 1): 1}
    # e_i ⧢ e_i ⧢ e_i = 3! e_iii, hence S^{iii} = (X^i_T - X^i_0)^3 / 3!.
    assert sg.shuffle_functionals(sg.shuffle((1,), (1,)), {(1,): 1.0}) == {(1, 1, 1): 6.0}
    assert sg.coefficient(S, (1, 1, 1), 3) == pytest.approx((p[-1, 1] - p[0, 1]) ** 3 / 6)


def test_invariances():
    # prop:sig-invariance: translation, reparametrisation (subdivided segments, repeated points).
    p = random_path(5, 2, seed=3)
    S = sg.signature(p, 6)
    assert max_diff(S, sg.signature(p + np.array([5.0, -2.0]), 6)) < 1e-12
    fine = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        for s in (0.1, 0.5, 0.5, 1.0):                     # a repeated point is a pause
            fine.append(a + s * (b - a))
    assert max_diff(S, sg.signature(np.array(fine), 6)) < 1e-12


def test_reversal_inverse_and_tree_like_path():
    # The reversed path has the inverse signature, so X followed by its reversal (a tree-like path,
    # thm:sig-uniqueness) has signature 1 = (1, 0, 0, ...).
    p = random_path(7, 3, seed=4)
    S = sg.signature(p, 5)
    assert max_diff(sg.signature(p[::-1], 5), sg.tensor_inverse(S)) < 1e-11
    loop = np.concatenate([p, p[::-1][1:]])
    assert max_diff(sg.signature(loop, 5), sg.zero_series(3, 5)) < 1e-10


def test_factorial_decay():
    # lem:sig-decay: |X^n| <= L^n / n! for the Euclidean norm on each level.
    for seed in range(5):
        p = random_path(12, 3, seed=seed)
        L = sg.path_length(p)
        S = sg.signature(p, 6)
        for n in range(1, 7):
            assert np.linalg.norm(S[n]) <= L**n / factorial(n) * (1 + 1e-12)


def test_levy_area_is_signed_area():
    # Lévy area of a closed polygon = the signed area it encloses (shoelace formula).
    theta = np.linspace(0.0, 2 * np.pi, 401)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    x, y = circle[:-1].T
    xn, yn = circle[1:].T
    shoelace = 0.5 * np.sum(x * yn - xn * y)
    assert sg.levy_area(circle) == pytest.approx(shoelace, rel=1e-12)
    assert sg.levy_area(circle) == pytest.approx(np.pi, rel=1e-3)
    assert sg.levy_area(circle[::-1]) == pytest.approx(-shoelace, rel=1e-12)


def test_time_augmented_words_are_moments():
    # prop:sig-time-injective: <e_{1 0^k}, S(X̂)_{0,T}> = ∫_0^T (X_u - X_0) (T-u)^{k-1}/(k-1)! du.
    rng = np.random.default_rng(5)
    t = np.linspace(0.0, 1.0, 2001)
    x = np.cumsum(np.concatenate([[0.3], rng.normal(size=2000) * 0.03]))
    S = sg.signature(sg.time_augment(t, x), 5)
    for k in range(1, 5):
        integrand = (x - x[0]) * (1.0 - t) ** (k - 1) / factorial(k - 1)
        quad = np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(t))
        assert sg.coefficient(S, (1,) + (0,) * k, 2) == pytest.approx(quad, abs=2e-6)


def test_brownian_level_two_is_ito_plus_half_covariation():
    # prop:sig-brownian-level2: along every partition, S^{ij}(Π) = Itô sum + C(W^i, W^j; Π)/2,
    # and in L^2 the error against ∫W^i dW^j + ρ_ij T/2 is controlled by the mesh.
    rho, T = 0.6, 1.0
    corr = [[1.0, rho], [rho, 1.0]]
    t, W = sg.correlated_brownian(T, 2**12, corr, 400, seed=6)
    for n in (2**4, 2**12):
        Wn = W[:, :: 2**12 // n]
        S = sg.signature(Wn, 2)
        dW = np.diff(Wn, axis=1)
        ito = np.sum(Wn[:, :-1, 0] * dW[:, :, 1], axis=1)
        cov = np.sum(dW[:, :, 0] * dW[:, :, 1], axis=1)
        assert np.allclose(S[2][:, 1], ito + 0.5 * cov, atol=1e-12)
        assert np.allclose(S[2][:, 0], 0.5 * Wn[:, -1, 0] ** 2, atol=1e-12)
    # On the fine grid, cov is close to rho T: Var(C) = (1 + rho^2) T^2 / n.
    assert np.sqrt(np.mean((cov - rho * T) ** 2)) < 3 * np.sqrt((1 + rho**2) / 2**12)


def test_ito_stratonovich_conversion():
    # prop:sig-brownian(iii): left-point Itô sums of <e_I, X̂> against W^1 converge to <ẽ_I, X̂_T>.
    corr = [[1.0, -0.5], [-0.5, 1.0]]
    t, W = sg.correlated_brownian(1.0, 2**9, corr, 300, seed=7)
    errs = []
    for n in (2**5, 2**9):
        X = sg.time_augment(t[:: 2**9 // n], W[:, :: 2**9 // n])
        stream = sg.signature(X, 3, at=range(n + 1))
        dW1 = np.diff(X[:, :, 1], axis=1)
        e = []
        for w in [(1,), (2,), (1, 1), (2, 1), (0, 2)]:
            ito = sum(sg.coefficient(stream[j], w, 3) * dW1[:, j] for j in range(n))
            f = sg.ito_integral_functional({w: 1.0}, 1, corr)
            e.append(np.sqrt(np.mean((ito - sg.pair(f, stream[-1], 3)) ** 2)))
        errs.append(np.array(e))
    assert np.all(errs[1] < 0.45 * errs[0])          # about 1/sqrt(16) = 0.25 expected
    # For I = (1) the difference is (V_2(W^1; Π) - T)/2 exactly, with L^2 norm sqrt(1/(2n)).
    assert errs[1][0] == pytest.approx(np.sqrt(1 / 2**10), rel=0.25)
    assert np.all(errs[1] < 0.05)


def test_expected_signature_of_brownian_motion():
    # prop:sig-brownian(iv): E[S(X̂)_{0,t}] = exp(t (e_0 + (1/2) sum ρ_ij e_i ⊗ e_j)).
    corr = [[1.0, 0.5], [0.5, 1.0]]
    E = sg.brownian_expected_signature(0.7, corr, 4)
    assert sg.coefficient(E, (1, 1), 3) == pytest.approx(0.35)              # E[(W^1)^2/2]
    assert sg.coefficient(E, (1, 2), 3) == pytest.approx(0.5 * 0.5 * 0.7)    # E[∫W^1∘dW^2]
    assert sg.coefficient(E, (1, 1, 1, 1), 3) == pytest.approx(0.7**2 / 8)   # E[W^4]/24
    assert sg.coefficient(E, (0, 1, 1), 3) == pytest.approx(0.7**2 / 4)
    t, W = sg.correlated_brownian(0.7, 100, corr, 8000, seed=8, antithetic=True)
    S = sg.signature(sg.time_augment(t, W), 3)
    for k in (2, 3):
        se = S[k].std(axis=0) / np.sqrt(S[k].shape[0])
        assert np.all(np.abs(S[k].mean(axis=0) - E[k]) < 4.5 * se + 5e-3)


def test_expected_signature_of_the_discretized_path():
    # Chen + independent increments: E[S(X̂^Π)] = (E exp(h e_0 + ΔW))^{⊗m}, within O(h) of exp(T A).
    corr = [[1.0, -0.5], [-0.5, 1.0]]
    E = sg.brownian_expected_signature(1.0, corr, 3)
    gaps = []
    for m in (30, 60):
        Em = sg.expected_signature_random_walk(1.0, m, corr, 3)
        gaps.append(max(np.max(np.abs(a - b)) for a, b in zip(Em, E)))
        assert sg.coefficient(Em, (1, 1), 3) == pytest.approx(0.5)            # (W_T^1)^2/2 exactly
    assert gaps[1] == pytest.approx(gaps[0] / 2, rel=1e-6)                    # first order in h
    t, W = sg.correlated_brownian(1.0, 30, corr, 20000, seed=16, antithetic=True)
    S = sg.signature(sg.time_augment(t, W), 3)
    Em = sg.expected_signature_random_walk(1.0, 30, corr, 3)
    se = S[3].std(axis=0) / np.sqrt(S[3].shape[0])
    assert np.all(np.abs(S[3].mean(axis=0) - Em[3]) < 4.5 * se + 1e-12)


def test_linear_model_mean_and_second_moment():
    # prop:sig-martingale and prop:sig-calibration(i): E[S_T] = S_0 and E[S_T^2] = aᵀ M a.
    corr = [[1.0]]
    mw = sg.linear_model_words(1, 3)
    ell = np.array([0.2, 0.05, 0.1, 0.02, -0.03, 0.04, 0.05])
    funcs = [{(): 1.0}] + [sg.ito_integral_functional({w: 1.0}, 1, corr) for w in mw]
    E = sg.brownian_expected_signature(1.0, corr, 6)
    a = np.concatenate([[1.0], ell])
    second = a @ sg.expected_product_matrix(funcs, E, 2) @ a
    # The expectation of each feature is 0 (a stochastic integral), hence E[S_T] = S_0.
    assert all(abs(sg.pair(f, E, 2)) < 1e-14 for f in funcs[1:])
    t, W = sg.correlated_brownian(1.0, 200, corr, 20000, seed=9, antithetic=True)
    _, F = sg.linear_model_features(sg.signature(sg.time_augment(t, W), 3), 3, corr)
    ST = 1.0 + F @ ell
    assert ST.mean() == pytest.approx(1.0, abs=3 * ST.std() / np.sqrt(ST.size) + 2e-3)
    assert (ST**2).mean() == pytest.approx(second, abs=4 * (ST**2).std() / np.sqrt(ST.size) + 2e-3)


def test_vix2_quadratic_form_against_nested_monte_carlo():
    # prop:sig-calibration(iii): E[∫_T^{T+Δ} σ^2 du | F_T] = ℓᵀ Q(T) ℓ.
    corr = [[1.0]]
    mw = sg.words(2, 2)
    ell = np.array([0.2, 0.05, 0.1, 0.02, -0.03, 0.04, 0.05])
    T, Delta = 0.5, 1 / 12
    t, W = sg.correlated_brownian(T, 100, corr, 1, seed=11)
    X = sg.time_augment(t, W)[0]
    Q = sg.vix2_quadratic_form(mw, sg.signature(X, 5), corr, Delta)
    assert np.all(np.linalg.eigvalsh(Q) > -1e-12)                      # Q is positive semidefinite
    m, N = 50, 6000
    t2, W2 = sg.correlated_brownian(Delta, m, corr, N, seed=12, antithetic=True)
    stream = sg.signature(sg.time_augment(t2 + T, W2), 2, at=range(m + 1))
    S_T2 = [np.broadcast_to(a, (N,) + np.shape(a)) for a in sg.signature(X, 2)]
    vals = np.array([sum(c * sg.coefficient(sg.tensor_mul(S_T2, s), w, 2) for w, c in zip(mw, ell)) ** 2
                     for s in stream])
    integral = np.sum(0.5 * (vals[1:] + vals[:-1]), axis=0) * Delta / m
    assert integral.mean() == pytest.approx(ell @ Q @ ell, abs=4 * integral.std() / np.sqrt(N) + 1e-5)


def test_log_price_formula_of_the_volatility_model():
    # prop:sig-calibration(ii): log(S_T/S_0) = sum_I ℓ_I <ẽ^B_I, Z_T> - ℓᵀ Q^0(T) ℓ / 2, against Itô sums.
    rho = -0.6
    corr = [[1.0, rho], [rho, 1.0]]                       # letters 1 = W, 2 = W^B
    mw = [(), (0,), (1,)]
    ell = np.array([0.2, 0.05, 0.1])
    t, Z = sg.correlated_brownian(1.0, 2**10, corr, 300, seed=17)
    errs = []
    for n in (2**5, 2**10):
        tn, Zn = t[:: 2**10 // n], Z[:, :: 2**10 // n]
        S = sg.signature(sg.time_augment(tn, Zn), 3)
        Q0 = np.array([[sg.pair(sg.append_letter(sg.shuffle(u, v), 0), S, 3) for v in mw] for u in mw])
        lin = sum(c * sg.pair(sg.ito_integral_functional({w: 1.0}, 2, corr), S, 3) for c, w in zip(ell, mw))
        formula = lin - 0.5 * np.einsum("i,pij,j->p", ell, Q0.transpose(2, 0, 1), ell)
        sigma = ell[0] + ell[1] * tn[:-1] + ell[2] * Zn[:, :-1, 0]
        ito = np.sum(sigma * np.diff(Zn[:, :, 1], axis=1) - 0.5 * sigma**2 * np.diff(tn), axis=1)
        errs.append(np.sqrt(np.mean((formula - ito) ** 2)))
    assert errs[1] < 0.3 * errs[0] and errs[1] < 3e-3          # O(sqrt(h)): ratio about 0.18


def test_gbm_truncation_error():
    # ex:sig-gbm: S_T = S_0 sum_k σ^k J_k(T) with E[J_k J_l] = 1{k=l} T^k / k!.
    sigma, T = 0.4, 1.0
    J = sg.iterated_ito_integrals(4)
    E = sg.brownian_expected_signature(T, [[1.0]], 8)
    for k in range(5):
        for l in range(5):
            val = sg.pair(sg.shuffle_functionals(J[k], J[l]), E, 2)
            assert val == pytest.approx(T**k / factorial(k) if k == l else 0.0, abs=1e-12)
    assert J[2] == {(1, 1): 1.0, (0,): -0.5}
    t, W = sg.correlated_brownian(T, 400, [[1.0]], 4000, seed=13)
    S = sg.signature(sg.time_augment(t, W), 4)
    ST = np.exp(sigma * W[:, -1, 0] - 0.5 * sigma**2 * T)
    approx = sum(sigma**k * sg.pair(J[k], S, 2) for k in range(5))
    exact_err = sum(sigma ** (2 * k) * T**k / factorial(k) for k in range(5, 30))
    assert np.mean((ST - approx) ** 2) == pytest.approx(exact_err, rel=0.5, abs=2e-6)


def test_calibration_fits_heston_smiles():
    # sec:sig-calibration: a 13-parameter model (n = 3) fits two Heston smiles to a few bp in-sample.
    from ieor222.volatility import black_vega_normalized, heston_call, implied_vol

    corr = [[1.0, -0.5], [-0.5, 1.0]]
    Ts = [0.25, 1.0]
    t, W = sg.correlated_brownian(1.0, 60, corr, 8000, seed=14, antithetic=True)
    sigs = sg.signature(sg.time_augment(t, W), 3, at=[15, 60])
    feats = {}
    for T, S in zip(Ts, sigs):
        mw, feats[T] = sg.linear_model_features(S, 3, corr)
    quotes = {}
    for T in Ts:
        K = np.exp(0.2 * np.sqrt(T) * np.linspace(-1.5, 1.5, 7))
        p = heston_call(1.0, K, T, 0.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        iv = implied_vol(p, 1.0, K, T)
        quotes[T] = (K, p, black_vega_normalized(np.log(K), iv * np.sqrt(T)) * np.sqrt(T))
    res = sg.calibrate_linear_model(1.0, quotes, feats, sg.gbm_initial_guess(0.2, 1.0, mw))
    for T in Ts:
        K, p, _ = quotes[T]
        iv_model = implied_vol(sg.call_prices(1.0 + feats[T] @ res.x, K, 1.0), 1.0, K, T)
        assert np.max(np.abs(iv_model - implied_vol(p, 1.0, K, T))) < 25e-4


def test_signature_regression_improves_with_depth():
    # thm:sig-universal in practice: out-of-sample R^2 of the Asian call payoff grows with the depth.
    t, W = sg.correlated_brownian(1.0, 50, [[1.0]], 4000, seed=15)
    X = sg.time_augment(t, W)
    S = sg.signature(X, 4)
    St = np.exp(0.3 * W[:, :, 0] - 0.045 * t)
    y = np.maximum(0.5 * (St[:, 1:] + St[:, :-1]).mean(axis=1) - 1.0, 0.0)
    r2 = []
    for depth in (1, 2, 4):
        Phi = sg.flatten(S[: depth + 1])
        coef = np.linalg.lstsq(Phi[:2000], y[:2000], rcond=None)[0]
        r2.append(1 - np.var(y[2000:] - Phi[2000:] @ coef) / np.var(y[2000:]))
    assert r2[0] < r2[1] < r2[2] and r2[2] > 0.95
