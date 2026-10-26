"""Checks of the Chapter 8 code against the results proved in the notes."""
import numpy as np
import pytest

from ieor222.brownian import brownian_paths
from ieor222.itoformula import (
    correlated_brownian,
    covariation,
    ito_check,
    left_sum,
    levy_brownian,
    sgn,
    time_integral,
    trapezoid_sum,
)


def test_left_sum_of_w_against_w_is_exact_algebra():
    # sum W_j (W_{j+1} - W_j) = (W_T^2 - sum (W_{j+1} - W_j)^2) / 2 for every path and grid.
    _, W = brownian_paths(T=2.0, n_steps=500, n_paths=4, seed=0)
    qv = covariation(W, W)
    assert np.allclose(left_sum(W, W), 0.5 * (W**2 - qv))


def test_left_sum_of_w_against_w_converges_to_ito_value():
    # int_0^T W dW = (W_T^2 - T) / 2 (Chapter 7); the Riemann sums differ from it by
    # (T - sum of squared increments) / 2, whose standard deviation is sqrt(T^2 / (2n)).
    T = 1.0
    _, W = brownian_paths(T=T, n_steps=2**16, n_paths=3, seed=1)
    assert np.allclose(left_sum(W, W)[:, -1], 0.5 * (W[:, -1] ** 2 - T), atol=0.02)


def test_trapezoid_sum_of_w_is_the_chain_rule():
    # The trapezoid (Stratonovich) sum of W against W telescopes to W_T^2 / 2 exactly.
    _, W = brownian_paths(T=1.0, n_steps=300, n_paths=3, seed=2)
    assert np.allclose(trapezoid_sum(W, W), 0.5 * W**2)


def test_ito_formula_for_the_square_has_the_exact_discrete_error():
    # For f(x) = x^2 the discrete Ito formula is exact up to sum (dW^2 - h), which is the
    # quadratic variation error of Theorem 5.16 (thm:bm-qv).
    t, W = brownian_paths(T=1.0, n_steps=1000, n_paths=5, seed=3)
    r = ito_check(t, W, lambda t, x: x**2, lambda t, x: 2 * x, lambda t, x: 2 + 0 * x)
    assert np.allclose(r["lhs"] - r["rhs"], covariation(W, W) - t)
    assert np.allclose(r["lhs"] - r["first_order"], covariation(W, W))


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_ito_formula_holds_path_by_path_for_cos(seed):
    t, W = brownian_paths(T=1.0, n_steps=2**16, n_paths=1, seed=seed)
    r = ito_check(t, W, lambda t, x: np.cos(x), lambda t, x: -np.sin(x), lambda t, x: -np.cos(x))
    assert np.abs(r["lhs"] - r["rhs"]).max() < 0.02
    # Without the second-order term the error is (1/2) int cos(W_s) ds, of order one.
    gap = r["lhs"] - r["first_order"]
    assert np.allclose(gap, r["second_order"], atol=0.02)
    assert abs(gap[0, -1]) > 0.1


def test_exponential_martingale_has_no_drift():
    # f(t, x) = exp(s x - s^2 t / 2): the dt-terms cancel and f(t, W_t) = 1 + int s f dW.
    s = 0.8
    f = lambda t, x: np.exp(s * x - 0.5 * s**2 * t)
    t, W = brownian_paths(T=1.0, n_steps=2**15, n_paths=3, seed=7)
    r = ito_check(t, W, f, lambda t, x: s * f(t, x), lambda t, x: s**2 * f(t, x),
                  f_t=lambda t, x: -0.5 * s**2 * f(t, x))
    assert np.abs(r["lhs"] - r["stochastic"]).max() < 0.02


def test_gbm_and_discounted_gbm_dynamics():
    # S = S0 exp(sigma W + (mu - sigma^2/2) t) solves dS = mu S dt + sigma S dW, and
    # d(e^{-rt} S) = (mu - r) e^{-rt} S dt + sigma e^{-rt} S dW (Proposition prop:itf-gbm).
    S0, mu, sigma, r = 100.0, 0.08, 0.3, 0.03
    t, W = brownian_paths(T=1.0, n_steps=2**16, n_paths=3, seed=8)
    S = S0 * np.exp(sigma * W + (mu - 0.5 * sigma**2) * t)
    rhs = S0 + mu * time_integral(S, t) + sigma * left_sum(S, W)
    assert np.allclose(S, rhs, atol=0.3)  # relative error below 0.3 percent
    D = np.exp(-r * t) * S
    rhs = S0 + (mu - r) * time_integral(D, t) + sigma * left_sum(D, W)
    assert np.allclose(D, rhs, atol=0.3)


def test_discrete_product_rule_is_exact():
    # X_n Y_n - X_0 Y_0 = sum X_j dY_j + sum Y_j dX_j + sum dX_j dY_j, the discrete product rule.
    rng = np.random.default_rng(9)
    X, Y = rng.normal(size=(2, 4, 50)).cumsum(axis=-1)
    lhs = X * Y - X[..., :1] * Y[..., :1]
    assert np.allclose(lhs, left_sum(X, Y) + left_sum(Y, X) + covariation(X, Y))


def test_correlated_brownian_covariations():
    # [B^1, B^2]_T = rho T along one fine path; the sum has standard deviation T sqrt((1 + rho^2) / n).
    for rho in (-0.7, 0.0, 0.6, 1.0):
        t, B = correlated_brownian(T=1.0, n_steps=2**16, corr=rho, n_paths=2, seed=10)
        c12 = covariation(B[:, 0], B[:, 1])[:, -1]
        c11 = covariation(B[:, 0], B[:, 0])[:, -1]
        c22 = covariation(B[:, 1], B[:, 1])[:, -1]
        assert np.allclose(c12, rho, atol=0.03)
        assert np.allclose(c11, 1.0, atol=0.03) and np.allclose(c22, 1.0, atol=0.03)


def test_correlated_brownian_three_assets():
    C = np.array([[1.0, 0.5, -0.2], [0.5, 1.0, 0.3], [-0.2, 0.3, 1.0]])
    t, B = correlated_brownian(T=1.0, n_steps=20, corr=C, n_paths=60_000, seed=11)
    emp = np.einsum("pi,pj->ij", B[:, :, -1], B[:, :, -1]) / B.shape[0]
    assert np.allclose(emp, C, atol=0.03)


def test_levy_brownian_motion_code_check():
    # On the grid, sgn(W_{t_j}) (W_{t_{j+1}} - W_{t_j}) are exactly independent N(0, h), so these
    # checks test the code (left-point evaluation, sgn convention), not Levy's theorem.
    t, W = brownian_paths(T=1.0, n_steps=256, n_paths=40_000, seed=12)
    B = levy_brownian(W)
    assert np.allclose(covariation(B, B), covariation(W, W))
    b = B[:, -1]
    assert abs(b.mean()) < 0.02
    assert b.var() == pytest.approx(1.0, abs=0.03)
    assert set(np.unique(sgn(np.array([-1.0, 0.0, 2.0])))) == {-1.0, 1.0}


def test_levy_brownian_motion_joint_law_with_w():
    # B and W are uncorrelated Brownian motions, yet not independent: by the product rule,
    # d(W^2 B) = W^2 dB + B (2W dW + dt) + 2 |W| dt, so E[W_1^2 B_1] = 2 int_0^1 E|W_s| ds
    # = (4/3) sqrt(2/pi), while independence would give E[W_1^2] E[B_1] = 0.
    t, W = brownian_paths(T=1.0, n_steps=256, n_paths=40_000, seed=13)
    B = levy_brownian(W)
    w, b = W[:, -1], B[:, -1]
    assert np.mean(w * b) == pytest.approx(0.0, abs=0.03)
    assert np.mean(w**2 * b) == pytest.approx(4 / 3 * np.sqrt(2 / np.pi), abs=0.08)


def test_levy_needs_predictable_unit_integrands():
    # Right-point evaluation sgn(W_{t_{j+1}}) is not predictable: the sums have mean about
    # 2 sqrt(2/pi) (twice the mean local time, Exercise ex:itf-tanaka), not 0. And |Delta| != 1
    # breaks the variance: int (1 + sgn(W)/2) dW has variance E int (5/4 + sgn(W_s)) ds = 5/4.
    t, W = brownian_paths(T=1.0, n_steps=256, n_paths=40_000, seed=14)
    right = np.sum(sgn(W[:, 1:]) * np.diff(W, axis=1), axis=1)
    assert right.mean() > 1.3
    M = left_sum(1 + 0.5 * sgn(W), W)[:, -1]
    assert M.var() == pytest.approx(1.25, abs=0.05)
