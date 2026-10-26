"""Checks of the Chapter 7 code against the results proved in the notes."""
import numpy as np
import pytest

from ieor222.brownian import brownian_paths, p_variation
from ieor222.itointegral import (
    covariation,
    isometry_check,
    ito_integral,
    lambda_sum,
    midpoint_sum,
    quadratic_variation,
    simulate,
    time_integral,
    wdw_limit,
)


def test_left_point_sum_is_the_algebraic_identity():
    # sum W_{t_j} (W_{t_{j+1}} - W_{t_j}) = (W_T^2 - V_2(W; Pi)) / 2 on every path, exactly.
    _, W = brownian_paths(T=2.0, n_steps=500, n_paths=5, seed=0)
    I = ito_integral(W, W)[:, -1]
    assert np.allclose(I, 0.5 * (W[:, -1] ** 2 - p_variation(W, 2)), atol=1e-10)
    assert np.allclose(I, lambda_sum(W, 0.0), atol=1e-10)


@pytest.mark.parametrize("lam", [0.0, 0.25, 0.5, 0.8, 1.0])
def test_lambda_sums_identity(lam):
    # S_lam = W_T^2 / 2 + (lam - 1/2) V_2(W; Pi) for every partition and every path.
    _, W = brownian_paths(T=1.0, n_steps=256, n_paths=4, seed=1)
    for every in (1, 4, 32):
        V2 = p_variation(W, 2, every=every)
        assert np.allclose(lambda_sum(W, lam, every), 0.5 * W[:, -1] ** 2 + (lam - 0.5) * V2, atol=1e-10)


def test_trapezoid_sum_is_exactly_half_W_squared():
    _, W = brownian_paths(T=1.0, n_steps=64, n_paths=3, seed=2)
    assert np.allclose(lambda_sum(W, 0.5), 0.5 * W[:, -1] ** 2, atol=1e-12)


def test_left_point_l2_error_is_T_over_sqrt_2n():
    # E[(S_0 - (W_T^2 - T)/2)^2] = (1/4) E[(V_2 - T)^2] = T^2 / (2n) for the uniform partition.
    T, n = 1.5, 64
    _, W = brownian_paths(T=T, n_steps=n, n_paths=40_000, seed=3)
    err = lambda_sum(W, 0.0) - wdw_limit(W[:, -1], T, 0.0)
    assert np.mean(err**2) == pytest.approx(T**2 / (2 * n), rel=0.05)


def test_midpoint_sums_converge_to_stratonovich_limit():
    # M - W_T^2 / 2 = (1/2) sum (A_j^2 - B_j^2), whose second moment is T^2 / (4n).
    T, n = 1.0, 64
    _, W = brownian_paths(T=T, n_steps=2 * n, n_paths=40_000, seed=4)
    err = midpoint_sum(W, every=2) - 0.5 * W[:, -1] ** 2
    assert np.mean(err**2) == pytest.approx(T**2 / (4 * n), rel=0.05)
    with pytest.raises(ValueError):
        midpoint_sum(W, every=3)


def test_ito_integral_of_W_is_a_martingale_with_the_isometry():
    # On the grid, E[I_T^2] = E sum W_{t_j}^2 h = T^2 (n - 1) / (2n) exactly.
    T, n = 1.0, 50
    t, W, H, I = simulate(lambda t, W: W, T=T, n_steps=n, n_paths=100_000, seed=5)
    assert np.allclose(I.mean(axis=0), 0.0, atol=0.01)
    assert np.mean(I[:, -1] ** 2) == pytest.approx(T**2 * (n - 1) / (2 * n), rel=0.02)


def test_isometry_check_for_a_bounded_integrand():
    res = isometry_check(lambda t, W: np.cos(W), T=1.0, n_steps=200, n_paths=40_000, seed=6)
    assert abs(res["lhs"] - res["rhs"]) < 4 * res["se"]
    assert abs(res["mean"]) < 4 * res["mean_se"]


def test_wiener_integral_is_gaussian_with_the_right_variance():
    # Deterministic integrand h(t) = e^{a t}: I_T ~ N(0, (e^{2aT} - 1) / (2a)).
    a, T, n = 0.7, 1.0, 400
    t, W, H, I = simulate(lambda t, W: np.exp(a * t), T=T, n_steps=n, n_paths=100_000, seed=7)
    x = I[:, -1]
    grid_var = np.sum(np.exp(2 * a * t[:-1]) * (T / n))  # exact variance of the left-point sum
    assert grid_var == pytest.approx((np.exp(2 * a * T) - 1) / (2 * a), rel=0.01)
    assert x.var() == pytest.approx(grid_var, rel=0.02)
    z = (x - x.mean()) / x.std()
    assert abs(np.mean(z**3)) < 0.03          # skewness 0
    assert np.mean(z**4) == pytest.approx(3.0, abs=0.06)  # kurtosis 3


def test_ito_integral_of_W_is_not_gaussian():
    # int_0^1 W dW = (W_1^2 - 1)/2 has the skewness of a chi-square(1) variable, sqrt(8).
    _, W, _, I = simulate(lambda t, W: W, T=1.0, n_steps=400, n_paths=100_000, seed=8)
    x = I[:, -1]
    z = (x - x.mean()) / x.std()
    assert np.mean(z**3) == pytest.approx(np.sqrt(8.0), rel=0.1)


def test_quadratic_variation_of_the_integral_is_int_H_squared():
    # [I, I]_T = int_0^T W_s^2 ds along one path (fine grid; the error has sd ~ sqrt(2 T^3 / n)).
    T, n = 1.0, 2**16
    t, W, H, I = simulate(lambda t, W: W, T=T, n_steps=n, n_paths=1, seed=9)
    qv = quadratic_variation(I[0])[-1]
    target = time_integral(W[0] ** 2, t)[-1]
    assert qv == pytest.approx(target, abs=0.03)
    # On a coarser sub-grid the realized quadratic variation is still close.
    assert quadratic_variation(I[0], every=64)[-1] == pytest.approx(target, abs=0.15)


def test_simple_integrand_does_not_depend_on_the_representation():
    # A simple process with jumps at k/8, integrated on the fine grid, equals the coarse sum.
    t, W = brownian_paths(T=1.0, n_steps=1024, n_paths=3, seed=11)
    coarse = np.arange(0, 1025, 128)
    xi = np.sin(W[:, coarse[:-1]])  # xi_j = sin(W_{t_j}) is F_{t_j}-measurable
    H = np.repeat(xi, 128, axis=1)
    H = np.concatenate([H, H[:, -1:]], axis=1)
    fine = ito_integral(H, W)[:, -1]
    direct = np.sum(xi * np.diff(W[:, coarse], axis=1), axis=1)
    assert np.allclose(fine, direct, atol=1e-10)


def test_independent_brownian_motions_have_zero_covariation():
    # E[(sum dW dB)^2] = sum h^2 = T^2 / n.
    T, n = 1.0, 100
    _, W = brownian_paths(T=T, n_steps=n, n_paths=50_000, seed=12)
    _, B = brownian_paths(T=T, n_steps=n, n_paths=50_000, seed=13)
    c = covariation(W, B)[:, -1]
    assert abs(c.mean()) < 0.002
    assert np.mean(c**2) == pytest.approx(T**2 / n, rel=0.03)


def test_covariation_of_two_integrals_is_int_HK():
    # [I, J]_T = int_0^T H K ds along one path, for I = int W dW and J = int cos(W) dW.
    T, n = 1.0, 2**16
    t, W = brownian_paths(T=T, n_steps=n, n_paths=1, seed=14)
    H, K = W[0], np.cos(W[0])
    I, J = ito_integral(H, W[0]), ito_integral(K, W[0])
    target = time_integral(H * K, t)[-1]
    assert covariation(I, J)[-1] == pytest.approx(target, abs=0.02)
    # polarization: [I, J] = ([I+J] - [I-J]) / 4, exactly for the discrete sums
    pol = (quadratic_variation(I + J)[-1] - quadratic_variation(I - J)[-1]) / 4
    assert pol == pytest.approx(covariation(I, J)[-1], abs=1e-10)
