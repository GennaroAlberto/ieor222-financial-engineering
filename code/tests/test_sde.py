"""Checks of the Chapter 9 code against the results proved in the notes."""
import numpy as np
import pytest

from ieor222.sde import (
    absorbed_fraction,
    brownian_increments,
    cev_paths,
    coarsen,
    euler_maruyama,
    gbm_euler_errors,
    gbm_euler_mean,
    gbm_exact,
    ou_exact,
    ou_mean,
    ou_stationary,
    ou_var,
    path_from_increments,
    picard_iterates,
)


def test_coarsening_keeps_the_path():
    dW = brownian_increments(1.0, 64, n_paths=3, seed=0)
    W_fine = path_from_increments(dW)
    W_coarse = path_from_increments(coarsen(dW, 8))
    assert np.allclose(W_coarse, W_fine[:, ::8])


def test_euler_is_exact_for_brownian_motion_with_drift():
    # b = mu, sigma = s constant: X_t = x0 + mu t + s W_t, and Euler reproduces it on the grid.
    dW = brownian_increments(2.0, 50, n_paths=4, seed=1)
    t, X = euler_maruyama(lambda t, x: 0.3 + 0 * x, lambda t, x: 0.7 + 0 * x, 1.0, 2.0, 50, dW=dW)
    assert np.allclose(X, 1.0 + 0.3 * t + 0.7 * path_from_increments(dW))


def test_gbm_exact_solves_the_equation_in_the_limit():
    # Strong convergence (Theorem on Euler--Maruyama): on the same path, the Euler scheme
    # with a small step is close to the exact solution S0 exp(sigma W + (mu - sigma^2/2) t).
    S0, mu, sigma, T = 1.0, 0.2, 0.3, 1.0
    dW = brownian_increments(T, 2**14, n_paths=200, seed=2)
    t, S = euler_maruyama(lambda t, x: mu * x, lambda t, x: sigma * x, S0, T, 2**14, dW=dW)
    exact = gbm_exact(S0, mu, sigma, t, path_from_increments(dW))
    assert np.max(np.abs(S[:, -1] - exact[:, -1])) < 0.01


def test_mean_of_euler_for_gbm_is_s0_one_plus_mu_h_to_the_n():
    # Example in the notes: E[S^h_T] = S0 (1 + mu h)^n, which differs from S0 e^{mu T} at order h.
    S0, mu, sigma, T, n = 1.0, 0.8, 0.4, 1.0, 8
    _, S = euler_maruyama(lambda t, x: mu * x, lambda t, x: sigma * x, S0, T, n,
                          n_paths=400_000, seed=3)
    est, se = S[:, -1].mean(), S[:, -1].std() / np.sqrt(S.shape[0])
    assert abs(est - gbm_euler_mean(S0, mu, T, n)) < 4 * se
    assert abs(est - S0 * np.exp(mu * T)) > 10 * se          # the bias is visible
    # first-order expansion of the bias: S0 e^{mu T} mu^2 T h / 2
    h = T / 2**12
    bias = S0 * np.exp(mu * T) - gbm_euler_mean(S0, mu, T, 2**12)
    assert bias == pytest.approx(S0 * np.exp(mu * T) * mu**2 * T * h / 2, rel=1e-3)


def test_strong_order_one_half_and_weak_order_one():
    r = gbm_euler_errors(1.0, 1.0, 0.5, 1.0, levels=range(5, 11), n_paths=40_000, seed=4)
    strong_slope = np.polyfit(np.log(r["h"]), np.log(r["strong"]), 1)[0]
    weak_slope = np.polyfit(np.log(r["h"]), np.log(np.abs(r["weak_exact"])), 1)[0]
    assert 0.45 < strong_slope < 0.6
    assert 0.95 < weak_slope < 1.05
    # the coupled Monte Carlo estimate of the weak error agrees with the exact one
    assert np.all(np.abs(r["weak"] - r["weak_exact"]) < 4 * r["weak_se"] + 1e-12)


def test_ou_exact_sampler_has_the_gaussian_law_of_the_notes():
    x0, kappa, theta, sigma, T = 2.0, 1.5, -0.5, 0.8, 1.2
    t, X = ou_exact(x0, kappa, theta, sigma, T, 12, n_paths=200_000, seed=5)
    for k in (1, 6, 12):
        assert X[:, k].mean() == pytest.approx(ou_mean(x0, kappa, theta, t[k]), abs=0.01)
        assert X[:, k].var() == pytest.approx(ou_var(kappa, sigma, t[k]), rel=0.02)


def test_ou_stationary_law_is_invariant():
    kappa, theta, sigma = 2.0, 1.0, 0.6
    m, v = ou_stationary(kappa, theta, sigma)
    x0 = np.random.default_rng(6).normal(m, np.sqrt(v), size=200_000)
    _, X = ou_exact(x0, kappa, theta, sigma, 3.0, 30, n_paths=200_000, seed=7)
    assert X[:, -1].mean() == pytest.approx(m, abs=0.01)
    assert X[:, -1].var() == pytest.approx(v, rel=0.02)
    # and from any start the law approaches it
    assert ou_var(kappa, sigma, 50.0) == pytest.approx(v, rel=1e-12)
    assert ou_mean(10.0, kappa, theta, 50.0) == pytest.approx(m, abs=1e-12)


def test_euler_for_ou_has_mean_one_minus_kappa_h_to_the_n():
    # Euler for a linear drift: E X^h_{t_k} = theta + (x0 - theta)(1 - kappa h)^k exactly.
    x0, kappa, theta, sigma, T, n = 3.0, 2.0, 1.0, 0.5, 1.0, 10
    _, X = euler_maruyama(lambda t, x: kappa * (theta - x), lambda t, x: sigma + 0 * x,
                          x0, T, n, n_paths=200_000, seed=8)
    exact = theta + (x0 - theta) * (1 - kappa * T / n) ** np.arange(n + 1)
    assert np.allclose(X.mean(axis=0), exact, atol=0.01)


def test_cev_paths_are_nonnegative_and_absorbed():
    t, S = cev_paths(1.0, 0.0, 1.0, 0.5, 2.0, 400, n_paths=2_000, seed=9)
    assert np.all(S >= 0)
    hit = S <= 0
    assert np.all(hit[:, 1:] >= hit[:, :-1])                 # once at zero, always at zero
    frac = absorbed_fraction(S)
    assert np.all(np.diff(frac) >= 0) and frac[-1] > 0.2      # zero is reached, often
    # with gamma = 1 the scheme is the Euler scheme for GBM (on paths that stay positive)
    dW = brownian_increments(1.0, 100, n_paths=50, seed=10)
    _, S1 = cev_paths(1.0, 0.1, 0.2, 1.0, 1.0, 100, dW=dW)
    _, E1 = euler_maruyama(lambda t, x: 0.1 * x, lambda t, x: 0.2 * x, 1.0, 1.0, 100, dW=dW)
    assert np.allclose(S1, E1)


def test_picard_iterates_converge_to_the_euler_path():
    T, n = 1.0, 200
    dW = brownian_increments(T, n, n_paths=1, seed=11)
    t = np.linspace(0, T, n + 1)
    W = path_from_increments(dW)[0]
    b = lambda t, x: np.sin(x)
    s = lambda t, x: 0.5 * np.cos(x) + 0.2
    it = picard_iterates(b, s, 0.3, t, W, n_iter=30)
    _, em = euler_maruyama(b, s, 0.3, T, n, dW=dW)
    gaps = np.max(np.abs(it - em[0]), axis=1)
    assert gaps[-1] < 1e-10                                   # the fixed point is the Euler path
    assert np.all(np.diff(gaps[5:]) <= 1e-12)                 # and the gap shrinks


def test_cev_half_absorption_matches_feller_formula():
    # Exercise 9.6: for gamma = 1/2, mu = 0, P(S_T = 0) = exp(-2 s / (sigma^2 T)).
    s, sigma, T = 1.0, 1.0, 2.0
    _, S = cev_paths(s, 0.0, sigma, 0.5, T, 2_000, n_paths=5_000, seed=12)
    exact = np.exp(-2 * s / (sigma**2 * T))
    assert absorbed_fraction(S)[-1] == pytest.approx(exact, abs=0.03)
