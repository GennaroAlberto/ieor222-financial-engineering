"""Checks of the Chapter 5 code against the results proved in the notes."""
import numpy as np
import pytest

from ieor222.brownian import (
    brownian_paths,
    exponential_martingale,
    p_variation,
    scaled_random_walk,
    symmetric_random_walk,
)


def test_random_walk_shape_and_steps():
    M = symmetric_random_walk(50, n_paths=7, seed=0)
    assert M.shape == (7, 51)
    assert np.all(M[:, 0] == 0)
    assert set(np.unique(np.diff(M, axis=1))) <= {-1, 1}


def test_scaled_walk_quadratic_variation_is_exactly_t():
    # The scaled walk has quadratic variation exactly t on every path, not just on average.
    for n in (1, 10, 250):
        t, W = scaled_random_walk(n, T=2.0, n_paths=5, seed=n)
        assert np.allclose(p_variation(W, 2), t[-1])


def test_scaled_walk_variance_is_t():
    t, W = scaled_random_walk(400, T=1.0, n_paths=40_000, seed=1)
    assert abs(W[:, -1].var() - 1.0) < 0.05
    assert abs(W[:, -1].mean()) < 0.03


def test_brownian_covariance_is_min():
    t, W = brownian_paths(T=1.0, n_steps=100, n_paths=40_000, seed=2)
    i, j = 30, 70  # s = 0.3, t = 0.7
    cov = np.mean(W[:, i] * W[:, j])
    assert cov == pytest.approx(0.3, abs=0.02)


def test_brownian_quadratic_variation_close_to_T():
    # E[(Q - T)^2] = 2 T^2 / n: with n = 2^16 the standard deviation is about 0.0055.
    T = 1.0
    _, W = brownian_paths(T=T, n_steps=2**16, n_paths=1, seed=3)
    assert p_variation(W[0], 2) == pytest.approx(T, abs=0.03)


def test_first_variation_grows_under_refinement():
    # Refining a partition can only increase the first variation (triangle inequality);
    # for Brownian motion it grows without bound (infinite first variation).
    _, W = brownian_paths(T=1.0, n_steps=2**14, n_paths=1, seed=4)
    v1 = [p_variation(W[0], 1, every=2**k) for k in range(14, -1, -1)]
    assert np.all(np.diff(v1) >= -1e-12)
    assert v1[-1] > 50 * v1[0]


def test_exponential_martingale_has_mean_one():
    t, W = brownian_paths(T=1.0, n_steps=50, n_paths=100_000, seed=5)
    Z = exponential_martingale(t, W, sigma=0.8)
    assert np.allclose(Z.mean(axis=0), 1.0, atol=0.02)
