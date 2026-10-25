"""Checks of the Chapter 1 code against the results proved in the notes."""
from math import comb

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.stats import norm

from ieor222.cointoss import (
    bayes_conditional_expectation,
    binomial_forecast,
    coin_density,
    conditional_expectation,
    expectation,
    gaussian_shift_density,
    gaussian_shift_sample,
    heads_count,
    is_measurable,
    least_squares,
    outcome_names,
    partition_of,
    stock_prices,
    toss_partition,
    toss_probabilities,
    toss_space,
)

N, P, S0, U, D = 6, 0.6, 4.0, 1.5, 0.5


@pytest.fixture
def space():
    heads = toss_space(N)
    return heads, toss_probabilities(heads, P), stock_prices(heads, S0, U, D)


def test_space_order_and_binomial_law(space):
    heads, prob, _ = space
    assert outcome_names(toss_space(2)) == ["HH", "HT", "TH", "TT"]
    assert heads.shape == (2**N, N) and len({tuple(r) for r in heads}) == 2**N
    assert prob.sum() == pytest.approx(1.0)
    h = heads.sum(axis=1)
    for j in range(N + 1):
        assert prob[h == j].sum() == pytest.approx(comb(N, j) * P**j * (1 - P) ** (N - j))


def test_atoms_of_Fk(space):
    heads, _, S = space
    for k in range(N + 1):
        lab = toss_partition(N, k)
        assert np.unique(lab).size == 2**k
        for a in np.unique(lab):  # an atom = all outcomes with a given first k tosses
            block = heads[lab == a, :k]
            assert len(block) == 2 ** (N - k) and np.all(block == block[0])
        # the stock is adapted: S_k is constant on the atoms of F_k, S_{k+1} is not
        assert is_measurable(S[:, k], lab)
        if k < N:
            assert not is_measurable(S[:, k + 1], lab)


def test_information_generated_by_S2():
    # Two tosses: sigma(S_2) has atoms {HH}, {HT, TH}, {TT}; S_1 is not sigma(S_2)-measurable.
    heads = toss_space(2)
    S = stock_prices(heads, S0, U, D)
    assert np.allclose(S[:, 2], [9, 3, 3, 1])
    lab = partition_of(S[:, 2])
    assert lab[1] == lab[2] and len(np.unique(lab)) == 3
    assert not is_measurable(S[:, 1], lab)
    assert is_measurable(np.abs(S[:, 2] - S[:, 1]), toss_partition(2, 1))  # = S_1 / 2


def test_stock_conditional_expectation(space):
    _, prob, S = space
    m = P * U + (1 - P) * D
    assert expectation(S[:, N], prob) == pytest.approx(S0 * m**N)
    for k in range(N + 1):
        ce = conditional_expectation(S[:, N], prob, toss_partition(N, k))
        assert np.allclose(ce, m ** (N - k) * S[:, k], rtol=1e-12)


def test_partial_averaging(space):
    _, prob, S = space
    rng = np.random.default_rng(0)
    X = S.max(axis=1) + rng.normal(size=2**N)
    for k in range(N + 1):
        lab = toss_partition(N, k)
        Y = conditional_expectation(X, prob, lab)
        for _ in range(20):  # random events of F_k: unions of atoms
            chosen = rng.random(2**k) < 0.5
            A = chosen[lab].astype(float)
            assert expectation(Y * A, prob) == pytest.approx(expectation(X * A, prob))


def test_tower_property_and_non_nested(space):
    _, prob, S = space
    X = S.max(axis=1)
    for k in range(N + 1):
        for j in range(k, N + 1):
            inner = conditional_expectation(X, prob, toss_partition(N, j))
            assert np.allclose(conditional_expectation(inner, prob, toss_partition(N, k)),
                               conditional_expectation(X, prob, toss_partition(N, k)))
    # sigma(S_2) and F_1 are not nested, and the order of conditioning matters
    a, b = toss_partition(N, 1), partition_of(S[:, 2])
    ab = conditional_expectation(conditional_expectation(X, prob, a), prob, b)
    ba = conditional_expectation(conditional_expectation(X, prob, b), prob, a)
    assert not np.allclose(ab, ba)


def test_known_factors_jensen_independence(space):
    heads, prob, S = space
    X = np.maximum(S[:, N] - S0, 0.0)
    for k in range(N + 1):
        lab = toss_partition(N, k)
        Zk = np.log(S[:, k]) ** 2 + 1  # F_k-measurable
        cX = conditional_expectation(X, prob, lab)
        assert np.allclose(conditional_expectation(Zk * X, prob, lab), Zk * cX)
        assert np.all(cX**2 <= conditional_expectation(X**2, prob, lab) + 1e-12)
        # a function of the tosses after k is independent of F_k
        future = heads[:, k:].sum(axis=1) ** 2
        assert np.allclose(conditional_expectation(future, prob, lab), expectation(future, prob))


def test_independence_lemma_binomial_formula(space):
    _, prob, S = space
    for h in (lambda x: np.maximum(x - S0, 0.0), np.log, np.sqrt):
        for k in range(N + 1):
            ce = conditional_expectation(h(S[:, N]), prob, toss_partition(N, k))
            assert np.allclose(ce, binomial_forecast(h, S[:, k], N - k, P, U, D))


def test_projection(space):
    _, prob, S = space
    k = 3
    X = np.maximum(S[:, N] - S0, 0.0)
    lab = partition_of(S[:, k])
    ce = conditional_expectation(X, prob, lab)
    # projection on all functions of S_k (indicators of its level sets) = E[X | S_k]
    indicators = np.stack([(lab == a).astype(float) for a in np.unique(lab)], axis=1)
    assert np.allclose(least_squares(X, indicators, prob), ce)
    # S_k takes k + 1 values, so polynomials of degree k in S_k span the same space
    x = S[:, k] / S0
    assert np.allclose(least_squares(X, np.vander(x, k + 1), prob), ce)
    # orthogonality and Pythagoras against the best affine predictor
    lin = least_squares(X, np.vander(x, 2), prob)
    assert expectation((X - ce) * x**2, prob) == pytest.approx(0.0, abs=1e-12)
    mse = lambda Y: expectation((X - Y) ** 2, prob)
    assert mse(lin) == pytest.approx(mse(ce) + expectation((ce - lin) ** 2, prob))
    assert mse(ce) < mse(lin)
    # variance decomposition: Var X = E[(X - E[X|G])^2] + Var(E[X|G])
    m = expectation(X, prob)
    assert expectation((X - m) ** 2, prob) == pytest.approx(mse(ce) + expectation((ce - m) ** 2, prob))


def test_change_of_measure_coins(space):
    heads, prob, S = space
    pt = 0.3
    Z = coin_density(heads, P, pt)
    q = toss_probabilities(heads, pt)
    assert np.allclose(Z * prob, q) and expectation(Z, prob) == pytest.approx(1.0)
    X = np.maximum(S[:, N] - S0, 0.0)
    assert expectation(X, q) == pytest.approx(expectation(Z * X, prob))
    H = heads_count(heads)
    for k in range(N + 1):
        lab = toss_partition(N, k)
        Zk = (pt / P) ** H[:, k] * ((1 - pt) / (1 - P)) ** (k - H[:, k])
        assert np.allclose(conditional_expectation(Z, prob, lab), Zk)  # density of Q on F_k
        direct = conditional_expectation(X, q, lab)                    # computed under Q
        assert np.allclose(bayes_conditional_expectation(X, Z, prob, lab), direct)


def test_gaussian_shift_by_quadrature():
    theta, b = 0.8, 0.3
    x = np.linspace(-12, 12, 200_001)
    phi = norm.pdf(x)
    w = gaussian_shift_density(x, theta) * phi
    assert np.allclose(w, norm.pdf(x - theta))                 # completing the square
    integral = lambda f: trapezoid(f * w, x)
    assert integral(np.ones_like(x)) == pytest.approx(1.0)
    assert integral(x) == pytest.approx(theta)
    assert integral(x**2) == pytest.approx(1 + theta**2)
    assert integral((x <= b).astype(float)) == pytest.approx(norm.cdf(b - theta), abs=1e-4)


def test_gaussian_shift_by_sampling():
    theta = 0.7
    X, Z = gaussian_shift_sample(theta, 400_000, seed=1)
    assert Z.mean() == pytest.approx(1.0, abs=0.01)
    mean = np.mean(Z * X)
    assert mean == pytest.approx(theta, abs=0.01)
    assert np.mean(Z * (X - theta) ** 2) == pytest.approx(1.0, abs=0.02)
    assert np.mean(Z**2) == pytest.approx(np.exp(theta**2), rel=0.03)  # E[Z^2] = e^{theta^2}
