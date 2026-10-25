"""Checks of the Chapter 2 code against the results proved in the notes."""
import numpy as np
import pytest

from ieor222.martingales import (
    doubling,
    expected_exit_time,
    first_exit,
    martingale_transform,
    mean_one_products,
    play,
    prob_upper,
    random_walk,
    simulate_exit,
)


def test_random_walk_shape_start_and_steps():
    S = random_walk(40, n_paths=6, p=0.3, x0=5, seed=0)
    assert S.shape == (6, 41)
    assert np.all(S[:, 0] == 5)
    assert set(np.unique(np.diff(S, axis=1))) <= {-1, 1}


def test_compensated_walk_has_constant_mean():
    # Martingales of the random walk (a): S_n - n(p - q) is a martingale, while S_n drifts.
    p, n = 0.6, 50
    S = random_walk(n, n_paths=100_000, p=p, seed=1)
    drift = (2 * p - 1) * np.arange(n + 1)
    assert np.allclose((S - drift).mean(axis=0), 0.0, atol=0.1)
    assert S[:, -1].mean() == pytest.approx(10.0, abs=0.1)


def test_de_moivre_martingale_has_constant_mean():
    # Martingales of the random walk (b): (q/p)^{S_n} is a martingale.
    p = 0.55
    S = random_walk(30, n_paths=200_000, p=p, seed=2)
    M = ((1 - p) / p) ** S
    assert np.allclose(M.mean(axis=0), 1.0, atol=0.02)


def test_ruin_formulas_boundary_values_and_continuity_at_one_half():
    for p in (0.3, 0.5, 0.7):
        assert prob_upper(-3, -3, 4, p) == pytest.approx(0.0)
        assert prob_upper(4, -3, 4, p) == pytest.approx(1.0)
        assert expected_exit_time(-3, -3, 4, p) == pytest.approx(0.0, abs=1e-12)
        assert expected_exit_time(4, -3, 4, p) == pytest.approx(0.0, abs=1e-12)
    # The biased formulas tend to the symmetric ones as p -> 1/2.
    x = np.arange(0, 11)
    assert np.allclose(prob_upper(x, 0, 10, 0.5 + 1e-7), prob_upper(x, 0, 10, 0.5), atol=1e-5)
    assert np.allclose(expected_exit_time(x, 0, 10, 0.5 - 1e-6), x * (10 - x), atol=1e-3)


def test_one_step_interval():
    # From 1 in (0, 2) the walk exits after exactly one step, at 2 with probability p.
    for p in (0.2, 0.5, 0.9):
        assert prob_upper(1, 0, 2, p) == pytest.approx(p)
        assert expected_exit_time(1, 0, 2, p) == pytest.approx(1.0)


@pytest.mark.parametrize("p", [0.5, 0.45, 0.6])
def test_gamblers_ruin_against_simulation(p):
    a, b, x = -4, 6, 0
    tau, S_tau = simulate_exit(x, a, b, p=p, n_paths=100_000, seed=3)
    assert set(np.unique(S_tau)) <= {a, b}
    P = prob_upper(x, a, b, p)
    assert np.mean(S_tau == b) == pytest.approx(P, abs=4 * np.sqrt(P * (1 - P) / len(tau)))
    se = tau.std() / np.sqrt(len(tau))
    assert tau.mean() == pytest.approx(expected_exit_time(x, a, b, p), abs=4 * se)


def test_homework_value_for_exit_from_plus_minus_ten():
    # p = 0.6, exit from (-10, 10) started at 0: E[tau] = 10(2P - 1)/(2p - 1) with P = 1/(1 + (2/3)^10).
    P = 1 / (1 + (2 / 3) ** 10)
    assert prob_upper(0, -10, 10, 0.6) == pytest.approx(P)
    assert expected_exit_time(0, -10, 10, 0.6) == pytest.approx(48.2954, abs=1e-4)


def test_wald_identities_on_exit_times():
    # Wald's identities: E[S_tau - x] = (p - q) E[tau] and, for p = 1/2, E[(S_tau - x)^2] = E[tau].
    tau, S_tau = simulate_exit(2, 0, 9, p=0.5, n_paths=100_000, seed=4)
    assert np.mean((S_tau - 2) ** 2) == pytest.approx(tau.mean(), rel=0.02)
    tau, S_tau = simulate_exit(2, 0, 9, p=0.58, n_paths=100_000, seed=5)
    assert np.mean(S_tau - 2) == pytest.approx(0.16 * tau.mean(), rel=0.03)


def test_first_exit_on_sampled_paths():
    paths = np.array([[0, 1, 2, 3], [0, -1, 0, 1], [0, 1, 0, -1]])
    tau, S_tau = first_exit(paths, -1, 2)
    assert list(tau) == [2, 1, 3]
    assert list(S_tau) == [2, -1, -1]
    tau, S_tau = first_exit(paths, -5, 5)
    assert list(tau) == [-1, -1, -1] and np.all(np.isnan(S_tau))


def test_martingale_transform_and_stopped_process():
    S = random_walk(25, n_paths=50, seed=6)
    # H = 1 gives back the increments of S.
    assert np.allclose(martingale_transform(np.ones((50, 25)), S), S - S[:, :1])
    # Stopped martingales: S_{n ^ tau} = S_0 + (H . S)_n with H_k = 1{tau >= k}.
    tau, _ = first_exit(S, -3, 3)
    tau = np.where(tau < 0, 10**6, tau)
    k = np.arange(1, 26)
    H = (tau[:, None] >= k[None, :]).astype(float)
    n = np.arange(26)
    stopped = S[np.arange(50)[:, None], np.minimum(n[None, :], tau[:, None])]
    assert np.allclose(S[:, :1] + martingale_transform(H, S), stopped)


def test_strategies_cannot_beat_a_fair_game():
    # You cannot beat a fair game: bounded predictable stakes give gains with mean zero.
    def after_two_losses(k, past, gains):          # the gambler's fallacy
        if k < 3:
            return np.zeros(past.shape[0])
        return np.where((past[:, -1] == -1) & (past[:, -2] == -1), 3.0, 0.5)
    _, _, G = play(after_two_losses, 40, n_paths=200_000, seed=7)
    assert G[:, -1].mean() == pytest.approx(0.0, abs=4 * G[:, -1].std() / np.sqrt(200_000))


def test_doubling_with_finite_bankroll():
    # Bankroll 2^m - 1: gain +1 with probability 1 - 2^{-m}, else -(2^m - 1); the mean is zero.
    m, n_paths = 5, 400_000
    _, H, G = play(doubling(2**m - 1), 12, n_paths=n_paths, seed=8)
    final = G[:, -1]
    assert set(np.unique(final)) == {1.0, -(2.0**m - 1)}
    assert np.all(H[:, m:] == 0)                    # never plays after round m
    assert np.mean(final < 0) == pytest.approx(2.0**-m, abs=0.002)
    assert final.mean() == pytest.approx(0.0, abs=4 * final.std() / np.sqrt(n_paths))


def test_doubling_without_bankroll_always_ends_ahead():
    # The doubling strategy: with unlimited credit the gain is +1 as soon as a round is won.
    _, _, G = play(doubling(), 40, n_paths=10_000, seed=9)
    assert np.all(G[:, -1] == 1.0)                  # P(no win in 40 rounds) = 2^-40


def test_mean_one_products():
    M = mean_one_products(60, n_paths=400_000, a=0.5, seed=10)
    assert np.allclose(M[:, :11].mean(axis=0), 1.0, atol=0.03)
    # Typical paths decay: the median after 60 steps is (1.5 * 0.5)^30 = 0.75^30.
    assert np.median(M[:, -1]) == pytest.approx(0.75**30, rel=1e-9)
    assert np.mean(M[:, -1] < 1) > 0.9
