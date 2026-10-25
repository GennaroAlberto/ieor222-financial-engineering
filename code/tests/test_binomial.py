"""Checks of the Chapter 4 code against the results proved in the notes."""
import numpy as np
import pytest

from ieor222.binomial import (
    american,
    early_exercise_boundary,
    european,
    path_dependent,
    price_by_expectation,
    replication_error,
    risk_neutral_probability,
    stock_paths,
    stock_tree,
    tosses,
    trinomial_moment_matching,
    trinomial_price_interval,
    trinomial_risk_neutral,
    trinomial_theta_max,
)

call = lambda K: (lambda s: np.maximum(s - K, 0.0))
put = lambda K: (lambda s: np.maximum(K - s, 0.0))


def test_risk_neutral_probability_and_no_arbitrage_condition():
    p = risk_neutral_probability(1.2, 0.9, 0.05)
    assert p == pytest.approx(0.5)
    assert p * 1.2 + (1 - p) * 0.9 == pytest.approx(1.05)       # E^Q[S_1] = (1 + r) S_0
    for u, d, r in ((1.2, 0.9, 0.25), (1.2, 0.9, -0.15), (1.2, 1.2, 0.0), (1.1, 0.0, 0.0)):
        with pytest.raises(ValueError):
            risk_neutral_probability(u, d, r)


def test_one_period_example_of_the_notes():
    V, D = european(call(100), 100, 1.2, 0.9, 0.05, 1)
    assert V[0][0] == pytest.approx(10 / 1.05) and D[0][0] == pytest.approx(2 / 3)


def test_backward_induction_equals_discounted_expectation():
    # thm:bin-multi-period (ii) at time 0: the CRR sum over the binomial law of the up-moves.
    for N in (1, 2, 7, 40):
        for f in (call(100), put(95), lambda s: s**2):
            V, _ = european(f, 100, 1.1, 0.92, 0.01, N)
            assert V[0][0] == pytest.approx(price_by_expectation(f, 100, 1.1, 0.92, 0.01, N))


def test_put_call_parity_on_the_tree():
    S0, K, r, N = 100, 104, 0.02, 12
    C, _ = european(call(K), S0, 1.08, 0.95, r, N)
    P, _ = european(put(K), S0, 1.08, 0.95, r, N)
    for n in range(N + 1):
        S = stock_tree(S0, 1.08, 0.95, N)[n]
        assert np.allclose(C[n] - P[n], S - K * (1 + r) ** (-(N - n)))


def test_replication_along_every_path():
    # thm:bin-multi-period (i): X_N = V_N on all 2^N paths, also for path-dependent claims.
    asian = lambda P: np.maximum(P.mean(axis=1) - 100, 0.0)
    lookback = lambda P: P.max(axis=1) - P[:, -1]
    european_call = lambda P: np.maximum(P[:, -1] - 100, 0.0)
    for f in (asian, lookback, european_call):
        assert replication_error(f, 100, 1.15, 0.9, 0.03, 10) < 1e-9


def test_full_tree_agrees_with_recombining_tree():
    S, V, D = path_dependent(lambda P: np.maximum(P[:, -1] - 100, 0.0), 100, 1.15, 0.9, 0.03, 6)
    Vr, Dr = european(call(100), 100, 1.15, 0.9, 0.03, 6)
    ups = tosses(6).sum(axis=1)
    assert np.allclose(V[6], Vr[6][ups])
    first3 = tosses(3).sum(axis=1)
    assert np.allclose(V[3], Vr[3][first3]) and np.allclose(D[3], Dr[3][first3])


def test_lookback_exercise_numbers():
    # Two-period lookback of the exercises: S0 = 8, u = 3/2, d = 1/2, r = 1/4.
    S, V, D = path_dependent(lambda P: P.max(axis=1) - P[:, -1], 8, 1.5, 0.5, 0.25, 2)
    assert V[0][0] == pytest.approx(1.2) and D[0][0] == pytest.approx(-0.15)
    assert np.allclose(V[1], [2.4, 1.2]) and np.allclose(D[1], [-1.0, -0.5])


def test_discounted_stock_is_a_Q_martingale_at_every_node():
    u, d, r, N = 1.3, 0.8, 0.04, 6
    p = risk_neutral_probability(u, d, r)
    S = stock_tree(50, u, d, N)
    for n in range(N):
        assert np.allclose(p * S[n + 1][1:] + (1 - p) * S[n + 1][:-1], (1 + r) * S[n])


def test_american_call_equals_european_call_when_r_nonnegative():
    for r in (0.0, 0.01, 0.05):
        U, _, _ = american(call(100), 100, 1.1, 0.9, r, 30)
        V, _ = european(call(100), 100, 1.1, 0.9, r, 30)
        for n in range(31):
            assert np.allclose(U[n], V[n])


def test_american_put_dominates_and_equals_european_when_r_is_zero():
    U, _, _ = american(put(100), 100, 1.1, 0.9, 0.02, 30)
    V, _ = european(put(100), 100, 1.1, 0.9, 0.02, 30)
    assert U[0][0] > V[0][0] + 0.1
    U0, _, _ = american(put(100), 100, 1.1, 0.9, 0.0, 30)
    V0, _ = european(put(100), 100, 1.1, 0.9, 0.0, 30)
    assert np.allclose(U0[0], V0[0])


def test_american_put_exercise_numbers():
    # Two-period American put of the exercises: S0 = 8, u = 3/2, d = 1/2, r = 1/4, K = 8.
    U, _, ex = american(put(8), 8, 1.5, 0.5, 0.25, 2)
    V, _ = european(put(8), 8, 1.5, 0.5, 0.25, 2)
    assert U[0][0] == pytest.approx(1.04) and V[0][0] == pytest.approx(0.72)
    assert ex[1].tolist() == [True, False]


def _stopping_times(N):
    """Every stopping time with values in {0,...,N}, as an array of values on the 2^N paths.

    Either tau = 0 everywhere, or tau >= 1 and tau - 1 is a stopping time of the subtree
    after the first toss (one for each value of the first toss).
    """
    if N == 0:
        return [np.array([0])]
    sub = _stopping_times(N - 1)
    return [np.zeros(2**N, dtype=int)] + [np.concatenate([1 + a, 1 + b]) for a in sub for b in sub]


def test_american_price_is_the_maximum_over_all_stopping_times():
    # thm:bin-american (ii), by brute force over the 677 stopping times of a 4-period tree.
    S0, u, d, r, N, K = 100, 1.1, 0.9, 0.02, 4, 100
    taus = _stopping_times(N)
    assert len(taus) == 677
    p = risk_neutral_probability(u, d, r)
    ups = tosses(N).sum(axis=1)
    Q = p**ups * (1 - p) ** (N - ups)
    paths = stock_paths(S0, u, d, N)
    values = [np.sum(Q * np.maximum(K - paths[np.arange(2**N), t], 0) / (1 + r) ** t) for t in taus]
    U, _, _ = american(put(K), S0, u, d, r, N)
    assert max(values) == pytest.approx(U[0][0], abs=1e-12)
    # the smallest optimal time tau* = min{n : U_n = G_n} attains it
    G = [np.maximum(K - s, 0) for s in stock_tree(S0, u, d, N)]
    ups_path = np.concatenate([np.zeros((2**N, 1), dtype=int), np.cumsum(tosses(N), axis=1)], axis=1)
    hit = np.array([np.isclose(U[n][ups_path[:, n]], G[n][ups_path[:, n]]) for n in range(N + 1)]).T
    tau_star = hit.argmax(axis=1)                       # first n with U_n = G_n on each path
    v_star = np.sum(Q * np.maximum(K - paths[np.arange(2**N), tau_star], 0) / (1 + r) ** tau_star)
    assert v_star == pytest.approx(U[0][0], abs=1e-12)


def test_american_seller_hedge_dominates_the_payoff():
    # thm:bin-american (iii): starting from U_0, the hedge keeps X_n >= U_n >= G_n everywhere.
    S0, u, d, r, N, K = 100, 1.1, 0.9, 0.02, 8, 100
    U, D, _ = american(put(K), S0, u, d, r, N)
    ups = np.concatenate([np.zeros((2**N, 1), dtype=int), np.cumsum(tosses(N), axis=1)], axis=1)
    paths = stock_paths(S0, u, d, N)
    X = np.full(2**N, U[0][0])
    for n in range(N):
        delta = np.array(D[n])[ups[:, n]]
        X = delta * paths[:, n + 1] + (1 + r) * (X - delta * paths[:, n])
        assert np.all(X >= np.array(U[n + 1])[ups[:, n + 1]] - 1e-10)


def test_put_exercise_boundary_is_below_strike():
    b = early_exercise_boundary(100, 100, 1.04, 1 / 1.04, 0.003, 60)
    assert np.all(b[~np.isnan(b)] < 100)


def test_trinomial_measures_and_price_interval():
    S0, u, m, d, r, K = 100, 1.2, 1.0, 0.8, 0.05, 100
    tmax = trinomial_theta_max(u, m, d, r)
    for theta in np.linspace(1e-6, tmax - 1e-6, 7):
        q = np.array(trinomial_risk_neutral(u, m, d, r, theta))
        assert np.all(q > 0) and q.sum() == pytest.approx(1)
        assert q @ np.array([u, m, d]) == pytest.approx(1 + r)
    low, high = trinomial_price_interval(call(K), S0, u, m, d, r)
    # high: binomial price on {d, u}; low: binomial price on {m, u} since m <= 1 + r
    assert high == pytest.approx(price_by_expectation(call(K), S0, u, d, r, 1))
    assert low == pytest.approx(price_by_expectation(call(K), S0, u, m, r, 1))
    grid = [np.array(trinomial_risk_neutral(u, m, d, r, t)) @ np.maximum(np.array([u, m, d]) * S0 - K, 0) / (1 + r)
            for t in np.linspace(1e-9, tmax - 1e-9, 101)]
    assert min(grid) == pytest.approx(low, abs=1e-6) and max(grid) == pytest.approx(high, abs=1e-6)


def test_trinomial_middle_at_interest_rate_and_replicable_claims():
    S0, u, d, r = 100, 1.25, 0.85, 0.03
    low, high = trinomial_price_interval(call(100), S0, u, 1 + r, d, r)
    assert low == pytest.approx(max((1 + r) * S0 - 100, 0) / (1 + r))
    # an affine claim is replicable: its price interval is a single point
    low, high = trinomial_price_interval(lambda s: 3 + 0.5 * s, S0, u, 1.0, d, r)
    assert low == pytest.approx(high) == pytest.approx(3 / (1 + r) + 0.5 * S0)


def test_trinomial_moment_matching_positivity():
    sigma, nu = 0.2, 0.05 - 0.02
    for dt in (1 / 12, 1 / 52, 1 / 365):
        dx, pu, pm, pd = trinomial_moment_matching(sigma, nu, dt)               # dx = sigma sqrt(3 dt)
        assert min(pu, pm, pd) > 0 and pu + pm + pd == pytest.approx(1)
        assert pu * dx - pd * dx == pytest.approx(nu * dt)
        assert (pu + pd) * dx**2 == pytest.approx(sigma**2 * dt + nu**2 * dt**2)
        _, _, pm1, _ = trinomial_moment_matching(sigma, nu, dt, lam=1.0)      # dx = sigma sqrt(dt)
        assert pm1 == pytest.approx(-nu**2 * dt / sigma**2) and pm1 < 0
