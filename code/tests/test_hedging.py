"""Checks of the Chapter 10 code against the results proved in the notes."""
import numpy as np
import pytest

from ieor222.blackscholes import call, digital_call, gbm_paths, greeks, put
from ieor222.hedging import (
    asymptotic_rms,
    bs_residual,
    call_delta,
    call_gamma,
    call_hedge,
    delta_hedge,
    digital_call_delta,
    gamma_moment,
    gamma_moment_call,
    gamma_theta,
    hedged_pnl,
    hedging_error,
    power_claim,
    rebalanced_wealth,
)

S0, K, T, r, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


def call_fns(K=K, T=T, r=r, sigma=SIGMA):
    return (lambda t, x: call(x, K, T - t, r, sigma),
            lambda t, x: call_delta(x, K, T - t, r, sigma),
            lambda t, x: call_gamma(x, K, T - t, r, sigma))


def square_fns(T=T, r=r, sigma=SIGMA):
    f = lambda t, x, key: power_claim(x, T - t, 2.0, r, sigma)[key]
    return (lambda t, x: f(t, x, "value"), lambda t, x: f(t, x, "delta"), lambda t, x: f(t, x, "gamma"))


def square_constant(S0, T, r, sigma, mu):
    """(sigma^4 / 2) int_0^T E[e^{-2ru} Gamma^2 S_u^4] du for the claim S_T^2, in closed form."""
    c = mu - r + sigma**2
    return 0.5 * sigma**4 * S0**4 * np.exp(2 * (r + sigma**2) * T) * np.expm1(4 * c * T) / c


# ------------------------------------------------------------ prop:dh-bs-solution

@pytest.mark.parametrize("x, tau, rate, sigma", [(100.0, 1.0, 0.05, 0.2), (70.0, 0.3, 0.0, 0.4),
                                                  (130.0, 2.0, -0.01, 0.15)])
def test_call_and_put_solve_the_pde(x, tau, rate, sigma):
    # Exactly, with the Greeks of prop:bs-greeks: theta + r x delta + sigma^2 x^2 gamma / 2 = r c.
    for kind, price in (("call", call), ("put", put)):
        g = greeks(x, K, tau, rate, sigma, kind=kind)
        lhs = g["theta"] + rate * x * g["delta"] + 0.5 * sigma**2 * x**2 * g["gamma"]
        assert lhs == pytest.approx(rate * price(x, K, tau, rate, sigma), abs=1e-10)
    # And by finite differences, as in the Week 5B exercise.
    v = lambda t, y: call(y, K, 3.0 - t, rate, sigma)
    assert abs(bs_residual(v, 3.0 - tau, x, rate, sigma)) < 1e-4


def test_call_is_continuous_at_maturity_and_has_bounded_delta():
    # Joint continuity at (T, x): along (t, y) -> (T, x) the error is at most |y - x| plus the
    # time value at x, which is of order sqrt(tau) at the money (prop:bs-atm) and tiny elsewhere.
    x = np.array([80.0, 100.0, 120.0])
    errs = []
    for tau in (1e-2, 1e-4, 1e-6, 1e-8):
        y = x + np.sqrt(tau) * np.array([1.0, -1.0, 1.0])
        err = np.abs(call(y, K, tau, r, SIGMA) - np.maximum(x - K, 0.0))
        assert np.all(err <= np.sqrt(tau) * (1 + 0.5 * K * SIGMA) + r * K * tau)
        errs.append(err.max())
    assert errs[-1] < 1e-3 and np.all(np.diff(errs) < 0)
    d = call_delta(np.linspace(1.0, 400.0, 2000), K, 0.01, r, SIGMA)
    assert np.all((d >= 0) & (d <= 1))      # 0 < N(d_+) < 1, up to rounding in the tails
    d = call_delta(np.linspace(60.0, 160.0, 500), K, 0.25, r, SIGMA)
    assert np.all((d > 0) & (d < 1))


def test_power_claims_solve_the_pde_and_price_by_expectation():
    for a in (-1.0, 0.5, 2.0, 3.0):
        v = lambda t, x: power_claim(x, T - t, a, r, SIGMA)["value"]
        assert abs(bs_residual(v, 0.4, 90.0, r, SIGMA)) < 1e-4 * abs(v(0.4, 90.0))
        expect = np.exp(-r * T) * S0**a * np.exp(a * (r - 0.5 * SIGMA**2) * T + 0.5 * a**2 * SIGMA**2 * T)
        assert v(0.0, S0) == pytest.approx(expect, rel=1e-12)


def test_digital_delta_is_the_derivative_of_the_digital_price():
    x, tau, h = np.array([80.0, 100.0, 125.0]), 0.25, 1e-4
    fd = (digital_call(x + h, K, tau, r, SIGMA) - digital_call(x - h, K, tau, r, SIGMA)) / (2 * h)
    assert np.allclose(digital_call_delta(x, K, tau, r, SIGMA), fd, rtol=1e-6)


# ------------------------------------------------------------ self-financing and verification

def test_rebalanced_wealth_is_self_financing():
    t, S = gbm_paths(S0, T, 0.1, SIGMA, 12, 5, seed=1)
    shares = np.random.default_rng(2).normal(size=(5, 12))
    X = rebalanced_wealth(t, S, shares, 3.0, r)
    h = t[1] - t[0]
    assert np.allclose(X[:, 1:], shares * S[:, 1:] + np.exp(r * h) * (X[:, :-1] - shares * S[:, :-1]))


def test_forward_is_hedged_exactly_by_one_share():
    # v = x - K e^{-r tau} has delta 1: the hedge is static and exact for every n and every mu.
    value = lambda t, x: x - K * np.exp(-r * (T - t))
    delta = lambda t, x: np.ones_like(x)
    for mu in (-0.2, 0.3):
        t, S = gbm_paths(S0, T, mu, SIGMA, 7, 50, seed=3)
        assert np.allclose(hedging_error(t, S, value, delta, lambda x: x - K, r), 0.0, atol=1e-10)


def test_error_is_minus_the_sum_of_the_period_pnls():
    # prop:dh-hedging-error, first identity: e^{-rT}(X_T - g(S_T)) = -sum_k P_k, exactly.
    value, delta, _ = call_fns()
    t, S = gbm_paths(S0, T, 0.12, SIGMA, 20, 200, seed=4)
    err = hedging_error(t, S, value, delta, lambda x: np.maximum(x - K, 0.0), r)
    assert np.allclose(np.exp(-r * T) * err, -hedged_pnl(t, S, value, delta, r).sum(axis=1), atol=1e-10)


def test_mean_error_is_zero_when_mu_equals_r():
    # With mu = r, discounted gains and discounted prices are martingales: E[e_n] = 0 exactly.
    out = call_hedge(S0, K, T, r, SIGMA, mu=r, n=10, n_paths=40_000, seed=5)
    e = out["error"]
    assert abs(e.mean()) < 4 * e.std() / np.sqrt(e.size)


# ------------------------------------------------------------ the size of the error

def test_gamma_moment_quadrature_for_the_squared_claim():
    _, _, gamma = square_fns()
    for mu in (-0.1, 0.05, 0.3):
        exact = square_constant(S0, T, r, SIGMA, mu)
        moment = lambda u: gamma_moment(gamma, u, S0, r, SIGMA, mu)
        n = 50
        assert asymptotic_rms(moment, T, SIGMA, n) ** 2 * n / T == pytest.approx(exact, rel=1e-8)


def test_call_moment_closed_form_agrees_with_quadrature():
    _, _, gamma = call_fns()
    for u in (0.2, 0.5, 0.8):
        quad = gamma_moment(gamma, u, S0, r, SIGMA, 0.1, n_nodes=120)
        assert gamma_moment_call(u, S0, K, T, r, SIGMA, 0.1) == pytest.approx(quad, rel=1e-6)


def test_squared_claim_error_matches_the_leading_term():
    value, delta, _ = square_fns()
    for mu, n in ((r, 64), (0.25, 64)):
        t, S = gbm_paths(S0, T, mu, SIGMA, n, 20_000, seed=6)
        e = np.exp(-r * T) * hedging_error(t, S, value, delta, lambda x: x**2, r)
        lead = square_constant(S0, T, r, SIGMA, mu) * T / n
        assert np.mean(e**2) == pytest.approx(lead, rel=0.08)


def test_gamma_theta_remainder_is_of_order_h():
    # rho_n = e_n + sum of the gamma-theta terms has E[rho_n^2] <= C h^2, while E[e_n^2] ~ h.
    value, delta, gamma = square_fns()
    ms = []
    for n in (16, 64):
        t, S = gbm_paths(S0, T, 0.15, SIGMA, n, 20_000, seed=7)
        e = np.exp(-r * T) * hedging_error(t, S, value, delta, lambda x: x**2, r)
        rho = e + gamma_theta(t, S, gamma, r, SIGMA).sum(axis=1)
        ms.append((np.mean(rho**2), np.mean(e**2)))
    (rho16, e16), (rho64, e64) = ms
    assert rho16 / rho64 > 8          # about 16 for h^2
    assert 3 < e16 / e64 < 5.5        # about 4 for h
    assert rho64 < 0.01 * e64


def test_call_error_rms_matches_the_leading_term():
    # The call is outside hypothesis (H), but the leading term still describes it (Bertsimas-Kogan-Lo).
    for mu in (0.05, 0.3):
        out = call_hedge(S0, K, T, r, SIGMA, mu=mu, n=128, n_paths=8_000, seed=8)
        moment = lambda u: gamma_moment_call(u, S0, K, T, r, SIGMA, mu)
        lead = np.exp(r * T) * asymptotic_rms(moment, T, SIGMA, 128)
        assert np.sqrt(np.mean(out["error"] ** 2)) == pytest.approx(lead, rel=0.06)


def test_numbers_quoted_in_the_notes():
    # Call price, and the leading term of eq:dh-error-variance for the call of fig:dh-hedging-error.
    assert call(S0, K, T, r, SIGMA) == pytest.approx(10.4506, abs=5e-5)
    lead = lambda mu, n: np.exp(r * T) * asymptotic_rms(
        lambda u: gamma_moment_call(u, S0, K, T, r, SIGMA, mu), T, SIGMA, n)
    for mu, value in ((-0.10, 6.78), (0.05, 7.02), (0.30, 5.44)):
        assert lead(mu, 1) == pytest.approx(value, abs=5e-3)
    for n, value in ((12, 1.91), (52, 0.92), (252, 0.42)):
        assert lead(0.15, n) == pytest.approx(value, abs=5e-3)


def test_call_gamma_theta_remainder_is_only_of_order_h_three_halves():
    # Outside (H): for the call, E[rho_n^2] falls by about 8 (not 16) when n is multiplied by 4.
    value, delta, gamma = call_fns()
    ms = []
    for n in (16, 64, 256):
        t, S = gbm_paths(S0, T, 0.15, SIGMA, n, 10_000, seed=n + 1)
        e = -hedged_pnl(t, S, value, delta, r).sum(axis=1)
        ms.append(np.mean((e + gamma_theta(t, S, gamma, r, SIGMA).sum(axis=1)) ** 2))
    for a, b in zip(ms, ms[1:]):
        assert 6 < a / b < 11
