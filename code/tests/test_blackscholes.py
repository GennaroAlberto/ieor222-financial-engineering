"""Checks of the Chapter 6 code against the results proved in the notes."""
import numpy as np
import pytest

from ieor222.blackscholes import (
    asset_or_nothing_call,
    atm_forward_call,
    binomial_call,
    binomial_put,
    brenner_subrahmanyam,
    call,
    crr_parameters,
    d_plus_minus,
    digital_call,
    digital_put,
    gbm_paths,
    gbm_terminal,
    greeks,
    log_return_law,
    norm_cdf,
    norm_pdf,
    put,
)

CASES = [  # (x, K, tau, r, sigma): in, at and out of the money, r > 0, r = 0, r < 0
    (100.0, 100.0, 1.0, 0.05, 0.2),
    (80.0, 100.0, 0.5, 0.0, 0.3),
    (120.0, 90.0, 2.0, -0.01, 0.15),
    (50.0, 55.0, 0.1, 0.03, 0.6),
]


def test_reference_value():
    # A value quoted in many textbooks: S = K = 100, T = 1, r = 5%, sigma = 20%.
    assert call(100, 100, 1, 0.05, 0.2) == pytest.approx(10.450583572185565, rel=1e-12)


@pytest.mark.parametrize("x, K, tau, r, sigma", CASES)
def test_put_call_parity(x, K, tau, r, sigma):
    assert call(x, K, tau, r, sigma) - put(x, K, tau, r, sigma) == pytest.approx(x - K * np.exp(-r * tau))


@pytest.mark.parametrize("x, K, tau, r, sigma", CASES)
def test_call_is_asset_or_nothing_minus_K_digitals(x, K, tau, r, sigma):
    # thm:bs-formula from prop:bs-digital, and digital call + digital put = discount factor.
    c = asset_or_nothing_call(x, K, tau, r, sigma) - K * digital_call(x, K, tau, r, sigma)
    assert c == pytest.approx(call(x, K, tau, r, sigma))
    assert digital_call(x, K, tau, r, sigma) + digital_put(x, K, tau, r, sigma) == pytest.approx(np.exp(-r * tau))


def test_edge_cases():
    # tau = 0 gives the payoff; K = 0 gives the stock; the price stays inside the bounds of Chapter 3.
    assert call(110.0, 100.0, 0.0, 0.05, 0.2) == 10.0 and put(90.0, 100.0, 0.0, 0.05, 0.2) == 10.0
    assert call(100.0, 0.0, 1.0, 0.05, 0.2) == 100.0 and put(100.0, 0.0, 1.0, 0.05, 0.2) == 0.0
    x = np.linspace(1.0, 300.0, 300)
    c = call(x, 100.0, 1.0, 0.05, 0.2)
    assert np.all(c >= np.maximum(x - 100.0 * np.exp(-0.05), 0.0) - 1e-12) and np.all(c <= x)
    # sigma -> 0: the call tends to its lower bound (x - K e^{-r tau})^+
    assert call(120.0, 100.0, 1.0, 0.05, 1e-6) == pytest.approx(120.0 - 100.0 * np.exp(-0.05))


@pytest.mark.parametrize("x, K, tau, r, sigma", CASES)
def test_monte_carlo(x, K, tau, r, sigma):
    # The price is a discounted expectation under Q: sample S_T exactly and compare.
    S = gbm_terminal(x, tau, r, sigma, 400_000, seed=11)
    pay = np.exp(-r * tau) * np.maximum(S - K, 0.0)
    assert abs(pay.mean() - call(x, K, tau, r, sigma)) < 4 * pay.std() / np.sqrt(pay.size)
    prob = np.mean(S > K)  # Q(S_T > K) = N(d_-)
    assert abs(prob - norm_cdf(d_plus_minus(x, K, tau, r, sigma)[1])) < 4 * np.sqrt(0.25 / S.size)


def test_discounted_price_is_a_martingale():
    # e^{-rt} c(t, S_t) has constant mean: the time-t formula is consistent with the time-0 one.
    S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.03, 0.25
    t, S = gbm_paths(S0, T, r, sigma, n_steps=4, n_paths=400_000, seed=12)
    for j in (1, 2, 3):
        v = np.exp(-r * t[j]) * call(S[:, j], K, T - t[j], r, sigma)
        assert abs(v.mean() - call(S0, K, T, r, sigma)) < 4 * v.std() / np.sqrt(v.size)


def test_risk_neutral_probability_of_the_tree():
    # q_n -> 1/2 and n E[xi] -> (r - sigma^2/2) T: the two facts behind thm:bs-limit.
    T, r, sigma = 2.0, 0.04, 0.3
    for n in (10, 100, 10_000):
        u, d, r_n, q = crr_parameters(T, n, r, sigma)
        assert q * u + (1 - q) * d == pytest.approx(1 + r_n)
    support, w = log_return_law(T, 10_000, r, sigma)
    assert w.sum() == pytest.approx(1.0)
    assert (w * support).sum() == pytest.approx((r - sigma**2 / 2) * T, abs=1e-4)
    assert (w * support**2).sum() - (w * support).sum() ** 2 == pytest.approx(sigma**2 * T, rel=1e-3)
    with pytest.raises(ValueError):
        crr_parameters(1.0, 1, 0.5, 0.2)  # n <= r^2 T / sigma^2: the tree has an arbitrage


@pytest.mark.parametrize("K", [80.0, 100.0, 113.0])
def test_binomial_prices_converge(K):
    S0, T, r, sigma = 100.0, 1.0, 0.05, 0.2
    errs = [abs(binomial_call(S0, K, T, r, sigma, n) - call(S0, K, T, r, sigma)) for n in (50, 500, 5000)]
    assert errs[2] < 1e-3 and errs[2] < errs[0]
    # put-call parity holds exactly in every tree
    for n in (25, 26):
        assert binomial_call(S0, K, T, r, sigma, n) - binomial_put(S0, K, T, r, sigma, n) == pytest.approx(
            S0 - K * np.exp(-r * T))


def test_odd_even_oscillation_at_the_money():
    # With K = S0, even n put the strike on a node and odd n do not: the errors alternate in sign.
    S0 = K = 100.0
    e = [binomial_call(S0, K, 1.0, 0.05, 0.2, n) - call(S0, K, 1.0, 0.05, 0.2) for n in range(200, 206)]
    assert all(e[i] * e[i + 1] < 0 for i in range(5))


@pytest.mark.parametrize("x, K, tau, r, sigma", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_against_finite_differences(x, K, tau, r, sigma, kind):
    f = call if kind == "call" else put
    g = greeks(x, K, tau, r, sigma, kind)
    h = 1e-4
    assert g["delta"] == pytest.approx((f(x + h, K, tau, r, sigma) - f(x - h, K, tau, r, sigma)) / (2 * h), abs=1e-6)
    assert g["gamma"] == pytest.approx(
        (f(x + 1e-2, K, tau, r, sigma) - 2 * f(x, K, tau, r, sigma) + f(x - 1e-2, K, tau, r, sigma)) / 1e-4, rel=1e-4)
    assert g["vega"] == pytest.approx((f(x, K, tau, r, sigma + h) - f(x, K, tau, r, sigma - h)) / (2 * h), rel=1e-6)
    assert g["rho"] == pytest.approx((f(x, K, tau, r + h, sigma) - f(x, K, tau, r - h, sigma)) / (2 * h), rel=1e-6)
    # theta is the derivative in calendar time t, i.e. minus the derivative in tau
    assert g["theta"] == pytest.approx(-(f(x, K, tau + h, r, sigma) - f(x, K, tau - h, r, sigma)) / (2 * h), rel=1e-5)


@pytest.mark.parametrize("x, K, tau, r, sigma", CASES)
def test_identities_of_prop_greeks(x, K, tau, r, sigma):
    dp, dm = d_plus_minus(x, K, tau, r, sigma)
    assert K * np.exp(-r * tau) * norm_pdf(dm) == pytest.approx(x * norm_pdf(dp))
    g = greeks(x, K, tau, r, sigma)
    # the Black--Scholes equation (Chapter 10) and vega = sigma tau x^2 gamma
    lhs = g["theta"] + r * x * g["delta"] + 0.5 * sigma**2 * x**2 * g["gamma"]
    assert lhs == pytest.approx(r * call(x, K, tau, r, sigma))
    assert g["vega"] == pytest.approx(sigma * tau * x**2 * g["gamma"])
    # dual delta: dC/dK = - digital call price (Breeden--Litzenberger)
    h = 1e-4
    dCdK = (call(x, K + h, tau, r, sigma) - call(x, K - h, tau, r, sigma)) / (2 * h)
    assert dCdK == pytest.approx(-digital_call(x, K, tau, r, sigma), rel=1e-6)


def test_brenner_subrahmanyam_bounds():
    # prop:bs-atm: 0 <= S0 v / sqrt(2 pi) - C <= S0 v^3 / (24 sqrt(2 pi)), and the delta and vega bounds.
    S0, r = 100.0, 0.04
    for T in (1 / 12, 0.5, 2.0):
        for sigma in (0.05, 0.2, 0.5, 1.0):
            v = sigma * np.sqrt(T)
            K = S0 * np.exp(r * T)
            exact = call(S0, K, T, r, sigma)
            assert exact == pytest.approx(atm_forward_call(S0, T, sigma))
            assert exact == pytest.approx(put(S0, K, T, r, sigma))
            approx = brenner_subrahmanyam(S0, T, sigma)
            gap = approx["price"] - exact
            assert -1e-12 <= gap <= S0 * v**3 / (24 * np.sqrt(2 * np.pi)) + 1e-12
            g = greeks(S0, K, T, r, sigma)
            assert 0 <= approx["delta"] - g["delta"] <= v**3 / (48 * np.sqrt(2 * np.pi)) + 1e-12
            assert 0 <= approx["vega"] - g["vega"] <= approx["vega"] * v**2 / 8 + 1e-12


def test_trees_with_negative_rate_converge():
    # thm:bs-limit allows any real r: with r < 0 the trees need n > r^2 T / sigma^2 and still converge.
    S0, K, T, r, sigma = 100.0, 95.0, 2.0, -0.02, 0.25
    for f, g in ((binomial_call, call), (binomial_put, put)):
        assert abs(f(S0, K, T, r, sigma, 4000) - g(S0, K, T, r, sigma)) < 2e-3


def test_other_scaling_of_rem_bs_scaling():
    # rem:bs-scaling: the trees u = 1 + sigma sqrt(dt), d = 1 - sigma sqrt(dt) have the same limit.
    from ieor222.binomial import price_by_expectation
    S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.03, 0.3
    exact = call(S0, K, T, r, sigma)
    errs = []
    for n in (50, 200, 800):
        dt = T / n
        u, d = 1 + sigma * np.sqrt(dt), 1 - sigma * np.sqrt(dt)
        price = price_by_expectation(lambda s: np.maximum(s - K, 0.0), S0, u, d, np.exp(r * dt) - 1, n)
        errs.append(abs(price - exact))
    assert errs[-1] < 5e-3 and errs[-1] < errs[0]
