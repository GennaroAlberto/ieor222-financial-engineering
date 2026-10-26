"""Checks of the Chapter 11 code against the results proved in the notes."""
import numpy as np
import pytest
from scipy.special import ndtr

from ieor222.brownian import brownian_paths
from ieor222.riskneutral import (
    bs_price,
    call_estimator_std,
    call_samples_two_numeraires,
    d_plus_minus,
    density_from_drift,
    density_second_moment,
    effective_sample_size,
    exercise_probabilities,
    forward_price,
    gbm_from_brownian,
    girsanov_density,
    weighted_estimate,
)

S0, K, T, r, sigma, mu = 100.0, 110.0, 1.0, 0.03, 0.2, 0.15
theta = (mu - r) / sigma


def test_density_has_mean_one_and_second_moment():
    # E[Z_t] = 1 for every t (Z is a martingale), and E[Z_T^2] = exp(theta^2 T).
    t, W = brownian_paths(T=T, n_steps=20, n_paths=200_000, seed=0)
    Z = girsanov_density(t, W, theta)
    assert np.allclose(Z.mean(axis=0), 1.0, atol=0.01)
    assert np.mean(Z[:, -1] ** 2) == pytest.approx(density_second_moment(theta, T), rel=0.03)


def test_girsanov_reweighted_brownian_motion_is_standard():
    # Under Q (weights Z_T), W~_t = W_t + theta t has mean 0, variance t and covariance min(s, t).
    t, W = brownian_paths(T=T, n_steps=10, n_paths=200_000, seed=1)
    Z = girsanov_density(t, W, theta)[:, -1]
    Wt = W + theta * t
    for j in (3, 7, 10):
        assert np.average(Wt[:, j], weights=Z) == pytest.approx(0.0, abs=0.01)
        assert np.average(Wt[:, j] ** 2, weights=Z) == pytest.approx(t[j], rel=0.02)
    assert np.average(Wt[:, 3] * Wt[:, 7], weights=Z) == pytest.approx(t[3], abs=0.01)
    # Under P it is a Brownian motion with drift theta: not centred.
    assert Wt[:, -1].mean() == pytest.approx(theta * T, abs=0.01)


def test_bayes_formula_for_conditional_mean():
    # E^Q[W_T | F_t] = W_t - theta (T - t), computed under P as E[Z_T W_T | F_t] / Z_t.
    # Given F_t, W_T = W_t + D with D ~ N(0, T - t): integrate D out by quadrature.
    t, w = 0.4, 0.7
    y = np.linspace(-12, 12, 40001)
    dens = np.exp(-0.5 * y * y) / np.sqrt(2 * np.pi) * (y[1] - y[0])
    D = np.sqrt(T - t) * y
    ZT_over_Zt = np.exp(-theta * D - 0.5 * theta**2 * (T - t))
    bayes = np.sum(ZT_over_Zt * (w + D) * dens)
    assert bayes == pytest.approx(w - theta * (T - t), abs=1e-9)


def test_discrete_density_is_exact_for_constant_drift():
    t, W = brownian_paths(T=T, n_steps=50, n_paths=4, seed=2)
    dW = np.diff(W, axis=1)
    Z = density_from_drift(np.full_like(dW, theta), dW, T / 50)
    assert np.allclose(Z, girsanov_density(t, W, theta))


def test_reweighted_price_equals_black_scholes():
    # Paths simulated under P (drift mu); weights Z_T turn them into Q-paths.
    t, W = brownian_paths(T=T, n_steps=1, n_paths=400_000, seed=3)
    S = gbm_from_brownian(t, W, S0, mu, sigma)
    Z = girsanov_density(t, W, theta)[:, -1]
    payoff = np.exp(-r * T) * np.maximum(S[:, -1] - K, 0.0)
    est, hw = weighted_estimate(payoff, Z)
    exact = bs_price(S0, K, T, r, sigma)
    assert abs(est - exact) < 1.5 * hw
    # The unweighted average estimates e^{-rT} E^P[(S_T - K)^+] = e^{(mu - r)T} c(S0; rate mu): not a price.
    naive, hw_naive = weighted_estimate(payoff)
    wrong = np.exp((mu - r) * T) * bs_price(S0, K, T, mu, sigma)
    assert abs(naive - wrong) < 1.5 * hw_naive
    assert naive > exact + 5 * hw


def test_discounted_stock_is_martingale_only_under_q():
    t, W = brownian_paths(T=2.0, n_steps=8, n_paths=200_000, seed=4)
    S = gbm_from_brownian(t, W, S0, mu, sigma)
    Z = girsanov_density(t, W, theta)
    disc = np.exp(-r * t) * S
    assert np.allclose((disc * Z).mean(axis=0), S0, rtol=0.01)       # E^Q[e^{-rt} S_t] = S_0
    assert np.allclose(disc.mean(axis=0), S0 * np.exp((mu - r) * t), rtol=0.01)


def test_effective_sample_size_matches_second_moment():
    t, W = brownian_paths(T=T, n_steps=1, n_paths=200_000, seed=5)
    for th in (0.3, 1.0):
        Z = girsanov_density(t, W, th)[:, -1]
        assert effective_sample_size(Z) / Z.size == pytest.approx(np.exp(-th**2 * T), rel=0.05)


def test_put_call_parity_with_dividends_and_reduction_to_q0():
    for q in (0.0, 0.02, -0.01):
        c = bs_price(S0, K, T, r, sigma, q, "call")
        p = bs_price(S0, K, T, r, sigma, q, "put")
        assert c - p == pytest.approx(S0 * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-12)
        dc = bs_price(S0, K, T, r, sigma, q, "digital_call")
        dp = bs_price(S0, K, T, r, sigma, q, "digital_put")
        assert dc + dp == pytest.approx(np.exp(-r * T), abs=1e-14)
        assert c == pytest.approx(bs_price(S0, K, T, r, sigma, q, "asset_call") - K * dc, abs=1e-12)
    # The dividend-paying stock is priced like a stock started at the prepaid forward S0 e^{-qT}.
    q = 0.04
    assert bs_price(S0, K, T, r, sigma, q) == pytest.approx(bs_price(S0 * np.exp(-q * T), K, T, r, sigma), abs=1e-12)
    # Known value: S0 = K = 100, T = 1, r = 5%, sigma = 20%, q = 0 gives 10.4506.
    assert bs_price(100.0, 100.0, 1.0, 0.05, 0.2) == pytest.approx(10.450583572185565, abs=1e-10)


def test_digital_is_derivative_of_call_in_strike():
    h = 1e-4
    for q in (0.0, 0.03):
        slope = (bs_price(S0, K + h, T, r, sigma, q) - bs_price(S0, K - h, T, r, sigma, q)) / (2 * h)
        assert -slope == pytest.approx(bs_price(S0, K, T, r, sigma, q, "digital_call"), abs=1e-7)


def test_forward_contract_three_ways():
    # Week 11A: the forward with delivery price K at T is worth S0 - K e^{-rT}.
    static = S0 - K * np.exp(-r * T)
    # risk-neutral: E^Q[e^{-rT}(S_T - K)], exactly by the martingale property, and by Monte Carlo
    t, W = brownian_paths(T=T, n_steps=1, n_paths=200_000, seed=6)
    S = gbm_from_brownian(t, W + (mu - r) / sigma * t, S0, r, sigma)  # Q-paths from W~ = W + theta t
    est, hw = weighted_estimate(np.exp(-r * T) * (S[:, -1] - K), girsanov_density(t, W, theta)[:, -1])
    assert abs(est - static) < 1.5 * hw
    # PDE: v(t, x) = x - K e^{-r(T - t)} solves v_t + r x v_x + sigma^2 x^2 v_xx / 2 = r v.
    x, tt, e = 87.0, 0.3, 1e-4
    v = lambda t_, x_: x_ - K * np.exp(-r * (T - t_))
    vt = (v(tt + e, x) - v(tt - e, x)) / (2 * e)
    vx = (v(tt, x + e) - v(tt, x - e)) / (2 * e)
    vxx = (v(tt, x + e) - 2 * v(tt, x) + v(tt, x - e)) / e**2
    assert vt + r * x * vx + 0.5 * sigma**2 * x**2 * vxx - r * v(tt, x) == pytest.approx(0.0, abs=1e-5)
    assert v(0.0, S0) == pytest.approx(static)
    # and the forward price, the delivery price that makes the contract worth zero
    assert forward_price(S0, T, r) == pytest.approx(S0 * np.exp(r * T))
    assert forward_price(S0, T, r, q=0.02) == pytest.approx(S0 * np.exp((r - 0.02) * T))


def test_numeraire_change_for_the_call():
    # Q(S_T > K) = N(d_-), Q^S(S_T > K) = N(d_+), and C = S0 Q^S(S_T > K) - K e^{-rT} Q(S_T > K).
    pQ, pS = exercise_probabilities(S0, K, T, r, sigma)
    dp, dm = d_plus_minus(S0, K, T, r, sigma)
    assert pQ == pytest.approx(ndtr(dm)) and pS == pytest.approx(ndtr(dp))
    assert S0 * pS - K * np.exp(-r * T) * pQ == pytest.approx(bs_price(S0, K, T, r, sigma), abs=1e-12)
    # Monte Carlo under Q^S: S has drift r + sigma^2.
    rng = np.random.default_rng(7)
    ST = S0 * np.exp(sigma * np.sqrt(T) * rng.standard_normal(400_000) + (r + 0.5 * sigma**2) * T)
    assert np.mean(ST > K) == pytest.approx(pS, abs=0.003)


def test_two_numeraire_estimators_are_unbiased_and_std_is_exact():
    samples = call_samples_two_numeraires(S0, K, T, r, sigma, 400_000, seed=8)
    exact = bs_price(S0, K, T, r, sigma)
    std = call_estimator_std(S0, K, T, r, sigma)
    for name in ("Q", "QS"):
        est, hw = weighted_estimate(samples[name])
        assert abs(est - exact) < 1.5 * hw
        assert samples[name].std() == pytest.approx(std[name][0], rel=0.01)
    assert np.all(samples["QS"] <= S0)
    # For a call the stock-numeraire estimator has the smaller variance at every strike tested,
    # for the volatility of the tests and for the one of Figure 11.2 (sigma = 30%).
    for vol, worst in ((sigma, 0.80), (0.3, 0.71)):
        s = call_estimator_std(S0, np.linspace(50, 200, 151), T, r, vol)
        assert np.all(s["QS"] < s["Q"])
        assert np.max(s["QS"] / s["Q"]) < worst


def test_weighted_estimator_beats_direct_q_for_the_figure_call():
    # Remark 11.4(d): for the call of Figure 11.1 the weighted P-estimator has about a quarter
    # of the variance of the plain estimator under Q (31.8 against 119.9).
    t, W = brownian_paths(T=T, n_steps=1, n_paths=400_000, seed=9)
    Z = girsanov_density(t, W, theta)[:, -1]
    S = gbm_from_brownian(t, W, S0, mu, sigma)
    var_weighted = np.var(Z * np.exp(-r * T) * np.maximum(S[:, -1] - K, 0.0))
    var_q = call_estimator_std(S0, K, T, r, sigma)["Q"][0] ** 2
    assert var_q == pytest.approx(119.90, abs=0.05)
    assert var_weighted / var_q == pytest.approx(0.265, abs=0.01)
