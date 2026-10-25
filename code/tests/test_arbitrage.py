"""Checks of the Chapter 3 code against the results proved in the notes."""
import numpy as np
import pytest
from scipy.special import ndtr

from ieor222.arbitrage import (
    Portfolio,
    carr_madan,
    find_arbitrages,
    lognormal_call,
    lognormal_density,
    lognormal_put,
    replicate_payoff,
    state_prices,
)

S0, B, V = 100.0, 0.97, 0.25
STRIKES = np.arange(60.0, 161.0, 5.0)


def reference_curve():
    """Calls priced by a pricing measure, hence arbitrage-free (pricing-measure lemma)."""
    return lognormal_call(S0, B, STRIKES, V), lognormal_put(S0, B, STRIKES, V)


def test_reference_curve_satisfies_every_relation():
    c, p = reference_curve()
    assert find_arbitrages(S0, B, STRIKES, c, p) == []
    # parity, bounds, monotonicity, slope bounds and convexity, checked directly
    assert np.allclose(c - p, S0 - B * STRIKES)
    assert np.all((c >= np.maximum(S0 - B * STRIKES, 0)) & (c <= S0))
    slopes = np.diff(c) / np.diff(STRIKES)
    assert np.all((slopes >= -B) & (slopes <= 0)) and np.all(np.diff(slopes) >= 0)


@pytest.mark.parametrize("i, bump, condition", [
    (0, -6.0, "lower bound"),      # C(60) pushed below S0 - 60 B
    (8, +3.0, "convexity"),        # a kink upwards in the middle of the curve
    (12, +5.0, "monotonicity"),    # C(120) above C(115)
    (20, -1.0, "positivity"),      # C(160) < 0
])
def test_violations_come_with_an_arbitrage(i, bump, condition):
    c, _ = reference_curve()
    c = c.copy()
    c[i] += bump
    found = find_arbitrages(S0, B, STRIKES, c)
    assert condition in {a.condition for a in found}
    quotes = dict(zip(STRIKES, c))
    for a in found:
        # payoff >= 0 in every state, cash collected today = minus the cost
        assert a.portfolio.min_payoff() >= -1e-9
        assert a.profit > 0
        assert a.portfolio.cost(S0, B, quotes) == pytest.approx(-a.profit)


def test_slope_and_upper_bound_violations():
    found = find_arbitrages(S0, B, [50.0, 60.0], [60.0, 45.0])   # drop 15 > 10 B
    assert {a.condition for a in found} >= {"slope"}
    found = find_arbitrages(S0, B, [50.0], [101.0])
    assert [a.condition for a in found] == ["upper bound"]
    assert found[0].portfolio.min_payoff() >= 0


def test_butterfly_example_of_the_notes():
    # S0 = 100, B = 0.97, C(95, 100, 105) = 8.0, 5.5, 2.6: butterfly costs -0.2 per middle call
    found = find_arbitrages(100.0, 0.97, [95.0, 100.0, 105.0], [8.0, 5.5, 2.6])
    assert [a.condition for a in found] == ["convexity"]
    a = found[0]
    assert a.profit == pytest.approx(0.2)
    x = np.linspace(0, 200, 4001)
    assert np.max(a.portfolio.payoff(x)) == pytest.approx(2.5)   # half a tent of height 5


def test_parity_violation_has_zero_payoff():
    c, p = reference_curve()
    p = p.copy()
    p[4] += 0.3
    found = [a for a in find_arbitrages(S0, B, STRIKES, c, p) if a.condition == "parity"]
    assert len(found) == 1 and found[0].profit == pytest.approx(0.3)
    x = np.linspace(0, 300, 301)
    assert np.allclose(found[0].portfolio.payoff(x), 0.0)


def test_state_prices_reproduce_the_quotes():
    c, _ = reference_curve()
    x, q = state_prices(S0, B, STRIKES, c)
    assert np.all(q > 0)
    assert q.sum() == pytest.approx(B)
    assert (q * x).sum() == pytest.approx(S0)
    model = np.array([(q * np.maximum(x - K, 0)).sum() for K in STRIKES])
    assert np.allclose(model, c)
    # with equal spacing h, interior state prices are butterflies divided by h
    h = STRIKES[1] - STRIKES[0]
    fly = (c[:-2] - 2 * c[1:-1] + c[2:]) / h
    assert np.allclose(q[2:2 + len(fly)], fly)


def test_flat_tail_is_reported_and_has_no_state_prices():
    found = find_arbitrages(100.0, 1.0, [100.0, 110.0], [5.0, 5.0])
    assert [a.condition for a in found] == ["flat tail"]
    assert found[0].profit == 0.0 and found[0].portfolio.min_payoff() >= 0
    with pytest.raises(ValueError):
        state_prices(100.0, 1.0, [100.0, 110.0], [5.0, 5.0])
    # a flat tail at price zero is fine: all mass below the strikes
    x, q = state_prices(100.0, 1.0, [100.0, 110.0], [0.0, 0.0])
    assert np.all(x <= 100.0) and q.sum() == pytest.approx(1.0)


def test_piecewise_linear_payoffs_are_replicated_exactly():
    fly = lambda s: np.maximum(s - 90, 0) - 2 * np.maximum(s - 100, 0) + np.maximum(s - 110, 0)
    p = replicate_payoff(fly, [80.0, 90.0, 100.0, 110.0, 120.0], kappa=100.0)
    x = np.linspace(0, 250, 2501)
    assert np.allclose(p.payoff(x), fly(x))


def test_interpolation_error_bound():
    # f_h = f at the nodes and |f - f_h| <= h^2 max|f''| / 8 inside the grid; for x^2 the
    # maximum is exactly h^2 / 4 (f'' = 2), attained at the midpoints.
    for h in (20.0, 5.0, 1.0):
        K = np.arange(40.0, 200.0 + h / 2, h)
        p = replicate_payoff(lambda s: s**2, K, kappa=100.0)
        assert np.allclose(p.payoff(K), K**2)
        x = np.linspace(40, 200, 16001)
        assert np.max(np.abs(p.payoff(x) - x**2)) == pytest.approx(h**2 / 4, rel=1e-6)
        q = replicate_payoff(np.log, K, kappa=100.0)
        assert np.max(np.abs(q.payoff(x) - np.log(x))) <= h**2 / 8 / 40.0**2 + 1e-12


def test_carr_madan_identity():
    f, df, d2f = np.log, lambda s: 1 / s, lambda s: -1 / s**2
    x = np.array([5.0, 50.0, 99.0, 100.0, 140.0, 400.0])
    for kappa in (60.0, 100.0, 250.0):
        assert np.allclose(carr_madan(f, df, d2f, x, kappa), np.log(x), atol=1e-9)
    g, dg, d2g = (lambda s: np.exp(-s / 50)), (lambda s: -np.exp(-s / 50) / 50), \
                 (lambda s: np.exp(-s / 50) / 2500)
    assert np.allclose(carr_madan(g, dg, d2g, x, 100.0), g(x), atol=1e-9)


def test_strip_prices_match_expectations():
    # Pricing with a strip (corollary): the price of S_T^2 is B E[S_T^2] = B F^2 e^{v^2}
    # and the price of -log(S_T / F) is B v^2 / 2, both under the lognormal pricing measure.
    F = S0 / B
    K = np.arange(1.0, 1200.0, 0.5)
    call = lambda k: float(lognormal_call(S0, B, k, V))
    put = lambda k: float(lognormal_put(S0, B, k, V))
    sq = replicate_payoff(lambda s: s**2, K, kappa=F)
    assert sq.cost(S0, B, call, put) == pytest.approx(B * F**2 * np.exp(V**2), rel=1e-4)
    lg = replicate_payoff(lambda s: -np.log(s / F), K, kappa=F)
    assert lg.cost(S0, B, call, put) == pytest.approx(B * V**2 / 2, rel=2e-3)


def test_breeden_litzenberger():
    # C'(K) = -B Q(S_T > K) = -B N(d-) and C''(K) = B q_T(K)
    h = 1e-2
    for K in (70.0, 100.0, 130.0):
        c = lognormal_call(S0, B, np.array([K - h, K, K + h]), V)
        d_minus = (np.log(S0 / (B * K)) - V**2 / 2) / V
        assert (c[2] - c[0]) / (2 * h) == pytest.approx(-B * ndtr(d_minus), rel=1e-6)
        assert (c[0] - 2 * c[1] + c[2]) / h**2 == pytest.approx(
            B * lognormal_density(K, S0, B, V), rel=1e-4)


def test_portfolio_cost_needs_put_prices():
    p = Portfolio(bond=1.0).add_put(100.0, 1.0)
    with pytest.raises(ValueError):
        p.cost(S0, B, {})
