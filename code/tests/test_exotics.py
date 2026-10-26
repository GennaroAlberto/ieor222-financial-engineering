"""Checks of the Chapter 13 code against the results proved in the notes."""
import numpy as np
import pytest
from scipy.integrate import quad, trapezoid

from ieor222 import exotics as ex
from ieor222.binomial import path_dependent
from ieor222.blackscholes import call, norm_cdf

S0, K, T, r, SIG = 100.0, 100.0, 1.0, 0.05, 0.2


# ---------------------------------------------------------------- reflection principle

def test_reflected_walk_is_brownian():
    # thm:exo-reflection (i), on a grid: reflecting a Gaussian walk after a stopping time keeps its law.
    t = np.linspace(0, 1, 51)
    rng = np.random.default_rng(0)
    W = np.concatenate([np.zeros((40_000, 1)), np.cumsum(rng.normal(0, np.sqrt(0.02), (40_000, 50)), axis=1)], axis=1)
    R = ex.reflect(W, 0.4)
    assert np.mean(np.any(R != W, axis=1)) > 0.5          # most paths were reflected
    i, j = 15, 40                                         # s = 0.3, t = 0.8
    assert np.mean(R[:, j]) == pytest.approx(0.0, abs=0.015)
    assert np.mean(R[:, i] * R[:, j]) == pytest.approx(0.3, abs=0.015)
    assert np.var(R[:, j]) == pytest.approx(0.8, abs=0.025)


def test_law_of_the_maximum():
    m = np.linspace(0, 3, 13)
    assert np.allclose(ex.max_tail(m, 2.0), 2 * norm_cdf(-m / np.sqrt(2.0)))   # M_T has the law of |W_T|
    for nu in (-0.4, 0.0, 0.7):
        tail = quad(lambda x: ex.max_density(x, 2.0, nu), 0.5, np.inf)[0]
        assert tail == pytest.approx(ex.max_tail(0.5, 2.0, nu), rel=1e-7)
        assert ex.max_tail(0.0, 2.0, nu) == pytest.approx(1.0)


def test_joint_law_with_drift():
    # P(M >= m, X_T <= w) from the formula and from the joint density, for w <= m.
    T_, nu, m, w = 1.5, 0.35, 0.8, 0.2
    a = np.linspace(m, m + 12, 2401)
    x = np.linspace(w - 12, w, 2401)
    A, X = np.meshgrid(a, x, indexing="ij")
    integral = trapezoid(trapezoid(ex.joint_density(A, X, T_, nu), x, axis=1), a)
    assert integral == pytest.approx(ex.joint_tail(m, w, T_, nu), rel=1e-4)


def test_maximum_by_simulation():
    # Exact simulation of the maximum between grid points (rem:exo-bridge) against the closed form.
    rng = np.random.default_rng(1)
    nu, h, n = 0.3, 1.0 / 20, 20
    X = ex.log_paths(1.0, n, 100_000, nu, seed=2)
    M = ex.bridge_max_sample(X[:, :-1], X[:, 1:], h, rng).max(axis=1)
    for m in (0.25, 0.75, 1.5):
        assert np.mean(M >= m) == pytest.approx(ex.max_tail(m, 1.0, nu), abs=0.005)
    # the maximum over the grid alone is smaller
    assert np.mean(X.max(axis=1) >= 0.75) < ex.max_tail(0.75, 1.0, nu) - 0.02


# ---------------------------------------------------------------- barriers

def _down_and_in_quadrature(x, B):
    """e^{-rT} E[(S_T - K)^+ 1{min S <= B}] by quadrature, from the reflection identity for the minimum.

    Paths ending below a = log(B/x)/sigma have crossed; for those ending above, thm:exo-reflection (iii)
    gives E[h(X_T) 1{min X <= a}] = e^{2 nu a} E[h(2a - W_T + nu T)] with h vanishing below a.
    """
    nu, a = ex.drift(r, SIG), np.log(B / x) / SIG
    pay = lambda y: max(x * np.exp(SIG * y) - K, 0.0)
    dens = lambda w: np.exp(-w * w / (2 * T)) / np.sqrt(2 * np.pi * T)
    below = quad(lambda w: pay(w + nu * T) * dens(w), -np.inf, a - nu * T, limit=200)[0]
    image = quad(lambda w: pay(2 * a - w + nu * T) * dens(w), -np.inf, a + nu * T, limit=200)[0]
    return np.exp(-r * T) * (below + np.exp(2 * nu * a) * image)


def test_barrier_limits_and_parity():
    c = call(S0, K, T, r, SIG)
    assert ex.up_and_out_call(S0, K, 1e6, T, r, SIG) == pytest.approx(c, rel=1e-9)     # barrier never reached
    assert ex.down_and_out_call(S0, K, 1e-6, T, r, SIG) == pytest.approx(c, rel=1e-9)
    assert ex.up_and_out_call(S0, K, 100.0, T, r, SIG) == 0.0                          # knocked out at once
    assert ex.up_and_out_call(S0, 130.0, 120.0, T, r, SIG) == 0.0                      # K >= B
    assert ex.up_and_out_call(119.999999, K, 120.0, T, r, SIG) == pytest.approx(0.0, abs=1e-4)
    for B in (80.0, 95.0, 105.0):
        # in--out parity checked against an independent computation of the down-and-in price
        assert ex.down_and_in_call(110.0, K, B, T, r, SIG) == pytest.approx(_down_and_in_quadrature(110.0, B), rel=1e-7)
    # the two formulas of prop:exo-barrier (ii) meet at B = K
    assert ex.down_and_out_call(110.0, K, K - 1e-9, T, r, SIG) == pytest.approx(
        ex.down_and_out_call(110.0, K, K + 1e-9, T, r, SIG), abs=1e-6)


def test_up_and_out_by_integrating_the_joint_density():
    nu = ex.drift(r, SIG)
    B = 125.0
    k, b = np.log(K / S0) / SIG, np.log(B / S0) / SIG
    a = np.linspace(0, b, 1601)
    w = np.linspace(k, b, 1601)
    A, Wg = np.meshgrid(a, w, indexing="ij")
    f = ex.joint_density(A, Wg, T, nu) * np.maximum(S0 * np.exp(SIG * Wg) - K, 0.0)
    integral = np.exp(-r * T) * trapezoid(trapezoid(f, w, axis=1), a)
    assert integral == pytest.approx(ex.up_and_out_call(S0, K, B, T, r, SIG), rel=2e-3)


@pytest.mark.parametrize("up,B", [(True, 120.0), (False, 90.0), (False, 105.0)])
def test_barrier_closed_form_against_continuous_monte_carlo(up, B):
    x = 100.0 if up else 110.0
    exact = (ex.up_and_out_call if up else ex.down_and_out_call)(x, K, B, T, r, SIG)
    est, se = ex.barrier_call_mc(x, K, B, T, r, SIG, up=up, n_monitor=50, n_paths=60_000, continuous=True, seed=3)
    assert abs(est - exact) < 4 * se + 1e-3


def test_discrete_monitoring_and_bgk_shift():
    # Discrete monitoring raises the up-and-out price; the shifted barrier removes most of the gap.
    B, m = 120.0, 52
    cont = ex.up_and_out_call(S0, K, B, T, r, SIG)
    shifted = ex.up_and_out_call(S0, K, ex.bgk_barrier(B, SIG, T, m, up=True), T, r, SIG)
    est, se = ex.barrier_call_mc(S0, K, B, T, r, SIG, up=True, n_monitor=m, n_paths=60_000, seed=4)
    assert est > cont + 0.2
    assert abs(est - shifted) < 0.15 * (est - cont)
    assert ex.BGK_BETA == pytest.approx(0.5825971579390106, rel=1e-12)


# ---------------------------------------------------------------- lookback

@pytest.mark.parametrize("rate", [0.05, 0.0, -0.03])
def test_lookback_against_quadrature(rate):
    # E[max S] = S0 (1 + int_0^oo sigma e^{sigma m} Q(M >= m) dm), the first step of the proof.
    nu = ex.drift(rate, SIG)
    Emax = S0 * (1 + quad(lambda m: SIG * np.exp(SIG * m) * ex.max_tail(m, T, nu), 0, 30)[0])
    assert ex.lookback_put(S0, T, rate, SIG) == pytest.approx(np.exp(-rate * T) * Emax - S0, rel=1e-8)


def test_lookback_rate_zero_is_the_limit():
    assert ex.lookback_put(S0, T, 0.0, SIG) == pytest.approx(ex.lookback_put(S0, T, 1e-8, SIG), abs=1e-5)


def test_lookback_against_monte_carlo():
    est, se = ex._lookback_put_mc(S0, T, r, SIG, n_monitor=50, n_paths=60_000, continuous=True, seed=5)
    assert abs(est - ex.lookback_put(S0, T, r, SIG)) < 4 * se


# ---------------------------------------------------------------- Asian options

def test_geometric_laws():
    m, v = ex.geometric_law(T, r, SIG)
    assert (m, v) == pytest.approx(((r - SIG**2 / 2) * T / 2, SIG**2 * T / 3))
    for n in (1, 3, 12):
        t = T * np.arange(1, n + 1) / n
        assert ex.geometric_law(T, r, SIG, n) == pytest.approx(ex.weighted_geometric_law(t, np.full(n, 1 / n), r, SIG))
    assert ex.geometric_law(T, r, SIG, 1) == pytest.approx(((r - SIG**2 / 2) * T, SIG**2 * T))   # n = 1: S_T
    assert ex.geometric_asian_call(S0, K, T, r, SIG, 1) == pytest.approx(call(S0, K, T, r, SIG))
    assert ex.geometric_asian_call(S0, K, T, r, SIG, 10**6) == pytest.approx(ex.geometric_asian_call(S0, K, T, r, SIG), rel=1e-5)


def test_discrete_geometric_against_monte_carlo():
    n = 12
    X = ex.log_paths(T, n, 200_000, ex.drift(r, SIG), seed=6)
    pay = np.exp(-r * T) * np.maximum(S0 * np.exp(SIG * X[:, 1:].mean(axis=1)) - K, 0.0)
    assert abs(pay.mean() - ex.geometric_asian_call(S0, K, T, r, SIG, n)) < 4 * pay.std() / np.sqrt(pay.size)


def test_vecer_gamma():
    assert ex.vecer_gamma(T, T, r) == 0.0
    assert ex.vecer_gamma(0.0, T, 0.0) == 1.0
    assert ex.vecer_gamma(0.3, T, 1e-9) == pytest.approx(ex.vecer_gamma(0.3, T, 0.0))


# Continuously averaged Asian calls: the benchmark cases of V. Linetsky, Operations Research 52 (2004).
BENCHMARKS = [(0.02, 0.10, 1, 2.0, 2.0, 0.0559860), (0.18, 0.30, 1, 2.0, 2.0, 0.2183875),
              (0.0125, 0.25, 2, 2.0, 2.0, 0.1722687), (0.05, 0.50, 1, 1.9, 2.0, 0.1931737),
              (0.05, 0.50, 1, 2.0, 2.0, 0.2464156), (0.05, 0.50, 1, 2.1, 2.0, 0.3062203),
              (0.05, 0.50, 2, 2.0, 2.0, 0.3500952)]


@pytest.mark.parametrize("rate,sigma,mat,spot,strike,value", BENCHMARKS)
def test_vecer_pde_against_benchmarks(rate, sigma, mat, spot, strike, value):
    assert ex.asian_call_vecer(spot, strike, mat, rate, sigma) == pytest.approx(value, abs=5e-6)


def test_vecer_against_monte_carlo_and_bounds():
    out = ex.asian_call_mc(S0, K, T, r, SIG, n_steps=100, n_paths=40_000, seed=7)
    pde = ex.asian_call_vecer(S0, K, T, r, SIG)
    assert out["se"] < out["plain_se"] / 10                      # the control variate works
    assert abs(out["price"] - pde) < 4 * out["se"] + 2e-3        # 2e-3: trapezoidal time step
    assert ex.geometric_asian_call(S0, K, T, r, SIG) < pde < call(S0, K, T, r, SIG)


def test_vecer_linear_region():
    # g(t, z) = z for z >= gamma(t): the option is surely exercised there.
    z, g = ex.vecer_solve(T, r, SIG, times=[0.0, 0.5])
    for t, gt in g.items():
        right = z >= ex.vecer_gamma(t, T, r)
        assert np.allclose(gt[right], z[right], atol=1e-6)


# ---------------------------------------------------------------- perpetual American put

def test_perpetual_put_smooth_fit_and_ode():
    Kp, rp, sp = 100.0, 0.05, 0.3
    L = ex.perpetual_put_boundary(Kp, rp, sp)
    assert L == pytest.approx(2 * rp * Kp / (2 * rp + sp**2))
    e = 1e-5
    left = (ex.perpetual_put(L, Kp, rp, sp) - ex.perpetual_put(L - e, Kp, rp, sp)) / e
    right = (ex.perpetual_put(L + e, Kp, rp, sp) - ex.perpetual_put(L, Kp, rp, sp)) / e
    assert left == pytest.approx(-1, abs=1e-4) and right == pytest.approx(-1, abs=1e-4)
    x, hx = 140.0, 1e-3
    v = lambda y: ex.perpetual_put(y, Kp, rp, sp)
    d1 = (v(x + hx) - v(x - hx)) / (2 * hx)
    d2 = (v(x + hx) - 2 * v(x) + v(x - hx)) / hx**2
    assert rp * x * d1 + 0.5 * sp**2 * x**2 * d2 - rp * v(x) == pytest.approx(0.0, abs=1e-5)
    xs = np.linspace(1, 300, 400)
    assert np.all(v(xs) >= np.maximum(Kp - xs, 0) - 1e-12)


def test_long_american_puts_approach_the_perpetual_put():
    Kp, rp, sp = 100.0, 0.05, 0.3
    prices = [ex.american_put_binomial(100.0, Kp, mat, rp, sp, n=600) for mat in (1.0, 10.0, 60.0)]
    perp = ex.perpetual_put(100.0, Kp, rp, sp)
    assert prices[0] < prices[1] < prices[2] < perp + 0.02
    assert prices[2] == pytest.approx(perp, abs=0.15)


# ---------------------------------------------------------------- course example

def test_asian_call_on_a_two_period_tree():
    # Course homework: S0 = 4, u = 2, d = 1/2, r = 1/4, payoff (S0 + S1 + S2)/3 - 3)^+.
    _, V, D = path_dependent(lambda P: np.maximum(P.mean(axis=1) - 3, 0.0), 4.0, 2.0, 0.5, 0.25, 2)
    assert V[0][0] == pytest.approx(1.44)
    assert D[0][0] == pytest.approx(5 / 9)
    assert V[1] == pytest.approx([2 / 15, 52 / 15]) and D[1] == pytest.approx([1 / 9, 1 / 3])
