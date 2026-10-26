"""Checks of the Chapter 12 code against the results proved in the notes."""
import numpy as np
import pytest

from ieor222.blackscholes import call, d_plus_minus, digital_call, norm_cdf, put
from ieor222.feynmankac import (bm_density, bs_price_by_heat, cameron_martin, density_at, digital_delta,
                                digital_hedge_integral, digital_hedge_variance, fk_monte_carlo,
                                gbm_density, heat_kernel, heat_solution, kolmogorov_residuals,
                                local_vol_mc, ou_density, pay_later_delta, pay_later_fee,
                                pay_later_value, price_at, prices_from_density, skew_vol,
                                solve_forward_pde, solve_pricing_pde)

K, T, r, SIGMA = 100.0, 1.0, 0.05, 0.2


def _trap(f, x):
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(x)))


# ------------------------------------------------------------ transition densities


def test_transition_densities_are_densities_with_the_right_mean():
    y = np.linspace(-12, 12, 20001)
    assert _trap(bm_density(0.3, 0.5, 1.3, y), y) == pytest.approx(1.0, abs=1e-10)
    yy = np.linspace(1e-3, 3000, 1_000_001)
    p = gbm_density(0.0, 100.0, 2.0, yy, 0.08, 0.3)
    assert _trap(p, yy) == pytest.approx(1.0, abs=1e-6)
    assert _trap(yy * p, yy) == pytest.approx(100.0 * np.exp(0.08 * 2.0), rel=1e-6)
    q = ou_density(0.0, 2.0, 1.0, y, 1.5, -0.5, 0.8)
    assert _trap(y * q, y) == pytest.approx(-0.5 + 2.5 * np.exp(-1.5), abs=1e-9)


@pytest.mark.parametrize("model", ["bm", "gbm", "ou"])
def test_kolmogorov_backward_and_forward_equations(model):
    # thm:fk-kolmogorov, checked on the explicit densities. Residuals are compared with the size
    # of the time derivative, the largest term of each equation.
    if model == "bm":
        dens, b, s = bm_density, (lambda t, x: 0.0 * x), (lambda t, x: 1.0 + 0.0 * x)
        pts, hs = [(0.2, 0.3, 1.0, 0.7), (0.0, -1.0, 0.5, 0.2)], None
    elif model == "gbm":
        mu, sig = 0.1, 0.3
        dens = lambda t, x, s_, y: gbm_density(t, x, s_, y, mu, sig)
        b, s = (lambda t, x: mu * x), (lambda t, x: sig * x)
        pts, hs = [(0.2, 100.0, 1.0, 110.0), (0.5, 80.0, 0.8, 95.0)], None
    else:
        kap, th, sg = 1.5, 0.04, 0.02
        dens = lambda t, x, s_, y: ou_density(t, x, s_, y, kap, th, sg)
        b, s = (lambda t, x: kap * (th - x)), (lambda t, x: sg + 0.0 * x)
        pts, hs = [(0.2, 0.05, 1.0, 0.045), (0.0, 0.0, 0.5, 0.03)], 1e-4
    for (t, x, s_, y) in pts:
        back, fwd = kolmogorov_residuals(dens, b, s, t, x, s_, y, h_space=hs)
        h = 1e-4
        scale_t = abs(dens(t + h, x, s_, y) - dens(t - h, x, s_, y)) / (2 * h) + dens(t, x, s_, y)
        scale_s = abs(dens(t, x, s_ + h, y) - dens(t, x, s_ - h, y)) / (2 * h) + dens(t, x, s_, y)
        assert abs(back) < 1e-4 * scale_t
        assert abs(fwd) < 1e-4 * scale_s


# ------------------------------------------------------------ the heat equation


def test_heat_kernel_solves_the_heat_equation():
    s, z, h = 0.7, 0.4, 1e-4
    ds = (heat_kernel(s + h, z) - heat_kernel(s - h, z)) / (2 * h)
    dzz = (heat_kernel(s, z + h) - 2 * heat_kernel(s, z) + heat_kernel(s, z - h)) / h**2
    assert ds == pytest.approx(0.5 * dzz, rel=1e-6)
    # u(s, z) = E phi(z + sqrt(s) Y) for phi(z) = z^2 is z^2 + s
    assert heat_solution(lambda w: w**2, s, z) == pytest.approx(z**2 + s, rel=1e-10)


@pytest.mark.parametrize("x", [70.0, 100.0, 135.0])
def test_heat_reduction_gives_the_black_scholes_prices(x):
    # prop:fk-heat: e^{-r tau} u(sigma^2 tau, log x + (r - sigma^2/2) tau) is the price.
    t = 0.3
    assert bs_price_by_heat(lambda y: np.maximum(y - K, 0), t, x, T, r, SIGMA) == pytest.approx(
        call(x, K, T - t, r, SIGMA), abs=1e-5)
    assert bs_price_by_heat(lambda y: np.maximum(K - y, 0), t, x, T, r, SIGMA) == pytest.approx(
        put(x, K, T - t, r, SIGMA), abs=1e-5)
    # a discontinuous payoff: the trapezoidal rule is only first order there
    assert bs_price_by_heat(lambda y: (y > K).astype(float), t, x, T, r, SIGMA) == pytest.approx(
        digital_call(x, K, T - t, r, SIGMA), abs=1e-3)


# ------------------------------------------------------------ pricing PDEs


def test_pricing_pde_with_constant_volatility_is_black_scholes():
    xs = np.array([70.0, 85.0, 100.0, 120.0, 140.0])
    vol = lambda t, x: SIGMA + 0.0 * x
    res = solve_pricing_pde(lambda x: np.maximum(x - K, 0), T, r, vol, 10.0, 1000.0, n_x=800, n_t=400)
    assert np.max(np.abs(price_at(res, xs) - call(xs, K, T, r, SIGMA))) < 2e-3
    res = solve_pricing_pde(lambda x: (x > K).astype(float), T, r, vol, 10.0, 1000.0, n_x=800, n_t=400)
    assert np.max(np.abs(price_at(res, xs) - digital_call(xs, K, T, r, SIGMA))) < 1e-2
    # time-dependent volatility: the price is Black--Scholes with the root-mean-square volatility
    vol_t = lambda t, x: (0.1 + 0.2 * t) + 0.0 * x
    rms = np.sqrt((0.1**2 + 0.1 * 0.2 + 0.2**2 / 3))          # (1/T) int_0^1 (0.1 + 0.2 t)^2 dt
    res = solve_pricing_pde(lambda x: np.maximum(x - K, 0), T, r, vol_t, 10.0, 1000.0, n_x=800, n_t=400)
    assert np.max(np.abs(price_at(res, xs) - call(xs, K, T, r, rms))) < 2e-3


def test_local_volatility_pde_agrees_with_monte_carlo():
    # cor:fk-pricing-pde: one PDE solve and a Monte Carlo average estimate the same number.
    rr, g = 0.03, (lambda x: np.maximum(x - K, 0))
    res = solve_pricing_pde(g, T, rr, skew_vol, 5.0, 2000.0, n_x=1600, n_t=800)
    for S0, seed in ((85.0, 1), (110.0, 2)):
        m = local_vol_mc(g, S0, T, rr, skew_vol, 100, 100_000, seed=seed)
        assert abs(m["mean"] - price_at(res, S0)) < 4 * m["se"] + 0.01
    # refining the grid changes the price by little
    fine = solve_pricing_pde(g, T, rr, skew_vol, 5.0, 2000.0, n_x=3200, n_t=1600)
    assert abs(price_at(fine, 100.0) - price_at(res, 100.0)) < 1e-3


def test_forward_equation_conserves_mass_and_prices_like_the_backward_equation():
    rr = 0.03
    fw = solve_forward_pde(100.0, T, rr, skew_vol, 5.0, 2000.0, n_x=1600, n_t=400, s0=0.005)
    y, q = fw["y"], fw["q"]
    assert _trap(q, y) == pytest.approx(1.0, abs=1e-8)
    assert _trap(np.exp(y) * q, y) == pytest.approx(100.0 * np.exp(rr * T), rel=1e-5)  # martingale
    Ks = np.array([80.0, 100.0, 125.0])
    back = [price_at(solve_pricing_pde(lambda x, k=k: np.maximum(x - k, 0), T, rr, skew_vol, 5.0, 2000.0,
                                       n_x=1600, n_t=800), 100.0) for k in Ks]
    assert np.max(np.abs(prices_from_density(fw, Ks, T, rr) - np.array(back).ravel())) < 1e-3
    # the interpolated density agrees with the grid values at grid points
    assert np.allclose(density_at(fw, fw["x"][400:1200]), fw["density"][400:1200], rtol=1e-10)


# ------------------------------------------------------------ digital and pay-later options


def test_digital_delta_is_a_discounted_transition_density():
    x, tau = np.array([80.0, 100.0, 125.0]), 0.25
    _, dm = d_plus_minus(x, K, tau, r, SIGMA)
    formula = np.exp(-r * tau) * np.exp(-0.5 * dm**2) / np.sqrt(2 * np.pi) / (SIGMA * x * np.sqrt(tau))
    assert np.allclose(digital_delta(x, K, tau, r, SIGMA), formula, rtol=1e-12)
    h = 1e-4
    fd = (digital_call(x + h, K, tau, r, SIGMA) - digital_call(x - h, K, tau, r, SIGMA)) / (2 * h)
    assert np.allclose(digital_delta(x, K, tau, r, SIGMA), fd, rtol=1e-6)
    # at the money it grows like 1 / (K sigma sqrt(2 pi tau))
    for tau in (1e-3, 1e-5):
        assert digital_delta(K, K, tau, r, SIGMA) * K * SIGMA * np.sqrt(2 * np.pi * tau) == pytest.approx(1.0, abs=0.01)


def test_pay_later_fee():
    P = pay_later_fee(100.0, K, T, r, SIGMA)
    dp, dm = d_plus_minus(100.0, K, T, r, SIGMA)
    assert P == pytest.approx(100.0 * np.exp(r * T) * norm_cdf(dp) / norm_cdf(dm) - K, rel=1e-12)
    assert P == pytest.approx(19.63196768, abs=1e-6)
    assert P >= np.exp(r * T) * call(100.0, K, T, r, SIGMA)
    # the zero-cost position has value zero at time 0, and its discounted payoff has mean zero under Q
    assert pay_later_value(100.0, K, T, r, SIGMA, P) == pytest.approx(0.0, abs=1e-12)
    ST = 100.0 * np.exp((r - 0.5 * SIGMA**2) * T + SIGMA * np.sqrt(T) * np.random.default_rng(5).normal(size=400_000))
    pay = (ST - K - P) * (ST > K)
    assert abs(pay.mean()) < 4 * pay.std() / np.sqrt(len(pay))
    # the hedge: near maturity and at the strike the delta is very negative
    assert pay_later_delta(K, K, 1e-4, r, SIGMA, P) < -30


def test_digital_hedge_is_square_integrable():
    # Ito isometry: E^Q int_0^T (e^{-rt} sigma S_t Delta_t)^2 dt = e^{-2rT} p (1 - p), and the
    # integral form (1/2pi) int_0^{pi/2} exp(-d^2 / (1 + sin u)) du = N(d)(1 - N(d)).
    for d in (-1.7, 0.0, 0.15, 2.5):
        assert digital_hedge_integral(d) == pytest.approx(norm_cdf(d) * (1 - norm_cdf(d)), rel=1e-10)
    _, dm = d_plus_minus(100.0, K, T, r, SIGMA)
    assert digital_hedge_variance(100.0, K, T, r, SIGMA) == pytest.approx(
        np.exp(-2 * r * T) * digital_hedge_integral(dm), rel=1e-10)


# ------------------------------------------------------------ a quadratic potential


def test_cameron_martin_formula_and_feynman_kac_monte_carlo():
    m = fk_monte_carlo(lambda t, x: 0 * x, lambda t, x: 1 + 0 * x, lambda x: np.ones_like(x), 0.0, 1.0,
                       200, 200_000, k=lambda t, x: x**2, seed=3)
    assert abs(m["mean"] - cameron_martin(1.0, 1.0)) < 4 * m["se"] + 2e-3
    assert cameron_martin(1.0, 1.0) == pytest.approx(np.cosh(np.sqrt(2.0)) ** -0.5, rel=1e-14)
    # moments of I = int_0^T W^2: E I = T^2 / 2 and Var I = T^4 / 3, from log E e^{-lam I}
    Tm, h = 1.3, 1e-4
    logL = lambda lam: np.log(cameron_martin(lam, Tm))
    assert -(logL(h) - logL(-h)) / (2 * h) == pytest.approx(Tm**2 / 2, rel=1e-6)
    assert (logL(h) - 2 * logL(0.0) + logL(-h)) / h**2 == pytest.approx(Tm**4 / 3, rel=1e-4)
    # with a nonzero start the formula is the Feynman--Kac solution u(0, x)
    m = fk_monte_carlo(lambda t, x: 0 * x, lambda t, x: 1 + 0 * x, lambda x: np.ones_like(x), 0.5, 1.0,
                       200, 200_000, k=lambda t, x: 0.5 * x**2, seed=4)
    assert abs(m["mean"] - cameron_martin(0.5, 1.0, x=0.5)) < 4 * m["se"] + 2e-3
    # negative lam: blow-up at w T = pi / 2
    assert np.isinf(cameron_martin(-0.5 * (np.pi / 2) ** 2, 1.0))
    assert np.isfinite(cameron_martin(-0.5 * (np.pi / 2 - 0.01) ** 2, 1.0))
