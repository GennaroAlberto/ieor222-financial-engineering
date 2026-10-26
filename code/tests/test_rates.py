"""Checks of the Chapter 14 code against the results proved in the notes and the exercises."""
import numpy as np
import pytest
from scipy.stats import norm

from ieor222.rates import (
    affine_B,
    affine_riccati,
    bootstrap_zeros,
    cir_A,
    cir_B,
    cir_bond,
    cir_euler,
    cir_exact,
    cir_long_yield,
    cir_mean,
    cir_stationary,
    cir_var,
    cir_yield,
    feller_ratio,
    forward_rates,
    gauss_B,
    gauss_B2_integral,
    gaussian_bond,
    ho_lee_beta,
    hull_white_beta,
    hull_white_bond,
    integrated_rate,
    jamshidian_coupon_option,
    mc_bond_prices,
    simple_rate_caplet,
    vasicek_A,
    vasicek_bond,
    vasicek_bond_option,
    vasicek_bond_option_vol,
    vasicek_euler,
    vasicek_exact,
    vasicek_forward,
    vasicek_forward_measure_rT,
    vasicek_integral_moments,
    vasicek_long_yield,
    vasicek_mean,
    vasicek_negative_prob,
    vasicek_shape_thresholds,
    vasicek_step_moments,
    vasicek_var,
    vasicek_yield,
    yields,
)

VAS = dict(kappa=0.15, theta=0.05, sigma=0.025)            # the parameters of the figures
CIR = dict(kappa=0.15, theta=0.05, sigma=0.025 / np.sqrt(0.05))
TAU = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])


# ---------------------------------------------------------------- the affine theorem

@pytest.mark.parametrize("beta1, alpha1", [(-0.4, 0.0), (0.3, 0.0), (0.0, 0.0), (-0.3, 0.02), (0.2, 0.05)])
def test_riccati_closed_form_solves_the_equation(beta1, alpha1):
    # Theorem (affine term structures), part (i): the closed form of B solves the Riccati equation
    _, B_num = affine_riccati(TAU, 0.0, beta1, 0.0, alpha1)
    assert np.allclose(affine_B(TAU, beta1, alpha1), B_num, rtol=1e-9, atol=1e-12)
    assert np.all(affine_B(TAU, beta1, alpha1) > 0)


def test_vasicek_and_cir_closed_forms_match_the_riccati_system():
    k, th, s = VAS.values()
    A, B = affine_riccati(TAU, k * th, -k, s**2, 0.0)
    assert np.allclose(A, vasicek_A(TAU, k, th, s), atol=1e-11)
    assert np.allclose(B, gauss_B(TAU, k), atol=1e-11)
    k, th, s = CIR.values()
    A, B = affine_riccati(TAU, k * th, -k, 0.0, s**2)
    assert np.allclose(A, cir_A(TAU, k, th, s), atol=1e-11)
    assert np.allclose(B, cir_B(TAU, k, s), atol=1e-11)


# ---------------------------------------------------------------- Vasicek

def test_vasicek_yields_forwards_and_limits():
    k, th, s = VAS.values()
    r = 0.03
    # Y -> r at the short end, Y -> R_inf at the long end
    assert vasicek_yield(r, 1e-7, k, th, s) == pytest.approx(r, abs=1e-8)
    assert vasicek_yield(r, 1e5, k, th, s) == pytest.approx(vasicek_long_yield(k, th, s), abs=1e-5)
    # f = -d_T log P, and f = E[r_T] - sigma^2 B^2 / 2
    h = 1e-5
    fd = -(np.log(vasicek_bond(r, TAU + h, k, th, s)) - np.log(vasicek_bond(r, TAU - h, k, th, s))) / (2 * h)
    assert np.allclose(vasicek_forward(r, TAU, k, th, s), fd, atol=1e-8)
    assert np.allclose(vasicek_forward(r, TAU, k, th, s),
                       vasicek_mean(TAU, r, k, th) - 0.5 * s**2 * gauss_B(TAU, k)**2)
    # the yield of the notes, R_inf + (r - R_inf) B / tau + sigma^2 B^2 / (4 kappa tau)
    R, B = vasicek_long_yield(k, th, s), gauss_B(TAU, k)
    assert np.allclose(vasicek_yield(r, TAU, k, th, s), R + (r - R) * B / TAU + s**2 * B**2 / (4 * k * TAU))


@pytest.mark.parametrize("k, th, s", [(0.15, 0.05, 0.025), (0.4, 0.05, 0.015), (1.0, 0.03, 0.05)])
def test_vasicek_yield_curve_shapes(k, th, s):
    # Vasicek (1977): increasing for r <= lo, decreasing for r >= hi, humped in between
    lo, hi = vasicek_shape_thresholds(k, th, s)
    assert hi == pytest.approx(th)                       # the upper threshold is theta
    tau = np.linspace(1e-4, 200.0, 20_001)
    for r in np.r_[np.linspace(lo - 0.02, hi + 0.02, 25), lo - 1e-4, lo + 1e-4, hi - 1e-4, hi + 1e-4]:
        d = np.diff(vasicek_yield(r, tau, k, th, s))
        if r <= lo:
            assert np.all(d >= -1e-13)
        elif r >= hi:
            assert np.all(d <= 1e-13)
        else:
            sign = np.sign(d[np.abs(d) > 1e-13])
            assert sign[0] > 0 and sign[-1] < 0 and np.all(np.diff(sign) <= 0)   # one maximum


def test_vasicek_exact_sampler_has_the_right_law_and_prices_bonds():
    k, th, s = VAS.values()
    r0, T, n, M = 0.03, 10.0, 20, 100_000
    t, r, I = vasicek_exact(r0, k, th, s, T, n, n_paths=M, seed=1)
    assert r[:, -1].mean() == pytest.approx(vasicek_mean(T, r0, k, th), abs=4 * np.sqrt(vasicek_var(T, k, s) / M))
    assert r[:, -1].var() == pytest.approx(vasicek_var(T, k, s), rel=0.02)
    m, v = vasicek_integral_moments(r0, T, k, th, s)
    assert I[:, -1].mean() == pytest.approx(m, abs=4 * np.sqrt(v / M))
    assert I[:, -1].var() == pytest.approx(v, rel=0.02)
    P, se = mc_bond_prices(I)
    exact = vasicek_bond(r0, t, k, th, s)
    assert np.all(np.abs(P - exact) <= 4 * se + 1e-15)
    # one step: covariance of (r_{t+h}, int r) is sigma^2 B(h)^2 / 2
    _, r1, I1 = vasicek_exact(r0, k, th, s, 2.0, 1, n_paths=200_000, seed=2)
    vx, vi, c = vasicek_step_moments(2.0, k, s)
    assert np.cov(r1[:, 1], I1[:, 1])[0, 1] == pytest.approx(c, rel=0.03)
    assert c == pytest.approx(0.5 * s**2 * gauss_B(2.0, k)**2)


def test_vasicek_bond_price_from_the_gaussian_integral():
    # Exercise: P = exp(-m + v/2) with the conditional mean and variance of int_t^T r
    k, th, s = VAS.values()
    for r in (-0.01, 0.03, 0.09):
        m, v = vasicek_integral_moments(r, TAU, k, th, s)
        assert np.allclose(gaussian_bond(m, v), vasicek_bond(r, TAU, k, th, s), rtol=1e-12)
    # the convexity gap v / (2 tau) increases to sigma^2 / (2 kappa^2)
    _, v = vasicek_integral_moments(0.03, np.array([1.0, 10.0, 30.0, 1e5]), k, th, s)
    gap = v / (2 * np.array([1.0, 10.0, 30.0, 1e5]))
    assert np.all(np.diff(gap) > 0) and gap[-1] == pytest.approx(s**2 / (2 * k**2), rel=1e-3)


def test_vasicek_negative_rate_probability():
    k, th, s, r0 = 0.2, 0.03, 0.025, 0.01
    t, r, _ = vasicek_exact(r0, k, th, s, 5.0, 5, n_paths=200_000, seed=3)
    for j in (1, 3, 5):
        p = vasicek_negative_prob(t[j], r0, k, th, s)
        assert np.mean(r[:, j] < 0) == pytest.approx(p, abs=4 * np.sqrt(p * (1 - p) / 200_000))
    # stationary limit N(-theta sqrt(2 kappa) / sigma), as in the caption of the figure
    assert vasicek_negative_prob(500.0, r0, k, th, s) == pytest.approx(norm.cdf(-th * np.sqrt(2 * k) / s))
    assert norm.cdf(-th * np.sqrt(2 * k) / s) == pytest.approx(0.224, abs=5e-4)
    assert vasicek_long_yield(k, th, 0.05) == pytest.approx(-0.00125)


def test_hjm_drift_condition_holds_in_vasicek():
    # f(t,T) = theta + e^{-kappa tau}(r - theta) - sigma^2 B^2/2 is affine in r, so its Q-drift is
    # d_t f + d_r f * kappa (theta - r); HJM: it equals sigma_f int sigma_f = sigma^2 e^{-kappa tau} B(tau)
    k, th, s = VAS.values()
    h = 1e-6
    for r in (0.0, 0.03, 0.08):
        dt_f = -(vasicek_forward(r, TAU + h, k, th, s) - vasicek_forward(r, TAU - h, k, th, s)) / (2 * h)
        drift = dt_f + np.exp(-k * TAU) * k * (th - r)
        assert np.allclose(drift, s**2 * np.exp(-k * TAU) * gauss_B(TAU, k), atol=1e-9)


# ---------------------------------------------------------------- Ho--Lee and Hull--White

def market_curve():
    f0 = lambda t: 0.02 + 0.015 * (1 - np.exp(-0.4 * np.asarray(t))) + 0.002 * np.asarray(t) * np.exp(-0.2 * np.asarray(t))
    df = lambda t: (0.006 * np.exp(-0.4 * np.asarray(t)) + 0.002 * np.exp(-0.2 * np.asarray(t))
                    - 0.0004 * np.asarray(t) * np.exp(-0.2 * np.asarray(t)))
    PM = lambda T: np.exp(-np.array([0.02 * x + 0.015 * (x - (1 - np.exp(-0.4 * x)) / 0.4)
                                     + 0.002 * (25 * (1 - np.exp(-0.2 * x)) - 5 * x * np.exp(-0.2 * x)) for x in T]))
    return f0, df, PM


def test_ho_lee_and_hull_white_fit_the_initial_curve():
    f0, df, PM = market_curve()
    T = np.array([0.5, 1.0, 3.0, 7.0, 15.0])
    sigma = 0.012
    # Proposition (Ho--Lee): beta = f' + sigma^2 t and r0 = f(0) reproduce P^M(0, T)
    beta = lambda t: ho_lee_beta(t, df, sigma)
    assert np.allclose(hull_white_bond(f0(0.0), 0.0, T, beta, 0.0, sigma), PM(T), rtol=1e-9)
    # Exercise (Hull--White): beta = f' + kappa f + sigma^2 (1 - e^{-2 kappa t}) / (2 kappa)
    for kappa in (0.05, 0.3, 1.0):
        beta = lambda t, k=kappa: hull_white_beta(t, f0, df, k, sigma)
        assert np.allclose(hull_white_bond(f0(0.0), 0.0, T, beta, kappa, sigma), PM(T), rtol=1e-9)
    # as kappa -> 0 the Hull--White drift tends to the Ho--Lee drift
    t = np.linspace(0, 10, 11)
    assert np.allclose(hull_white_beta(t, f0, df, 1e-7, sigma), ho_lee_beta(t, df, sigma), atol=1e-8)


def test_hull_white_with_constant_drift_is_vasicek():
    k, th, s = VAS.values()
    r, t = 0.04, 1.5
    T = t + TAU
    assert np.allclose(hull_white_bond(r, t, T, lambda u: k * th, k, s), vasicek_bond(r, TAU, k, th, s), rtol=1e-10)


def test_course_gaussian_rate_models():
    # IEOR 222 homework (Gaussian rate r_t = r0 + rho t - sigma W_t) and midterm
    # (r_t = r0 + alpha t + 2 sigma t W_t): closed forms against Monte Carlo on Riemann sums.
    rng = np.random.default_rng(4)
    T, n, M = 2.0, 400, 40_000
    h = T / n
    t = np.linspace(0, T, n + 1)
    W = np.concatenate([np.zeros((M, 1)), np.cumsum(rng.normal(0, np.sqrt(h), (M, n)), axis=1)], axis=1)
    trap = lambda X: (0.5 * (X[:, 1:] + X[:, :-1]) * h).sum(axis=1)
    r0, rho, sigma = 0.03, 0.01, 0.02
    exact = np.exp(-r0 * T - rho * T**2 / 2 + sigma**2 * T**3 / 6)
    D = np.exp(-trap(r0 + rho * t - sigma * W))
    assert D.mean() == pytest.approx(exact, abs=4 * D.std() / np.sqrt(M) + 1e-5)
    assert gaussian_bond(r0 * T + rho * T**2 / 2, sigma**2 * T**3 / 3) == pytest.approx(exact)
    r0, alpha, sigma = 0.03, 0.01, 0.02
    exact = np.exp(-r0 * T - alpha * T**2 / 2 + 4 * sigma**2 * T**5 / 15)
    D = np.exp(-trap(r0 + alpha * t + 2 * sigma * t * W))
    assert D.mean() == pytest.approx(exact, abs=4 * D.std() / np.sqrt(M) + 1e-5)
    # with constant drift the Ho--Lee bond eventually exceeds one: the cubic term wins
    r0, rho, sigma = 0.03, 0.01, 0.02
    assert gaussian_bond(r0 * 30 + rho * 30**2 / 2, sigma**2 * 30**3 / 3) < 1
    assert gaussian_bond(r0 * 100 + rho * 100**2 / 2, sigma**2 * 100**3 / 3) > 1


# ---------------------------------------------------------------- CIR

def test_cir_closed_form_solves_the_riccati_equations_and_limits():
    k, th, s = CIR.values()
    h = 1e-6
    dB = (cir_B(TAU + h, k, s) - cir_B(TAU - h, k, s)) / (2 * h)
    dA = (cir_A(TAU + h, k, th, s) - cir_A(TAU - h, k, th, s)) / (2 * h)
    B = cir_B(TAU, k, s)
    assert np.allclose(dB, 1 - k * B - 0.5 * s**2 * B**2, atol=1e-8)
    assert np.allclose(dA, -k * th * B, atol=1e-8)
    assert cir_yield(0.03, 1e-7, k, th, s) == pytest.approx(0.03, abs=1e-8)
    assert cir_yield(0.03, 1e5, k, th, s) == pytest.approx(cir_long_yield(k, th, s), abs=1e-5)
    assert cir_long_yield(k, th, s) == pytest.approx(0.0408, abs=5e-5)
    assert feller_ratio(k, th, s) == pytest.approx(1.2)


def test_cir_exact_sampler_moments_and_positivity():
    # Exercise (CIR moments): mean and variance; stationary gamma law
    k, th, s, r0 = 0.5, 0.04, 0.1, 0.02
    t, r = cir_exact(r0, k, th, s, 4.0, 4, n_paths=200_000, seed=5)
    assert np.all(r >= 0)
    for j in (1, 4):
        sd = np.sqrt(cir_var(t[j], r0, k, th, s))
        assert r[:, j].mean() == pytest.approx(cir_mean(t[j], r0, k, th), abs=4 * sd / np.sqrt(2e5))
        assert r[:, j].var() == pytest.approx(cir_var(t[j], r0, k, th, s), rel=0.02)
    assert cir_var(1.0, r0, k, th, s) == pytest.approx(1.5739e-4, rel=1e-4)
    shape, scale = cir_stationary(k, th, s)
    assert shape * scale == pytest.approx(th) and shape * scale**2 == pytest.approx(cir_var(1e3, r0, k, th, s))
    # with nu < 1 the exact sampler visits the neighbourhood of 0 often
    _, r = cir_exact(0.04, 0.5, 0.04, np.sqrt(0.08), 5.0, 500, n_paths=2_000, seed=6)
    assert np.mean(r.min(axis=1) < 1e-4) > 0.5


def test_cir_monte_carlo_bond_prices():
    k, th, s = CIR.values()
    t, r = cir_exact(0.03, k, th, s, 10.0, 100, n_paths=60_000, seed=7)
    P, se = mc_bond_prices(integrated_rate(t, r))
    exact = cir_bond(0.03, t, k, th, s)
    assert np.all(np.abs(P - exact) <= 4 * se + 2e-5)


def test_course_homework_vasicek_versus_cir_simulation():
    # IEOR 222 homework: alpha = 1, beta = 0.1, sigma = 0.3, T = 10, 10^4 steps, 10^3 paths
    # (r0 = beta, which the homework leaves unspecified). Vasicek goes negative, CIR does not,
    # and both averages at T are close to the long-run level beta.
    a, b, s, T, n, M = 1.0, 0.1, 0.3, 10.0, 10_000, 1_000
    dW = np.random.default_rng(8).normal(0, np.sqrt(T / n), (M, n))
    _, rv = vasicek_euler(b, a, b, s, T, n, dW=dW)
    _, rc = cir_euler(b, a, b, s, T, n, dW=dW)
    assert np.mean(rv[:, -1] < 0) > 0.25
    assert rv[:, -1].mean() == pytest.approx(b, abs=4 * rv[:, -1].std() / np.sqrt(M))
    assert rc[:, -1].mean() == pytest.approx(b, abs=4 * rc[:, -1].std() / np.sqrt(M))
    assert np.mean(rc < 0) < 1e-3 and feller_ratio(a, b, s) >= 1


# ---------------------------------------------------------------- forward measure and options

def test_forward_measure_and_bond_options():
    k, th, s = VAS.values()
    r0, T, S, M = 0.03, 1.0, 5.0, 200_000
    # E^{Q^T}[r_T] = f(0,T), by reweighting Q-paths with D_T / P(0,T)
    _, r, I = vasicek_exact(r0, k, th, s, T, 1, n_paths=M, seed=9)
    w = np.exp(-I[:, -1]) / vasicek_bond(r0, T, k, th, s)
    assert np.mean(w * r[:, -1]) == pytest.approx(vasicek_forward(r0, T, k, th, s), abs=2e-4)
    rT = vasicek_forward_measure_rT(r0, T, k, th, s, M, seed=10)
    assert rT.mean() == pytest.approx(vasicek_forward(r0, T, k, th, s), abs=4 * rT.std() / np.sqrt(M))
    PTS_q, PTS_f = vasicek_bond(r[:, -1], S - T, k, th, s), vasicek_bond(rT, S - T, k, th, s)
    PT, PS = vasicek_bond(r0, T, k, th, s), vasicek_bond(r0, S, k, th, s)
    assert PTS_f.mean() == pytest.approx(PS / PT, abs=4 * PTS_f.std() / np.sqrt(M))   # forward price
    for K in (0.8, 0.85, 0.9):
        exact = vasicek_bond_option(K, T, S, r0, k, th, s)
        a = np.exp(-I[:, -1]) * np.maximum(PTS_q - K, 0)
        b = PT * np.maximum(PTS_f - K, 0)
        assert a.mean() == pytest.approx(exact, abs=4 * a.std() / np.sqrt(M))
        assert b.mean() == pytest.approx(exact, abs=4 * b.std() / np.sqrt(M))
        put = vasicek_bond_option(K, T, S, r0, k, th, s, call=False)
        assert exact - put == pytest.approx(PS - K * PT, abs=1e-14)
    # Sigma for expiry 1 on the 5-year bond (the default of the notebook's bond-option demo)
    assert vasicek_bond_option_vol(T, S, k, s) == pytest.approx(0.0699, abs=5e-5)
    assert vasicek_bond_option_vol(T, S, k, s) == pytest.approx(s * gauss_B(S - T, k) * np.sqrt(vasicek_var(T, k, 1.0)))


def test_jamshidian_decomposition_and_price():
    k, th, s = VAS.values()
    r0, T, pay, cf, K = 0.03, 1.0, np.array([2.0, 3.0, 4.0]), np.array([5.0, 5.0, 105.0]), 100.0
    price, rs, Ks = jamshidian_coupon_option(K, T, pay, cf, r0, k, th, s)
    assert np.dot(cf, Ks) == pytest.approx(K)
    x = np.linspace(-0.1, 0.2, 301)
    zc = vasicek_bond(x[:, None], pay - T, k, th, s)
    assert np.allclose(np.maximum(zc @ cf - K, 0), np.maximum(zc - Ks, 0) @ cf, atol=1e-10)
    assert price == pytest.approx(4.6895, abs=5e-5) and rs == pytest.approx(0.04932, abs=5e-6)
    _, r, I = vasicek_exact(r0, k, th, s, T, 1, n_paths=200_000, seed=2026)
    payoff = np.exp(-I[:, -1]) * np.maximum(vasicek_bond(r[:, -1:], pay - T, k, th, s) @ cf - K, 0)
    assert payoff.mean() == pytest.approx(price, abs=4 * payoff.std() / np.sqrt(len(payoff)))
    put, _, _ = jamshidian_coupon_option(K, T, pay, cf, r0, k, th, s, call=False)
    bond0 = np.dot(cf, vasicek_bond(r0, pay, k, th, s))
    assert price - put == pytest.approx(bond0 - K * vasicek_bond(r0, T, k, th, s), abs=1e-10)


def test_caplet_is_a_bond_put():
    k, th, s = VAS.values()
    r0, T, d, kk = 0.03, 1.0, 0.5, 0.04
    cap = simple_rate_caplet(kk, T, d, r0, k, th, s)
    assert cap == pytest.approx(0.0029204, abs=5e-8)
    _, r, I = vasicek_exact(r0, k, th, s, T, 1, n_paths=200_000, seed=11)
    P = vasicek_bond(r[:, -1], d, k, th, s)
    L = (1 / P - 1) / d
    payoff = np.exp(-I[:, -1]) * P * d * np.maximum(L - kk, 0)     # value at T of the payment at T + d
    assert payoff.mean() == pytest.approx(cap, abs=4 * payoff.std() / np.sqrt(len(payoff)))


# ---------------------------------------------------------------- curves

def test_bootstrap_reprices_the_bonds():
    c, p = [0.02, 0.03, 0.04], [99.20, 99.60, 100.30]
    P = bootstrap_zeros(c, p)
    for n in range(3):
        assert 100 * (c[n] * P[: n + 1].sum() + P[n]) == pytest.approx(p[n])
    assert P == pytest.approx([0.97255, 0.93866, 0.89091], abs=5e-6)
    T = np.array([1.0, 2.0, 3.0])
    Y = yields(P, T)
    F = forward_rates(np.r_[1.0, P], np.r_[0.0, T])
    assert np.allclose(T * Y, np.cumsum(F))                       # k Y(0,k) = sum of forwards
    assert 100 * (0.05 * P.sum() + P[-1]) == pytest.approx(103.10, abs=5e-3)


def test_term_premium_formula():
    # Exercise: TP(T) = (R_inf - theta_P)(1 - B/T) + sigma^2 B^2 / (4 kappa T), independent of r0
    k, thP, s, lam = 0.3, 0.04, 0.01, -0.3
    th = thP - s * lam / k
    T = np.array([1.0, 10.0, 30.0])
    B = gauss_B(T, k)
    tp = (vasicek_long_yield(k, th, s) - thP) * (1 - B / T) + s**2 * B**2 / (4 * k * T)
    for r0 in (0.0, 0.03, 0.07):
        direct = vasicek_yield(r0, T, k, th, s) - (thP + (r0 - thP) * B / T)
        assert np.allclose(direct, tp)
    assert tp[1] == pytest.approx(0.006537, abs=5e-7)
    assert gauss_B2_integral(T, k) == pytest.approx([(x - gauss_B(x, k) - k * gauss_B(x, k)**2 / 2) / k**2 for x in T])
