"""Checks of the Chapter 15 code against the results proved in the notes."""
import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from ieor222.blackscholes import call as bs_call_q0
from ieor222.blackscholes import digital_call
from ieor222 import volatility as V

HESTON = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
S0, R = 100.0, 0.02


# ------------------------------------------------------------ Black--Scholes in forward form


def test_normalized_black_matches_the_black_scholes_formula():
    K = np.array([60.0, 90.0, 100.0, 125.0, 200.0])
    for q in (0.0, 0.03):
        T, sig = 1.5, 0.3
        # with a dividend yield the price is the q = 0 price at the spot S0 e^{-qT} (prop:rn-dividends)
        expected = bs_call_q0(S0 * np.exp(-q * T), K, T, R, sig)
        assert np.allclose(V.bs_call(S0, K, T, R, sig, q), expected, rtol=1e-12, atol=1e-12)
        assert np.allclose(V.bs_call(S0, K, T, R, sig, q) - V.bs_put(S0, K, T, R, sig, q),
                           S0 * np.exp(-q * T) - K * np.exp(-R * T))
    # put--call symmetry b(k, s) - (1 - e^k) = e^k b(-k, s), used in prop:iv-mixing
    k = np.linspace(-2, 2, 41)
    for s in (0.05, 0.4, 2.0):
        assert np.allclose(V.black_normalized(k, s) - (1 - np.exp(k)), np.exp(k) * V.black_normalized(-k, s), atol=1e-14)


def test_limits_and_vega_of_prop_iv_existence():
    k = np.array([-0.5, 0.0, 0.7])
    # b(k, s) -> (1 - e^k)^+ as s -> 0 (at k = 0 the gap is s N'(0) to first order)
    assert np.allclose(V.black_normalized(k, 1e-8), np.maximum(1 - np.exp(k), 0), atol=1e-8)
    assert np.allclose(V.black_normalized(k, 60.0), 1.0, atol=1e-12)
    s, h = 0.37, 1e-6
    fd = (V.black_normalized(k, s + h) - V.black_normalized(k, s - h)) / (2 * h)
    assert np.allclose(fd, V.black_vega_normalized(k, s), atol=1e-8)


# ------------------------------------------------------------ implied volatility


def test_implied_volatility_round_trip_with_out_of_the_money_quotes():
    k = np.linspace(-3, 3, 61)
    for s in (0.01, 0.05, 0.2, 0.5, 1.0, 3.0, 6.0):
        call = V.black_normalized(k, s)
        put = np.exp(k) * V.black_normalized(-k, s)
        s_hat = np.where(k >= 0, V.implied_total_vol(call, k, "call"), V.implied_total_vol(put, k, "put"))
        fin = np.isfinite(s_hat)
        # nan only where the out-of-the-money price underflows to zero
        otm = V.black_normalized(np.abs(k), s)
        assert np.all(otm[~fin] == 0.0)
        assert np.allclose(s_hat[fin], s, rtol=1e-10)


def test_calls_and_puts_have_the_same_implied_vol_and_bounds_are_enforced():
    K, T, q = np.array([80.0, 100.0, 130.0]), 0.75, 0.01
    C = V.bs_call(S0, K, T, R, 0.27, q)
    P = V.bs_put(S0, K, T, R, 0.27, q)
    assert np.allclose(V.implied_vol(C, S0, K, T, R, q, "call"), 0.27, rtol=1e-10)
    assert np.allclose(V.implied_vol(P, S0, K, T, R, q, "put"), 0.27, rtol=1e-10)
    # prices outside the open interval eq:iv-bounds have no implied volatility
    lower = np.maximum(S0 * np.exp(-q * T) - K * np.exp(-R * T), 0.0)
    assert np.all(np.isnan(V.implied_vol(lower - 1e-6, S0, K, T, R, q)))
    assert np.all(np.isnan(V.implied_vol(np.full(3, S0 * np.exp(-q * T) + 1e-6), S0, K, T, R, q)))
    assert np.isnan(V.implied_vol(0.0, S0, 130.0, T, R, q))


def test_tehranchi_upper_bound_with_equality_at_the_forward():
    k = np.linspace(-2, 2, 81)
    for s in (0.05, 0.3, 1.0, 2.5):
        c = V.black_normalized(k, s)
        assert np.all(V.tehranchi_upper(c, k) >= s - 1e-12)
    for s in (0.05, 0.3, 1.0):
        assert V.tehranchi_upper(V.black_normalized(0.0, s), 0.0) == pytest.approx(s, rel=1e-10)


def test_worked_example_at_the_money():
    # Week 5A: a 3-month at-the-money option priced 36 on a spot of 1000 (r = q = 0). The rule of
    # Brenner and Subrahmanyam gives 36 / (0.4 * 1000 * 0.5) = 0.18; rem:iv-solver gives the exact value.
    exact = 2.0 / np.sqrt(0.25) * ndtri((1 + 36.0 / 1000.0) / 2)
    assert V.implied_vol(36.0, 1000.0, 1000.0, 0.25) == pytest.approx(exact, rel=1e-12)
    assert exact == pytest.approx(0.18054, abs=1e-5)


def test_week14a_bachelier_versus_black_scholes_at_the_money():
    # Week 14A (with the at-the-money-forward strike): 0 <= sigma_BS - sigma_N / S0 <= sigma_BS^3 T / 24
    for sig, T in ((0.1, 0.5), (0.2, 1.0), (0.5, 2.0), (0.8, 1.0)):
        C = V.bs_call(S0, S0, T, 0.0, sig)
        sigma_n = C / np.sqrt(T / (2 * np.pi))
        assert V.bachelier_call(S0, S0, T, sigma_n) == pytest.approx(C, rel=1e-12)
        gap = sig - sigma_n / S0
        assert 0 <= gap <= sig**3 * T / 24
    # a flat normal smile is a Black--Scholes skew, with the slope of ex:iv-bachelier at K = S0
    K = np.array([80.0, 100.0, 120.0])
    iv = V.implied_vol(V.bachelier_call(S0, K, 1.0, 20.0), S0, K, 1.0)
    assert iv[0] > iv[1] > iv[2]


# ------------------------------------------------------------ stylised facts


def test_mixture_kurtosis_and_uncorrelated_but_dependent_returns():
    # prop:iv-mixture: kurtosis 3 E[V^2] / E[V]^2 and Cov(r_i^2, r_j^2) = Cov(V_i, V_j)
    rng = np.random.default_rng(0)
    n = 400_000
    V1 = rng.choice([0.5, 2.0], size=n)
    V2 = np.where(rng.random(n) < 0.8, V1, rng.choice([0.5, 2.0], size=n))   # persistent variance
    r1 = np.sqrt(V1) * rng.standard_normal(n)
    r2 = np.sqrt(V2) * rng.standard_normal(n)
    assert V.kurtosis(r1) == pytest.approx(V.mixture_kurtosis([0.5, 2.0], [0.5, 0.5]), rel=0.03)
    assert abs(np.corrcoef(r1, r2)[0, 1]) < 0.01
    assert np.cov(r1**2, r2**2)[0, 1] == pytest.approx(np.cov(V1, V2)[0, 1], rel=0.08)
    assert V.mixture_kurtosis([1.0], [1.0]) == pytest.approx(3.0)


# ------------------------------------------------------------ static arbitrage


def test_density_formula_of_prop_iv_static_arbitrage():
    k = np.linspace(-1.5, 1.5, 3001)
    w, wk, wkk = V.svi_raw(k, a=0.02, b=0.1, rho=-0.5, m=0.0, sig=0.2)
    dk = k[1] - k[0]
    c = V.black_normalized(k, np.sqrt(w))
    ck = np.gradient(c, dk)
    ckk = np.gradient(ck, dk)
    fd = (ckk - ck) * np.exp(-k)            # density of log(S_T / F) from the prices
    assert np.max(np.abs(V.density_from_w(k, w, wk, wkk) - fd)[20:-20]) < 2e-4
    # flat smile: g = 1 and the lognormal density
    flat = np.full_like(k, 0.09)
    assert np.allclose(V.butterfly_g(k, flat, 0 * k, 0 * k), 1.0)
    lognormal = np.exp(-(k + 0.045) ** 2 / (2 * 0.09)) / np.sqrt(2 * np.pi * 0.09)
    assert np.allclose(V.density_from_w(k, flat, 0 * k, 0 * k), lognormal, atol=1e-12)


def test_vogt_slice_has_butterfly_arbitrage_and_ssvi_slice_does_not():
    k = np.linspace(-1.5, 1.5, 3001)
    w, wk, wkk = V.svi_raw(k, a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sig=0.4153)
    g = V.butterfly_g(k, w, wk, wkk)
    neg = k[g < 0]
    assert neg.min() == pytest.approx(0.64, abs=0.01) and neg.max() == pytest.approx(1.26, abs=0.01)
    for theta in (0.01, 0.04, 0.16):
        # Gatheral--Jacquier Theorem 4.2: theta phi (1 + |rho|) < 4 and theta phi^2 (1 + |rho|) <= 4
        w, wk, wkk, _ = V.ssvi(k, theta, -0.7, 1.0, 0.5)
        assert V.butterfly_g(k, w, wk, wkk).min() > 0


def test_heston_total_variance_is_nondecreasing_in_maturity():
    # prop:iv-static-arbitrage(ii) at fixed log-moneyness
    k = np.linspace(-0.4, 0.3, 15)
    ws = []
    for T in (0.25, 0.5, 1.0, 2.0):
        F = S0 * np.exp(R * T)
        C = V.heston_call(S0, F * np.exp(k), T, R, **HESTON)
        ws.append(V.implied_total_vol(C / S0, k) ** 2)
    assert np.all(np.diff(np.array(ws), axis=0) > 0)


# ------------------------------------------------------------ Fourier pricing and the Heston model


def test_lewis_formula_reproduces_black_scholes():
    w = 0.09
    cf = lambda z: np.exp(-0.5j * z * w - 0.5 * z * z * w)     # log(S_T / F) ~ N(-w/2, w)
    k = np.linspace(-1.5, 1.5, 31)
    assert np.allclose(V.call_normalized_from_cf(cf, k), V.black_normalized(k, np.sqrt(w)), atol=1e-12)


def test_heston_characteristic_function_basic_identities():
    u = np.linspace(-30, 30, 121)
    for T in (0.1, 1.0, 5.0):
        phi = V.heston_cf(u, T, **HESTON)[0]
        assert np.all(np.abs(phi) <= 1 + 1e-12)
        assert V.heston_cf(0.0, T, **HESTON)[0] == pytest.approx(1.0)
        assert V.heston_cf(-1j, T, **HESTON)[0] == pytest.approx(1.0)         # E[S_T / F_T] = 1
    # the T-derivative from the Riccati equations (ex:iv-heston-riccati) against a finite difference
    z = np.array([0.3, 2.0, 7.0]) - 0.5j
    h = 1e-5
    fd = (V.heston_cf(z, 1.0 + h, **HESTON)[0] - V.heston_cf(z, 1.0 - h, **HESTON)[0]) / (2 * h)
    assert np.allclose(V.heston_cf_dT(z, 1.0, **HESTON), fd, atol=1e-7)


def _riccati_rk4(z, T, v0, kappa, theta, xi, rho, n=1500):
    """exp(A + D v0) from the Riccati equations of ex:iv-heston-riccati, by RK4 for the pair (A, D)."""
    f = lambda D: 0.5 * xi**2 * D * D - (kappa - rho * xi * 1j * z) * D - 0.5 * (z * z + 1j * z)
    D = np.zeros_like(z)
    A = np.zeros_like(z)
    h = T / n
    for _ in range(n):
        k1 = f(D); D2 = D + 0.5 * h * k1; k2 = f(D2); D3 = D + 0.5 * h * k2; k3 = f(D3); D4 = D + h * k3
        A = A + kappa * theta * h * (D + 2 * D2 + 2 * D3 + D4) / 6
        D = D + h * (k1 + 2 * k2 + 2 * k3 + f(D4)) / 6
    return np.exp(A + D * v0)


def test_heston_closed_form_is_the_continuation_at_u_minus_i_over_2():
    # the check behind the remark after prop:iv-heston, for the parameter sets used in the chapter
    z = np.concatenate([np.linspace(0, 20, 41), np.linspace(25, 150, 11)]) - 0.5j
    sets = [dict(HESTON, rho=r) for r in (-0.7, 0.0, 0.5)] + [dict(v0=0.03, kappa=2.0, theta=0.05, xi=0.5, rho=-0.7)]
    for p in sets:
        for T in (0.05, 1.0, 3.0):
            assert np.max(np.abs(V.heston_cf(z, T, **p)[0] - _riccati_rk4(z, T, **p))) < 1e-9


def test_heston_with_small_vol_of_vol_is_black_scholes_with_deterministic_variance():
    # ex:iv-heston-riccati(d): xi -> 0 gives log(S_T / F_T) ~ N(-V/2, V), V = T * expected variance.
    # With rho = 0 the correction is of order xi^2; with rho != 0 it is of order rho * xi.
    K, T = np.array([70.0, 100.0, 140.0]), 2.0
    p = dict(v0=0.02, kappa=1.5, theta=0.06, xi=1e-4, rho=0.0)
    bs = V.bs_call(S0, K, T, R, np.sqrt(V.heston_expected_variance(T, p["v0"], p["kappa"], p["theta"])))
    assert np.allclose(V.heston_call(S0, K, T, R, **p), bs, atol=1e-6)
    gaps = [V.heston_call(S0, K, T, R, **dict(p, rho=-0.5, xi=xi)) - bs for xi in (1e-3, 1e-4)]
    assert np.allclose(gaps[0][[0, 2]] / gaps[1][[0, 2]], 10.0, rtol=0.01)   # the wings, where the gap is largest


def test_uncorrelated_heston_smile_is_symmetric_and_below_the_mean_at_the_money():
    # prop:iv-heston(iii) and prop:iv-mixing
    p = dict(HESTON, rho=0.0)
    T = 1.0
    F = S0 * np.exp(R * T)
    k = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    C = V.heston_call(S0, F * np.exp(k), T, R, **p)
    w = V.implied_total_vol(C / S0, k) ** 2
    assert np.allclose(w, w[::-1], rtol=1e-7)
    assert w[2] < T * V.heston_expected_variance(T, p["v0"], p["kappa"], p["theta"])


def test_week14b_random_variance_smile():
    # Week 14B / prop:iv-mixing(iii): two-point variance, below the mean at the money, above in the wings
    m, d = 0.04, 0.5
    vals = np.array([m * (1 - d), m * (1 + d)])
    vbar = vals.max()
    k = np.array([0.0, np.sqrt(vbar + vbar**2 / 4) + 0.01, -np.sqrt(vbar + vbar**2 / 4) - 0.01])
    c = 0.5 * (V.black_normalized(k, np.sqrt(vals[0])) + V.black_normalized(k, np.sqrt(vals[1])))
    w = V.implied_total_vol(c, k) ** 2
    assert w[0] < m and w[1] > m and w[2] > m
    assert w[1] == pytest.approx(w[2], rel=1e-9)


def test_heston_fourier_against_monte_carlo():
    T = 0.5
    K = np.array([85.0, 100.0, 115.0])
    out = V.heston_paths(S0, T, R, n_steps=100, n_paths=60_000, seed=7, **HESTON)
    price, se = V.mc_call_prices(out["S"], K, S0 * np.exp(R * T), np.exp(-R * T))
    exact = V.heston_call(S0, K, T, R, **HESTON)
    assert np.all(np.abs(price - exact) < 4 * se + 0.01)


def test_heston_expected_variance_by_simulation():
    T = 1.0
    out = V.heston_paths(S0, T, R, n_steps=200, n_paths=20_000, seed=3, keep=True, **HESTON)
    vbar = out["v_path"][:, :-1].mean(axis=1).mean()
    exact = V.heston_expected_variance(T, HESTON["v0"], HESTON["kappa"], HESTON["theta"])
    assert vbar == pytest.approx(exact, rel=0.02)


# ------------------------------------------------------------ local volatility


def test_flat_and_time_dependent_surfaces_have_the_expected_local_volatility():
    # ex:iv-dupire-w(c): w = int_0^T vartheta^2 independent of k gives sigma_loc = vartheta(T)
    k = np.linspace(-1, 1, 11)
    T = 0.7
    vartheta2 = 0.04 + 0.02 * T
    w = np.full_like(k, 0.04 * T + 0.01 * T**2)
    assert np.allclose(V.local_variance_from_w(k, w, 0 * k, 0 * k, np.full_like(k, vartheta2)), vartheta2)


def test_dupire_local_vol_of_heston_at_short_maturity_and_its_repricing():
    tg = np.linspace(0.0, 0.5, 51)
    tl = tg.copy()
    tl[0] = 0.5 * tg[1]
    kg = np.linspace(-1.2, 0.8, 161)
    lv = V.dupire_heston_grid(tl, kg, **HESTON)
    # at short maturity the at-the-money local volatility is close to sqrt(v0)
    assert lv[0, np.argmin(np.abs(kg))] == pytest.approx(np.sqrt(HESTON["v0"]), abs=0.003)
    # the local volatility model reprices the Heston vanillas (Gyongy / rem:iv-dupire-use)
    out = V.local_vol_paths(S0, 0.5, R, 0.0, tg, kg, lv, 60_000, seed=11)
    K = np.array([85.0, 100.0, 115.0])
    price, se = V.mc_call_prices(out["S"], K, S0 * np.exp(R * 0.5), np.exp(-R * 0.5))
    exact = V.heston_call(S0, K, 0.5, R, **HESTON)
    assert np.all(np.abs(price - exact) < 4 * se + 0.02)


def test_digital_under_the_smile_matches_the_heston_digital():
    # prop:iv-digital, with the slope of the Heston smile from a finite difference
    T = 1.0
    F = S0 * np.exp(R * T)
    h = 0.01
    K3 = np.array([F - h, F, F + h])
    iv = V.implied_vol(V.heston_call(S0, K3, T, R, **HESTON), S0, K3, T, R)
    smile = V.digital_from_smile(S0, F, T, R, iv[1], (iv[2] - iv[0]) / (2 * h))
    srf = V.heston_surface(np.array([0.0]), np.array([T]), **HESTON)
    exact = -np.exp(-R * T) * srf["c_k"][0, 0]           # e^{-rT} Q(S_T > F) = -e^{-rT} e^{-k} c_k at k = 0
    assert smile == pytest.approx(exact, abs=1e-5)
    flat = V.digital_from_smile(S0, F, T, R, iv[1], 0.0)
    assert flat == pytest.approx(digital_call(S0, F, T, R, iv[1]), rel=1e-12)
    assert exact - flat > 0.08                             # the skew correction of sec:iv-consistency


# ------------------------------------------------------------ variance swaps and the VIX


def test_strip_recovers_flat_variance_and_heston_expected_variance():
    # Week 14C: the fair strike from a strip of Black--Scholes prices is sigma^2
    T = 0.25
    F = S0 * np.exp(R * T)
    K = np.arange(5.0, 600.0, 0.25)
    C = V.bs_call(S0, K, T, R, 0.2)
    P = V.bs_put(S0, K, T, R, 0.2)
    # up to the O(h^2 / (T K^2)) error of the trapezoidal rule at the kink at F (h = 0.25 here)
    assert V.strip_variance(K, np.where(K < F, P, C), F, T, R) == pytest.approx(0.04, abs=0.25**2 / (4 * T * F**2))
    # Heston: prop:iv-variance-swap(ii) equals prop:iv-heston(i)
    p = dict(v0=0.03, kappa=2.0, theta=0.05, xi=0.5, rho=-0.7)
    T = 1.0
    F = S0 * np.exp(R * T)
    K = np.arange(1.0, 1500.0, 0.5)
    C = V.heston_call(S0, K, T, R, **p)
    P = C - S0 + K * np.exp(-R * T)
    kvar = V.strip_variance(K, np.where(K < F, P, C), F, T, R)
    assert kvar == pytest.approx(V.heston_expected_variance(T, p["v0"], p["kappa"], p["theta"]), rel=2e-4)


def test_week14c_random_volatility_variance_swap():
    # Week 14C with the random volatility of Week 14B: the strike is E[sigma^2]
    T, vals = 0.5, np.array([0.02, 0.08])
    F = S0 * np.exp(R * T)
    K = np.arange(2.0, 800.0, 0.25)
    C = 0.5 * (V.bs_call(S0, K, T, R, np.sqrt(vals[0])) + V.bs_call(S0, K, T, R, np.sqrt(vals[1])))
    P = C - S0 + K * np.exp(-R * T)
    assert V.strip_variance(K, np.where(K < F, P, C), F, T, R) == pytest.approx(vals.mean(), rel=1e-4)


def test_vix_formula_bias_is_h_squared_over_6_T_K0_squared():
    p = dict(v0=0.03, kappa=2.0, theta=0.05, xi=0.5, rho=-0.7)
    T = 30 / 365
    F = S0 * np.exp(R * T)
    exact = V.heston_expected_variance(T, p["v0"], p["kappa"], p["theta"])
    for h in (1.0, 2.5, 5.0):
        K = np.arange(20.0, 300.0 + 1e-9, h)
        C = V.heston_call(S0, K, T, R, **p)
        P = C - S0 + K * np.exp(-R * T)
        K0 = K[np.searchsorted(K, F, side="right") - 1]
        bias = V.vix_sigma2(K, C, P, F, T, R) - exact
        assert bias == pytest.approx(h**2 / (6 * T * K0**2), rel=0.03)


def test_log_contract_identity_with_k0_below_the_forward():
    # the identity before fig:iv-variance-swap, with x = F / K0 - 1 (exact, fine strike grid)
    T, sig = 0.5, 0.25
    F = S0 * np.exp(R * T)
    K0 = 98.0
    K = np.arange(1.0, 900.0, 0.05)
    C = V.bs_call(S0, K, T, R, sig)
    P = V.bs_put(S0, K, T, R, sig)
    x = F / K0 - 1
    otm = np.where(K < K0, P, C)
    i0 = np.argmin(np.abs(K - K0))
    otm[i0] = 0.5 * (P[i0] + C[i0])              # the integrand jumps at K0: average, as the CBOE does
    lhs = V.strip_variance(K, otm, F, T, R) - 2 / T * (x - np.log1p(x))
    assert lhs == pytest.approx(sig**2, rel=1e-5)
