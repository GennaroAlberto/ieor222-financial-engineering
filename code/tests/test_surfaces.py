"""Checks of the Chapter 17 code against the results proved in the notes."""
import numpy as np
import pytest
from scipy.linalg import expm

from ieor222 import surfaces as S
from ieor222.volatility import black_normalized, heston_call

T_MODEL = S.T_MODEL


@pytest.fixture(scope="module")
def example():
    return S.sanos_example()


def random_martingale_density(K, T, seed=0):
    rng = np.random.default_rng(seed)
    Sigma = 0.1 + 0.3 * rng.random((len(T), K.size))
    return S.dlv_densities(K, Sigma, T), Sigma


# ------------------------------------------------------------ Black--Scholes in normalized units


def test_call_function_of_sanos():
    s, x, v = 1.1, np.array([0.5, 0.9, 1.0, 1.3, 2.0]), 0.04
    c = S.bs_call(s, x, v)
    # Call(s, x, v) = s b(log(x/s), sqrt v), the normalized price of eq:iv-b
    assert np.allclose(c, s * black_normalized(np.log(x / s), np.sqrt(v)))
    # E[(sY - x)^+] for log Y ~ N(-v/2, v), by quadrature
    z = np.linspace(-9, 9, 20001)
    Y = np.exp(np.sqrt(v) * z - v / 2)
    dens = np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
    quad = np.array([np.trapezoid(np.maximum(s * Y - xi, 0) * dens, z) for xi in x])
    assert np.allclose(c, quad, atol=1e-8)
    assert np.allclose(S.bs_call(s, x, 0.0), np.maximum(s - x, 0))
    h = 1e-4
    fd = (S.bs_call(s, x + h, v) - 2 * c + S.bs_call(s, x - h, v)) / h**2
    assert np.allclose(fd, S.bs_call_dxx(s, x, v), rtol=1e-5)


# ------------------------------------------------------------ static arbitrage on a grid


def test_discrete_density_and_linear_interpolation():
    nodes = np.array([0.0, 0.5, 1.0, 1.5, 3.0])
    w = np.array([0.1, 0.3, 0.2, 0.3, 0.1])
    w = w / w.sum()
    nodes = nodes * (1.0 / (w @ nodes))                 # mean 1
    x = nodes[1:]
    c = S.call_function(nodes, w, x)
    assert c[-1] == pytest.approx(0.0)
    # thm:afs-grid: the weights are the changes of slope, and the call function is the linear interpolation
    assert np.allclose(S.discrete_density(x, c), w, atol=1e-14)
    xx = np.linspace(0, 4, 401)
    assert np.allclose(S.linear_interpolation(x, c, xx), S.call_function(nodes, w, xx), atol=1e-14)
    g = S.grid_arbitrage(x, c[None, :])
    assert g["n_butterfly"] == 0 and g["n_slope"] == 0


def test_grid_conditions_detect_each_violation():
    K = S.strike_grid(T_MODEL)
    q, _ = random_martingale_density(K, T_MODEL)
    C = np.array([S.call_function(K, qj, K) for qj in q])
    g = S.grid_arbitrage(K, C, tol=1e-12)
    assert g["n_butterfly"] == g["n_slope"] == g["n_calendar"] == 0
    bad = C.copy()
    bad[2, 40] += 0.01                                   # a call too expensive: negative butterflies around it
    assert S.grid_arbitrage(K, bad, tol=1e-12)["n_butterfly"] > 0
    bad = C.copy()
    bad[3] = C[2] - 1e-3 * (C[2] > 1e-3)                 # a later slice below an earlier one
    assert S.grid_arbitrage(K, bad, tol=1e-12)["n_calendar"] > 0


def test_convex_order_and_strassen_coupling():
    # the example of ex:afs-grids: calls ordered at the inner grid points, so a martingale coupling exists
    xa, pa = np.array([0.5, 1.5]), np.array([0.5, 0.5])
    xb = np.array([0.0, 0.5, 1.5, 3.0])
    pb = S.discrete_density(xb[1:], np.array([0.6, 0.13, 0.0]))
    pi = S.martingale_coupling(xa, pa, xb, pb)
    assert pi is not None
    assert np.allclose(pi.sum(1), pa) and np.allclose(pi.sum(0), pb)
    assert np.allclose(pi @ xb, xa * pa)
    # same means but the call at 1 decreases: no coupling (lem:afs-convex-order and lem:afs-strassen)
    assert S.martingale_coupling(np.array([0.5, 1.5]), pa, np.array([1.0]), np.array([1.0])) is None


# ------------------------------------------------------------ naive interpolation


def test_concave_kink_slope_jump():
    k0, w0, s_left, s_right = 0.05, 0.02, -0.03, -0.06       # a concave kink of w at k0
    w = lambda k: w0 + np.where(k < k0, s_left, s_right) * (k - k0)
    x0, h = np.exp(k0), 1e-6
    c = lambda x: black_normalized(np.log(x), np.sqrt(w(np.log(x))))
    left = (c(x0) - c(x0 - h)) / h
    right = (c(x0 + h) - c(x0)) / h
    jump = S.kink_slope_jump(k0, w0, s_right - s_left)
    assert right - left == pytest.approx(jump, rel=1e-3)
    assert jump < 0
    hb = 1e-3
    assert c(x0 - hb) - 2 * c(x0) + c(x0 + hb) < 0         # prop:afs-kink: a negative butterfly


def test_linear_interpolation_is_the_most_expensive():
    T = 0.5
    x = np.exp(np.linspace(-0.8, 0.6, 15))
    c = heston_call(1.0, x, T, 0.0, **S.HESTON)
    xx = np.linspace(x[0], x[-1], 2001)
    lin = np.interp(xx, x, c)
    true = heston_call(1.0, xx, T, 0.0, **S.HESTON)
    assert np.all(lin >= true - 1e-7)            # prop:afs-expensive, up to the Fourier error (about 3e-8)
    assert np.max(lin - true) > 1e-3             # strictly above between strikes


def test_naive_interpolations_create_arbitrage(example):
    xg = np.exp(np.linspace(-1.2, 0.8, 1601))
    for kind in ("linear", "spline"):
        chk = S.surface_checks(S.naive_vol(example["quotes"], example["T"], kind), example["T"], xg)
        assert chk["butterfly"] < -1.0
    for qt in example["quotes"]:                          # concave kinks at every expiry
        w = S.implied_total_variance(qt["mid"], qt["x"])
        assert 7 <= len(S.concave_kinks(np.log(qt["x"]), w)) <= 11


# ------------------------------------------------------------ SANOS


def test_sanos_surface_is_free_of_static_arbitrage():
    K = S.strike_grid(T_MODEL)
    q, _ = random_martingale_density(K, T_MODEL, seed=3)
    v = 0.25 * 0.04 * T_MODEL
    xg = np.exp(np.linspace(-1.0, 0.7, 801))
    Tg = np.linspace(0.005, 1.3, 80)
    chk = S.surface_checks(lambda t, x: S.sanos_surface(q, K, T_MODEL, v, t, x), Tg, xg)
    assert chk["butterfly"] > -1e-7 and chk["calendar"] > -1e-10 and chk["bounds"] < 1e-10 and chk["slope"] < 1e-9
    # thm:afs-sanos (ii): positive density; (iii): smoothing raises prices, equality without smoothing
    for j in range(len(T_MODEL)):
        assert np.all(S.sanos_density(q[j], K, xg, v[j]) > 0)
        disc = S.call_function(K, q[j], xg)
        assert np.all(S.sanos_surface(q, K, T_MODEL, v, T_MODEL[j], xg) >= disc - 1e-14)
        assert np.allclose(S.sanos_surface(q, K, T_MODEL, 0 * v, T_MODEL[j], xg), disc, atol=1e-14)


def test_sanos_martingale_representation_by_monte_carlo():
    """thm:afs-sanos (i): Z = X Y, with X switching from xi_{j-1} to xi_j at a uniform time and Y lognormal."""
    K = np.array([0.6, 0.9, 1.0, 1.2, 1.5])
    T = np.array([0.5, 1.0])
    q1 = np.array([0.0, 2 / 3, 0.0, 1 / 3, 0.0])              # mean 1
    Qm = np.linalg.inv(S.dlv_matrix(K, np.full(5, 0.3), 0.5))  # a martingale kernel (prop:afs-dlv)
    q2 = Qm @ q1
    q = np.vstack([q1, q2])
    v = np.array([0.01, 0.02])
    rng = np.random.default_rng(1)
    n = 400_000
    i1 = np.where(rng.random(n) < 2 / 3, 1, 3)               # xi_1 = K[i1]
    cum = np.cumsum(Qm, axis=0)                              # column l: the law of xi_2 given xi_1 = K[l]
    i2 = np.argmax(rng.random(n)[:, None] < cum[:, i1].T, axis=1)
    t = 0.8                                                  # between the expiries: alpha = 0.6
    a = (t - 0.5) / 0.5
    X = np.where(rng.random(n) <= a, K[i2], K[i1])
    vt = S.smoothing_variance(t, T, v)
    Z = X * np.exp(np.sqrt(vt) * rng.standard_normal(n) - vt / 2)
    x = np.array([0.8, 1.0, 1.25])
    pay = np.maximum(Z[:, None] - x[None, :], 0)
    se = pay.std(0) / np.sqrt(n)
    assert np.all(np.abs(pay.mean(0) - S.sanos_surface(q, K, T, v, t, x)) < 4 * se)
    assert np.mean(Z) == pytest.approx(1.0, abs=4 * np.std(Z) / np.sqrt(n))


def test_lp_calibration(example):
    fit, K, quotes = example["fit"], example["K"], example["quotes"]
    assert fit["status"] == 0
    q = fit["q"]
    assert np.all(q >= 0)
    assert np.allclose(q.sum(1), 1, atol=1e-9) and np.allclose(q @ K, 1, atol=1e-9)
    calls = np.array([S.call_function(K, qj, K) for qj in q])
    assert np.all(np.diff(calls, axis=0) >= -1e-9)           # the calendar constraints of eq:afs-lp
    assert example["inside"] == 1.0                           # every fitted price inside its spread
    hard = S.sanos_calibrate(K, quotes, example["v"], hard=True)
    assert hard["status"] == 0
    # the LP value is min J(q) of prop:afs-lp (ii); check it where J is not tiny (heavy smoothing, eta = 0.9)
    eps = 0.05
    fit9 = S.sanos_calibrate(K, quotes, example["V"] * 0.9, eps=eps)
    J = 0.0
    for c, qt in zip(fit9["c"], quotes):
        w = 1 / (qt["ask"] - qt["bid"])
        J += np.sum(w * (np.maximum(c - qt["ask"], 0) + np.maximum(qt["bid"] - c, 0) + eps * np.abs(c - qt["mid"])))
    assert fit9["value"] > 10 and J == pytest.approx(fit9["value"], rel=1e-6)


# ------------------------------------------------------------ discrete local volatility


def test_dlv_transition_matrix():
    K = S.strike_grid(T_MODEL)
    rng = np.random.default_rng(2)
    sig = 0.05 + 0.5 * rng.random(K.size)
    M = S.dlv_matrix(K, sig, 0.3)
    Q = np.linalg.inv(M)
    assert Q.min() > -1e-15
    assert np.allclose(Q.sum(0), 1) and np.allclose(K @ Q, K)
    assert np.allclose(Q[:, 0], np.eye(K.size)[:, 0]) and np.allclose(Q[:, -1], np.eye(K.size)[:, -1])
    p = rng.random(K.size)
    sub, main, sup = S.dlv_bands(K, sig, 0.3)
    assert np.allclose(S.thomas(sub, main, sup, p), np.linalg.solve(M, p), atol=1e-14)


def test_dlv_three_strikes_closed_form():
    # ex:afs-dlv-three with a = b = 0.1, Sigma = 0.2, Delta = 1
    K = np.array([0.9, 1.0, 1.1])
    Q = np.linalg.inv(S.dlv_matrix(K, np.array([0, 0.2, 0]), 1.0))
    q1 = Q[:, 1]
    assert np.allclose(q1, [0.4, 0.2, 0.4])
    assert q1 @ (K - 1) ** 2 == pytest.approx(0.04 / 5)
    C = S.call_function(K, q1, K)
    assert S.implied_dlv(K, C[None, :], [1.0])[0, 1] == pytest.approx(0.2)


def test_implied_dlv_round_trip():
    K = S.strike_grid(T_MODEL)
    q, Sigma = random_martingale_density(K, T_MODEL, seed=1)
    C = np.array([S.call_function(K, qj, K) for qj in q])
    Sh = S.implied_dlv(K, C, T_MODEL)
    mask = q[:, 1:-1] > 1e-6
    assert np.max(np.abs(Sh - Sigma)[:, 1:-1][mask]) < 1e-7     # prop:afs-dlv (ii)


def test_implied_dlv_approaches_dupire():
    # the numbers quoted in the notes (Dupire's formula with a converged Fourier reference)
    assert S.dlv_dupire_gap(0.5, 1 / 12, 0.05) == pytest.approx(0.0275, abs=3e-4)
    assert S.dlv_dupire_gap(0.5, 1 / 52, 0.02) == pytest.approx(0.0059, abs=2e-4)
    assert S.dlv_dupire_gap(0.5, 1 / 252, 0.01) == pytest.approx(0.0013, abs=1e-4)


# ------------------------------------------------------------ the generator


def test_decoded_surfaces_are_martingale_densities():
    K = S.strike_grid(T_MODEL)
    basis = S.dlv_basis(K, T_MODEL)
    rng = np.random.default_rng(4)
    u = rng.normal(0, 1.5, size=(50, basis.shape[0] * basis.shape[2] + basis.shape[0]))
    q, v = S.decode(u, K, T_MODEL, basis)
    assert np.all(q >= -1e-15)
    assert np.allclose(q.sum(-1), 1) and np.allclose(q @ K, 1)
    calls = np.einsum("pjn,nl->pjl", q, np.maximum(K[:, None] - K[None, :], 0))
    assert np.all(np.diff(calls, axis=1) >= -1e-12)          # convex order: prop:afs-generator (i)
    assert np.all(np.diff(v, axis=-1) > 0)


def test_generator_stability():
    rng = np.random.default_rng(5)
    d, kmin, D = 6, 0.0002, 5.0
    L = rng.normal(size=(d, d))
    Rm = rng.normal(size=(d, d))
    kappa = L @ L.T + 0.5 * (Rm - Rm.T) + kmin * np.eye(d)
    assert np.all(np.linalg.eigvals(kappa).real >= kmin - 1e-10)     # prop:afs-generator (ii)
    A = expm(-kappa * D)
    assert np.max(np.abs(np.linalg.eigvals(A))) <= np.exp(-kmin * D) + 1e-12


def test_ar1_fit_and_invariant_law():
    rng = np.random.default_rng(6)
    A = np.array([[0.9, 0.05], [0.0, 0.5]])
    Om = np.array([[1.0, 0.3], [0.3, 0.5]])
    m = np.array([1.0, -2.0])
    h = S.simulate_ar1(m, A, Om, m, 20000, 1, seed=7)[0]
    m_hat, A_hat, Om_hat, rho = S.fit_ar1(h)
    assert np.allclose(A_hat, A, atol=0.02) and np.allclose(m_hat, m, atol=0.15) and np.allclose(Om_hat, Om, atol=0.05)
    P = S.ar1_invariant(A, Om)
    assert np.allclose(P, A @ P @ A.T + Om)
    U = rng.normal(size=(200, 8)) @ rng.normal(size=(8, 8))
    b, B, H = S.compress(U, 8)
    assert np.allclose(b + H @ B.T, U)                         # full rank: exact
    assert np.allclose(H.mean(0), 0, atol=1e-12) and np.allclose(np.cov(H.T, bias=True), np.eye(8), atol=1e-10)


def test_history_fit_small():
    hist = S.synthetic_history(n_days=12)
    assert hist["U"].shape == (12, 36)
    assert hist["inside"].mean() > 0.85


# ------------------------------------------------------------ drift tests


def test_conditional_test_gaussian_action_and_power():
    rng = np.random.default_rng(8)
    mu = np.array([0.1, -0.05])
    G = np.array([[0.5, 0.2], [0.2, 0.3]])
    R = rng.multivariate_normal(mu, G, size=100000)
    Th = S.fit_actions(R, np.ones((len(R), 1)))
    assert np.allclose(Th[:, 0], np.linalg.solve(np.eye(2) + G, mu), atol=0.01)     # ex:afs-gaussian-test
    n = 4000
    Z = rng.multivariate_normal(np.zeros(2), G, size=(n, 5))
    T0, p0, _ = S.conditional_test(Z, antithetic=False)       # no drift: no rejection
    assert abs(T0) < 3 and p0 > 0.003
    Z = Z[: n // 2]
    T1, p1, _ = S.conditional_test(np.concatenate([mu + Z, mu - Z]))   # drift, antithetic pairs
    assert T1 > 5


def test_entropy_weights_remove_the_mean():
    rng = np.random.default_rng(9)
    R = rng.normal(0.05, 0.2, size=(5000, 3))
    D, a = S.entropy_weights(R)
    assert D.mean() == pytest.approx(1.0) and np.all(D > 0)
    m, z = S.drift_zscores(R, D)
    assert np.max(np.abs(m)) < 1e-9
    # for Gaussian returns the minimal-entropy tilt is exponential with a close to Sigma^{-1} mu
    assert np.allclose(a, 0.05 / 0.04, rtol=0.15)
    m0, z0 = S.drift_zscores(R)
    assert np.allclose(z0, m0 / (R.std(0, ddof=0) / np.sqrt(len(R))), rtol=1e-12)


def test_implied_dlv_undefined_without_mass():
    # Gamma = 0 < Theta at the middle strike: mass leaves 1 entirely, no finite discrete local volatility
    K = np.array([0.5, 1.0, 1.5])
    C = S.call_function(K, np.array([0.5, 0.0, 0.5]), K)
    assert np.isnan(S.implied_dlv(K, C[None, :], [1.0])[0, 1])
    assert S.implied_dlv(K, np.maximum(1 - K, 0)[None, :], [1.0])[0, 1] == 0.0      # 0/0 := 0


def test_pair_standard_errors():
    rng = np.random.default_rng(10)
    Z = rng.standard_normal((1000, 3))
    noise = 0.1 * rng.standard_normal((2000, 3))
    R = np.concatenate([0.01 + Z, 0.01 - Z]) + noise          # antithetic pairs, strongly negatively correlated
    m, z = S.drift_zscores(R, antithetic=True)
    pairs = 0.5 * (R[:1000] + R[1000:])
    assert np.allclose(z, pairs.mean(0) / (pairs.std(0) / np.sqrt(1000)))
    assert np.all(np.abs(z) > 3 * np.abs(S.drift_zscores(R)[1]))   # the iid formula understates the evidence
