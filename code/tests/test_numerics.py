"""Checks of the Chapter 16 code against the results proved in the notes."""
import numpy as np
import pytest

from ieor222.blackscholes import call, greeks, put
from ieor222.exotics import american_put_binomial, perpetual_put_boundary
from ieor222.numerics import (_bands, american_put_fd, apply_operator, balanced_allocation, brennan_schwartz,
                              bs_log_coefficients, call_payoff_sd, convergence_table, euler_power_bias,
                              euler_power_moments, euler_terminal, european_fd, explicit_heat,
                              explicit_s_grid, heat_eigenvalues, log_grid, mc_european, mc_summary,
                              pinn_eval, pinn_init, pinn_loss_and_grad, pinn_train, power_claim_price,
                              power_claim_rmse, power_claim_variance, psor, theta_scheme)

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


# ------------------------------------------------------------ Monte Carlo (thm:num-mc, prop:num-mc-cost)


def test_call_payoff_sd_and_estimator():
    payoff = lambda s: np.maximum(s - K, 0.0)
    res = mc_european(payoff, S0, T, R, SIGMA, 1_000_000, seed=1)
    assert res["sd"] == pytest.approx(call_payoff_sd(S0, K, T, R, SIGMA), rel=0.01)
    assert abs(res["mean"] - call(S0, K, T, R, SIGMA)) < 4 * res["se"]
    assert res["half_width"] == pytest.approx(1.96 * res["se"], rel=1e-3)


def test_confidence_intervals_cover_about_95_percent():
    # thm:num-mc (iii): the asymptotic interval covers with probability close to 0.95 for large M.
    p = call(S0, K, T, R, SIGMA)
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((400, 2000))
    S = S0 * np.exp(SIGMA * np.sqrt(T) * Z + (R - 0.5 * SIGMA**2) * T)
    Y = np.exp(-R * T) * np.maximum(S - K, 0.0)
    cover = np.mean([mc_summary(y)["low"] <= p <= mc_summary(y)["high"] for y in Y])
    assert 0.91 <= cover <= 0.98


def test_euler_power_moments_closed_form():
    # exa:num-power: the moments of the Euler scheme on S are explicit; check them by simulation.
    S1, r, s = 1.0, 0.05, 0.4
    mean, var = euler_power_moments(S1, T, r, s, 4)
    Y = np.exp(-r * T) * euler_terminal(S1, T, r, s, 4, 400_000, seed=3) ** 2
    assert Y.mean() == pytest.approx(mean, abs=4 * np.sqrt(var / Y.size))
    assert Y.var() == pytest.approx(var, rel=0.05)
    # the bias is beta_1 h + O(h^2) with beta_1 = p T (r^2 - (2r + sigma^2)^2 / 2)
    p = power_claim_price(S1, T, r, s)
    beta1 = p * T * (r * r - 0.5 * (2 * r + s * s) ** 2)
    for n in (100, 1000):
        assert euler_power_bias(S1, T, r, s, n) * n / T == pytest.approx(beta1, rel=2e-3 * 100 / n + 1e-4)
    assert mean - p == pytest.approx(euler_power_bias(S1, T, r, s, 4), rel=1e-9)
    # with infinitely many steps the variance tends to that of exact sampling
    assert euler_power_moments(S1, T, r, s, 10**6)[1] == pytest.approx(power_claim_variance(S1, T, r, s), rel=1e-4)


def test_cost_allocation_reaches_the_target():
    # prop:num-mc-cost (i), with c_2 and v_2 bounds for the power claim of exa:num-power.
    S1, r, s = 1.0, 0.05, 0.4
    n_grid = np.arange(1, 5000)
    ratio = np.abs(euler_power_bias(S1, T, r, s, n_grid)) * n_grid / T
    var = euler_power_moments(S1, T, r, s, n_grid)[1]
    assert 0.032 < ratio.min() and ratio.max() < 0.039 and var.max() < 1.37
    for eps in (1e-2, 3e-3, 1e-3):
        n, M, cost = balanced_allocation(eps, 0.039, 1.37, T)
        assert power_claim_rmse(S1, T, r, s, n, M) <= eps
        assert cost <= 12 * 0.039 * 1.37 * T / eps**3
    # exact sampling: no bias, RMSE = sd / sqrt(M)
    assert power_claim_rmse(S1, T, r, s, 0, 10**4) == pytest.approx(np.sqrt(power_claim_variance(S1, T, r, s) / 1e4))


# ------------------------------------------------------------ finite differences


def test_difference_quotients_are_second_order():
    # lem:num-taylor through the operator L_h: errors fall by 4 when h halves.
    a, b, c = 0.3, -0.2, 0.1
    errs = []
    for h in (0.1, 0.05, 0.025):
        y = np.arange(-1, 1 + h / 2, h)
        f = np.sin(2 * y)
        exact = a * (-4 * np.sin(2 * y)) + b * 2 * np.cos(2 * y) - c * f
        errs.append(np.max(np.abs(apply_operator(f, a, b, c, h) - exact[1:-1])))
    assert errs[0] / errs[1] == pytest.approx(4, rel=0.05) and errs[1] / errs[2] == pytest.approx(4, rel=0.05)


def test_explicit_scheme_stability_threshold():
    # prop:num-explicit-stability: lambda <= 1/2 gives a convex combination (maximum principle),
    # lambda > 1/2 lets the sawtooth grow like |1 - 4 lambda|^n.
    J = 40
    y = np.linspace(0.0, 1.0, J + 1)
    u0 = np.sin(np.pi * y) + 1e-12 * (-1.0) ** np.arange(J + 1)
    u0[0] = u0[-1] = 0.0
    good = explicit_heat(u0, 0.5, 2000)
    assert np.max(np.abs(good)) <= np.max(np.abs(u0)) + 1e-15
    bad = explicit_heat(u0, 0.55, 300)
    assert np.max(np.abs(bad)) > 1e3
    saw = np.array([(-1.0) ** j for j in range(-30, 31)])
    out = explicit_heat(saw, 0.6, 1)     # interior nodes far from the ends see the infinite grid
    assert out[30] == pytest.approx((1 - 4 * 0.6) * saw[30])
    # eigenvalues of the iteration matrix
    lam = 0.6
    A = (np.diag(-2.0 * np.ones(J - 1)) + np.diag(np.ones(J - 2), 1) + np.diag(np.ones(J - 2), -1))
    G = np.eye(J - 1) + lam * A
    assert np.sort(np.linalg.eigvalsh(G)) == pytest.approx(np.sort(heat_eigenvalues(J, lam, 0.0)), abs=1e-12)
    assert min(heat_eigenvalues(J, lam, 0.0)) == pytest.approx(1 - 4 * lam * np.cos(np.pi / (2 * J)) ** 2)


def test_symmetrization_and_l2_stability():
    # proof of thm:num-schemes (ii): D^{-1} A D is symmetric, and ||G^m B^{-1}||_2 <= e^{|b| L / a}.
    a, b, c = 0.02, -0.15, 0.05
    J, L = 60, 3.0
    h = L / J
    assert h * abs(b) <= a
    pm, p0, pp = _bands(J - 1, a, b, c, h)
    A = np.diag(np.full(J - 1, p0)) + np.diag(np.full(J - 2, pp), 1) + np.diag(np.full(J - 2, pm), -1)
    d = (pm / pp) ** (np.arange(1, J) / 2)
    S = np.diag(1 / d) @ A @ np.diag(d)
    assert np.allclose(S, S.T)
    assert np.max(np.linalg.eigvalsh(S)) <= -c + 1e-12
    for theta, k in ((0.5, 0.05), (1.0, 0.05), (0.0, 0.9 / (4 * a / h**2 + c))):
        B = np.eye(J - 1) - theta * k * A
        G = np.linalg.solve(B, np.eye(J - 1) + (1 - theta) * k * A)
        M = np.linalg.inv(B)
        for _ in range(200):
            assert np.linalg.norm(M, 2) <= np.exp(abs(b) * L / a) + 1e-9
            M = G @ M


@pytest.mark.parametrize("scheme,order", [("explicit", 2.0), ("implicit", 1.0), ("cn-rannacher", 2.0)])
def test_convergence_orders_on_a_call(scheme, order):
    # thm:num-schemes and Rannacher's start: observed orders at S0 = K (fig:num-convergence).
    lev = [100, 200, 400] if scheme == "explicit" else [100, 200, 400, 800]
    tab = convergence_table("call", S0, K, T, R, SIGMA, lev, scheme)
    e = np.abs(tab["error"])
    orders = np.log2(e[:-1] / e[1:])
    assert np.all(np.abs(orders - order) < 0.15)


def test_crank_nicolson_gamma_oscillates_without_rannacher():
    plain = european_fd("call", S0, K, T, R, SIGMA, 800, 67, theta=0.5, rannacher=0)
    smooth = european_fd("call", S0, K, T, R, SIGMA, 800, 67, theta=0.5, rannacher=2)
    g = greeks(S0, K, T, R, SIGMA, "call")["gamma"]
    assert abs(plain["gamma"] - g) > 1.0 * g          # wrong by more than the gamma itself
    assert abs(smooth["gamma"] - g) < 1e-3 * g


def test_put_call_parity_and_boundary_error():
    c = european_fd("call", S0, K, T, R, SIGMA, 400, 200, rannacher=2)
    p = european_fd("put", S0, K, T, R, SIGMA, 400, 200, rannacher=2)
    assert c["price"] - p["price"] == pytest.approx(S0 - K * np.exp(-R * T), abs=1e-3)
    # sec:num-bs: with m = 6 the boundary values are wrong by less than 1e-8
    y = log_grid(S0, T, SIGMA, 200, 6.0)
    taus = np.linspace(1e-4, T, 500)
    assert np.max(call(np.exp(y[0]), K, taus, R, SIGMA)) < 1e-8
    assert np.max(put(np.exp(y[-1]), K, taus, R, SIGMA)) < 1e-8


def test_discrete_maximum_principle_for_boundary_data():
    # thm:num-schemes (iv): perturbing the boundary values by delta moves the solution by at most delta.
    a, b, c = bs_log_coefficients(R, SIGMA)
    y = log_grid(S0, T, SIGMA, 100, 4.0)
    x = np.exp(y)
    g = np.maximum(x - K, 0.0)
    h = y[1] - y[0]
    n_t = int(np.ceil(T * a / (0.4 * h * h)))
    for theta, nt in ((1.0, 40), (0.0, n_t)):
        u1 = theta_scheme(g, y, T, nt, a, b, c, theta, lambda t: (0.0, x[-1] - K * np.exp(-R * t)))["u"]
        u2 = theta_scheme(g, y, T, nt, a, b, c, theta, lambda t: (0.3, x[-1] - K * np.exp(-R * t) - 0.2))["u"]
        assert np.max(np.abs(u1 - u2)) <= 0.3 + 1e-12


def test_explicit_scheme_in_the_price_variable():
    # rem:num-s-grid, the worked example of the course: N = 10^4 steps, S_max = 400. M = 100 and 505
    # are fine; M = 510 explodes near S_max (spectral radius 1.0055).
    for M in (100, 505):
        res = explicit_s_grid(K, R, SIGMA, T, 400.0, M, 10_000)
        inside = (res["S"] <= 350) & (res["S"] > 0)
        assert np.max(np.abs(res["f"][inside] - call(res["S"][inside], K, T, R, SIGMA))) < 0.05
    assert explicit_s_grid(K, R, SIGMA, T, 400.0, 500, 10_000)["positive"]
    assert not explicit_s_grid(K, R, SIGMA, T, 400.0, 501, 10_000)["positive"]
    assert explicit_s_grid(K, R, SIGMA, T, 400.0, 510, 10_000)["max_abs"] > 1e6
    dt, M = T / 10_000, 510
    j = np.arange(1, M)
    G = (np.diag(1 - dt * (SIGMA**2 * j**2 + R)) + np.diag(0.5 * dt * (SIGMA**2 * j[1:] ** 2 - R * j[1:]), -1)
         + np.diag(0.5 * dt * (SIGMA**2 * j[:-1] ** 2 + R * j[:-1]), 1))
    assert np.max(np.abs(np.linalg.eigvals(G))) == pytest.approx(1.0055, abs=1e-4)


# ------------------------------------------------------------ the American put (prop:num-american-grid)


def test_psor_solves_the_complementarity_problem_and_matches_brennan_schwartz():
    a, b, c = bs_log_coefficients(R, SIGMA)
    y = log_grid(K, T, SIGMA, 200)
    x = np.exp(y)[1:-1]
    h, k, theta = y[1] - y[0], 0.01, 0.5
    pm, p0, pp = _bands(len(x), a, b, c, h)
    m = len(x)
    ab = np.zeros((3, m))
    ab[0, 1:], ab[1], ab[2, :-1] = -theta * k * pp, 1 - theta * k * p0, -theta * k * pm
    g = np.maximum(K - x, 0.0)
    q = g + (1 - theta) * k * apply_operator(np.r_[K - np.exp(y[0]), g, 0.0], a, b, c, h)
    q[0] += theta * k * pm * (K - np.exp(y[0]))
    for omega in (0.8, 1.0, 1.5):
        z, it = psor(ab, q, g, np.zeros(m), omega=omega, tol=1e-13)
        B = np.diag(ab[1]) + np.diag(ab[0, 1:], 1) + np.diag(ab[2, :-1], -1)
        w = B @ z - q
        assert np.all(z >= g) and np.all(w >= -1e-9) and abs((z - g) @ w) < 1e-8
        assert z == pytest.approx(brennan_schwartz(ab, q, g), abs=1e-9)


def test_american_put_price_and_boundary():
    res = american_put_fd(S0, K, T, R, SIGMA, n_x=400, n_t=200, keep=True)
    assert res["price"] == pytest.approx(american_put_binomial(S0, K, T, R, SIGMA, n=4000), abs=3e-3)
    assert res["price"] > put(S0, K, T, R, SIGMA) + 0.5
    euro = european_fd("put", S0, K, T, R, SIGMA, 400, 200, rannacher=2, width=6.0)
    assert np.all(res["v"] >= np.maximum(K - res["x"], 0.0) - 1e-12)
    # Crank--Nicolson here has lambda = a k / h^2 about 2.8, outside prop:num-american-grid (iii); the
    # dominance below is observed, not proved. The implicit test that follows is covered by (iii).
    assert res["v"][200] >= euro["v"][200] - 1e-9          # same node: S0 = K is the center of both grids
    bnd = res["boundary"][1:]                              # calendar times from T - k down to 0
    assert np.all(np.diff(bnd) <= 1e-12)                   # rises towards maturity
    assert perpetual_put_boundary(K, R, SIGMA) <= bnd.min() and bnd.max() < K
    assert res["boundary"][-1] == pytest.approx(80.8, abs=1.0)


# ------------------------------------------------------------ the physics-informed network (rem:num-pinn)


def _pinn_points(seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = np.log(0.5), np.log(2.0)
    tb = np.linspace(0, T, 11)
    y0 = np.linspace(lo, hi, 51)
    return dict(tau_r=rng.uniform(0, T, 200), y_r=rng.uniform(lo, hi, 200), y_0=y0,
                g_0=np.maximum(np.exp(y0) - 1, 0), tau_b=np.r_[tb, tb],
                y_b=np.r_[np.full(11, lo), np.full(11, hi)], g_b=np.r_[np.zeros(11), np.exp(hi) - np.exp(-R * tb)])


def test_pinn_gradients_match_finite_differences():
    a, b, c = bs_log_coefficients(R, SIGMA)
    pts = _pinn_points()
    p = pinn_init(6, seed=4)
    _, g, _ = pinn_loss_and_grad(p, pts, a, b, c)
    eps = 1e-6
    for key in p:
        for i in range(min(3, p[key].size)):
            up = {kk: v.copy() for kk, v in p.items()}
            dn = {kk: v.copy() for kk, v in p.items()}
            up[key][i] += eps
            dn[key][i] -= eps
            fd = (pinn_loss_and_grad(up, pts, a, b, c)[0] - pinn_loss_and_grad(dn, pts, a, b, c)[0]) / (2 * eps)
            assert fd == pytest.approx(g[key][i], rel=1e-5, abs=1e-8)
    # derivatives in (tau, y) of the network
    e0, e1 = pinn_eval(p, np.array([0.3]), np.array([0.1])), pinn_eval(p, np.array([0.3]), np.array([0.1 + 1e-5]))
    assert (e1["u"] - e0["u"]) / 1e-5 == pytest.approx(e0["u_y"], rel=1e-3)


def test_pinn_training_reduces_the_loss():
    a, b, c = bs_log_coefficients(R, SIGMA)
    p, hist = pinn_train(_pinn_points(), a, b, c, n_hidden=16, n_iter=800, lr=0.01, seed=0)
    assert hist[-1, 1] < 0.05 * hist[0, 1]
    yy = np.linspace(np.log(0.6), np.log(1.8), 9)
    err = pinn_eval(p, np.full(9, 0.5), yy)["u"] - call(np.exp(yy), 1.0, 0.5, R, SIGMA)
    assert np.max(np.abs(err)) < 0.05


def test_implicit_american_dominates_european():
    # prop:num-american-grid (iii) with theta = 1: the condition (1 - theta) k (2a/h^2 + c) <= 1 holds,
    # the European boundary values K e^{-r tau} - x_0 and 0 are not larger than the American ones.
    amer = american_put_fd(S0, K, T, R, SIGMA, n_x=200, n_t=100, theta=1.0, rannacher=0)
    euro = european_fd("put", S0, K, T, R, SIGMA, 200, 100, theta=1.0)
    assert np.allclose(amer["x"], euro["x"])
    assert np.all(amer["v"] >= euro["v"] - 1e-10)
    assert np.all(amer["v"] >= np.maximum(K - amer["x"], 0.0) - 1e-12)


def test_brennan_schwartz_agrees_with_psor_on_the_put():
    a = american_put_fd(S0, K, T, R, SIGMA, n_x=400, n_t=200)
    b = american_put_fd(S0, K, T, R, SIGMA, n_x=400, n_t=200, solver="brennan-schwartz")
    assert a["price"] == pytest.approx(b["price"], abs=1e-6)


def test_theta_scheme_uses_the_old_boundary_values():
    # def:num-theta: the explicit part of a step uses the boundary values at the old time, not the
    # end values of u0. A payoff that is wrong at the end nodes gives the same solution.
    a, b, c = bs_log_coefficients(R, SIGMA)
    y = log_grid(S0, T, SIGMA, 100, 4.0)
    x = np.exp(y)
    g = np.maximum(x - K, 0.0)
    bc = lambda t: (0.0, 0.0)
    clean = g.copy()
    clean[0] = clean[-1] = 0.0
    for theta in (0.0, 0.5):
        n_t = 400 if theta == 0.0 else 20
        u1 = theta_scheme(g, y, 0.1, n_t, a, b, c, theta, bc)["u"]
        u2 = theta_scheme(clean, y, 0.1, n_t, a, b, c, theta, bc)["u"]
        assert np.allclose(u1, u2, atol=1e-13)
