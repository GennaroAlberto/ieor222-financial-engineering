"""Checks of the Chapter 19 code against the results proved or cited in the notes."""
import gzip
import struct

import numpy as np
import pytest
from scipy.integrate import quad

from ieor222 import roughvol as rv
from ieor222.volatility import black_normalized, heston_cf


# ---------------------------------------------------------------- fBM and the Riemann--Liouville process


def test_abs_moments():
    assert rv.abs_moment(2.0) == pytest.approx(1.0)
    assert rv.abs_moment(1.0) == pytest.approx(np.sqrt(2 / np.pi))
    assert rv.abs_moment(4.0) == pytest.approx(3.0)
    q = 0.5
    val = quad(lambda x: abs(x) ** q * np.exp(-x * x / 2) / np.sqrt(2 * np.pi), -np.inf, np.inf)[0]
    assert rv.abs_moment(q) == pytest.approx(val, rel=1e-7)


def test_fbm_covariance_reduces_to_brownian_motion():
    s, t = np.meshgrid(np.linspace(0, 2, 9), np.linspace(0, 2, 9))
    assert np.allclose(rv.fbm_cov(s, t, 0.5), np.minimum(s, t))


@pytest.mark.parametrize("H", [0.1, 0.3, 0.7, 0.9])
def test_fgn_autocovariance_sign_asymptotics_and_telescoping(H):
    n = np.arange(1, 50)
    r = rv.fgn_autocov(n, H)
    assert np.all(np.sign(r) == np.sign(H - 0.5))                      # prop:rv-fbm-properties(iv)
    big = 10.0**4
    assert rv.fgn_autocov(big, H) / (H * (2 * H - 1) * big ** (2 * H - 2)) == pytest.approx(1, rel=1e-3)
    m = 30                                                              # sum_{|k|<=m} rho(k) = (m+1)^{2H} - m^{2H}
    k = np.arange(-m, m + 1)
    assert rv.fgn_autocov(k, H).sum() == pytest.approx((m + 1) ** (2 * H) - m ** (2 * H))


@pytest.mark.parametrize("H", [0.07, 0.25, 0.5, 0.8])
def test_mandelbrot_van_ness_constant(H):
    a = H - 0.5
    left = quad(lambda u: ((1 - u) ** a - (-u) ** a) ** 2, -np.inf, 0, limit=400)[0]
    right = quad(lambda u: (1 - u) ** (2 * a), 0, 1, limit=400)[0]
    assert rv.mvn_constant(H) == pytest.approx(left + right, rel=1e-7)


@pytest.mark.parametrize("H", [0.05, 0.1, 0.3, 0.7])
def test_riemann_liouville_covariances(H):
    a = H - 0.5
    for s, t in [(0.2, 1.0), (0.97, 1.0), (1.5, 0.4), (0.7, 0.7)]:
        lo, hi = min(s, t), max(s, t)
        val = 2 * H * quad(lambda u: (lo - u) ** a * (hi - u) ** a, 0, lo, limit=400)[0]
        assert rv.rl_cov(s, t, H) == pytest.approx(val, rel=1e-6)
        wv = np.sqrt(2 * H) * quad(lambda u: (t - u) ** a, 0, min(s, t), limit=400)[0]
        assert rv.rl_brownian_cov(t, s, H) == pytest.approx(wv, rel=1e-6)
    assert rv.rl_cov(2.0, 2.0, H) == pytest.approx(2.0 ** (2 * H))     # Var Y_t = t^{2H}


def test_riemann_liouville_scaling():
    # prop:rv-rl(ii): Cov(Y_cs, Y_ct) = c^{2H} Cov(Y_s, Y_t) and Cov(Y_ct, W_cs) = c^{H+1/2} Cov(Y_t, W_s)
    H, c = 0.12, 3.7
    assert rv.rl_cov(c * 0.3, c * 0.8, H) == pytest.approx(c ** (2 * H) * rv.rl_cov(0.3, 0.8, H))
    assert rv.rl_brownian_cov(c * 0.8, c * 0.3, H) == pytest.approx(c ** (H + 0.5) * rv.rl_brownian_cov(0.8, 0.3, H))


def test_rl_increment_bound():
    # prop:rv-rl(iii): (t-s)^{2H} <= E(Y_t - Y_s)^2 <= 2H c_H^2 (t-s)^{2H}
    H = 0.1
    for s, t in [(0.1, 0.2), (1.0, 1.01), (5.0, 5.5)]:
        inc = rv.rl_cov(t, t, H) + rv.rl_cov(s, s, H) - 2 * rv.rl_cov(s, t, H)
        assert (t - s) ** (2 * H) <= inc <= 2 * H * rv.mvn_constant(H) * (t - s) ** (2 * H)


@pytest.mark.parametrize("sampler", ["cholesky", "davies-harte"])
def test_exact_fbm_samplers_have_the_fbm_covariance(sampler):
    H, n = 0.14, 32
    f = rv.fbm_cholesky if sampler == "cholesky" else rv.fbm_davies_harte
    t, X = f(H, 2.0, n, 40_000, seed=7)
    emp = X.T @ X / X.shape[0]
    exact = rv.fbm_cov(t[:, None], t[None, :], H)
    assert np.abs(emp - exact).max() < 0.05
    inc = np.diff(X, axis=1)                                           # stationary increments
    assert np.allclose(inc.var(axis=0), (2.0 / n) ** (2 * H), rtol=0.05)


def test_rl_cholesky_joint_law():
    H = 0.1
    t, Y, dW = rv.rl_cholesky(H, 1.0, 20, 40_000, seed=8)
    W = np.cumsum(dW, axis=1)
    assert Y[:, -1].var() == pytest.approx(1.0, abs=0.03)
    assert np.mean(Y[:, 10] * W[:, -1]) == pytest.approx(rv.rl_brownian_cov(t[10], 1.0, H), abs=0.02)


# ---------------------------------------------------------------- the hybrid scheme


def _hybrid_moments(H, n, ell, points):
    """Var(Yhat_1), Cov(Y_1, Yhat_1) from the definition of the scheme (independent of the module's MSE formula)."""
    a, dt = H - 0.5, 1.0 / n
    k = np.arange(ell + 1, n + 1, dtype=float)
    b = rv.optimal_points(k, a) if points == "optimal" else k
    cell = lambda kk, p: quad(lambda y: y ** p, kk - 1, kk)[0] * dt ** (p + 1)    # int over cell of (t - s)^p ds
    var_hat = sum((bk * dt) ** (2 * a) * dt for bk in b) + (cell(1, 2 * a) if ell == 1 else 0.0)
    cov = sum((bk * dt) ** a * cell(kk, a) for kk, bk in zip(k, b)) + (cell(1, 2 * a) if ell == 1 else 0.0)
    return 2 * H * var_hat, 2 * H * cov


@pytest.mark.parametrize("ell,points", [(0, "left"), (0, "optimal"), (1, "optimal")])
def test_hybrid_mse_formula(ell, points):
    H, n = 0.1, 40
    var_hat, cov = _hybrid_moments(H, n, ell, points)
    assert rv.hybrid_mse(H, n, n, ell, points) == pytest.approx(1.0 - 2 * cov + var_hat, rel=1e-6)
    _, Y, _ = rv.hybrid_rl(H, 1.0, n, 60_000, seed=9, ell=ell, points=points)
    assert Y[:, -1].var() == pytest.approx(var_hat, rel=0.03)


def test_hybrid_mse_rate_and_constant():
    # MSE ~ 2H J n^{-2H} (Bennedsen, Lunde and Pakkanen, Theorem 2.12)
    H = 0.1
    for ell, points in [(0, "left"), (1, "optimal")]:
        J = rv.hybrid_J(H, ell, points)
        n = 2**14
        assert rv.hybrid_mse(H, n, n, ell, points) / (2 * H * J * n ** (-2 * H)) == pytest.approx(1, rel=0.02)
    assert rv.hybrid_J(H, 1) < rv.hybrid_J(H, 0) / 100


def test_optimal_points_preserve_the_covariance_with_w():
    # with b_k^*, the Riemann part has the exact covariance with W: Cov(Yhat_1, W_1) = Cov(Y_1, W_1)
    H = 0.1
    _, Y, dW = rv.hybrid_rl(H, 1.0, 50, 100_000, seed=10, antithetic=True)
    assert np.mean(Y[:, -1] * dW.sum(axis=1)) == pytest.approx(rv.rl_brownian_cov(1.0, 1.0, H), abs=0.01)


# ---------------------------------------------------------------- RFSV and the GJR estimator


def test_gjr_recovers_h_on_fbm():
    # prop:rv-scaling(i): zeta_q = qH; with nu = 1, m(2, 1) estimates nu^2 = 1
    H = 0.14
    _, X = rv.fbm_davies_harte(H, T=4096.0, n_steps=4096, n_paths=1, seed=12)
    est = rv.gjr_estimate(X[0])
    assert est["H"] == pytest.approx(H, abs=0.02)
    assert np.allclose(est["zeta"] / est["qs"], H, atol=0.025)
    assert rv.gjr_m(X[0], 2.0, 1) == pytest.approx(1.0, rel=0.08)


def test_rfsv_spot_and_smoothed_proxies():
    # the 8-hour average biases H upwards (Gatheral, Jaisson and Rosenbaum, Section 3.4)
    t, X = rv.rfsv_logvol(n_days=2000, steps_per_day=96, seed=13)
    spot, proxy = rv.daily_proxies(t, X, 96, window=1 / 3)
    h_spot, h_proxy = rv.gjr_estimate(spot)["H"], rv.gjr_estimate(proxy)["H"]
    assert h_spot == pytest.approx(0.14, abs=0.025)
    assert h_proxy > h_spot + 0.02


def test_rfsv_is_the_integral_equation():
    # X_t = m - lam int_0^t (X_s - m) ds + nu W^H_t, checked on the grid with the trapezoidal rule
    lam, nu, m = 0.05, 0.3, -5.0
    t, X = rv.rfsv_logvol(lam=lam, nu=nu, m=m, n_days=200, steps_per_day=50, seed=14)
    _, B = rv.fbm_davies_harte(0.14, T=200.0, n_steps=200 * 50, n_paths=1, seed=14)
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (X[1:] + X[:-1] - 2 * m) * np.diff(t))])
    assert np.abs(X - (m - lam * integral + nu * B[0])).max() < 1e-3


def test_smoothing_factor():
    H = 0.14
    p = 2 * H + 2
    for th in (1e-3, 1e-2):                                             # f = 1 - 2 th^{2H} / (p (p-1)) + O(th^2)
        assert rv.smoothing_factor(th, H) == pytest.approx(1 - 2 * th ** (2 * H) / (p * (p - 1)), abs=2 * th ** 2)
    # direct double integral of the moving-average increment variance at Delta = 1, delta = 1/3
    d = 1 / 3
    g = lambda x: 0.5 * (abs(x + 1) ** (2 * H) + abs(x - 1) ** (2 * H) - 2 * abs(x) ** (2 * H))
    val = quad(lambda x: (d - abs(x)) * g(x), -d, d, limit=200, points=[0.0])[0] / d ** 2
    assert rv.smoothing_factor(d, H) == pytest.approx(val, rel=1e-6)


# ---------------------------------------------------------------- rough Bergomi


def test_iv_and_skew_recovers_a_linear_smile():
    T, s0, slope = 0.5, 0.2, -0.3
    k, h = 0.0, 1e-5
    c = lambda x: black_normalized(x, (s0 + slope * x) * np.sqrt(T))
    ck = (c(k + h) - c(k - h)) / (2 * h)
    iv, sk = rv.iv_and_skew(c(k), ck, k, T)
    assert iv == pytest.approx(s0, rel=1e-8)
    assert sk == pytest.approx(slope, rel=1e-5)


def test_rbergomi_forward_variance_and_martingale():
    H, eta, rho, xi = 0.1, 1.5, -0.7, 0.04
    t, Y, dW = rv.rl_cholesky(H, 1.0, 16, 50_000, seed=15)
    v = xi * np.exp(eta * Y - 0.5 * eta ** 2 * t ** (2 * H))
    assert np.allclose(v.mean(axis=0), xi, rtol=0.04)                  # E v_t = xi_0(t)
    A, V = rv.rbergomi_integrals(Y, dW, t, xi, eta, rho, H)
    assert np.exp(A).mean() == pytest.approx(1.0, abs=0.01)             # E[e^A] = 1 (exponential martingale)


def test_rbergomi_skew_small_vol_of_vol():
    # at small eta the ATM skew is close to the first-order term (rho eta / 2) E_H T^{H-1/2}
    H, eta, rho, xi = 0.1, 0.15, -0.7, 0.04
    Ts = np.array([0.1, 1.0])
    o = rv.rbergomi_smiles([0.0], Ts, H=H, eta=eta, rho=rho, xi=xi, n_steps=128, n_paths=2**14, seed=16)
    first = rv.rbergomi_first_order_skew(Ts, H, eta, rho)
    assert np.allclose(o["skew"], first, rtol=0.1)
    assert np.allclose(o["iv"][:, 0], np.sqrt(xi), rtol=0.02)


def test_rbergomi_cholesky_and_hybrid_agree():
    k = np.array([-0.1, 0.0, 0.05])
    a = rv.rbergomi_smiles(k, [0.25], n_steps=32, n_paths=2**14, seed=17, scheme="cholesky")
    b = rv.rbergomi_smiles(k, [0.25], n_steps=32, n_paths=2**14, seed=18, scheme="hybrid")
    se = np.sqrt(a["iv_se"] ** 2 + b["iv_se"] ** 2)
    assert np.all(np.abs(a["iv"] - b["iv"]) < 4 * se + 2e-3)


# ---------------------------------------------------------------- rough Heston


def test_rough_heston_reduces_to_heston():
    z = np.linspace(0, 40, 9) - 0.5j
    cf = rv.rough_heston_cf(z, 1.0, V0=0.04, theta=0.05, lam=1.5, nu=0.4, rho=-0.7, H=0.5, n_steps=200)
    ref = heston_cf(z, 1.0, 0.04, 1.5, 0.05, 0.6, -0.7)[0]
    assert np.abs(cf - ref).max() < 1e-4


def test_rough_heston_trivial_points_and_convergence():
    p = dict(V0=0.04, theta=0.04, lam=0.3, nu=0.3, rho=-0.7, H=0.1)
    assert rv.rough_heston_cf(np.array([0.0, -1j]), 0.5, **p).real == pytest.approx([1.0, 1.0])
    z = np.array([1.0, 5.0 - 0.5j, 20.0])
    a = rv.rough_heston_cf(z, 0.5, n_steps=100, **p)
    b = rv.rough_heston_cf(z, 0.5, n_steps=400, **p)
    assert np.abs(a - b).max() < 2e-4
    assert np.all(np.abs(b) <= 1 + 1e-9)


# ---------------------------------------------------------------- the public data file


def _rdata_blob(dates, values):
    """A gzip-compressed RDX2 file holding data.frame(DateID = <int>, SPX2.rv = <double>) under the name rvsp500."""
    out = bytearray(b"RDX2\nX\n")
    i32 = lambda v: struct.pack(">i", v)
    def charsxp(s):
        return i32(0x00040009) + i32(len(s)) + s.encode()
    out += i32(2) + i32(0x030500) + i32(0x020300)
    out += i32(0x00000402) + i32(1) + charsxp("rvsp500")              # pairlist cell with tag
    out += i32(0x00000313) + i32(2)                                     # list (object, attributes), length 2
    out += i32(13) + i32(len(dates)) + b"".join(i32(int(d)) for d in dates)
    out += i32(14) + i32(len(values)) + b"".join(struct.pack(">d", v) for v in values)
    out += i32(0x00000402) + i32(1) + charsxp("names")                 # attributes: names = c("DateID", "SPX2.rv")
    out += i32(16) + i32(2) + charsxp("DateID") + charsxp("SPX2.rv")
    out += i32(0x00000402) + i32(1) + charsxp("class") + i32(16) + i32(1) + charsxp("data.frame")
    out += i32(254)                                                     # end of attributes
    out += i32(254)                                                     # end of the top pairlist
    return gzip.compress(bytes(out))


def test_offline_switch(monkeypatch):
    monkeypatch.setenv("IEOR222_OFFLINE", "1")
    assert rv.load_spx_realized_variance() is None


def test_optimal_points_at_h_one_half():
    # at H = 1/2 the kernel is constant: every evaluation point is exact
    assert np.allclose(rv.optimal_points(np.arange(1, 5), 0.0), np.arange(1, 5))
    assert rv.hybrid_mse(0.5, 64, 64, 0) == pytest.approx(0.0, abs=1e-14)


def test_parse_rdata_reads_a_data_frame():
    dates, vals = [20000103, 20000104, 20000105], [1.5e-4, 3.0e-4, -1.0]
    df = rv.parse_rdata(_rdata_blob(dates, vals))["rvsp500"]
    assert list(df["DateID"]) == dates
    assert np.allclose(df["SPX2.rv"], vals)
    with pytest.raises(ValueError):
        rv.parse_rdata(gzip.compress(b"not an R file"))
