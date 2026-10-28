"""Checks of the Chapter 18 code against the results proved or stated in the notes."""
import numpy as np
import pytest
from scipy import stats
from scipy.linalg import expm

from ieor222 import hawkes as H

UNI = dict(mu=0.5, alpha=0.6, beta=2.0)


# ------------------------------------------------------------- mean intensity (prop:hk-stationary)


def test_mean_intensity_is_the_neumann_series():
    mu, G, _ = H.lob_toy_model()
    series = sum(np.linalg.matrix_power(G, n) @ mu for n in range(400))
    assert np.allclose(H.mean_intensity(mu, G), series, rtol=1e-12)
    assert H.spectral_radius(G) < 1
    with pytest.raises(ValueError):
        H.mean_intensity([1.0], [[1.0]])


def test_transient_mean_matches_closed_form_for_a_common_decay():
    # With beta_ij = beta: m(t) = Lambda + exp(-beta (I - G) t) (mu - Lambda).
    mu, G, beta = H.lob_toy_model()
    Lam = H.mean_intensity(mu, G)
    ts = np.array([0.0, 0.1, 1.0, 5.0, 50.0])
    m, EN = H.exp_hawkes_mean(ts, mu, G, beta)
    for t, mt in zip(ts, m):
        assert np.allclose(mt, Lam + expm(-4.0 * (np.eye(6) - G) * t) @ (mu - Lam), atol=1e-10)
    # E N(t) = Lambda t - (I - G)^{-1} (Lambda - mu) / beta + o(1)
    corr = np.linalg.solve(np.eye(6) - G, Lam - mu) / 4.0
    assert np.allclose(EN[-1], Lam * 50.0 - corr, atol=1e-8)
    # univariate: E N(T) = Lambda T - mu alpha (1 - e^{-beta (1 - alpha) T}) / (beta (1 - alpha)^2)
    T = 7.0
    a, b, mu1 = UNI["alpha"], UNI["beta"], UNI["mu"]
    exact = mu1 / (1 - a) * T - mu1 * a * (1 - np.exp(-b * (1 - a) * T)) / (b * (1 - a) ** 2)
    assert H.exp_hawkes_mean([T], mu1, a, b)[1][0, 0] == pytest.approx(exact, rel=1e-12)


def test_divergence_when_the_branching_ratio_is_one():
    _, EN = H.exp_hawkes_mean([10.0, 100.0, 1000.0], 1.0, 1.0, 2.0)
    m, _ = H.exp_hawkes_mean([10.0, 100.0, 1000.0], 1.0, 1.0, 2.0)
    assert np.all(np.diff(m[:, 0]) > 0) and m[-1, 0] > 1000      # E lambda(t) = mu (1 + beta t)
    assert m[-1, 0] == pytest.approx(1.0 + 2.0 * 1000.0)


# ------------------------------------------------------------- simulation (prop:hk-ogata, prop:hk-branching)


def test_ogata_univariate_mean_count():
    T, n = 50.0, 1500
    counts = np.array([H.simulate_ogata(T, **UNI, seed=s)[0].size for s in range(n)])
    exact = H.exp_hawkes_mean([T], **UNI)[1][0, 0]
    assert abs(counts.mean() - exact) < 4 * counts.std() / np.sqrt(n)


def test_ogata_multivariate_counts_match_the_mean():
    mu, G, beta = H.lob_toy_model()
    T, n = 100.0, 150
    counts = np.array([np.bincount(H.simulate_ogata(T, mu, G, beta, seed=s)[1], minlength=6)
                       for s in range(n)])
    exact = H.exp_hawkes_mean([T], mu, G, beta)[1][0]
    assert np.all(np.abs(counts.mean(0) - exact) < 4 * counts.std(0) / np.sqrt(n))


def test_branching_and_thinning_give_the_same_law():
    T, n = 30.0, 1500
    og = np.array([H.simulate_ogata(T, **UNI, seed=10_000 + s)[0].size for s in range(n)])
    _, _, p = H.simulate_branching(T, [UNI["mu"]], [[UNI["alpha"]]], H.exponential_lags(UNI["beta"]),
                                   seed=4, n_paths=n)
    br = np.bincount(p, minlength=n)
    assert stats.ks_2samp(og, br).pvalue > 0.001
    exact = H.exp_hawkes_mean([T], **UNI)[1][0, 0]
    assert abs(br.mean() - exact) < 4 * br.std() / np.sqrt(n)
    # the generations: a cluster has on average 1 / (1 - alpha) events (Galton--Watson)
    t, k, g = H.simulate_branching(5000.0, [0.2], [[0.6]], H.exponential_lags(2.0), seed=5,
                                   return_generation=True)
    assert np.sum(g == 0) * 1 / (1 - 0.6) == pytest.approx(t.size, rel=0.06)


# ------------------------------------------------------------- likelihood (prop:hk-mle)


def _brute_loglik(theta, t, T):
    mu, a, b = theta
    lam = np.array([mu + a * b * np.sum(np.exp(-b * (s - t[t < s]))) for s in t])
    return np.sum(np.log(lam)) - mu * T - a * np.sum(1 - np.exp(-b * (T - t)))


def test_loglik_recursion_and_derivatives():
    T = 150.0
    t, k = H.simulate_ogata(T, **UNI, seed=1)
    th = np.array([0.4, 0.7, 1.5])
    ll, g, Hs = H.exp_loglik(th, t, T, derivatives=True)
    assert ll == pytest.approx(_brute_loglik(th, t, T), rel=1e-12)
    assert ll == pytest.approx(H.multivariate_loglik(t, k, T, *th), rel=1e-12)
    eps = 1e-6
    gfd = [(_brute_loglik(th + eps * e, t, T) - _brute_loglik(th - eps * e, t, T)) / (2 * eps) for e in np.eye(3)]
    assert np.allclose(g, gfd, rtol=1e-6, atol=1e-6)
    Hfd = np.array([(H.exp_loglik(th + eps * e, t, T, True)[1] - H.exp_loglik(th - eps * e, t, T, True)[1])
                    / (2 * eps) for e in np.eye(3)])
    assert np.allclose(Hs, Hfd, rtol=1e-5, atol=1e-5)


def test_mle_recovers_the_parameters():
    T = 4000.0
    t, _ = H.simulate_ogata(T, **UNI, seed=3)
    f = H.fit_exp_hawkes(t, T)
    truth = np.array([UNI["mu"], UNI["alpha"], UNI["beta"]])
    assert np.all(np.abs(f["theta"] - truth) < 3 * f["se"])
    # the standard errors shrink like T^{-1/2}: compare with a path four times shorter
    f4 = H.fit_exp_hawkes(t[t <= T / 4], T / 4)
    assert np.allclose(f4["se"] / f["se"], 2.0, rtol=0.35)


def test_mle_coverage_in_a_small_study():
    est, se = H.mle_study(40, 600.0, **UNI, seed=2)
    truth = np.array([UNI["mu"], UNI["alpha"], UNI["beta"]])
    cover = np.mean(np.abs(est - truth) < 1.96 * se, axis=0)
    assert np.all(cover >= 0.8)


def test_fixed_decay_likelihood_is_concave_and_matches_counts():
    mu, G, beta = H.lob_toy_model()
    T = 1500.0
    t, k = H.simulate_ogata(T, mu, G, beta, seed=8)
    fit = H.fit_fixed_beta(t, k, T, beta, 6)
    X, c = H.design_fixed_beta(t, k, T, beta, 6)
    for i in range(6):
        th = np.concatenate([[fit["mu"][i]], fit["alpha"][i]])
        lam = X[i] @ th
        hess = -(X[i] / lam[:, None]).T @ (X[i] / lam[:, None])
        assert np.all(np.linalg.eigvalsh(hess) <= 1e-9)
        # at the maximum the fitted compensator at T equals the number of type-i events
        assert th @ c[i] == pytest.approx(X[i].shape[0], rel=1e-5)
    z = (fit["alpha"] - G) / np.where(fit["se_alpha"] > 0, fit["se_alpha"], np.inf)
    assert np.all(np.abs(z[G > 0]) < 4)
    assert np.all(np.abs(fit["mu"] - mu) < 4 * fit["se_mu"])


# ------------------------------------------------------------- time change (thm:hk-time-change)


def test_time_changed_gaps_are_standard_exponential():
    T = 2000.0
    t, k = H.simulate_ogata(T, **UNI, seed=11)
    gaps = H.time_changed_gaps(t, k, UNI["mu"], UNI["alpha"], UNI["beta"])[0]
    assert H.ks_exponential(gaps)[1] > 0.01
    assert gaps.mean() == pytest.approx(1.0, abs=4 / np.sqrt(gaps.size))
    poisson = np.diff(np.concatenate([[0.0], t])) * t.size / T
    assert H.ks_exponential(poisson)[1] < 1e-6
    # multivariate, each component separately
    mu, G, beta = H.lob_toy_model()
    t, k = H.simulate_ogata(1500.0, mu, G, beta, seed=12)
    pv = [H.ks_exponential(g)[1] for g in H.time_changed_gaps(t, k, mu, G, beta)]
    assert min(pv) > 0.001


# ------------------------------------------------------------- nearly unstable regime (thm:hk-scaling-limit)


def test_rescaled_intensity_moments_converge_to_cir():
    mu, lam, beta = 1.0, 1.0, 0.5
    shape, scale = H.cir_from_zero_gamma(mu, lam, 1 / beta)
    errs = []
    for T in (10.0, 100.0, 1000.0, 10000.0):
        a = 1 - lam / T
        m, v = H.exp_hawkes_intensity_moments([T], mu, a, beta)
        errs.append((abs((1 - a) * m[0] - shape * scale), abs((1 - a) ** 2 * v[0] - shape * scale**2)))
    errs = np.array(errs)
    assert np.all(np.diff(errs, axis=0) < 0) and errs[-1].max() < 1e-3
    # CIR mean mu (1 - e^{-lam/m}) and variance mu (1 - e^{-lam/m})^2 / (2 m) at t = 1
    assert shape * scale == pytest.approx(mu * (1 - np.exp(-lam * beta)))
    assert shape * scale**2 == pytest.approx(mu * (1 - np.exp(-lam * beta)) ** 2 * beta / 2)


def test_nearly_unstable_simulation_matches_the_exact_prelimit_moments():
    # compare with the exact moments at finite T (the limit law differs by an O(1/T) bias)
    T, n = 40.0, 1500
    C = H.nearly_unstable_light(T, 1.0, 1.0, 0.5, n_paths=n, seed=6)
    m, v = H.exp_hawkes_intensity_moments([T], 1.0, 1 - 1 / T, 0.5)
    mean, var = m[0] / T, v[0] / T**2
    assert abs(C.mean() - mean) < 4 * C.std() / np.sqrt(n)
    se_var = np.sqrt((np.mean((C - C.mean()) ** 4) - C.var() ** 2) / n)
    assert abs(C.var() - var) < 4 * se_var


def test_heavy_tail_scaling_and_mean():
    a, muT, sc = H.heavy_tail_scaling(1000.0, 0.6)
    assert 0 < a < 1 and muT > 0
    _, _, p = H.simulate_branching(1000.0, [muT], [[a]], H.lomax_lags(0.6), seed=7, n_paths=600)
    X1 = sc * np.bincount(p, minlength=600)
    pre = H.heavy_tail_mean_prelimit(1000.0, 0.6)
    assert abs(X1.mean() - pre) < 4 * X1.std() / np.sqrt(600)
    # the exact mean moves towards the limit int_0^1 F^{0.6,1} as T grows
    lim = H.heavy_tail_mean_limit(1.0, 0.6)
    assert abs(H.heavy_tail_mean_prelimit(1e4, 0.6, h=0.5) - lim) < abs(pre - lim)
    # the Lomax sampler has the tail (1 + x)^{-tail}
    lags = H.lomax_lags(0.6)(np.random.default_rng(0), 0, 0, 200_000)
    assert np.mean(lags > 10.0) == pytest.approx(11.0 ** -0.6, rel=0.02)


def test_binned_fbm_variogram():
    assert np.allclose(H.binned_fbm_variogram([1, 2, 7], 0.5), np.array([1, 2, 7]) - 1 / 3)
    # Monte Carlo: bin averages of a Brownian path
    rng = np.random.default_rng(1)
    W = np.cumsum(rng.normal(size=(400, 2000)), axis=1) / np.sqrt(2000)
    Y = W.reshape(400, 200, 10).mean(axis=2) * np.sqrt(200)          # h = 1/200, h^{2H} = 1/200
    v = np.array([np.mean((Y[:, k:] - Y[:, :-k]) ** 2) for k in (1, 3)])
    assert np.allclose(v, H.binned_fbm_variogram([1, 3], 0.5), rtol=0.05)


# ------------------------------------------------------------- order book data


def test_lobster_parser_codes():
    msg = np.array([
        [34200.0, 1, 1, 100, 5850000, 1],     # limit buy        -> LB (2)
        [34200.5, 1, 2, 100, 5851000, -1],    # limit sell       -> LA (3)
        [34201.0, 4, 2, 100, 5851000, -1],    # sell order hit   -> market buy (0)
        [34201.0, 4, 3, 50, 5851000, -1],     # same market order, merged
        [34202.0, 3, 1, 100, 5850000, 1],     # deletion at bid  -> CB (4)
        [34203.0, 2, 4, 10, 5852000, -1],     # partial cancel   -> CA (5)
        [34204.0, 4, 5, 100, 5850000, 1],     # buy order hit    -> market sell (1)
        [34205.0, 5, 0, 100, 5850000, 1],     # hidden execution: dropped
    ])
    t, k = H.lobster_to_events(msg)
    assert k.tolist() == [2, 3, 0, 4, 5, 1]
    assert np.allclose(t, [0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
