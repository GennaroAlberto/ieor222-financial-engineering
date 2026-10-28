"""Figures for Chapter 18 (Hawkes processes and limit order books)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ieor222 import hawkes as H
from ieor222.style import COBALT, GOLD, GREY, INK, PALETTE, use_style

GREEN = PALETTE[2]
UNI = dict(mu=0.5, alpha=0.6, beta=2.0)


def path(out):
    """A Hawkes path from the cluster construction, its intensity, and a Poisson path with the same rate."""
    T = 60.0
    t, k, g = H.simulate_branching(T, [0.3], [[0.75]], H.exponential_lags(1.5), seed=14,
                                   return_generation=True)
    grid = np.linspace(0, T, 3001)
    lam = H.intensity_on_grid(grid, t, k, 0.3, 0.75, 1.5)[:, 0]
    Lam = 0.3 / (1 - 0.75)
    rng = np.random.default_rng(3)
    tp = np.sort(rng.uniform(0, T, rng.poisson(Lam * T)))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 2.85), sharex=True,
                                  gridspec_kw={"height_ratios": [2.1, 1]})
    ax.plot(grid, lam, color=COBALT, lw=1.1, label=r"intensity $\lambda(t)$")
    ax.axhline(Lam, color=INK, lw=0.8, ls="--", label=r"mean rate $\mu/(1-\|\varphi\|)$")
    ax.set_ylabel("events per unit time")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_ylim(0, None)
    imm = g == 0
    ax2.vlines(t[imm], 1.55, 2.45, color=COBALT, lw=0.9, label="immigrants")
    ax2.vlines(t[~imm], 1.55, 2.45, color=GOLD, lw=0.9, label="descendants")
    ax2.vlines(tp, 0.15, 1.05, color=GREY, lw=0.9, label="Poisson, same rate")
    ax2.set_yticks([])
    ax2.set_ylim(0, 2.6)
    ax2.grid(False)
    ax2.set_xlabel("$t$")
    ax2.set_xlim(0, T)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.78), ncol=3, fontsize=8)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(out / "ch18-path.pdf")
    plt.close(fig)
    return dict(n=t.size, immigrants=int(imm.sum()), poisson=tp.size)


def estimation(out, n_rep=100):
    """Time-change QQ plots under the fitted Hawkes and Poisson models, and the MLE sampling distribution."""
    T = 1000.0
    truth = np.array([UNI["mu"], UNI["alpha"], UNI["beta"]])
    t, k = H.simulate_ogata(T, **UNI, seed=21)
    fit = H.fit_exp_hawkes(t, T)
    gaps_h = H.time_changed_gaps(t, k, *fit["theta"])[0]
    gaps_p = np.diff(np.concatenate([[0.0], t])) * t.size / T
    p_h, p_p = H.ks_exponential(gaps_h)[1], H.ks_exponential(gaps_p)[1]
    est, se = H.mle_study(n_rep, T, **UNI, seed=18)
    z = (est - truth) / se

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.45))
    q = (np.arange(1, gaps_h.size + 1) - 0.5) / gaps_h.size
    ex = -np.log1p(-q)
    ax.plot(ex, np.sort(gaps_p), color=GOLD, lw=1.6, label=f"Poisson fit, KS $p$ = {p_p:.0e}")
    ax.plot(ex, np.sort(gaps_h), color=COBALT, lw=1.6, label=f"Hawkes fit, KS $p$ = {p_h:.2f}")
    lim = max(ex.max(), gaps_p.max()) * 1.02
    ax.plot([0, lim], [0, lim], color=INK, lw=0.8, ls="--")
    ax.set_xlim(0, ex.max() * 1.02)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Exp(1) quantiles")
    ax.set_ylabel("transformed inter-arrival times")
    ax.legend(loc="upper left", fontsize=7.5)

    xs = np.linspace(-3.2, 3.2, 300)
    ax2.plot(xs, stats.norm.cdf(xs), color=INK, lw=1.0, ls="--", label=r"$N(0,1)$")
    for j, (name, col) in enumerate(((r"$\hat\mu$", COBALT), (r"$\hat\alpha$", GOLD), (r"$\hat\beta$", GREEN))):
        zz = np.sort(z[:, j])
        ax2.step(zz, np.arange(1, zz.size + 1) / zz.size, where="post", color=col, lw=1.4, label=name)
    ax2.set_xlim(-3.2, 3.2)
    ax2.set_xlabel(r"$(\hat\theta-\theta)/\mathrm{se}(\hat\theta)$")
    ax2.set_ylabel("empirical distribution function")
    ax2.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out / "ch18-estimation.pdf")
    plt.close(fig)
    cover = np.mean(np.abs(z) < 1.96, axis=0)
    return dict(n=t.size, theta=fit["theta"], se=fit["se"], p_hawkes=p_h, p_poisson=p_p,
                mean=est.mean(0), sd=est.std(0, ddof=1), mean_se=se.mean(0), coverage=cover)


def lob(out):
    """Toy order book: event rates against (I - G)^{-1} mu, and the mid-price."""
    mu, G, beta = H.lob_toy_model()
    T = 4 * 3600.0
    t, k = H.simulate_ogata(T, mu, G, beta, seed=2026)
    Lam = H.mean_intensity(mu, G)
    emp = np.bincount(k, minlength=6) / T
    mid = H.lob_mid_price(t, k)
    fit = H.fit_fixed_beta(t, k, T, beta, 6)
    pv = [H.ks_exponential(gg)[1] for gg in H.time_changed_gaps(t, k, fit["mu"], fit["alpha"], beta)]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.35), gridspec_kw={"width_ratios": [1, 1.35]})
    x = np.arange(6)
    ax.bar(x - 0.2, emp, width=0.38, color=COBALT, label="simulated, 4 hours")
    ax.bar(x + 0.2, Lam, width=0.38, color=GOLD, label=r"$(I-G)^{-1}\mu$")
    ax.bar(x + 0.2, mu, width=0.38, color="white", alpha=0.55, hatch="////", edgecolor=GOLD, lw=0,
           label=r"baseline $\mu$")
    ax.set_xticks(x, H.LOB_TYPES, fontsize=8)
    ax.set_ylabel("events per second")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_ylim(0, max(emp.max(), Lam.max()) * 1.45)
    ax.grid(axis="x", visible=False)
    ax2.step(t / 3600, mid / 0.01, where="post", color=COBALT, lw=1.0)
    ax2.set_xlabel("time (hours)")
    ax2.set_ylabel("mid-price change (ticks)")
    ax2.set_xlim(0, 4)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out / "ch18-lob.pdf")
    plt.close(fig)
    return dict(n=t.size, rho=H.spectral_radius(G), Lam=Lam, total=Lam.sum(), emp=emp,
                moves=int(np.sum(np.diff(mid) != 0)), max_alpha_err=np.abs(fit["alpha"] - G).max(),
                max_alpha_z=np.max(np.abs(fit["alpha"] - G)[G > 0] / fit["se_alpha"][G > 0]),
                ks_min=min(pv))


def nearly_unstable(out, n_vario=8):
    """Light tails: C^T_1 against the Gamma law of the CIR limit. Light versus heavy tails: rescaled event rates."""
    mu, lam, beta = 1.0, 1.0, 0.5
    T = 50.0
    C = H.nearly_unstable_light(T, mu, lam, beta, n_paths=3000, seed=6)
    shape, scale = H.cir_from_zero_gamma(mu, lam, 1 / beta)
    nb, lags = 500, np.arange(1, 21)
    Tl, Th, tail = 2000.0, 1e6, 0.6
    al = 1 - lam / Tl
    a_h, mu_h, sc_h = H.heavy_tail_scaling(Th, tail, lam)
    vl, vh = [], []
    for s in range(n_vario):
        tl, _ = H.simulate_branching(Tl, [mu], [[al]], H.exponential_lags(beta), seed=100 + s)
        yl, nl = H.binned_rates(tl, Tl, (1 - al) / Tl, nb)
        th, _ = H.simulate_branching(Th, [mu_h], [[a_h]], H.lomax_lags(tail), seed=200 + s)
        yh, nh = H.binned_rates(th, Th, sc_h, nb)
        if s == 0:
            path_l, path_h = yl, yh
        vl.append(H.binned_variogram(yl[nb // 5:], lags, nl))
        vh.append(H.binned_variogram(yh[nb // 5:], lags, nh))
    vl, vh = np.mean(vl, axis=0), np.mean(vh, axis=0)

    fig = plt.figure(figsize=(6.6, 2.25))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.3, 1], hspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    ax2a, ax2b = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[:, 2])
    ax.hist(C, bins=40, density=True, color=COBALT, alpha=0.35, label=f"$C^T_1$, $T={T:.0f}$")
    xs = np.linspace(1e-3, C.max(), 300)
    ax.plot(xs, stats.gamma(shape, scale=scale).pdf(xs), color=INK, lw=1.3, label="CIR limit")
    ax.set_xlabel("$C^T_1$")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    u = (np.arange(nb) + 0.5) / nb
    for a2, y, col, name in ((ax2a, path_l, COBALT, "light tail"), (ax2b, path_h, GOLD, r"heavy tail, $\alpha=0.6$")):
        a2.plot(u, y / y.mean(), color=col, lw=0.7)
        a2.set_xlim(0, 1)
        a2.set_ylim(0, None)
        a2.text(0.02, 0.93, name, transform=a2.transAxes, fontsize=7, va="top", color=INK,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
        a2.set_xticks([0, 0.5, 1])
    ax2a.set_xticklabels([])
    ax2b.set_xlabel("rescaled time $t$")
    fig.text(0.333, 0.58, "rate / mean", rotation=90, va="center", ha="center", fontsize=9, color=INK)
    for v, col, Hh, name in ((vl, COBALT, 0.5, "light"), (vh, GOLD, tail - 0.5, "heavy")):
        ax3.plot(lags, v / v[0], "o", ms=3, color=col, label=f"{name} tail")
        ax3.plot(lags, H.binned_fbm_variogram(lags, Hh) / H.binned_fbm_variogram(1, Hh), color=col, lw=1.0,
                 label=f"$H={Hh:g}$" if Hh != 0.5 else r"$H=\frac{1}{2}$")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("lag (bins)")
    ax3.set_ylabel("variogram / lag-1 value")
    ax3.legend(fontsize=6.5, loc="upper left", handlelength=1.5)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.97, wspace=0.42)
    fig.savefig(out / "ch18-nearly-unstable.pdf")
    plt.close(fig)
    m_ex, v_ex = H.exp_hawkes_intensity_moments([T], mu, 1 - lam / T, beta)
    # first moment of the heavy-tailed limit: E[X^T_1] against int_0^1 F^{0.6,1}
    a_s, mu_s, sc_s = H.heavy_tail_scaling(1000.0, tail, lam)
    _, _, p = H.simulate_branching(1000.0, [mu_s], [[a_s]], H.lomax_lags(tail), seed=7, n_paths=600)
    X1 = sc_s * np.bincount(p, minlength=600)
    naive_H = np.polyfit(np.log(lags), np.log(vh), 1)[0] / 2
    return dict(C_mean=C.mean(), C_var=C.var(), exact_mean=(lam / T) * m_ex[0],
                exact_var=(lam / T) ** 2 * v_ex[0], cir_mean=shape * scale, cir_var=shape * scale**2,
                ks=stats.kstest(C, stats.gamma(shape, scale=scale).cdf).pvalue,
                vario_light=vl / vl[0], vario_heavy=vh / vh[0], X1_mean=X1.mean(),
                X1_se=X1.std() / np.sqrt(X1.size), X1_limit=H.heavy_tail_mean_limit(1.0, tail, lam),
                X1_prelimit=[H.heavy_tail_mean_prelimit(TT, tail, lam) for TT in (1e3, 1e4)],
                naive_H_heavy=naive_H)


def main(out):
    use_style()
    info = {}
    info["path"] = path(out)
    info["estimation"] = estimation(out)
    info["lob"] = lob(out)
    info["nearly_unstable"] = nearly_unstable(out)
    return info


if __name__ == "__main__":
    import pathlib
    import pprint
    pprint.pprint(main(pathlib.Path("latex/figures")))
