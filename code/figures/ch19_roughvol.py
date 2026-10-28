"""Figures for Chapter 19 (rough volatility). All data are simulated: the figures do not need the network."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from ieor222.roughvol import (daily_proxies, fbm_cholesky, gjr_estimate, hybrid_J, hybrid_mse, iv_and_skew, powerlaw_fit,
                              rbergomi_first_order_skew, rbergomi_smiles, rfsv_logvol, rough_heston_smile)
from ieor222.style import COBALT, GOLD, GREY, INK, PALETTE, use_style
from ieor222.volatility import heston_surface

GREEN, VIOLET = PALETTE[2], PALETTE[3]

# Rough Bergomi parameters of Bayer, Friz and Gatheral's SPX fit of 4 February 2010, with a flat forward variance.
RB = dict(H=0.07, eta=1.9, rho=-0.9, xi=0.235 ** 2)
# Rough Heston parameters of El Euch and Rosenbaum (2019), Section 5.2.
RH = dict(V0=0.4, theta=0.04, lam=2.0, nu=0.05, rho=-0.5)


def _plain_log(ax, axis="y"):
    """Decimal labels at 1, 2, 5 times powers of ten on a log axis, and no minor labels."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def fbm_paths(out):
    """fBM paths for three Hurst indices from the same normal draws (exact Cholesky sampling)."""
    n = 1000
    G = np.random.default_rng(19).standard_normal((1, n))
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.1), sharey=True)
    for ax, H, color in zip(axes, (0.1, 0.5, 0.9), (COBALT, INK, GOLD)):
        t, X = fbm_cholesky(H, 1.0, n, normals=G)
        ax.plot(t, X[0], color=color, lw=0.8)
        ax.set_title(f"$H = {H}$")
        ax.set_xlabel("$t$")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("$W^H_t$")
    fig.tight_layout()
    fig.savefig(out / "ch19-fbm.pdf")
    plt.close(fig)


def gjr(out):
    """The GJR regression on a simulated RFSV path: spot volatility and realized-variance proxies."""
    spd = 192                       # 8 grid points per hour for the averages
    t, X = rfsv_logvol(H=0.14, nu=0.3, lam=5e-4, m=-5.0, n_days=3500, steps_per_day=spd, seed=5)
    spot, proxy = daily_proxies(t, X, spd, window=1 / 3)
    _, proxy1h = daily_proxies(t, X, spd, window=1 / 24)
    e_spot, e_8h, e_1h = gjr_estimate(spot), gjr_estimate(proxy), gjr_estimate(proxy1h)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.7))
    L = np.log(e_spot["lags"])
    for i, (q, color) in enumerate(zip(e_spot["qs"], PALETTE)):
        ax.plot(L, e_spot["logm"][i], "o", ms=2.2, color=color, label=f"$q={q:g}$")
        ax.plot(L, e_spot["intercept"][i] + e_spot["zeta"][i] * L, color=color, lw=0.9)
    ax.set_xlabel(r"$\log\Delta$ ($\Delta$ in days)")
    ax.set_ylabel(r"$\log m(q,\Delta)$")
    ax.legend(fontsize=6.5, ncol=2, loc="lower right")
    qq = np.linspace(0, 3.2, 10)
    ax2.plot(qq, 0.14 * qq, color=GREY, lw=0.9, ls="--", label="$0.14\\,q$ (true $H$)")
    for e, color, mk, lab in ((e_spot, COBALT, "o", "spot"), (e_1h, GREEN, "s", "1-hour average"),
                              (e_8h, GOLD, "^", "8-hour average")):
        ax2.plot(e["qs"], e["zeta"], mk, ms=4, color=color, label=f"{lab}: $\\hat H={e['H']:.3f}$")
    ax2.set_xlabel("$q$")
    ax2.set_ylabel(r"$\zeta_q$")
    ax2.set_xlim(0, 3.2)
    ax2.set_ylim(0, None)
    ax2.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ch19-gjr.pdf")
    plt.close(fig)
    return e_spot, e_1h, e_8h


def hybrid(out):
    """Exact mean-square error of the hybrid scheme, and rough Bergomi smiles by Cholesky and by the hybrid scheme."""
    H = 0.1
    ns = 2 ** np.arange(3, 13)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for (ell, pts), color, lab in (((0, "left"), GOLD, r"$\ell=0$, $b_k=k$"),
                                   ((0, "optimal"), GREEN, r"$\ell=0$, $b_k^*$"),
                                   ((1, "optimal"), COBALT, r"$\ell=1$, $b_k^*$")):
        rmse = np.sqrt([hybrid_mse(H, n, n, ell, pts) for n in ns])
        ax.loglog(ns, np.sqrt(2 * H * hybrid_J(H, ell, pts)) * ns ** (-H), color=color, lw=1.0, ls="-", zorder=1)
        ax.loglog(ns, rmse, "o", ms=3.5, mfc="white", mec=color, mew=1.0, color=color, label=lab, zorder=2)
    ax.set_xlabel("steps $n$ on $[0,1]$")
    ax.set_ylabel(r"$\|Y_1-\hat Y_1\|_{L^2}$  ($\|Y_1\|_{L^2}=1$)")
    _plain_log(ax)
    ax.legend(fontsize=6.5, loc="center right")
    T = 0.5
    k = np.linspace(-0.3, 0.12, 8)
    for scheme, n, paths, color, mk, lab in (("cholesky", 64, 2 ** 15, INK, "s", "Cholesky, 64 steps"),
                                             ("hybrid", 64, 2 ** 15, COBALT, "o", "hybrid, 64 steps"),
                                             ("hybrid", 512, 2 ** 14, GOLD, "^", "hybrid, 512 steps")):
        o = rbergomi_smiles(k, [T], n_steps=n, n_paths=paths, seed=21, scheme=scheme, **RB)
        ax2.errorbar(k, 100 * o["iv"][0], yerr=200 * o["iv_se"][0], fmt=mk, ms=3, color=color, lw=0.8,
                     capsize=1.5, label=lab)
    ax2.set_xlabel(r"log-moneyness $k$")
    ax2.set_ylabel(r"implied volatility (\%), $T=0.5$")
    ax2.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "ch19-hybrid.pdf")
    plt.close(fig)


def heston_skew(T, v0, kappa, theta, xi, rho):
    """ATM skew d sigma_imp / dk at k = 0 in the Heston model, from Fourier prices and their k-derivative (Chapter 15)."""
    sk = []
    for TT in np.atleast_1d(T):
        s = heston_surface(np.array([0.0]), np.array([TT]), v0, kappa, theta, xi, rho)
        sk.append(iv_and_skew(s["c"][0, 0], s["c_k"][0, 0], 0.0, TT)[1])
    return np.array(sk)


def skew(out):
    """Rough Bergomi smiles and the term structure of the ATM skew, against the power law and Heston."""
    Ts = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    k = np.linspace(-0.2, 0.1, 13)
    o = rbergomi_smiles(k, Ts, n_steps=256, n_paths=2 ** 15, seed=11, **RB)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.8))
    for T, color in zip((0.05, 0.5, 2.0), (COBALT, GOLD, GREEN)):
        j = int(np.argmin(np.abs(Ts - T)))
        ok = np.isfinite(o["iv"][j])
        ax.plot(k[ok], 100 * o["iv"][j][ok], "o-", ms=2.5, lw=1.0, color=color, label=f"$T={T:g}$")
    ax.set_xlabel("log-moneyness $k$")
    ax.set_ylabel("implied volatility (\\%)")
    ax.legend(fontsize=7)
    A, beta = powerlaw_fit(Ts, o["skew"])
    TT = np.geomspace(Ts[0], Ts[-1], 100)
    ax2.errorbar(Ts, -o["skew"], yerr=2 * o["skew_se"], fmt="o", ms=3, color=COBALT, capsize=1.5, lw=0.8,
                 label="rough Bergomi, Monte Carlo")
    ax2.loglog(TT, A * TT ** beta, color=COBALT, lw=0.8, label=f"fit $\\propto T^{{{beta:.2f}}}$")
    ax2.loglog(TT, -rbergomi_first_order_skew(TT, RB["H"], RB["eta"], RB["rho"]), color=INK, ls="--", lw=0.8,
               label=r"$\frac{\rho\eta}{2}E_H T^{H-1/2}$")
    hs = heston_skew(Ts, 0.235 ** 2, 2.0, 0.235 ** 2, 0.45, -0.9)
    ax2.loglog(Ts, -hs, "s-", ms=2.5, lw=0.8, color=GOLD, label="Heston")
    ax2.set_xlabel("maturity $T$ (years)")
    ax2.set_ylabel(r"$-\partial_k\sigma_{\mathrm{imp}}(0,T)$")
    _plain_log(ax2)
    ax2.legend(fontsize=6.3, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch19-skew.pdf")
    plt.close(fig)
    return o, (A, beta), hs


def rough_heston(out):
    """Rough Heston (H = 0.1) against classical Heston (H = 1/2) with El Euch and Rosenbaum's parameters."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for T, ls in ((1 / 52, "-"), (1.0, "--")):
        k = np.linspace(-0.25, 0.25, 41) * np.sqrt(T) / np.sqrt(0.2)
        for H, color in ((0.1, COBALT), (0.5, GOLD)):
            iv, _ = rough_heston_smile(k, T, H=H, v_eff=RH["V0"] if T < 0.2 else None, **RH)
            ax.plot(k / np.sqrt(T), 100 * iv, ls=ls, color=color, lw=1.1,
                    label=f"$H={H}$, $T={'1/52' if T < 1 else '1'}$")
    ax.set_xlabel(r"$k/\sqrt{T}$")
    ax.set_ylabel("implied volatility (\\%)")
    ax.legend(fontsize=6.3)
    Ts = np.geomspace(0.005, 2.0, 12)
    for H, color, mk in ((0.1, COBALT, "o"), (0.5, GOLD, "s")):
        sk = [rough_heston_smile([0.0], T, H=H, v_eff=RH["V0"] if T < 0.2 else None, **RH)[1] for T in Ts]
        ax2.loglog(Ts, -np.array(sk), mk + "-", ms=3, lw=0.9, color=color, label=f"$H={H}$")
    hs = heston_skew(Ts, RH["V0"], RH["lam"], RH["theta"], RH["lam"] * RH["nu"], RH["rho"])
    ax2.loglog(Ts, -hs, "x", ms=4, color=INK, label="Heston, closed form")
    ax2.set_xlabel("maturity $T$ (years)")
    ax2.set_ylabel(r"$-\partial_k\sigma_{\mathrm{imp}}(0,T)$")
    _plain_log(ax2)
    ax2.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "ch19-rough-heston.pdf")
    plt.close(fig)


def main(out):
    use_style()
    fbm_paths(out)
    gjr(out)
    hybrid(out)
    skew(out)
    rough_heston(out)
