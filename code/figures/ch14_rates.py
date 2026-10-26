"""Figures for Chapter 14 (short-rate models)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma as gamma_law
from scipy.stats import norm

from ieor222.rates import (cir_bond, cir_exact, cir_long_yield, cir_stationary, cir_yield,
                           integrated_rate, mc_bond_prices, vasicek_bond, vasicek_exact,
                           vasicek_long_yield, vasicek_negative_prob, vasicek_shape_thresholds,
                           vasicek_yield, yields)
from ieor222.style import COBALT, GOLD, GREY, INK, PALETTE, use_style

GREEN = PALETTE[2]

# The Vasicek model of the yield-curve and Monte Carlo figures, and a CIR model with the same mean reversion, the
# same long-run level, and the same volatility of rates at r = theta: sigma_cir sqrt(theta) = sigma.
KAPPA, THETA, SIGMA = 0.15, 0.05, 0.025
SIGMA_CIR = SIGMA / np.sqrt(THETA)


def yield_curves(out):
    """Vasicek and CIR yield curves from three starting rates (one in each Vasicek shape regime)."""
    tau = np.linspace(0.0, 30.0, 301)
    lo, hi = vasicek_shape_thresholds(KAPPA, THETA, SIGMA)
    assert 0.01 <= lo < 0.04 < hi <= 0.08
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.1), sharey=True)
    starts = ((0.01, COBALT, "-"), (0.04, GOLD, (0, (5, 2))), (0.08, GREEN, (0, (1.5, 1.5))))
    for ax, name in zip(axes, ("Vasicek", "CIR")):
        for r0, c, ls in starts:
            if name == "Vasicek":
                y = vasicek_yield(r0, tau, KAPPA, THETA, SIGMA)
            else:
                y = cir_yield(r0, tau, KAPPA, THETA, SIGMA_CIR)
            ax.plot(tau, 100 * y, color=c, ls=ls, label=f"$r_0={100 * r0:g}\\%$")
        R = (vasicek_long_yield(KAPPA, THETA, SIGMA) if name == "Vasicek"
             else cir_long_yield(KAPPA, THETA, SIGMA_CIR))
        ax.axhline(100 * R, color=GREY, lw=0.8, ls=(0, (4, 3)), label="long yield")
        ax.set_xlim(0, 30)
        ax.set_xlabel("maturity $T-t$ (years)")
        ax.set_title(name, fontsize=9)
    axes[0].set_ylabel("yield (%)")
    axes[0].legend(loc="upper right", fontsize=7, ncol=2, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(out / "ch14-yield-curves.pdf")
    plt.close(fig)


def monte_carlo(out):
    """Monte Carlo bond prices, by exact simulation, against the closed forms."""
    r0, T, n, M = 0.03, 10.0, 200, 20_000
    t, _, I_v = vasicek_exact(r0, KAPPA, THETA, SIGMA, T, n, n_paths=M, seed=14)
    t2, r_c = cir_exact(r0, KAPPA, THETA, SIGMA_CIR, T, n, n_paths=M, seed=15)
    I_c = integrated_rate(t2, r_c)
    keep = np.arange(10, n + 1, 10)                 # maturities 0.5, 1, ..., 10
    tau = t[keep]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.1))
    series = (("Vasicek", I_v, lambda x: vasicek_bond(r0, x, KAPPA, THETA, SIGMA), COBALT, "o"),
              ("CIR", I_c, lambda x: cir_bond(r0, x, KAPPA, THETA, SIGMA_CIR), GOLD, "s"))
    fine = np.linspace(0.05, T, 200)
    for k, (name, I, closed, c, mk) in enumerate(series):
        P, se = mc_bond_prices(I[:, keep])
        ax.plot(fine, 100 * yields(closed(fine), fine), color=c, lw=1.4, label=f"{name}, closed form")
        ax.plot(tau, 100 * yields(P, tau), ls="none", marker=mk, ms=3.2, mfc="white", mec=c,
                label=f"{name}, Monte Carlo")
        err = 1e4 * (yields(P, tau) - yields(closed(tau), tau))
        band = 1e4 * 1.96 * se / (P * tau)          # delta method: dY = -dP / (P tau)
        ax2.errorbar(tau + (k - 0.5) * 0.12, err, yerr=band, ls="none", marker=mk, ms=3, color=c,
                     mfc="white", elinewidth=1.0, capsize=0, label=name)
    ax2.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("maturity $T$ (years)")
    ax.set_ylabel("yield (%)")
    ax.legend(loc="lower right", fontsize=7)
    ax2.set_xlabel("maturity $T$ (years)")
    ax2.set_ylabel("Monte Carlo $-$ exact (bp)")
    ax2.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "ch14-monte-carlo.pdf")
    plt.close(fig)


def negative_rates(out):
    """Vasicek: probability of a negative short rate, and the yield curve, for three volatilities."""
    kappa, theta, r0 = 0.2, 0.03, 0.01
    t = np.linspace(0.01, 30.0, 300)
    tau = np.linspace(0.0, 60.0, 301)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.1))
    for sigma, c, ls in ((0.01, COBALT, "-"), (0.025, GOLD, (0, (5, 2))), (0.05, GREEN, (0, (1.5, 1.5)))):
        lab = f"$\\sigma={100 * sigma:g}\\%$"
        ax.plot(t, vasicek_negative_prob(t, r0, kappa, theta, sigma), color=c, ls=ls, label=lab)
        ax.axhline(norm.cdf(-theta * np.sqrt(2 * kappa) / sigma), color=c, lw=0.6, alpha=0.6)
        ax2.plot(tau, 100 * vasicek_yield(r0, tau, kappa, theta, sigma), color=c, ls=ls, label=lab)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 0.4)
    ax.set_xlabel("$t$ (years)")
    ax.set_ylabel(r"$\mathbb{Q}(r_t<0)$")
    ax.legend(loc="lower right", fontsize=7.5)
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.set_xlim(0, 60)
    ax2.set_xlabel("maturity $T$ (years)")
    ax2.set_ylabel("yield $Y(0,T)$ (%)")
    fig.tight_layout()
    fig.savefig(out / "ch14-negative-rates.pdf")
    plt.close(fig)


def feller(out):
    """CIR paths with and without the Feller condition, and the stationary densities."""
    kappa, theta, r0, T = 0.5, 0.04, 0.04, 5.0
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.1), gridspec_kw={"width_ratios": [1.35, 1]})
    for nu, c, seed in ((3.0, COBALT, 3), (0.5, GOLD, 4)):
        sigma = np.sqrt(2 * kappa * theta / nu)
        t, r = cir_exact(r0, kappa, theta, sigma, T, 2_000, n_paths=3, seed=seed)
        for k, row in enumerate(r):
            ax.plot(t, 100 * row, color=c, lw=0.8, alpha=0.9,
                    label=f"$\\nu={nu:g}$, $\\sigma={100 * sigma:.1f}\\%$" if k == 0 else None)
    ax.set_xlim(0, T)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("$t$ (years)")
    ax.set_ylabel("short rate (%)")
    ax.set_ylim(0, 1.3 * ax.get_ylim()[1])
    ax.legend(loc="upper left", fontsize=7.5, ncol=2, frameon=True, framealpha=1.0, edgecolor="none")

    x = np.linspace(1e-4, 0.15, 600)
    for nu, c, ls in ((0.5, GOLD, "-"), (1.0, GREY, (0, (1.5, 1.5))), (3.0, COBALT, "-"),
                      (8.0, GREEN, (0, (5, 2)))):
        sigma = np.sqrt(2 * kappa * theta / nu)
        shape, scale = cir_stationary(kappa, theta, sigma)
        ax2.plot(100 * x, gamma_law.pdf(x, shape, scale=scale) / 100, color=c, ls=ls,
                 label=f"$\\nu={nu:g}$")
    ax2.set_xlim(0, 15)
    ax2.set_ylim(0, 0.35)
    ax2.set_xlabel("short rate (%)")
    ax2.set_ylabel("stationary density")
    ax2.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "ch14-feller.pdf")
    plt.close(fig)


def main(out):
    use_style()
    yield_curves(out)
    monte_carlo(out)
    negative_rates(out)
    feller(out)
