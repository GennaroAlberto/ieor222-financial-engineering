"""Figures for Chapter 9 (stochastic differential equations)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieor222.sde import (absorbed_fraction, cev_paths, gbm_euler_errors, ou_exact, ou_mean,
                         ou_stationary, ou_var)
from ieor222.style import COBALT, GOLD, GREY, INK, PALETTE, use_style

GREEN = PALETTE[2]


def euler_convergence(out):
    """Strong and weak errors of Euler--Maruyama for GBM, against the exact solution."""
    S0, mu, sigma, T = 1.0, 0.8, 0.5, 1.0
    r = gbm_euler_errors(S0, mu, sigma, T, levels=range(1, 11), n_paths=100_000, seed=2026)
    h = r["h"]
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    a = ax.errorbar(h, r["strong"], yerr=1.96 * r["strong_se"], color=COBALT, marker="o", ms=4,
                    lw=1.6, capsize=0, label=r"strong error $\mathbb{E}|S_T-S^h_T|$ (Monte Carlo)")
    b = ax.errorbar(h, np.abs(r["weak"]), yerr=1.96 * r["weak_se"], color=GOLD, marker="s", ms=4,
                    lw=0, elinewidth=1.2, capsize=0,
                    label=r"weak error $|\mathbb{E}S^h_T-\mathbb{E}S_T|$ (Monte Carlo)")
    (c,) = ax.plot(h, np.abs(r["weak_exact"]), color=GOLD, lw=1.2,
                   label=r"weak error, exact: $S_0|e^{\mu T}-(1+\mu h)^{T/h}|$")
    # reference slopes through the smallest step, labelled at the small-h end
    for slope, y0, txt in ((0.5, r["strong"][-1], "slope 1/2"), (1.0, np.abs(r["weak_exact"][-1]), "slope 1")):
        ref = y0 * (h / h[-1]) ** slope / 1.7
        ax.plot(h, ref, color=GREY, lw=0.9, ls=(0, (4, 3)))
        ax.text(h[-2], ref[-2] / 1.3, txt, color=GREY, fontsize=8, ha="center", va="top")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("step $h$")
    ax.set_ylabel("error at $T=1$")
    ax.legend(handles=[a, b, c], loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "ch09-euler.pdf")
    plt.close(fig)


def ou_paths(out):
    """Vasicek short-rate paths from several starting points, and the stationary law."""
    kappa, theta, sigma = 1.5, 0.04, 0.02
    m, v = ou_stationary(kappa, theta, sigma)
    T, n = 4.0, 800
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax.axhspan(100 * (m - 2 * np.sqrt(v)), 100 * (m + 2 * np.sqrt(v)), color=COBALT, alpha=0.08, lw=0)
    ax.axhline(100 * theta, color=INK, lw=0.8, alpha=0.5)
    for k, (x0, c) in enumerate(((0.11, GOLD), (0.04, COBALT), (-0.02, GREEN))):
        t, X = ou_exact(x0, kappa, theta, sigma, T, n, n_paths=2, seed=10 + k)
        ax.plot(t, 100 * X.T, color=c, lw=0.9, alpha=0.9)
        ax.plot(t, 100 * ou_mean(x0, kappa, theta, t), color=c, lw=1.4, ls=(0, (4, 2)),
                label=f"$r_0={100 * x0:g}\\%$")
    ax.set_xlim(0, T)
    ax.set_xlabel("$t$ (years)")
    ax.set_ylabel("short rate (%)")
    ax.legend(loc="upper right", fontsize=7.5, ncol=3, handlelength=1.6, columnspacing=1.0)

    t, X = ou_exact(0.11, kappa, theta, sigma, T, 40, n_paths=100_000, seed=20)
    ax2.hist(100 * X[:, -1], bins=60, density=True, color=COBALT, alpha=0.35,
             label="simulated $r_4$, $r_0=11\\%$")
    x = np.linspace(100 * (m - 4.5 * np.sqrt(v)), 100 * (m + 4.5 * np.sqrt(v)), 300)
    sd = 100 * np.sqrt(v)
    ax2.plot(x, np.exp(-0.5 * ((x - 100 * m) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color=INK,
             lw=1.4, label="stationary law")
    ax2.set_xlabel("short rate (%)")
    ax2.set_ylabel("density")
    ax2.legend(loc="upper left", fontsize=7, handlelength=1.4)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.35)
    fig.tight_layout()
    fig.savefig(out / "ch09-ou.pdf")
    plt.close(fig)


def cev(out):
    """CEV paths that reach zero, and the fraction absorbed by time t for several gamma."""
    S0, mu, sigma, T, n = 1.0, 0.0, 0.6, 5.0, 5_000
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.2))
    t, S = cev_paths(S0, mu, sigma, 0.5, T, n, n_paths=14, seed=1)
    dead = S[:, -1] <= 0
    for row, d in zip(S, dead):
        ax.plot(t, row, color=GOLD if d else COBALT, lw=0.9, alpha=0.9, zorder=3 if d else 2)
    ax.plot([], [], color=COBALT, lw=1.2, label="positive at $T$")
    ax.plot([], [], color=GOLD, lw=1.2, label="absorbed at 0")
    ax.set_xlim(0, T)
    ax.set_ylim(bottom=-0.05)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$S_t$")
    ax.set_title(r"$\gamma=1/2$", fontsize=9)
    ax.legend(loc="upper left", fontsize=7.5)

    styles = ((0.25, GREEN, (0, (1, 1.5))), (0.5, GOLD, "-"), (0.75, COBALT, (0, (5, 2))))
    for gamma, c, ls in styles:
        t, S = cev_paths(S0, mu, sigma, gamma, T, n, n_paths=4_000, seed=40)
        ax2.plot(t, absorbed_fraction(S), color=c, ls=ls, lw=1.6, label=f"$\\gamma={gamma:g}$")
    ax2.set_xlim(0, T)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("$t$")
    ax2.set_ylabel("fraction of paths at 0")
    ax2.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "ch09-cev.pdf")
    plt.close(fig)


def main(out):
    use_style()
    euler_convergence(out)
    ou_paths(out)
    cev(out)
