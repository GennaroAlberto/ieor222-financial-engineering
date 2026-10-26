"""Figures for Chapter 11 (change of measure and risk-neutral pricing)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieor222.brownian import brownian_paths
from ieor222.riskneutral import (bs_price, call_estimator_std, exercise_probabilities,
                                 gbm_from_brownian, girsanov_density)
from ieor222.style import COBALT, GOLD, GREY, INK, use_style

S0, K, T, R, SIGMA, MU = 100.0, 110.0, 1.0, 0.03, 0.2, 0.15


def reweighting(out):
    """Left: W_T + theta T under P, raw and reweighted by Z_T. Right: call prices from P-paths."""
    theta = (MU - R) / SIGMA
    t, W = brownian_paths(T=T, n_steps=1, n_paths=100_000, seed=11)
    Z = girsanov_density(t, W, theta)[:, -1]
    Wt = W[:, -1] + theta * T

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    edges = np.linspace(-4, 5, 61)
    ax.hist(Wt, bins=edges, density=True, color=GOLD, alpha=0.35,
            label=r"$\widetilde W_T$ on $\mathbb{P}$-paths")
    ax.hist(Wt, bins=edges, weights=Z / Z.sum() / np.diff(edges)[0], histtype="step",
            color=COBALT, lw=1.4, label=r"same paths, weights $Z_T$")
    y = np.linspace(-4, 5, 400)
    ax.plot(y, np.exp(-y**2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1.0, ls="--",
            label=r"$\mathcal{N}(0,1)$ density")
    ax.set(xlabel=r"$\widetilde W_T=W_T+\theta T$", ylabel="density", xlim=(-4, 5), ylim=(0, 0.62))
    ax.legend(fontsize=7, loc="upper left")

    S_T = gbm_from_brownian(T, W[:, -1], S0, MU, SIGMA)
    pay = np.exp(-R * T) * np.maximum(S_T - K, 0.0)
    M = np.unique(np.logspace(2, 5, 40).astype(int))
    for values, color, label in ((pay, GOLD, "plain average"),
                                 (pay * Z, COBALT, r"weighted by $Z_T$")):
        cum = np.cumsum(values)[M - 1] / M
        sq = np.cumsum(values**2)[M - 1] / M
        hw = 1.96 * np.sqrt(np.maximum(sq - cum**2, 0) / M)
        ax2.plot(M, cum, color=color, lw=1.3, label=label)
        ax2.fill_between(M, cum - hw, cum + hw, color=color, alpha=0.18, lw=0)
    ax2.axhline(bs_price(S0, K, T, R, SIGMA), color=INK, lw=0.9, ls="--", label="Black\u2013Scholes")
    ax2.set(xscale="log", xlabel=r"number of $\mathbb{P}$-paths", ylabel="call price estimate",
            ylim=(0, 16))
    ax2.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "ch11-reweighting.pdf")
    plt.close(fig)


def numeraire(out):
    """Left: log(S_T/S_0) under Q and under Q^S. Right: Monte Carlo error of the two estimators."""
    sigma, T_ = 0.3, 1.0
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    y = np.linspace(-1.2, 1.4, 500)
    k = np.log(K / S0)
    pQ, pS = exercise_probabilities(S0, K, T_, R, sigma)
    for m, color, name, p in (((R - sigma**2 / 2) * T_, COBALT, r"$\mathbb{Q}$", pQ),
                              ((R + sigma**2 / 2) * T_, GOLD, r"$\mathbb{Q}^S$", pS)):
        dens = np.exp(-(y - m) ** 2 / (2 * sigma**2 * T_)) / np.sqrt(2 * np.pi * sigma**2 * T_)
        ax.plot(y, dens, color=color, lw=1.4, label=f"{name}: mass {p:.3f} right of $K$")
        ax.fill_between(y, 0, dens, where=y > k, color=color, alpha=0.12, lw=0)
    ax.axvline(k, color=GREY, lw=0.8)
    ax.text(k + 0.03, 0.05, r"$\log(K/S_0)$", fontsize=7, color=GREY)
    ax.set(xlabel=r"$\log(S_T/S_0)$", ylabel="density", ylim=(0, 2.1))
    ax.legend(fontsize=7, loc="upper left")

    Ks = np.linspace(50, 200, 61)
    std = call_estimator_std(S0, Ks, T_, R, sigma)
    price = bs_price(S0, Ks, T_, R, sigma)
    ax2.plot(Ks / S0, std["Q"] / price, color=COBALT, lw=1.4,
             label=r"$e^{-rT}(S_T-K)^+$ under $\mathbb{Q}$")
    ax2.plot(Ks / S0, std["QS"] / price, color=GOLD, lw=1.4,
             label=r"$S_0(1-K/S_T)^+$ under $\mathbb{Q}^S$")
    ax2.set(yscale="log", xlabel=r"strike $K/S_0$", ylabel="std. deviation / price")
    ax2.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ch11-numeraire.pdf")
    plt.close(fig)


def main(out):
    use_style()
    reweighting(out)
    numeraire(out)
