"""Figures for Chapter 2 (Martingales and random walks)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieor222.martingales import (expected_exit_time, mean_one_products, prob_upper,
                                 simulate_exit)
from ieor222.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"


def products(out):
    """Paths of a product of mean-one factors 1 +- a: the mean stays at 1, the paths die."""
    a, n = 0.5, 100
    M = mean_one_products(n, n_paths=100_000, a=a, seed=11)
    k = np.arange(n + 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for path in M[:25]:
        ax.plot(k, path, color=GREY, lw=0.5, alpha=0.45)
    ax.plot(k, M.mean(axis=0), color=COBALT, lw=1.8, label="average of $10^5$ paths")
    ax.plot(k, (1 - a**2) ** (k / 2), color=GOLD, lw=1.8, label=r"typical path $(1-a^2)^{n/2}$")
    ax.axhline(1.0, color=INK, lw=0.9, ls="--", label=r"exact mean $\mathbb{E}[M_n]=1$")
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1e4)
    ax.set_xlim(0, n)
    ax.set_xlabel("$n$")
    ax.set_ylabel("$M_n$")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ch02-products.pdf")
    plt.close(fig)


def ruin(out):
    """Gambler's ruin on (0, 20): exit probabilities and expected durations, exact and simulated."""
    a, b = 0, 20
    xs = np.arange(a, b + 1)
    xs_mc = np.arange(2, b, 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for p, color in ((0.45, GOLD), (0.5, COBALT), (0.55, GREEN)):
        ax1.plot(xs, prob_upper(xs, a, b, p), color=color, lw=1.6, label=f"$p={p}$")
        ax2.plot(xs, expected_exit_time(xs, a, b, p), color=color, lw=1.6, label=f"$p={p}$")
        prob, dur = [], []
        for i, x in enumerate(xs_mc):
            tau, S_tau = simulate_exit(int(x), a, b, p=p, n_paths=4_000, seed=100 * i + int(100 * p))
            prob.append(np.mean(S_tau == b))
            dur.append(tau.mean())
        ax1.plot(xs_mc, prob, "o", color=color, ms=3.5, mfc="white", mew=1.1)
        ax2.plot(xs_mc, dur, "o", color=color, ms=3.5, mfc="white", mew=1.1)
    ax1.set_xlabel("starting point $x$")
    ax1.set_ylabel(r"$\mathbb{P}(S_\tau = b)$")
    ax2.set_xlabel("starting point $x$")
    ax2.set_ylabel(r"$\mathbb{E}[\tau]$")
    for ax in (ax1, ax2):
        ax.set_xlim(a, b)
        ax.set_xticks([0, 5, 10, 15, 20])
    ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ch02-ruin.pdf")
    plt.close(fig)


def main(out):
    use_style()
    products(out)
    ruin(out)
