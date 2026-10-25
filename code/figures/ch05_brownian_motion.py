"""Figures for Chapter 5 (Brownian motion)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieor222.brownian import brownian_paths, p_variation, scaled_random_walk
from ieor222.style import COBALT, GOLD, GREY, INK, use_style


def scaled_walks(out):
    """Scaled random walks for growing n: the paths look more and more like Brownian motion."""
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.2), sharey=True)
    for ax, n in zip(axes, (8, 64, 2048)):
        seed = n
        t, W = scaled_random_walk(n, T=1.0, n_paths=3, seed=seed)
        while len({tuple(row) for row in W}) < 3:  # three distinct paths, even for small n
            seed += 1000
            t, W = scaled_random_walk(n, T=1.0, n_paths=3, seed=seed)
        for k, color in enumerate((COBALT, GOLD, GREY)):
            ax.plot(t, W[k], color=color, lw=1.1, drawstyle="default")
        ax.set_title(f"$n = {n}$")
        ax.set_xlabel("$t$")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("$W^{(n)}(t)$")
    fig.tight_layout()
    fig.savefig(out / "ch05-scaled-walks.pdf")
    plt.close(fig)


def variations(out):
    """p-variations of one Brownian path along dyadic partitions of [0, 1]."""
    K = 18
    _, W = brownian_paths(T=1.0, n_steps=2**K, n_paths=1, seed=7)
    w = W[0]
    n_intervals = np.array([2**k for k in range(2, K + 1)])
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for p, color, label in ((1, GOLD, "$p=1$: first variation"),
                            (2, COBALT, "$p=2$: quadratic variation"),
                            (3, "#1BAF7A", "$p=3$")):
        v = np.array([p_variation(w, p, every=2 ** (K - k)) for k in range(2, K + 1)])
        ax.plot(n_intervals, v, color=color, lw=1.8, marker="o", ms=3, label=label)
    ax.axhline(1.0, color=INK, lw=0.8, ls=(0, (1, 0)), alpha=0.35)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of intervals $n$ in the partition of $[0,1]$")
    ax.set_ylabel(r"$\sum_j |W_{t_{j+1}}-W_{t_j}|^p$")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ch05-variations.pdf")
    plt.close(fig)


def main(out):
    use_style()
    scaled_walks(out)
    variations(out)
