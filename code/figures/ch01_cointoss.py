"""Figures for Chapter 1 (Probability as information)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieor222.cointoss import (
    conditional_expectation,
    expectation,
    least_squares,
    partition_of,
    stock_prices,
    toss_partition,
    toss_probabilities,
    toss_space,
)
from ieor222.style import COBALT, GOLD, GREY, INK, use_style


def condexp_atoms(out):
    """E[X | F_k] for the running maximum of the stock, k = 0, 1, 2, 3: averages over atoms."""
    n, p = 4, 0.6
    heads = toss_space(n)
    prob = toss_probabilities(heads, p)
    X = stock_prices(heads, 4.0, 1.5, 0.5).max(axis=1)
    edges = np.concatenate([[0.0], np.cumsum(prob)])  # outcome i occupies [edges[i], edges[i+1])
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.05), sharey=True)
    for k, ax in enumerate(axes):
        lab = toss_partition(n, k)
        Y = conditional_expectation(X, prob, lab)
        ax.bar(edges[:-1], X, width=prob, align="edge", color=GREY, alpha=0.22,
               edgecolor="white", linewidth=0.5)
        for a in np.unique(lab):
            idx = np.flatnonzero(lab == a)
            x0, x1 = edges[idx[0]], edges[idx[-1] + 1]
            ax.hlines(Y[idx[0]], x0, x1, color=COBALT, lw=1.8)
            if a > 0:
                ax.axvline(x0, color=INK, lw=0.5, alpha=0.35)
        ax.set_title(f"$k={k}$", pad=3)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.grid(False)
    axes[0].set_ylabel("value")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.22),
               plt.Line2D([], [], color=COBALT, lw=1.8)]
    fig.legend(handles, ["$X(\\omega)$, the running maximum $\\max_{j\\leq 4}S_j$",
                         "$E[X\\mid\\mathcal{F}_k]$"],
               loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    left, right = axes[0].get_position().x0, axes[-1].get_position().x1
    fig.text((left + right) / 2, axes[0].get_position().y0 - 0.17,
             "outcomes from HHHH to TTTT; width = probability", ha="center", va="top",
             fontsize=9, color=INK)
    fig.savefig(out / "ch01-condexp.pdf")
    plt.close(fig)


def projection(out):
    """A call payoff at date 10 against S_6: conditional law, E[X | S_6] and the best affine predictor."""
    n, k, p, S0, K = 10, 6, 0.5, 100.0, 100.0
    u = 1.15
    heads = toss_space(n)
    prob = toss_probabilities(heads, p)
    S = stock_prices(heads, S0, u, 1 / u)
    X = np.maximum(S[:, n] - K, 0.0)
    lab = partition_of(S[:, k])
    ce = conditional_expectation(X, prob, lab)
    lin = least_squares(X, np.vander(S[:, k], 2), prob)
    mse = lambda Y: expectation((X - Y) ** 2, prob)

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    # conditional law of X given S_k = s: dot area proportional to the conditional probability
    pts = {}
    for i in range(len(X)):
        key = (round(S[i, k], 8), round(X[i], 8))
        pts[key] = pts.get(key, 0.0) + prob[i]
    s_vals = np.array(sorted({key[0] for key in pts}))
    mass_s = {s: sum(m for (t, _), m in pts.items() if t == s) for s in s_vals}
    xs, ys, sz = zip(*[(s, x, m / mass_s[s]) for (s, x), m in pts.items()])
    ax.scatter(xs, ys, s=260 * np.array(sz), color=GREY, alpha=0.35, lw=0,
               label="law of $X$ given $S_6$")
    order = np.argsort(S[:, k])
    ax.plot(S[order, k], lin[order], color=GOLD, lw=1.6,
            label=f"best affine predictor (MSE {mse(lin):.0f})")
    g = [ce[np.argmin(np.abs(S[:, k] - s))] for s in s_vals]
    ax.plot(s_vals, g, color=COBALT, lw=1.8, marker="o", ms=4,
            label=f"$E[X\\mid S_6]$ (MSE {mse(ce):.0f})")
    ax.set_xlabel("$S_6$")
    ax.set_ylabel("$X=(S_{10}-100)^+$")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ch01-projection.pdf")
    plt.close(fig)


def main(out):
    use_style()
    condexp_atoms(out)
    projection(out)
