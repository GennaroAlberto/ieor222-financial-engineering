"""Figures for Chapter 8 (Ito's formula)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieor222.brownian import brownian_paths
from ieor222.itoformula import ito_check, levy_brownian
from ieor222.style import COBALT, GOLD, GREY, INK, use_style


def pathwise(out):
    """Ito's formula on one Brownian path, for f(x) = cosh(2x)."""
    t, W = brownian_paths(T=1.0, n_steps=2**16, n_paths=1, seed=3)
    r = ito_check(t, W, lambda t, x: np.cosh(2 * x), lambda t, x: 2 * np.sinh(2 * x),
                  lambda t, x: 4 * np.cosh(2 * x))
    k = 8  # computed on 2^16 steps, plotted on every 8th point to keep the PDF small
    t = t[::k]
    r = {key: val[:, ::k] for key, val in r.items()}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax.plot(t, r["lhs"][0], color=INK, lw=1.4, label="$f(W_t)-f(0)$")
    ax.plot(t, r["rhs"][0], color=GOLD, lw=1.2, ls=(0, (4, 2)), label="Itô: first + second order")
    ax.plot(t, r["first_order"][0], color=COBALT, lw=1.1, label=r"first order only: $\int_0^t f'(W)\,\mathrm{d}W$")
    ax.set(xlabel="$t$", xlim=(0, 1), title=r"$f(x)=\cosh(2x)$")
    ax.legend(loc="upper left", fontsize=7)
    gap = (r["lhs"] - r["first_order"])[0]
    ax2.plot(t, gap, color=COBALT, lw=1.4, label=r"$f(W_t)-f(0)-\int_0^t f'(W)\,\mathrm{d}W$")
    ax2.plot(t, r["second_order"][0], color=GOLD, lw=1.2, ls=(0, (4, 2)),
             label=r"$\frac{1}{2}\int_0^t f''(W_s)\,\mathrm{d}s$")
    ax2.set(xlabel="$t$", xlim=(0, 1), title="the missing term")
    ax2.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch08-ito-pathwise.pdf")
    plt.close(fig)


def levy(out):
    """B = int sgn(W) dW: a path next to W, and the law of B_1."""
    t, W = brownian_paths(T=1.0, n_steps=2**12, n_paths=1, seed=277)  # a long negative excursion
    B = levy_brownian(W)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax.plot(t, W[0], color=GREY, lw=1.0, label="$W_t$")
    ax.plot(t, B[0], color=COBALT, lw=1.2, label=r"$B_t=\int_0^t \mathrm{sgn}(W_s)\,\mathrm{d}W_s$")
    ax.axhline(0.0, color=INK, lw=0.6, alpha=0.4)
    ax.set(xlabel="$t$", xlim=(0, 1), title="one path")
    ax.legend(loc="upper left", fontsize=7)
    ts, Ws = brownian_paths(T=1.0, n_steps=500, n_paths=20_000, seed=22)
    b = levy_brownian(Ws)[:, -1]
    ax2.hist(b, bins=60, density=True, color=COBALT, alpha=0.35, label="$B_1$, 20000 paths")
    x = np.linspace(-4, 4, 400)
    ax2.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1.3, label=r"$\mathcal{N}(0,1)$ density")
    ax2.set(xlim=(-4, 4), xlabel="value", title=f"sample variance {b.var():.3f}")
    ax2.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch08-levy.pdf")
    plt.close(fig)


def main(out):
    use_style()
    pathwise(out)
    levy(out)
