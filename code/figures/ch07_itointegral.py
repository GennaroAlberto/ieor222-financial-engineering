"""Figures for Chapter 7 (the Ito integral)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ieor222.brownian import brownian_paths
from ieor222.itointegral import (
    lambda_sum,
    midpoint_sum,
    quadratic_variation,
    simulate,
    time_integral,
    wdw_limit,
)
from ieor222.style import COBALT, GOLD, GREY, INK, PALETTE, use_style

GREEN = PALETTE[2]


def riemann_sums(out):
    """Left-point, midpoint and right-point sums for int_0^1 W dW, on one path and in L^2."""
    T, K = 1.0, 16
    _, W = brownian_paths(T=T, n_steps=2**K, n_paths=1, seed=12)
    w = W[0]
    ks = np.arange(1, 15)
    n = 2**ks
    left = [lambda_sum(w, 0.0, every=2 ** (K - k)) for k in ks]
    right = [lambda_sum(w, 1.0, every=2 ** (K - k)) for k in ks]
    mid = [midpoint_sum(w, every=2 ** (K - k)) for k in ks]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.2))
    for vals, lam, color, label in ((right, 1.0, GREEN, "right point"),
                                    (mid, 0.5, GOLD, "midpoint"),
                                    (left, 0.0, COBALT, "left point (Itô)")):
        ax.axhline(wdw_limit(w[-1], T, lam), color=color, lw=0.8, ls="--", alpha=0.8)
        ax.plot(n, vals, color=color, marker="o", ms=2.5, lw=1.2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of intervals $n$")
    ax.set_ylabel("Riemann sum")
    ax.set_title("one path; dashed lines: the limits", fontsize=9)
    ax.set_ylim(top=max(max(right), wdw_limit(w[-1], T, 1.0)) + 0.9)
    ax.legend(loc="upper right", fontsize=7.5)

    Kp = 12
    _, Wp = brownian_paths(T=T, n_steps=2**Kp, n_paths=2000, seed=13)
    kk = np.arange(1, Kp)
    nn = 2**kk
    err_left = [np.sqrt(np.mean((lambda_sum(Wp, 0.0, every=2 ** (Kp - k)) - wdw_limit(Wp[:, -1], T)) ** 2))
                for k in kk]
    err_mid = [np.sqrt(np.mean((midpoint_sum(Wp, every=2 ** (Kp - k)) - 0.5 * Wp[:, -1] ** 2) ** 2))
               for k in kk]
    ax2.plot(nn, err_left, color=COBALT, marker="o", ms=2.5, lw=0, label="left point vs $\\frac{1}{2}(W_1^2-1)$")
    ax2.plot(nn, T / np.sqrt(2 * nn), color=COBALT, lw=0.9, ls="--")
    ax2.plot(nn, err_mid, color=GOLD, marker="s", ms=2.5, lw=0, label="midpoint vs $\\frac{1}{2}W_1^2$")
    ax2.plot(nn, T / (2 * np.sqrt(nn)), color=GOLD, lw=0.9, ls="--")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("number of intervals $n$")
    ax2.set_ylabel("root-mean-square error")
    ax2.set_title("2000 paths; dashed: $1/\\sqrt{2n}$ and $1/(2\\sqrt{n})$", fontsize=9)
    ax2.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "ch07-riemann-sums.pdf")
    plt.close(fig)


def quadratic_variation_vs_variance(out):
    """[I, I]_t = int_0^t W^2 ds is random; the variance t^2/2 of I_t is its mean."""
    T, n_coarse, every = 1.0, 1000, 16  # fine grid of 1000 * 16 steps; the coarse grid uses every 16th point
    t, W, H, I = simulate(lambda t, W: W, T=T, n_steps=n_coarse * every, n_paths=3, seed=31)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.2))
    ax.plot(t, W[0], color=GOLD, lw=1.0, label="$W_t$")
    ax.plot(t, I[0], color=COBALT, lw=1.4, label="$I_t$, the It\u00f4 integral of $W$")
    ax.axhline(0, color=INK, lw=0.6, alpha=0.4)
    ax.set_xlabel("$t$")
    ax.set_xlim(0, T)
    lo, hi = min(W[0].min(), I[0].min()), max(W[0].max(), I[0].max())
    ax.set_ylim(lo - 0.05, hi + 0.45 * (hi - lo))
    ax.legend(loc="upper center", ncol=2, fontsize=7.5)
    ax.set_title("one path", fontsize=9)

    for k, color in enumerate((COBALT, GOLD, GREEN)):
        ax2.plot(t, time_integral(W[k] ** 2, t), color=color, lw=1.4)
        qv = quadratic_variation(I[k], every=every)
        ax2.plot(t[::every][::25], qv[::25], color=color, lw=0, marker="o", ms=2.2)
    ax2.plot(t, t**2 / 2, color=INK, lw=1.2, ls="--")
    handles = [Line2D([], [], color=GREY, lw=1.4, label="$\\int_0^t W_s^2\\,\\mathrm{d}s$"),
               Line2D([], [], color=GREY, lw=0, marker="o", ms=3,
                      label=f"$\\sum (I_{{t_{{j+1}}}}-I_{{t_j}})^2$, $n={n_coarse}$"),
               Line2D([], [], color=INK, lw=1.2, ls="--", label="$E[I_t^2]=t^2/2$")]
    ax2.legend(handles=handles, loc="upper left", fontsize=7.5)
    ax2.set_xlabel("$t$")
    ax2.set_xlim(0, T)
    ax2.set_title("three paths: $[I,I]_t$ and the variance", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "ch07-qv.pdf")
    plt.close(fig)


def main(out):
    use_style()
    riemann_sums(out)
    quadratic_variation_vs_variance(out)
