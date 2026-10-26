"""Figures for Chapter 6 (from the binomial model to Black--Scholes)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieor222.blackscholes import (atm_forward_call, binomial_call, brenner_subrahmanyam, call,
                                  greeks)
from ieor222.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"


def binomial_convergence(out):
    """Binomial call prices against the Black--Scholes price: odd/even oscillation and 1/n decay."""
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    exact = call(S0, K, T, r, sigma)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    n = np.arange(4, 81)
    prices = np.array([binomial_call(S0, K, T, r, sigma, k) for k in n])
    even = n % 2 == 0
    ax.axhline(exact, color=INK, lw=0.9, label="Black–Scholes")
    ax.plot(n[even], prices[even], "o", ms=3, color=COBALT, label="$n$ even")
    ax.plot(n[~even], prices[~even], "o", ms=3, color=GOLD, label="$n$ odd")
    ax.set(xlabel="number of periods $n$", ylabel="call price", title=r"$S_0=K=100$, $T=1$, $r=5\%$, $\sigma=20\%$")
    ax.legend(fontsize=7.5, loc="upper right")

    m = np.unique(np.round(np.logspace(1, 3.5, 60)).astype(int))
    for parity, color, label in ((0, COBALT, "$K=100$, $n$ even"), (1, GOLD, "$K=100$, $n$ odd")):
        k = np.unique(2 * (m // 2) + parity)
        err = np.array([binomial_call(S0, K, T, r, sigma, j) - exact for j in k])
        ax2.plot(k, k * err, "o-", ms=2.5, lw=1.0, color=color, label=label, zorder=2)
    k = np.arange(10, 3001)
    exact110 = call(S0, 110.0, T, r, sigma)
    err = np.array([binomial_call(S0, 110.0, T, r, sigma, j) - exact110 for j in k])
    ax2.plot(k, k * err, ".", ms=1.2, color=GREEN, label="$K=110$, all $n$", zorder=1)
    ax2.axhline(0.0, color=GREY, lw=0.6)
    ax2.set(xscale="log", xlabel="number of periods $n$", ylabel=r"$n\times$(binomial $-$ BS)",
            title="the error times $n$")
    ax2.set_ylim(-3.3, 2.4)
    ax2.legend(fontsize=6.5, loc="lower center", ncol=3, markerscale=1.5, handlelength=1.2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(out / "ch06-binomial-convergence.pdf")
    plt.close(fig)


def greek_curves(out):
    """Delta, gamma, vega and theta of a call against the spot, for three maturities."""
    K, r, sigma = 100.0, 0.05, 0.2
    x = np.linspace(50, 150, 401)
    taus = ((1 / 12, COBALT, r"$\tau=1$ month"), (0.5, GOLD, r"$\tau=6$ months"), (2.0, GREEN, r"$\tau=2$ years"))
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.2), sharex=True)
    names = (("delta", r"delta $\partial_x c$"), ("gamma", r"gamma $\partial_{xx} c$"),
             ("vega", r"vega $\partial_\sigma c$"), ("theta", r"theta $\partial_t c$ (per year)"))
    for ax, (key, title) in zip(axes.flat, names):
        for tau, color, label in taus:
            ax.plot(x, greeks(x, K, tau, r, sigma)[key], color=color, lw=1.4, label=label)
        ax.axvline(K, color=GREY, lw=0.6, alpha=0.6)
        ax.set_title(title)
    axes[0, 0].legend(fontsize=7.5, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("spot $x$ (strike $K=100$)")
    fig.tight_layout()
    fig.savefig(out / "ch06-greeks.pdf")
    plt.close(fig)


def atm_approximation(out):
    """At-the-money-forward call: exact price against the Brenner--Subrahmanyam approximation."""
    S0, T = 1.0, 1.0
    v = np.linspace(1e-3, 2.0, 400)
    exact = atm_forward_call(S0, T, v)
    approx = brenner_subrahmanyam(S0, T, v)["price"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax.plot(v, exact, color=COBALT, label="exact $2N(v/2)-1$")
    ax.plot(v, approx, color=GOLD, label=r"$v/\sqrt{2\pi}$")
    ax.set(xlabel=r"total volatility $v=\sigma\sqrt{T}$", ylabel="price / $S_0$", title=r"call with $K=S_0e^{rT}$")
    ax.legend(fontsize=7.5, loc="upper left")
    vv = np.logspace(-2, np.log10(2.0), 200)
    ex = atm_forward_call(S0, T, vv)
    ax2.plot(vv, brenner_subrahmanyam(S0, T, vv)["price"] - ex, color=GOLD, label=r"$v/\sqrt{2\pi}-$ exact")
    ax2.plot(vv, vv**3 / (24 * np.sqrt(2 * np.pi)), color=INK, lw=1.0, ls=(0, (2, 2)), zorder=3,
             label=r"bound $v^3/(24\sqrt{2\pi})$")
    ax2.plot(vv, 0.4 * vv - ex, color=GREY, lw=0.9, ls="--", label="$0.4\\,v-$ exact")
    ax2.set(xscale="log", yscale="log", xlabel=r"total volatility $v=\sigma\sqrt{T}$", title="error / $S_0$")
    ax2.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ch06-atm.pdf")
    plt.close(fig)


def main(out):
    use_style()
    binomial_convergence(out)
    greek_curves(out)
    atm_approximation(out)
