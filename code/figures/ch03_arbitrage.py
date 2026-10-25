"""Figures for Chapter 3 (Arbitrage and static replication)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieor222.arbitrage import lognormal_call, replicate_payoff
from ieor222.style import COBALT, GOLD, GREY, INK, PALETTE, use_style

GREEN = PALETTE[2]


def payoffs(out):
    """Payoffs at T of a long call, a long put, a long forward and a butterfly."""
    K = 100.0
    x = np.linspace(0, 200, 801)
    panels = (
        ("call $(S_T-K)^+$", np.maximum(x - K, 0)),
        ("put $(K-S_T)^+$", np.maximum(K - x, 0)),
        ("forward $S_T-K$", x - K),
        ("butterfly", np.maximum(x - 70, 0) - 2 * np.maximum(x - K, 0) + np.maximum(x - 130, 0)),
    )
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.95))
    for ax, (title, y) in zip(axes, panels):
        ax.axhline(0, color=GREY, lw=0.6)
        ax.plot(x, y, color=COBALT, lw=1.6)
        ax.set_title(title)
        ax.set_xticks([0, K, 200])
        ax.set_xticklabels(["0", "$K$", ""])
        ax.set_yticks([])
        ax.set_xlabel("$S_T$")
        ax.set_xlim(0, 200)
    axes[3].set_xticks([0, 70, K, 130])
    axes[3].set_xticklabels(["0", "$K_1$", "$K_2$", "$K_3$"])
    axes[3].set_ylim(-8, 40)
    axes[2].set_ylim(-110, 110)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "ch03-payoffs.pdf")
    plt.close(fig)


def call_bounds(out):
    """The region allowed by no-arbitrage for K -> C(K), with two admissible curves."""
    S0, B = 100.0, 0.9
    K = np.linspace(0, 260, 1041)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    lower = np.maximum(S0 - B * K, 0)
    ax.fill_between(K, lower, S0, color=COBALT, alpha=0.06, lw=0)
    ax.plot(K, lower, color=INK, lw=1.0, label=r"bounds $(S_0-BK)^+$ and $S_0$")
    ax.axhline(S0, color=INK, lw=1.0)
    for v, color in ((0.15, COBALT), (0.5, GOLD)):
        ax.plot(K, lognormal_call(S0, B, K, v), color=color, lw=1.7,
                label=f"an admissible curve (lognormal, $v={v:g}$)")
    ax.annotate("slope $-B$", xy=(60, S0 - B * 60), xytext=(20, 18), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    ax.set_xticks([0, S0 / B, 200])
    ax.set_xticklabels(["0", "$F=S_0/B$", "200"])
    ax.set_yticks([0, S0])
    ax.set_yticklabels(["0", "$S_0$"])
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 108)
    ax.set_xlabel("strike $K$")
    ax.set_ylabel("call price $C(K)$")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ch03-call-bounds.pdf")
    plt.close(fig)


def carr_madan(out):
    """Discrete Carr--Madan replication of the log payoff, and its error as the spacing shrinks."""
    kappa, lo, hi = 100.0, 50.0, 200.0
    f = lambda s: -np.log(s / kappa)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7))

    h = 25.0
    K = np.arange(lo, hi + h / 2, h)
    p = replicate_payoff(f, K, kappa)
    x = np.linspace(20, 240, 1101)
    ax.axvspan(lo, hi, color=COBALT, alpha=0.05, lw=0)
    ax.plot(x, f(x), color=COBALT, lw=1.7, label=r"$f(x)=-\log(x/\kappa)$")
    ax.plot(x, p.payoff(x), color=GOLD, lw=1.4, ls="--", label=f"strip with spacing $h={h:g}$")
    ax.plot(K, f(K), "o", color=GOLD, ms=3)
    ax.set_xlabel("$S_T$")
    ax.set_xlim(20, 240)
    ax.set_ylim(-1.0, 1.7)
    ax.legend(loc="upper right", fontsize=7.5)

    hs = 50.0 / 2.0 ** np.arange(0, 8)
    err = []
    xx = np.linspace(lo, hi, 60001)
    for h in hs:
        q = replicate_payoff(f, np.arange(lo, hi + h / 2, h), kappa)
        err.append(np.max(np.abs(q.payoff(xx) - f(xx))))
    ax2.plot(hs, err, "o-", color=COBALT, ms=3.5, label="largest error on $[50,200]$")
    ax2.plot(hs, hs**2 / (8 * lo**2), color=GOLD, lw=1.2, ls="--",
             label=r"bound $h^2\max|f''|/8$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("strike spacing $h$")
    ax2.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out / "ch03-carr-madan.pdf")
    plt.close(fig)


def main(out):
    use_style()
    payoffs(out)
    call_bounds(out)
    carr_madan(out)
