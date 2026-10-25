"""Figures for Chapter 4 (the binomial model)."""
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieor222.binomial import (american, european, stock_tree, trinomial_price_interval,
                              trinomial_risk_neutral, trinomial_theta_max)
from ieor222.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"


def tree(out):
    """Three-period tree of a call: stock price, option value and hedge ratio at every node."""
    S0, u, d, r, K, N = 100, 1.2, 0.9, 0.05, 100, 3
    S = stock_tree(S0, u, d, N)
    V, D = european(lambda s: np.maximum(s - K, 0.0), S0, u, d, r, N)
    fig, ax = plt.subplots(figsize=(6.4, 2.75))
    pos = lambda n, j: (n, 2 * j - n)
    for n in range(N):
        for j in range(n + 1):
            for jj in (j, j + 1):
                (x0, y0), (x1, y1) = pos(n, j), pos(n + 1, jj)
                ax.plot([x0 + 0.17, x1 - 0.17], [y0, y1], color=GREY, lw=0.8, alpha=0.6, zorder=1)
    for n in range(N + 1):
        for j in range(n + 1):
            x, y = pos(n, j)
            lines = [rf"$S={S[n][j]:.1f}$", rf"$V={V[n][j]:.2f}$"]
            if n < N:
                lines.append(rf"$\Delta={D[n][j]:.3f}$")
            ax.text(x, y, "\n".join(lines), ha="center", va="center", fontsize=7.5, color=INK, zorder=2,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COBALT if n < N else GOLD, lw=0.8))
    ax.set_xticks(range(N + 1))
    ax.set_xticklabels([f"$n={n}$" for n in range(N + 1)])
    ax.set_yticks([])
    ax.set_xlim(-0.45, N + 0.45)
    ax.set_ylim(-N - 0.9, N + 0.9)
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out / "ch04-tree.pdf")
    plt.close(fig)


def american_put(out):
    """American versus European put: prices at time 0, and the early-exercise region."""
    u, r, K, N = 1.04, 0.003, 100.0, 40
    d = 1 / u
    put = lambda s: np.maximum(K - s, 0.0)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.55))
    S0s = np.linspace(60, 140, 161)
    am = [american(put, s, u, d, r, N)[0][0][0] for s in S0s]
    eu = [european(put, s, u, d, r, N)[0][0][0] for s in S0s]
    ax.plot(S0s, put(S0s), color=INK, lw=0.9, ls="--", label="payoff $(K-S_0)^+$")
    ax.plot(S0s, am, color=COBALT, label="American")
    ax.plot(S0s, eu, color=GOLD, label="European")
    ax.set(xlabel="$S_0$", ylabel="price at time 0", title=f"$N={N}$ periods, $K={K:.0f}$")
    ax.legend(fontsize=7.5)

    S = stock_tree(K, u, d, N)
    _, _, ex = american(put, K, u, d, r, N)
    n_all = np.concatenate([np.full(n + 1, n) for n in range(N + 1)])
    s_all = np.concatenate(S)
    e_all = np.concatenate(ex)
    ax2.scatter(n_all[~e_all], s_all[~e_all], s=3, color=GREY, alpha=0.45, label="continue")
    ax2.scatter(n_all[e_all], s_all[e_all], s=3, color=COBALT, label="exercise")
    ax2.axhline(K, color=INK, lw=0.7, alpha=0.4)
    ax2.set(yscale="log", xlabel="period $n$", ylabel="$S_n$", title=r"nodes of the tree, $S_0=K$")
    ax2.set_ylim(K * d**(N * 0.75), K * u**(N * 0.75))
    ax2.set_yticks([40, 60, 100, 150, 250])
    ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.legend(fontsize=7.5, loc="upper left", markerscale=3)
    fig.tight_layout()
    fig.savefig(out / "ch04-american-put.pdf")
    plt.close(fig)


def trinomial(out):
    """One-period trinomial model: the interval of arbitrage-free call prices."""
    S0, u, d, r, K = 100.0, 1.2, 0.8, 0.05, 100.0
    call = lambda s: np.maximum(s - K, 0.0)
    ms = np.linspace(d + 1e-3, u - 1e-3, 400)
    ends = np.array([trinomial_price_interval(call, S0, u, m, d, r) for m in ms])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.55))
    ax.fill_between(ms, ends[:, 0], ends[:, 1], color=COBALT, alpha=0.18, lw=0)
    ax.plot(ms, ends[:, 1], color=COBALT, label="upper end (binomial on $\\{d,u\\}$)")
    ax.plot(ms, ends[:, 0], color=GOLD, label="lower end")
    ax.axvline(1 + r, color=INK, lw=0.7, ls=":", alpha=0.7)
    ax.text(1 + r + 0.005, 0.6, "$m=1+r$", fontsize=7.5, color=INK)
    ax.set(xlabel="middle factor $m$", ylabel="call price", title=f"$u={u}$, $d={d}$, $r={r}$, $K=S_0$")
    ax.set_ylim(0, ends[:, 1].max() * 1.35)
    ax.legend(fontsize=7.5, loc="upper left")

    m = 1.0
    tmax = trinomial_theta_max(u, m, d, r)
    th = np.linspace(0, tmax, 50)
    price = [np.array(trinomial_risk_neutral(u, m, d, r, t)) @ call(np.array([u, m, d]) * S0) / (1 + r) for t in th]
    ax2.plot(th, price, color=COBALT, label=r"$\mathbb{E}^{\mathbb{Q}_\theta}[(S_1-K)^+]/(1+r)$")
    ax2.plot([0, tmax], [price[0], price[-1]], "o", ms=5, mfc="white", mec=COBALT, label="end points (excluded)")
    ax2.set(xlabel=r"$\theta=\mathbb{Q}_\theta(S_1=mS_0)$", ylabel="call price",
            title=f"$m={m}$: one price per risk-neutral measure")
    ax2.set_xlim(-0.03, tmax + 0.03)
    ax2.set_ylim(0, price[0] * 1.35)
    ax2.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch04-trinomial.pdf")
    plt.close(fig)


def main(out):
    use_style()
    tree(out)
    american_put(out)
    trinomial(out)
