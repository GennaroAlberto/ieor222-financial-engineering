"""Figures for Chapter 13 (Exotic options)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieor222 import exotics as ex
from ieor222.blackscholes import call
from ieor222.style import COBALT, GOLD, GREY, INK, PALETTE, use_style

GREEN = PALETTE[2]


def reflection(out):
    """A Brownian path reflected at its first passage at m, and the law of the maximum with drift."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.3))
    n, m = 4000, 0.6
    t = np.linspace(0, 1, n + 1)
    rng = np.random.default_rng(11)
    for seed in range(11, 200):                    # a path that crosses m around mid-way and ends below it
        rng = np.random.default_rng(seed)
        W = np.concatenate([[0.0], np.cumsum(rng.normal(0, np.sqrt(1 / n), n))])
        k = np.argmax(W >= m) if (W >= m).any() else n
        if 0.3 * n < k < 0.6 * n and W[-1] < m - 0.2:
            break
    R = ex.reflect(W, m)
    ax.plot(t, W, color=COBALT, lw=1.0, label="$W$")
    ax.plot(t[k:], R[k:], color=GOLD, lw=1.0, label=r"reflected $W^*$")
    ax.axhline(m, color=INK, lw=0.8, ls="--", alpha=0.6)
    ax.axvline(t[k], color=GREY, lw=0.6, ls=":")
    ax.text(0.01, m + 0.04, "$m$", color=INK, fontsize=9)
    ax.text(t[k] + 0.01, ax.get_ylim()[0] + 0.05 if False else -0.9, r"$\tau_m$", color=GREY, fontsize=9)
    ax.set(xlabel="$t$", xlim=(0, 1), ylim=(-1.0, 2.0), title="reflection after $\\tau_m$")
    ax.legend(loc="upper left", fontsize=8)

    T, n_steps, n_paths = 1.0, 2000, 10_000
    grid = np.linspace(0, 2.5, 300)
    for nu, color, ls in ((-1.0, GOLD, "-"), (0.0, COBALT, "-"), (1.0, GREEN, "--")):
        X = ex.log_paths(T, n_steps, n_paths, nu, seed=int(10 * nu) + 20)
        ax2.hist(X.max(axis=1), bins=np.linspace(0, 2.5, 41), density=True, color=color, alpha=0.22)
        ax2.plot(grid, ex.max_density(grid, T, nu), color=color, ls=ls, lw=1.5, label=f"$\\nu={nu:g}$")
    ax2.set(xlabel="$m$", ylabel="density", xlim=(0, 2.5), title=r"law of $\max_{t\leq1}(W_t+\nu t)$")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ch13-reflection.pdf")
    plt.close(fig)


def barrier(out):
    """Up-and-out call: closed form, discrete monitoring by Monte Carlo, and the shifted barrier."""
    K, B, T, r, sigma = 100.0, 120.0, 1.0, 0.05, 0.2
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.3))
    x = np.linspace(70, 119.9, 300)
    ax.plot(x, ex.up_and_out_call(x, K, B, T, r, sigma), color=COBALT, label="continuous")
    m = 12
    ax.plot(x, ex.up_and_out_call(x, K, ex.bgk_barrier(B, sigma, T, m), T, r, sigma), color=GOLD, ls="--",
            lw=1.2, label="shifted barrier")
    xs = np.arange(75, 120, 5.0)
    est = [ex.barrier_call_mc(s, K, B, T, r, sigma, n_monitor=m, n_paths=20_000, seed=int(s)) for s in xs]
    ax.errorbar(xs, [e[0] for e in est], yerr=[2 * e[1] for e in est], fmt="o", ms=3.5, color=GOLD,
                label="Monte Carlo")
    ax.set(xlabel="$S_0$", ylabel="price", xlim=(70, 121), ylim=(0, 2.1), title=f"$K={K:g}$, $B={B:g}$")
    ax.legend(fontsize=7, loc="lower center", handlelength=1.6)

    S0 = 100.0
    ms = np.array([4, 12, 26, 52, 126, 252, 504])
    est = [ex.barrier_call_mc(S0, K, B, T, r, sigma, n_monitor=int(k), n_paths=20_000, seed=int(k) + 7) for k in ms]
    fine = np.unique(np.geomspace(3, 600, 80).astype(int))
    cont = ex.up_and_out_call(S0, K, B, T, r, sigma)
    ax2.axhline(cont, color=COBALT, label="continuous")
    ax2.plot(fine, [ex.up_and_out_call(S0, K, ex.bgk_barrier(B, sigma, T, k), T, r, sigma) for k in fine],
             color=GOLD, ls="--", lw=1.2, label="shifted barrier")
    ax2.errorbar(ms, [e[0] for e in est], yerr=[2 * e[1] for e in est], fmt="o", ms=3.5, color=GOLD,
                 label="Monte Carlo, discrete")
    ax2.set(xscale="log", xlabel="monitoring dates $m$", ylabel="price at $S_0=100$", title="discrete monitoring")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch13-barrier.pdf")
    plt.close(fig)


def asian(out):
    """Arithmetic and geometric Asian calls; the geometric average as a control variate."""
    S0, T, r, sigma = 100.0, 1.0, 0.05, 0.3
    Ks = np.linspace(80, 120, 9)
    grid = np.linspace(75, 125, 201)
    z, g0 = ex.vecer_solve(T, r, sigma)
    vecer = S0 * np.interp(ex.vecer_gamma(0, T, r) - np.exp(-r * T) * grid / S0, z, g0)
    mc = [ex.asian_call_mc(S0, k, T, r, sigma, n_steps=100, n_paths=10_000, seed=3) for k in Ks]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.3))
    ax.plot(grid, call(S0, grid, T, r, sigma), color=GREY, lw=1.1, label="European call")
    ax.plot(grid, vecer, color=COBALT, label="arithmetic Asian (PDE)")
    ax.plot(Ks, [o["price"] for o in mc], "o", ms=3.5, color=COBALT, label="arithmetic, Monte Carlo")
    ax.plot(grid, ex.geometric_asian_call(S0, grid, T, r, sigma), color=GOLD, ls="--", label="geometric Asian")
    ax.set(xlabel="strike $K$", ylabel="price", title=rf"$S_0={S0:g}$, $\sigma={sigma:g}$, $T={T:g}$")
    ax.legend(fontsize=7)
    ax2.plot(Ks, [o["plain_se"] * np.sqrt(1e4) for o in mc], "o-", ms=3.5, color=GOLD, label="plain")
    ax2.plot(Ks, [o["se"] * np.sqrt(1e4) for o in mc], "s--", ms=3.5, color=COBALT, label="with geometric control")
    ax2.set(yscale="log", xlabel="strike $K$", ylabel="standard deviation per path",
            title="Monte Carlo, arithmetic Asian")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ch13-asian.pdf")
    plt.close(fig)


def main(out):
    use_style()
    reflection(out)
    barrier(out)
    asian(out)
