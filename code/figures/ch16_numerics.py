"""Figures for Chapter 16 (numerical methods)."""
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieor222.binomial import early_exercise_boundary
from ieor222.blackscholes import call, crr_parameters, greeks, put
from ieor222.numerics import (american_put_fd, bs_log_coefficients, call_payoff_sd, convergence_table,
                              european_fd, grid_greeks, log_grid, mc_european, power_claim_rmse,
                              theta_scheme)
from ieor222.style import COBALT, GOLD, GREY, INK, PALETTE, use_style

GREEN, PURPLE = PALETTE[2], PALETTE[3]
S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


def monte_carlo(out):
    """Left: call estimates with 95% intervals against M. Right: RMSE against cost with a discretised model."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.3))
    exact = call(S0, K, T, R, SIGMA)
    sd = call_payoff_sd(S0, K, T, R, SIGMA)
    Ms = np.unique(np.round(np.logspace(2, 6, 17)).astype(int))
    payoff = lambda s: np.maximum(s - K, 0.0)
    est = [mc_european(payoff, S0, T, R, SIGMA, M, seed=1600 + i) for i, M in enumerate(Ms)]
    grid = np.logspace(2, 6, 200)
    ax.fill_between(grid, exact - 1.96 * sd / np.sqrt(grid), exact + 1.96 * sd / np.sqrt(grid),
                    color=COBALT, alpha=0.12, lw=0, label=r"$p\pm1.96\,s/\sqrt{M}$")
    ax.axhline(exact, color=GREY, lw=0.8)
    ax.errorbar(Ms, [e["mean"] for e in est], yerr=[e["half_width"] for e in est], fmt="o", ms=2.8,
                color=INK, elinewidth=0.9, capsize=0, label="estimate, 95% CI")
    ax.set_xscale("log")
    ax.set(xlabel="number of paths $M$", ylabel="call price", ylim=(exact - 4.5, exact + 4.5))
    ax.legend(fontsize=7, loc="upper right")

    S1, r1, s1 = 1.0, 0.05, 0.4
    C = np.logspace(2, 9, 120)
    ax2.loglog(C, power_claim_rmse(S1, T, r1, s1, 0, C), color=COBALT, label="exact sampling")
    for n, c, ls in ((4, GOLD, (0, (4, 2))), (32, GREEN, (0, (1, 1.5)))):
        ax2.loglog(C, power_claim_rmse(S1, T, r1, s1, n, C / n), color=c, ls=ls, label=f"Euler, $n={n}$")
    ns = np.arange(1, 4000)
    best = np.array([np.min(power_claim_rmse(S1, T, r1, s1, ns, c / ns)) for c in C])
    ax2.loglog(C, best, color=INK, lw=1.3, label="Euler, best $n$")
    ax2.set(xlabel="cost $nM$ (normal draws)", ylabel="root mean square error", ylim=(1e-5, 0.3))
    ax2.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch16-monte-carlo.pdf")
    plt.close(fig)


def stability(out):
    """Left: error of the explicit scheme for a call at three values of lambda. Right: amplification factors."""
    a, b, c = bs_log_coefficients(R, SIGMA)
    y = log_grid(K, T, SIGMA, 200)
    x = np.exp(y)
    h = y[1] - y[0]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.3))
    n_max = 120
    bc = lambda tau: (0.0, x[-1] - K * np.exp(-R * tau))
    for lam, col in ((0.45, COBALT), (0.5, GOLD), (0.55, GREEN)):
        k = lam * h * h / a
        res = theta_scheme(np.maximum(x - K, 0.0), y, n_max * k, n_max, a, b, c, 0.0, bc, keep=True)
        steps = np.arange(1, n_max + 1)
        err = [np.max(np.abs(res["history"][n] - call(x, K, n * k, R, SIGMA))) for n in steps]
        ax.semilogy(steps, err, color=col, label=rf"$\lambda={lam:g}$")
    ax.set(xlabel="time step $n$", ylabel=r"$\max_j|U^n_j-u(\tau_n,y_j)|$", ylim=(1e-5, 1e6))
    ax.legend(fontsize=7.5, loc="upper left")

    xi = np.linspace(0, np.pi, 300)
    s = np.sin(xi / 2) ** 2
    ax2.axhspan(-1, 1, color=GREY, alpha=0.08, lw=0)
    for lam, col, ls in ((0.4, COBALT, "-"), (0.6, COBALT, (0, (4, 2)))):
        ax2.plot(xi, 1 - 4 * lam * s, color=col, ls=ls, label=rf"explicit, $\lambda={lam:g}$")
    ax2.plot(xi, 1 / (1 + 4 * 5 * s), color=GOLD, label=r"implicit, $\lambda=5$")
    ax2.plot(xi, (1 - 2 * 5 * s) / (1 + 2 * 5 * s), color=GREEN, label=r"Crank–Nicolson, $\lambda=5$")
    ax2.set(xlabel=r"frequency $\xi h$", ylabel=r"amplification factor $g$", xlim=(0, np.pi), ylim=(-1.5, 1.1))
    ax2.set_xticks([0, np.pi / 2, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
    ax2.legend(fontsize=6.6, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch16-stability.pdf")
    plt.close(fig)


def convergence(out):
    """Left: errors of the four schemes at the money against h. Right: the gamma near the strike."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.4))
    styles = (("explicit", r"explicit, $\lambda=0.4$", COBALT, "o"), ("implicit", "implicit", GOLD, "s"),
              ("cn", "Crank–Nicolson", GREEN, "^"), ("cn-rannacher", "C–N + Rannacher", INK, "D"))
    levels = [50, 100, 200, 400, 800, 1600]
    for scheme, lab, col, mk in styles:
        lev = levels[:-1] if scheme == "explicit" else levels
        tab = convergence_table("call", S0, K, T, R, SIGMA, lev, scheme, lam=0.4, ratio=1.0)
        ax.loglog(tab["h"], np.abs(tab["error"]), color=col, marker=mk, ms=3, lw=1.2, label=lab)
    hh = np.array([0.0015, 0.05])
    ax.loglog(hh, 0.25 * hh, color=GREY, lw=0.8, ls=":")
    ax.loglog(hh, 3.0 * hh**2, color=GREY, lw=0.8, ls=":")
    ax.text(0.03, 0.25 * 0.03 * 0.45, "slope 1", fontsize=6.5, color=GREY, ha="center")
    ax.text(0.03, 3.0 * 0.03**2 * 0.3, "slope 2", fontsize=6.5, color=GREY, ha="center")
    ax.set_xticks([0.002, 0.005, 0.01, 0.02, 0.05], ["0.002", "0.005", "0.01", "0.02", "0.05"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set(xlabel="space step $h$ (log-price)", ylabel="|price error| at $S_0=K$", xlim=(0.0012, 0.06), ylim=(4e-6, 8.0))
    ax.legend(fontsize=6.4, loc="upper left")

    n_x, n_t = 400, 25
    for ran, lab, col in ((0, "Crank–Nicolson", GREEN), (2, "C–N + Rannacher", INK)):
        res = european_fd("call", S0, K, T, R, SIGMA, n_x, n_t, theta=0.5, rannacher=ran)
        _, gam = grid_greeks(res["x"], res["v"])
        xs = res["x"][1:-1]
        m = (xs > 88) & (xs < 112)
        ax2.plot(xs[m], gam[m], color=col, lw=1.2 if ran else 1.0, marker="." if not ran else None, ms=3, label=lab)
    xx = np.linspace(88, 112, 200)
    ax2.plot(xx, greeks(xx, K, T, R, SIGMA, "call")["gamma"], color=GOLD, ls=(0, (4, 2)), lw=1.3, label="exact")
    ax2.set(xlabel="stock price $x$", ylabel=r"$\Gamma$ at $t=0$", xlim=(88, 112))
    ax2.legend(fontsize=6.6, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch16-convergence.pdf")
    plt.close(fig)


def american(out):
    """Left: American and European puts. Right: the early-exercise boundary, grid against tree."""
    res = american_put_fd(S0, K, T, R, SIGMA, n_x=800, n_t=400, keep=True)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.35))
    m = (res["x"] > 60) & (res["x"] < 140)
    xs = res["x"][m]
    ax.plot(xs, np.maximum(K - xs, 0.0), color=GREY, lw=0.9, label="payoff $(K-x)^+$")
    ax.plot(xs, put(xs, K, T, R, SIGMA), color=GOLD, ls=(0, (4, 2)), label="European put")
    ax.plot(xs, res["v"][m], color=COBALT, label="American put")
    b0 = res["boundary"][-1]   # the last time level is t = 0
    ax.axvline(b0, color=COBALT, lw=0.7, ls=":")
    ax.set(xlabel="stock price $x$", ylabel="price at $t=0$", xlim=(60, 140), ylim=(0, 40))
    ax.legend(fontsize=7, loc="upper right")

    N = 2000
    u, d, rn, _ = crr_parameters(T, N, R, SIGMA)
    bb = early_exercise_boundary(K, S0, u, d, rn, N)
    tt = np.linspace(0, T, N + 1)
    ax2.plot(tt[::10], bb[::10], ".", color=GOLD, ms=2.2, label=f"binomial tree, $N={N}$")
    ax2.plot(res["t"], res["boundary"], color=COBALT, lw=1.3, label="grid (PSOR)")
    L = 2 * R * K / (2 * R + SIGMA**2)
    ax2.axhline(K, color=GREY, lw=0.8)
    ax2.axhline(L, color=GREEN, lw=1.0, ls=(0, (4, 2)), label=r"perpetual level $L^*$")
    ax2.set(xlabel="time $t$", ylabel="exercise boundary $b(t)$", xlim=(0, T), ylim=(68, 102))
    ax2.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "ch16-american.pdf")
    plt.close(fig)


def main(out):
    use_style()
    monte_carlo(out)
    stability(out)
    convergence(out)
    american(out)
