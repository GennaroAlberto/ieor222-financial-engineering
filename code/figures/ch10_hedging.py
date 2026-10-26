"""Figures for Chapter 10 (the Black--Scholes equation by hedging)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from ieor222.blackscholes import call, gbm_paths
from ieor222.hedging import (asymptotic_rms, call_delta, call_gamma, call_hedge, gamma_moment_call,
                             gamma_theta, hedged_pnl, hedging_error)
from ieor222.style import COBALT, GOLD, GREY, INK, use_style

GREEN = "#1BAF7A"
S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


def _call_fns():
    value = lambda t, x: call(x, K, T - t, R, SIGMA)
    delta = lambda t, x: call_delta(x, K, T - t, R, SIGMA)
    gamma = lambda t, x: call_gamma(x, K, T - t, R, SIGMA)
    return value, delta, gamma


def hedging_error_figure(out):
    """Terminal error of the discrete delta hedge of a call: histograms, and rms against n for three drifts."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    mu = 0.15
    bins = np.linspace(-8, 8, 81)
    for n, color, label in ((12, GOLD, "monthly"), (52, COBALT, "weekly"), (252, GREEN, "daily")):
        e = call_hedge(S0, K, T, R, SIGMA, mu, n, 20_000, seed=n)["error"]
        ax.hist(np.clip(e, bins[0], bins[-1]), bins=bins, density=True, histtype="step", lw=1.3,
                color=color, label=f"{label}, rms {np.sqrt(np.mean(e**2)):.2f}")
    ax.set(xlabel=r"hedging error $X_T-(S_T-K)^+$", ylabel="density",
           title=r"$n=12,52,252$ rebalancing dates, $\mu=15\%$")
    ax.legend(fontsize=7, loc="upper left")

    ns = 2 ** np.arange(2, 11)
    value, delta, _ = _call_fns()
    payoff = lambda x: np.maximum(x - K, 0.0)
    for mu, color, marker in ((-0.10, GOLD, "s"), (0.05, COBALT, "o"), (0.30, GREEN, "^")):
        t, S = gbm_paths(S0, T, mu, SIGMA, ns[-1], 6_000, seed=int(100 * (mu + 1)))
        rms = []
        for n in ns:
            step = ns[-1] // n
            e = hedging_error(t[::step], S[:, ::step], value, delta, payoff, R)
            rms.append(np.sqrt(np.mean(e**2)))
        moment = lambda u, mu=mu: gamma_moment_call(u, S0, K, T, R, SIGMA, mu)
        lead = [np.exp(R * T) * asymptotic_rms(moment, T, SIGMA, n) for n in ns]
        ax2.plot(ns, rms, marker, ms=3.5, color=color, label=rf"$\mu={100 * mu:.0f}\%$")
        ax2.plot(ns, lead, "--", lw=0.9, color=color)
    ax2.set(xscale="log", yscale="log", xlabel="number of rebalancing dates $n$",
            ylabel="root-mean-square error", title="error $\\approx$ constant $\\times\\, n^{-1/2}$")
    ax2.set_xscale("log", base=2)
    ax2.set_yticks([0.2, 0.5, 1, 2, 4], ["0.2", "0.5", "1", "2", "4"])
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch10-hedging-error.pdf")
    plt.close(fig)


def gamma_theta_figure(out):
    """One-period P&L of a hedged long call against the move, and the gamma--theta sum along paths."""
    value, delta, gamma = _call_fns()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    h = 1 / 52
    R_move = np.linspace(-0.08, 0.08, 401)
    Y1 = S0 * (1 + R_move)                      # discounted price at time h
    exact = np.exp(-R * h) * value(h, np.exp(R * h) * Y1) - value(0.0, S0) - delta(0.0, S0) * S0 * R_move
    G = gamma(0.0, S0) * S0**2
    ax.plot(100 * R_move, exact, color=COBALT, label="exact P&L")
    ax.plot(100 * R_move, 0.5 * G * (R_move**2 - SIGMA**2 * h), "--", color=GOLD,
            label=r"$\frac{1}{2}\Gamma S^2(R^2-\sigma^2h)$")
    for side in (-1, 1):
        ax.axvline(100 * side * SIGMA * np.sqrt(h), color=GREY, lw=0.6)
    ax.axhline(0.0, color=GREY, lw=0.6)
    ax.set(xlabel="discounted return $R$ over one week (%)", ylabel="P&L of the hedged long call",
           title=r"$S=K=100$, one year to maturity")
    ax.legend(fontsize=7, loc="upper center")

    n, mu = 52, 0.15
    t, S = gbm_paths(S0, T, mu, SIGMA, n, 3_000, seed=11)
    err = -hedged_pnl(t, S, value, delta, R).sum(axis=1)               # discounted hedging error
    lead = -gamma_theta(t, S, gamma, R, SIGMA).sum(axis=1)
    lim = np.quantile(np.abs(err), 0.995)
    ax2.plot(lead, err, ".", ms=1.5, color=COBALT, alpha=0.5)
    ax2.plot([-lim, lim], [-lim, lim], color=INK, lw=0.8)
    corr = np.corrcoef(lead, err)[0, 1]
    ax2.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=r"$-\sum_k\frac{1}{2}e^{-rt_k}\Gamma_kS_{t_k}^2(R_k^2-\sigma^2h)$",
            ylabel="discounted hedging error", title=f"weekly hedge, 3000 paths, correlation {corr:.3f}")
    fig.tight_layout()
    fig.savefig(out / "ch10-gamma-theta.pdf")
    plt.close(fig)


def main(out):
    use_style()
    hedging_error_figure(out)
    gamma_theta_figure(out)
