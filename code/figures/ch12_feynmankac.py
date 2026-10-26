"""Figures for Chapter 12 (the Feynman--Kac formula)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ieor222.blackscholes import call, digital_call
from ieor222.feynmankac import (digital_delta, gbm_density, local_vol_mc, price_at, skew_vol,
                                solve_pricing_pde)
from ieor222.style import COBALT, GOLD, GREY, INK, PALETTE, use_style

GREEN = PALETTE[2]
K, T, R = 100.0, 1.0, 0.03          # the local-volatility examples
LV = lambda t, x: skew_vol(t, x)     # 25% at x = 100, from about 13% to 37%


def mc_versus_pde(out):
    """The same prices computed two ways: one PDE solve for all spots, Monte Carlo spot by spot."""
    g = lambda x: np.maximum(x - K, 0.0)
    res = solve_pricing_pde(g, T, R, LV, 5.0, 2000.0, n_x=1600, n_t=800)
    xs = np.linspace(60.0, 150.0, 400)
    spots = np.arange(70.0, 141.0, 10.0)
    mc = [local_vol_mc(g, s, T, R, LV, 100, 100_000, seed=100 + i) for i, s in enumerate(spots)]
    est = np.array([m["mean"] for m in mc])
    se = np.array([m["se"] for m in mc])
    pde_at = price_at(res, spots)

    # plot the time value v - (x - K e^{-rT})^+, where the two models differ visibly
    intrinsic = lambda x: np.maximum(x - K * np.exp(-R * T), 0.0)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.15), gridspec_kw={"width_ratios": [1.25, 1]})
    ax.plot(xs, price_at(res, xs) - intrinsic(xs), color=COBALT, lw=1.6, label="PDE, local volatility")
    ax.plot(xs, call(xs, K, T, R, 0.25) - intrinsic(xs), color=GOLD, lw=1.2, ls=(0, (4, 2)),
            label=r"Black–Scholes, $\sigma=25\%$")
    ax.errorbar(spots, est - intrinsic(spots), yerr=1.96 * se, fmt="o", ms=3.2, color=INK,
                elinewidth=1.0, capsize=0, label="Monte Carlo, 95% CI")
    ax.set(xlabel="spot $x$", ylabel=r"$v(0,x)-(x-Ke^{-rT})^+$", xlim=(60, 150), ylim=(0, None))
    ax.legend(loc="upper right", fontsize=7.2)

    ax2.axhline(0.0, color=GREY, lw=0.8)
    ax2.errorbar(spots, est - pde_at, yerr=1.96 * se, fmt="o", ms=3.2, color=INK, elinewidth=1.0, capsize=0)
    ax2.set(xlabel="spot $x$", ylabel="Monte Carlo $-$ PDE", xlim=(65, 145))
    lim = 1.15 * np.max(np.abs(est - pde_at) + 1.96 * se)
    ax2.set_ylim(-lim, lim)
    fig.tight_layout()
    fig.savefig(out / "ch12-mc-pde.pdf")
    plt.close(fig)


def kolmogorov_gbm(out):
    """GBM transition densities: forward in the arrival point, backward in the starting point."""
    mu, sigma, x0, y0 = 0.08, 0.3, 100.0, 100.0
    rng = np.random.default_rng(12)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.2))
    y = np.linspace(1.0, 300.0, 600)
    for s, c in ((0.1, COBALT), (1.0, GOLD), (4.0, GREEN)):
        S = x0 * np.exp((mu - 0.5 * sigma**2) * s + sigma * np.sqrt(s) * rng.normal(size=40_000))
        ax.hist(S, bins=np.linspace(0, 300, 121), density=True, color=c, alpha=0.22, lw=0)
        ax.plot(y, gbm_density(0.0, x0, s, y, mu, sigma), color=c, lw=1.5, label=f"$s={s:g}$")
    ax.set(xlabel="arrival point $y$", ylabel=r"$p(0,100;s,y)$", xlim=(0, 300), ylim=(0, None),
           title="forward: a density in $y$")
    ax.legend(fontsize=7.5, loc="upper right")

    x = np.linspace(1.0, 300.0, 600)
    for t, c in ((0.9, COBALT), (0.5, GOLD), (0.0, GREEN)):
        ax2.plot(x, gbm_density(t, x, 1.0, y0, mu, sigma), color=c, lw=1.5, label=f"$t={t:g}$")
    ax2.set(xlabel="starting point $x$", ylabel=r"$p(t,x;1,100)$", xlim=(0, 300), ylim=(0, None),
            title="backward: a function of $x$")
    ax2.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch12-kolmogorov.pdf")
    plt.close(fig)


def local_vol_surface(out):
    """The price v(t, x) of a call in the local volatility model, and its difference from Black--Scholes."""
    g = lambda x: np.maximum(x - K, 0.0)
    res = solve_pricing_pde(g, T, R, LV, 5.0, 2000.0, n_x=800, n_t=200, keep=True)
    keep = (res["x"] > 50) & (res["x"] < 160)
    x, t = res["x"][keep], res["t"]
    V = res["surface"][:, keep]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.35), gridspec_kw={"width_ratios": [1.25, 1]})
    cmap = LinearSegmentedColormap.from_list("cobalt", ["#FFFFFF", "#9FB5FF", COBALT, "#0B2A8F"])
    cs = ax.contourf(x, t, V, levels=np.arange(0, 62, 4), cmap=cmap)
    ax.contour(x, t, V, levels=np.arange(0, 62, 4), colors=INK, linewidths=0.35, alpha=0.5)
    ax.axvline(K, color=GOLD, lw=1.0, ls=(0, (4, 2)))
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label("$v(t,x)$", color=INK)
    cb.ax.tick_params(labelsize=7)
    ax.set(xlabel="stock price $x$", ylabel="time $t$", title="call price, local volatility")
    ax.grid(False)

    xs = np.linspace(50.0, 160.0, 300)
    for tau, c in ((0.25, COBALT), (0.5, GOLD), (1.0, GREEN)):
        row = np.argmin(np.abs(t - (T - tau)))
        sl = _interp(res["x"], res["surface"][row])
        ax2.plot(xs, sl(xs) - call(xs, K, tau, R, 0.25), color=c, lw=1.5, label=rf"$T-t={tau:g}$")
    ax2.axhline(0.0, color=GREY, lw=0.8)
    ax2.axvline(K, color=GOLD, lw=0.8, ls=(0, (4, 2)))
    ax2.set(xlabel="stock price $x$", ylabel=r"$v(t,x)-c_{25\%}(t,x)$", title="against Black–Scholes at 25%")
    ax2.legend(fontsize=7.2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch12-local-vol.pdf")
    plt.close(fig)


def _interp(x, v):
    """Cubic interpolation in log x of values on a price grid."""
    return lambda z: price_at(dict(x=x, v=v), z)


def digital(out):
    """Price and delta of the digital call near maturity; the delta is a transition density."""
    r, sigma = 0.05, 0.2
    x = np.linspace(70.0, 130.0, 601)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.15))
    ax.plot(x, (x > K).astype(float), color=GREY, lw=1.0, label="payoff")
    for tau, c in ((1.0, GREEN), (0.1, GOLD), (0.01, COBALT)):
        ax.plot(x, digital_call(x, K, tau, r, sigma), color=c, lw=1.5, label=rf"$\tau={tau:g}$")
        ax2.plot(x, digital_delta(x, K, tau, r, sigma), color=c, lw=1.5, label=rf"$\tau={tau:g}$")
    ax.set(xlabel="stock price $x$", ylabel=r"price $D(t,x)$", title="digital call: price")
    ax.legend(fontsize=7.2, loc="upper left")
    ax2.set(xlabel="stock price $x$", ylabel=r"delta $\partial_xD(t,x)$", title="digital call: delta")
    ax2.legend(fontsize=7.2, loc="upper right")
    ax2.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out / "ch12-digital.pdf")
    plt.close(fig)


def main(out):
    use_style()
    mc_versus_pde(out)
    kolmogorov_gbm(out)
    local_vol_surface(out)
    digital(out)
