"""Figures for Chapter 15 (implied volatility and the smile)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieor222.style import COBALT, GOLD, GREY, INK, PALETTE, use_style
from ieor222.volatility import (autocorrelation, black_normalized, butterfly_g, cross_correlation,
                                density_from_w, dupire_heston_grid, heston_call, heston_expected_variance,
                                heston_paths, implied_vol, kurtosis, local_vol_paths, mc_call_prices,
                                ssvi, strip_variance, svi_raw, up_and_out_call, vix_sigma2)

GREEN = PALETTE[2]
S0, R = 100.0, 0.02
# the "market" of the chapter: Heston under Q
HESTON = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
# stylised facts: Heston under P, daily returns
P_MODEL = dict(mu=0.07, v0=0.04, kappa=4.0, theta=0.04, xi=0.6, rho=-0.7)
VOGT = dict(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sig=0.4153)   # Gatheral--Jacquier (2014), Vogt's example


def daily_returns(years=20, per_day=5, seed=15, **model):
    """Daily log returns of one Heston path (5 Euler steps per day), and of GBM with the same mean variance."""
    n_days = 252 * years
    out = heston_paths(1.0, years, model["mu"], model["v0"], model["kappa"], model["theta"], model["xi"],
                       model["rho"], n_days * per_day, 1, seed=seed, keep=True)
    logS = np.log(out["S_path"][0, ::per_day])
    r = np.diff(logS)
    rng = np.random.default_rng(seed + 1)
    g = rng.normal(np.mean(r), np.std(r), size=r.size)
    return r, g


def stylised(out):
    r, g = daily_returns(**P_MODEL)
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 1.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axs[0]
    t = np.arange(r.size) / 252
    ax.plot(t, 100 * r, color=COBALT, lw=0.35)
    ax.set(xlabel="years", ylabel="daily log return (%)", xlim=(0, t[-1]), title="returns: Heston under $\\mathbb{P}$")
    ax = axs[1]
    z = (r - r.mean()) / r.std()
    bins = np.linspace(-7, 7, 57)
    hist, edges = np.histogram(z, bins=bins, density=True)
    mid = 0.5 * (edges[1:] + edges[:-1])
    keep = hist > 0
    ax.semilogy(mid[keep], hist[keep], "o", ms=2.4, color=COBALT, label="Heston")
    x = np.linspace(-7, 7, 300)
    ax.semilogy(x, np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi), color=GOLD, lw=1.2, label="normal")
    ax.set(xlabel="standardised return", ylabel="density", ylim=(2e-7, 2), xlim=(-7, 7), title="heavy tails")
    ax.legend(fontsize=6.5, loc="lower center", handlelength=1.2)
    ax = axs[2]
    lags = np.arange(1, 61)
    ax.axhline(0, color=GREY, lw=0.6)
    band = 2 / np.sqrt(r.size)
    ax.fill_between(lags, -band, band, color=GREY, alpha=0.15, lw=0)
    ax.plot(lags, autocorrelation(r, 60), color=GOLD, lw=1.1, label="corr$(r_t,r_{t+j})$")
    ax.plot(lags, autocorrelation(np.abs(r), 60), color=COBALT, lw=1.3, label="corr$(|r_t|,|r_{t+j}|)$")
    ax.plot(lags, cross_correlation(r, np.abs(r), 60), color=GREEN, lw=1.3, label="corr$(r_t,|r_{t+j}|)$")
    ax.set(xlabel="lag $j$ (days)", title="clustering and leverage", xlim=(1, 60), ylim=(-0.2, 0.5))
    ax.legend(fontsize=5.8, loc="upper right", handlelength=1.2)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "ch15-stylised.pdf")
    plt.close(fig)


def static_arbitrage(out):
    """Butterfly arbitrage (a negative density) and calendar arbitrage (crossing total variances)."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.05))
    k = np.linspace(-1.5, 1.5, 1201)
    w, wk, wkk = svi_raw(k, **VOGT)
    dens = density_from_w(k, w, wk, wkk)
    ws, wks, wkks, _ = ssvi(k, 0.04, -0.7, 1.0, 0.5)
    ax.axhline(0, color=GREY, lw=0.6)
    ax.plot(k, density_from_w(k, ws, wks, wkks), color=GOLD, lw=1.3, label="SSVI")
    ax.plot(k, dens, color=COBALT, lw=1.5, label="Vogt's SVI")
    ax.fill_between(k, dens, 0, where=dens < 0, color=COBALT, alpha=0.35, lw=0)
    ax.set(xlabel="log-moneyness $k$", ylabel="density of $\\log(S_T/F)$", xlim=(-1.5, 1.5),
           title="butterfly condition: $g\\geq0$")
    ax.legend(fontsize=6.8, loc="upper left", handlelength=1.4)
    axin = ax.inset_axes([0.63, 0.40, 0.35, 0.42])
    axin.axhline(0, color=GREY, lw=0.5)
    axin.plot(k, butterfly_g(k, w, wk, wkk), color=COBALT, lw=1.0)
    axin.set_xlim(0.2, 1.5)
    axin.set_ylim(-0.06, 0.3)
    axin.tick_params(labelsize=5.5)
    axin.set_title("$g(k)$", fontsize=6.5, pad=1.5)
    w1 = ssvi(k, 0.02, -0.3, 1.5, 0.5)[0]
    w2 = ssvi(k, 0.04, -0.9, 1.5, 0.5)[0]
    ax2.plot(k, w1, color=GOLD, lw=1.3, label="$w(k,T_1)$")
    ax2.plot(k, w2, color=COBALT, lw=1.3, label="$w(k,T_2)$, $T_2>T_1$")
    ax2.fill_between(k, w1, w2, where=w1 > w2, color=COBALT, alpha=0.3, lw=0)
    ax2.set(xlabel="log-moneyness $k$", ylabel="total implied variance", xlim=(-1.5, 1.5), ylim=(0, None),
            title="calendar condition: $\\partial_Tw\\geq0$")
    ax2.legend(fontsize=7, loc="upper center")
    fig.tight_layout()
    fig.savefig(out / "ch15-static-arbitrage.pdf")
    plt.close(fig)


def heston_smile(out):
    """Heston smiles by Fourier inversion against Monte Carlo, and their term structure."""
    T = 1.0
    K = np.linspace(60, 150, 61)
    Kmc = np.arange(65, 146, 10.0)
    F = S0 * np.exp(R * T)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.1))
    for i, (rho, c) in enumerate(((-0.7, COBALT), (0.0, GOLD), (0.5, GREEN))):
        p = dict(HESTON, rho=rho)
        iv = implied_vol(heston_call(S0, K, T, R, **p), S0, K, T, R)
        ax.plot(np.log(K / F), 100 * iv, color=c, lw=1.4, label=f"$\\rho={rho:g}$")
        mc = heston_paths(S0, T, R, n_steps=250, n_paths=100_000, seed=20 + i, **p)
        price, se = mc_call_prices(mc["S"], Kmc, F, np.exp(-R * T))
        lo = implied_vol(price - 1.96 * se, S0, Kmc, T, R)
        hi = implied_vol(price + 1.96 * se, S0, Kmc, T, R)
        mid = implied_vol(price, S0, Kmc, T, R)
        ax.errorbar(np.log(Kmc / F), 100 * mid, yerr=100 * np.vstack([mid - lo, hi - mid]), fmt="o", ms=2.6,
                    color=INK, elinewidth=0.9, capsize=0, label="Monte Carlo, 95% CI" if i == 0 else None)
    ax.set(xlabel="log-moneyness $k=\\log(K/F)$", ylabel="implied volatility (%)", title="$T=1$", ylim=(8, 30))
    ax.legend(fontsize=6.6, loc="lower center", ncol=2, handlelength=1.4)
    for T2, c in ((0.1, COBALT), (0.5, GOLD), (2.0, GREEN)):
        F2 = S0 * np.exp(R * T2)
        z = np.linspace(-2.5, 2.0, 91)
        KK = F2 * np.exp(z * 0.2 * np.sqrt(T2))
        iv = implied_vol(heston_call(S0, KK, T2, R, **HESTON), S0, KK, T2, R)
        ax2.plot(z, 100 * iv, color=c, lw=1.4, label=f"$T={T2:g}$")
    ax2.set(xlabel="$k/(0.2\\sqrt{T})$", ylabel="implied volatility (%)", title="$\\rho=-0.7$")
    ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ch15-heston-smile.pdf")
    plt.close(fig)


def local_vol_vs_heston(out, n_paths=100_000):
    """Dupire's local volatility of the Heston surface; same vanillas, different barrier prices."""
    T, n_steps = 1.0, 200
    F = S0 * np.exp(R * T)
    tg = np.linspace(0.0, T, n_steps + 1)
    tl = tg.copy()
    tl[0] = 0.5 * tg[1]
    kg = np.linspace(-1.5, 1.0, 251)
    lv = dupire_heston_grid(tl, kg, **HESTON)
    lvp = local_vol_paths(S0, T, R, 0.0, tg, kg, lv, n_paths, seed=31, running_max=True)
    hp = heston_paths(S0, T, R, n_steps=n_steps, n_paths=n_paths, seed=32, running_max=True, **HESTON)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.1))
    K = np.linspace(65, 145, 81)
    k = np.log(K / F)
    iv = implied_vol(heston_call(S0, K, T, R, **HESTON), S0, K, T, R)
    ax.plot(k, 100 * iv, color=COBALT, lw=1.5, label="Heston implied vol.")
    for frac, ls in ((0.5, (0, (4, 2))), (1.0, "-")):
        j = int(round(frac * n_steps))
        kf = np.log(K / (S0 * np.exp(R * tg[j])))
        ax.plot(kf, 100 * np.interp(kf, kg, lv[j]), color=GOLD, lw=1.2, ls=ls,
                label=f"local vol. $\\sigma(t,\\cdot)$, $t={tg[j]:g}$")
    Kmc = np.arange(70, 141, 10.0)
    price, se = mc_call_prices(lvp["S"], Kmc, F, np.exp(-R * T))
    ax.plot(np.log(Kmc / F), 100 * implied_vol(price, S0, Kmc, T, R), "o", ms=3, color=INK,
            label="local vol. model, Monte Carlo")
    ax.set(xlabel="log-moneyness $k$", ylabel="volatility (%)", title="$T=1$: the same vanillas", ylim=(10, 47))
    ax.legend(fontsize=6.3, loc="upper right", frameon=True, facecolor="white", edgecolor=GREY, framealpha=1)
    Bs = np.arange(105, 171, 5.0)
    disc = np.exp(-R * T)
    for paths, c, lab in ((hp, COBALT, "Heston"), (lvp, GOLD, "local volatility")):
        v = np.array([up_and_out_call(paths["S"], paths["max"], 100.0, B, disc) for B in Bs])
        ax2.errorbar(Bs, v[:, 0], yerr=1.96 * v[:, 1], fmt="o-", ms=2.6, lw=1.1, color=c, elinewidth=0.9,
                     capsize=0, label=lab)
    vanilla = heston_call(S0, 100.0, T, R, **HESTON)[0]
    ax2.axhline(vanilla, color=GREY, lw=0.8, ls=(0, (3, 2)))
    ax2.text(106, vanilla + 0.25, "call, $K=100$", fontsize=6.5, color=GREY)
    ax2.set(xlabel="barrier $B$", ylabel="price", title="up-and-out call, $K=100$", ylim=(0, 10.8))
    ax2.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "ch15-local-vol.pdf")
    plt.close(fig)


def variance_swap(out):
    """The variance-swap strike from a strip of options: discretization error and term structure."""
    p = dict(v0=0.03, kappa=2.0, theta=0.05, xi=0.5, rho=-0.7)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.1))
    T = 30 / 365
    F = S0 * np.exp(R * T)
    exact = heston_expected_variance(T, p["v0"], p["kappa"], p["theta"])
    hs = np.array([0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0])
    for (lo, hi), c, lab in (((20.0, 300.0), COBALT, "strikes in $[20,300]$"),
                             ((85.0, 115.0), GOLD, "strikes in $[85,115]$")):
        err = []
        for h in hs:
            K = np.arange(lo, hi + 1e-9, h)
            C = heston_call(S0, K, T, R, **p)
            P = C - S0 + K * np.exp(-R * T)
            err.append(vix_sigma2(K, C, P, F, T, R) - exact)
        err = np.array(err)
        ax.loglog(hs, np.abs(err), "-", lw=1.1, color=c)
        ax.loglog(hs[err > 0], err[err > 0], "o", ms=3.2, color=c, label=lab)
        ax.loglog(hs[err < 0], -err[err < 0], "o", ms=3.2, mfc="white", color=c)
    ax.loglog(hs, hs**2 / (6 * T * 100.0**2), color=GREY, lw=0.9, ls=(0, (3, 2)), label="$h^2/(6TK_0^2)$")
    ax.set(xlabel="strike spacing $h$", ylabel="|CBOE estimate $-$ exact|", title="one month")
    ax.legend(fontsize=6.5, loc="upper left")
    Ts = np.array([1 / 12, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    tt = np.linspace(0.04, 3.0, 200)
    ax2.plot(tt, 100 * np.sqrt(heston_expected_variance(tt, p["v0"], p["kappa"], p["theta"])), color=COBALT, lw=1.5,
             label="$\\sqrt{K_{\\mathrm{var}}}$, exact")
    strip, atm = [], []
    for T2 in Ts:
        F2 = S0 * np.exp(R * T2)
        K = np.arange(1.0, 1500.0, 0.5)
        C = heston_call(S0, K, T2, R, **p)
        P = C - S0 + K * np.exp(-R * T2)
        strip.append(strip_variance(K, np.where(K < F2, P, C), F2, T2, R))
        atm.append(implied_vol(heston_call(S0, F2, T2, R, **p), S0, F2, T2, R)[0])
    ax2.plot(Ts, 100 * np.sqrt(strip), "o", ms=3.2, color=INK, label="strip of options")
    ax2.plot(Ts, 100 * np.array(atm), color=GOLD, lw=1.2, ls=(0, (4, 2)), marker="s", ms=2.5,
             label="at-the-money implied vol.")
    ax2.set(xlabel="maturity $T$", ylabel="volatility (%)", title="term structure", ylim=(15.0, 22.3))
    ax2.legend(fontsize=6.8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "ch15-variance-swap.pdf")
    plt.close(fig)


def main(out):
    use_style()
    stylised(out)
    static_arbitrage(out)
    heston_smile(out)
    local_vol_vs_heston(out)
    variance_swap(out)


if __name__ == "__main__":
    import pathlib

    main(pathlib.Path(__file__).resolve().parents[2] / "latex" / "figures")
