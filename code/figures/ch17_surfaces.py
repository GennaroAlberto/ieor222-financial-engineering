"""Figures for Chapter 17 (arbitrage-free option surfaces: SANOS and DYSANOS)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieor222 import surfaces as S
from ieor222.style import COBALT, GOLD, GREY, INK, PALETTE, use_style
from ieor222.volatility import heston_call, heston_surface

GREEN = PALETTE[2]


def _iv(c, x, T):
    return 100 * np.sqrt(S.implied_total_variance(c, x) / T)


def naive(out, ex=None):
    """Naive interpolation of mid implied volatilities against SANOS, at the 3-month expiry."""
    ex = ex or S.sanos_example()
    j = 2
    qt, T = ex["quotes"][j], ex["T"][j]
    xq = qt["x"]
    k = np.linspace(np.log(xq[0]) - 0.04, np.log(xq[-1]) + 0.04, 1201)
    x = np.exp(k)
    lin = S.naive_vol(ex["quotes"], ex["T"], "linear")
    spl = S.naive_vol(ex["quotes"], ex["T"], "spline")
    sanos = S.sanos_surface(ex["fit"]["q"], ex["K"], ex["T"], ex["v"], T, x)
    true = heston_call(1.0, x, T, 0.0, **S.HESTON)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.15))
    ivb, iva = _iv(qt["bid"], xq, T), _iv(qt["ask"], xq, T)
    ax.vlines(np.log(xq), ivb, iva, color=INK, lw=1.0, label="bid--ask")
    ax.plot(k, _iv(true, x, T), color=GOLD, ls="--", lw=1.1, label="Heston")
    ax.plot(k, _iv(lin(T, x), x, T), color=GREEN, lw=0.9, label="linear in $w$")
    ax.plot(k, _iv(sanos, x, T), color=COBALT, lw=1.4, label="SANOS")
    ax.set(xlabel="log-moneyness $k=\\log x$", ylabel="implied volatility (%)", title="3-month smile",
           xlim=(k[0], k[-1]))
    ax.legend(fontsize=6.5, loc="upper right", handlelength=1.4)
    # densities of log(Z_T): x * d^2c/dx^2
    def dens(c):
        h = np.diff(x)
        s = np.diff(c) / h
        return x[1:-1] * np.diff(s) / (0.5 * (x[2:] - x[:-2]))
    srf = heston_surface(k, [T], **S.HESTON)
    ax2.axhline(0, color=GREY, lw=0.6)
    ax2.plot(k, srf["density"][0], color=GOLD, ls="--", lw=1.1, label="Heston")
    d_spl = dens(spl(T, x))
    ax2.plot(k[1:-1], d_spl, color=GREEN, lw=0.9, label="cubic spline in $\\sigma$")
    ax2.fill_between(k[1:-1], d_spl, 0, where=d_spl < 0, color=GREEN, alpha=0.35, lw=0)
    ax2.plot(k, x * S.sanos_density(ex["fit"]["q"][j], ex["K"], x, ex["v"][j]), color=COBALT, lw=1.4,
             label="SANOS")
    w_mid = S.implied_total_variance(qt["mid"], xq)
    kinks = S.concave_kinks(np.log(xq), w_mid)
    ax2.plot(np.log(xq[kinks]), np.full(kinks.size, -0.55), "v", ms=3.2, color=GREEN, mec="none",
             label="concave kinks")
    ax2.set(xlabel="log-moneyness $k$", ylabel="density of $\\log Z_T$", title="implied densities",
            xlim=(k[0], k[-1]), ylim=(-1.1, 5.4))
    ax2.legend(fontsize=6.3, loc="upper left", handlelength=1.4)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "ch17-naive.pdf")
    plt.close(fig)


def sanos_fit(out, ex=None):
    """The SANOS fit: total implied variance across maturities, and the LP density with its smoothing."""
    ex = ex or S.sanos_example()
    q, K, T, v = ex["fit"]["q"], ex["K"], ex["T"], ex["v"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.15))
    Tall = np.sort(np.concatenate([T, 0.5 * (T[1:] + T[:-1])]))
    for t in Tall:
        k = np.linspace(-3.5 * 0.2 * np.sqrt(t), 2.5 * 0.2 * np.sqrt(t), 300)   # about the quoted range
        x = np.exp(k)
        w = S.implied_total_variance(S.sanos_surface(q, K, T, v, t, x), x)
        model = np.isclose(t, T).any()
        ax.plot(k, w, color=COBALT if model else GREY, lw=1.1 if model else 0.7, ls="-" if model else ":")
    for qt in ex["quotes"]:
        ax.plot(np.log(qt["x"]), S.implied_total_variance(qt["mid"], qt["x"]), "o", ms=1.6, color=INK)
    ax.plot([], [], color=COBALT, lw=1.1, label="expiries")
    ax.plot([], [], color=GREY, lw=0.7, ls=":", label="in between")
    ax.plot([], [], "o", ms=1.6, color=INK, label="mid quotes")
    ax.set(xlabel="log-moneyness $k$", ylabel="total implied variance $w$", xlim=(-0.72, 0.5), ylim=(0, 0.1),
           title="no calendar crossing")
    ax.legend(fontsize=6.5, loc="upper center", handlelength=1.4)
    j = 5
    kk = np.linspace(-0.75, 0.5, 600)
    xx = np.exp(kk)
    srf = heston_surface(kk, [T[j]], **S.HESTON)
    ax2.plot(kk, srf["density"][0], color=GOLD, ls="--", lw=1.1, label="Heston")
    ax2.plot(kk, xx * S.sanos_density(q[j], K, xx, v[j]), color=COBALT, lw=1.4, label="SANOS")
    nz = q[j] > 1e-10
    scale = 10.0
    ax2.vlines(np.log(K[nz]), 0, scale * q[j][nz], color=GREEN, lw=1.6, label=f"LP weights $q_j^i$ ($\\times{scale:g}$)")
    ax2.set(xlabel="log-moneyness $k$", ylabel="density of $\\log Z_T$", xlim=(kk[0], kk[-1]),
            title=f"$T={T[j]:g}$: weights and smoothing")
    ax2.legend(fontsize=6.3, loc="upper left", handlelength=1.4)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "ch17-sanos.pdf")
    plt.close(fig)


def dlv_and_dysanos(out, hist=None, gen=None):
    """Left: implied discrete local volatility against Dupire's. Right: DYSANOS-type generated paths."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.15))
    T1 = 0.5
    ke = np.linspace(-0.45, 0.35, 161)
    lv = S.dupire_reference(T1, ke)
    ax.plot(ke, 100 * lv, color=INK, lw=1.3, label="Dupire (Ch. 15)")
    for (dT, h, c, lab) in ((1 / 12, 0.05, GOLD, "$\\Delta T=1/12$, $h=0.05$"),
                            (1 / 52, 0.02, COBALT, "$\\Delta T=1/52$, $h=0.02$")):
        y, sig = S.heston_implied_dlv(T1, dT, h)
        keep = (y >= ke[0]) & (y <= ke[-1])
        ax.plot(y[keep], 100 * sig[keep], "o", ms=2.4, color=c, label=lab)
    ax.set(xlabel="log-moneyness $\\log x$", ylabel="local volatility (%)", title=f"discrete local vol., $T={T1:g}$")
    ax.legend(fontsize=6.3, loc="upper right", handlelength=1.4)
    hist = hist or S.synthetic_history()
    gen = gen or S.fit_generator(hist)
    Tm, K, basis = hist["T"], hist["K"], hist["basis"]
    q, v = S.decode(hist["U"], K, Tm, basis, n_expiries=3)
    atm_hist = 100 * S.atm_vol(q, v, K, Tm, 2)
    n_days = len(atm_hist)
    days = np.arange(n_days)
    ax2.plot(days, atm_hist, color=INK, lw=0.8, label="fitted history")
    horizon, n_paths = 120, 400
    hp = S.simulate_ar1(gen["m"], gen["A"], gen["Omega"], gen["H"][-1], horizon, n_paths, seed=17)
    qg, vg = S.decode(gen["u0"] + hp @ gen["Lam"].T, K, Tm, basis, n_expiries=3)
    atm_gen = 100 * S.atm_vol(qg, vg, K, Tm, 2)
    fut = n_days - 1 + np.arange(horizon + 1)
    lo, hi = np.percentile(atm_gen, [5, 95], axis=0)
    ax2.fill_between(fut, lo, hi, color=COBALT, alpha=0.18, lw=0, label="generated, 5--95%")
    for i, c in zip(range(3), (COBALT, GOLD, GREEN)):
        ax2.plot(fut, atm_gen[i], color=c, lw=0.7)
    ax2.set(xlabel="trading day", ylabel="3-month ATM vol. (%)", title="AR(1) generator", xlim=(0, fut[-1]))
    ax2.legend(fontsize=6.3, loc="upper left", handlelength=1.4)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "ch17-dlv-dysanos.pdf")
    plt.close(fig)


def main(out):
    use_style()
    ex = S.sanos_example()
    naive(out, ex)
    sanos_fit(out, ex)
    dlv_and_dysanos(out)


if __name__ == "__main__":
    import pathlib
    main(pathlib.Path("."))
