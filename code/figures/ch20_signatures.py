"""Figures for Chapter 20 (signatures and signature-based models)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieor222 import signatures as sg
from ieor222.style import COBALT, GOLD, GREY, INK, PALETTE, use_style

GREEN = PALETTE[2]


def _planar_path():
    """A smooth planar curve from (0, 0) to (1, 0.4), sampled at 200 points."""
    s = np.linspace(0.0, 1.0, 200)
    return np.stack([s + 0.1 * np.sin(2 * np.pi * s), 0.4 * s - 0.3 * np.sin(np.pi * s)], axis=1)


def levy_area(out):
    """Left: the Lévy area is the signed area between a path and its chord. Right: a tree-like excursion."""
    X = _planar_path()
    S = sg.signature(X, 4)
    area = 0.5 * (S[2][1] - S[2][2])
    # A tree-like excursion: at the 80th point, go out along a zig-zag branch and come back.
    a = X[80]
    branch = a + np.array([[0.0, 0.0], [0.12, -0.18], [0.02, -0.33], [0.2, -0.45]])
    Y = np.concatenate([X[:81], branch[1:], branch[::-1][1:], X[81:]])
    SY = sg.signature(Y, 4)
    gap = max(np.max(np.abs(u - v)) for u, v in zip(S, SY))
    assert gap < 1e-12, gap

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax = axes[0]
    ax.fill(np.r_[X[:, 0], X[0, 0]], np.r_[X[:, 1], X[0, 1]], color=COBALT, alpha=0.15, lw=0)
    ax.plot(X[:, 0], X[:, 1], color=COBALT, label="path $X$")
    ax.plot([X[0, 0], X[-1, 0]], [X[0, 1], X[-1, 1]], color=INK, lw=0.9, ls="--", label="chord")
    ax.annotate("", xy=X[66], xytext=X[56], arrowprops=dict(arrowstyle="-|>", color=COBALT, lw=1.2))
    ax.text(0.5, 0.06, f"$A = {area:.3f}$", color=INK, fontsize=9, ha="center")
    ax.set_title(r"Lévy area $\frac{1}{2}(\mathrm{Sig}^{12}-\mathrm{Sig}^{21})$")
    ax.set_xlabel("$X^1$")
    ax.set_ylabel("$X^2$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.plot(Y[:, 0], Y[:, 1], color=COBALT, lw=1.2, label="$X$ with an excursion")
    ax.plot(branch[:, 0], branch[:, 1], color=GOLD, lw=2.2, label="out and back")
    ax.set_title("same signature (tree-like part)")
    ax.set_xlabel("$X^1$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ch20-levy-area.pdf")
    plt.close(fig)
    return area


def brownian_level2(out):
    """Level 2 of piecewise-linear Brownian paths: Itô sum plus half the covariation (Stratonovich)."""
    rho, K, n_paths = 0.5, 13, 800
    corr = [[1.0, rho], [rho, 1.0]]
    t, W = sg.correlated_brownian(1.0, 2**K, corr, n_paths, seed=20)
    ns = np.array([2**k for k in range(2, K + 1)])
    s12, ito, cov = [], [], []
    for n in ns:
        Wn = W[:, :: 2**K // n]
        dW = np.diff(Wn, axis=1)
        ito.append(np.sum(Wn[:, :-1, 0] * dW[:, :, 1], axis=1))
        cov.append(np.sum(dW[:, :, 0] * dW[:, :, 1], axis=1))
        s12.append(sg.level2(Wn)[:, 0, 1])
        if n <= 2**8:                              # the closed form agrees with Chen's recursion
            assert np.allclose(s12[-1], sg.signature(Wn, 2)[2][:, 1], atol=1e-12)
    s12, ito, cov = map(np.array, (s12, ito, cov))
    assert np.allclose(s12, ito + 0.5 * cov, atol=1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax = axes[0]
    j = 0
    ax.plot(ns, s12[:, j], color=COBALT, marker="o", ms=2.5, label=r"$\mathrm{Sig}^{12}(\Pi_n)$")
    ax.plot(ns, ito[:, j], color=GOLD, marker="o", ms=2.5, label="Itô sum")
    ax.plot(ns, s12[:, j] - ito[:, j], color=GREEN, marker="o", ms=2.5, label="difference")
    ax.axhline(rho / 2, color=INK, lw=0.8, ls="--")
    ax.text(ns[-1], rho / 2 - 0.13, r"$\rho/2$", color=INK, fontsize=9, ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("intervals $n$")
    ax.set_title(r"one path, $\rho=0.5$")
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    rms_sig = np.sqrt(np.mean((s12[:-1] - s12[-1]) ** 2, axis=1))
    rms_diff = np.sqrt(np.mean((0.5 * cov - rho / 2) ** 2, axis=1))
    ax.plot(ns[:-1], rms_sig, color=COBALT, marker="o", ms=2.5, label=r"$\mathrm{Sig}^{12}(\Pi_n)-\mathrm{Sig}^{12}(\Pi_{2^{13}})$")
    ax.plot(ns, rms_diff, color=GREEN, marker="o", ms=2.5, label=r"$\frac{1}{2} C(W^1,W^2;\Pi_n)-\frac{\rho}{2}$")
    ax.plot(ns, 0.5 * np.sqrt((1 + rho**2) / ns), color=INK, lw=0.8, ls="--", label=r"$\frac{1}{2}\sqrt{(1+\rho^2)/n}$")
    exact_sig = 0.5 * np.sqrt((1 - rho**2) * (1 / ns[:-1] - 1 / ns[-1]))   # Lévy areas lost on each interval
    ax.plot(ns[:-1], exact_sig, color=INK, lw=0.8, ls=":", label=r"$\frac{1}{2}\sqrt{(1-\rho^2)(1/n-1/N)}$")
    print("level-2 figure: simulated/exact RMS of the signature error:", np.round(rms_sig / exact_sig, 3))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("intervals $n$")
    ax.set_title(f"root-mean-square errors, {n_paths} paths")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch20-brownian-level2.pdf")
    plt.close(fig)


def regression(out):
    """Out-of-sample fit of path-dependent payoffs by linear functionals of the signature."""
    depths = (1, 2, 3, 4, 5, 6)
    res = sg.regression_experiment(depths=depths)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax = axes[0]
    for (name, label), color in zip((("asian", "Asian call"), ("max", "maximum of $W$")), (COBALT, GOLD)):
        ax.plot(depths, 1 - res[name][0], color=color, marker="o", ms=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("truncation depth $N$")
    ax.set_ylabel(r"$1-R^2$ (test paths)")
    ax.legend(fontsize=8)
    ax = axes[1]
    _, y, pred = res["asian"]
    ax.scatter(y, pred, s=2, color=COBALT, alpha=0.4, lw=0)
    lim = [min(y.min(), pred.min()), y.max()]
    ax.plot(lim, lim, color=INK, lw=0.8)
    ax.set_xlabel("Asian call payoff")
    ax.set_ylabel(f"linear functional, $N={depths[-1]}$")
    fig.tight_layout()
    fig.savefig(out / "ch20-regression.pdf")
    plt.close(fig)
    return res


def calibration(out):
    """Linear signature model (n = 2 and n = 3) calibrated to four Heston smiles."""
    r3 = sg.calibration_experiment(n=3, n_paths=50_000, n_check=60_000)
    r2 = sg.calibration_experiment(n=2, n_paths=30_000, n_check=30_000)
    Ts = sorted(r3["iv_market"])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    ax = axes[0]
    for T, color in zip(Ts, PALETTE):
        K = r3["quotes"][T][0]
        lab = f"$T={T:.2f}$" if T >= 0.25 else "$T=1/12$"
        ax.plot(K, 100 * r3["iv_market"][T], ls="none", marker="o", ms=3.5, color=color, label=lab)
        ax.plot(K, 100 * r3["iv_out"][T], color=color, lw=1.2)
        ax.plot(K, 100 * r2["iv_out"][T], color=color, lw=0.8, ls=":")
    ax.set_xlabel("strike $K/S_0$")
    ax.set_ylabel("implied volatility (%)")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    ax = axes[1]
    x = r3["x"]
    for T, color in zip(Ts, PALETTE):
        ax.plot(x, 1e4 * (r3["iv_out"][T] - r3["iv_market"][T]), color=color, marker="o", ms=2.5, lw=1.0)
    se = np.max([np.hypot(r3["iv_se"][T], r3["iv_se_fit"][T]) for T in Ts], axis=0)
    ax.fill_between(x, -2e4 * se, 2e4 * se, color=GREY, alpha=0.18, lw=0, label=r"$\pm2$ MC s.e.")
    ax.axhline(0.0, color=INK, lw=0.6)
    ax.set_xlabel(r"$\log(K/S_0)\,/\,(0.2\sqrt{T})$")
    ax.set_ylabel("model $-$ Heston (bp)")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ch20-calibration.pdf")
    plt.close(fig)
    return r2, r3


def _report(r, name):
    Ts = sorted(r["iv_market"])
    e_in = np.concatenate([r["iv_in"][T] - r["iv_market"][T] for T in Ts])
    e_out = np.concatenate([r["iv_out"][T] - r["iv_market"][T] for T in Ts])
    se = np.concatenate([np.hypot(r["iv_se"][T], r["iv_se_fit"][T]) for T in Ts])
    print(f"{name}: {len(r['words'])} parameters; in-sample RMS {1e4 * np.sqrt(np.mean(e_in**2)):.1f} bp "
          f"(max {1e4 * np.abs(e_in).max():.1f}); independent paths RMS {1e4 * np.sqrt(np.mean(e_out**2)):.1f} bp "
          f"(max {1e4 * np.abs(e_out).max():.1f}); combined MC s.e. RMS {1e4 * np.sqrt(np.mean(se**2)):.1f} bp (max {1e4 * se.max():.1f}); "
          f"P(S_T<=0) at T=1: {r['prob_negative'][Ts[-1]]:.1e}; E[S_T]-S_0 at T=1: {r['mean_bias'][Ts[-1]]:.1e}; sample {r['times'][0]:.1f}s, fit {r['times'][1]:.1f}s")


def main(out):
    use_style()
    area = levy_area(out)
    print(f"Lévy area of the planar path: {area:.4f}")
    brownian_level2(out)
    res = regression(out)
    for k, v in res.items():
        print(f"regression {k}: R^2 by depth {np.round(v[0], 3)}")
    r2, r3 = calibration(out)
    _report(r2, "n=2")
    _report(r3, "n=3")
