"""Static arbitrage and static replication (Chapter 3).

Conventions follow the notes. There are two dates, 0 and T. ``B`` is the time-0 price of
the zero-coupon bond paying 1 at T (``B = (1 + r)**-T`` with annual compounding), ``S0``
the stock price, and ``C(K)``, ``P(K)`` the time-0 prices of European calls and puts with
strike K and maturity T. A :class:`Portfolio` is static: chosen at 0, held to T, and its
payoff is a function of S_T alone.

Three groups of functions:

* :func:`find_arbitrages` checks quoted call prices (and puts, through parity) against the
  relations of the notes and returns, for each violation, a portfolio that is an arbitrage;
* :func:`state_prices` builds, for quotes that pass every check, state prices that reproduce
  them (the construction in the proof of the sufficiency proposition);
* :func:`replicate_payoff` gives the static replication of a payoff f(S_T) by a bond, the
  stock and finitely many puts and calls (a discrete Carr--Madan strip).

:func:`lognormal_call` and friends produce an arbitrage-free reference curve: the
Black--Scholes formula of Chapter 6, written with B and the total volatility v = sigma sqrt(T).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr


# ---------------------------------------------------------------------------
# Portfolios
# ---------------------------------------------------------------------------

@dataclass
class Portfolio:
    """Units of the bond, the stock, and calls and puts (dicts strike -> units).

    Positive units are long positions, negative units short positions.
    """
    bond: float = 0.0
    stock: float = 0.0
    calls: dict = field(default_factory=dict)
    puts: dict = field(default_factory=dict)

    def add_call(self, K: float, units: float) -> "Portfolio":
        """Add ``units`` calls with strike K; strike 0 means the stock (a call with zero strike)."""
        if K == 0:
            self.stock += units
        else:
            self.calls[K] = self.calls.get(K, 0.0) + units
        return self

    def add_put(self, K: float, units: float) -> "Portfolio":
        """Add ``units`` puts with strike K; a put with zero strike is worthless and is dropped."""
        if K != 0:
            self.puts[K] = self.puts.get(K, 0.0) + units
        return self

    def payoff(self, ST):
        """Value at T as a function of S_T (vectorized)."""
        ST = np.asarray(ST, dtype=float)
        v = self.bond + self.stock * ST
        for K, q in self.calls.items():
            v = v + q * np.maximum(ST - K, 0.0)
        for K, q in self.puts.items():
            v = v + q * np.maximum(K - ST, 0.0)
        return v

    def cost(self, S0: float, B: float, call_price, put_price=None) -> float:
        """Time-0 cost. ``call_price`` and ``put_price`` map a strike to a price
        (a dict or a function)."""
        get_c = call_price.__getitem__ if isinstance(call_price, dict) else call_price
        get_p = (put_price.__getitem__ if isinstance(put_price, dict) else put_price)
        v = self.bond * B + self.stock * S0
        v += sum(q * get_c(K) for K, q in self.calls.items())
        if self.puts:
            if get_p is None:
                raise ValueError("the portfolio holds puts: pass put prices")
            v += sum(q * get_p(K) for K, q in self.puts.items())
        return float(v)

    def kinks(self) -> np.ndarray:
        """Strikes at which the payoff can change slope (plus 0)."""
        return np.array(sorted({0.0, *self.calls, *self.puts}))

    def min_payoff(self) -> float:
        """Minimum of the payoff over S_T >= 0, or -inf if it is unbounded below.

        The payoff is piecewise linear, so it is enough to look at the kinks and at
        the slope beyond the largest one.
        """
        k = self.kinks()
        slope_at_infinity = self.stock + sum(self.calls.values())
        if slope_at_infinity < -1e-12:
            return -np.inf
        return float(np.min(self.payoff(k)))

    def describe(self, digits: int = 4) -> str:
        parts = []
        fmt = lambda q: f"{q:+.{digits}g}"
        if abs(self.bond) > 1e-14:
            parts.append(f"{fmt(self.bond)} bond")
        if abs(self.stock) > 1e-14:
            parts.append(f"{fmt(self.stock)} stock")
        for K in sorted(self.puts):
            if abs(self.puts[K]) > 1e-14:
                parts.append(f"{fmt(self.puts[K])} put({K:g})")
        for K in sorted(self.calls):
            if abs(self.calls[K]) > 1e-14:
                parts.append(f"{fmt(self.calls[K])} call({K:g})")
        return ", ".join(parts) if parts else "empty"


@dataclass
class Arbitrage:
    """A violated relation and a portfolio exploiting it.

    ``profit`` is the cash collected at time 0 (minus the cost of ``portfolio``); the
    payoff of ``portfolio`` at T is nonnegative for every value of S_T >= 0. For the
    condition ``"flat tail"`` the profit is 0 and the payoff is positive when S_T exceeds
    the lower strike: it is an arbitrage unless that event has probability zero
    (see ``note``).
    """
    condition: str
    strikes: tuple
    portfolio: Portfolio
    profit: float
    note: str = ""

    def __str__(self) -> str:
        ks = ", ".join(f"{K:g}" for K in self.strikes)
        s = (f"{self.condition} at K = {ks}: collect {self.profit:.4g} today; "
             f"hold {self.portfolio.describe()}")
        return s + (f" ({self.note})" if self.note else "")


# ---------------------------------------------------------------------------
# The arbitrage checker
# ---------------------------------------------------------------------------

def _prepare(strikes, prices):
    K = np.asarray(strikes, dtype=float)
    c = np.asarray(prices, dtype=float)
    if K.ndim != 1 or K.shape != c.shape:
        raise ValueError("strikes and prices must be 1-d arrays of the same length")
    if np.any(K <= 0) or np.any(np.diff(K) <= 0):
        raise ValueError("strikes must be positive and strictly increasing")
    return K, c


def find_arbitrages(S0: float, B: float, strikes, calls, puts=None, tol: float = 1e-9):
    """Check quoted calls (and puts) at one maturity against the relations of the notes.

    Parameters
    ----------
    S0, B : stock price and bond price at time 0 (both > 0).
    strikes : increasing positive strikes K_1 < ... < K_n.
    calls : call prices at those strikes.
    puts : optional put prices at the same strikes; they are checked through put--call parity.
    tol : violations smaller than ``tol`` are ignored.

    Returns a list of :class:`Arbitrage`, one per violated relation:

    * ``"positivity"``, ``"lower bound"``, ``"upper bound"``: (S0 - B K)^+ <= C(K) <= S0;
    * ``"monotonicity"``, ``"slope"``: -B <= (C(K_{i+1}) - C(K_i)) / (K_{i+1} - K_i) <= 0;
    * ``"convexity"``: consecutive triples, including the stock as the call with strike 0;
    * ``"parity"``: C(K) - P(K) = S0 - B K;
    * ``"flat tail"``: the last slope is 0 while the last price is positive.

    The checks on consecutive strikes are complete: when the list is empty, the quotes are
    reproduced by the state prices of :func:`state_prices`, so they are arbitrage-free prices
    in at least one model.
    """
    if S0 <= 0 or B <= 0:
        raise ValueError("S0 and B must be positive")
    K, c = _prepare(strikes, calls)
    n = len(K)
    out = []

    # Anchor: the stock is the call with strike 0.
    KK = np.concatenate([[0.0], K])
    cc = np.concatenate([[S0], c])

    # Bounds at each strike.
    for i in range(n):
        Ki, ci = K[i], c[i]
        lower = S0 - B * Ki
        if lower > 0 and ci < lower - tol:
            # buy the call, short the stock, lend B K: payoff (K - S_T)^+
            p = Portfolio(bond=Ki, stock=-1.0).add_call(Ki, 1.0)
            out.append(Arbitrage("lower bound", (Ki,), p, lower - ci))
        elif lower <= 0 and ci < -tol:
            out.append(Arbitrage("positivity", (Ki,), Portfolio().add_call(Ki, 1.0), -ci))
        if ci > S0 + tol:
            # sell the call, buy the stock: payoff min(S_T, K)
            p = Portfolio(stock=1.0).add_call(Ki, -1.0)
            out.append(Arbitrage("upper bound", (Ki,), p, ci - S0))

    # Consecutive pairs of quoted strikes: monotonicity and slope.
    for i in range(n - 1):
        K1, K2, c1, c2 = K[i], K[i + 1], c[i], c[i + 1]
        if c2 > c1 + tol:
            p = Portfolio().add_call(K1, 1.0).add_call(K2, -1.0)
            out.append(Arbitrage("monotonicity", (K1, K2), p, c2 - c1))
        if c1 - c2 > B * (K2 - K1) + tol:
            p = Portfolio(bond=K2 - K1).add_call(K1, -1.0).add_call(K2, 1.0)
            out.append(Arbitrage("slope", (K1, K2), p, c1 - c2 - B * (K2 - K1)))

    # Consecutive triples (the first one may involve the stock): convexity.
    for j in range(1, n):
        Ka, Kb, Kc = KK[j - 1], KK[j], KK[j + 1]
        lam = (Kc - Kb) / (Kc - Ka)
        excess = cc[j] - lam * cc[j - 1] - (1 - lam) * cc[j + 1]
        if excess > tol:
            p = Portfolio().add_call(Ka, lam).add_call(Kb, -1.0).add_call(Kc, 1 - lam)
            out.append(Arbitrage("convexity", (Ka, Kb, Kc), p, excess))

    # Flat tail: last slope zero, last price positive.
    s_last = (cc[-1] - cc[-2]) / (KK[-1] - KK[-2])
    if abs(s_last) <= tol and cc[-1] > tol:
        Ka, Kb = KK[-2], KK[-1]
        p = Portfolio().add_call(Ka, 1.0).add_call(Kb, -1.0)
        note = (f"payoff > 0 when S_T > {Ka:g}; if that event has probability zero, "
                f"selling the call with strike {Kb:g} is an arbitrage instead")
        out.append(Arbitrage("flat tail", (Ka, Kb), p, 0.0, note))

    # Put--call parity.
    if puts is not None:
        _, pp = _prepare(strikes, puts)
        for i in range(n):
            d = (c[i] - pp[i]) - (S0 - B * K[i])
            if d > tol:      # call rich: sell call, buy put, buy stock, borrow B K
                p = Portfolio(bond=-K[i], stock=1.0).add_call(K[i], -1.0).add_put(K[i], 1.0)
                out.append(Arbitrage("parity", (K[i],), p, d))
            elif d < -tol:   # call cheap: buy call, sell put, short stock, lend B K
                p = Portfolio(bond=K[i], stock=-1.0).add_call(K[i], 1.0).add_put(K[i], -1.0)
                out.append(Arbitrage("parity", (K[i],), p, -d))
    return out


def state_prices(S0: float, B: float, strikes, calls, tol: float = 1e-12):
    """State prices reproducing arbitrage-free call quotes (the sufficiency proposition).

    With K_0 = 0, C(K_0) = S0 and slopes s_i = (C(K_i) - C(K_{i-1})) / (K_i - K_{i-1}),
    the quotes must satisfy -B <= s_1 <= ... <= s_n <= 0, C(K_n) >= 0, and C(K_n) = 0 if
    s_n = 0. Returns ``(x, q)``: values of S_T and positive state prices with
    sum(q) = B, sum(q x) = S0 and sum(q (x - K_i)^+) = C(K_i) for every i.
    Raises ``ValueError`` if a condition fails.
    """
    K, c = _prepare(strikes, calls)
    KK = np.concatenate([[0.0], K])
    cc = np.concatenate([[S0], c])
    s = np.diff(cc) / np.diff(KK)
    if s[0] < -B - tol or np.any(np.diff(s) < -tol) or s[-1] > tol or cc[-1] < -tol:
        raise ValueError("the quotes violate a no-arbitrage relation: use find_arbitrages")
    if abs(s[-1]) <= tol and cc[-1] > tol:
        raise ValueError("flat tail with a positive price: no state prices exist")
    x = list(KK)                                   # atoms at 0, K_1, ..., K_n
    q = [B + s[0]] + list(np.diff(s)) + [0.0]       # mass at K_n is put in the tail atom
    if s[-1] < -tol:
        x.append(K[-1] + cc[-1] / (-s[-1]))
        q.append(-s[-1])
    x, q = np.array(x), np.array(q)
    keep = q > tol
    return x[keep], q[keep]


# ---------------------------------------------------------------------------
# Static replication (discrete Carr--Madan)
# ---------------------------------------------------------------------------

def replicate_payoff(f, strikes, kappa: float) -> Portfolio:
    """Static replication of the linear interpolation of f on a strike grid.

    Let f_h be the continuous piecewise-linear function equal to f at the strikes
    K_0 < ... < K_m (``kappa`` is added to the grid if missing) and extended linearly
    beyond K_0 and K_m. Then f_h(S_T) is replicated exactly by

        f_h(kappa) - s kappa bonds + s shares            (a forward at kappa, s = mean slope at kappa)
        + w_i puts at the strikes K_i < kappa,
        + w_i calls at the strikes K_i > kappa,
        + w_kappa / 2 puts and w_kappa / 2 calls at kappa,

    where w_i is the change of slope of f_h at K_i: a second difference, about f''(K_i) times
    the strike spacing. This is the Carr--Madan strip with the integral replaced by a sum.
    Returns the :class:`Portfolio`.
    """
    K = np.unique(np.concatenate([np.asarray(strikes, dtype=float), [float(kappa)]]))
    if K[0] <= 0 or len(K) < 3:
        raise ValueError("need at least three positive strikes")
    y = np.array([f(k) for k in K], dtype=float)
    slopes = np.diff(y) / np.diff(K)               # slope on [K_i, K_{i+1}]
    w = np.concatenate([[0.0], np.diff(slopes), [0.0]])   # slope jumps; none at the ends
    j = int(np.searchsorted(K, kappa))
    if j == 0:
        s = slopes[0]
    elif j == len(K) - 1:
        s = slopes[-1]
    else:
        s = 0.5 * (slopes[j - 1] + slopes[j])
    p = Portfolio(bond=y[j] - s * K[j], stock=s)
    for i, Ki in enumerate(K):
        if w[i] == 0.0:
            continue
        if i < j:
            p.add_put(Ki, w[i])
        elif i > j:
            p.add_call(Ki, w[i])
        else:
            p.add_put(Ki, 0.5 * w[i]).add_call(Ki, 0.5 * w[i])
    return p


def carr_madan(f, df, d2f, x, kappa: float, n_quad: int = 2001):
    """Right-hand side of the Carr--Madan identity at the points x (numerical quadrature).

    f(kappa) + f'(kappa)(x - kappa) + int_0^kappa f''(K)(K - x)^+ dK
    + int_kappa^infty f''(K)(x - K)^+ dK. Each integral runs over the interval between x
    and kappa only, and is computed with the composite Simpson rule. Used to check the
    identity in the tests and the notebook.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = f(kappa) + df(kappa) * (x - kappa)
    for k, xi in enumerate(x):
        a, b = sorted((xi, kappa))
        if b > a:
            t = np.linspace(a, b, n_quad)
            g = d2f(t) * np.abs(xi - t)        # (K - x)^+ below kappa, (x - K)^+ above
            h = t[1] - t[0]
            out[k] += h / 3 * (g[0] + g[-1] + 4 * g[1:-1:2].sum() + 2 * g[2:-1:2].sum())
    return out


# ---------------------------------------------------------------------------
# A reference arbitrage-free curve: lognormal S_T (Black--Scholes, Chapter 6)
# ---------------------------------------------------------------------------

def lognormal_call(S0: float, B: float, K, v: float):
    """C(K) = S0 N(d+) - B K N(d-), d+- = (log(F/K) +- v^2/2) / v, F = S0 / B.

    The price of a call when, under the pricing measure, log S_T is normal with mean
    log F - v^2/2 and variance v^2 (v = sigma sqrt(T) is the total volatility).
    """
    K = np.asarray(K, dtype=float)
    F = S0 / B
    with np.errstate(divide="ignore"):
        d1 = (np.log(F / K) + 0.5 * v**2) / v
    d2 = d1 - v
    return S0 * ndtr(d1) - B * K * ndtr(d2)


def lognormal_put(S0: float, B: float, K, v: float):
    """Put price in the same model, by put--call parity."""
    return lognormal_call(S0, B, K, v) - S0 + B * np.asarray(K, dtype=float)


def lognormal_density(x, S0: float, B: float, v: float):
    """Density of S_T under the pricing measure in the lognormal model."""
    x = np.asarray(x, dtype=float)
    F = S0 / B
    z = (np.log(x / F) + 0.5 * v**2) / v
    return np.exp(-0.5 * z**2) / (x * v * np.sqrt(2 * np.pi))
