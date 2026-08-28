"""Hull-White one-factor model ``dr = (theta(t) - a r) dt + sigma dW``.

The model is fitted to an input :class:`~irm.curve.DiscountCurve` so that
model ZCB prices at ``t = 0`` reproduce the market discount factors
*exactly*.  All formulas use market DFs ``P^M(0, .)`` directly:

* ``B(t,T) = (1 - e^{-a (T-t)}) / a``
* ``A(t,T) = (P^M(0,T)/P^M(0,t))
             * exp[B(t,T) f^M(0,t)
                   - (sigma^2/(4a)) (1 - e^{-2 a t}) B(t,T)^2]``
* ``P(t,T) = A(t,T) e^{-B(t,T) r(t)}``

``f^M(0,t)`` is the market instantaneous forward, computed by the *exact*
finite-difference scheme documented in API_SPEC.md (step ``h = 1e-5``,
central for ``t >= h``, one-sided at the short end) so that every language
port reproduces bit-comparable values.

theta(t) fitting: matching the curve requires

``theta(t) = d f^M(0,t)/dt + a f^M(0,t) + (sigma^2/(2a)) (1 - e^{-2 a t})``

— the drift must (i) track the slope of the forward curve, (ii) offset the
pull ``-a r`` at the forward level, and (iii) add back the convexity
``sigma^2`` term.  Pricing never needs theta(t) explicitly (it is absorbed
into A via the market DFs); the function is provided for inspection and
testing.

Options: ZBP/ZBC closed forms, caplets/floorlets as ZCB puts/calls, caps as
caplet strips, and Jamshidian's decomposition for European swaptions.
Monte Carlo simulates ``r(t) = x(t) + alpha(t)`` with exact Gaussian OU
moments (trinomial-tree-free), ``alpha(t) = f^M(0,t)
+ (sigma^2/(2a^2)) (1 - e^{-a t})^2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .curve import DiscountCurve
from .curve import par_swap_rate as _curve_par_rate
from .mathutils import norm_cdf, ou_step_moments
from .rootfind import brentq

__all__ = ["HullWhite", "JamshidianResult"]

#: Finite-difference step for the market instantaneous forward f(0,t).
FD_STEP = 1e-5


@dataclass(frozen=True)
class JamshidianResult:
    """Jamshidian swaption decomposition output."""

    value: float
    r_star: float
    residual: float
    strikes: Tuple[float, ...]


class HullWhite:
    """Hull-White 1F model fitted to a market discount curve."""

    def __init__(self, a: float, sigma: float, curve: DiscountCurve) -> None:
        if not math.isfinite(a) or a <= 0.0:
            raise ValueError(f"mean reversion a must be finite and > 0, got {a}")
        if not math.isfinite(sigma) or sigma < 0.0:
            raise ValueError(f"sigma must be finite and >= 0, got {sigma}")
        if not isinstance(curve, DiscountCurve):
            raise ValueError("curve must be a DiscountCurve")
        self.a = float(a)
        self.sigma = float(sigma)
        self.curve = curve

    # ----------------------- market forward via FD -------------------- #

    def fwd0(self, t: float) -> float:
        """Market instantaneous forward ``f^M(0,t)`` by finite difference.

        Exact scheme (h = FD_STEP = 1e-5), reproduced identically by ports:

        * ``t >= h``:  central,  ``(ln DF(t-h) - ln DF(t+h)) / (2h)``
        * ``t <  h``:  one-sided, ``(ln DF(t) - ln DF(t+h)) / h``

        With piecewise-constant forwards the FD equals the segment forward
        away from pillars and the average of the two adjacent forwards when
        the stencil straddles a pillar.
        """
        if not math.isfinite(t) or t < 0.0:
            raise ValueError(f"time must be finite and >= 0, got {t}")
        h = FD_STEP
        if t >= h:
            return (math.log(self.curve.df(t - h)) - math.log(self.curve.df(t + h))) / (
                2.0 * h
            )
        return (math.log(self.curve.df(t)) - math.log(self.curve.df(t + h))) / h

    def theta(self, t: float, h: float = 1e-4) -> float:
        """Fitted drift ``theta(t)`` (documentation/testing helper).

        ``theta(t) = df/dt + a f + (sigma^2/(2a)) (1 - e^{-2 a t})`` with
        ``df/dt`` approximated by the forward difference
        ``(fwd0(t+h) - fwd0(t))/h``.
        """
        dfdt = (self.fwd0(t + h) - self.fwd0(t)) / h
        return dfdt + self.a * self.fwd0(t) + (
            self.sigma * self.sigma / (2.0 * self.a)
        ) * (1.0 - math.exp(-2.0 * self.a * t))

    # ----------------------- affine ZCB ------------------------------- #

    def b_factor(self, t: float, maturity: float) -> float:
        """``B(t,T) = (1 - e^{-a (T-t)}) / a``."""
        if maturity < t:
            raise ValueError(f"maturity {maturity} before t {t}")
        return (1.0 - math.exp(-self.a * (maturity - t))) / self.a

    def a_factor(self, t: float, maturity: float) -> float:
        """``A(t,T)`` built from market DFs and ``f^M(0,t)`` (see module doc)."""
        b = self.b_factor(t, maturity)
        pm_t = self.curve.df(t)
        pm_mat = self.curve.df(maturity)
        conv = (
            self.sigma
            * self.sigma
            / (4.0 * self.a)
            * (1.0 - math.exp(-2.0 * self.a * t))
            * b
            * b
        )
        return (pm_mat / pm_t) * math.exp(b * self.fwd0(t) - conv)

    def zcb_price(self, t: float, maturity: float, r: Optional[float] = None) -> float:
        """``P(t,T) = A(t,T) e^{-B(t,T) r}``.

        ``r`` defaults to ``r(0) = f^M(0,0)`` (so ``P(0,T)`` equals the
        market DF exactly: the ``e^{B f}`` and ``e^{-B r}`` terms cancel).
        """
        rr = self.fwd0(0.0) if r is None else float(r)
        if not math.isfinite(rr):
            raise ValueError(f"short rate must be finite, got {r}")
        return self.a_factor(t, maturity) * math.exp(-self.b_factor(t, maturity) * rr)

    # ----------------------- ZCB options ------------------------------ #

    def _sigma_p(self, expiry: float, bond_maturity: float) -> float:
        """Std dev of ln P at expiry:
        ``sigma_p = sigma sqrt((1 - e^{-2 a T})/(2a)) B(T,S)``."""
        return (
            self.sigma
            * math.sqrt((1.0 - math.exp(-2.0 * self.a * expiry)) / (2.0 * self.a))
            * self.b_factor(expiry, bond_maturity)
        )

    def _zb_option(
        self, expiry: float, bond_maturity: float, strike: float, call: bool
    ) -> float:
        if not (
            math.isfinite(expiry) and math.isfinite(bond_maturity) and math.isfinite(strike)
        ):
            raise ValueError("option inputs must be finite")
        if expiry < 0.0 or bond_maturity < expiry:
            raise ValueError(
                f"need 0 <= expiry <= bond maturity, got {expiry}, {bond_maturity}"
            )
        if strike <= 0.0:
            raise ValueError(f"strike must be > 0, got {strike}")
        p_s = self.curve.df(bond_maturity)
        p_t = self.curve.df(expiry)
        sp = self._sigma_p(expiry, bond_maturity)
        if sp == 0.0:
            intrinsic = p_s - strike * p_t
            if not call:
                intrinsic = -intrinsic
            return max(intrinsic, 0.0)
        h = math.log(p_s / (p_t * strike)) / sp + 0.5 * sp
        if call:
            return p_s * norm_cdf(h) - strike * p_t * norm_cdf(h - sp)
        return strike * p_t * norm_cdf(sp - h) - p_s * norm_cdf(-h)

    def zbc(self, expiry: float, bond_maturity: float, strike: float) -> float:
        """European call on a unit ZCB (uses market DFs — model-consistent)."""
        return self._zb_option(expiry, bond_maturity, strike, call=True)

    def zbp(self, expiry: float, bond_maturity: float, strike: float) -> float:
        """European put on a unit ZCB."""
        return self._zb_option(expiry, bond_maturity, strike, call=False)

    # ----------------------- caplets / caps --------------------------- #

    def caplet(
        self, reset: float, pay: float, strike: float, notional: float = 1.0
    ) -> float:
        """Caplet on the simple forward over ``[reset, pay]``.

        Pays ``notional * tau * max(F(reset,pay) - K, 0)`` at ``pay``.  A
        caplet is ``(1 + K tau)`` ZCB *puts* struck at ``1/(1 + K tau)``:

        ``caplet = notional (1 + K tau) ZBP(reset, pay, 1/(1 + K tau))``.

        With ``sigma = 0`` this degenerates to discounted intrinsic on the
        forward, ``notional * DF(pay) * tau * max(F - K, 0)``.
        """
        tau = pay - reset
        if tau <= 0.0 or reset < 0.0:
            raise ValueError(f"need 0 <= reset < pay, got {reset}, {pay}")
        if not math.isfinite(strike) or 1.0 + strike * tau <= 0.0:
            raise ValueError(
                f"caplet strike implies non-positive 1 + K*tau ({strike})"
            )
        k_bond = 1.0 / (1.0 + strike * tau)
        return notional * (1.0 + strike * tau) * self.zbp(reset, pay, k_bond)

    def floorlet(
        self, reset: float, pay: float, strike: float, notional: float = 1.0
    ) -> float:
        """Floorlet: ``notional (1 + K tau) ZBC(reset, pay, 1/(1 + K tau))``."""
        tau = pay - reset
        if tau <= 0.0 or reset < 0.0:
            raise ValueError(f"need 0 <= reset < pay, got {reset}, {pay}")
        if not math.isfinite(strike) or 1.0 + strike * tau <= 0.0:
            raise ValueError(
                f"floorlet strike implies non-positive 1 + K*tau ({strike})"
            )
        k_bond = 1.0 / (1.0 + strike * tau)
        return notional * (1.0 + strike * tau) * self.zbc(reset, pay, k_bond)

    def cap(
        self, schedule: Sequence[float], strike: float, notional: float = 1.0
    ) -> float:
        """Cap = strip of caplets over consecutive schedule times.

        ``schedule = [t0, t1, ..., tn]`` prices caplets on
        ``[t0,t1], ..., [t_{n-1}, t_n]`` (first reset ``t0 >= 0``).
        """
        ts = [float(t) for t in schedule]
        if len(ts) < 2:
            raise ValueError("cap schedule needs at least two times")
        total = 0.0
        for t_reset, t_pay in zip(ts[:-1], ts[1:]):
            total += self.caplet(t_reset, t_pay, strike, notional)
        return total

    def par_swap_rate(self, times: Sequence[float]) -> float:
        """Par swap rate helper off the model's market curve."""
        return _curve_par_rate(self.curve, times)

    # ----------------------- Jamshidian swaption ---------------------- #

    def jamshidian_swaption(
        self,
        expiry: float,
        pay_times: Sequence[float],
        fixed_rate: float,
        notional: float = 1.0,
        payer: bool = True,
    ) -> JamshidianResult:
        """European swaption via Jamshidian's decomposition.

        The underlying swap exchanges the fixed rate for the (single-curve)
        floating leg over ``[expiry, pay_times]``; accruals are
        ``tau_1 = t_1 - expiry``, ``tau_i = t_i - t_{i-1}``.  A payer
        swaption's payoff at expiry is ``max(1 - sum_i c_i P(T, t_i), 0)``
        with coupons ``c_i = X tau_i`` and ``c_n = 1 + X tau_n`` — a put
        with strike 1 on a coupon bond.

        Because P(T, t; r) is strictly decreasing in r (B > 0), the coupon
        bond price is monotone in r, so there is a unique ``r*`` with
        ``sum_i c_i P(T, t_i; r*) = 1`` (found with the native Brent solver
        on an expanding bracket, |residual| < 1e-12 required).  Setting
        ``K_i = P(T, t_i; r*)``, the coupon-bond option splits *exactly*
        into ZCB options that are all exercised on the same event {r_T > r*}:

        * payer    = ``sum_i c_i ZBP(T, t_i, K_i)``
        * receiver = ``sum_i c_i ZBC(T, t_i, K_i)``
        """
        ts = [float(t) for t in pay_times]
        if not math.isfinite(expiry) or expiry <= 0.0:
            raise ValueError(f"swaption expiry must be > 0, got {expiry}")
        if len(ts) < 1:
            raise ValueError("swaption needs at least one payment time")
        if not math.isfinite(fixed_rate):
            raise ValueError("fixed rate must be finite")
        prev = expiry
        taus = []
        for t in ts:
            if t <= prev:
                raise ValueError(
                    "payment times must be strictly increasing and after expiry"
                )
            taus.append(t - prev)
            prev = t
        coupons = [fixed_rate * tau for tau in taus]
        coupons[-1] += 1.0

        def bond_minus_one(r: float) -> float:
            return sum(
                c * self.zcb_price(expiry, t, r) for c, t in zip(coupons, ts)
            ) - 1.0

        # Expanding bracket: f is strictly decreasing in r.
        lo, hi = -1.0, 1.0
        for _ in range(24):
            if bond_minus_one(lo) > 0.0 and bond_minus_one(hi) < 0.0:
                break
            if bond_minus_one(lo) <= 0.0:
                lo *= 2.0
            if bond_minus_one(hi) >= 0.0:
                hi *= 2.0
        else:
            raise ValueError(
                "jamshidian: could not bracket r* (degenerate coupon bond — "
                "check fixed rate/schedule)"
            )
        r_star = brentq(bond_minus_one, lo, hi, xtol=1e-15)
        residual = bond_minus_one(r_star)
        strikes = tuple(self.zcb_price(expiry, t, r_star) for t in ts)
        if payer:
            value = sum(
                c * self.zbp(expiry, t, k) for c, t, k in zip(coupons, ts, strikes)
            )
        else:
            value = sum(
                c * self.zbc(expiry, t, k) for c, t, k in zip(coupons, ts, strikes)
            )
        return JamshidianResult(
            value=notional * value,
            r_star=r_star,
            residual=residual,
            strikes=strikes,
        )

    # ----------------------- Monte Carlo ------------------------------ #

    def alpha(self, t: float) -> float:
        """Deterministic shift ``alpha(t) = f^M(0,t)
        + (sigma^2/(2 a^2)) (1 - e^{-a t})^2`` with ``r(t) = x(t) + alpha(t)``."""
        return self.fwd0(t) + (self.sigma**2 / (2.0 * self.a**2)) * (
            1.0 - math.exp(-self.a * t)
        ) ** 2

    def _variance_integral(self, t: float) -> float:
        """``V(t) = int_0^t (sigma^2/(2a^2))(1 - e^{-a s})^2 ds`` so that
        ``int_0^t alpha ds = -ln P^M(0,t) + V(t)``; closed form:

        ``V(t) = (sigma^2/(2a^2)) [t - 2(1-e^{-a t})/a + (1-e^{-2 a t})/(2a)]``.
        """
        a = self.a
        return (self.sigma**2 / (2.0 * a * a)) * (
            t
            - 2.0 * (1.0 - math.exp(-a * t)) / a
            + (1.0 - math.exp(-2.0 * a * t)) / (2.0 * a)
        )

    def mc_caplet(
        self,
        reset: float,
        pay: float,
        strike: float,
        notional: float,
        n_steps: int,
        n_paths: int,
        seed: int,
    ) -> Tuple[float, float]:
        """Unbiased MC caplet price and standard error.

        Simulates ``x`` to the reset date with exact OU joint moments of
        ``(x, int x dt)``; the pathwise discount factor is exact:

        ``D(0,T) = P^M(0,T) exp(-V(T) - int_0^T x dt)``.

        At reset, ``P(T, pay)`` is computed analytically from
        ``r(T) = x(T) + alpha(T)``; the payoff (paid at ``pay``, known at
        ``T``) is valued as ``D(0,T) P(T,pay) * notional * tau * max(F-K,0)``.
        No discretisation bias at any step count (sigma = 0 gives the
        deterministic intrinsic).
        """
        tau = pay - reset
        if tau <= 0.0 or reset <= 0.0:
            raise ValueError(f"need 0 < reset < pay, got {reset}, {pay}")
        if n_steps < 1 or n_paths < 1:
            raise ValueError("n_steps and n_paths must be >= 1")
        dt = reset / n_steps
        m = ou_step_moments(self.a, self.sigma, dt)
        rng = np.random.default_rng(seed)
        x = np.zeros(n_paths)
        integral_x = np.zeros(n_paths)
        if m.var_x > 0.0:
            sd_x = math.sqrt(m.var_x)
            beta = m.cov / m.var_x
            sd_resid = math.sqrt(max(m.var_i - m.cov * m.cov / m.var_x, 0.0))
            for _ in range(n_steps):
                z1 = rng.standard_normal(n_paths)
                z2 = rng.standard_normal(n_paths)
                mean_x = x * m.decay
                mean_i = x * m.b
                x_new = mean_x + sd_x * z1
                integral_x += mean_i + beta * (x_new - mean_x) + sd_resid * z2
                x = x_new
        # sigma = 0: x stays 0, integral stays 0 (deterministic curve replay).
        disc = self.curve.df(reset) * np.exp(-self._variance_integral(reset) - integral_x)
        r_reset = x + self.alpha(reset)
        b = self.b_factor(reset, pay)
        a_f = self.a_factor(reset, pay)
        p_reset_pay = a_f * np.exp(-b * r_reset)
        fwd = (1.0 / p_reset_pay - 1.0) / tau
        payoff = notional * tau * np.maximum(fwd - strike, 0.0)
        pv = disc * p_reset_pay * payoff
        price = float(np.mean(pv))
        se = float(np.std(pv, ddof=1) / math.sqrt(n_paths)) if n_paths > 1 else 0.0
        return price, se
