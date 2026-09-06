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

``theta(t) = d f^M(0,t)/dt + a f^M(0,t) + (sigma^2/(2a)) (1 - e^{-2a t})``

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

Calibration: :func:`calibrate_hullwhite` fits ``(a, sigma)`` (or ``sigma``
with ``a`` fixed) to caplet / payer-swaption prices by least squares with
the native Nelder-Mead from several starts; :mod:`irm.bachelier` converts
prices to and from normal (bp) implied volatilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .curve import DiscountCurve
from .curve import par_swap_rate as _curve_par_rate
from .mathutils import norm_cdf, ou_integral_variance, ou_step_moments
from .optimize import nelder_mead
from .rootfind import brentq

__all__ = [
    "HullWhite",
    "JamshidianResult",
    "CapletQuote",
    "SwaptionQuote",
    "HullWhiteCalibration",
    "calibrate_hullwhite",
    "HW_DOMAIN",
    "HW_CALIBRATION_STARTS",
    "JAMSHIDIAN_RESIDUAL_TOL",
]

#: Finite-difference step for the market instantaneous forward f(0,t).
FD_STEP = 1e-5
#: The coupon-bond residual ``|g(r*)|`` must be below this after the root find.
JAMSHIDIAN_RESIDUAL_TOL = 1e-10
#: Maximum bracket doublings when searching for r*.
_JAMSHIDIAN_MAX_DOUBLINGS = 24
#: Largest exponent passed to exp() before the result is declared not
#: representable (exp(709.78) is the double overflow point).
_MAX_EXP_ARG = 700.0

#: Calibration domain ``(a_lo, a_hi, sigma_hi)`` (sigma_lo = 0); outside gets
#: the penalty ``1e6 (1 + distance)``.
HW_DOMAIN = (1e-6, 5.0, 1.0)
#: Default multi-start ``(a, sigma)`` pairs (sigma-only starts use the sigmas).
HW_CALIBRATION_STARTS: Tuple[Tuple[float, float], ...] = (
    (0.03, 0.005), (0.1, 0.01), (0.5, 0.02),
)
_A_LO_BOUND = 1e-5
_A_HI_BOUND = 4.995
_SIGMA_LO_BOUND = 1e-5
_SIGMA_HI_BOUND = 0.999
_EQUIV_RMSE_FACTOR = 1.05
_EQUIV_RMSE_FLOOR = 1e-14
_IDENT_A = 1e-2
_IDENT_SIGMA = 1e-4
#: sigma is optimised as sigma * _SIGMA_SCALE so one Nelder-Mead step fits both.
_SIGMA_SCALE = 10.0


@dataclass(frozen=True)
class JamshidianResult:
    """Jamshidian swaption decomposition output."""

    value: float
    r_star: float
    residual: float
    strikes: Tuple[float, ...]


def _check_finite(**kwargs: float) -> None:
    for name, v in kwargs.items():
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v}")


class HullWhite:
    """Hull-White 1F model fitted to a market discount curve.

    ``a``, ``sigma`` and the curve are validated once and read-only.
    """

    __slots__ = ("_a", "_sigma", "_curve")

    def __init__(self, a: float, sigma: float, curve: DiscountCurve) -> None:
        if not math.isfinite(a) or a <= 0.0:
            raise ValueError(f"mean reversion a must be finite and > 0, got {a}")
        if not math.isfinite(sigma) or sigma < 0.0:
            raise ValueError(f"sigma must be finite and >= 0, got {sigma}")
        if not isinstance(curve, DiscountCurve):
            raise ValueError("curve must be a DiscountCurve")
        self._a = float(a)
        self._sigma = float(sigma)
        self._curve = curve

    @property
    def a(self) -> float:
        """Mean-reversion speed (> 0)."""
        return self._a

    @property
    def sigma(self) -> float:
        """Absolute (normal) short-rate volatility (>= 0)."""
        return self._sigma

    @property
    def curve(self) -> DiscountCurve:
        """The market discount curve the model is fitted to."""
        return self._curve

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"HullWhite(a={self._a}, sigma={self._sigma}, curve={self._curve!r})"

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
            return (math.log(self._curve.df(t - h)) - math.log(self._curve.df(t + h))) / (
                2.0 * h
            )
        return (math.log(self._curve.df(t)) - math.log(self._curve.df(t + h))) / h

    def theta(self, t: float, h: float = 1e-4) -> float:
        """Fitted drift ``theta(t)`` (documentation/testing helper).

        ``theta(t) = df/dt + a f + (sigma^2/(2a)) (1 - e^{-2 a t})`` with
        ``df/dt`` approximated by the forward difference
        ``(fwd0(t+h) - fwd0(t))/h``.
        """
        if not math.isfinite(h) or h <= 0.0:
            raise ValueError(f"theta step h must be finite and > 0, got {h}")
        dfdt = (self.fwd0(t + h) - self.fwd0(t)) / h
        return dfdt + self._a * self.fwd0(t) + (
            self._sigma * self._sigma / (2.0 * self._a)
        ) * (-math.expm1(-2.0 * self._a * t))

    # ----------------------- affine ZCB ------------------------------- #

    def b_factor(self, t: float, maturity: float) -> float:
        """``B(t,T) = (1 - e^{-a (T-t)}) / a`` (``expm1`` form), ``0 <= t <= T``."""
        _check_finite(t=t, maturity=maturity)
        if t < 0.0 or maturity < t:
            raise ValueError(f"need 0 <= t <= maturity, got t={t}, maturity={maturity}")
        return -math.expm1(-self._a * (maturity - t)) / self._a

    def a_factor(self, t: float, maturity: float) -> float:
        """``A(t,T)`` built from market DFs and ``f^M(0,t)`` (see module doc)."""
        b = self.b_factor(t, maturity)
        pm_t = self._curve.df(t)
        pm_mat = self._curve.df(maturity)
        conv = (
            self._sigma
            * self._sigma
            / (4.0 * self._a)
            * (-math.expm1(-2.0 * self._a * t))
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
        expo = -self.b_factor(t, maturity) * rr
        if expo > _MAX_EXP_ARG:
            raise ValueError(f"ZCB price not representable: exp({expo}) overflows")
        return self.a_factor(t, maturity) * math.exp(expo)

    # ----------------------- ZCB options ------------------------------ #

    def _sigma_p(self, expiry: float, bond_maturity: float) -> float:
        """Std dev of ln P at expiry:
        ``sigma_p = sigma sqrt((1 - e^{-2 a T})/(2a)) B(T,S)``."""
        return (
            self._sigma
            * math.sqrt((-math.expm1(-2.0 * self._a * expiry)) / (2.0 * self._a))
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
        p_s = self._curve.df(bond_maturity)
        p_t = self._curve.df(expiry)
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

    @staticmethod
    def _check_caplet_inputs(
        what: str, reset: float, pay: float, strike: float, notional: float
    ) -> float:
        """Validate and return ``tau``; shared by caplet/floorlet/mc_caplet."""
        _check_finite(reset=reset, pay=pay, strike=strike, notional=notional)
        tau = pay - reset
        if tau <= 0.0 or reset < 0.0:
            raise ValueError(f"{what}: need 0 <= reset < pay, got {reset}, {pay}")
        if 1.0 + strike * tau <= 0.0:
            raise ValueError(f"{what}: strike implies non-positive 1 + K*tau ({strike})")
        return tau

    def caplet(
        self, reset: float, pay: float, strike: float, notional: float = 1.0
    ) -> float:
        """Caplet on the simple forward over ``[reset, pay]``.

        Pays ``notional * tau * max(F(reset,pay) - K, 0)`` at ``pay``.  A
        caplet is ``(1 + K tau)`` ZCB *puts* struck at ``1/(1 + K tau)``:

        ``caplet = notional (1 + K tau) ZBP(reset, pay, 1/(1 + K tau))``.

        With ``sigma = 0`` this degenerates to discounted intrinsic on the
        forward, ``notional * DF(pay) * tau * max(F - K, 0)``.  All inputs
        must be finite (``notional`` may be negative: a short position).
        """
        tau = self._check_caplet_inputs("caplet", reset, pay, strike, notional)
        k_bond = 1.0 / (1.0 + strike * tau)
        return notional * (1.0 + strike * tau) * self.zbp(reset, pay, k_bond)

    def floorlet(
        self, reset: float, pay: float, strike: float, notional: float = 1.0
    ) -> float:
        """Floorlet: ``notional (1 + K tau) ZBC(reset, pay, 1/(1 + K tau))``."""
        tau = self._check_caplet_inputs("floorlet", reset, pay, strike, notional)
        k_bond = 1.0 / (1.0 + strike * tau)
        return notional * (1.0 + strike * tau) * self.zbc(reset, pay, k_bond)

    def cap(
        self, schedule: Sequence[float], strike: float, notional: float = 1.0
    ) -> float:
        """Cap = strip of caplets over consecutive schedule times.

        ``schedule = [t0, t1, ..., tn]`` (>= 2 finite, strictly increasing
        times, ``t0 >= 0``) prices caplets on ``[t0,t1], ..., [t_{n-1}, t_n]``.
        """
        ts = [float(t) for t in schedule]
        if len(ts) < 2:
            raise ValueError("cap schedule needs at least two times")
        if any(not math.isfinite(t) for t in ts):
            raise ValueError("cap schedule times must be finite")
        if ts[0] < 0.0 or any(b <= a for a, b in zip(ts[:-1], ts[1:])):
            raise ValueError("cap schedule must be strictly increasing with t0 >= 0")
        _check_finite(strike=strike, notional=notional)
        total = 0.0
        for t_reset, t_pay in zip(ts[:-1], ts[1:]):
            total += self.caplet(t_reset, t_pay, strike, notional)
        return total

    def par_swap_rate(self, times: Sequence[float]) -> float:
        """Par swap rate helper off the model's market curve."""
        return _curve_par_rate(self._curve, times)

    # ----------------------- Jamshidian swaption ---------------------- #

    def _swaption_coupons(
        self, expiry: float, pay_times: Sequence[float], fixed_rate: float, notional: float
    ) -> Tuple[List[float], List[float]]:
        ts = [float(t) for t in pay_times]
        if not math.isfinite(expiry) or expiry <= 0.0:
            raise ValueError(f"swaption expiry must be finite and > 0, got {expiry}")
        if len(ts) < 1:
            raise ValueError("swaption needs at least one payment time")
        _check_finite(fixed_rate=fixed_rate, notional=notional)
        prev = expiry
        taus = []
        for t in ts:
            if not math.isfinite(t) or t <= prev:
                raise ValueError(
                    "payment times must be finite, strictly increasing and after expiry"
                )
            taus.append(t - prev)
            prev = t
        coupons = [fixed_rate * tau for tau in taus]
        coupons[-1] += 1.0
        if coupons[-1] <= 0.0:
            raise ValueError(
                "jamshidian: degenerate coupon bond — 1 + X*tau_n must be > 0 "
                f"(got {coupons[-1]}); no r* exists"
            )
        return ts, coupons

    def jamshidian_at_r_star(
        self,
        expiry: float,
        pay_times: Sequence[float],
        fixed_rate: float,
        r_star: float,
        notional: float = 1.0,
        payer: bool = True,
    ) -> JamshidianResult:
        """Jamshidian decomposition for a *given* critical rate ``r*``.

        Enforces the contract of :meth:`jamshidian_swaption` step 2: the
        coupon-bond residual ``g(r*) = sum_i c_i P(T,u_i;r*) - 1`` must
        satisfy ``|g(r*)| < 1e-10`` and the slope ``g'(r*) = -sum_i c_i B_i
        K_i`` must be negative (the crossing is on the decreasing branch of
        ``g``); otherwise ``ValueError``.  Exposed so the contract is
        testable and so payer/receiver can share one root find.
        """
        ts, coupons = self._swaption_coupons(expiry, pay_times, fixed_rate, notional)
        if not math.isfinite(r_star):
            raise ValueError(f"jamshidian: r* must be finite, got {r_star}")
        strikes = tuple(self.zcb_price(expiry, t, r_star) for t in ts)
        residual = sum(c * k for c, k in zip(coupons, strikes)) - 1.0
        if not (abs(residual) < JAMSHIDIAN_RESIDUAL_TOL):
            raise ValueError(
                f"jamshidian: r* residual {residual} exceeds {JAMSHIDIAN_RESIDUAL_TOL}"
            )
        slope = -sum(c * self.b_factor(expiry, t) * k for c, t, k in zip(coupons, ts, strikes))
        if not (slope < 0.0):
            raise ValueError(
                f"jamshidian: coupon bond not decreasing at r* (slope {slope}); "
                "decomposition invalid"
            )
        if payer:
            value = sum(c * self.zbp(expiry, t, k) for c, t, k in zip(coupons, ts, strikes))
        else:
            value = sum(c * self.zbc(expiry, t, k) for c, t, k in zip(coupons, ts, strikes))
        return JamshidianResult(
            value=notional * value, r_star=r_star, residual=residual, strikes=strikes
        )

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
        with strike 1 on a coupon bond ``g(r) = sum_i c_i P(T, t_i; r)``.

        Exactness needs ``g(r) - 1`` to change sign exactly once.  For
        ``X >= 0`` every coupon is positive and ``g`` is strictly decreasing
        (each ``B > 0``).  For ``X < 0`` the coupon signs are
        ``(-, ..., -, +)`` when ``1 + X tau_n > 0``: ``g'`` is an exponential
        sum with one sign change, so ``g`` has a single minimum, tends to
        ``+inf`` as ``r -> -inf`` and to ``0^-`` as ``r -> +inf``, and still
        crosses 1 exactly once — the decomposition remains exact (verified
        against direct integration in the tests).  If ``1 + X tau_n <= 0``
        every coupon is non-positive, ``g < 1`` everywhere and the bracket
        search fails with ``ValueError``.

        ``r*`` is found with the native Brent solver on an expanding bracket
        (start ``[-1, 1]``, double the failing side, at most 24 doublings),
        ``xtol = 1e-15``; ``|g(r*)| < 1e-10`` and ``g'(r*) < 0`` are then
        enforced by :meth:`jamshidian_at_r_star`.  Setting
        ``K_i = P(T, t_i; r*)``, the coupon-bond option splits *exactly*
        into ZCB options that are all exercised on the same event {r_T > r*}:

        * payer    = ``sum_i c_i ZBP(T, t_i, K_i)``
        * receiver = ``sum_i c_i ZBC(T, t_i, K_i)``
        """
        ts, coupons = self._swaption_coupons(expiry, pay_times, fixed_rate, notional)

        def bond_minus_one(r: float) -> float:
            return sum(
                c * self.zcb_price(expiry, t, r) for c, t in zip(coupons, ts)
            ) - 1.0

        # Expanding bracket: g(lo) > 0 > g(hi) is needed.  g(lo) grows like
        # exp(B_n |lo|); refuse to evaluate past the exp() overflow point.
        b_last = self.b_factor(expiry, ts[-1])
        lo, hi = -1.0, 1.0
        bracketed = False
        for _ in range(_JAMSHIDIAN_MAX_DOUBLINGS):
            if b_last * (-lo) > _MAX_EXP_ARG:
                break
            g_lo, g_hi = bond_minus_one(lo), bond_minus_one(hi)
            if g_lo > 0.0 and g_hi < 0.0:
                bracketed = True
                break
            if g_lo <= 0.0:
                lo *= 2.0
            if g_hi >= 0.0:
                hi *= 2.0
        if not bracketed:
            raise ValueError(
                "jamshidian: could not bracket r* (degenerate coupon bond — "
                "check fixed rate/schedule)"
            )
        r_star = brentq(bond_minus_one, lo, hi, xtol=1e-15)
        return self.jamshidian_at_r_star(expiry, ts, fixed_rate, r_star, notional, payer)

    # ----------------------- Monte Carlo ------------------------------ #

    def alpha(self, t: float) -> float:
        """Deterministic shift ``alpha(t) = f^M(0,t)
        + (sigma^2/(2 a^2)) (1 - e^{-a t})^2`` with ``r(t) = x(t) + alpha(t)``."""
        one_m = -math.expm1(-self._a * t)
        return self.fwd0(t) + (self._sigma**2 / (2.0 * self._a**2)) * one_m * one_m

    def _variance_integral(self, t: float) -> float:
        """``V(t) = int_0^t (sigma^2/(2a^2))(1 - e^{-a s})^2 ds`` so that
        ``int_0^t alpha ds = -ln P^M(0,t) + V(t)``; closed form:

        ``V(t) = (sigma^2/(2a^2)) [t - 2(1-e^{-a t})/a + (1-e^{-2 a t})/(2a)]``

        which is half the integrated-OU variance kernel
        (:func:`irm.mathutils.ou_integral_variance`), evaluated without
        cancellation.  ``V(0) = 0``.
        """
        if t == 0.0:
            return 0.0
        return 0.5 * ou_integral_variance(self._a, self._sigma, t)

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
        deterministic intrinsic and a standard error of exactly 0).  Same
        input validation as :meth:`caplet` plus ``reset > 0``.
        """
        tau = self._check_caplet_inputs("mc_caplet", reset, pay, strike, notional)
        if reset <= 0.0:
            raise ValueError(f"mc_caplet: need 0 < reset < pay, got {reset}, {pay}")
        if n_steps < 1 or n_paths < 1:
            raise ValueError("n_steps and n_paths must be >= 1")
        dt = reset / n_steps
        m = ou_step_moments(self._a, self._sigma, dt)
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
        disc = self._curve.df(reset) * np.exp(-self._variance_integral(reset) - integral_x)
        r_reset = x + self.alpha(reset)
        b = self.b_factor(reset, pay)
        a_f = self.a_factor(reset, pay)
        p_reset_pay = a_f * np.exp(-b * r_reset)
        fwd = (1.0 / p_reset_pay - 1.0) / tau
        payoff = notional * tau * np.maximum(fwd - strike, 0.0)
        pv = disc * p_reset_pay * payoff
        price = float(np.mean(pv))
        if n_paths > 1 and self._sigma > 0.0:
            se = float(np.std(pv, ddof=1) / math.sqrt(n_paths))
        else:
            se = 0.0
        return price, se


# --------------------------------------------------------------------- #
# Calibration of (a, sigma) to caplet / swaption prices
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class CapletQuote:
    """Market caplet price per unit notional on ``[reset, pay]`` at ``strike``."""

    reset: float
    pay: float
    strike: float
    price: float

    def __post_init__(self) -> None:
        _check_finite(reset=self.reset, pay=self.pay, strike=self.strike, price=self.price)
        if self.reset < 0.0 or self.pay <= self.reset:
            raise ValueError(f"caplet quote needs 0 <= reset < pay, got {self.reset}, {self.pay}")
        if 1.0 + self.strike * (self.pay - self.reset) <= 0.0:
            raise ValueError("caplet quote strike implies non-positive 1 + K*tau")
        if self.price < 0.0:
            raise ValueError(f"caplet quote price must be >= 0, got {self.price}")


@dataclass(frozen=True)
class SwaptionQuote:
    """Market *payer* swaption price per unit notional."""

    expiry: float
    pay_times: Tuple[float, ...]
    fixed_rate: float
    price: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pay_times", tuple(float(t) for t in self.pay_times))
        _check_finite(expiry=self.expiry, fixed_rate=self.fixed_rate, price=self.price)
        if self.expiry <= 0.0:
            raise ValueError(f"swaption quote expiry must be > 0, got {self.expiry}")
        if len(self.pay_times) < 1:
            raise ValueError("swaption quote needs at least one payment time")
        prev = self.expiry
        for t in self.pay_times:
            if not math.isfinite(t) or t <= prev:
                raise ValueError(
                    "swaption quote payment times must be finite, strictly increasing "
                    "and after expiry"
                )
            prev = t
        if self.price < 0.0:
            raise ValueError(f"swaption quote price must be >= 0, got {self.price}")


@dataclass(frozen=True)
class HullWhiteCalibration:
    """Outcome of :func:`calibrate_hullwhite` (same diagnostics as Vasicek).

    ``rmse`` is in price units per unit notional.  ``converged`` means the
    optimiser converged at the best start *and* the solution is not at a
    domain bound (``a < 1e-5`` or ``> 4.995``, ``sigma < 1e-5`` or
    ``> 0.999``).  ``identified`` requires ``n_starts >= 2`` and
    ``a_spread <= 1e-2``, ``sigma_spread <= 1e-4`` over the starts whose
    rmse is within 5 % of the best.  ``a_fixed`` is ``True`` when ``a`` was
    supplied, not fitted.
    """

    a: float
    sigma: float
    rmse: float
    iterations: int
    converged: bool
    at_bound: bool
    n_starts: int
    a_spread: float
    sigma_spread: float
    identified: bool
    a_fixed: bool

    def model(self, curve: DiscountCurve) -> HullWhite:
        return HullWhite(self.a, self.sigma, curve)


def calibrate_hullwhite(
    curve: DiscountCurve,
    caplets: Sequence[CapletQuote] = (),
    swaptions: Sequence[SwaptionQuote] = (),
    a_fixed: Optional[float] = None,
    x0: Optional[Sequence[float]] = None,
    maxiter: int = 4000,
) -> HullWhiteCalibration:
    """Fit ``(a, sigma)`` to caplet and payer-swaption prices by least squares.

    Minimises ``sum_j (model_price_j - market_price_j)^2`` over all quotes
    (prices per unit notional) with Nelder-Mead on ``(a, 10 sigma)``
    (``initial_step = 0.05``, i.e. 0.005 in ``sigma``); out-of-domain points (``a <= 1e-6``,
    ``a > 5``, ``sigma < 0``, ``sigma > 1``) get the penalty
    ``1e6 (1 + distance)``.

    * ``x0 = None``: three starts from ``HW_CALIBRATION_STARTS``, best
      objective returned with spreads over equivalent starts.
    * ``x0 = (a, sigma)``: single warm start.
    * ``a_fixed``: fit ``sigma`` only (1-D), starts at the ``sigma`` values
      of ``HW_CALIBRATION_STARTS``; ``x0[0]`` is ignored.

    At least one quote is required; with both parameters free at least two
    quotes of different expiry/tenor are needed for identifiability (a
    single quote fits ``sigma`` for any ``a`` — the result will report
    ``identified = False``).  Non-convergence is reported, never raised.
    Quotes are trusted as given: convert broker normal vols with
    :func:`irm.bachelier.bachelier_price` first.
    """
    if not isinstance(curve, DiscountCurve):
        raise ValueError("curve must be a DiscountCurve")
    caps = list(caplets)
    swps = list(swaptions)
    if not caps and not swps:
        raise ValueError("calibrate_hullwhite needs at least one caplet or swaption quote")
    for q in caps:
        if not isinstance(q, CapletQuote):
            raise ValueError("caplets must be CapletQuote instances")
    for q in swps:
        if not isinstance(q, SwaptionQuote):
            raise ValueError("swaptions must be SwaptionQuote instances")
    if maxiter < 1:
        raise ValueError("maxiter must be >= 1")
    if x0 is not None:
        x0 = [float(v) for v in x0]
        if len(x0) != 2:
            raise ValueError("x0 must have 2 entries (a, sigma)")
        if any(not math.isfinite(v) for v in x0):
            raise ValueError("x0 must be finite")
    if a_fixed is not None:
        a_fixed = float(a_fixed)
        if not math.isfinite(a_fixed) or a_fixed <= 0.0:
            raise ValueError(f"a_fixed must be finite and > 0, got {a_fixed}")
    a_lo, a_hi, s_hi = HW_DOMAIN
    fixed = a_fixed is not None
    n_quotes = len(caps) + len(swps)

    def sse(a: float, sigma: float) -> float:
        if a <= a_lo or a > a_hi or sigma < 0.0 or sigma > s_hi:
            return 1e6 * (
                1.0
                + max(0.0, a_lo - a)
                + max(0.0, a - a_hi)
                + max(0.0, -sigma)
                + max(0.0, sigma - s_hi)
            )
        hw = HullWhite(a, sigma, curve)
        total = 0.0
        for q in caps:
            d = hw.caplet(q.reset, q.pay, q.strike, 1.0) - q.price
            total += d * d
        for q in swps:
            d = hw.jamshidian_swaption(q.expiry, q.pay_times, q.fixed_rate, 1.0, True).value
            d -= q.price
            total += d * d
        return total

    # Nelder-Mead uses one initial_step (0.05) for every coordinate, so the
    # optimiser works on sigma * SIGMA_SCALE (step 0.005 in sigma units).
    if fixed:
        def objective(p: Sequence[float]) -> float:
            return sse(a_fixed, p[0] / _SIGMA_SCALE)  # type: ignore[arg-type]
        starts: List[Tuple[float, ...]] = (
            [(x0[1] * _SIGMA_SCALE,)] if x0 is not None
            else [(s * _SIGMA_SCALE,) for _, s in HW_CALIBRATION_STARTS]
        )
    else:
        def objective(p: Sequence[float]) -> float:
            return sse(p[0], p[1] / _SIGMA_SCALE)
        starts = (
            [(x0[0], x0[1] * _SIGMA_SCALE)] if x0 is not None
            else [(a, s * _SIGMA_SCALE) for a, s in HW_CALIBRATION_STARTS]
        )

    results = []
    for st in starts:
        res = nelder_mead(objective, st, initial_step=0.05, maxiter=maxiter)
        if fixed:
            a_val, s_val = a_fixed, abs(res.x[0]) / _SIGMA_SCALE
        else:
            a_val, s_val = res.x[0], abs(res.x[1]) / _SIGMA_SCALE
        results.append((res.fx, a_val, s_val, res.iterations, res.converged))

    best = min(results, key=lambda r: r[0])
    fx, a_val, s_val, iterations, nm_converged = best
    rmse = math.sqrt(max(fx, 0.0) / n_quotes)
    at_bound = (
        (not fixed and (a_val < _A_LO_BOUND or a_val > _A_HI_BOUND))
        or s_val < _SIGMA_LO_BOUND
        or s_val > _SIGMA_HI_BOUND
    )
    equiv = [
        r for r in results
        if math.sqrt(max(r[0], 0.0) / n_quotes) <= _EQUIV_RMSE_FACTOR * rmse + _EQUIV_RMSE_FLOOR
    ]
    a_spread = max(r[1] for r in equiv) - min(r[1] for r in equiv)
    sigma_spread = max(r[2] for r in equiv) - min(r[2] for r in equiv)
    identified = len(results) >= 2 and a_spread <= _IDENT_A and sigma_spread <= _IDENT_SIGMA
    return HullWhiteCalibration(
        a=float(a_val),  # type: ignore[arg-type]
        sigma=float(s_val),
        rmse=rmse,
        iterations=iterations,
        converged=bool(nm_converged and not at_bound),
        at_bound=bool(at_bound),
        n_starts=len(results),
        a_spread=a_spread,
        sigma_spread=sigma_spread,
        identified=identified,
        a_fixed=fixed,
    )
