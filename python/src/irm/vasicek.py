"""Vasicek short-rate model ``dr = kappa (theta - r) dt + sigma dW``.

Gaussian model: rates can go negative, which is a *feature* here (EUR/JPY
style curves) and the reason no lognormal short-rate model is included in
this project.

Closed forms implemented (affine term structure):

* ``B(t,T) = (1 - e^{-kappa (T-t)}) / kappa``
* ``A(t,T) = exp[(theta - sigma^2/(2 kappa^2)) (B(t,T) - (T-t))
              - sigma^2 B(t,T)^2 / (4 kappa)]``
* ``P(t,T) = A(t,T) e^{-B(t,T) r(t)}``
* ``r_T | r_t ~ N(theta + (r_t - theta) e^{-kappa (T-t)},
                  sigma^2 (1 - e^{-2 kappa (T-t)}) / (2 kappa))``
* Jamshidian (1989) bond-option formula for European calls/puts on ZCBs.

Monte Carlo uses the *exact* OU transition of the pair ``(r, int r dt)``
(see :mod:`irm.mathutils`), so the ZCB estimator is unbiased at any step
count.  Calibration fits ``(kappa, theta, sigma)`` — or ``(kappa, theta)``
with ``sigma`` fixed from option prices — to market zero yields by least
squares with the native Nelder-Mead from :mod:`irm.optimize`, from several
starting points, and reports identifiability and boundary diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .mathutils import norm_cdf, ou_step_moments
from .optimize import nelder_mead

__all__ = [
    "Vasicek",
    "VasicekCalibration",
    "calibrate_vasicek",
    "VASICEK_DOMAIN",
    "CALIBRATION_STARTS",
]

#: Calibration domain ``(kappa_lo, kappa_hi, theta_abs_max, sigma_hi)``:
#: points outside get the penalty ``1e6 (1 + distance)``.
VASICEK_DOMAIN = (1e-6, 50.0, 5.0, 5.0)

#: Default multi-start kappa / sigma pairs; theta starts at the mean yield.
CALIBRATION_STARTS: Tuple[Tuple[float, float], ...] = ((0.2, 0.005), (0.5, 0.01), (1.5, 0.02))

#: Largest exponent passed to exp() before a ZCB price is declared not
#: representable (exp(709.78) is the double overflow point).
_MAX_EXP_ARG = 700.0

# at_bound thresholds (documented in API_SPEC section 7).
_KAPPA_LO_BOUND = 1e-5      # 10x the penalty floor 1e-6
_KAPPA_HI_BOUND = 49.95     # within 1e-3 of the range from 50
_THETA_BOUND = 4.995
_SIGMA_LO_BOUND = 1e-5
_SIGMA_HI_BOUND = 4.995
# identifiability: starts whose rmse is within 5 % of the best are
# "equivalent fits"; their parameter spreads must be below these to call the
# solution identified.
_EQUIV_RMSE_FACTOR = 1.05
_EQUIV_RMSE_FLOOR = 1e-12
_IDENT_KAPPA = 1e-2
_IDENT_THETA = 1e-3
_IDENT_SIGMA = 1e-3


class Vasicek:
    """Vasicek model with parameters ``kappa > 0``, ``sigma >= 0``.

    Parameters are validated once and exposed as read-only properties, so a
    model instance cannot be put into an invalid state after construction.
    """

    __slots__ = ("_kappa", "_theta", "_sigma", "_r0")

    def __init__(self, kappa: float, theta: float, sigma: float, r0: float) -> None:
        for name, v in (("kappa", kappa), ("theta", theta), ("sigma", sigma), ("r0", r0)):
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")
        if kappa <= 0.0:
            raise ValueError(f"kappa must be > 0, got {kappa}")
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self._kappa = float(kappa)
        self._theta = float(theta)
        self._sigma = float(sigma)
        self._r0 = float(r0)

    @property
    def kappa(self) -> float:
        """Mean-reversion speed (> 0)."""
        return self._kappa

    @property
    def theta(self) -> float:
        """Long-run mean level."""
        return self._theta

    @property
    def sigma(self) -> float:
        """Absolute (normal) volatility (>= 0)."""
        return self._sigma

    @property
    def r0(self) -> float:
        """Initial short rate."""
        return self._r0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Vasicek(kappa={self._kappa}, theta={self._theta}, "
            f"sigma={self._sigma}, r0={self._r0})"
        )

    # ----------------------------- affine functions ------------------- #

    def b_factor(self, tau: float) -> float:
        """``B(t,T)`` with ``tau = T - t``: duration of the ZCB w.r.t. r.

        ``B = (1 - e^{-kappa tau})/kappa`` (evaluated with ``expm1``); it
        is the sensitivity ``-d ln P / d r`` and appears in every
        option/moment formula.
        """
        if not math.isfinite(tau) or tau < 0.0:
            raise ValueError(f"tau must be finite and >= 0, got {tau}")
        return -math.expm1(-self._kappa * tau) / self._kappa

    def a_factor(self, tau: float) -> float:
        """``A(t,T)`` with ``tau = T - t`` (depends on tau only):

        ``A = exp[(theta - sigma^2/(2 kappa^2)) (B - tau)
                 - sigma^2 B^2/(4 kappa)]``.

        The first term is the risk-adjusted drift contribution of the mean
        level; the second is the convexity from rate volatility.
        """
        b = self.b_factor(tau)
        k, s = self._kappa, self._sigma
        return math.exp(
            (self._theta - s * s / (2.0 * k * k)) * (b - tau)
            - s * s * b * b / (4.0 * k)
        )

    # ----------------------------- ZCB / yields ----------------------- #

    def zcb_price(self, maturity: float, t: float = 0.0, r: Optional[float] = None) -> float:
        """Zero-coupon bond price ``P(t, maturity) = A e^{-B r(t)}``.

        ``r`` defaults to ``r0`` (only sensible for ``t = 0``).
        """
        if not (math.isfinite(maturity) and math.isfinite(t)):
            raise ValueError("maturity and t must be finite")
        if maturity < t:
            raise ValueError(f"maturity {maturity} before valuation time {t}")
        rr = self._r0 if r is None else float(r)
        if not math.isfinite(rr):
            raise ValueError(f"short rate must be finite, got {r}")
        tau = maturity - t
        expo = -self.b_factor(tau) * rr
        if expo > _MAX_EXP_ARG:
            raise ValueError(f"ZCB price not representable: exp({expo}) overflows")
        return self.a_factor(tau) * math.exp(expo)

    def zero_yield(self, maturity: float, t: float = 0.0, r: Optional[float] = None) -> float:
        """Continuously compounded zero yield ``-ln P(t,T) / (T - t)``."""
        tau = maturity - t
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError(f"need maturity > t, got tau={tau}")
        return -math.log(self.zcb_price(maturity, t, r)) / tau

    # ----------------------------- r_T distribution ------------------- #

    def r_mean(self, horizon: float, r: Optional[float] = None) -> float:
        """``E[r_T | r_0] = theta + (r_0 - theta) e^{-kappa T}``."""
        if not math.isfinite(horizon) or horizon < 0.0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        rr = self._r0 if r is None else float(r)
        if not math.isfinite(rr):
            raise ValueError(f"short rate must be finite, got {r}")
        return self._theta + (rr - self._theta) * math.exp(-self._kappa * horizon)

    def r_var(self, horizon: float) -> float:
        """``Var[r_T | r_0] = sigma^2 (1 - e^{-2 kappa T}) / (2 kappa)``."""
        if not math.isfinite(horizon) or horizon < 0.0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        return (
            self._sigma
            * self._sigma
            * (-math.expm1(-2.0 * self._kappa * horizon))
            / (2.0 * self._kappa)
        )

    # ----------------------------- bond options ----------------------- #

    def _sigma_p(self, expiry: float, bond_maturity: float) -> float:
        """Std dev of ln P(T,S) at option expiry T for bond maturity S:

        ``sigma_p = sigma sqrt((1 - e^{-2 kappa T})/(2 kappa)) B(T,S)``.
        """
        return (
            self._sigma
            * math.sqrt((-math.expm1(-2.0 * self._kappa * expiry)) / (2.0 * self._kappa))
            * self.b_factor(bond_maturity - expiry)
        )

    def _zb_option(self, expiry: float, bond_maturity: float, strike: float, call: bool) -> float:
        if not (math.isfinite(expiry) and math.isfinite(bond_maturity) and math.isfinite(strike)):
            raise ValueError("option inputs must be finite")
        if expiry < 0.0 or bond_maturity < expiry:
            raise ValueError(
                f"need 0 <= expiry <= bond maturity, got {expiry}, {bond_maturity}"
            )
        if strike <= 0.0:
            raise ValueError(f"strike must be > 0, got {strike}")
        p_s = self.zcb_price(bond_maturity)
        p_t = self.zcb_price(expiry)
        sp = self._sigma_p(expiry, bond_maturity)
        if sp == 0.0:
            # sigma = 0, expiry = 0, or S = T: option collapses to intrinsic
            # on the forward bond price.
            intrinsic = p_s - strike * p_t
            if not call:
                intrinsic = -intrinsic
            return max(intrinsic, 0.0)
        h = math.log(p_s / (p_t * strike)) / sp + 0.5 * sp
        if call:
            return p_s * norm_cdf(h) - strike * p_t * norm_cdf(h - sp)
        return strike * p_t * norm_cdf(sp - h) - p_s * norm_cdf(-h)

    def zbc(self, expiry: float, bond_maturity: float, strike: float) -> float:
        """European call on a unit ZCB (Jamshidian 1989 closed form).

        ``ZBC = P(0,S) N(h) - K P(0,T) N(h - sigma_p)`` with
        ``h = ln(P(0,S)/(K P(0,T)))/sigma_p + sigma_p/2``.
        """
        return self._zb_option(expiry, bond_maturity, strike, call=True)

    def zbp(self, expiry: float, bond_maturity: float, strike: float) -> float:
        """European put on a unit ZCB:
        ``ZBP = K P(0,T) N(sigma_p - h) - P(0,S) N(-h)``."""
        return self._zb_option(expiry, bond_maturity, strike, call=False)

    # ----------------------------- Monte Carlo ------------------------ #

    def simulate(
        self, horizon: float, n_steps: int, n_paths: int, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Exact-transition simulation of ``(r_T, int_0^T r dt)``.

        Writes ``r = theta + x`` with ``x`` a mean-zero OU process and
        samples ``(x_next, int x dt)`` per step from their exact joint
        Gaussian (conditional decomposition), so both outputs are free of
        discretisation bias.  Returns ``(r_T paths, integral paths)``.
        """
        if not math.isfinite(horizon) or horizon <= 0.0:
            raise ValueError(f"horizon must be finite and > 0, got {horizon}")
        if n_steps < 1 or n_paths < 1:
            raise ValueError("n_steps and n_paths must be >= 1")
        dt = horizon / n_steps
        m = ou_step_moments(self._kappa, self._sigma, dt)
        rng = np.random.default_rng(seed)
        x = np.full(n_paths, self._r0 - self._theta)
        integral_x = np.zeros(n_paths)
        if m.var_x > 0.0:
            sd_x = math.sqrt(m.var_x)
            beta = m.cov / m.var_x  # regression of the integral on x_next
            resid_var = max(m.var_i - m.cov * m.cov / m.var_x, 0.0)
            sd_resid = math.sqrt(resid_var)
            for _ in range(n_steps):
                z1 = rng.standard_normal(n_paths)
                z2 = rng.standard_normal(n_paths)
                mean_x = x * m.decay
                mean_i = x * m.b
                x_new = mean_x + sd_x * z1
                integral_x += mean_i + beta * (x_new - mean_x) + sd_resid * z2
                x = x_new
        else:  # sigma = 0: fully deterministic decay
            for _ in range(n_steps):
                integral_x += x * m.b
                x = x * m.decay
        r_end = self._theta + x
        integral_r = self._theta * horizon + integral_x
        return r_end, integral_r

    def mc_zcb(
        self, maturity: float, n_steps: int, n_paths: int, seed: int
    ) -> Tuple[float, float]:
        """MC ZCB price ``E[e^{-int r}]`` and its standard error.

        ``sigma = 0`` is fully deterministic and reports a standard error of
        exactly 0 (as does ``n_paths = 1``).
        """
        _, integral = self.simulate(maturity, n_steps, n_paths, seed)
        disc = np.exp(-integral)
        price = float(np.mean(disc))
        if n_paths > 1 and self._sigma > 0.0:
            se = float(np.std(disc, ddof=1) / math.sqrt(n_paths))
        else:
            se = 0.0
        return price, se


@dataclass(frozen=True)
class VasicekCalibration:
    """Least-squares calibration outcome (non-convergence reported, not raised).

    * ``converged``: the optimiser met its tolerances at the reported
      solution **and** the solution is not at a domain bound.
    * ``at_bound``: the solution sits at the edge of the calibration domain
      (``kappa < 1e-5`` or ``> 49.95``, ``|theta| > 4.995``,
      ``sigma < 1e-5`` or ``> 4.995``) — a penalty-boundary artefact, not a
      fit.
    * ``n_starts``: number of Nelder-Mead starts run (3 by default; 1 when
      ``x0`` is given).
    * ``kappa_spread`` / ``theta_spread`` / ``sigma_spread``: max minus min of
      each parameter over the starts whose rmse is within 5 % of the best
      (0 for a single start).
    * ``identified``: ``n_starts >= 2`` and all equivalent starts agree
      (``kappa_spread <= 1e-2``, ``theta_spread <= 1e-3``,
      ``sigma_spread <= 1e-3``).  With a single start identifiability is
      not assessed and this is ``False``.
    * ``sigma_fixed``: ``True`` when ``sigma`` was supplied, not fitted.

    Price on the result only if ``converged and (identified or sigma_fixed)``.
    """

    kappa: float
    theta: float
    sigma: float
    r0: float
    rmse: float
    iterations: int
    converged: bool
    at_bound: bool
    n_starts: int
    kappa_spread: float
    theta_spread: float
    sigma_spread: float
    identified: bool
    sigma_fixed: bool

    def model(self) -> Vasicek:
        return Vasicek(self.kappa, self.theta, self.sigma, self.r0)


def _at_bound(kappa: float, theta: float, sigma: float, sigma_is_fixed: bool) -> bool:
    if kappa < _KAPPA_LO_BOUND or kappa > _KAPPA_HI_BOUND:
        return True
    if abs(theta) > _THETA_BOUND:
        return True
    if not sigma_is_fixed and (sigma < _SIGMA_LO_BOUND or sigma > _SIGMA_HI_BOUND):
        return True
    return False


def calibrate_vasicek(
    maturities: Sequence[float],
    yields: Sequence[float],
    r0: float,
    x0: Optional[Sequence[float]] = None,
    maxiter: int = 4000,
    sigma_fixed: Optional[float] = None,
) -> VasicekCalibration:
    """Fit ``(kappa, theta, sigma)`` to market zero yields by least squares.

    Minimises ``sum_i (y_model(T_i) - y_i)^2`` with Nelder-Mead
    (``initial_step = 0.05``); out-of-domain points (``kappa <= 1e-6``,
    ``kappa > 50``, ``|theta| > 5``, ``sigma < 0``, ``sigma > 5``) get the
    smooth penalty ``1e6 (1 + distance)`` so the simplex is pushed back
    inside the domain.  ``r0`` is held fixed.

    * ``x0 = None`` (default): three starts ``(kappa, mean(y), sigma)`` for
      ``(kappa, sigma)`` in ``CALIBRATION_STARTS``; the lowest objective is
      returned together with the parameter spread across equivalent starts.
    * ``x0 = (kappa, theta, sigma)``: a single warm start from ``x0``
      (identifiability is then not assessed).
    * ``sigma_fixed``: fit only ``(kappa, theta)`` with ``sigma`` held at the
      given value (>= 0) — the practitioner workflow, since yields identify
      ``sigma`` only through convexity.  ``x0[2]`` is ignored in this mode.

    Yields identify the parameters weakly: the bundled ``zero_yields.csv``
    has two least-squares minima with rmse within 2 % of each other.  Always
    check ``converged``, ``at_bound`` and ``identified`` on the result.
    """
    ts = [float(t) for t in maturities]
    ys = [float(y) for y in yields]
    if len(ts) != len(ys) or len(ts) < 3:
        raise ValueError("need at least 3 (maturity, yield) pairs of equal length")
    if any((not math.isfinite(t)) or t <= 0.0 for t in ts):
        raise ValueError("maturities must be finite and > 0")
    if any(not math.isfinite(y) for y in ys):
        raise ValueError("yields must be finite")
    if not math.isfinite(r0):
        raise ValueError("r0 must be finite")
    if maxiter < 1:
        raise ValueError("maxiter must be >= 1")
    if x0 is not None:
        x0 = [float(v) for v in x0]
        if len(x0) != 3:
            raise ValueError("x0 must have 3 entries (kappa, theta, sigma)")
        if any(not math.isfinite(v) for v in x0):
            raise ValueError("x0 must be finite")
    if sigma_fixed is not None:
        sigma_fixed = float(sigma_fixed)
        if not math.isfinite(sigma_fixed) or sigma_fixed < 0.0:
            raise ValueError(f"sigma_fixed must be finite and >= 0, got {sigma_fixed}")
    k_lo, k_hi, th_max, s_hi = VASICEK_DOMAIN
    fixed = sigma_fixed is not None

    def penalty(kappa: float, theta: float, sigma: float) -> Optional[float]:
        if kappa <= k_lo or sigma < 0.0 or kappa > k_hi or abs(theta) > th_max or sigma > s_hi:
            return 1e6 * (
                1.0
                + max(0.0, k_lo - kappa)
                + max(0.0, -sigma)
                + max(0.0, kappa - k_hi)
                + max(0.0, abs(theta) - th_max)
                + max(0.0, sigma - s_hi)
            )
        return None

    def sse(kappa: float, theta: float, sigma: float) -> float:
        pen = penalty(kappa, theta, sigma)
        if pen is not None:
            return pen
        model = Vasicek(kappa, theta, sigma, r0)
        return sum((model.zero_yield(t) - y) ** 2 for t, y in zip(ts, ys))

    if fixed:
        def objective(p: Sequence[float]) -> float:
            return sse(p[0], p[1], sigma_fixed)  # type: ignore[arg-type]
    else:
        def objective(p: Sequence[float]) -> float:
            return sse(p[0], p[1], p[2])

    mean_y = sum(ys) / len(ys)
    if x0 is not None:
        starts: List[Tuple[float, ...]] = [tuple(x0)]
    else:
        starts = [(k, mean_y, s) for k, s in CALIBRATION_STARTS]

    results = []
    for st in starts:
        p0 = st[:2] if fixed else st
        res = nelder_mead(objective, p0, initial_step=0.05, maxiter=maxiter)
        kappa, theta = res.x[0], res.x[1]
        sigma = sigma_fixed if fixed else abs(res.x[2])
        results.append((res.fx, kappa, theta, sigma, res.iterations, res.converged))

    best = min(results, key=lambda r: r[0])
    fx, kappa, theta, sigma, iterations, nm_converged = best
    n = len(ts)
    rmse = math.sqrt(max(fx, 0.0) / n)
    at_bound = _at_bound(kappa, theta, sigma, fixed)  # type: ignore[arg-type]

    equiv = [
        r for r in results
        if math.sqrt(max(r[0], 0.0) / n) <= _EQUIV_RMSE_FACTOR * rmse + _EQUIV_RMSE_FLOOR
    ]
    kappa_spread = max(r[1] for r in equiv) - min(r[1] for r in equiv)
    theta_spread = max(r[2] for r in equiv) - min(r[2] for r in equiv)
    sigma_spread = max(r[3] for r in equiv) - min(r[3] for r in equiv)
    identified = (
        len(results) >= 2
        and kappa_spread <= _IDENT_KAPPA
        and theta_spread <= _IDENT_THETA
        and sigma_spread <= _IDENT_SIGMA
    )
    return VasicekCalibration(
        kappa=kappa,
        theta=theta,
        sigma=float(sigma),  # type: ignore[arg-type]
        r0=float(r0),
        rmse=rmse,
        iterations=iterations,
        converged=bool(nm_converged and not at_bound),
        at_bound=at_bound,
        n_starts=len(results),
        kappa_spread=kappa_spread,
        theta_spread=theta_spread,
        sigma_spread=sigma_spread,
        identified=identified,
        sigma_fixed=fixed,
    )
