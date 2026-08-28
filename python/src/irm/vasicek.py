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
count.  Calibration fits ``(kappa, theta, sigma)`` to market zero yields by
least squares with the native Nelder-Mead from :mod:`irm.optimize`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .mathutils import norm_cdf, ou_step_moments
from .optimize import nelder_mead

__all__ = ["Vasicek", "VasicekCalibration", "calibrate_vasicek"]


class Vasicek:
    """Vasicek model with parameters ``kappa > 0``, ``sigma >= 0``."""

    def __init__(self, kappa: float, theta: float, sigma: float, r0: float) -> None:
        for name, v in (("kappa", kappa), ("theta", theta), ("sigma", sigma), ("r0", r0)):
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")
        if kappa <= 0.0:
            raise ValueError(f"kappa must be > 0, got {kappa}")
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.r0 = float(r0)

    # ----------------------------- affine functions ------------------- #

    def b_factor(self, tau: float) -> float:
        """``B(t,T)`` with ``tau = T - t``: duration of the ZCB w.r.t. r.

        ``B = (1 - e^{-kappa tau})/kappa``; it is the sensitivity
        ``-d ln P / d r`` and appears in every option/moment formula.
        """
        if not math.isfinite(tau) or tau < 0.0:
            raise ValueError(f"tau must be finite and >= 0, got {tau}")
        return (1.0 - math.exp(-self.kappa * tau)) / self.kappa

    def a_factor(self, tau: float) -> float:
        """``A(t,T)`` with ``tau = T - t`` (depends on tau only):

        ``A = exp[(theta - sigma^2/(2 kappa^2)) (B - tau)
                 - sigma^2 B^2/(4 kappa)]``.

        The first term is the risk-adjusted drift contribution of the mean
        level; the second is the convexity from rate volatility.
        """
        b = self.b_factor(tau)
        k, s = self.kappa, self.sigma
        return math.exp(
            (self.theta - s * s / (2.0 * k * k)) * (b - tau)
            - s * s * b * b / (4.0 * k)
        )

    # ----------------------------- ZCB / yields ----------------------- #

    def zcb_price(self, maturity: float, t: float = 0.0, r: Optional[float] = None) -> float:
        """Zero-coupon bond price ``P(t, maturity) = A e^{-B r(t)}``.

        ``r`` defaults to ``r0`` (only sensible for ``t = 0``).
        """
        if maturity < t:
            raise ValueError(f"maturity {maturity} before valuation time {t}")
        rr = self.r0 if r is None else float(r)
        tau = maturity - t
        return self.a_factor(tau) * math.exp(-self.b_factor(tau) * rr)

    def zero_yield(self, maturity: float, t: float = 0.0, r: Optional[float] = None) -> float:
        """Continuously compounded zero yield ``-ln P(t,T) / (T - t)``."""
        tau = maturity - t
        if tau <= 0.0:
            raise ValueError(f"need maturity > t, got tau={tau}")
        return -math.log(self.zcb_price(maturity, t, r)) / tau

    # ----------------------------- r_T distribution ------------------- #

    def r_mean(self, horizon: float, r: Optional[float] = None) -> float:
        """``E[r_T | r_0] = theta + (r_0 - theta) e^{-kappa T}``."""
        if not math.isfinite(horizon) or horizon < 0.0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        rr = self.r0 if r is None else float(r)
        return self.theta + (rr - self.theta) * math.exp(-self.kappa * horizon)

    def r_var(self, horizon: float) -> float:
        """``Var[r_T | r_0] = sigma^2 (1 - e^{-2 kappa T}) / (2 kappa)``."""
        if not math.isfinite(horizon) or horizon < 0.0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        return (
            self.sigma
            * self.sigma
            * (1.0 - math.exp(-2.0 * self.kappa * horizon))
            / (2.0 * self.kappa)
        )

    # ----------------------------- bond options ----------------------- #

    def _sigma_p(self, expiry: float, bond_maturity: float) -> float:
        """Std dev of ln P(T,S) at option expiry T for bond maturity S:

        ``sigma_p = sigma sqrt((1 - e^{-2 kappa T})/(2 kappa)) B(T,S)``.
        """
        return (
            self.sigma
            * math.sqrt((1.0 - math.exp(-2.0 * self.kappa * expiry)) / (2.0 * self.kappa))
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
        if horizon <= 0.0 or not math.isfinite(horizon):
            raise ValueError(f"horizon must be finite and > 0, got {horizon}")
        if n_steps < 1 or n_paths < 1:
            raise ValueError("n_steps and n_paths must be >= 1")
        dt = horizon / n_steps
        m = ou_step_moments(self.kappa, self.sigma, dt)
        rng = np.random.default_rng(seed)
        x = np.full(n_paths, self.r0 - self.theta)
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
        r_end = self.theta + x
        integral_r = self.theta * horizon + integral_x
        return r_end, integral_r

    def mc_zcb(
        self, maturity: float, n_steps: int, n_paths: int, seed: int
    ) -> Tuple[float, float]:
        """MC ZCB price ``E[e^{-int r}]`` and its standard error."""
        _, integral = self.simulate(maturity, n_steps, n_paths, seed)
        disc = np.exp(-integral)
        price = float(np.mean(disc))
        se = float(np.std(disc, ddof=1) / math.sqrt(n_paths)) if n_paths > 1 else 0.0
        return price, se


@dataclass(frozen=True)
class VasicekCalibration:
    """Least-squares calibration outcome (non-convergence reported, not raised)."""

    kappa: float
    theta: float
    sigma: float
    r0: float
    rmse: float
    iterations: int
    converged: bool

    def model(self) -> Vasicek:
        return Vasicek(self.kappa, self.theta, self.sigma, self.r0)


def calibrate_vasicek(
    maturities: Sequence[float],
    yields: Sequence[float],
    r0: float,
    x0: Optional[Sequence[float]] = None,
    maxiter: int = 4000,
) -> VasicekCalibration:
    """Fit ``(kappa, theta, sigma)`` to market zero yields by least squares.

    Minimises ``sum_i (y_model(T_i) - y_i)^2`` with Nelder-Mead; invalid
    regions (kappa <= 0, sigma < 0) are handled with a smooth penalty so the
    simplex is pushed back inside the domain.  ``r0`` is held fixed.
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

    def objective(p: Sequence[float]) -> float:
        kappa, theta, sigma = p
        if kappa <= 1e-6 or sigma < 0.0 or kappa > 50.0 or abs(theta) > 5.0 or sigma > 5.0:
            # Smooth penalty pointing back toward the feasible region.
            return 1e6 * (
                1.0
                + max(0.0, 1e-6 - kappa)
                + max(0.0, -sigma)
                + max(0.0, kappa - 50.0)
                + max(0.0, abs(theta) - 5.0)
                + max(0.0, sigma - 5.0)
            )
        model = Vasicek(kappa, theta, sigma, r0)
        return sum((model.zero_yield(t) - y) ** 2 for t, y in zip(ts, ys))

    start = tuple(x0) if x0 is not None else (0.5, sum(ys) / len(ys), 0.01)
    res = nelder_mead(objective, start, initial_step=0.05, maxiter=maxiter)
    kappa, theta, sigma = res.x
    rmse = math.sqrt(max(res.fx, 0.0) / len(ts))
    return VasicekCalibration(
        kappa=kappa,
        theta=theta,
        sigma=abs(sigma),
        r0=float(r0),
        rmse=rmse,
        iterations=res.iterations,
        converged=res.converged,
    )
