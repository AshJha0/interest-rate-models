"""Small numerical utilities shared by the models.

Contains the standard normal CDF (via ``math.erfc``, accurate to ~1e-15,
so no external special-function library is needed), the exact
Ornstein-Uhlenbeck transition moments used by both the Vasicek and the
Hull-White Monte Carlo engines, and the cancellation-free integrated OU
variance kernel shared with the Hull-White pathwise discount factor.
"""

from __future__ import annotations

import math
from typing import NamedTuple

_SQRT2 = math.sqrt(2.0)

#: Below this value of ``x = a * dt`` the integrated-OU variance uses its
#: power series instead of the closed form (see :func:`ou_integral_variance`).
OU_SERIES_THRESHOLD = 1e-2

# Coefficients of  x - 2(1 - e^{-x}) + (1 - e^{-2x})/2  =  x^3 * sum_k c_k x^k,
# c_k = (-1)^(k+3) (2 - 2^(k+2)) / (k+3)!  for k = 0, 1, ...
_OU_SERIES = (
    1.0 / 3.0,
    -1.0 / 4.0,
    7.0 / 60.0,
    -1.0 / 24.0,
    31.0 / 2520.0,
    -1.0 / 320.0,
)


def norm_cdf(x: float) -> float:
    """Standard normal CDF ``N(x) = 0.5 * erfc(-x / sqrt(2))``.

    ``erfc`` is used instead of ``erf`` for better relative accuracy in the
    far tails (large negative ``x``).
    """
    return 0.5 * math.erfc(-x / _SQRT2)


def norm_pdf(x: float) -> float:
    """Standard normal density ``phi(x) = exp(-x^2/2) / sqrt(2 pi)``."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def ou_integral_variance(a: float, sigma: float, dt: float) -> float:
    """``Var[int_0^dt x(s) ds | x(0)]`` for ``dx = -a x dt + sigma dW``.

    Closed form ``(sigma^2/a^2) (dt - 2 b + (1 - e^{-2 a dt})/(2a))`` with
    ``b = (1 - e^{-a dt})/a``.  The three O(dt) terms cancel to an
    O(a^2 dt^3) result, which loses all precision for small ``a * dt``
    (80x too large at ``a*dt = 1e-4`` in plain double arithmetic).  With
    ``x = a * dt`` this therefore evaluates

    * ``x < OU_SERIES_THRESHOLD``: the series
      ``sigma^2 dt^3 (1/3 - x/4 + 7x^2/60 - x^3/24 + 31x^4/2520 - x^5/320)``
      (relative truncation error < 3e-15 at the threshold);
    * otherwise the closed form written with ``expm1``
      (relative rounding error below ~1e-11 for ``x >= 1e-2``).

    This is also ``2 V(t)`` of the Hull-White model with ``dt = t``.
    """
    if not (a > 0.0) or not math.isfinite(a):
        raise ValueError(f"mean reversion must be positive and finite, got {a}")
    if not (dt > 0.0) or not math.isfinite(dt):
        raise ValueError(f"time step must be positive and finite, got {dt}")
    if not math.isfinite(sigma):
        raise ValueError(f"sigma must be finite, got {sigma}")
    x = a * dt
    s2 = sigma * sigma
    if x < OU_SERIES_THRESHOLD:
        poly = 0.0
        for c in reversed(_OU_SERIES):
            poly = poly * x + c
        return s2 * dt * dt * dt * poly
    bracket = dt + 2.0 * math.expm1(-x) / a + (-math.expm1(-2.0 * x)) / (2.0 * a)
    return max((s2 / (a * a)) * bracket, 0.0)


class OUStepMoments(NamedTuple):
    """Exact one-step transition moments of a mean-zero OU process.

    For ``dx = -a x dt + sigma dW`` and a step of length ``dt``, conditional
    on the current state ``x``:

    * ``x_next  ~ N(x * decay, var_x)`` with ``decay = exp(-a dt)``
    * ``I = int_t^{t+dt} x(s) ds ~ N(x * b, var_i)``
    * ``Cov(x_next, I) = cov``

    where (writing ``b = (1 - decay)/a``):

    * ``var_x = sigma^2 (1 - decay^2) / (2a)``
    * ``var_i = (sigma^2/a^2) (dt - 2 b + (1 - decay^2)/(2a))``
    * ``cov   = (sigma^2/(2 a^2)) (1 - decay)^2``

    These are the classic closed-form OU moments; sampling the pair
    ``(x_next, I)`` from this bivariate Gaussian gives an *exact* (bias-free)
    discretisation of both the short rate and its time integral, which is
    what makes the MC zero-coupon-bond estimators unbiased at any step size.
    ``1 - decay`` terms are evaluated with ``expm1`` and ``var_i`` through
    :func:`ou_integral_variance`, so the moments stay accurate down to
    ``a * dt ~ 1e-12`` (plain evaluation is wrong by orders of magnitude
    below ``a * dt ~ 1e-3``).
    """

    decay: float
    b: float
    var_x: float
    var_i: float
    cov: float


def ou_step_moments(a: float, sigma: float, dt: float) -> OUStepMoments:
    """Compute :class:`OUStepMoments` for mean-reversion ``a > 0``, ``dt > 0``."""
    if not (a > 0.0) or not math.isfinite(a):
        raise ValueError(f"mean reversion must be positive and finite, got {a}")
    if not (dt > 0.0) or not math.isfinite(dt):
        raise ValueError(f"time step must be positive and finite, got {dt}")
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"sigma must be finite and >= 0, got {sigma}")
    x = a * dt
    decay = math.exp(-x)
    one_minus_decay = -math.expm1(-x)
    b = one_minus_decay / a
    s2 = sigma * sigma
    var_x = s2 * (-math.expm1(-2.0 * x)) / (2.0 * a)
    var_i = ou_integral_variance(a, sigma, dt)
    cov = (s2 / (2.0 * a * a)) * one_minus_decay * one_minus_decay
    return OUStepMoments(decay=decay, b=b, var_x=var_x, var_i=var_i, cov=cov)
