"""Small numerical utilities shared by the models.

Contains the standard normal CDF (via ``math.erfc``, accurate to ~1e-15,
so no external special-function library is needed) and the exact
Ornstein-Uhlenbeck transition moments used by both the Vasicek and the
Hull-White Monte Carlo engines.
"""

from __future__ import annotations

import math
from typing import NamedTuple

_SQRT2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    """Standard normal CDF ``N(x) = 0.5 * erfc(-x / sqrt(2))``.

    ``erfc`` is used instead of ``erf`` for better relative accuracy in the
    far tails (large negative ``x``).
    """
    return 0.5 * math.erfc(-x / _SQRT2)


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
    """

    decay: float
    b: float
    var_x: float
    var_i: float
    cov: float


def ou_step_moments(a: float, sigma: float, dt: float) -> OUStepMoments:
    """Compute :class:`OUStepMoments` for mean-reversion ``a > 0``, ``dt > 0``."""
    if a <= 0.0:
        raise ValueError(f"mean reversion must be positive, got {a}")
    if dt <= 0.0:
        raise ValueError(f"time step must be positive, got {dt}")
    decay = math.exp(-a * dt)
    b = (1.0 - decay) / a
    s2 = sigma * sigma
    var_x = s2 * (1.0 - decay * decay) / (2.0 * a)
    var_i = (s2 / (a * a)) * (dt - 2.0 * b + (1.0 - decay * decay) / (2.0 * a))
    cov = (s2 / (2.0 * a * a)) * (1.0 - decay) ** 2
    # Guard tiny negative values from floating-point cancellation.
    var_x = max(var_x, 0.0)
    var_i = max(var_i, 0.0)
    return OUStepMoments(decay=decay, b=b, var_x=var_x, var_i=var_i, cov=cov)
