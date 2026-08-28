"""Native 1-D root finders: Brent's method and plain bisection.

Implemented from scratch (no ``scipy``) per project rules; ``scipy`` is only
allowed in the test suite as a cross-check.  Brent's method combines
bisection, secant and inverse quadratic interpolation and is guaranteed to
converge for a continuous function with a sign change on the bracket, with
superlinear speed on well-behaved functions.
"""

from __future__ import annotations

import math
from typing import Callable

__all__ = ["brentq", "bisect"]

# Machine-epsilon-scale relative tolerance (2 * eps, as in the classic zbrent).
_RTOL = 4.44e-16


def brentq(
    f: Callable[[float], float],
    a: float,
    b: float,
    xtol: float = 1e-14,
    rtol: float = _RTOL,
    maxiter: int = 100,
) -> float:
    """Find a root of ``f`` in ``[a, b]`` using Brent's method.

    ``f(a)`` and ``f(b)`` must have opposite signs (or one endpoint be an
    exact root); otherwise a ``ValueError`` is raised.  Convergence tolerance
    on the abscissa is ``2*rtol*|x| + xtol/2`` (Numerical-Recipes zbrent
    convention).
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("brentq: bracket endpoints must be finite")
    if xtol <= 0.0:
        raise ValueError("brentq: xtol must be positive")
    fa = f(a)
    fb = f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if (fa > 0.0) == (fb > 0.0):
        raise ValueError(
            f"brentq: root not bracketed: f({a})={fa}, f({b})={fb}"
        )
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxiter):
        if (fb > 0.0) == (fc > 0.0):
            # Rename so that the root lies between b and c.
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * rtol * abs(b) + 0.5 * xtol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            # Attempt inverse quadratic interpolation (secant if a == c).
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d  # accept interpolation
                d = p / q
            else:
                d = xm  # interpolation failed; bisect
                e = d
        else:
            d = xm  # bounds decreasing too slowly; bisect
            e = d
        a, fa = b, fb
        if abs(d) > tol1:
            b += d
        else:
            b += tol1 if xm > 0.0 else -tol1
        fb = f(b)
    raise ValueError(f"brentq: no convergence after {maxiter} iterations")


def bisect(
    f: Callable[[float], float],
    a: float,
    b: float,
    xtol: float = 1e-12,
    maxiter: int = 200,
) -> float:
    """Plain bisection on ``[a, b]``; robust fallback, linear convergence."""
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("bisect: bracket endpoints must be finite")
    fa = f(a)
    fb = f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if (fa > 0.0) == (fb > 0.0):
        raise ValueError(
            f"bisect: root not bracketed: f({a})={fa}, f({b})={fb}"
        )
    lo, hi = (a, b) if a < b else (b, a)
    flo = fa if a < b else fb
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0 or 0.5 * (hi - lo) < xtol:
            return mid
        if (fm > 0.0) == (flo > 0.0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)
