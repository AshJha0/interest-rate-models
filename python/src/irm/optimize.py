"""Native Nelder-Mead simplex minimiser (no scipy), P02-style API.

Used by the Vasicek least-squares calibration.  Standard coefficients:
reflection 1, expansion 2, contraction 0.5, shrink 0.5.  Non-convergence is
*reported* via ``converged=False`` in the result, never raised — callers
decide how to react (per the shared conventions: report, don't crash).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

__all__ = ["NelderMeadResult", "nelder_mead"]


@dataclass(frozen=True)
class NelderMeadResult:
    """Outcome of a Nelder-Mead minimisation."""

    x: Tuple[float, ...]
    fx: float
    iterations: int
    converged: bool


def nelder_mead(
    f: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    initial_step: float = 0.1,
    xtol: float = 1e-10,
    ftol: float = 1e-14,
    maxiter: int = 5000,
) -> NelderMeadResult:
    """Minimise ``f`` starting from ``x0``.

    The initial simplex is ``x0`` plus ``initial_step`` added to each
    coordinate in turn.  Converged when both the simplex diameter is below
    ``xtol`` and the function-value spread below ``ftol``.
    """
    n = len(x0)
    if n == 0:
        raise ValueError("x0 must be non-empty")
    if not all(math.isfinite(v) for v in x0):
        raise ValueError("x0 must be finite")

    pts: List[List[float]] = [list(map(float, x0))]
    for i in range(n):
        p = list(map(float, x0))
        p[i] += initial_step
        pts.append(p)
    vals = [float(f(p)) for p in pts]

    def order() -> None:
        idx = sorted(range(n + 1), key=lambda k: vals[k])
        pts[:] = [pts[k] for k in idx]
        vals[:] = [vals[k] for k in idx]

    order()
    it = 0
    converged = False
    while it < maxiter:
        it += 1
        spread_f = abs(vals[-1] - vals[0])
        spread_x = max(
            max(abs(pts[j][i] - pts[0][i]) for i in range(n)) for j in range(1, n + 1)
        )
        if spread_f <= ftol and spread_x <= xtol:
            converged = True
            break

        centroid = [sum(pts[j][i] for j in range(n)) / n for i in range(n)]
        worst = pts[-1]
        refl = [centroid[i] + (centroid[i] - worst[i]) for i in range(n)]
        f_refl = float(f(refl))

        if f_refl < vals[0]:
            # Try expanding further along the reflection direction.
            exp_pt = [centroid[i] + 2.0 * (centroid[i] - worst[i]) for i in range(n)]
            f_exp = float(f(exp_pt))
            if f_exp < f_refl:
                pts[-1], vals[-1] = exp_pt, f_exp
            else:
                pts[-1], vals[-1] = refl, f_refl
        elif f_refl < vals[-2]:
            pts[-1], vals[-1] = refl, f_refl
        else:
            # Contract toward the centroid (outside if reflection helped a bit).
            if f_refl < vals[-1]:
                contr = [centroid[i] + 0.5 * (refl[i] - centroid[i]) for i in range(n)]
            else:
                contr = [centroid[i] + 0.5 * (worst[i] - centroid[i]) for i in range(n)]
            f_contr = float(f(contr))
            if f_contr < min(f_refl, vals[-1]):
                pts[-1], vals[-1] = contr, f_contr
            else:
                # Shrink the whole simplex toward the best vertex.
                for j in range(1, n + 1):
                    pts[j] = [
                        pts[0][i] + 0.5 * (pts[j][i] - pts[0][i]) for i in range(n)
                    ]
                    vals[j] = float(f(pts[j]))
        order()

    return NelderMeadResult(
        x=tuple(pts[0]), fx=vals[0], iterations=it, converged=converged
    )
