"""Discount curve with log-linear interpolation in ln(DF).

Conventions (see API_SPEC.md):

* Day count is ACT/365F everywhere; all times are year fractions
  ``days/365`` (a documented simplification versus the ACT/360 money-market
  convention used for real deposits/FRAs).
* Rates are decimals (0.05 = 5%).
* Interpolation is *linear in ln(DF)* between pillars, with an implicit
  node ``(t=0, DF=1)``.  Linear-in-ln(DF) is exactly the piecewise-constant
  instantaneous-forward interpolation:  on a segment
  ``[t_i, t_{i+1}]`` the instantaneous forward is the constant
  ``f_i = -(ln DF_{i+1} - ln DF_i) / (t_{i+1} - t_i)``.
* Extrapolation beyond the last pillar is flat-forward: the last segment's
  constant forward is carried on indefinitely.
* Negative rates are fully supported: DF > 1 is legal (EUR/JPY style),
  the only requirement is DF > 0.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import Sequence

__all__ = ["DiscountCurve", "par_swap_rate"]


class DiscountCurve:
    """Immutable discount curve defined by pillar times and discount factors.

    Parameters
    ----------
    times:
        Strictly increasing pillar times in years (ACT/365F), all > 0.
    dfs:
        Discount factors at the pillars, all finite and > 0.  Values > 1
        are allowed (negative rates).
    """

    __slots__ = ("times", "dfs", "_t", "_lnp")

    def __init__(self, times: Sequence[float], dfs: Sequence[float]) -> None:
        times = tuple(float(t) for t in times)
        dfs = tuple(float(p) for p in dfs)
        if len(times) == 0:
            raise ValueError("curve needs at least one pillar")
        if len(times) != len(dfs):
            raise ValueError(
                f"times and dfs length mismatch: {len(times)} vs {len(dfs)}"
            )
        prev = 0.0
        for t in times:
            if not math.isfinite(t):
                raise ValueError(f"non-finite pillar time {t}")
            if t <= prev:
                raise ValueError(
                    "pillar times must be strictly increasing and positive; "
                    f"offending time {t} after {prev}"
                )
            prev = t
        for t, p in zip(times, dfs):
            if not math.isfinite(p) or p <= 0.0:
                raise ValueError(
                    f"discount factor at t={t} must be finite and > 0, got {p}"
                )
        self.times = times
        self.dfs = dfs
        # Internal nodes include the implicit (0, ln 1 = 0).
        self._t = (0.0,) + times
        self._lnp = (0.0,) + tuple(math.log(p) for p in dfs)

    # ------------------------------------------------------------------ #

    def _segment(self, t: float) -> int:
        """Index i of the segment [t_i, t_{i+1}] used for time ``t``.

        Times at or beyond the last pillar map to the last segment
        (flat-forward extrapolation); a time exactly at an interior node
        maps to the segment starting at that node.
        """
        n_seg = len(self._t) - 1
        i = bisect_right(self._t, t) - 1
        if i < 0:
            i = 0
        if i > n_seg - 1:
            i = n_seg - 1
        return i

    def _slope(self, i: int) -> float:
        """d ln(DF)/dt on segment i (equals minus the constant forward)."""
        return (self._lnp[i + 1] - self._lnp[i]) / (self._t[i + 1] - self._t[i])

    def _ln_df(self, t: float) -> float:
        if not math.isfinite(t) or t < 0.0:
            raise ValueError(f"time must be finite and >= 0, got {t}")
        if t == 0.0:
            return 0.0
        i = self._segment(t)
        return self._lnp[i] + self._slope(i) * (t - self._t[i])

    # ------------------------------------------------------------------ #

    def df(self, t: float) -> float:
        """Discount factor DF(t); DF(0) = 1, flat-forward beyond last pillar."""
        return math.exp(self._ln_df(t))

    def zero_rate(self, t: float) -> float:
        """Continuously compounded zero rate ``z(t) = -ln DF(t) / t``.

        At ``t = 0`` the limit is returned, i.e. the instantaneous forward
        of the first segment.
        """
        if not math.isfinite(t) or t < 0.0:
            raise ValueError(f"time must be finite and >= 0, got {t}")
        if t == 0.0:
            return -self._slope(0)
        return -self._ln_df(t) / t

    def inst_forward(self, t: float) -> float:
        """Instantaneous forward ``f(t) = -d ln DF / dt``.

        Piecewise constant by construction.  At a pillar the *right*
        segment's forward is returned; at/after the last pillar the last
        segment's forward (the flat extrapolation level).
        """
        if not math.isfinite(t) or t < 0.0:
            raise ValueError(f"time must be finite and >= 0, got {t}")
        return -self._slope(self._segment(t))

    def fwd_rate(self, t1: float, t2: float) -> float:
        """Simply compounded forward rate over ``[t1, t2]``:

        ``F(t1, t2) = (DF(t1)/DF(t2) - 1) / (t2 - t1)``.
        """
        if not (math.isfinite(t1) and math.isfinite(t2)):
            raise ValueError("forward-rate times must be finite")
        if t1 < 0.0 or t2 <= t1:
            raise ValueError(
                f"need 0 <= t1 < t2 for a forward rate, got t1={t1}, t2={t2}"
            )
        return (self.df(t1) / self.df(t2) - 1.0) / (t2 - t1)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DiscountCurve(pillars={len(self.times)}, last_t={self.times[-1]})"


def par_swap_rate(curve: DiscountCurve, times: Sequence[float]) -> float:
    """Forward-starting par swap rate off a single curve.

    ``times = [t0, t1, ..., tn]`` gives the swap start ``t0 >= 0`` and the
    fixed payment times.  With accruals ``tau_i = t_i - t_{i-1}`` and the
    single-curve identity (floating leg PV = DF(t0) - DF(tn)):

    ``par = (DF(t0) - DF(tn)) / sum_i tau_i DF(t_i)``.
    """
    ts = [float(t) for t in times]
    if len(ts) < 2:
        raise ValueError("par_swap_rate needs a start and at least one payment")
    prev = ts[0]
    if prev < 0.0:
        raise ValueError(f"swap start must be >= 0, got {prev}")
    annuity = 0.0
    for t in ts[1:]:
        if t <= prev:
            raise ValueError("swap schedule times must be strictly increasing")
        annuity += (t - prev) * curve.df(t)
        prev = t
    return (curve.df(ts[0]) - curve.df(ts[-1])) / annuity
