"""Sequential single-curve bootstrap from deposits, FRAs and par swaps.

Framework
---------
Single-curve: the floating leg of a swap is projected off the same curve it
is discounted on, so its PV telescopes to ``DF(0) - DF(T)``.  (Real desks
use multi-curve setups — OIS discounting plus tenor-specific projection
curves; the OIS variant below bootstraps such a discount curve, but
projection basis is out of scope and documented as a simplification.)

Algorithm
---------
Instruments are sorted by pillar (maturity) time, which must be strictly
increasing.  For each instrument in order, the discount factor at its
pillar is the single unknown ``x``; all earlier dates interpolate on the
already-solved pillars (plus the trial pillar), so each pillar is found by
a 1-D root find of the instrument's *residual* (PV mispricing):

* Deposit (simple interest):  ``residual = DF(T) (1 + R T) - 1``
* FRA:                        ``residual = DF(T1) - (1 + R (T2-T1)) DF(T2)``
* Par swap / OIS swap:        ``residual = R * annuity - (1 - DF(Tn))``
  with ``annuity = sum_i tau_i DF(t_i)`` over the annual fixed schedule.

The root is found with the native Brent solver on the bracket
``DF in [1e-10, 100]`` with ``xtol = 1e-14``.  A bracket failure means no
positive discount factor can reprice the quote — i.e. crossed/arbitrageable
quotes (the "negative implied DF" case) — and raises ``ValueError``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

from .curve import DiscountCurve
from .rootfind import brentq

__all__ = [
    "Deposit",
    "FRA",
    "Swap",
    "OISSwap",
    "Instrument",
    "bootstrap",
    "annual_schedule",
]


def _check_rate(rate: float) -> None:
    if not math.isfinite(rate):
        raise ValueError(f"quote rate must be finite, got {rate}")


def annual_schedule(maturity: float) -> Tuple[float, ...]:
    """Annual fixed-leg payment times ending exactly at ``maturity``.

    ``n = ceil(maturity)`` payments at ``maturity - (n-1), ..., maturity``;
    e.g. 2.0 -> (1.0, 2.0); 2.5 -> (0.5, 1.5, 2.5); 0.25 -> (0.25,).
    A short first stub (< 1y) is created for non-integer maturities.
    """
    if not math.isfinite(maturity) or maturity <= 0.0:
        raise ValueError(f"maturity must be finite and > 0, got {maturity}")
    n = math.ceil(maturity - 1e-12)
    return tuple(maturity - (n - 1 - k) for k in range(n))


@dataclass(frozen=True)
class Deposit:
    """Cash deposit paying simple interest ``R`` at ``maturity``."""

    maturity: float
    rate: float

    def __post_init__(self) -> None:
        _check_rate(self.rate)
        if not math.isfinite(self.maturity) or self.maturity <= 0.0:
            raise ValueError(f"deposit maturity must be > 0, got {self.maturity}")
        if 1.0 + self.rate * self.maturity <= 0.0:
            raise ValueError(
                "deposit quote implies non-positive discount factor "
                f"(1 + R*T = {1.0 + self.rate * self.maturity})"
            )

    @property
    def pillar(self) -> float:
        return self.maturity

    def residual(self, curve: DiscountCurve) -> float:
        """Mispricing of investing 1 at the simple rate: DF(T)(1+RT) - 1."""
        return curve.df(self.maturity) * (1.0 + self.rate * self.maturity) - 1.0


@dataclass(frozen=True)
class FRA:
    """Forward rate agreement fixing the simple forward over [start, end]."""

    start: float
    end: float
    rate: float

    def __post_init__(self) -> None:
        _check_rate(self.rate)
        if not (math.isfinite(self.start) and math.isfinite(self.end)):
            raise ValueError("FRA times must be finite")
        if self.start < 0.0 or self.end <= self.start:
            raise ValueError(
                f"FRA needs 0 <= start < end, got start={self.start}, end={self.end}"
            )
        if 1.0 + self.rate * (self.end - self.start) <= 0.0:
            raise ValueError("FRA quote implies non-positive discount factor")

    @property
    def pillar(self) -> float:
        return self.end

    def residual(self, curve: DiscountCurve) -> float:
        """DF-linear mispricing: DF(T1) - (1 + R tau) DF(T2)."""
        tau = self.end - self.start
        return curve.df(self.start) - (1.0 + self.rate * tau) * curve.df(self.end)


@dataclass(frozen=True)
class Swap:
    """Par interest-rate swap: annual fixed leg vs single-curve floating leg.

    Floating leg PV telescopes to ``1 - DF(T)`` because projection and
    discounting share the curve.
    """

    maturity: float
    rate: float

    def __post_init__(self) -> None:
        _check_rate(self.rate)
        annual_schedule(self.maturity)  # validates maturity

    @property
    def pillar(self) -> float:
        return self.maturity

    def residual(self, curve: DiscountCurve) -> float:
        """R * annuity - (1 - DF(T)), the fixed-minus-floating PV."""
        annuity = 0.0
        prev = 0.0
        for t in annual_schedule(self.maturity):
            annuity += (t - prev) * curve.df(t)
            prev = t
        return self.rate * annuity - (1.0 - curve.df(self.maturity))


@dataclass(frozen=True)
class OISSwap:
    """OIS par swap: annual fixed leg vs daily-compounded overnight leg.

    Daily compounding of the overnight rate is approximated as continuous
    compounding at the curve's short rate, under which the compounded
    floating coupon over ``[t_{i-1}, t_i]`` pays ``DF(t_{i-1})/DF(t_i) - 1``
    and the floating leg PV telescopes to ``1 - DF(T)`` — identical in form
    to :class:`Swap`.  The daily-vs-continuous compounding gap is O(rate^2 /
    365) and is a documented approximation.
    """

    maturity: float
    rate: float

    def __post_init__(self) -> None:
        _check_rate(self.rate)
        annual_schedule(self.maturity)

    @property
    def pillar(self) -> float:
        return self.maturity

    def residual(self, curve: DiscountCurve) -> float:
        annuity = 0.0
        prev = 0.0
        for t in annual_schedule(self.maturity):
            annuity += (t - prev) * curve.df(t)
            prev = t
        return self.rate * annuity - (1.0 - curve.df(self.maturity))


Instrument = Union[Deposit, FRA, Swap, OISSwap]

_DF_LO = 1e-10
_DF_HI = 100.0


def bootstrap(instruments: Sequence[Instrument]) -> DiscountCurve:
    """Sequentially bootstrap a discount curve from sorted instruments.

    Instruments must be ordered with strictly increasing pillar times
    (duplicates rejected).  Each pillar DF is solved with Brent's method on
    ``[1e-10, 100]`` (DF > 1 allowed: negative rates).  Raises
    ``ValueError`` on ordering violations or when no positive DF can
    reprice a quote (crossed/arbitrageable inputs).
    """
    if len(instruments) == 0:
        raise ValueError("bootstrap needs at least one instrument")
    prev_pillar = 0.0
    for ins in instruments:
        if ins.pillar <= prev_pillar:
            raise ValueError(
                "instrument pillars must be strictly increasing "
                f"(duplicate or out-of-order pillar at t={ins.pillar})"
            )
        prev_pillar = ins.pillar

    times: List[float] = []
    dfs: List[float] = []
    for ins in instruments:
        pillar = ins.pillar

        def objective(x: float, _ins: Instrument = ins, _t: float = pillar) -> float:
            trial = DiscountCurve(times + [_t], dfs + [x])
            return _ins.residual(trial)

        try:
            df = brentq(objective, _DF_LO, _DF_HI, xtol=1e-14)
        except ValueError as exc:
            raise ValueError(
                f"bootstrap failed at pillar t={pillar} "
                f"({type(ins).__name__}, rate={ins.rate}): no admissible "
                f"positive discount factor — crossed/arbitrageable quotes? "
                f"[{exc}]"
            ) from exc
        times.append(pillar)
        dfs.append(df)
    return DiscountCurve(times, dfs)
