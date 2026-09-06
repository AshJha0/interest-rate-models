"""Bachelier (normal) model for caplets and swaptions: price and implied vol.

Rates desks quote cap/floor and swaption prices as *normal* ("bp")
volatilities: the volatility ``sigma_N`` such that the Bachelier formula
reproduces the price.  Hull-White is a Gaussian model, so its option prices
are naturally compared with the market in this unit.  For a payer option
on a forward ``F`` with strike ``K``, expiry ``T`` and annuity ``A``
(``A = tau DF(T_pay)`` for a caplet, ``sum_i tau_i DF(u_i)`` for a swaption):

``price = A [ (F - K) N(d) + sigma_N sqrt(T) phi(d) ]``,
``d = (F - K) / (sigma_N sqrt(T))``

and the receiver is ``A [ (K - F) N(-d) + sigma_N sqrt(T) phi(d) ]``.
With ``sigma_N = 0`` both collapse to discounted intrinsic.  The price is
strictly increasing in ``sigma_N`` (vega ``A sqrt(T) phi(d) > 0``), so the
implied volatility is unique and found with the native Brent solver on an
expanding bracket.  Negative forwards and strikes are first-class (the
normal model is the reason the market moved to bp vols after 2012).
"""

from __future__ import annotations

import math

from .mathutils import norm_cdf, norm_pdf
from .rootfind import brentq

__all__ = ["bachelier_price", "bachelier_implied_vol"]

#: Absolute price tolerance used by :func:`bachelier_implied_vol` to decide
#: that a quote is at intrinsic (implied vol 0).
INTRINSIC_TOL = 1e-15
#: Root tolerance on the implied volatility.
IMPLIED_VOL_XTOL = 1e-14
#: Upper bracket search: start at this vol and double at most this many times.
_VOL_BRACKET_START = 0.01
_VOL_BRACKET_DOUBLINGS = 40


def _check_common(forward: float, strike: float, expiry: float, annuity: float) -> None:
    if not (math.isfinite(forward) and math.isfinite(strike)):
        raise ValueError("bachelier: forward and strike must be finite")
    if not math.isfinite(expiry) or expiry <= 0.0:
        raise ValueError(f"bachelier: expiry must be finite and > 0, got {expiry}")
    if not math.isfinite(annuity) or annuity <= 0.0:
        raise ValueError(f"bachelier: annuity must be finite and > 0, got {annuity}")


def bachelier_price(
    forward: float,
    strike: float,
    expiry: float,
    vol: float,
    annuity: float,
    payer: bool = True,
) -> float:
    """Normal-model option price (per unit notional, annuity included).

    ``vol`` is the absolute (normal) volatility in rate units per sqrt(year):
    ``0.0075`` means 75 bp.  ``vol = 0`` returns discounted intrinsic.
    """
    _check_common(forward, strike, expiry, annuity)
    if not math.isfinite(vol) or vol < 0.0:
        raise ValueError(f"bachelier: vol must be finite and >= 0, got {vol}")
    moneyness = forward - strike if payer else strike - forward
    if vol == 0.0:
        return annuity * max(moneyness, 0.0)
    sd = vol * math.sqrt(expiry)
    d = moneyness / sd
    return annuity * (moneyness * norm_cdf(d) + sd * norm_pdf(d))


def bachelier_implied_vol(
    price: float,
    forward: float,
    strike: float,
    expiry: float,
    annuity: float,
    payer: bool = True,
) -> float:
    """Normal implied volatility of a payer/receiver price.

    ``price`` must be finite and at least the discounted intrinsic
    ``annuity * max(+-(F - K), 0)``; a price within ``INTRINSIC_TOL`` of
    intrinsic returns ``0.0``.  Otherwise Brent's method solves
    ``bachelier_price(vol) = price`` on ``[0, hi]`` with ``hi`` found by
    doubling from ``0.01`` (at most 40 doublings, i.e. vol up to ~1e10 —
    a quote needing more is rejected).  Root tolerance ``1e-14`` on the vol.
    """
    _check_common(forward, strike, expiry, annuity)
    if not math.isfinite(price):
        raise ValueError(f"bachelier: price must be finite, got {price}")
    intrinsic = bachelier_price(forward, strike, expiry, 0.0, annuity, payer)
    if price < intrinsic - INTRINSIC_TOL:
        raise ValueError(
            f"bachelier: price {price} below intrinsic {intrinsic} — no implied vol"
        )
    if price <= intrinsic + INTRINSIC_TOL:
        return 0.0

    def diff(v: float) -> float:
        return bachelier_price(forward, strike, expiry, v, annuity, payer) - price

    hi = _VOL_BRACKET_START
    bracketed = False
    for _ in range(_VOL_BRACKET_DOUBLINGS):
        if diff(hi) > 0.0:
            bracketed = True
            break
        hi *= 2.0
    if not bracketed:
        raise ValueError(f"bachelier: price {price} too large to imply a volatility")
    return brentq(diff, 0.0, hi, xtol=IMPLIED_VOL_XTOL)
