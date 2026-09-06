"""Bachelier (normal) model: price, implied vol, parity, validation."""

from __future__ import annotations

import math

import pytest

from irm import bachelier_implied_vol, bachelier_price


def test_atm_closed_form() -> None:
    # ATM: price = A sigma sqrt(T) / sqrt(2 pi).
    a, vol, t = 0.98, 0.005, 1.0
    p = bachelier_price(-0.002, -0.002, t, vol, a, True)
    assert p == pytest.approx(a * vol * math.sqrt(t) / math.sqrt(2.0 * math.pi), abs=1e-17)
    assert bachelier_price(-0.002, -0.002, t, vol, a, False) == pytest.approx(p, abs=1e-17)
    assert bachelier_implied_vol(p, -0.002, -0.002, t, a, True) == pytest.approx(vol, abs=1e-14)


@pytest.mark.parametrize("forward", [-0.01, 0.0, 0.02, 0.05])
@pytest.mark.parametrize("strike", [-0.02, 0.0, 0.03])
@pytest.mark.parametrize("vol", [0.001, 0.0075, 0.02])
@pytest.mark.parametrize("payer", [True, False])
def test_implied_vol_round_trip_grid(forward: float, strike: float, vol: float, payer: bool) -> None:
    t, annuity = 2.5, 3.7
    p = bachelier_price(forward, strike, t, vol, annuity, payer)
    iv = bachelier_implied_vol(p, forward, strike, t, annuity, payer)
    d = abs(forward - strike) / (vol * math.sqrt(t))
    if d < 4.0:
        # Time value resolvable in double precision: the vol round-trips.
        assert iv == pytest.approx(vol, abs=1e-12)
    else:
        # Deep in/out of the money the time value is below rounding: the
        # implied vol is whatever reprices the quote, and the *price* must
        # round-trip exactly.
        assert 0.0 <= iv <= vol * 1.001
    assert bachelier_price(forward, strike, t, iv, annuity, payer) == pytest.approx(p, abs=1e-15)


def test_payer_receiver_parity_and_monotonicity() -> None:
    f, k, t, a = 0.03, 0.025, 2.0, 4.2
    for vol in [0.0, 0.002, 0.0075, 0.03]:
        payer = bachelier_price(f, k, t, vol, a, True)
        receiver = bachelier_price(f, k, t, vol, a, False)
        assert payer - receiver == pytest.approx(a * (f - k), abs=1e-15)
    vols = [0.001, 0.002, 0.005, 0.01, 0.02]
    prices = [bachelier_price(f, k, t, v, a, True) for v in vols]
    assert all(b > x for x, b in zip(prices[:-1], prices[1:]))


def test_intrinsic_and_below_intrinsic() -> None:
    f, k, t, a = 0.03, 0.025, 2.0, 4.2
    intrinsic = a * (f - k)
    assert bachelier_price(f, k, t, 0.0, a, True) == intrinsic
    assert bachelier_price(f, k, t, 0.0, a, False) == 0.0
    assert bachelier_implied_vol(intrinsic, f, k, t, a, True) == 0.0
    assert bachelier_implied_vol(0.0, f, k, t, a, False) == 0.0
    with pytest.raises(ValueError, match="intrinsic"):
        bachelier_implied_vol(intrinsic - 1e-6, f, k, t, a, True)
    with pytest.raises(ValueError, match="intrinsic"):
        bachelier_implied_vol(-1e-9, f, k, t, a, False)


def test_validation() -> None:
    nan = float("nan")
    with pytest.raises(ValueError):
        bachelier_price(nan, 0.02, 1.0, 0.01, 1.0)
    with pytest.raises(ValueError):
        bachelier_price(0.02, 0.02, 0.0, 0.01, 1.0)
    with pytest.raises(ValueError):
        bachelier_price(0.02, 0.02, 1.0, -0.01, 1.0)
    with pytest.raises(ValueError):
        bachelier_price(0.02, 0.02, 1.0, 0.01, 0.0)
    with pytest.raises(ValueError):
        bachelier_implied_vol(nan, 0.02, 0.02, 1.0, 1.0)
    with pytest.raises(ValueError):
        bachelier_implied_vol(0.01, 0.02, 0.02, float("inf"), 1.0)
    with pytest.raises(ValueError, match="too large"):
        bachelier_implied_vol(1e300, 0.02, 0.02, 1.0, 1.0)
