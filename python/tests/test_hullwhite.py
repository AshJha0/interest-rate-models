"""Hull-White: curve fit, options, caps, Jamshidian, MC, theta(t)."""

from __future__ import annotations

import math

import pytest

from irm import DiscountCurve, HullWhite

HW_T = [1.0, 2.0, 3.0, 5.0, 10.0]
HW_DF = [math.exp(-(0.02 + 0.0015 * t) * t) for t in HW_T]
A, SIGMA = 0.1, 0.01


@pytest.fixture()
def curve() -> DiscountCurve:
    return DiscountCurve(HW_T, HW_DF)


@pytest.fixture()
def hw(curve: DiscountCurve) -> HullWhite:
    return HullWhite(A, SIGMA, curve)


def test_zcb_t0_matches_market_exactly(hw: HullWhite, curve: DiscountCurve) -> None:
    # By construction the e^{B f(0,0)} and e^{-B r0} factors cancel: P(0,T)
    # must equal the market DF to machine precision, at and between pillars.
    for t in [0.5, 1.0, 2.0, 2.5, 3.0, 5.0, 7.0, 10.0]:
        assert hw.zcb_price(0.0, t) == pytest.approx(curve.df(t), rel=1e-13)


def test_zcb_t0_matches_bootstrapped_eur(eur_curve: DiscountCurve) -> None:
    hw = HullWhite(0.05, 0.008, eur_curve)
    for t in [1.0, 5.0, 10.0, 30.0]:
        assert hw.zcb_price(0.0, t) == pytest.approx(eur_curve.df(t), rel=1e-13)
    # Negative-rate curve: DF > 1 flows straight through the model.
    assert hw.zcb_price(0.0, 1.0) > 1.0


def test_zcb_monotone_decreasing_in_r(hw: HullWhite) -> None:
    # Strict monotonicity in r is what Jamshidian's decomposition relies on.
    prices = [hw.zcb_price(1.0, 5.0, r) for r in [-0.02, 0.0, 0.02, 0.05, 0.10]]
    for p, q in zip(prices[:-1], prices[1:]):
        assert q < p


def test_fwd0_piecewise_and_pillar_average(hw: HullWhite, curve: DiscountCurve) -> None:
    # Inside a segment the FD reproduces the constant forward exactly.
    assert hw.fwd0(1.5) == pytest.approx(curve.inst_forward(1.5), rel=1e-9)
    # Straddling a pillar it returns the average of adjacent forwards.
    left = curve.inst_forward(1.5)
    right = curve.inst_forward(2.5)
    assert hw.fwd0(2.0) == pytest.approx(0.5 * (left + right), rel=1e-9)


def test_theta_on_flat_forward_curve() -> None:
    # Flat continuous curve at 3%: f(0,t) = 0.03 for all t, so
    # theta(t) = a*0.03 + sigma^2/(2a) (1 - e^{-2at}).
    flat = DiscountCurve([10.0], [math.exp(-0.03 * 10.0)])
    hw = HullWhite(A, SIGMA, flat)
    for t in [0.5, 1.0, 3.0]:
        expected = A * 0.03 + SIGMA**2 / (2 * A) * (1 - math.exp(-2 * A * t))
        assert hw.theta(t) == pytest.approx(expected, abs=1e-6)
    assert hw.alpha(0.0) == pytest.approx(0.03, abs=1e-9)


def test_zb_option_put_call_parity_grid(hw: HullWhite, curve: DiscountCurve) -> None:
    for expiry, s_mat in [(0.5, 2.0), (1.0, 3.0), (2.0, 5.0)]:
        for k in [0.85, 0.92, 0.96, 1.00]:
            lhs = hw.zbc(expiry, s_mat, k) - hw.zbp(expiry, s_mat, k)
            rhs = curve.df(s_mat) - k * curve.df(expiry)
            assert lhs == pytest.approx(rhs, abs=1e-13)


def test_caplet_floorlet_parity(hw: HullWhite, curve: DiscountCurve) -> None:
    # caplet - floorlet = tau * DF(pay) * (F - K)  (an off-market FRA PV).
    reset, pay, k, notional = 1.0, 2.0, 0.025, 100.0
    tau = pay - reset
    fwd = curve.fwd_rate(reset, pay)
    lhs = hw.caplet(reset, pay, k, notional) - hw.floorlet(reset, pay, k, notional)
    assert lhs == pytest.approx(notional * tau * curve.df(pay) * (fwd - k), abs=1e-10)


def test_cap_equals_caplet_strip(hw: HullWhite) -> None:
    schedule = [0.5, 1.0, 2.0, 3.0]
    strip = sum(
        hw.caplet(a, b, 0.024, 100.0) for a, b in zip(schedule[:-1], schedule[1:])
    )
    assert hw.cap(schedule, 0.024, 100.0) == pytest.approx(strip, rel=1e-14)


def test_sigma_zero_caplet_is_discounted_intrinsic(curve: DiscountCurve) -> None:
    hw0 = HullWhite(A, 0.0, curve)
    reset, pay, k = 1.0, 2.0, 0.02
    tau = pay - reset
    fwd = curve.fwd_rate(reset, pay)
    intrinsic = curve.df(pay) * tau * max(fwd - k, 0.0)
    assert hw0.caplet(reset, pay, k, 1.0) == pytest.approx(intrinsic, abs=1e-15)
    # OTM side is exactly zero.
    assert hw0.caplet(reset, pay, fwd + 0.01, 1.0) == 0.0


def test_caplet_positive_and_increasing_in_vol(curve: DiscountCurve) -> None:
    vals = [
        HullWhite(A, s, curve).caplet(1.0, 2.0, 0.025, 100.0)
        for s in [0.0, 0.005, 0.01, 0.02]
    ]
    for lo, hi in zip(vals[:-1], vals[1:]):
        assert hi > lo - 1e-15  # vega >= 0


def test_mc_caplet_within_3se(hw: HullWhite) -> None:
    analytic = hw.caplet(1.0, 2.0, 0.025, 100.0)
    price, se = hw.mc_caplet(1.0, 2.0, 0.025, 100.0, n_steps=4, n_paths=50_000, seed=5)
    assert se > 0.0
    assert abs(price - analytic) < 3.0 * se


def test_mc_caplet_sigma0_deterministic(curve: DiscountCurve) -> None:
    hw0 = HullWhite(A, 0.0, curve)
    price, se = hw0.mc_caplet(1.0, 2.0, 0.02, 100.0, n_steps=2, n_paths=64, seed=1)
    assert se == pytest.approx(0.0, abs=1e-14)
    assert price == pytest.approx(hw0.caplet(1.0, 2.0, 0.02, 100.0), rel=1e-12)


def test_jamshidian_swaption(hw: HullWhite, curve: DiscountCurve) -> None:
    expiry, pay_times, fixed = 1.0, [2.0, 3.0, 4.0, 5.0, 6.0], 0.028
    res = hw.jamshidian_swaption(expiry, pay_times, fixed, notional=100.0)
    # r* really solves the coupon-bond equation.
    assert abs(res.residual) < 1e-10
    coupon_bond = sum(
        c * hw.zcb_price(expiry, t, res.r_star)
        for c, t in zip([fixed] * 4 + [1.0 + fixed], pay_times)
    )
    assert coupon_bond == pytest.approx(1.0, abs=1e-10)
    # Decomposition strikes are the ZCB prices at r*.
    assert res.strikes[0] == pytest.approx(hw.zcb_price(expiry, 2.0, res.r_star))
    # Swaption value >= forward-swap intrinsic (and >= 0).
    annuity = sum(curve.df(t) for t in pay_times)
    intrinsic = (curve.df(expiry) - curve.df(pay_times[-1])) - fixed * annuity
    assert res.value >= max(100.0 * intrinsic, 0.0) - 1e-12
    assert res.value > 0.0


def test_jamshidian_payer_receiver_parity(hw: HullWhite, curve: DiscountCurve) -> None:
    # payer - receiver = PV of the forward payer swap.
    expiry, pay_times, fixed = 1.0, [2.0, 3.0, 4.0], 0.026
    payer = hw.jamshidian_swaption(expiry, pay_times, fixed, payer=True).value
    receiver = hw.jamshidian_swaption(expiry, pay_times, fixed, payer=False).value
    annuity = sum(curve.df(t) for t in pay_times)
    fwd_swap = (curve.df(expiry) - curve.df(pay_times[-1])) - fixed * annuity
    assert payer - receiver == pytest.approx(fwd_swap, abs=1e-12)


def test_jamshidian_atm_strike_grid(hw: HullWhite) -> None:
    # Property-style: payer swaption decreasing in the fixed rate.
    expiry, pay_times = 1.0, [2.0, 3.0, 4.0, 5.0]
    vals = [
        hw.jamshidian_swaption(expiry, pay_times, x).value
        for x in [0.015, 0.020, 0.025, 0.030, 0.035]
    ]
    for lo, hi in zip(vals[:-1], vals[1:]):
        assert hi < lo


def test_jamshidian_negative_curve() -> None:
    neg = DiscountCurve(HW_T, [math.exp(-(-0.004 + 0.0012 * t) * t) for t in HW_T])
    hw = HullWhite(A, SIGMA, neg)
    res = hw.jamshidian_swaption(1.0, [2.0, 3.0, 4.0], 0.0, notional=100.0)
    assert abs(res.residual) < 1e-10
    assert res.value > 0.0
    assert res.r_star < 0.0  # ATM-ish zero strike on a negative curve


def test_validation(curve: DiscountCurve, hw: HullWhite) -> None:
    with pytest.raises(ValueError):
        HullWhite(0.0, 0.01, curve)
    with pytest.raises(ValueError):
        HullWhite(0.1, -0.01, curve)
    with pytest.raises(ValueError):
        HullWhite(0.1, 0.01, "not a curve")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        hw.caplet(2.0, 1.0, 0.02)  # pay before reset
    with pytest.raises(ValueError):
        hw.caplet(1.0, 2.0, -1.5)  # 1 + K*tau <= 0
    with pytest.raises(ValueError):
        hw.cap([1.0], 0.02)
    with pytest.raises(ValueError):
        hw.zbc(1.0, 3.0, -0.5)
    with pytest.raises(ValueError):
        hw.jamshidian_swaption(0.0, [1.0, 2.0], 0.02)
    with pytest.raises(ValueError):
        hw.jamshidian_swaption(1.0, [3.0, 2.0], 0.02)
    with pytest.raises(ValueError):
        hw.jamshidian_swaption(1.0, [], 0.02)
    with pytest.raises(ValueError):
        hw.fwd0(-1.0)
