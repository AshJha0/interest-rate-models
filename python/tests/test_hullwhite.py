"""Hull-White: curve fit, options, caps, Jamshidian, MC, theta(t)."""

from __future__ import annotations

import math

import pytest

from irm import (
    JAMSHIDIAN_RESIDUAL_TOL,
    CapletQuote,
    DiscountCurve,
    HullWhite,
    SwaptionQuote,
    calibrate_hullwhite,
)
from irm.mathutils import ou_integral_variance

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


# ----------------------------------------------------------------------- #
# Validation of notional / strike / schedule (MAJOR-5, test #7, #14)
# ----------------------------------------------------------------------- #


def test_non_finite_notional_and_mc_strike_rejected(hw: HullWhite) -> None:
    nan, inf = float("nan"), float("inf")
    with pytest.raises(ValueError):
        hw.caplet(1.0, 2.0, 0.025, nan)
    with pytest.raises(ValueError):
        hw.floorlet(1.0, 2.0, 0.025, inf)
    with pytest.raises(ValueError):
        hw.cap([1.0, 2.0, 3.0], 0.02, inf)
    with pytest.raises(ValueError):
        hw.jamshidian_swaption(1.0, [2.0, 3.0], 0.02, notional=nan)
    with pytest.raises(ValueError):
        hw.mc_caplet(1.0, 2.0, nan, 100.0, 8, 100, 1)
    with pytest.raises(ValueError):
        hw.mc_caplet(1.0, 2.0, -1.5, 100.0, 8, 100, 1)  # 1 + K tau <= 0
    with pytest.raises(ValueError):
        hw.mc_caplet(1.0, 2.0, 0.02, inf, 8, 100, 1)
    with pytest.raises(ValueError):
        hw.caplet(nan, 2.0, 0.02)
    with pytest.raises(ValueError):
        hw.b_factor(nan, 5.0)
    with pytest.raises(ValueError):
        hw.b_factor(-1.0, 5.0)
    with pytest.raises(ValueError):
        hw.jamshidian_swaption(1.0, [2.0, inf], 0.02)
    with pytest.raises(ValueError):
        hw.theta(1.0, h=0.0)
    # A negative notional is a short position, not an error.
    assert hw.caplet(1.0, 2.0, 0.025, -100.0) == pytest.approx(-hw.caplet(1.0, 2.0, 0.025, 100.0))


def test_cap_schedule_non_increasing_rejected(hw: HullWhite) -> None:
    with pytest.raises(ValueError):
        hw.cap([1.0, 2.0, 2.0, 3.0], 0.02)
    with pytest.raises(ValueError):
        hw.cap([1.0, float("nan"), 3.0], 0.02)
    with pytest.raises(ValueError):
        hw.cap([-1.0, 1.0], 0.02)
    with pytest.raises(ValueError):
        hw.cap([1.0, 2.0], float("nan"))


def test_parameters_are_read_only(hw: HullWhite) -> None:
    with pytest.raises(AttributeError):
        hw.a = -1.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        hw.sigma = 2.0  # type: ignore[misc]


# ----------------------------------------------------------------------- #
# Jamshidian contract (MAJOR-6, MINOR-4, tests #8, #10)
# ----------------------------------------------------------------------- #


def test_jamshidian_residual_contract_enforced(hw: HullWhite) -> None:
    expiry, pays, x = 1.0, [2.0, 3.0, 4.0, 5.0, 6.0], 0.028
    res = hw.jamshidian_swaption(expiry, pays, x, notional=100.0)
    # Re-running the decomposition from the found r* reproduces the value.
    again = hw.jamshidian_at_r_star(expiry, pays, x, res.r_star, notional=100.0)
    assert again.value == pytest.approx(res.value, abs=1e-14)
    assert again.residual == res.residual
    assert abs(again.residual) < JAMSHIDIAN_RESIDUAL_TOL
    # A wrong r* (the bracket endpoint) violates |g(r*)| < 1e-10 -> error.
    with pytest.raises(ValueError, match="residual"):
        hw.jamshidian_at_r_star(expiry, pays, x, -1.0, notional=100.0)
    with pytest.raises(ValueError, match="residual"):
        hw.jamshidian_at_r_star(expiry, pays, x, res.r_star + 1e-6, notional=100.0)
    with pytest.raises(ValueError):
        hw.jamshidian_at_r_star(expiry, pays, x, float("nan"))
    # Receiver from the same root.
    rec = hw.jamshidian_at_r_star(expiry, pays, x, res.r_star, notional=100.0, payer=False)
    assert rec.value == pytest.approx(
        hw.jamshidian_swaption(expiry, pays, x, notional=100.0, payer=False).value, abs=1e-14)


def _swaption_by_integration(hw: HullWhite, expiry: float, pays, x: float, n: int = 2000) -> float:
    """Payer swaption per unit notional by direct integration of the payoff
    (1 - g(r_T))^+ under the T-forward Gaussian law of r_T.

    Independent of the decomposition: the payoff is evaluated as a max over
    the coupon bond.  Simpson's rule with ``n`` intervals from the payoff
    kink (found by bisection on g) to mu + 12 sd; the integrand is smooth
    there so the error is O(h^4).
    """
    c = hw.curve
    v = hw.sigma**2 * (-math.expm1(-2.0 * hw.a * expiry)) / (2.0 * hw.a)
    b1, a1 = hw.b_factor(expiry, pays[0]), hw.a_factor(expiry, pays[0])
    mu = (math.log(a1 * c.df(expiry) / c.df(pays[0])) + b1 * b1 * v / 2.0) / b1
    prev = expiry
    coupons = []
    for t in pays:
        coupons.append(x * (t - prev))
        prev = t
    coupons[-1] += 1.0
    sd = math.sqrt(v)

    def payoff(r: float) -> float:
        return max(1.0 - sum(ci * hw.zcb_price(expiry, t, r) for ci, t in zip(coupons, pays)), 0.0)

    # Kink: bisection for the smallest r with positive payoff.
    lo, hi = mu - 12.0 * sd, mu + 12.0 * sd
    while lo < mu - 400.0 * sd or payoff(lo) > 0.0:
        lo -= 12.0 * sd
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if payoff(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    kink = hi
    top = mu + 12.0 * sd
    h = (top - kink) / n
    total = 0.0
    for i in range(n + 1):
        r = kink + i * h
        w = math.exp(-0.5 * ((r - mu) / sd) ** 2) / (sd * math.sqrt(2.0 * math.pi))
        coef = 1.0 if i in (0, n) else (4.0 if i % 2 == 1 else 2.0)
        total += coef * payoff(r) * w
    return c.df(expiry) * total * h / 3.0


def test_jamshidian_negative_fixed_rate_stays_exact(hw: HullWhite) -> None:
    # For X < 0 the coupon signs are (-, ..., -, +): g is no longer monotone,
    # but g - 1 still has a single crossing, so the decomposition is exact.
    # Direct integration confirms it (the review's "silently wrong" claim
    # does not hold); 1 + X tau_n <= 0 is the genuinely degenerate case.
    expiry, pays = 1.0, [2.0, 3.0, 4.0, 5.0, 6.0]
    for x in (-0.005, -0.2, -0.5):
        res = hw.jamshidian_swaption(expiry, pays, x, notional=1.0)
        assert res.value > 0.0
        assert abs(res.residual) < 1e-10
        assert res.value == pytest.approx(_swaption_by_integration(hw, expiry, pays, x), rel=1e-9)
    assert hw.jamshidian_swaption(expiry, pays, -0.005).r_star < 0.0
    with pytest.raises(ValueError, match="degenerate"):
        hw.jamshidian_swaption(expiry, pays, -1.0)  # 1 + X tau_n = 0: g <= 0 everywhere
    with pytest.raises(ValueError, match="degenerate"):
        hw.jamshidian_swaption(expiry, pays, -1.5)
    # X = -0.8 still brackets and satisfies the residual contract; beyond
    # about -0.95 the coupon-bond terms are O(1e8+) and |g(r*)| cannot beat
    # 1e-10 in double precision -> the contract rejects it honestly.
    ok = hw.jamshidian_swaption(expiry, pays, -0.8)
    assert abs(ok.residual) < 1e-10 and ok.value > 0.0
    with pytest.raises(ValueError, match="residual"):
        hw.jamshidian_swaption(expiry, pays, -0.99)
    # An absurd short rate makes the ZCB price overflow: error, not inf.
    with pytest.raises(ValueError, match="not representable"):
        hw.zcb_price(1.0, 5.0, -1e6)


def test_jamshidian_positive_rate_matches_integration(hw: HullWhite) -> None:
    res = hw.jamshidian_swaption(1.0, [2.0, 3.0, 4.0, 5.0, 6.0], 0.028)
    assert res.value == pytest.approx(
        _swaption_by_integration(hw, 1.0, [2.0, 3.0, 4.0, 5.0, 6.0], 0.028), rel=1e-9)


# ----------------------------------------------------------------------- #
# Monte Carlo determinism / degenerate sizes (tests #16, #17)
# ----------------------------------------------------------------------- #


def test_mc_caplet_same_seed_and_single_path(hw: HullWhite) -> None:
    a = hw.mc_caplet(1.0, 2.0, 0.025, 100.0, n_steps=4, n_paths=500, seed=3)
    b = hw.mc_caplet(1.0, 2.0, 0.025, 100.0, n_steps=4, n_paths=500, seed=3)
    c = hw.mc_caplet(1.0, 2.0, 0.025, 100.0, n_steps=4, n_paths=500, seed=4)
    assert a == b and a != c
    price, se = hw.mc_caplet(1.0, 2.0, 0.02, 100.0, n_steps=1, n_paths=1, seed=1)
    assert math.isfinite(price) and price >= 0.0
    assert se == 0.0


def test_mc_caplet_tiny_mean_reversion_unbiased(curve: DiscountCurve) -> None:
    # a*dt = 1e-4/200 = 5e-7 per step exercises the series branch of var_i
    # and V(t); the estimator must remain unbiased.
    slow = HullWhite(1e-4, 0.01, curve)
    analytic = slow.caplet(1.0, 2.0, 0.025, 100.0)
    price, se = slow.mc_caplet(1.0, 2.0, 0.025, 100.0, n_steps=200, n_paths=20_000, seed=9)
    assert abs(price - analytic) < 3.0 * se


def test_variance_integral_uses_cancellation_free_kernel(curve: DiscountCurve) -> None:
    slow = HullWhite(1e-5, 0.02, curve)
    # V(t) = sigma^2 t^3 / 6 (1 + O(a t)) for a t -> 0.
    assert slow._variance_integral(1.0) == pytest.approx(0.5 * ou_integral_variance(1e-5, 0.02, 1.0))
    assert slow._variance_integral(1.0) == pytest.approx(0.02**2 / 6.0, rel=1e-5)
    assert slow._variance_integral(0.0) == 0.0


# ----------------------------------------------------------------------- #
# Calibration of (a, sigma) to caplet / swaption prices (MAJOR-7)
# ----------------------------------------------------------------------- #


def _quotes(true: HullWhite):
    caps = [CapletQuote(r, r + 1.0, 0.025, true.caplet(r, r + 1.0, 0.025, 1.0))
            for r in (1.0, 2.0, 4.0)]
    pays = (2.0, 3.0, 4.0, 5.0, 6.0)
    swps = [SwaptionQuote(1.0, pays, 0.028, true.jamshidian_swaption(1.0, pays, 0.028, 1.0).value)]
    return caps, swps


def test_calibrate_hullwhite_recovers_a_and_sigma(curve: DiscountCurve) -> None:
    true = HullWhite(0.08, 0.012, curve)
    caps, swps = _quotes(true)
    cal = calibrate_hullwhite(curve, caps, swps)
    assert cal.converged and cal.identified and not cal.at_bound and not cal.a_fixed
    assert cal.n_starts == 3
    assert cal.a == pytest.approx(0.08, abs=1e-8)
    assert cal.sigma == pytest.approx(0.012, abs=1e-9)
    assert cal.rmse < 1e-10
    model = cal.model(curve)
    for q in caps:
        assert model.caplet(q.reset, q.pay, q.strike, 1.0) == pytest.approx(q.price, abs=1e-10)


def test_calibrate_hullwhite_a_fixed_and_warm_start(curve: DiscountCurve) -> None:
    true = HullWhite(0.08, 0.012, curve)
    caps, swps = _quotes(true)
    cal = calibrate_hullwhite(curve, caps, swps, a_fixed=0.08)
    assert cal.a_fixed and cal.a == 0.08
    assert cal.converged and cal.identified
    assert cal.sigma == pytest.approx(0.012, abs=1e-9)
    warm = calibrate_hullwhite(curve, caps, swps, x0=(0.2, 0.02))
    assert warm.n_starts == 1 and not warm.identified
    assert warm.converged
    assert warm.a == pytest.approx(0.08, abs=1e-6)
    assert warm.sigma == pytest.approx(0.012, abs=1e-8)
    # Caplets only, swaptions only.
    only_caps = calibrate_hullwhite(curve, caps)
    assert only_caps.sigma == pytest.approx(0.012, abs=1e-8)
    only_swp = calibrate_hullwhite(curve, swaptions=swps, a_fixed=0.08)
    assert only_swp.sigma == pytest.approx(0.012, abs=1e-8)


def test_calibrate_hullwhite_single_quote_not_identified(curve: DiscountCurve) -> None:
    true = HullWhite(0.08, 0.012, curve)
    caps, _ = _quotes(true)
    cal = calibrate_hullwhite(curve, caps[:1])
    assert cal.converged
    assert not cal.identified  # one price, two parameters: a ridge of solutions
    assert cal.a_spread > 1e-2
    assert cal.rmse < 1e-9


def test_calibrate_hullwhite_non_convergence_and_validation(curve: DiscountCurve) -> None:
    true = HullWhite(0.08, 0.012, curve)
    caps, swps = _quotes(true)
    res = calibrate_hullwhite(curve, caps, swps, maxiter=2)
    assert not res.converged and math.isfinite(res.rmse)
    with pytest.raises(ValueError):
        calibrate_hullwhite(curve)  # no quotes
    with pytest.raises(ValueError):
        calibrate_hullwhite(curve, caps, swps, a_fixed=0.0)
    with pytest.raises(ValueError):
        calibrate_hullwhite(curve, caps, swps, x0=(0.1,))
    with pytest.raises(ValueError):
        calibrate_hullwhite(curve, caps, swps, x0=(float("nan"), 0.01))
    with pytest.raises(ValueError):
        calibrate_hullwhite("curve", caps)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CapletQuote(1.0, 2.0, 0.02, float("nan"))
    with pytest.raises(ValueError):
        CapletQuote(2.0, 1.0, 0.02, 0.001)
    with pytest.raises(ValueError):
        CapletQuote(1.0, 2.0, 0.02, -0.001)
    with pytest.raises(ValueError):
        SwaptionQuote(1.0, (0.5, 2.0), 0.02, 0.001)
    with pytest.raises(ValueError):
        SwaptionQuote(1.0, (), 0.02, 0.001)
    with pytest.raises(ValueError):
        SwaptionQuote(float("inf"), (2.0,), 0.02, 0.001)
