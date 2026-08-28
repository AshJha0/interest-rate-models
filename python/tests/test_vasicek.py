"""Vasicek: closed forms, moments, options, exact MC, calibration."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.stats as sstats

from irm import Vasicek, calibrate_vasicek
from irm.mathutils import norm_cdf

BASE = dict(kappa=0.5, theta=0.03, sigma=0.01, r0=0.02)


@pytest.fixture()
def model() -> Vasicek:
    return Vasicek(**BASE)


def test_norm_cdf_vs_scipy() -> None:
    for x in [-8.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 8.0]:
        assert norm_cdf(x) == pytest.approx(float(sstats.norm.cdf(x)), abs=1e-14)


def test_b_and_a_factors(model: Vasicek) -> None:
    k, th, s = BASE["kappa"], BASE["theta"], BASE["sigma"]
    tau = 5.0
    b = (1.0 - math.exp(-k * tau)) / k
    assert model.b_factor(tau) == pytest.approx(b, abs=1e-15)
    a = math.exp((th - s * s / (2 * k * k)) * (b - tau) - s * s * b * b / (4 * k))
    assert model.a_factor(tau) == pytest.approx(a, rel=1e-14)
    assert model.zcb_price(tau) == pytest.approx(a * math.exp(-b * BASE["r0"]), rel=1e-14)


def test_zcb_degenerate_and_yield(model: Vasicek) -> None:
    assert model.zcb_price(0.0) == 1.0
    assert model.zero_yield(5.0) == pytest.approx(-math.log(model.zcb_price(5.0)) / 5.0)
    with pytest.raises(ValueError):
        model.zero_yield(0.0)
    with pytest.raises(ValueError):
        model.zcb_price(1.0, t=2.0)


def test_negative_rates_supported() -> None:
    m = Vasicek(kappa=0.3, theta=-0.01, sigma=0.008, r0=-0.005)
    p = m.zcb_price(1.0)
    assert p > 1.0  # negative rates -> ZCB above par
    assert m.zero_yield(1.0) < 0.0


def test_rt_moments_closed_form(model: Vasicek) -> None:
    k, th, s, r0 = BASE["kappa"], BASE["theta"], BASE["sigma"], BASE["r0"]
    t = 2.0
    assert model.r_mean(t) == pytest.approx(th + (r0 - th) * math.exp(-k * t), abs=1e-15)
    assert model.r_var(t) == pytest.approx(s * s * (1 - math.exp(-2 * k * t)) / (2 * k), abs=1e-18)
    assert model.r_var(0.0) == 0.0


def test_rt_moments_vs_simulation(model: Vasicek) -> None:
    n = 60_000
    r_end, _ = model.simulate(2.0, n_steps=4, n_paths=n, seed=3)
    se_mean = math.sqrt(model.r_var(2.0) / n)
    assert abs(float(np.mean(r_end)) - model.r_mean(2.0)) < 3.0 * se_mean
    assert float(np.var(r_end, ddof=1)) == pytest.approx(model.r_var(2.0), rel=0.05)


def test_mc_zcb_within_3se(model: Vasicek) -> None:
    analytic = model.zcb_price(5.0)
    price, se = model.mc_zcb(5.0, n_steps=4, n_paths=50_000, seed=12)
    assert se > 0.0
    assert abs(price - analytic) < 3.0 * se
    # Exact scheme: no discretisation bias — 1 step must also agree.
    price1, se1 = model.mc_zcb(5.0, n_steps=1, n_paths=50_000, seed=13)
    assert abs(price1 - analytic) < 3.0 * se1


def test_sigma_zero_mc_is_deterministic() -> None:
    m = Vasicek(kappa=0.5, theta=0.03, sigma=0.0, r0=0.02)
    price, se = m.mc_zcb(5.0, n_steps=16, n_paths=100, seed=1)
    assert se == pytest.approx(0.0, abs=1e-16)
    assert price == pytest.approx(m.zcb_price(5.0), rel=1e-13)


def test_bond_option_put_call_parity_grid(model: Vasicek) -> None:
    # Property-style: ZBC - ZBP = P(0,S) - K P(0,T) across strikes/expiries.
    for expiry, s_mat in [(0.5, 2.0), (1.0, 3.0), (2.0, 7.0)]:
        for k in [0.80, 0.90, 0.95, 1.00, 1.05]:
            lhs = model.zbc(expiry, s_mat, k) - model.zbp(expiry, s_mat, k)
            rhs = model.zcb_price(s_mat) - k * model.zcb_price(expiry)
            assert lhs == pytest.approx(rhs, abs=1e-13)


def test_bond_option_edges(model: Vasicek) -> None:
    # Deep ITM call ~ forward minus strike; deep OTM ~ 0.
    assert model.zbc(1.0, 3.0, 1e-8) == pytest.approx(
        model.zcb_price(3.0) - 1e-8 * model.zcb_price(1.0), rel=1e-9
    )
    assert model.zbc(1.0, 3.0, 10.0) == pytest.approx(0.0, abs=1e-12)
    # expiry = 0 and sigma = 0 collapse to intrinsic.
    assert model.zbc(0.0, 3.0, 0.9) == pytest.approx(
        max(model.zcb_price(3.0) - 0.9, 0.0), abs=1e-15
    )
    m0 = Vasicek(kappa=0.5, theta=0.03, sigma=0.0, r0=0.02)
    assert m0.zbp(1.0, 3.0, 0.99) == pytest.approx(
        max(0.99 * m0.zcb_price(1.0) - m0.zcb_price(3.0), 0.0), abs=1e-15
    )


def test_option_positivity_and_monotonicity(model: Vasicek) -> None:
    strikes = [0.85, 0.90, 0.95, 1.00]
    calls = [model.zbc(1.0, 3.0, k) for k in strikes]
    for c in calls:
        assert c >= 0.0
    for a, b in zip(calls[:-1], calls[1:]):
        assert b <= a + 1e-15  # call value decreasing in strike


def test_parameter_validation() -> None:
    with pytest.raises(ValueError):
        Vasicek(kappa=0.0, theta=0.03, sigma=0.01, r0=0.02)
    with pytest.raises(ValueError):
        Vasicek(kappa=-0.5, theta=0.03, sigma=0.01, r0=0.02)
    with pytest.raises(ValueError):
        Vasicek(kappa=0.5, theta=0.03, sigma=-0.01, r0=0.02)
    with pytest.raises(ValueError):
        Vasicek(kappa=0.5, theta=float("nan"), sigma=0.01, r0=0.02)
    m = Vasicek(**BASE)
    with pytest.raises(ValueError):
        m.zbc(1.0, 3.0, 0.0)  # strike must be > 0
    with pytest.raises(ValueError):
        m.zbc(3.0, 1.0, 0.9)  # bond matures before expiry
    with pytest.raises(ValueError):
        m.simulate(1.0, 0, 10, seed=1)
    with pytest.raises(ValueError):
        m.b_factor(-1.0)


def test_calibration_recovers_parameters() -> None:
    true = Vasicek(kappa=0.8, theta=0.05, sigma=0.02, r0=0.03)
    ts = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0]
    ys = [true.zero_yield(t) for t in ts]  # noise-free targets
    res = calibrate_vasicek(ts, ys, r0=0.03)
    assert res.converged
    assert res.rmse < 1e-7
    assert res.kappa == pytest.approx(0.8, abs=2e-3)
    assert res.theta == pytest.approx(0.05, abs=2e-3)
    # sigma has weak curvature influence -> looser bound.
    assert res.sigma == pytest.approx(0.02, abs=5e-3)
    refit = res.model()
    for t, y in zip(ts, ys):
        assert refit.zero_yield(t) == pytest.approx(y, abs=1e-6)


def test_calibration_from_csv(data_dir) -> None:
    from irm import load_zero_yields

    ts, ys = load_zero_yields(data_dir / "zero_yields.csv")
    res = calibrate_vasicek(ts, ys, r0=0.03)
    assert res.converged
    assert res.rmse < 1e-4
    assert res.kappa == pytest.approx(0.8, rel=0.2)


def test_calibration_non_convergence_reported_not_raised() -> None:
    true = Vasicek(kappa=0.8, theta=0.05, sigma=0.02, r0=0.03)
    ts = [1.0, 2.0, 5.0, 10.0]
    ys = [true.zero_yield(t) for t in ts]
    res = calibrate_vasicek(ts, ys, r0=0.03, maxiter=2)
    assert not res.converged  # reported, no exception
    assert math.isfinite(res.rmse)


def test_calibration_input_validation() -> None:
    with pytest.raises(ValueError):
        calibrate_vasicek([1.0, 2.0], [0.02, 0.03], r0=0.02)  # too few points
    with pytest.raises(ValueError):
        calibrate_vasicek([1.0, -2.0, 3.0], [0.02, 0.03, 0.03], r0=0.02)
    with pytest.raises(ValueError):
        calibrate_vasicek([1.0, 2.0, 3.0], [0.02, float("nan"), 0.03], r0=0.02)
