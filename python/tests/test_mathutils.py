"""Shared kernels: normal CDF/PDF and the exact OU step moments.

Reference values for the integrated-OU variance are computed here with
60-digit ``decimal`` arithmetic straight from the closed form, so the tests
do not copy numbers from the code under test.
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext

import pytest

from irm import norm_cdf, norm_pdf, ou_integral_variance, ou_step_moments
from irm.mathutils import OU_SERIES_THRESHOLD


def _ref_var_i(a: float, sigma: float, dt: float) -> float:
    getcontext().prec = 60
    a_, s_, dt_ = Decimal(repr(a)), Decimal(repr(sigma)), Decimal(repr(dt))
    x = a_ * dt_
    e1 = (-x).exp()
    e2 = (-2 * x).exp()
    return float((s_ * s_ / (a_ * a_)) * (dt_ - 2 * (1 - e1) / a_ + (1 - e2) / (2 * a_)))


def _ref_var_x(a: float, sigma: float, dt: float) -> float:
    getcontext().prec = 60
    a_, s_, dt_ = Decimal(repr(a)), Decimal(repr(sigma)), Decimal(repr(dt))
    return float(s_ * s_ * (1 - (-2 * a_ * dt_).exp()) / (2 * a_))


@pytest.mark.parametrize(
    "a,sigma,dt",
    [
        (1e-3, 0.01, 0.01),
        (1e-4, 0.01, 1e-3),
        (1e-6, 0.02, 0.125),
        (1e-2, 0.01, 1.0),   # x = 1e-2: exactly at the series threshold
        (0.05, 0.02, 0.01),
        (0.1, 0.01, 0.125),
        (0.8, 0.02, 2.5),
        (5.0, 0.01, 1.0),
    ],
)
def test_ou_moments_small_a_dt_accurate(a: float, sigma: float, dt: float) -> None:
    m = ou_step_moments(a, sigma, dt)
    ref_i = _ref_var_i(a, sigma, dt)
    ref_x = _ref_var_x(a, sigma, dt)
    assert m.var_i == pytest.approx(ref_i, rel=1e-9)
    assert m.var_x == pytest.approx(ref_x, rel=1e-12)
    assert m.b == pytest.approx(-math.expm1(-a * dt) / a, rel=1e-14)
    # Cauchy-Schwarz: the (x', I) covariance matrix must be PSD.
    assert m.cov * m.cov <= m.var_x * m.var_i * (1.0 + 1e-12)
    # The reviewer's 4-term series is what the fix promised for tiny x.
    x = a * dt
    if x < 1e-3:
        series = sigma**2 * dt**3 * (1 / 3 - x / 4 + 7 * x**2 / 60 - x**3 / 24)
        assert m.var_i == pytest.approx(series, rel=1e-9)


def test_naive_formula_is_wrong_for_tiny_a_dt() -> None:
    # Documents the bug the series fixes: plain double evaluation is off by
    # orders of magnitude at a*dt = 1e-4 (80x per the review measurement).
    a, sigma, dt = 1e-4, 0.01, 1e-3
    decay = math.exp(-a * dt)
    b = (1.0 - decay) / a
    naive = (sigma**2 / a**2) * (dt - 2.0 * b + (1.0 - decay * decay) / (2.0 * a))
    ref = _ref_var_i(a, sigma, dt)
    assert abs(naive - ref) / ref > 10.0
    assert ou_integral_variance(a, sigma, dt) == pytest.approx(ref, rel=1e-12)


def test_series_and_closed_form_agree_at_threshold() -> None:
    x = OU_SERIES_THRESHOLD
    below = ou_integral_variance(x * (1 - 1e-12), 0.01, 1.0)
    above = ou_integral_variance(x * (1 + 1e-12), 0.01, 1.0)
    assert below == pytest.approx(above, rel=1e-9)


def test_ou_moments_validation() -> None:
    with pytest.raises(ValueError):
        ou_step_moments(0.0, 0.01, 0.1)
    with pytest.raises(ValueError):
        ou_step_moments(0.1, 0.01, 0.0)
    with pytest.raises(ValueError):
        ou_step_moments(float("nan"), 0.01, 0.1)
    with pytest.raises(ValueError):
        ou_step_moments(0.1, -0.01, 0.1)
    with pytest.raises(ValueError):
        ou_integral_variance(0.1, float("inf"), 0.1)
    # sigma = 0: all variances exactly zero, decay/b still defined.
    m = ou_step_moments(0.3, 0.0, 0.5)
    assert m.var_x == 0.0 and m.var_i == 0.0 and m.cov == 0.0
    assert m.decay == pytest.approx(math.exp(-0.15), abs=1e-16)


def test_norm_cdf_and_pdf() -> None:
    assert norm_cdf(0.0) == 0.5
    assert norm_cdf(1.0) == pytest.approx(0.8413447460685429, abs=1e-15)
    assert norm_cdf(-8.0) == pytest.approx(6.220960574271786e-16, rel=1e-10)
    assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi), abs=1e-16)
    assert norm_pdf(2.0) == pytest.approx(math.exp(-2.0) / math.sqrt(2.0 * math.pi), rel=1e-15)
