"""DiscountCurve: interpolation, extrapolation, rates, validation."""

from __future__ import annotations

import math

import pytest

from irm import DiscountCurve, par_swap_rate

TIMES = [1.0, 2.0, 3.0, 5.0]
DFS = [0.99, 0.975, 0.955, 0.90]


@pytest.fixture()
def curve() -> DiscountCurve:
    return DiscountCurve(TIMES, DFS)


def test_pillar_exactness(curve: DiscountCurve) -> None:
    for t, p in zip(TIMES, DFS):
        assert curve.df(t) == pytest.approx(p, abs=1e-15)
    assert curve.df(0.0) == 1.0


def test_log_linear_midpoint(curve: DiscountCurve) -> None:
    # Linear in ln DF: midpoint is the geometric mean of neighbours.
    expected = math.exp(0.5 * (math.log(0.975) + math.log(0.955)))
    assert curve.df(2.5) == pytest.approx(expected, abs=1e-15)
    # First segment interpolates against the implicit (0, 1) node.
    expected05 = math.exp(0.5 * math.log(0.99))
    assert curve.df(0.5) == pytest.approx(expected05, abs=1e-15)


def test_flat_forward_extrapolation(curve: DiscountCurve) -> None:
    f_last = -(math.log(0.90) - math.log(0.955)) / (5.0 - 3.0)
    assert curve.inst_forward(10.0) == pytest.approx(f_last, abs=1e-15)
    assert curve.df(7.0) == pytest.approx(0.90 * math.exp(-f_last * 2.0), rel=1e-14)


def test_zero_rate_definition(curve: DiscountCurve) -> None:
    for t in [0.5, 1.0, 2.5, 4.0, 6.0]:
        assert curve.zero_rate(t) == pytest.approx(-math.log(curve.df(t)) / t, abs=1e-15)
    # At t = 0 the limit equals the first segment's instantaneous forward.
    assert curve.zero_rate(0.0) == pytest.approx(-math.log(0.99) / 1.0, abs=1e-15)


def test_piecewise_constant_forwards(curve: DiscountCurve) -> None:
    # Instantaneous forward is constant inside each segment...
    assert curve.inst_forward(2.2) == pytest.approx(curve.inst_forward(2.8), abs=1e-15)
    # ...equals the log-DF slope of the segment...
    f_23 = -(math.log(0.955) - math.log(0.975)) / 1.0
    assert curve.inst_forward(2.5) == pytest.approx(f_23, abs=1e-14)
    # ...and at a pillar the right segment's forward is used.
    assert curve.inst_forward(2.0) == pytest.approx(f_23, abs=1e-14)


def test_forward_rate_consistency(curve: DiscountCurve) -> None:
    # Property-style grid: F(t1,t2) always reproduces the DF ratio.
    grid = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    for i, t1 in enumerate(grid):
        for t2 in grid[i + 1 :]:
            f = curve.fwd_rate(t1, t2)
            assert (1.0 + f * (t2 - t1)) * curve.df(t2) == pytest.approx(
                curve.df(t1), rel=1e-13
            )


def test_positive_curve_df_decreasing(curve: DiscountCurve) -> None:
    ts = [0.1 * k for k in range(1, 71)]
    for a, b in zip(ts[:-1], ts[1:]):
        assert curve.df(b) < curve.df(a)


def test_negative_rates_df_above_one() -> None:
    # EUR/JPY style: DF > 1 is explicitly supported.
    neg = DiscountCurve([1.0, 2.0], [1.004, 1.006])
    assert neg.df(1.0) > 1.0
    assert neg.df(1.5) > 1.0
    assert neg.zero_rate(2.0) < 0.0
    assert neg.inst_forward(0.5) < 0.0


def test_single_pillar_curve() -> None:
    c = DiscountCurve([2.0], [0.96])
    assert c.df(2.0) == pytest.approx(0.96)
    assert c.df(1.0) == pytest.approx(math.sqrt(0.96), rel=1e-14)
    f = -math.log(0.96) / 2.0
    assert c.df(4.0) == pytest.approx(0.96 * math.exp(-2.0 * f), rel=1e-14)


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="at least one pillar"):
        DiscountCurve([], [])
    with pytest.raises(ValueError, match="length mismatch"):
        DiscountCurve([1.0, 2.0], [0.9])
    with pytest.raises(ValueError, match="strictly increasing"):
        DiscountCurve([1.0, 1.0], [0.9, 0.8])  # duplicate pillar
    with pytest.raises(ValueError, match="strictly increasing"):
        DiscountCurve([2.0, 1.0], [0.9, 0.95])  # non-monotone
    with pytest.raises(ValueError, match="strictly increasing"):
        DiscountCurve([0.0, 1.0], [1.0, 0.9])  # t = 0 pillar not allowed
    with pytest.raises(ValueError, match="> 0"):
        DiscountCurve([1.0], [0.0])
    with pytest.raises(ValueError, match="> 0"):
        DiscountCurve([1.0], [-0.5])
    with pytest.raises(ValueError):
        DiscountCurve([float("nan")], [0.9])
    with pytest.raises(ValueError):
        DiscountCurve([1.0], [float("inf")])


def test_query_validation(curve: DiscountCurve) -> None:
    with pytest.raises(ValueError):
        curve.df(-0.5)
    with pytest.raises(ValueError):
        curve.df(float("nan"))
    with pytest.raises(ValueError):
        curve.zero_rate(-1.0)
    with pytest.raises(ValueError):
        curve.fwd_rate(2.0, 2.0)
    with pytest.raises(ValueError):
        curve.fwd_rate(-1.0, 2.0)


def test_par_swap_rate(curve: DiscountCurve) -> None:
    times = [0.0, 1.0, 2.0, 3.0]
    annuity = sum(curve.df(t) for t in [1.0, 2.0, 3.0])
    expected = (1.0 - curve.df(3.0)) / annuity
    assert par_swap_rate(curve, times) == pytest.approx(expected, rel=1e-14)
    with pytest.raises(ValueError):
        par_swap_rate(curve, [0.0])
    with pytest.raises(ValueError):
        par_swap_rate(curve, [0.0, 2.0, 1.0])


def test_fwd_rate_underflow_is_standard_error() -> None:
    c = DiscountCurve([1.0], [0.95])
    with pytest.raises(ValueError, match="not representable"):
        c.fwd_rate(0.0, 1e6)  # DF(1e6) underflows to 0
    # Large but representable horizons still work.
    assert math.isfinite(c.fwd_rate(0.0, 500.0))
