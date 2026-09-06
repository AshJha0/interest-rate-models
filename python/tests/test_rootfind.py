"""Native root finders, cross-checked against scipy (tests only)."""

from __future__ import annotations

import math

import pytest
import scipy.optimize as sopt

from irm import bisect, brentq


def test_brentq_transcendental_vs_scipy() -> None:
    f = lambda x: math.cos(x) - x  # noqa: E731
    ours = brentq(f, 0.0, 1.0)
    ref = sopt.brentq(f, 0.0, 1.0, xtol=1e-14)
    assert abs(ours - ref) < 1e-12
    assert abs(f(ours)) < 1e-14


def test_brentq_polynomial_grid_property() -> None:
    # Property-style: cube roots across a grid of targets.
    for a in [0.1, 0.5, 1.0, 2.0, 5.0, 27.0, 100.0]:
        root = brentq(lambda x: x**3 - a, 0.0, 10.0)
        assert abs(root - a ** (1.0 / 3.0)) < 1e-11


def test_brentq_endpoint_roots() -> None:
    assert brentq(lambda x: x, 0.0, 1.0) == 0.0
    assert brentq(lambda x: x - 1.0, 0.0, 1.0) == 1.0


def test_brentq_not_bracketed_raises() -> None:
    with pytest.raises(ValueError, match="not bracketed"):
        brentq(lambda x: x * x + 1.0, -1.0, 1.0)


def test_brentq_bad_inputs() -> None:
    with pytest.raises(ValueError):
        brentq(lambda x: x, float("nan"), 1.0)
    with pytest.raises(ValueError):
        brentq(lambda x: x, -1.0, 1.0, xtol=0.0)


def test_bisect_matches_brentq() -> None:
    f = lambda x: math.exp(x) - 2.0  # noqa: E731
    assert abs(bisect(f, 0.0, 1.0) - math.log(2.0)) < 1e-10
    with pytest.raises(ValueError, match="not bracketed"):
        bisect(lambda x: 1.0, 0.0, 1.0)


def test_brentq_steep_function() -> None:
    # Nearly-flat then steep: exercises the bisection fallback branch.
    f = lambda x: math.tanh(50.0 * (x - 0.7))  # noqa: E731
    root = brentq(f, 0.0, 1.0)
    assert abs(root - 0.7) < 1e-12


def test_brentq_non_convergence_is_error() -> None:
    with pytest.raises(ValueError, match="no convergence"):
        brentq(lambda x: math.tanh(1e3 * (x - 0.123456)), -10.0, 10.0, maxiter=1)


def test_bisect_non_convergence_is_error() -> None:
    with pytest.raises(ValueError, match="no convergence"):
        bisect(lambda x: x - 0.3, 0.0, 1.0, xtol=1e-300, maxiter=5)
    with pytest.raises(ValueError):
        bisect(lambda x: x, 0.0, float("nan"))
    with pytest.raises(ValueError):
        bisect(lambda x: x, -1.0, 1.0, xtol=0.0)
    assert bisect(lambda x: x - 0.25, 0.0, 1.0) == pytest.approx(0.25, abs=1e-11)


def test_non_finite_function_values_rejected() -> None:
    # A NaN residual must never be returned as a "root" or misreported as a
    # bracketing failure.
    with pytest.raises(ValueError, match="not finite"):
        brentq(lambda x: float("nan") if 0.2 < x < 0.8 else (1.0 if x < 0.5 else -1.0), 0.0, 1.0)
    with pytest.raises(ValueError, match="not finite"):
        brentq(lambda x: float("nan"), 0.0, 1.0)
    with pytest.raises(ValueError, match="not finite"):
        bisect(lambda x: float("inf") if x > 0.4 else -1.0, 0.0, 1.0)
