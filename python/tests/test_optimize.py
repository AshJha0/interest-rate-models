"""Native Nelder-Mead, cross-checked against scipy (tests only)."""

from __future__ import annotations

import pytest
import scipy.optimize as sopt

from irm import nelder_mead


def _rosenbrock(p) -> float:
    x, y = p
    return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2


def test_quadratic_bowl() -> None:
    res = nelder_mead(lambda p: (p[0] - 2.0) ** 2 + (p[1] + 1.0) ** 2, [0.0, 0.0])
    assert res.converged
    assert res.x[0] == pytest.approx(2.0, abs=1e-6)
    assert res.x[1] == pytest.approx(-1.0, abs=1e-6)
    assert res.fx == pytest.approx(0.0, abs=1e-12)


def test_rosenbrock_vs_scipy() -> None:
    ours = nelder_mead(_rosenbrock, [-1.2, 1.0], maxiter=10_000)
    ref = sopt.minimize(
        _rosenbrock, [-1.2, 1.0], method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 10_000},
    )
    assert ours.converged
    assert ours.x[0] == pytest.approx(1.0, abs=1e-5)
    assert ours.x[1] == pytest.approx(1.0, abs=1e-5)
    assert ours.fx == pytest.approx(float(ref.fun), abs=1e-9)


def test_non_convergence_reported() -> None:
    res = nelder_mead(_rosenbrock, [-1.2, 1.0], maxiter=3)
    assert not res.converged
    assert res.iterations == 3


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        nelder_mead(lambda p: 0.0, [])
    with pytest.raises(ValueError):
        nelder_mead(lambda p: p[0], [float("nan")])
