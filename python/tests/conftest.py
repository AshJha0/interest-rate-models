"""Shared fixtures: paths to the bundled data and bootstrapped curves."""

from __future__ import annotations

from pathlib import Path

import pytest

from irm import DiscountCurve, bootstrap, load_curve_quotes, load_ois_quotes

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def eur_curve() -> DiscountCurve:
    return bootstrap(load_curve_quotes(DATA_DIR / "curve_quotes.csv", "EUR"))


@pytest.fixture(scope="session")
def usd_curve() -> DiscountCurve:
    return bootstrap(load_curve_quotes(DATA_DIR / "curve_quotes.csv", "USD"))


@pytest.fixture(scope="session")
def ois_curve() -> DiscountCurve:
    return bootstrap(load_ois_quotes(DATA_DIR / "ois_quotes.csv"))
