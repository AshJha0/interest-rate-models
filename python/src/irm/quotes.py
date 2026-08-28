"""CSV loaders for the bundled sample market data (see ../data).

File formats (headers included, plain ASCII, comma-separated):

* ``curve_quotes.csv``: ``curve_id,instrument,start,maturity,rate``
  where ``instrument`` is ``deposit``, ``fra`` or ``swap``; ``start`` is
  used only for FRAs (0 otherwise); times in ACT/365F years, rates decimal.
* ``ois_quotes.csv``: ``maturity,rate`` (OIS par swap quotes).
* ``zero_yields.csv``: ``maturity,yield`` (Vasicek calibration targets).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple, Union

from .bootstrap import FRA, Deposit, Instrument, OISSwap, Swap

__all__ = ["load_curve_quotes", "load_ois_quotes", "load_zero_yields"]

PathLike = Union[str, Path]


def load_curve_quotes(path: PathLike, curve_id: str) -> List[Instrument]:
    """Load the instruments of one curve, sorted by pillar time."""
    out: List[Instrument] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["curve_id"] != curve_id:
                continue
            kind = row["instrument"]
            rate = float(row["rate"])
            maturity = float(row["maturity"])
            if kind == "deposit":
                out.append(Deposit(maturity=maturity, rate=rate))
            elif kind == "fra":
                out.append(FRA(start=float(row["start"]), end=maturity, rate=rate))
            elif kind == "swap":
                out.append(Swap(maturity=maturity, rate=rate))
            else:
                raise ValueError(f"unknown instrument type {kind!r} in {path}")
    if not out:
        raise ValueError(f"no quotes found for curve_id={curve_id!r} in {path}")
    out.sort(key=lambda ins: ins.pillar)
    return out


def load_ois_quotes(path: PathLike) -> List[OISSwap]:
    """Load OIS par swap quotes, sorted by maturity."""
    out: List[OISSwap] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(OISSwap(maturity=float(row["maturity"]), rate=float(row["rate"])))
    if not out:
        raise ValueError(f"no OIS quotes in {path}")
    out.sort(key=lambda ins: ins.pillar)
    return out


def load_zero_yields(path: PathLike) -> Tuple[List[float], List[float]]:
    """Load (maturities, yields) calibration targets."""
    ts: List[float] = []
    ys: List[float] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            ts.append(float(row["maturity"]))
            ys.append(float(row["yield"]))
    if not ts:
        raise ValueError(f"no zero yields in {path}")
    return ts, ys
