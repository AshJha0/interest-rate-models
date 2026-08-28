"""irm — interest rate models: curve bootstrap, Vasicek, Hull-White 1F.

Reference Python implementation for project P04.  See API_SPEC.md at the
project root for the precise, language-neutral contract every port follows.
"""

from .bootstrap import FRA, Deposit, Instrument, OISSwap, Swap, annual_schedule, bootstrap
from .curve import DiscountCurve, par_swap_rate
from .hullwhite import FD_STEP, HullWhite, JamshidianResult
from .mathutils import norm_cdf, ou_step_moments
from .optimize import NelderMeadResult, nelder_mead
from .quotes import load_curve_quotes, load_ois_quotes, load_zero_yields
from .rootfind import bisect, brentq
from .vasicek import Vasicek, VasicekCalibration, calibrate_vasicek

__version__ = "1.0.0"

__all__ = [
    "DiscountCurve",
    "par_swap_rate",
    "Deposit",
    "FRA",
    "Swap",
    "OISSwap",
    "Instrument",
    "annual_schedule",
    "bootstrap",
    "Vasicek",
    "VasicekCalibration",
    "calibrate_vasicek",
    "HullWhite",
    "JamshidianResult",
    "FD_STEP",
    "brentq",
    "bisect",
    "nelder_mead",
    "NelderMeadResult",
    "norm_cdf",
    "ou_step_moments",
    "load_curve_quotes",
    "load_ois_quotes",
    "load_zero_yields",
    "__version__",
]
