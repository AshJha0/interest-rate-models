"""irm — interest rate models: curve bootstrap, Vasicek, Hull-White 1F.

Reference Python implementation of the interest-rate-models repository.  See
API_SPEC.md at the project root for the precise, language-neutral contract every port follows.
"""

from .bootstrap import (
    FRA,
    MAX_MATURITY,
    MIN_MATURITY,
    Deposit,
    Instrument,
    OISSwap,
    Swap,
    annual_schedule,
    bootstrap,
    solve_pillar_df,
)
from .curve import DiscountCurve, par_swap_rate
from .bachelier import bachelier_implied_vol, bachelier_price
from .hullwhite import (
    FD_STEP,
    JAMSHIDIAN_RESIDUAL_TOL,
    CapletQuote,
    HullWhite,
    HullWhiteCalibration,
    JamshidianResult,
    SwaptionQuote,
    calibrate_hullwhite,
)
from .mathutils import norm_cdf, norm_pdf, ou_integral_variance, ou_step_moments
from .optimize import NelderMeadResult, nelder_mead
from .quotes import load_curve_quotes, load_ois_quotes, load_zero_yields
from .rootfind import bisect, brentq
from .vasicek import Vasicek, VasicekCalibration, calibrate_vasicek

__version__ = "1.1.0"

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
    "solve_pillar_df",
    "MIN_MATURITY",
    "MAX_MATURITY",
    "Vasicek",
    "VasicekCalibration",
    "calibrate_vasicek",
    "HullWhite",
    "JamshidianResult",
    "FD_STEP",
    "JAMSHIDIAN_RESIDUAL_TOL",
    "CapletQuote",
    "SwaptionQuote",
    "HullWhiteCalibration",
    "calibrate_hullwhite",
    "bachelier_price",
    "bachelier_implied_vol",
    "brentq",
    "bisect",
    "nelder_mead",
    "NelderMeadResult",
    "norm_cdf",
    "norm_pdf",
    "ou_step_moments",
    "ou_integral_variance",
    "load_curve_quotes",
    "load_ois_quotes",
    "load_zero_yields",
    "__version__",
]
