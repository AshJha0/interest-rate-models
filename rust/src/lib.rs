//! # `irm` — interest rate models (P04 Rust port)
//!
//! Curve bootstrapping (deposits / FRAs / par swaps / OIS), the Vasicek
//! short-rate model and the Hull-White one-factor model, ported from the
//! validated Python reference implementation under `../python/src/irm`.
//!
//! Conventions (see `../API_SPEC.md`, the normative contract):
//!
//! * Day count ACT/365F everywhere; times are year fractions `days/365`.
//! * Rates are decimals; continuous compounding for zero rates, simple
//!   compounding for deposits/FRAs/forwards, annual fixed swap legs.
//! * Negative rates are first-class: discount factors above 1 are legal
//!   (only `DF > 0` is required); both models are Gaussian precisely so
//!   that `r < 0` works.
//! * Invalid input returns `Err(IrmError)` — no panics; calibration
//!   non-convergence is reported via a `converged` flag, never an error.

pub mod bootstrap;
pub mod curve;
pub mod error;
pub mod hullwhite;
pub mod mathutils;
pub mod optimize;
pub mod quotes;
pub mod rootfind;
pub mod vasicek;

pub use bootstrap::{annual_schedule, bootstrap, Instrument};
pub use curve::{par_swap_rate, DiscountCurve};
pub use error::{IrmError, Result};
pub use hullwhite::{HullWhite, JamshidianResult, FD_STEP};
pub use mathutils::{erfc, norm_cdf, ou_step_moments, OuStepMoments};
pub use optimize::{nelder_mead, NelderMeadResult};
pub use quotes::{load_curve_quotes, load_ois_quotes, load_zero_yields};
pub use rootfind::{bisect, brentq, BRENT_MAXITER, BRENT_RTOL, BRENT_XTOL};
pub use vasicek::{calibrate_vasicek, Vasicek, VasicekCalibration};
