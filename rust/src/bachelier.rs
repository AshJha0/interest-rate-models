//! Bachelier (normal) model for caplets and swaptions: price and implied vol.
//!
//! Rates desks quote cap/floor and swaption prices as *normal* ("bp")
//! volatilities: the volatility `sigma_N` such that the Bachelier formula
//! reproduces the price.  For a payer option on a forward `F` with strike
//! `K`, expiry `T` and annuity `A` (`A = tau DF(T_pay)` for a caplet,
//! `sum_i tau_i DF(u_i)` for a swaption):
//!
//! `price = A [ (F - K) N(d) + sigma_N sqrt(T) phi(d) ]`,
//! `d = (F - K) / (sigma_N sqrt(T))`
//!
//! and the receiver is `A [ (K - F) N(-d) + sigma_N sqrt(T) phi(d) ]`.
//! With `sigma_N = 0` both collapse to discounted intrinsic.  The price is
//! strictly increasing in `sigma_N`, so the implied volatility is unique and
//! found with the native Brent solver on an expanding bracket.  Negative
//! forwards and strikes are first-class.

use crate::error::{IrmError, Result};
use crate::mathutils::{norm_cdf, norm_pdf};
use crate::rootfind::{brentq, BRENT_MAXITER, BRENT_RTOL};

/// Absolute price tolerance for deciding a quote is at intrinsic (vol 0).
pub const INTRINSIC_TOL: f64 = 1e-15;
/// Root tolerance on the implied volatility.
pub const IMPLIED_VOL_XTOL: f64 = 1e-14;
const VOL_BRACKET_START: f64 = 0.01;
const VOL_BRACKET_DOUBLINGS: usize = 40;

fn check_common(forward: f64, strike: f64, expiry: f64, annuity: f64) -> Result<()> {
    if !forward.is_finite() || !strike.is_finite() {
        return Err(IrmError::InvalidInput(
            "bachelier: forward and strike must be finite".into(),
        ));
    }
    if !expiry.is_finite() || expiry <= 0.0 {
        return Err(IrmError::InvalidInput(format!(
            "bachelier: expiry must be finite and > 0, got {expiry}"
        )));
    }
    if !annuity.is_finite() || annuity <= 0.0 {
        return Err(IrmError::InvalidInput(format!(
            "bachelier: annuity must be finite and > 0, got {annuity}"
        )));
    }
    Ok(())
}

/// Normal-model option price (per unit notional, annuity included).
///
/// `vol` is the absolute (normal) volatility in rate units per sqrt(year):
/// `0.0075` means 75 bp.  `vol = 0` returns discounted intrinsic.
pub fn bachelier_price(
    forward: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    annuity: f64,
    payer: bool,
) -> Result<f64> {
    check_common(forward, strike, expiry, annuity)?;
    if !vol.is_finite() || vol < 0.0 {
        return Err(IrmError::InvalidInput(format!(
            "bachelier: vol must be finite and >= 0, got {vol}"
        )));
    }
    let moneyness = if payer { forward - strike } else { strike - forward };
    if vol == 0.0 {
        return Ok(annuity * moneyness.max(0.0));
    }
    let sd = vol * expiry.sqrt();
    let d = moneyness / sd;
    Ok(annuity * (moneyness * norm_cdf(d) + sd * norm_pdf(d)))
}

/// Normal implied volatility of a payer/receiver price.
///
/// `price` must be finite and at least the discounted intrinsic
/// `annuity * max(+-(F - K), 0)`; a price within [`INTRINSIC_TOL`] of
/// intrinsic returns `0.0`.  Otherwise Brent's method solves
/// `bachelier_price(vol) = price` on `[0, hi]` with `hi` found by doubling
/// from `0.01` (at most 40 doublings).  Root tolerance `1e-14` on the vol.
pub fn bachelier_implied_vol(
    price: f64,
    forward: f64,
    strike: f64,
    expiry: f64,
    annuity: f64,
    payer: bool,
) -> Result<f64> {
    check_common(forward, strike, expiry, annuity)?;
    if !price.is_finite() {
        return Err(IrmError::InvalidInput(format!(
            "bachelier: price must be finite, got {price}"
        )));
    }
    let intrinsic = bachelier_price(forward, strike, expiry, 0.0, annuity, payer)?;
    if price < intrinsic - INTRINSIC_TOL {
        return Err(IrmError::InvalidInput(format!(
            "bachelier: price {price} below intrinsic {intrinsic} — no implied vol"
        )));
    }
    if price <= intrinsic + INTRINSIC_TOL {
        return Ok(0.0);
    }
    let diff = |v: f64| -> Result<f64> {
        Ok(bachelier_price(forward, strike, expiry, v, annuity, payer)? - price)
    };
    let mut hi = VOL_BRACKET_START;
    let mut bracketed = false;
    for _ in 0..VOL_BRACKET_DOUBLINGS {
        if diff(hi)? > 0.0 {
            bracketed = true;
            break;
        }
        hi *= 2.0;
    }
    if !bracketed {
        return Err(IrmError::InvalidInput(format!(
            "bachelier: price {price} too large to imply a volatility"
        )));
    }
    brentq(diff, 0.0, hi, IMPLIED_VOL_XTOL, BRENT_RTOL, BRENT_MAXITER)
}
