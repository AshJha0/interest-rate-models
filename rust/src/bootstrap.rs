//! Sequential single-curve bootstrap from deposits, FRAs and par swaps.
//!
//! # Framework
//!
//! Single-curve: the floating leg of a swap is projected off the same curve
//! it is discounted on, so its PV telescopes to `DF(0) - DF(T)`.  (Real
//! desks use multi-curve setups — OIS discounting plus tenor-specific
//! projection curves; the OIS variant below bootstraps such a discount
//! curve, but projection basis is out of scope and documented as a
//! simplification.)
//!
//! # Algorithm
//!
//! Instruments must be supplied with strictly increasing pillar (maturity)
//! times.  For each instrument in order, the discount factor at its pillar
//! is the single unknown `x`; all earlier dates interpolate on the
//! already-solved pillars (plus the trial pillar), so each pillar is found
//! by a 1-D root find of the instrument's *residual* (PV mispricing):
//!
//! * Deposit (simple interest): `residual = DF(T) (1 + R T) - 1`
//! * FRA:                       `residual = DF(T1) - (1 + R (T2-T1)) DF(T2)`
//! * Par swap / OIS swap:       `residual = R * annuity - (1 - DF(Tn))`
//!   with `annuity = sum_i tau_i DF(t_i)` over the annual fixed schedule.
//!
//! The root is found with the native Brent solver on the bracket
//! `DF in [1e-10, 100]` with `xtol = 1e-14`.  A bracket failure means no
//! positive discount factor can reprice the quote — i.e. crossed or
//! arbitrageable quotes (the "negative implied DF" case) — and yields
//! `IrmError::Bootstrap`.

use crate::curve::DiscountCurve;
use crate::error::{IrmError, Result};
use crate::rootfind::{brentq, BRENT_MAXITER, BRENT_RTOL};

const DF_LO: f64 = 1e-10;
const DF_HI: f64 = 100.0;

fn check_rate(rate: f64) -> Result<()> {
    if !rate.is_finite() {
        return Err(IrmError::InvalidInput(format!(
            "quote rate must be finite, got {rate}"
        )));
    }
    Ok(())
}

/// Annual fixed-leg payment times ending exactly at `maturity`.
///
/// `n = ceil(maturity)` payments at `maturity - (n-1), ..., maturity`;
/// e.g. `2.0 -> [1.0, 2.0]`; `2.5 -> [0.5, 1.5, 2.5]`; `0.25 -> [0.25]`.
/// A short first stub (< 1y) is created for non-integer maturities.
pub fn annual_schedule(maturity: f64) -> Result<Vec<f64>> {
    if !maturity.is_finite() || maturity <= 0.0 {
        return Err(IrmError::InvalidInput(format!(
            "maturity must be finite and > 0, got {maturity}"
        )));
    }
    let n = (maturity - 1e-12).ceil() as usize;
    Ok((0..n).map(|k| maturity - (n - 1 - k) as f64).collect())
}

/// Fixed-vs-floating swap residual shared by [`Instrument::Swap`] and
/// [`Instrument::OisSwap`]: `R * annuity - (1 - DF(T))`.
fn swap_residual(curve: &DiscountCurve, maturity: f64, rate: f64) -> Result<f64> {
    let mut annuity = 0.0;
    let mut prev = 0.0;
    for t in annual_schedule(maturity)? {
        annuity += (t - prev) * curve.df(t)?;
        prev = t;
    }
    Ok(rate * annuity - (1.0 - curve.df(maturity)?))
}

/// A bootstrap instrument (validated at construction).
///
/// * `Deposit`: cash deposit paying simple interest `R` at `maturity`.
/// * `Fra`: forward rate agreement fixing the simple forward over
///   `[start, end]`.
/// * `Swap`: par swap, annual fixed leg vs single-curve floating leg
///   (floating PV telescopes to `1 - DF(T)`).
/// * `OisSwap`: OIS par swap — daily compounding of the overnight rate is
///   approximated as continuous compounding at the curve's short rate,
///   under which the compounded floating coupon over `[t_{i-1}, t_i]` pays
///   `DF(t_{i-1})/DF(t_i) - 1` and the leg PV telescopes to `1 - DF(T)`,
///   identical in form to `Swap`.  The daily-vs-continuous gap is
///   `O(rate^2/365)` — documented approximation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Instrument {
    /// Cash deposit `(maturity, rate)`, simple interest.
    Deposit { maturity: f64, rate: f64 },
    /// FRA `(start, end, rate)`, simple forward.
    Fra { start: f64, end: f64, rate: f64 },
    /// Par swap `(maturity, rate)`, annual fixed leg.
    Swap { maturity: f64, rate: f64 },
    /// OIS par swap `(maturity, rate)`, annual fixed leg.
    OisSwap { maturity: f64, rate: f64 },
}

impl Instrument {
    /// Validated deposit; requires `T > 0` and `1 + R*T > 0`.
    pub fn deposit(maturity: f64, rate: f64) -> Result<Self> {
        check_rate(rate)?;
        if !maturity.is_finite() || maturity <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "deposit maturity must be > 0, got {maturity}"
            )));
        }
        if 1.0 + rate * maturity <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "deposit quote implies non-positive discount factor \
                 (1 + R*T = {})",
                1.0 + rate * maturity
            )));
        }
        Ok(Instrument::Deposit { maturity, rate })
    }

    /// Validated FRA; requires `0 <= start < end` and `1 + R*(end-start) > 0`.
    pub fn fra(start: f64, end: f64, rate: f64) -> Result<Self> {
        check_rate(rate)?;
        if !start.is_finite() || !end.is_finite() {
            return Err(IrmError::InvalidInput("FRA times must be finite".into()));
        }
        if start < 0.0 || end <= start {
            return Err(IrmError::InvalidInput(format!(
                "FRA needs 0 <= start < end, got start={start}, end={end}"
            )));
        }
        if 1.0 + rate * (end - start) <= 0.0 {
            return Err(IrmError::InvalidInput(
                "FRA quote implies non-positive discount factor".into(),
            ));
        }
        Ok(Instrument::Fra { start, end, rate })
    }

    /// Validated par swap with annual fixed leg.
    pub fn swap(maturity: f64, rate: f64) -> Result<Self> {
        check_rate(rate)?;
        annual_schedule(maturity)?; // validates maturity
        Ok(Instrument::Swap { maturity, rate })
    }

    /// Validated OIS par swap with annual fixed leg.
    pub fn ois_swap(maturity: f64, rate: f64) -> Result<Self> {
        check_rate(rate)?;
        annual_schedule(maturity)?;
        Ok(Instrument::OisSwap { maturity, rate })
    }

    /// The curve pillar this instrument determines (its end date).
    pub fn pillar(&self) -> f64 {
        match *self {
            Instrument::Deposit { maturity, .. } => maturity,
            Instrument::Fra { end, .. } => end,
            Instrument::Swap { maturity, .. } => maturity,
            Instrument::OisSwap { maturity, .. } => maturity,
        }
    }

    /// The quoted rate.
    pub fn rate(&self) -> f64 {
        match *self {
            Instrument::Deposit { rate, .. }
            | Instrument::Fra { rate, .. }
            | Instrument::Swap { rate, .. }
            | Instrument::OisSwap { rate, .. } => rate,
        }
    }

    /// PV mispricing of the instrument on `curve` (zero when repriced).
    pub fn residual(&self, curve: &DiscountCurve) -> Result<f64> {
        match *self {
            Instrument::Deposit { maturity, rate } => {
                Ok(curve.df(maturity)? * (1.0 + rate * maturity) - 1.0)
            }
            Instrument::Fra { start, end, rate } => {
                let tau = end - start;
                Ok(curve.df(start)? - (1.0 + rate * tau) * curve.df(end)?)
            }
            Instrument::Swap { maturity, rate } | Instrument::OisSwap { maturity, rate } => {
                swap_residual(curve, maturity, rate)
            }
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Instrument::Deposit { .. } => "Deposit",
            Instrument::Fra { .. } => "FRA",
            Instrument::Swap { .. } => "Swap",
            Instrument::OisSwap { .. } => "OISSwap",
        }
    }
}

/// Sequentially bootstrap a discount curve from sorted instruments.
///
/// Instruments must be ordered with strictly increasing pillar times
/// (duplicates rejected).  Each pillar DF is solved with Brent's method on
/// `[1e-10, 100]` (DF > 1 allowed: negative rates).  Returns
/// `IrmError::Bootstrap` when no positive DF can reprice a quote
/// (crossed/arbitrageable inputs).
pub fn bootstrap(instruments: &[Instrument]) -> Result<DiscountCurve> {
    if instruments.is_empty() {
        return Err(IrmError::InvalidInput(
            "bootstrap needs at least one instrument".into(),
        ));
    }
    let mut prev_pillar = 0.0;
    for ins in instruments {
        if ins.pillar() <= prev_pillar {
            return Err(IrmError::InvalidInput(format!(
                "instrument pillars must be strictly increasing \
                 (duplicate or out-of-order pillar at t={})",
                ins.pillar()
            )));
        }
        prev_pillar = ins.pillar();
    }

    let mut times: Vec<f64> = Vec::with_capacity(instruments.len());
    let mut dfs: Vec<f64> = Vec::with_capacity(instruments.len());
    for ins in instruments {
        let pillar = ins.pillar();
        let objective = |x: f64| -> Result<f64> {
            let mut trial_t = times.clone();
            trial_t.push(pillar);
            let mut trial_df = dfs.clone();
            trial_df.push(x);
            let trial = DiscountCurve::new(&trial_t, &trial_df)?;
            ins.residual(&trial)
        };
        let df = brentq(objective, DF_LO, DF_HI, 1e-14, BRENT_RTOL, BRENT_MAXITER)
            .map_err(|exc| {
                IrmError::Bootstrap(format!(
                    "bootstrap failed at pillar t={pillar} ({}, rate={}): no \
                     admissible positive discount factor — \
                     crossed/arbitrageable quotes? [{exc}]",
                    ins.kind(),
                    ins.rate()
                ))
            })?;
        times.push(pillar);
        dfs.push(df);
    }
    DiscountCurve::new(&times, &dfs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_examples() {
        assert_eq!(annual_schedule(2.0).unwrap(), vec![1.0, 2.0]);
        assert_eq!(annual_schedule(2.5).unwrap(), vec![0.5, 1.5, 2.5]);
        assert_eq!(annual_schedule(0.25).unwrap(), vec![0.25]);
        assert!(annual_schedule(0.0).is_err());
    }

    #[test]
    fn deposit_validation() {
        assert!(Instrument::deposit(1.0, -1.5).is_err()); // 1 + R*T <= 0
        assert!(Instrument::deposit(-1.0, 0.02).is_err());
        assert!(Instrument::deposit(1.0, f64::NAN).is_err());
        assert!(Instrument::fra(2.0, 1.0, 0.02).is_err());
    }
}
