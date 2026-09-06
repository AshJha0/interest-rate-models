//! Discount curve with log-linear interpolation in ln(DF).
//!
//! Conventions (see `API_SPEC.md`):
//!
//! * Day count is ACT/365F everywhere; all times are year fractions
//!   `days/365` (a documented simplification versus the ACT/360
//!   money-market convention used for real deposits/FRAs).
//! * Rates are decimals (0.05 = 5%).
//! * Interpolation is *linear in ln(DF)* between pillars, with an implicit
//!   node `(t = 0, DF = 1)`.  Linear-in-ln(DF) is exactly the
//!   piecewise-constant instantaneous-forward interpolation: on a segment
//!   `[t_i, t_{i+1}]` the instantaneous forward is the constant
//!   `f_i = -(ln DF_{i+1} - ln DF_i) / (t_{i+1} - t_i)`.
//! * Extrapolation beyond the last pillar is flat-forward: the last
//!   segment's constant forward is carried on indefinitely.
//! * Negative rates are fully supported: DF > 1 is legal (EUR/JPY style);
//!   the only requirement is DF > 0.

use crate::error::{IrmError, Result};

/// Immutable discount curve defined by pillar times and discount factors.
///
/// Pillar times must be strictly increasing and `> 0`; discount factors
/// must be finite and `> 0` (values above 1 are allowed — negative rates).
#[derive(Debug, Clone)]
pub struct DiscountCurve {
    times: Vec<f64>,
    dfs: Vec<f64>,
    /// Pillar times with the implicit `t = 0` node prepended.
    t_nodes: Vec<f64>,
    /// `ln DF` at each node of `t_nodes` (`ln 1 = 0` at the front).
    ln_dfs: Vec<f64>,
}

impl DiscountCurve {
    /// Build a curve from pillar `times` (years, ACT/365F) and `dfs`.
    pub fn new(times: &[f64], dfs: &[f64]) -> Result<Self> {
        if times.is_empty() {
            return Err(IrmError::InvalidInput(
                "curve needs at least one pillar".into(),
            ));
        }
        if times.len() != dfs.len() {
            return Err(IrmError::InvalidInput(format!(
                "times and dfs length mismatch: {} vs {}",
                times.len(),
                dfs.len()
            )));
        }
        let mut prev = 0.0_f64;
        for &t in times {
            if !t.is_finite() {
                return Err(IrmError::InvalidInput(format!(
                    "non-finite pillar time {t}"
                )));
            }
            if t <= prev {
                return Err(IrmError::InvalidInput(format!(
                    "pillar times must be strictly increasing and positive; \
                     offending time {t} after {prev}"
                )));
            }
            prev = t;
        }
        for (&t, &p) in times.iter().zip(dfs.iter()) {
            if !p.is_finite() || p <= 0.0 {
                return Err(IrmError::InvalidInput(format!(
                    "discount factor at t={t} must be finite and > 0, got {p}"
                )));
            }
        }
        let mut t_nodes = Vec::with_capacity(times.len() + 1);
        t_nodes.push(0.0);
        t_nodes.extend_from_slice(times);
        let mut ln_dfs = Vec::with_capacity(dfs.len() + 1);
        ln_dfs.push(0.0);
        ln_dfs.extend(dfs.iter().map(|p| p.ln()));
        Ok(Self {
            times: times.to_vec(),
            dfs: dfs.to_vec(),
            t_nodes,
            ln_dfs,
        })
    }

    /// Pillar times (excluding the implicit zero node).
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Discount factors at the pillars.
    pub fn dfs(&self) -> &[f64] {
        &self.dfs
    }

    /// Index `i` of the segment `[t_i, t_{i+1}]` used for time `t`.
    ///
    /// Times at or beyond the last pillar map to the last segment
    /// (flat-forward extrapolation); a time exactly at an interior node
    /// maps to the segment *starting* at that node (right-hand segment).
    fn segment(&self, t: f64) -> usize {
        let n_seg = self.t_nodes.len() - 1;
        // partition_point == Python's bisect_right for a sorted slice.
        let i = self.t_nodes.partition_point(|&x| x <= t).saturating_sub(1);
        i.min(n_seg - 1)
    }

    /// `d ln(DF)/dt` on segment `i` (equals minus the constant forward).
    fn slope(&self, i: usize) -> f64 {
        (self.ln_dfs[i + 1] - self.ln_dfs[i]) / (self.t_nodes[i + 1] - self.t_nodes[i])
    }

    fn ln_df(&self, t: f64) -> Result<f64> {
        if !t.is_finite() || t < 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "time must be finite and >= 0, got {t}"
            )));
        }
        if t == 0.0 {
            return Ok(0.0);
        }
        let i = self.segment(t);
        Ok(self.ln_dfs[i] + self.slope(i) * (t - self.t_nodes[i]))
    }

    /// Discount factor `DF(t)`; `DF(0) = 1`, flat-forward beyond last pillar.
    pub fn df(&self, t: f64) -> Result<f64> {
        Ok(self.ln_df(t)?.exp())
    }

    /// Continuously compounded zero rate `z(t) = -ln DF(t) / t`.
    ///
    /// At `t = 0` the limit is returned, i.e. the instantaneous forward of
    /// the first segment.
    pub fn zero_rate(&self, t: f64) -> Result<f64> {
        if !t.is_finite() || t < 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "time must be finite and >= 0, got {t}"
            )));
        }
        if t == 0.0 {
            return Ok(-self.slope(0));
        }
        Ok(-self.ln_df(t)? / t)
    }

    /// Instantaneous forward `f(t) = -d ln DF / dt`.
    ///
    /// Piecewise constant by construction.  At a pillar the *right*
    /// segment's forward is returned; at/after the last pillar the last
    /// segment's forward (the flat extrapolation level).
    pub fn inst_forward(&self, t: f64) -> Result<f64> {
        if !t.is_finite() || t < 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "time must be finite and >= 0, got {t}"
            )));
        }
        Ok(-self.slope(self.segment(t)))
    }

    /// Simply compounded forward rate over `[t1, t2]`:
    /// `F(t1, t2) = (DF(t1)/DF(t2) - 1) / (t2 - t1)`.  A result that is not
    /// representable (DF(t2) underflowed to 0, ratio overflowed) is an
    /// `InvalidInput` error — never `inf`.
    pub fn fwd_rate(&self, t1: f64, t2: f64) -> Result<f64> {
        if !t1.is_finite() || !t2.is_finite() {
            return Err(IrmError::InvalidInput(
                "forward-rate times must be finite".into(),
            ));
        }
        if t1 < 0.0 || t2 <= t1 {
            return Err(IrmError::InvalidInput(format!(
                "need 0 <= t1 < t2 for a forward rate, got t1={t1}, t2={t2}"
            )));
        }
        let df1 = self.df(t1)?;
        let df2 = self.df(t2)?;
        if df2 == 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "forward rate over [{t1}, {t2}] not representable: DF(t2) underflowed to 0"
            )));
        }
        let fwd = (df1 / df2 - 1.0) / (t2 - t1);
        if !fwd.is_finite() {
            return Err(IrmError::InvalidInput(format!(
                "forward rate over [{t1}, {t2}] not representable: {fwd}"
            )));
        }
        Ok(fwd)
    }
}

/// Forward-starting par swap rate off a single curve.
///
/// `times = [t0, t1, ..., tn]` gives the swap start `t0 >= 0` and the fixed
/// payment times.  With accruals `tau_i = t_i - t_{i-1}` and the
/// single-curve identity (floating leg PV = `DF(t0) - DF(tn)`):
///
/// `par = (DF(t0) - DF(tn)) / sum_i tau_i DF(t_i)`.
pub fn par_swap_rate(curve: &DiscountCurve, times: &[f64]) -> Result<f64> {
    if times.len() < 2 {
        return Err(IrmError::InvalidInput(
            "par_swap_rate needs a start and at least one payment".into(),
        ));
    }
    let mut prev = times[0];
    if prev < 0.0 {
        return Err(IrmError::InvalidInput(format!(
            "swap start must be >= 0, got {prev}"
        )));
    }
    let mut annuity = 0.0;
    for &t in &times[1..] {
        if t <= prev {
            return Err(IrmError::InvalidInput(
                "swap schedule times must be strictly increasing".into(),
            ));
        }
        annuity += (t - prev) * curve.df(t)?;
        prev = t;
    }
    Ok((curve.df(times[0])? - curve.df(times[times.len() - 1])?) / annuity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_selection_matches_bisect_right() {
        let c = DiscountCurve::new(&[1.0, 2.0, 5.0], &[0.99, 0.97, 0.90]).unwrap();
        // Right-hand segment at a pillar: forward just after 2.0.
        let f_left = c.inst_forward(1.5).unwrap();
        let f_right = c.inst_forward(2.0).unwrap();
        let f_mid = c.inst_forward(3.0).unwrap();
        assert!(f_left != f_right);
        assert_eq!(f_right, f_mid);
    }

    #[test]
    fn zero_rate_limit_at_origin() {
        let c = DiscountCurve::new(&[2.0], &[0.96]).unwrap();
        let f0 = -(0.96_f64.ln()) / 2.0;
        assert!((c.zero_rate(0.0).unwrap() - f0).abs() < 1e-15);
    }
}
