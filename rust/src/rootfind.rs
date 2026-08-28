//! Native 1-D root finders: Brent's method and plain bisection.
//!
//! Implemented from scratch (no external numerical library) per project
//! rules.  Brent's method combines bisection, secant and inverse quadratic
//! interpolation and is guaranteed to converge for a continuous function
//! with a sign change on the bracket, with superlinear speed on
//! well-behaved functions.
//!
//! Objective functions are fallible (`FnMut(f64) -> Result<f64>`) so that
//! callers such as the bootstrap — whose residuals build trial curves —
//! can propagate their own errors through the solver.

use crate::error::{IrmError, Result};

/// Machine-epsilon-scale relative tolerance (2 * eps, classic `zbrent`).
pub const BRENT_RTOL: f64 = 4.44e-16;

/// Default absolute tolerance for [`brentq`].
pub const BRENT_XTOL: f64 = 1e-14;

/// Default iteration cap for [`brentq`].
pub const BRENT_MAXITER: usize = 100;

/// Find a root of `f` in `[a, b]` using Brent's method (NR `zbrent` variant).
///
/// `f(a)` and `f(b)` must have opposite signs (or one endpoint be an exact
/// root); otherwise `IrmError::RootFind` is returned.  The per-iteration
/// convergence test on the abscissa is `|c - b|/2 <= 2*rtol*|b| + xtol/2`
/// (Numerical-Recipes convention).  Non-convergence after `maxiter`
/// iterations is an error.
pub fn brentq<F>(
    mut f: F,
    a: f64,
    b: f64,
    xtol: f64,
    rtol: f64,
    maxiter: usize,
) -> Result<f64>
where
    F: FnMut(f64) -> Result<f64>,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(IrmError::InvalidInput(
            "brentq: bracket endpoints must be finite".into(),
        ));
    }
    if xtol <= 0.0 {
        return Err(IrmError::InvalidInput("brentq: xtol must be positive".into()));
    }
    let (mut a, mut b) = (a, b);
    let mut fa = f(a)?;
    let mut fb = f(b)?;
    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if (fa > 0.0) == (fb > 0.0) {
        return Err(IrmError::RootFind(format!(
            "brentq: root not bracketed: f({a})={fa}, f({b})={fb}"
        )));
    }
    let (mut c, mut fc) = (a, fa);
    let mut d = b - a;
    let mut e = b - a;
    for _ in 0..maxiter {
        if (fb > 0.0) == (fc > 0.0) {
            // Rename so that the root lies between b and c.
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol1 = 2.0 * rtol * b.abs() + 0.5 * xtol;
        let xm = 0.5 * (c - b);
        if xm.abs() <= tol1 || fb == 0.0 {
            return Ok(b);
        }
        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            // Attempt inverse quadratic interpolation (secant if a == c).
            let s = fb / fa;
            let (mut p, mut q);
            if a == c {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                q = fa / fc;
                let r = fb / fc;
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 {
                q = -q;
            }
            p = p.abs();
            if 2.0 * p < (3.0 * xm * q - (tol1 * q).abs()).min((e * q).abs()) {
                e = d; // accept interpolation
                d = p / q;
            } else {
                d = xm; // interpolation failed; bisect
                e = d;
            }
        } else {
            d = xm; // bounds decreasing too slowly; bisect
            e = d;
        }
        a = b;
        fa = fb;
        if d.abs() > tol1 {
            b += d;
        } else {
            b += if xm > 0.0 { tol1 } else { -tol1 };
        }
        fb = f(b)?;
    }
    Err(IrmError::RootFind(format!(
        "brentq: no convergence after {maxiter} iterations"
    )))
}

/// Plain bisection on `[a, b]`; robust fallback, linear convergence.
pub fn bisect<F>(mut f: F, a: f64, b: f64, xtol: f64, maxiter: usize) -> Result<f64>
where
    F: FnMut(f64) -> Result<f64>,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(IrmError::InvalidInput(
            "bisect: bracket endpoints must be finite".into(),
        ));
    }
    let fa = f(a)?;
    let fb = f(b)?;
    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if (fa > 0.0) == (fb > 0.0) {
        return Err(IrmError::RootFind(format!(
            "bisect: root not bracketed: f({a})={fa}, f({b})={fb}"
        )));
    }
    let (mut lo, mut hi, mut flo) = if a < b { (a, b, fa) } else { (b, a, fb) };
    for _ in 0..maxiter {
        let mid = 0.5 * (lo + hi);
        let fm = f(mid)?;
        if fm == 0.0 || 0.5 * (hi - lo) < xtol {
            return Ok(mid);
        }
        if (fm > 0.0) == (flo > 0.0) {
            lo = mid;
            flo = fm;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brent_finds_cubic_root() {
        let root = brentq(|x| Ok(x * x * x - 2.0), 0.0, 2.0, 1e-14, BRENT_RTOL, 100).unwrap();
        assert!((root - 2.0_f64.cbrt()).abs() < 1e-12);
    }

    #[test]
    fn brent_rejects_unbracketed() {
        let err = brentq(|x| Ok(x * x + 1.0), -1.0, 1.0, 1e-14, BRENT_RTOL, 100);
        assert!(matches!(err, Err(IrmError::RootFind(_))));
    }

    #[test]
    fn bisect_matches_brent() {
        let f = |x: f64| Ok((x - 0.3).exp() - 1.0);
        let rb = brentq(f, -1.0, 1.0, 1e-14, BRENT_RTOL, 100).unwrap();
        let ri = bisect(f, -1.0, 1.0, 1e-12, 200).unwrap();
        assert!((rb - 0.3).abs() < 1e-12);
        assert!((ri - 0.3).abs() < 1e-11);
    }
}
