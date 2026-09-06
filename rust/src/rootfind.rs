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

/// Reject NaN/inf function values: a silent NaN would otherwise be treated
/// as "same sign" and either misreported as a bracketing failure or, worse,
/// returned as a converged root.
fn finite_value(who: &str, x: f64, fx: f64) -> Result<f64> {
    if !fx.is_finite() {
        return Err(IrmError::RootFind(format!(
            "{who}: function value not finite at x={x}: {fx}"
        )));
    }
    Ok(fx)
}

/// Find a root of `f` in `[a, b]` using Brent's method (NR `zbrent` variant).
///
/// `f(a)` and `f(b)` must have opposite signs (or one endpoint be an exact
/// root); otherwise `IrmError::RootFind` is returned.  The per-iteration
/// convergence test on the abscissa is `|c - b|/2 <= 2*rtol*|b| + xtol/2`
/// (Numerical-Recipes convention).  Non-convergence after `maxiter`
/// iterations and a non-finite function value anywhere are
/// `IrmError::RootFind` errors — an unconverged or NaN root is never
/// returned; an unbracketed root is `RootFind` too, bad inputs are
/// `InvalidInput`.
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
    let mut fa = finite_value("brentq", a, f(a)?)?;
    let mut fb = finite_value("brentq", b, f(b)?)?;
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
        fb = finite_value("brentq", b, f(b)?)?;
    }
    Err(IrmError::RootFind(format!(
        "brentq: no convergence after {maxiter} iterations"
    )))
}

/// Plain bisection on `[a, b]`; robust fallback, linear convergence.
/// Same bracket contract as [`brentq`]: unbracketed root, non-finite
/// values and exhausting `maxiter` are `IrmError::RootFind` errors.
pub fn bisect<F>(mut f: F, a: f64, b: f64, xtol: f64, maxiter: usize) -> Result<f64>
where
    F: FnMut(f64) -> Result<f64>,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(IrmError::InvalidInput(
            "bisect: bracket endpoints must be finite".into(),
        ));
    }
    if xtol <= 0.0 {
        return Err(IrmError::InvalidInput("bisect: xtol must be positive".into()));
    }
    let fa = finite_value("bisect", a, f(a)?)?;
    let fb = finite_value("bisect", b, f(b)?)?;
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
        let fm = finite_value("bisect", mid, f(mid)?)?;
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
    Err(IrmError::RootFind(format!(
        "bisect: no convergence after {maxiter} iterations"
    )))
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

    #[test]
    fn brent_non_convergence_is_error() {
        let err = brentq(|x| Ok((1e3 * (x - 0.123456)).tanh()), -10.0, 10.0, 1e-14, BRENT_RTOL, 1);
        assert!(matches!(err, Err(IrmError::RootFind(_))));
    }

    #[test]
    fn bisect_non_convergence_is_error() {
        assert!(matches!(
            bisect(|x| Ok(x - 0.3), 0.0, 1.0, 1e-300, 5),
            Err(IrmError::RootFind(_))
        ));
        assert!(bisect(|x| Ok(x), 0.0, f64::NAN, 1e-12, 200).is_err());
        assert!(bisect(|x| Ok(x), -1.0, 1.0, 0.0, 200).is_err());
        assert!((bisect(|x| Ok(x - 0.25), 0.0, 1.0, 1e-12, 200).unwrap() - 0.25).abs() < 1e-11);
    }

    #[test]
    fn non_finite_function_values_rejected() {
        // A NaN residual must never be returned as a "root" or misreported
        // as a bracketing failure.
        let nan_inside = |x: f64| Ok(if x > 0.2 && x < 0.8 { f64::NAN } else if x < 0.5 { 1.0 } else { -1.0 });
        assert!(matches!(
            brentq(nan_inside, 0.0, 1.0, 1e-14, BRENT_RTOL, 100),
            Err(IrmError::RootFind(msg)) if msg.contains("not finite")
        ));
        assert!(brentq(|_| Ok(f64::NAN), 0.0, 1.0, 1e-14, BRENT_RTOL, 100).is_err());
        assert!(matches!(
            bisect(|x| Ok(if x > 0.4 { f64::INFINITY } else { -1.0 }), 0.0, 1.0, 1e-12, 200),
            Err(IrmError::RootFind(_))
        ));
    }
}
