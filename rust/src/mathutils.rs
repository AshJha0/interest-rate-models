//! Small numerical utilities shared by the models.
//!
//! Contains a native `erfc` (Cody's rational-Chebyshev CALERF algorithm,
//! ~1e-16 relative accuracy — Rust's standard library has no error
//! function), the standard normal CDF built on it, and the exact
//! Ornstein-Uhlenbeck transition moments used by both the Vasicek and the
//! Hull-White Monte Carlo engines.

use crate::error::{IrmError, Result};

const SQRT_2: f64 = std::f64::consts::SQRT_2;

// ------------------------------------------------------------------ //
// Cody (1969) rational approximations for erf/erfc ("CALERF").
// ------------------------------------------------------------------ //

#[allow(clippy::excessive_precision)]
const ERF_A: [f64; 5] = [
    3.16112374387056560e0,
    1.13864154151050156e2,
    3.77485237685302021e2,
    3.20937758913846947e3,
    1.85777706184603153e-1,
];
#[allow(clippy::excessive_precision)]
const ERF_B: [f64; 4] = [
    2.36012909523441209e1,
    2.44024637934444173e2,
    1.28261652607737228e3,
    2.84423683343917062e3,
];
#[allow(clippy::excessive_precision)]
const ERF_C: [f64; 9] = [
    5.64188496988670089e-1,
    8.88314979438837594e0,
    6.61191906371416295e1,
    2.98635138197400131e2,
    8.81952221241769090e2,
    1.71204761263407058e3,
    2.05107837782607147e3,
    1.23033935479799725e3,
    2.15311535474403846e-8,
];
#[allow(clippy::excessive_precision)]
const ERF_D: [f64; 8] = [
    1.57449261107098347e1,
    1.17693950891312499e2,
    5.37181101862009858e2,
    1.62138957456669019e3,
    3.29079923573345963e3,
    4.36261909014324716e3,
    3.43936767414372164e3,
    1.23033935480374942e3,
];
#[allow(clippy::excessive_precision)]
const ERF_P: [f64; 6] = [
    3.05326634961232344e-1,
    3.60344899949804439e-1,
    1.25781726111229246e-1,
    1.60837851487422766e-2,
    6.58749161529837803e-4,
    1.63153871373020978e-2,
];
#[allow(clippy::excessive_precision)]
const ERF_Q: [f64; 5] = [
    2.56852019228982242e0,
    1.87295284992346047e0,
    5.27905102951428412e-1,
    6.05183413124413191e-2,
    2.33520497626869185e-3,
];
/// `1/sqrt(pi)`.
#[allow(clippy::excessive_precision)]
const FRTPI: f64 = 5.6418958354775628695e-1;

/// Complementary error function, Cody's CALERF algorithm.
///
/// Absolute accuracy is far better than the 1e-12 the API contract asks
/// for (roughly machine precision over the whole real line).
pub fn erfc(x: f64) -> f64 {
    let y = x.abs();
    if y <= 0.46875 {
        // |x| small: compute erf directly, erfc = 1 - erf.
        let z = if y > 1.11e-16 { y * y } else { 0.0 };
        let mut num = ERF_A[4] * z;
        let mut den = z;
        for i in 0..3 {
            num = (num + ERF_A[i]) * z;
            den = (den + ERF_B[i]) * z;
        }
        let erf = x * (num + ERF_A[3]) / (den + ERF_B[3]);
        return 1.0 - erf;
    }
    let result = if y <= 4.0 {
        let mut num = ERF_C[8] * y;
        let mut den = y;
        for i in 0..7 {
            num = (num + ERF_C[i]) * y;
            den = (den + ERF_D[i]) * y;
        }
        (num + ERF_C[7]) / (den + ERF_D[7])
    } else {
        let z = 1.0 / (y * y);
        let mut num = ERF_P[5] * z;
        let mut den = z;
        for i in 0..4 {
            num = (num + ERF_P[i]) * z;
            den = (den + ERF_Q[i]) * z;
        }
        let r = z * (num + ERF_P[4]) / (den + ERF_Q[4]);
        (FRTPI - r) / y
    };
    // exp(-y^2) split as exp(-ysq^2)*exp(-del) for accuracy (Cody's trick).
    let ysq = (y * 16.0).trunc() / 16.0;
    let del = (y - ysq) * (y + ysq);
    let erfc_abs = (-ysq * ysq).exp() * (-del).exp() * result;
    if x < 0.0 {
        2.0 - erfc_abs
    } else {
        erfc_abs
    }
}

/// Standard normal CDF `N(x) = 0.5 * erfc(-x / sqrt(2))`.
///
/// `erfc` is used instead of `erf` for better relative accuracy in the far
/// tails (large negative `x`).
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / SQRT_2)
}

// ------------------------------------------------------------------ //
// Exact OU transition moments.
// ------------------------------------------------------------------ //

/// Exact one-step transition moments of a mean-zero OU process.
///
/// For `dx = -a x dt + sigma dW` and a step of length `dt`, conditional on
/// the current state `x` (writing `b = (1 - decay)/a`, `decay = e^{-a dt}`):
///
/// * `x_next ~ N(x * decay, var_x)` with `var_x = sigma^2 (1 - decay^2)/(2a)`
/// * `I = int x ds ~ N(x * b, var_i)` with
///   `var_i = (sigma^2/a^2)(dt - 2b + (1 - decay^2)/(2a))`
/// * `Cov(x_next, I) = (sigma^2/(2a^2)) (1 - decay)^2`
///
/// Sampling the pair `(x_next, I)` from this bivariate Gaussian gives an
/// *exact* (bias-free) discretisation of both the short rate and its time
/// integral — which is what makes the MC bond estimators unbiased at any
/// step size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OuStepMoments {
    /// `exp(-a dt)`.
    pub decay: f64,
    /// `(1 - decay)/a`.
    pub b: f64,
    /// Conditional variance of `x_next`.
    pub var_x: f64,
    /// Conditional variance of the step integral `I`.
    pub var_i: f64,
    /// Conditional covariance of `(x_next, I)`.
    pub cov: f64,
}

/// Compute [`OuStepMoments`] for mean reversion `a > 0` and step `dt > 0`.
pub fn ou_step_moments(a: f64, sigma: f64, dt: f64) -> Result<OuStepMoments> {
    if a <= 0.0 {
        return Err(IrmError::InvalidInput(format!(
            "mean reversion must be positive, got {a}"
        )));
    }
    if dt <= 0.0 {
        return Err(IrmError::InvalidInput(format!(
            "time step must be positive, got {dt}"
        )));
    }
    let decay = (-a * dt).exp();
    let b = (1.0 - decay) / a;
    let s2 = sigma * sigma;
    // Guard tiny negatives from floating-point cancellation.
    let var_x = (s2 * (1.0 - decay * decay) / (2.0 * a)).max(0.0);
    let var_i = ((s2 / (a * a)) * (dt - 2.0 * b + (1.0 - decay * decay) / (2.0 * a))).max(0.0);
    let cov = (s2 / (2.0 * a * a)) * (1.0 - decay) * (1.0 - decay);
    Ok(OuStepMoments {
        decay,
        b,
        var_x,
        var_i,
        cov,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erfc_reference_values() {
        // Reference values from a correctly rounded erfc (mpmath/libm).
        let cases = [
            (0.0, 1.0),
            (0.5, 0.4795001221869535),
            (1.0, 0.15729920705028513),
            (-1.0, 1.8427007929497148),
            (2.0, 0.004677734981047265),
            (3.5, 7.430983723414128e-7),
            (5.0, 1.5374597944280351e-12),
            (-3.0, 1.9999779095030014),
        ];
        for (x, want) in cases {
            assert!(
                (erfc(x) - want).abs() < 1e-14,
                "erfc({x}) = {} want {want}",
                erfc(x)
            );
        }
    }

    #[test]
    fn norm_cdf_reference_values() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-16);
        assert!((norm_cdf(1.0) - 0.8413447460685429).abs() < 1e-14);
        assert!((norm_cdf(-1.0) - 0.15865525393145707).abs() < 1e-14);
        assert!((norm_cdf(-8.0) - 6.220960574271786e-16).abs() < 1e-24);
        assert!(norm_cdf(8.0) < 1.0 + 1e-15 && norm_cdf(8.0) > 1.0 - 1e-14);
    }

    #[test]
    fn ou_moments_small_dt_limits() {
        // For a*dt -> 0: var_x ~ sigma^2 dt, var_i ~ sigma^2 dt^3/3,
        // cov ~ sigma^2 dt^2/2 (corrections are O(a dt) = 5e-4 here).
        let m = ou_step_moments(0.05, 0.02, 0.01).unwrap();
        let s2 = 0.02f64 * 0.02;
        assert!((m.var_x - s2 * 0.01).abs() / (s2 * 0.01) < 1e-3);
        assert!((m.var_i - s2 * 0.01f64.powi(3) / 3.0).abs() / (s2 * 0.01f64.powi(3) / 3.0) < 1e-3);
        assert!((m.cov - s2 * 0.01f64.powi(2) / 2.0).abs() / (s2 * 0.01f64.powi(2) / 2.0) < 1e-3);
        // Cauchy-Schwarz: correlation valid.
        assert!(m.cov * m.cov <= m.var_x * m.var_i * (1.0 + 1e-12));
    }
}
