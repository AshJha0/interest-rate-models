//! Shared kernels: normal CDF/PDF and the exact OU step moments.
//!
//! Reference values for the integrated-OU variance come from a 60-digit
//! evaluation of the closed form (python/tests/test_mathutils.py derives
//! them with `decimal`; the same numbers are pinned here).

use irm::mathutils::OU_SERIES_THRESHOLD;
use irm::{norm_cdf, norm_pdf, ou_integral_variance, ou_step_moments};

// (a, sigma, dt, var_i, var_x) with the variances at 60-digit precision.
const CASES: [(f64, f64, f64, f64, f64); 8] = [
    (0.001, 0.01, 0.01, 3.3333083334499993e-11, 9.9999000006666625e-07),
    (0.0001, 0.01, 0.001, 3.3333330833333452e-14, 9.9999990000000661e-08),
    (1e-06, 0.02, 0.125, 2.6041664225260560e-07, 4.9999993750000522e-05),
    (0.01, 0.01, 1.0, 3.3084495845603743e-05, 9.9006633466223494e-05),
    (0.05, 0.02, 0.01, 1.3328334499791697e-10, 3.9980006665000334e-06),
    (0.1, 0.01, 0.125, 6.4497359652182101e-08, 1.2345043985833665e-05),
    (0.8, 0.02, 2.5, 5.9493183361629550e-04, 2.4542109027781648e-04),
    (5.0, 0.01, 1.0, 2.8107625552266318e-06, 9.9995460007023755e-06),
];

#[test]
fn ou_moments_small_a_dt_accurate() {
    for &(a, sigma, dt, var_i_ref, var_x_ref) in &CASES {
        let m = ou_step_moments(a, sigma, dt).unwrap();
        assert!((m.var_i / var_i_ref - 1.0).abs() < 1e-9, "a={a} dt={dt}: {}", m.var_i);
        assert!((m.var_x / var_x_ref - 1.0).abs() < 1e-12, "a={a} dt={dt}");
        assert!((m.b - (-(-a * dt).exp_m1()) / a).abs() <= 1e-14 * m.b);
        // Cauchy-Schwarz: the (x', I) covariance matrix must be PSD.
        assert!(m.cov * m.cov <= m.var_x * m.var_i * (1.0 + 1e-12));
        let x = a * dt;
        if x < 1e-3 {
            let series = sigma * sigma * dt * dt * dt
                * (1.0 / 3.0 - x / 4.0 + 7.0 * x * x / 60.0 - x * x * x / 24.0);
            assert!((m.var_i / series - 1.0).abs() < 1e-9);
        }
    }
}

#[test]
fn naive_formula_is_wrong_for_tiny_a_dt() {
    // Documents the bug the series fixes: plain evaluation is off by orders
    // of magnitude at a*dt = 1e-7 (80x per the review measurement).
    let (a, sigma, dt) = (1e-4_f64, 0.01_f64, 1e-3_f64);
    let decay = (-a * dt).exp();
    let b = (1.0 - decay) / a;
    let naive = (sigma * sigma / (a * a)) * (dt - 2.0 * b + (1.0 - decay * decay) / (2.0 * a));
    let reference = 3.3333330833333452e-14;
    assert!((naive - reference).abs() / reference > 10.0);
    assert!((ou_integral_variance(a, sigma, dt).unwrap() / reference - 1.0).abs() < 1e-12);
}

#[test]
fn series_and_closed_form_agree_at_threshold() {
    let x = OU_SERIES_THRESHOLD;
    let below = ou_integral_variance(x * (1.0 - 1e-12), 0.01, 1.0).unwrap();
    let above = ou_integral_variance(x * (1.0 + 1e-12), 0.01, 1.0).unwrap();
    assert!((below / above - 1.0).abs() < 1e-9);
}

#[test]
fn ou_moments_validation() {
    assert!(ou_step_moments(0.0, 0.01, 0.1).is_err());
    assert!(ou_step_moments(0.1, 0.01, 0.0).is_err());
    assert!(ou_step_moments(f64::NAN, 0.01, 0.1).is_err());
    assert!(ou_step_moments(0.1, -0.01, 0.1).is_err());
    assert!(ou_integral_variance(0.1, f64::INFINITY, 0.1).is_err());
    let m = ou_step_moments(0.3, 0.0, 0.5).unwrap();
    assert_eq!(m.var_x, 0.0);
    assert_eq!(m.var_i, 0.0);
    assert_eq!(m.cov, 0.0);
    assert!((m.decay - (-0.15_f64).exp()).abs() < 1e-16);
}

#[test]
fn normal_cdf_and_pdf() {
    assert_eq!(norm_cdf(0.0), 0.5);
    assert!((norm_cdf(1.0) - 0.8413447460685429).abs() < 1e-15);
    assert!((norm_cdf(-8.0) / 6.220960574271786e-16 - 1.0).abs() < 1e-10);
    assert!((norm_pdf(0.0) - 1.0 / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-16);
    assert!((norm_pdf(2.0) - (-2.0_f64).exp() / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-16);
}
