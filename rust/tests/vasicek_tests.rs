//! Vasicek: closed forms, moments, options, exact MC, calibration.

use std::path::{Path, PathBuf};

use irm::{calibrate_vasicek, load_zero_yields, Vasicek};

fn model() -> Vasicek {
    Vasicek::new(0.5, 0.03, 0.01, 0.02).unwrap()
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")
}

#[test]
fn b_and_a_factors_closed_form() {
    let m = model();
    let (k, th, s, tau) = (0.5_f64, 0.03_f64, 0.01_f64, 5.0_f64);
    let b = (1.0 - (-k * tau).exp()) / k;
    assert!((m.b_factor(tau).unwrap() - b).abs() < 1e-15);
    let a = ((th - s * s / (2.0 * k * k)) * (b - tau) - s * s * b * b / (4.0 * k)).exp();
    assert!((m.a_factor(tau).unwrap() - a).abs() < 1e-15);
    let p = a * (-b * 0.02).exp();
    assert!((m.zcb_price(tau, 0.0, None).unwrap() - p).abs() < 1e-15);
}

#[test]
fn zcb_degenerate_and_yield() {
    let m = model();
    assert_eq!(m.zcb_price(0.0, 0.0, None).unwrap(), 1.0);
    let y = -m.zcb_price(5.0, 0.0, None).unwrap().ln() / 5.0;
    assert!((m.zero_yield(5.0, 0.0, None).unwrap() - y).abs() < 1e-15);
    assert!(m.zero_yield(0.0, 0.0, None).is_err());
    assert!(m.zcb_price(1.0, 2.0, None).is_err()); // maturity before t
}

#[test]
fn negative_rates_supported() {
    let m = Vasicek::new(0.3, -0.01, 0.008, -0.005).unwrap();
    assert!(m.zcb_price(1.0, 0.0, None).unwrap() > 1.0); // ZCB above par
    assert!(m.zero_yield(1.0, 0.0, None).unwrap() < 0.0);
}

#[test]
fn rt_moments_closed_form() {
    let m = model();
    let (k, th, s, r0, t) = (0.5_f64, 0.03_f64, 0.01_f64, 0.02_f64, 2.0_f64);
    let mean = th + (r0 - th) * (-k * t).exp();
    let var = s * s * (1.0 - (-2.0 * k * t).exp()) / (2.0 * k);
    assert!((m.r_mean(t, None).unwrap() - mean).abs() < 1e-15);
    assert!((m.r_var(t).unwrap() - var).abs() < 1e-18);
    assert_eq!(m.r_var(0.0).unwrap(), 0.0);
}

#[test]
fn mc_zcb_within_3se_of_analytic() {
    let m = model();
    let analytic = m.zcb_price(5.0, 0.0, None).unwrap();
    let (price, se) = m.mc_zcb(5.0, 4, 50_000, 12).unwrap();
    assert!(se > 0.0);
    assert!(
        (price - analytic).abs() < 3.0 * se,
        "MC {price} vs analytic {analytic}, se {se}"
    );
    // Exact scheme: no discretisation bias — 1 step must also agree.
    let (p1, se1) = m.mc_zcb(5.0, 1, 50_000, 13).unwrap();
    assert!((p1 - analytic).abs() < 3.0 * se1);
}

#[test]
fn mc_moments_match_distribution() {
    let m = model();
    let n = 50_000;
    let (r_end, _) = m.simulate(2.0, 4, n, 3).unwrap();
    let mean = r_end.iter().sum::<f64>() / n as f64;
    let se_mean = (m.r_var(2.0).unwrap() / n as f64).sqrt();
    assert!((mean - m.r_mean(2.0, None).unwrap()).abs() < 3.0 * se_mean);
    let var = r_end.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / (n as f64 - 1.0);
    assert!((var - m.r_var(2.0).unwrap()).abs() / m.r_var(2.0).unwrap() < 0.05);
}

#[test]
fn sigma_zero_mc_is_deterministic() {
    let m = Vasicek::new(0.5, 0.03, 0.0, 0.02).unwrap();
    let (price, se) = m.mc_zcb(5.0, 16, 100, 1).unwrap();
    assert_eq!(se, 0.0);
    let analytic = m.zcb_price(5.0, 0.0, None).unwrap();
    assert!((price - analytic).abs() / analytic < 1e-13);
}

#[test]
fn bond_option_put_call_parity_grid() {
    // Property-style: ZBC - ZBP = P(0,S) - K P(0,T) across strikes/expiries.
    let m = model();
    for (expiry, s_mat) in [(0.5, 2.0), (1.0, 3.0), (2.0, 7.0)] {
        for k in [0.80, 0.90, 0.95, 1.00, 1.05] {
            let lhs = m.zbc(expiry, s_mat, k).unwrap() - m.zbp(expiry, s_mat, k).unwrap();
            let rhs = m.zcb_price(s_mat, 0.0, None).unwrap()
                - k * m.zcb_price(expiry, 0.0, None).unwrap();
            assert!((lhs - rhs).abs() < 1e-13, "parity broken at K={k}, T={expiry}");
        }
    }
}

#[test]
fn bond_option_edges_and_degeneration() {
    let m = model();
    // Deep ITM call ~ forward minus strike; deep OTM ~ 0.
    let deep_itm = m.zbc(1.0, 3.0, 1e-8).unwrap();
    let fwd_minus_k = m.zcb_price(3.0, 0.0, None).unwrap()
        - 1e-8 * m.zcb_price(1.0, 0.0, None).unwrap();
    assert!((deep_itm - fwd_minus_k).abs() / fwd_minus_k < 1e-9);
    assert!(m.zbc(1.0, 3.0, 10.0).unwrap().abs() < 1e-12);
    // expiry = 0 and sigma = 0 collapse to intrinsic.
    let intrinsic = (m.zcb_price(3.0, 0.0, None).unwrap() - 0.9).max(0.0);
    assert!((m.zbc(0.0, 3.0, 0.9).unwrap() - intrinsic).abs() < 1e-15);
    let m0 = Vasicek::new(0.5, 0.03, 0.0, 0.02).unwrap();
    let put_intr = (0.99 * m0.zcb_price(1.0, 0.0, None).unwrap()
        - m0.zcb_price(3.0, 0.0, None).unwrap())
    .max(0.0);
    assert!((m0.zbp(1.0, 3.0, 0.99).unwrap() - put_intr).abs() < 1e-15);
}

#[test]
fn call_value_decreasing_in_strike() {
    let m = model();
    let strikes = [0.85, 0.90, 0.95, 1.00];
    let calls: Vec<f64> = strikes.iter().map(|&k| m.zbc(1.0, 3.0, k).unwrap()).collect();
    for c in &calls {
        assert!(*c >= 0.0);
    }
    for w in calls.windows(2) {
        assert!(w[1] <= w[0] + 1e-15);
    }
}

#[test]
fn parameter_validation() {
    assert!(Vasicek::new(0.0, 0.03, 0.01, 0.02).is_err());
    assert!(Vasicek::new(-0.5, 0.03, 0.01, 0.02).is_err());
    assert!(Vasicek::new(0.5, 0.03, -0.01, 0.02).is_err());
    assert!(Vasicek::new(0.5, f64::NAN, 0.01, 0.02).is_err());
    let m = model();
    assert!(m.zbc(1.0, 3.0, 0.0).is_err()); // strike must be > 0
    assert!(m.zbc(3.0, 1.0, 0.9).is_err()); // bond matures before expiry
    assert!(m.simulate(1.0, 0, 10, 1).is_err());
    assert!(m.b_factor(-1.0).is_err());
}

#[test]
fn calibration_recovers_parameters() {
    let truth = Vasicek::new(0.8, 0.05, 0.02, 0.03).unwrap();
    let ts = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0];
    let ys: Vec<f64> = ts
        .iter()
        .map(|&t| truth.zero_yield(t, 0.0, None).unwrap())
        .collect();
    let res = calibrate_vasicek(&ts, &ys, 0.03, None, 4000).unwrap();
    assert!(res.converged);
    assert!(res.rmse < 1e-7);
    assert!((res.kappa - 0.8).abs() < 2e-3);
    assert!((res.theta - 0.05).abs() < 2e-3);
    // sigma has weak curvature influence -> looser bound.
    assert!((res.sigma - 0.02).abs() < 5e-3);
    let refit = res.model().unwrap();
    for (&t, &y) in ts.iter().zip(ys.iter()) {
        assert!((refit.zero_yield(t, 0.0, None).unwrap() - y).abs() < 1e-6);
    }
}

#[test]
fn calibration_from_bundled_csv() {
    let (ts, ys) = load_zero_yields(&data_dir().join("zero_yields.csv")).unwrap();
    let res = calibrate_vasicek(&ts, &ys, 0.03, None, 4000).unwrap();
    assert!(res.converged);
    assert!(res.rmse < 1e-4);
    assert!((res.kappa - 0.8).abs() / 0.8 < 0.2);
}

#[test]
fn calibration_non_convergence_reported_not_error() {
    let truth = Vasicek::new(0.8, 0.05, 0.02, 0.03).unwrap();
    let ts = [1.0, 2.0, 5.0, 10.0];
    let ys: Vec<f64> = ts
        .iter()
        .map(|&t| truth.zero_yield(t, 0.0, None).unwrap())
        .collect();
    let res = calibrate_vasicek(&ts, &ys, 0.03, None, 2).unwrap();
    assert!(!res.converged); // reported, no error
    assert!(res.rmse.is_finite());
}

#[test]
fn calibration_input_validation() {
    assert!(calibrate_vasicek(&[1.0, 2.0], &[0.02, 0.03], 0.02, None, 100).is_err());
    assert!(calibrate_vasicek(&[1.0, -2.0, 3.0], &[0.02, 0.03, 0.03], 0.02, None, 100).is_err());
    assert!(
        calibrate_vasicek(&[1.0, 2.0, 3.0], &[0.02, f64::NAN, 0.03], 0.02, None, 100).is_err()
    );
    assert!(calibrate_vasicek(&[1.0, 2.0, 3.0], &[0.02, 0.02, 0.03], f64::NAN, None, 100).is_err());
}
