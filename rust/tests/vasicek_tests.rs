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
    let res = calibrate_vasicek(&ts, &ys, 0.03, None, 4000, None).unwrap();
    assert!(res.converged && !res.at_bound);
    assert_eq!(res.n_starts, 3);
    assert!(!res.sigma_fixed);
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
fn calibration_multistart_reports_non_identifiability() {
    // The bundled noisy yields have two least-squares minima with rmse within
    // 2 % of each other, at (0.80, 0.050, 0.023) and (0.41, 0.069, 0.080):
    // the diagnostics must say so instead of claiming a unique fit.
    let (ts, ys) = load_zero_yields(&data_dir().join("zero_yields.csv")).unwrap();
    let res = calibrate_vasicek(&ts, &ys, 0.03, None, 4000, None).unwrap();
    assert!(res.converged && !res.at_bound);
    assert!(res.rmse < 2e-5);
    assert_eq!(res.n_starts, 3);
    assert!(!res.identified);
    assert!(res.sigma_spread > 0.01);
    assert!(res.kappa_spread > 0.1);
    assert!((0.04..=0.07).contains(&res.theta));
    let mean_y = ys.iter().sum::<f64>() / ys.len() as f64;
    let a = calibrate_vasicek(&ts, &ys, 0.03, Some([0.5, mean_y, 0.01]), 4000, None).unwrap();
    let b = calibrate_vasicek(&ts, &ys, 0.03, Some([0.5, 0.05, 0.05]), 4000, None).unwrap();
    assert_eq!(a.n_starts, 1);
    assert!(a.rmse < 2e-5 && b.rmse < 2e-5);
    assert!((a.rmse - b.rmse).abs() < 5e-6);
    assert!((a.sigma - b.sigma).abs() > 0.01);
    assert!(!a.identified); // single start: identifiability not assessed
}

#[test]
fn calibration_sigma_fixed_identifies_kappa_theta() {
    let (ts, ys) = load_zero_yields(&data_dir().join("zero_yields.csv")).unwrap();
    let res = calibrate_vasicek(&ts, &ys, 0.03, None, 4000, Some(0.02)).unwrap();
    assert!(res.sigma_fixed && res.sigma == 0.02);
    assert!(res.converged && res.identified && !res.at_bound);
    assert!((res.kappa - 0.8002161).abs() < 1e-6);
    assert!((res.theta - 0.0499880).abs() < 1e-6);
    assert!(res.kappa_spread < 1e-6 && res.sigma_spread == 0.0);
    let warm = calibrate_vasicek(&ts, &ys, 0.03, Some([0.3, 0.04, 99.0]), 4000, Some(0.02)).unwrap();
    assert!((warm.kappa - res.kappa).abs() < 1e-6);
    assert_eq!(warm.sigma, 0.02);
    assert!(calibrate_vasicek(&ts, &ys, 0.03, None, 4000, Some(-0.01)).is_err());
}

#[test]
fn calibration_penalty_start_not_reported_converged() {
    let (ts, ys) = load_zero_yields(&data_dir().join("zero_yields.csv")).unwrap();
    let res = calibrate_vasicek(&ts, &ys, 0.03, Some([-1.0, 0.0, -1.0]), 4000, None).unwrap();
    assert!(res.at_bound);
    assert!(!res.converged); // stopped at the penalty wall
    assert!(res.kappa <= 1e-5);
    assert!(res.rmse.is_finite());
}

#[test]
fn calibration_flat_yields_sigma_at_zero_bound() {
    let ts = [1.0, 2.0, 3.0, 5.0, 10.0];
    let ys = [0.03; 5];
    let res = calibrate_vasicek(&ts, &ys, 0.03, None, 4000, None).unwrap();
    assert!(res.at_bound && !res.converged);
    assert!(res.sigma < 1e-5);
    assert!((res.theta - 0.03).abs() < 1e-8);
    assert!(!res.identified);
}

#[test]
fn calibration_x0_non_finite_and_maxiter_rejected() {
    // x0 is a fixed-size array: the wrong-length case is a compile error.
    let (ts, ys) = ([1.0, 2.0, 3.0], [0.02, 0.021, 0.022]);
    assert!(calibrate_vasicek(&ts, &ys, 0.02, Some([0.5, f64::NAN, 0.01]), 100, None).is_err());
    assert!(calibrate_vasicek(&ts, &ys, 0.02, None, 0, None).is_err());
}

#[test]
fn mc_zcb_many_small_steps_unbiased() {
    // kappa*dt = 2.5e-6 per step: the naive var_i would be wrong by orders of
    // magnitude; with the series the estimator stays unbiased.
    let m = Vasicek::new(1e-3, 0.03, 0.01, 0.02).unwrap();
    let analytic = m.zcb_price(5.0, 0.0, None).unwrap();
    let (price, se) = m.mc_zcb(5.0, 2000, 10_000, 77).unwrap();
    assert!(se > 0.0);
    assert!((price - analytic).abs() < 3.0 * se, "MC {price} vs {analytic}, se {se}");
}

#[test]
fn same_seed_same_result_and_single_path() {
    let m = model();
    let a = m.mc_zcb(3.0, 4, 2000, 11).unwrap();
    let b = m.mc_zcb(3.0, 4, 2000, 11).unwrap();
    let c = m.mc_zcb(3.0, 4, 2000, 12).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
    let (price, se) = m.mc_zcb(1.0, 1, 1, 5).unwrap();
    assert!(price.is_finite() && price > 0.0);
    assert_eq!(se, 0.0);
}

#[test]
fn non_finite_and_overflow_rejected() {
    let m = model();
    assert!(m.zcb_price(f64::NAN, 0.0, None).is_err());
    assert!(m.zcb_price(5.0, 0.0, Some(f64::INFINITY)).is_err());
    assert!(m.r_mean(1.0, Some(f64::NAN)).is_err());
    assert!(m.zero_yield(f64::INFINITY, 0.0, None).is_err());
    assert!(m.zcb_price(5.0, 0.0, Some(-1e6)).is_err()); // exp overflow
}


#[test]
fn calibration_non_convergence_reported_not_error() {
    let truth = Vasicek::new(0.8, 0.05, 0.02, 0.03).unwrap();
    let ts = [1.0, 2.0, 5.0, 10.0];
    let ys: Vec<f64> = ts
        .iter()
        .map(|&t| truth.zero_yield(t, 0.0, None).unwrap())
        .collect();
    let res = calibrate_vasicek(&ts, &ys, 0.03, None, 2, None).unwrap();
    assert!(!res.converged); // reported, no error
    assert_eq!(res.iterations, 2);
    assert!(res.rmse.is_finite());
}

#[test]
fn calibration_input_validation() {
    assert!(calibrate_vasicek(&[1.0, 2.0], &[0.02, 0.03], 0.02, None, 100, None).is_err());
    assert!(
        calibrate_vasicek(&[1.0, -2.0, 3.0], &[0.02, 0.03, 0.03], 0.02, None, 100, None).is_err()
    );
    assert!(
        calibrate_vasicek(&[1.0, 2.0, 3.0], &[0.02, f64::NAN, 0.03], 0.02, None, 100, None)
            .is_err()
    );
    assert!(
        calibrate_vasicek(&[1.0, 2.0, 3.0], &[0.02, 0.02, 0.03], f64::NAN, None, 100, None)
            .is_err()
    );
}
