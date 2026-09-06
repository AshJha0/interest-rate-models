//! Hull-White: curve fit, options, caps, Jamshidian, MC, theta(t).

use std::path::{Path, PathBuf};

use irm::{
    bootstrap, calibrate_hullwhite, load_curve_quotes, ou_integral_variance, CapletQuote,
    DiscountCurve, HullWhite, IrmError, SwaptionQuote, JAMSHIDIAN_RESIDUAL_TOL,
};

const A: f64 = 0.1;
const SIGMA: f64 = 0.01;

fn curve() -> DiscountCurve {
    let ts: [f64; 5] = [1.0, 2.0, 3.0, 5.0, 10.0];
    let dfs: Vec<f64> = ts.iter().map(|&t| (-(0.02 + 0.0015 * t) * t).exp()).collect();
    DiscountCurve::new(&ts, &dfs).unwrap()
}

fn hw() -> HullWhite {
    HullWhite::new(A, SIGMA, curve()).unwrap()
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")
}

fn eur_curve() -> DiscountCurve {
    bootstrap(&load_curve_quotes(&data_dir().join("curve_quotes.csv"), "EUR").unwrap()).unwrap()
}

#[test]
fn zcb_t0_matches_market_exactly_grid() {
    // The e^{B f(0,0)} and e^{-B r0} factors cancel: P(0,T) equals the
    // market DF to machine precision, at and between pillars.
    let hw = hw();
    let c = curve();
    for t in [0.5, 1.0, 2.0, 2.5, 3.0, 5.0, 7.0, 10.0] {
        let p = hw.zcb_price(0.0, t, None).unwrap();
        let m = c.df(t).unwrap();
        assert!((p - m).abs() / m < 1e-13, "P(0,{t}) = {p} vs market {m}");
    }
}

#[test]
fn zcb_t0_matches_bootstrapped_negative_curve() {
    let eur = eur_curve();
    let hw = HullWhite::new(0.05, 0.008, eur.clone()).unwrap();
    for t in [1.0, 5.0, 10.0, 30.0] {
        let p = hw.zcb_price(0.0, t, None).unwrap();
        let m = eur.df(t).unwrap();
        assert!((p - m).abs() / m < 1e-13);
    }
    // Negative-rate curve: DF > 1 flows straight through the model.
    assert!(hw.zcb_price(0.0, 1.0, None).unwrap() > 1.0);
}

#[test]
fn zcb_strictly_decreasing_in_r() {
    // Strict monotonicity in r is what Jamshidian's decomposition relies on.
    let hw = hw();
    let prices: Vec<f64> = [-0.02, 0.0, 0.02, 0.05, 0.10]
        .iter()
        .map(|&r| hw.zcb_price(1.0, 5.0, Some(r)).unwrap())
        .collect();
    for w in prices.windows(2) {
        assert!(w[1] < w[0]);
    }
}

#[test]
fn fwd0_piecewise_and_pillar_average() {
    let hw = hw();
    let c = curve();
    // Inside a segment the FD reproduces the constant forward exactly.
    let f_mid = hw.fwd0(1.5).unwrap();
    assert!((f_mid - c.inst_forward(1.5).unwrap()).abs() < 1e-9 * f_mid.abs().max(1.0));
    // Straddling a pillar it returns the average of adjacent forwards.
    let left = c.inst_forward(1.5).unwrap();
    let right = c.inst_forward(2.5).unwrap();
    let avg = 0.5 * (left + right);
    assert!((hw.fwd0(2.0).unwrap() - avg).abs() < 1e-9);
}

#[test]
fn theta_on_flat_forward_curve() {
    // Flat continuous curve at 3%: f(0,t) = 0.03 for all t, so
    // theta(t) = a*0.03 + sigma^2/(2a) (1 - e^{-2at}).
    let flat = DiscountCurve::new(&[10.0], &[(-0.03_f64 * 10.0).exp()]).unwrap();
    let hw = HullWhite::new(A, SIGMA, flat).unwrap();
    for t in [0.5, 1.0, 3.0] {
        let expected = A * 0.03 + SIGMA * SIGMA / (2.0 * A) * (1.0 - (-2.0 * A * t).exp());
        assert!((hw.theta(t).unwrap() - expected).abs() < 1e-6);
    }
    assert!((hw.alpha(0.0).unwrap() - 0.03).abs() < 1e-9);
}

#[test]
fn zb_option_put_call_parity_grid() {
    let hw = hw();
    let c = curve();
    for (expiry, s_mat) in [(0.5, 2.0), (1.0, 3.0), (2.0, 5.0)] {
        for k in [0.85, 0.92, 0.96, 1.00] {
            let lhs = hw.zbc(expiry, s_mat, k).unwrap() - hw.zbp(expiry, s_mat, k).unwrap();
            let rhs = c.df(s_mat).unwrap() - k * c.df(expiry).unwrap();
            assert!((lhs - rhs).abs() < 1e-13, "parity broken at K={k}, T={expiry}");
        }
    }
}

#[test]
fn caplet_floorlet_parity_is_fra_pv() {
    // caplet - floorlet = notional * tau * DF(pay) * (F - K).
    let hw = hw();
    let c = curve();
    let (reset, pay, k, notional) = (1.0, 2.0, 0.025, 100.0);
    let tau = pay - reset;
    let lhs = hw.caplet(reset, pay, k, notional).unwrap()
        - hw.floorlet(reset, pay, k, notional).unwrap();
    let f = c.fwd_rate(reset, pay).unwrap();
    let rhs = notional * tau * c.df(pay).unwrap() * (f - k);
    assert!((lhs - rhs).abs() < 1e-10);
}

#[test]
fn cap_equals_sum_of_caplets() {
    let hw = hw();
    let sched = [0.5, 1.0, 2.0, 3.0];
    let strip: f64 = sched
        .windows(2)
        .map(|w| hw.caplet(w[0], w[1], 0.024, 100.0).unwrap())
        .sum();
    let cap = hw.cap(&sched, 0.024, 100.0).unwrap();
    assert!((cap - strip).abs() < 1e-12);
}

#[test]
fn sigma_zero_caplet_is_discounted_intrinsic() {
    let hw0 = HullWhite::new(A, 0.0, curve()).unwrap();
    let c = curve();
    let (reset, pay, k, notional) = (1.0, 2.0, 0.02, 100.0);
    let tau = pay - reset;
    let f = c.fwd_rate(reset, pay).unwrap();
    let intrinsic = notional * c.df(pay).unwrap() * tau * (f - k).max(0.0);
    assert!((hw0.caplet(reset, pay, k, notional).unwrap() - intrinsic).abs() < 1e-12);
    // Deep OTM: strike far above the forward is worthless with sigma = 0.
    assert_eq!(hw0.caplet(reset, pay, 0.5, notional).unwrap(), 0.0);
}

#[test]
fn jamshidian_root_residual_and_value() {
    let hw = hw();
    let pay_times = [2.0, 3.0, 4.0, 5.0, 6.0];
    let res = hw.jamshidian_swaption(1.0, &pay_times, 0.028, 100.0, true).unwrap();
    assert!(res.residual.abs() < 1e-10);
    assert!(res.value > 0.0);
    assert_eq!(res.strikes.len(), pay_times.len());
    // Strikes are the ZCB prices at r*: decreasing in maturity for r* > 0.
    for w in res.strikes.windows(2) {
        assert!(w[1] < w[0]);
    }
}

#[test]
fn swaption_at_least_intrinsic_and_parity() {
    let hw = hw();
    let c = curve();
    let expiry = 1.0;
    let pay_times = [2.0, 3.0, 4.0, 5.0];
    let fixed = 0.028;
    // Payer - receiver = PV(floating) - PV(fixed) = value of the fwd swap.
    let payer = hw.jamshidian_swaption(expiry, &pay_times, fixed, 1.0, true).unwrap();
    let receiver = hw.jamshidian_swaption(expiry, &pay_times, fixed, 1.0, false).unwrap();
    let mut annuity = 0.0;
    let mut prev = expiry;
    for &t in &pay_times {
        annuity += (t - prev) * c.df(t).unwrap();
        prev = t;
    }
    let swap_pv =
        c.df(expiry).unwrap() - c.df(*pay_times.last().unwrap()).unwrap() - fixed * annuity;
    assert!((payer.value - receiver.value - swap_pv).abs() < 1e-12);
    // Swaption >= intrinsic (forward swap PV floored at zero).
    assert!(payer.value >= swap_pv.max(0.0) - 1e-14);
    assert!(receiver.value >= (-swap_pv).max(0.0) - 1e-14);
}

#[test]
fn jamshidian_on_negative_rate_curve() {
    let eur = eur_curve();
    let hw = HullWhite::new(0.05, 0.008, eur).unwrap();
    let res = hw.jamshidian_swaption(1.0, &[2.0, 3.0, 4.0], 0.0, 100.0, true).unwrap();
    assert!(res.residual.abs() < 1e-10);
    assert!(res.value > 0.0);
    assert!(res.r_star < 0.05); // negative-rate environment keeps r* low
}

#[test]
fn mc_caplet_within_3se_of_analytic() {
    let hw = hw();
    let analytic = hw.caplet(1.0, 2.0, 0.025, 100.0).unwrap();
    let (price, se) = hw.mc_caplet(1.0, 2.0, 0.025, 100.0, 4, 40_000, 5).unwrap();
    assert!(se > 0.0);
    assert!(
        (price - analytic).abs() < 3.0 * se,
        "MC {price} vs analytic {analytic}, se {se}"
    );
    // Exactness: a single step is also unbiased.
    let (p1, se1) = hw.mc_caplet(1.0, 2.0, 0.025, 100.0, 1, 40_000, 6).unwrap();
    assert!((p1 - analytic).abs() < 3.0 * se1);
}

#[test]
fn invalid_inputs_rejected() {
    assert!(HullWhite::new(0.0, 0.01, curve()).is_err());
    assert!(HullWhite::new(0.1, -0.01, curve()).is_err());
    let hw = hw();
    assert!(hw.zbc(1.0, 3.0, 0.0).is_err());
    assert!(hw.zbc(3.0, 1.0, 0.9).is_err());
    assert!(hw.caplet(2.0, 1.0, 0.02, 1.0).is_err()); // pay before reset
    assert!(hw.caplet(1.0, 2.0, -1.5, 1.0).is_err()); // 1 + K tau <= 0
    assert!(hw.cap(&[1.0], 0.02, 1.0).is_err()); // schedule too short
    assert!(hw.jamshidian_swaption(0.0, &[1.0], 0.02, 1.0, true).is_err());
    assert!(hw.jamshidian_swaption(1.0, &[], 0.02, 1.0, true).is_err());
    assert!(hw.jamshidian_swaption(1.0, &[0.5], 0.02, 1.0, true).is_err());
    assert!(hw.fwd0(-1.0).is_err());
    assert!(hw.mc_caplet(0.0, 1.0, 0.02, 1.0, 4, 10, 1).is_err());
}

// ---- validation of notional / strike / schedule (MAJOR-5) --------------------

#[test]
fn non_finite_notional_and_mc_strike_rejected() {
    let hw = hw();
    assert!(hw.caplet(1.0, 2.0, 0.025, f64::NAN).is_err());
    assert!(hw.floorlet(1.0, 2.0, 0.025, f64::INFINITY).is_err());
    assert!(hw.cap(&[1.0, 2.0, 3.0], 0.02, f64::INFINITY).is_err());
    assert!(hw.jamshidian_swaption(1.0, &[2.0, 3.0], 0.02, f64::NAN, true).is_err());
    assert!(hw.mc_caplet(1.0, 2.0, f64::NAN, 100.0, 8, 100, 1).is_err());
    assert!(hw.mc_caplet(1.0, 2.0, -1.5, 100.0, 8, 100, 1).is_err());
    assert!(hw.mc_caplet(1.0, 2.0, 0.02, f64::INFINITY, 8, 100, 1).is_err());
    assert!(hw.caplet(f64::NAN, 2.0, 0.02, 1.0).is_err());
    assert!(hw.b_factor(f64::NAN, 5.0).is_err());
    assert!(hw.b_factor(-1.0, 5.0).is_err());
    assert!(hw.jamshidian_swaption(1.0, &[2.0, f64::INFINITY], 0.02, 1.0, true).is_err());
    assert!(hw.theta_with_step(1.0, 0.0).is_err());
    assert!(hw.zcb_price(1.0, 5.0, Some(-1e6)).is_err()); // exp overflow
    // A negative notional is a short position, not an error.
    let long = hw.caplet(1.0, 2.0, 0.025, 100.0).unwrap();
    assert!((hw.caplet(1.0, 2.0, 0.025, -100.0).unwrap() + long).abs() < 1e-15);
}

#[test]
fn cap_schedule_non_increasing_rejected() {
    let hw = hw();
    assert!(hw.cap(&[1.0, 2.0, 2.0, 3.0], 0.02, 1.0).is_err());
    assert!(hw.cap(&[1.0, f64::NAN, 3.0], 0.02, 1.0).is_err());
    assert!(hw.cap(&[-1.0, 1.0], 0.02, 1.0).is_err());
    assert!(hw.cap(&[1.0, 2.0], f64::NAN, 1.0).is_err());
}

#[test]
fn theta_with_step_matches_default() {
    let hw = hw();
    assert_eq!(hw.theta(0.5).unwrap(), hw.theta_with_step(0.5, 1e-4).unwrap());
}

// ---- Jamshidian contract (MAJOR-6, MINOR-4) ----------------------------------

#[test]
fn jamshidian_residual_contract_enforced() {
    let hw = hw();
    let pays = [2.0, 3.0, 4.0, 5.0, 6.0];
    let res = hw.jamshidian_swaption(1.0, &pays, 0.028, 100.0, true).unwrap();
    let again = hw.jamshidian_at_r_star(1.0, &pays, 0.028, res.r_star, 100.0, true).unwrap();
    assert!((again.value - res.value).abs() < 1e-14);
    assert_eq!(again.residual, res.residual);
    assert!(again.residual.abs() < JAMSHIDIAN_RESIDUAL_TOL);
    // A wrong r* violates |g(r*)| < 1e-10 -> RootFind error.
    assert!(matches!(
        hw.jamshidian_at_r_star(1.0, &pays, 0.028, -1.0, 100.0, true),
        Err(IrmError::RootFind(_))
    ));
    assert!(matches!(
        hw.jamshidian_at_r_star(1.0, &pays, 0.028, res.r_star + 1e-6, 100.0, true),
        Err(IrmError::RootFind(_))
    ));
    assert!(matches!(
        hw.jamshidian_at_r_star(1.0, &pays, 0.028, f64::NAN, 100.0, true),
        Err(IrmError::InvalidInput(_))
    ));
    let rec = hw.jamshidian_at_r_star(1.0, &pays, 0.028, res.r_star, 100.0, false).unwrap();
    let rec2 = hw.jamshidian_swaption(1.0, &pays, 0.028, 100.0, false).unwrap();
    assert!((rec.value - rec2.value).abs() < 1e-14);
}

/// Payer swaption per unit notional by direct integration of the payoff
/// (1 - g(r_T))^+ under the T-forward Gaussian law of r_T (Simpson from the
/// payoff kink to mu + 12 sd) — independent of the decomposition.
fn swaption_by_integration(hw: &HullWhite, expiry: f64, pays: &[f64], x: f64) -> f64 {
    let n = 2000;
    let c = hw.curve();
    let v = hw.sigma() * hw.sigma() * (-(-2.0 * hw.a() * expiry).exp_m1()) / (2.0 * hw.a());
    let b1 = hw.b_factor(expiry, pays[0]).unwrap();
    let a1 = hw.a_factor(expiry, pays[0]).unwrap();
    let mu = ((a1 * c.df(expiry).unwrap() / c.df(pays[0]).unwrap()).ln() + b1 * b1 * v / 2.0) / b1;
    let mut coupons = Vec::new();
    let mut prev = expiry;
    for &t in pays {
        coupons.push(x * (t - prev));
        prev = t;
    }
    let last = coupons.len() - 1;
    coupons[last] += 1.0;
    let sd = v.sqrt();
    let payoff = |r: f64| -> f64 {
        let g: f64 = coupons
            .iter()
            .zip(pays.iter())
            .map(|(ci, &t)| ci * hw.zcb_price(expiry, t, Some(r)).unwrap())
            .sum();
        (1.0 - g).max(0.0)
    };
    let (mut lo, mut hi) = (mu - 12.0 * sd, mu + 12.0 * sd);
    while lo < mu - 400.0 * sd || payoff(lo) > 0.0 {
        lo -= 12.0 * sd;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if payoff(mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let (kink, top) = (hi, mu + 12.0 * sd);
    let h = (top - kink) / n as f64;
    let mut total = 0.0;
    for i in 0..=n {
        let r = kink + i as f64 * h;
        let w = (-0.5 * ((r - mu) / sd).powi(2)).exp() / (sd * (2.0 * std::f64::consts::PI).sqrt());
        let coef = if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
        total += coef * payoff(r) * w;
    }
    c.df(expiry).unwrap() * total * h / 3.0
}

#[test]
fn jamshidian_negative_fixed_rate_stays_exact() {
    // For X < 0 the coupon signs are (-, ..., -, +): g is not monotone but
    // g - 1 still has a single crossing, so the decomposition is exact;
    // 1 + X tau_n <= 0 is the genuinely degenerate case.
    let hw = hw();
    let pays = [2.0, 3.0, 4.0, 5.0, 6.0];
    for x in [-0.005, -0.2, -0.5] {
        let res = hw.jamshidian_swaption(1.0, &pays, x, 1.0, true).unwrap();
        assert!(res.value > 0.0);
        assert!(res.residual.abs() < 1e-10);
        let by_int = swaption_by_integration(&hw, 1.0, &pays, x);
        assert!((res.value / by_int - 1.0).abs() < 1e-9, "X={x}: {} vs {by_int}", res.value);
    }
    assert!(hw.jamshidian_swaption(1.0, &pays, -0.005, 1.0, true).unwrap().r_star < 0.0);
    assert!(matches!(
        hw.jamshidian_swaption(1.0, &pays, -1.0, 1.0, true),
        Err(IrmError::InvalidInput(_))
    ));
    assert!(hw.jamshidian_swaption(1.0, &pays, -1.5, 1.0, true).is_err());
    let ok = hw.jamshidian_swaption(1.0, &pays, -0.8, 1.0, true).unwrap();
    assert!(ok.residual.abs() < 1e-10);
    // Beyond about -0.95 the coupon-bond terms are O(1e8+) and |g(r*)| cannot
    // beat 1e-10 in double precision: the contract rejects it honestly.
    assert!(matches!(
        hw.jamshidian_swaption(1.0, &pays, -0.99, 1.0, true),
        Err(IrmError::RootFind(_))
    ));
    let pos = hw.jamshidian_swaption(1.0, &pays, 0.028, 1.0, true).unwrap();
    assert!((pos.value / swaption_by_integration(&hw, 1.0, &pays, 0.028) - 1.0).abs() < 1e-9);
}

// ---- Monte Carlo determinism / small mean reversion ---------------------------

#[test]
fn mc_caplet_same_seed_and_single_path() {
    let hw = hw();
    let a = hw.mc_caplet(1.0, 2.0, 0.025, 100.0, 4, 500, 3).unwrap();
    let b = hw.mc_caplet(1.0, 2.0, 0.025, 100.0, 4, 500, 3).unwrap();
    let c = hw.mc_caplet(1.0, 2.0, 0.025, 100.0, 4, 500, 4).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
    let (price, se) = hw.mc_caplet(1.0, 2.0, 0.02, 100.0, 1, 1, 1).unwrap();
    assert!(price.is_finite() && price >= 0.0);
    assert_eq!(se, 0.0);
}

#[test]
fn mc_caplet_tiny_mean_reversion_unbiased() {
    // a*dt = 5e-7 per step exercises the series branch of var_i and V(t).
    let slow = HullWhite::new(1e-4, 0.01, curve()).unwrap();
    let analytic = slow.caplet(1.0, 2.0, 0.025, 100.0).unwrap();
    let (price, se) = slow.mc_caplet(1.0, 2.0, 0.025, 100.0, 200, 20_000, 9).unwrap();
    assert!((price - analytic).abs() < 3.0 * se, "MC {price} vs {analytic}, se {se}");
}

#[test]
fn variance_integral_uses_cancellation_free_kernel() {
    let slow = HullWhite::new(1e-5, 0.02, curve()).unwrap();
    let v = slow.variance_integral(1.0).unwrap();
    assert!((v - 0.5 * ou_integral_variance(1e-5, 0.02, 1.0).unwrap()).abs() < 1e-20);
    assert!((v / (0.02_f64 * 0.02 / 6.0) - 1.0).abs() < 1e-5);
    assert_eq!(slow.variance_integral(0.0).unwrap(), 0.0);
}

// ---- calibration of (a, sigma) (MAJOR-7) --------------------------------------

fn quotes(truth: &HullWhite) -> (Vec<CapletQuote>, Vec<SwaptionQuote>) {
    let caps: Vec<CapletQuote> = [1.0, 2.0, 4.0]
        .iter()
        .map(|&r| CapletQuote::new(r, r + 1.0, 0.025, truth.caplet(r, r + 1.0, 0.025, 1.0).unwrap()).unwrap())
        .collect();
    let pays = [2.0, 3.0, 4.0, 5.0, 6.0];
    let swp = SwaptionQuote::new(
        1.0,
        &pays,
        0.028,
        truth.jamshidian_swaption(1.0, &pays, 0.028, 1.0, true).unwrap().value,
    )
    .unwrap();
    (caps, vec![swp])
}

#[test]
fn calibrate_hullwhite_recovers_a_and_sigma() {
    let truth = HullWhite::new(0.08, 0.012, curve()).unwrap();
    let (caps, swps) = quotes(&truth);
    let cal = calibrate_hullwhite(&curve(), &caps, &swps, None, None, 4000).unwrap();
    assert!(cal.converged && cal.identified && !cal.at_bound && !cal.a_fixed);
    assert_eq!(cal.n_starts, 3);
    assert!((cal.a - 0.08).abs() < 1e-8, "a = {}", cal.a);
    assert!((cal.sigma - 0.012).abs() < 1e-9, "sigma = {}", cal.sigma);
    assert!(cal.rmse < 1e-10);
    let model = cal.model(curve()).unwrap();
    for q in &caps {
        assert!((model.caplet(q.reset, q.pay, q.strike, 1.0).unwrap() - q.price).abs() < 1e-10);
    }
}

#[test]
fn calibrate_hullwhite_a_fixed_and_warm_start() {
    let truth = HullWhite::new(0.08, 0.012, curve()).unwrap();
    let (caps, swps) = quotes(&truth);
    let cal = calibrate_hullwhite(&curve(), &caps, &swps, Some(0.08), None, 4000).unwrap();
    assert!(cal.a_fixed && cal.a == 0.08);
    assert!(cal.converged && cal.identified);
    assert!((cal.sigma - 0.012).abs() < 1e-9);
    let warm = calibrate_hullwhite(&curve(), &caps, &swps, None, Some([0.2, 0.02]), 4000).unwrap();
    assert_eq!(warm.n_starts, 1);
    assert!(!warm.identified && warm.converged);
    assert!((warm.a - 0.08).abs() < 1e-6);
    assert!((warm.sigma - 0.012).abs() < 1e-8);
    let only_caps = calibrate_hullwhite(&curve(), &caps, &[], None, None, 4000).unwrap();
    assert!((only_caps.sigma - 0.012).abs() < 1e-8);
    let only_swp = calibrate_hullwhite(&curve(), &[], &swps, Some(0.08), None, 4000).unwrap();
    assert!((only_swp.sigma - 0.012).abs() < 1e-8);
}

#[test]
fn calibrate_hullwhite_single_quote_not_identified() {
    let truth = HullWhite::new(0.08, 0.012, curve()).unwrap();
    let (caps, _) = quotes(&truth);
    let cal = calibrate_hullwhite(&curve(), &caps[..1], &[], None, None, 4000).unwrap();
    assert!(cal.converged);
    assert!(!cal.identified); // one price, two parameters: a ridge
    assert!(cal.a_spread > 1e-2);
    assert!(cal.rmse < 1e-9);
}

#[test]
fn calibrate_hullwhite_non_convergence_and_validation() {
    let truth = HullWhite::new(0.08, 0.012, curve()).unwrap();
    let (caps, swps) = quotes(&truth);
    let res = calibrate_hullwhite(&curve(), &caps, &swps, None, None, 2).unwrap();
    assert!(!res.converged && res.rmse.is_finite());
    assert!(calibrate_hullwhite(&curve(), &[], &[], None, None, 4000).is_err());
    assert!(calibrate_hullwhite(&curve(), &caps, &swps, Some(0.0), None, 4000).is_err());
    assert!(calibrate_hullwhite(&curve(), &caps, &swps, None, Some([f64::NAN, 0.01]), 4000).is_err());
    assert!(calibrate_hullwhite(&curve(), &caps, &swps, None, None, 0).is_err());
    assert!(CapletQuote::new(1.0, 2.0, 0.02, f64::NAN).is_err());
    assert!(CapletQuote::new(2.0, 1.0, 0.02, 0.001).is_err());
    assert!(CapletQuote::new(1.0, 2.0, 0.02, -0.001).is_err());
    assert!(SwaptionQuote::new(1.0, &[0.5, 2.0], 0.02, 0.001).is_err());
    assert!(SwaptionQuote::new(1.0, &[], 0.02, 0.001).is_err());
    assert!(SwaptionQuote::new(f64::INFINITY, &[2.0], 0.02, 0.001).is_err());
}
