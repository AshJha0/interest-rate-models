//! Hull-White: curve fit, options, caps, Jamshidian, MC, theta(t).

use std::path::{Path, PathBuf};

use irm::{bootstrap, load_curve_quotes, DiscountCurve, HullWhite};

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
