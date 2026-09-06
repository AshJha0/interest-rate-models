//! Discount curve: interpolation exactness, derived rates, validation.

use irm::{par_swap_rate, DiscountCurve, IrmError};

const TS: [f64; 4] = [1.0, 2.0, 3.0, 5.0];
const DFS: [f64; 4] = [0.99, 0.975, 0.955, 0.9];

fn curve() -> DiscountCurve {
    DiscountCurve::new(&TS, &DFS).unwrap()
}

#[test]
fn pillar_exactness_property() {
    // Interpolation must reproduce every pillar DF exactly (property loop).
    let c = curve();
    for (&t, &p) in TS.iter().zip(DFS.iter()) {
        assert!(
            (c.df(t).unwrap() - p).abs() < 1e-15,
            "DF({t}) != pillar value"
        );
    }
    assert_eq!(c.df(0.0).unwrap(), 1.0);
}

#[test]
fn log_linear_interpolation_between_pillars() {
    let c = curve();
    // ln DF linear on [2, 3]: manual midpoint value.
    let expected = (0.5 * (0.975_f64.ln() + 0.955_f64.ln())).exp();
    assert!((c.df(2.5).unwrap() - expected).abs() < 1e-15);
}

#[test]
fn flat_forward_extrapolation() {
    let c = curve();
    // Beyond 5y the last segment forward continues: ln DF(7) = ln DF(5) + 2*slope.
    let slope = (0.9_f64.ln() - 0.955_f64.ln()) / 2.0;
    let expected = (0.9_f64.ln() + 2.0 * slope).exp();
    assert!((c.df(7.0).unwrap() - expected).abs() < 1e-15);
    // Instantaneous forward beyond the last pillar is the last segment level.
    assert!((c.inst_forward(9.0).unwrap() - (-slope)).abs() < 1e-15);
}

#[test]
fn zero_rate_definition_and_origin_limit() {
    let c = curve();
    assert!((c.zero_rate(3.0).unwrap() - (-(0.955_f64.ln()) / 3.0)).abs() < 1e-15);
    // t = 0 returns the first segment's instantaneous forward.
    let f0 = -(0.99_f64.ln()) / 1.0;
    assert!((c.zero_rate(0.0).unwrap() - f0).abs() < 1e-15);
}

#[test]
fn forward_rate_consistency_from_dfs_grid() {
    let c = curve();
    // Property loop: F(t1,t2) must satisfy DF(t1) = DF(t2)(1 + F (t2-t1)).
    for &t1 in &[0.0, 0.5, 1.0, 2.0, 3.5] {
        for &dt in &[0.25, 1.0, 2.0] {
            let t2 = t1 + dt;
            let f = c.fwd_rate(t1, t2).unwrap();
            let lhs = c.df(t1).unwrap();
            let rhs = c.df(t2).unwrap() * (1.0 + f * dt);
            assert!((lhs - rhs).abs() < 1e-14, "F({t1},{t2}) inconsistent");
        }
    }
}

#[test]
fn negative_rates_df_above_one_supported() {
    let c = DiscountCurve::new(&[1.0, 2.0], &[1.005, 1.008]).unwrap();
    assert!(c.df(1.0).unwrap() > 1.0);
    assert!(c.df(1.5).unwrap() > 1.0);
    assert!(c.zero_rate(2.0).unwrap() < 0.0);
}

#[test]
fn df_positive_and_decreasing_for_positive_rate_curve() {
    let c = curve();
    let mut prev = c.df(0.0).unwrap();
    for i in 1..=60 {
        let t = i as f64 * 0.1;
        let d = c.df(t).unwrap();
        assert!(d > 0.0 && d < prev);
        prev = d;
    }
}

#[test]
fn par_swap_rate_helper() {
    let c = curve();
    let times = [0.0, 1.0, 2.0, 3.0];
    let annuity: f64 = (1..4).map(|i| c.df(i as f64).unwrap()).sum();
    let expected = (1.0 - c.df(3.0).unwrap()) / annuity;
    assert!((par_swap_rate(&c, &times).unwrap() - expected).abs() < 1e-15);
    assert!(par_swap_rate(&c, &[0.0]).is_err());
    assert!(par_swap_rate(&c, &[0.0, 2.0, 1.0]).is_err());
    assert!(par_swap_rate(&c, &[-1.0, 1.0]).is_err());
}

#[test]
fn invalid_construction_rejected() {
    assert!(matches!(
        DiscountCurve::new(&[], &[]),
        Err(IrmError::InvalidInput(_))
    ));
    assert!(DiscountCurve::new(&[1.0, 2.0], &[0.99]).is_err()); // length mismatch
    assert!(DiscountCurve::new(&[2.0, 1.0], &[0.99, 0.98]).is_err()); // non-monotone
    assert!(DiscountCurve::new(&[1.0, 1.0], &[0.99, 0.98]).is_err()); // duplicate
    assert!(DiscountCurve::new(&[0.0], &[1.0]).is_err()); // t must be > 0
    assert!(DiscountCurve::new(&[-1.0], &[0.99]).is_err());
    assert!(DiscountCurve::new(&[1.0], &[0.0]).is_err()); // DF <= 0
    assert!(DiscountCurve::new(&[1.0], &[-0.5]).is_err());
    assert!(DiscountCurve::new(&[f64::NAN], &[0.99]).is_err());
    assert!(DiscountCurve::new(&[1.0], &[f64::INFINITY]).is_err());
}

#[test]
fn invalid_queries_rejected() {
    let c = curve();
    assert!(c.df(-0.5).is_err());
    assert!(c.df(f64::NAN).is_err());
    assert!(c.df(f64::INFINITY).is_err());
    assert!(c.zero_rate(-1.0).is_err());
    assert!(c.inst_forward(-1.0).is_err());
    assert!(c.fwd_rate(2.0, 2.0).is_err());
    assert!(c.fwd_rate(3.0, 1.0).is_err());
    assert!(c.fwd_rate(-1.0, 1.0).is_err());
}

#[test]
fn inst_forward_uses_right_segment_at_pillar() {
    let c = curve();
    let f_23 = -((0.955_f64.ln() - 0.975_f64.ln()) / 1.0);
    // At t = 2.0 exactly, the right-hand segment [2, 3] applies.
    assert!((c.inst_forward(2.0).unwrap() - f_23).abs() < 1e-15);
}

#[test]
fn fwd_rate_underflow_is_standard_error() {
    let c = DiscountCurve::new(&[1.0], &[0.95]).unwrap();
    assert!(matches!(c.fwd_rate(0.0, 1e6), Err(IrmError::InvalidInput(_)))); // DF underflows
    assert!(c.fwd_rate(0.0, 500.0).unwrap().is_finite());
}
