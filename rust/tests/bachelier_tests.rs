//! Bachelier (normal) model: price, implied vol, parity, validation.

use irm::{bachelier_implied_vol, bachelier_price};

#[test]
fn atm_closed_form() {
    // ATM: price = A sigma sqrt(T) / sqrt(2 pi).
    let (a, vol, t) = (0.98_f64, 0.005_f64, 1.0_f64);
    let p = bachelier_price(-0.002, -0.002, t, vol, a, true).unwrap();
    assert!((p - a * vol * t.sqrt() / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-17);
    assert!((bachelier_price(-0.002, -0.002, t, vol, a, false).unwrap() - p).abs() < 1e-17);
    assert!((bachelier_implied_vol(p, -0.002, -0.002, t, a, true).unwrap() - vol).abs() < 1e-14);
}

#[test]
fn implied_vol_round_trip_grid() {
    let (t, annuity) = (2.5_f64, 3.7_f64);
    for forward in [-0.01, 0.0, 0.02, 0.05] {
        for strike in [-0.02, 0.0, 0.03] {
            for vol in [0.001, 0.0075, 0.02] {
                for payer in [true, false] {
                    let p = bachelier_price(forward, strike, t, vol, annuity, payer).unwrap();
                    let iv = bachelier_implied_vol(p, forward, strike, t, annuity, payer).unwrap();
                    let d = (forward - strike).abs() / (vol * t.sqrt());
                    if d < 4.0 {
                        assert!((iv - vol).abs() < 1e-12, "{forward} {strike} {vol} -> {iv}");
                    } else {
                        // Time value below rounding: only the price round-trips.
                        assert!((0.0..=vol * 1.001).contains(&iv));
                    }
                    let back = bachelier_price(forward, strike, t, iv, annuity, payer).unwrap();
                    assert!((back - p).abs() < 1e-15);
                }
            }
        }
    }
}

#[test]
fn payer_receiver_parity_and_monotonicity() {
    let (f, k, t, a) = (0.03, 0.025, 2.0, 4.2);
    for vol in [0.0, 0.002, 0.0075, 0.03] {
        let payer = bachelier_price(f, k, t, vol, a, true).unwrap();
        let receiver = bachelier_price(f, k, t, vol, a, false).unwrap();
        assert!((payer - receiver - a * (f - k)).abs() < 1e-15);
    }
    let prices: Vec<f64> = [0.001, 0.002, 0.005, 0.01, 0.02]
        .iter()
        .map(|&v| bachelier_price(f, k, t, v, a, true).unwrap())
        .collect();
    assert!(prices.windows(2).all(|w| w[1] > w[0]));
}

#[test]
fn intrinsic_and_below_intrinsic() {
    let (f, k, t, a) = (0.03, 0.025, 2.0, 4.2);
    let intrinsic = a * (f - k);
    assert_eq!(bachelier_price(f, k, t, 0.0, a, true).unwrap(), intrinsic);
    assert_eq!(bachelier_price(f, k, t, 0.0, a, false).unwrap(), 0.0);
    assert_eq!(bachelier_implied_vol(intrinsic, f, k, t, a, true).unwrap(), 0.0);
    assert_eq!(bachelier_implied_vol(0.0, f, k, t, a, false).unwrap(), 0.0);
    assert!(bachelier_implied_vol(intrinsic - 1e-6, f, k, t, a, true).is_err());
    assert!(bachelier_implied_vol(-1e-9, f, k, t, a, false).is_err());
}

#[test]
fn validation() {
    assert!(bachelier_price(f64::NAN, 0.02, 1.0, 0.01, 1.0, true).is_err());
    assert!(bachelier_price(0.02, 0.02, 0.0, 0.01, 1.0, true).is_err());
    assert!(bachelier_price(0.02, 0.02, 1.0, -0.01, 1.0, true).is_err());
    assert!(bachelier_price(0.02, 0.02, 1.0, 0.01, 0.0, true).is_err());
    assert!(bachelier_implied_vol(f64::NAN, 0.02, 0.02, 1.0, 1.0, true).is_err());
    assert!(bachelier_implied_vol(0.01, 0.02, 0.02, f64::INFINITY, 1.0, true).is_err());
    assert!(bachelier_implied_vol(1e300, 0.02, 0.02, 1.0, 1.0, true).is_err());
}
