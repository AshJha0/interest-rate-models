//! Bootstrap: round-trip repricing, ordering validation, crossed quotes,
//! gap handling, negative-rate curves.

use std::path::{Path, PathBuf};

use irm::{
    annual_schedule, bootstrap, load_curve_quotes, load_ois_quotes, par_swap_rate,
    solve_pillar_df, DiscountCurve, Instrument, IrmError, MAX_MATURITY, MIN_MATURITY,
};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")
}

fn eur_instruments() -> Vec<Instrument> {
    load_curve_quotes(&data_dir().join("curve_quotes.csv"), "EUR").unwrap()
}

fn usd_instruments() -> Vec<Instrument> {
    load_curve_quotes(&data_dir().join("curve_quotes.csv"), "USD").unwrap()
}

fn reprice_all(instruments: &[Instrument], curve: &DiscountCurve) {
    for ins in instruments {
        let res = ins.residual(curve).unwrap();
        assert!(
            res.abs() < 1e-9,
            "instrument at pillar {} misprices by {res}",
            ins.pillar()
        );
    }
}

#[test]
fn eur_round_trip_reprices_every_instrument() {
    let ins = eur_instruments();
    let curve = bootstrap(&ins).unwrap();
    reprice_all(&ins, &curve);
    // Negative short end: DF(1y) above par must be supported.
    assert!(curve.df(1.0).unwrap() > 1.0);
}

#[test]
fn usd_round_trip_reprices_every_instrument() {
    let ins = usd_instruments();
    let curve = bootstrap(&ins).unwrap();
    reprice_all(&ins, &curve);
    // Normal curve: DF decreasing at pillars.
    let dfs = curve.dfs();
    for w in dfs.windows(2) {
        assert!(w[1] < w[0]);
    }
}

#[test]
fn ois_round_trip_reprices_every_instrument() {
    let ins = load_ois_quotes(&data_dir().join("ois_quotes.csv")).unwrap();
    let curve = bootstrap(&ins).unwrap();
    reprice_all(&ins, &curve);
}

#[test]
fn bootstrapped_curve_reproduces_input_par_rates() {
    // The par swap rate read back off the curve equals the input quote.
    let ins = usd_instruments();
    let curve = bootstrap(&ins).unwrap();
    for ins in &ins {
        if let Instrument::Swap { maturity, rate } = ins {
            let n = maturity.round() as i32;
            let mut times = vec![0.0];
            times.extend((1..=n).map(f64::from));
            let par = par_swap_rate(&curve, &times).unwrap();
            assert!(
                (par - rate).abs() < 1e-10,
                "par({maturity}) = {par} vs quote {rate}"
            );
        }
    }
}

#[test]
fn deposit_only_short_end_and_swap_only_long_end() {
    // Gap between 1y deposit and 5y swap: intermediate cashflow dates (1..4y)
    // interpolate on the trial curve; the finished curve reprices both.
    let ins = vec![
        Instrument::deposit(1.0, 0.03).unwrap(),
        Instrument::swap(5.0, 0.035).unwrap(),
    ];
    let curve = bootstrap(&ins).unwrap();
    reprice_all(&ins, &curve);
    assert_eq!(curve.times(), &[1.0, 5.0]);
}

#[test]
fn fra_bootstrap_supported() {
    let ins = vec![
        Instrument::deposit(0.5, 0.02).unwrap(),
        Instrument::fra(0.5, 1.0, 0.025).unwrap(),
        Instrument::swap(2.0, 0.024).unwrap(),
    ];
    let curve = bootstrap(&ins).unwrap();
    reprice_all(&ins, &curve);
    // FRA forward read back from the curve equals the quote.
    assert!((curve.fwd_rate(0.5, 1.0).unwrap() - 0.025).abs() < 1e-10);
}

#[test]
fn ordering_violations_rejected() {
    let a = Instrument::deposit(1.0, 0.02).unwrap();
    let b = Instrument::swap(2.0, 0.025).unwrap();
    assert!(matches!(
        bootstrap(&[b, a]),
        Err(IrmError::InvalidInput(_))
    )); // out of order
    let dup = Instrument::deposit(1.0, 0.021).unwrap();
    assert!(bootstrap(&[a, dup]).is_err()); // duplicate pillar
    assert!(bootstrap(&[]).is_err()); // empty
}

#[test]
fn crossed_quotes_produce_bootstrap_error() {
    // A 2y swap paying 300% cannot be repriced by any admissible DF chain:
    // the implied DF(2) would have to be negative.
    let ins = vec![
        Instrument::deposit(1.0, 0.02).unwrap(),
        Instrument::swap(2.0, 3.0).unwrap(),
    ];
    match bootstrap(&ins) {
        Err(IrmError::Bootstrap(msg)) => {
            assert!(msg.contains("t=2"), "message should name the pillar: {msg}");
            assert!(msg.contains("crossed"));
        }
        other => panic!("expected Bootstrap error, got {other:?}"),
    }
}

#[test]
fn non_convergence_message_not_crossed_quotes() {
    // A solver failure that is not a bracketing failure must not be
    // relabelled as crossed quotes (MINOR-10).
    match solve_pillar_df(|x| Ok(if x > 1.0 { f64::INFINITY } else { -1.0 }), 2.0, "test") {
        Err(IrmError::Bootstrap(msg)) => {
            assert!(msg.contains("solver failed") && msg.contains("not finite"), "{msg}");
            assert!(!msg.contains("crossed"));
        }
        other => panic!("expected Bootstrap error, got {other:?}"),
    }
    match solve_pillar_df(|x| Ok(1.0 + x), 2.0, "test") {
        Err(IrmError::Bootstrap(msg)) => assert!(msg.contains("crossed")),
        other => panic!("expected Bootstrap error, got {other:?}"),
    }
    let df = solve_pillar_df(|x| Ok(x - 0.9), 2.0, "test").unwrap();
    assert!((df - 0.9).abs() < 1e-13);
}

#[test]
fn swap_maturity_out_of_range_rejected() {
    // 1e12 used to panic with a Vec capacity overflow; now Err, never panic.
    for bad in [1e12, 1e9, MAX_MATURITY * (1.0 + 1e-9), 1e-13, MIN_MATURITY / 2.0, f64::INFINITY] {
        assert!(Instrument::swap(bad, 0.03).is_err(), "{bad}");
        assert!(Instrument::ois_swap(bad, 0.03).is_err(), "{bad}");
        assert!(Instrument::deposit(bad, 0.03).is_err(), "{bad}");
        assert!(Instrument::fra(0.0, bad, 0.03).is_err(), "{bad}");
        assert!(annual_schedule(bad).is_err(), "{bad}");
    }
    assert_eq!(annual_schedule(MAX_MATURITY).unwrap().len(), 200);
    assert_eq!(annual_schedule(MIN_MATURITY).unwrap(), vec![MIN_MATURITY]);
    assert_eq!(annual_schedule(199.5).unwrap().len(), 200);
}

#[test]
fn fra_strip_reprices() {
    let ins = vec![
        Instrument::deposit(0.5, 0.030).unwrap(),
        Instrument::fra(0.5, 1.0, 0.032).unwrap(),
        Instrument::fra(1.0, 1.5, 0.034).unwrap(),
        Instrument::swap(3.0, 0.033).unwrap(),
    ];
    let curve = bootstrap(&ins).unwrap();
    reprice_all(&ins, &curve);
    assert!((curve.fwd_rate(0.5, 1.0).unwrap() - 0.032).abs() < 1e-12);
}

#[test]
fn ois_sub_year_maturity() {
    let c = bootstrap(&[Instrument::ois_swap(1.0 / 12.0, 0.03).unwrap()]).unwrap();
    assert!((c.df(1.0 / 12.0).unwrap() - 1.0 / (1.0 + 0.03 / 12.0)).abs() < 1e-12);
}

#[test]
fn dv01_localises_in_quote_space() {
    let quotes = usd_instruments();
    let times: Vec<f64> = (0..=10).map(f64::from).collect();
    let bumped = |pillar: f64| -> DiscountCurve {
        let out: Vec<Instrument> = quotes
            .iter()
            .map(|q| {
                let bump = if q.pillar() == pillar { 1e-4 } else { 0.0 };
                match *q {
                    Instrument::Deposit { maturity, rate } => {
                        Instrument::deposit(maturity, rate + bump).unwrap()
                    }
                    Instrument::Swap { maturity, rate } => {
                        Instrument::swap(maturity, rate + bump).unwrap()
                    }
                    other => other,
                }
            })
            .collect();
        bootstrap(&out).unwrap()
    };
    assert!((par_swap_rate(&bootstrap(&quotes).unwrap(), &times).unwrap() - 0.0425).abs() < 1e-12);
    assert!((par_swap_rate(&bumped(5.0), &times).unwrap() - 0.0425).abs() < 1e-12);
    assert!((par_swap_rate(&bumped(10.0), &times).unwrap() - 0.0425 - 1e-4).abs() < 1e-12);
}

#[test]
fn negative_rate_instruments_bootstrap_cleanly() {
    let ins = vec![
        Instrument::deposit(0.5, -0.006).unwrap(),
        Instrument::deposit(1.0, -0.005).unwrap(),
        Instrument::swap(2.0, -0.004).unwrap(),
    ];
    let curve = bootstrap(&ins).unwrap();
    reprice_all(&ins, &curve);
    for &df in curve.dfs() {
        assert!(df > 1.0); // all-negative curve: every DF above par
    }
}
