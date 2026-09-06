//! End-to-end demo for the `irm` crate.
//!
//! Bootstraps the EUR (negative short end) and USD (normal) curves plus
//! the OIS curve, prints zero/forward tables, calibrates a Vasicek model
//! to the bundled zero yields, and prices Hull-White caps and a Jamshidian
//! payer swaption.  Run:  `cd rust && cargo run --bin demo`

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use irm::{
    bachelier_implied_vol, bootstrap, calibrate_hullwhite, calibrate_vasicek, load_curve_quotes,
    load_ois_quotes, load_zero_yields, par_swap_rate, CapletQuote, DiscountCurve, HullWhite,
    IrmError, SwaptionQuote,
};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")
}

fn print_curve_table(name: &str, curve: &DiscountCurve) -> Result<(), IrmError> {
    println!("\n{name}  (ACT/365F, cont. comp. zeros, simple 1y-forwards)");
    println!("  {:>5} {:>10} {:>8} {:>13}", "T", "DF", "zero %", "fwd(T,T+1) %");
    for t in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0] {
        let fwd = curve.fwd_rate(t, t + 1.0)?;
        println!(
            "  {t:5.2} {:10.6} {:8.4} {:13.4}",
            curve.df(t)?,
            100.0 * curve.zero_rate(t)?,
            100.0 * fwd
        );
    }
    Ok(())
}

fn run() -> Result<(), IrmError> {
    println!("{}", "=".repeat(64));
    println!("irm demo — bootstrapping, Vasicek, Hull-White (Rust)");
    println!("{}", "=".repeat(64));

    // ---- 1. bootstrap both quote curves + OIS ------------------------ //
    let data = data_dir();
    let eur = bootstrap(&load_curve_quotes(&data.join("curve_quotes.csv"), "EUR")?)?;
    let usd = bootstrap(&load_curve_quotes(&data.join("curve_quotes.csv"), "USD")?)?;
    let ois = bootstrap(&load_ois_quotes(&data.join("ois_quotes.csv"))?)?;
    print_curve_table("EUR curve (negative short end)", &eur)?;
    print_curve_table("USD curve (normal/inverted)", &usd)?;
    println!(
        "\nOIS discount curve: DF(1y)={:.6}  DF(10y)={:.6}",
        ois.df(1.0)?,
        ois.df(10.0)?
    );
    println!(
        "Sanity: EUR DF(1y) = {:.6} > 1 (negative rates)",
        eur.df(1.0)?
    );

    // ---- 2. Vasicek calibration -------------------------------------- //
    let (ts, ys) = load_zero_yields(&data.join("zero_yields.csv"))?;
    let cal = calibrate_vasicek(&ts, &ys, 0.03, None, 4000, None)?;
    println!("\nVasicek calibration to data/zero_yields.csv (r0 = 3%), 3 starts:");
    println!(
        "  kappa={:.4}  theta={:.4}  sigma={:.4}  rmse={:.2e}  converged={}  at_bound={}",
        cal.kappa, cal.theta, cal.sigma, cal.rmse, cal.converged, cal.at_bound
    );
    println!(
        "  identified={}  spreads across equivalent starts: kappa {:.3}, theta {:.4}, sigma {:.4}",
        cal.identified, cal.kappa_spread, cal.theta_spread, cal.sigma_spread
    );
    let cal = calibrate_vasicek(&ts, &ys, 0.03, None, 4000, Some(0.02))?;
    println!("  with sigma fixed at 0.02 (from option prices):");
    println!(
        "  kappa={:.4}  theta={:.4}  rmse={:.2e}  converged={}  identified={}",
        cal.kappa, cal.theta, cal.rmse, cal.converged, cal.identified
    );
    let model = cal.model()?;
    println!("  {:>4} {:>11} {:>9}", "T", "mkt yield %", "model %");
    for (t, y) in ts.iter().zip(ys.iter()).step_by(3) {
        println!(
            "  {t:4.1} {:11.4} {:9.4}",
            100.0 * y,
            100.0 * model.zero_yield(*t, 0.0, None)?
        );
    }
    let (mc, se) = model.mc_zcb(5.0, 8, 50_000, 42)?;
    println!(
        "  ZCB(5y): analytic {:.6}  MC {mc:.6} (se {se:.1e})",
        model.zcb_price(5.0, 0.0, None)?
    );

    // ---- 3. Hull-White caps on the USD curve ------------------------- //
    let hw = HullWhite::new(0.1, 0.01, usd.clone())?;
    let atm = par_swap_rate(&usd, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])?;
    println!(
        "\nHull-White (a=0.10, sigma=0.01) on USD curve; 5y par = {:.4}%",
        100.0 * atm
    );
    println!("  {:>14} {:>9} {:>18}", "cap", "strike %", "PV (notional 100)");
    for (label, strike) in [
        ("5y ATM", atm),
        ("5y ATM+100bp", atm + 0.01),
        ("5y ATM-100bp", atm - 0.01),
    ] {
        let pv = hw.cap(&[1.0, 2.0, 3.0, 4.0, 5.0], strike, 100.0)?;
        println!("  {label:>14} {:9.4} {pv:18.6}", 100.0 * strike);
    }
    let caplet = hw.caplet(1.0, 2.0, atm, 100.0)?;
    let (mc_c, se_c) = hw.mc_caplet(1.0, 2.0, atm, 100.0, 8, 50_000, 7)?;
    println!("  caplet(1y->2y) analytic {caplet:.6} | MC {mc_c:.6} (se {se_c:.1e})");

    // ---- 4. Jamshidian swaption -------------------------------------- //
    let expiry = 1.0;
    let pay_times = [2.0, 3.0, 4.0, 5.0, 6.0];
    let mut sched = vec![expiry];
    sched.extend_from_slice(&pay_times);
    let fwd_par = par_swap_rate(&usd, &sched)?;
    let res = hw.jamshidian_swaption(expiry, &pay_times, fwd_par, 100.0, true)?;
    println!(
        "\nJamshidian 1y-into-5y payer swaption (USD, strike = fwd par {:.4}%):",
        100.0 * fwd_par
    );
    println!(
        "  value = {:.6}   r* = {:.6}   |residual| = {:.1e}",
        res.value,
        res.r_star,
        res.residual.abs()
    );
    let eur_hw = HullWhite::new(0.05, 0.008, eur.clone())?;
    let eur_par = par_swap_rate(&eur, &[1.0, 2.0, 3.0, 4.0])?;
    let eur_res = eur_hw.jamshidian_swaption(1.0, &[2.0, 3.0, 4.0], eur_par, 100.0, true)?;
    println!(
        "  EUR (negative rates) 1y-into-3y @ {:.4}%: value = {:.6}, r* = {:.6}",
        100.0 * eur_par,
        eur_res.value,
        eur_res.r_star
    );

    // ---- 5. Hull-White calibration + Bachelier vols ------------------ //
    println!("\nHull-White calibration to USD caplet/swaption prices (a, sigma free):");
    let truth = HullWhite::new(0.08, 0.012, usd.clone())?;
    let mut caplets = Vec::new();
    for r in [1.0, 2.0, 4.0] {
        caplets.push(CapletQuote::new(r, r + 1.0, atm, truth.caplet(r, r + 1.0, atm, 1.0)?)?);
    }
    let swaptions = [SwaptionQuote::new(
        expiry,
        &pay_times,
        fwd_par,
        truth
            .jamshidian_swaption(expiry, &pay_times, fwd_par, 1.0, true)?
            .value,
    )?];
    let hcal = calibrate_hullwhite(&usd, &caplets, &swaptions, None, None, 4000)?;
    println!(
        "  a={:.6}  sigma={:.6}  rmse={:.1e}  converged={}  identified={}  (true: 0.08, 0.012)",
        hcal.a, hcal.sigma, hcal.rmse, hcal.converged, hcal.identified
    );
    println!("  normal (bp) implied vols of the calibrated model:");
    for q in &caplets {
        let fwd = usd.fwd_rate(q.reset, q.pay)?;
        let ann = (q.pay - q.reset) * usd.df(q.pay)?;
        let vol = bachelier_implied_vol(q.price, fwd, q.strike, q.reset, ann, true)?;
        println!(
            "    caplet {:.0}y->{:.0}y  {:6.1} bp",
            q.reset,
            q.pay,
            1e4 * vol
        );
    }
    let mut ann = 0.0;
    let mut prev = expiry;
    for &t in &pay_times {
        ann += (t - prev) * usd.df(t)?;
        prev = t;
    }
    let svol = bachelier_implied_vol(swaptions[0].price, fwd_par, fwd_par, expiry, ann, true)?;
    println!("    1y-into-5y swaption   {:6.1} bp", 1e4 * svol);

    println!("\ndone.");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("demo failed: {e}");
            ExitCode::FAILURE
        }
    }
}
