//! Golden-value suite: every case in `../data/golden/golden.json` is
//! recomputed through the public API and must agree within its stored
//! tolerance.  The dispatcher mirrors the Python reference test
//! (`python/tests/test_golden.py`), the executable definition of each case.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use irm::{
    bachelier_implied_vol, bachelier_price, bootstrap, calibrate_hullwhite, calibrate_vasicek,
    load_curve_quotes, load_ois_quotes, load_zero_yields, par_swap_rate, CapletQuote,
    DiscountCurve, HullWhite, SwaptionQuote, Vasicek,
};
use serde_json::Value;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")
}

fn num(inputs: &serde_json::Map<String, Value>, key: &str) -> f64 {
    inputs
        .get(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("missing/non-numeric input {key:?}"))
}

/// Rebuild a curve from flat keys `t1..tn` / `df1..dfn`.
fn curve_from_inputs(inputs: &serde_json::Map<String, Value>) -> DiscountCurve {
    let mut ts = Vec::new();
    let mut dfs = Vec::new();
    let mut i = 1;
    while inputs.contains_key(&format!("df{i}")) {
        ts.push(num(inputs, &format!("t{i}")));
        dfs.push(num(inputs, &format!("df{i}")));
        i += 1;
    }
    DiscountCurve::new(&ts, &dfs).expect("golden curve inputs are valid")
}

fn vasicek_from_inputs(inputs: &serde_json::Map<String, Value>) -> Vasicek {
    Vasicek::new(
        num(inputs, "kappa"),
        num(inputs, "theta"),
        num(inputs, "sigma"),
        num(inputs, "r0"),
    )
    .expect("golden Vasicek inputs are valid")
}

fn hw_from_inputs(inputs: &serde_json::Map<String, Value>) -> HullWhite {
    HullWhite::new(num(inputs, "a"), num(inputs, "sigma"), curve_from_inputs(inputs))
        .expect("golden HW inputs are valid")
}

fn boot(cache: &mut HashMap<String, DiscountCurve>, curve_id: &str) -> DiscountCurve {
    cache
        .entry(curve_id.to_string())
        .or_insert_with(|| {
            let data = data_dir();
            let instruments = if curve_id == "OIS" {
                load_ois_quotes(&data.join("ois_quotes.csv")).expect("ois quotes load")
            } else {
                load_curve_quotes(&data.join("curve_quotes.csv"), curve_id)
                    .expect("curve quotes load")
            };
            bootstrap(&instruments).expect("bootstrap succeeds")
        })
        .clone()
}

/// Compute the expected quantities for one golden case.
fn evaluate(
    name: &str,
    inp: &serde_json::Map<String, Value>,
    cache: &mut HashMap<String, DiscountCurve>,
) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    if let Some(rest) = name.strip_prefix("curve_") {
        let c = curve_from_inputs(inp);
        match rest {
            "df_pillar" | "df_mid" | "df_extrap" => {
                out.insert("df".into(), c.df(num(inp, "t")).unwrap());
            }
            "zero_mid" => {
                out.insert("zero".into(), c.zero_rate(num(inp, "t")).unwrap());
            }
            "fwd_simple" => {
                out.insert(
                    "fwd".into(),
                    c.fwd_rate(num(inp, "ta"), num(inp, "tb")).unwrap(),
                );
            }
            other => panic!("no rule for curve case {other:?}"),
        }
        return out;
    }
    if name.starts_with("boot_") {
        let curve_id = inp
            .get("curve_id")
            .and_then(Value::as_str)
            .expect("curve_id present");
        let c = boot(cache, curve_id);
        if name.ends_with("_par_10y") {
            let times: Vec<f64> = (0..=10).map(f64::from).collect();
            out.insert("par".into(), par_swap_rate(&c, &times).unwrap());
        } else if name.contains("zero") {
            out.insert("zero".into(), c.zero_rate(num(inp, "t")).unwrap());
        } else {
            out.insert("df".into(), c.df(num(inp, "t")).unwrap());
        }
        return out;
    }
    if name == "vas_calib_sigma_fixed" {
        let (ts, ys) = load_zero_yields(&data_dir().join("zero_yields.csv")).unwrap();
        let cal = calibrate_vasicek(&ts, &ys, num(inp, "r0"), None, 4000, Some(num(inp, "sigma_fixed")))
            .unwrap();
        assert!(cal.converged && cal.identified && !cal.at_bound);
        out.insert("kappa".into(), cal.kappa);
        out.insert("theta".into(), cal.theta);
        return out;
    }
    if name.starts_with("bach_") {
        if name == "bach_implied_vol_payer" {
            out.insert(
                "vol".into(),
                bachelier_implied_vol(
                    num(inp, "price"),
                    num(inp, "forward"),
                    num(inp, "strike"),
                    num(inp, "expiry"),
                    num(inp, "annuity"),
                    true,
                )
                .unwrap(),
            );
        } else {
            out.insert(
                "price".into(),
                bachelier_price(
                    num(inp, "forward"),
                    num(inp, "strike"),
                    num(inp, "expiry"),
                    num(inp, "vol"),
                    num(inp, "annuity"),
                    true,
                )
                .unwrap(),
            );
        }
        return out;
    }
    if name == "hw_calib_recover" {
        let curve = curve_from_inputs(inp);
        let mut caps = Vec::new();
        let mut i = 1;
        while inp.contains_key(&format!("cap_price_{i}")) {
            caps.push(
                CapletQuote::new(
                    num(inp, &format!("cap_reset_{i}")),
                    num(inp, &format!("cap_pay_{i}")),
                    num(inp, "strike"),
                    num(inp, &format!("cap_price_{i}")),
                )
                .unwrap(),
            );
            i += 1;
        }
        let e = num(inp, "swpt_expiry");
        let n = num(inp, "swpt_tenor_years").round() as usize;
        let pays: Vec<f64> = (1..=n).map(|k| e + k as f64).collect();
        let swps = [SwaptionQuote::new(e, &pays, num(inp, "swpt_fixed_rate"), num(inp, "swpt_price"))
            .unwrap()];
        let cal = calibrate_hullwhite(&curve, &caps, &swps, None, None, 4000).unwrap();
        assert!(cal.converged && cal.identified && !cal.at_bound);
        out.insert("a".into(), cal.a);
        out.insert("sigma".into(), cal.sigma);
        return out;
    }
    if name.starts_with("vas_") {
        let m = vasicek_from_inputs(inp);
        match name {
            "vas_rT_mean" => {
                out.insert("mean".into(), m.r_mean(num(inp, "T"), None).unwrap());
            }
            "vas_rT_var" => {
                out.insert("var".into(), m.r_var(num(inp, "T")).unwrap());
            }
            "vas_zbc" => {
                out.insert(
                    "price".into(),
                    m.zbc(num(inp, "expiry"), num(inp, "bond_maturity"), num(inp, "strike"))
                        .unwrap(),
                );
            }
            "vas_zbp" => {
                out.insert(
                    "price".into(),
                    m.zbp(num(inp, "expiry"), num(inp, "bond_maturity"), num(inp, "strike"))
                        .unwrap(),
                );
            }
            "vas_mc_zcb" => {
                let n_paths = num(inp, "min_paths") as usize;
                let (price, _) = m.mc_zcb(num(inp, "T"), 8, n_paths, 901).unwrap();
                out.insert("price".into(), price);
            }
            _ => {
                out.insert("price".into(), m.zcb_price(num(inp, "T"), 0.0, None).unwrap());
            }
        }
        return out;
    }
    if name.starts_with("hw_") {
        let hw = hw_from_inputs(inp);
        match name {
            "hw_zcb_t0" => {
                out.insert("price".into(), hw.zcb_price(0.0, num(inp, "T"), None).unwrap());
            }
            "hw_zcb_t1_explicit_r" => {
                out.insert(
                    "price".into(),
                    hw.zcb_price(num(inp, "t"), num(inp, "T"), Some(num(inp, "r"))).unwrap(),
                );
            }
            "hw_caplet" | "hw_caplet_sigma0" => {
                out.insert(
                    "price".into(),
                    hw.caplet(
                        num(inp, "reset"),
                        num(inp, "pay"),
                        num(inp, "strike"),
                        num(inp, "notional"),
                    )
                    .unwrap(),
                );
            }
            "hw_floorlet" => {
                out.insert(
                    "price".into(),
                    hw.floorlet(
                        num(inp, "reset"),
                        num(inp, "pay"),
                        num(inp, "strike"),
                        num(inp, "notional"),
                    )
                    .unwrap(),
                );
            }
            "hw_cap_3y" => {
                // Schedule keys are s0..s3 (t1..t5 are the curve pillars).
                let sched = [num(inp, "s0"), num(inp, "s1"), num(inp, "s2"), num(inp, "s3")];
                out.insert(
                    "price".into(),
                    hw.cap(&sched, num(inp, "strike"), num(inp, "notional")).unwrap(),
                );
            }
            "hw_swaption_payer" | "hw_swaption_neg_curve" | "hw_swaption_receiver" => {
                let expiry = num(inp, "expiry");
                let n = num(inp, "tenor_years").round() as usize;
                let pay_times: Vec<f64> = (1..=n).map(|i| expiry + i as f64).collect();
                let res = hw
                    .jamshidian_swaption(
                        expiry,
                        &pay_times,
                        num(inp, "fixed_rate"),
                        num(inp, "notional"),
                        name != "hw_swaption_receiver",
                    )
                    .unwrap();
                out.insert("price".into(), res.value);
                out.insert("r_star".into(), res.r_star);
                for (i, k) in res.strikes.iter().enumerate() {
                    out.insert(format!("strike_{}", i + 1), *k);
                }
            }
            "hw_mc_caplet" => {
                let n_paths = num(inp, "min_paths") as usize;
                let (price, _) = hw
                    .mc_caplet(
                        num(inp, "reset"),
                        num(inp, "pay"),
                        num(inp, "strike"),
                        num(inp, "notional"),
                        8,
                        n_paths,
                        902,
                    )
                    .unwrap();
                out.insert("price".into(), price);
            }
            other => panic!("no rule for HW case {other:?}"),
        }
        return out;
    }
    panic!("golden dispatcher has no rule for case {name:?}");
}

#[test]
fn golden_cases() {
    let path = data_dir().join("golden/golden.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let doc: Value = serde_json::from_str(&text).expect("valid JSON");
    let cases = doc["cases"].as_array().expect("cases array");
    assert_eq!(cases.len(), 33, "expected exactly 33 golden cases");

    // Every case is evaluated and every failure collected, so a single bad
    // case cannot mask the others.
    let mut names = std::collections::HashSet::new();
    let mut cache: HashMap<String, DiscountCurve> = HashMap::new();
    let mut failures: Vec<String> = Vec::new();
    for case in cases {
        let name = case["name"].as_str().expect("case name");
        assert!(names.insert(name.to_string()), "duplicate case name {name}");
        let inputs = case["inputs"].as_object().expect("inputs object");
        let expect = case["expect"].as_object().expect("expect object");
        let tol = case["tol"].as_f64().expect("tol");
        let got = evaluate(name, inputs, &mut cache);
        for (key, want) in expect {
            let want = want.as_f64().expect("numeric expectation");
            match got.get(key) {
                None => failures.push(format!("{name}: missing output {key}")),
                Some(&got_v) => {
                    if !((got_v - want).abs() <= tol) {
                        failures.push(format!(
                            "{name}.{key}: got {got_v}, want {want} +- {tol}"
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "golden failures:\n{}", failures.join("\n"));
}
