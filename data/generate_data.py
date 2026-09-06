"""Regenerate the bundled sample data and cross-language golden values.

Deterministic: quotes are hard-coded; the only randomness (calibration
target noise) uses a fixed seed.  Before writing golden.json this script
VALIDATES the numbers:

* every bootstrapped curve reprices all of its input instruments to 1e-9;
* Vasicek closed forms are cross-checked against exact-transition Monte
  Carlo within 3 standard errors (ZCB and bond option via parity);
* the Hull-White caplet closed form is cross-checked against MC within 3 SE;
* HW P(0,T) is checked to match the market DF to 1e-12;
* the Bachelier implied vol round-trips its price to 1e-12;
* the Hull-White calibration recovers the generating (a, sigma) to 1e-9.

Run:  python3 generate_data.py   (from this directory or anywhere)
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python" / "src"))

from irm import (  # noqa: E402
    CapletQuote,
    DiscountCurve,
    HullWhite,
    SwaptionQuote,
    Vasicek,
    bachelier_implied_vol,
    bachelier_price,
    bootstrap,
    calibrate_hullwhite,
    calibrate_vasicek,
    load_curve_quotes,
    load_ois_quotes,
    load_zero_yields,
    par_swap_rate,
)

# --------------------------------------------------------------------- #
# Market quotes (rates decimal, times ACT/365F years)
# --------------------------------------------------------------------- #

ON = 1.0 / 365.0

# EUR-like: negative short end, upward sloping long end.
EUR_DEPOSITS = [(ON, -0.0050), (1.0 / 12.0, -0.0048), (0.25, -0.0045),
                (0.5, -0.0040), (1.0, -0.0030)]
EUR_SWAPS = [(2.0, -0.0020), (3.0, -0.0010), (4.0, 0.0000), (5.0, 0.0010),
             (7.0, 0.0040), (10.0, 0.0080), (15.0, 0.0120), (20.0, 0.0140),
             (30.0, 0.0150)]

# USD-like: normal (high, mildly inverted) curve.
USD_DEPOSITS = [(ON, 0.0530), (1.0 / 12.0, 0.0531), (0.25, 0.0533),
                (0.5, 0.0528), (1.0, 0.0510)]
USD_SWAPS = [(2.0, 0.0480), (3.0, 0.0455), (4.0, 0.0442), (5.0, 0.0435),
             (7.0, 0.0428), (10.0, 0.0425), (15.0, 0.0427), (20.0, 0.0428),
             (30.0, 0.0420)]

# OIS par swap quotes (USD-like, ~ swap curve minus a spread).
OIS_QUOTES = [(1.0 / 12.0, 0.0528), (0.25, 0.0530), (0.5, 0.0524),
              (1.0, 0.0505), (2.0, 0.0473), (3.0, 0.0448), (5.0, 0.0428),
              (7.0, 0.0421), (10.0, 0.0418), (15.0, 0.0420), (20.0, 0.0421),
              (30.0, 0.0413)]

# "True" Vasicek used to manufacture calibration targets.
CAL_TRUE = dict(kappa=0.8, theta=0.05, sigma=0.02, r0=0.03)
CAL_MATURITIES = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]
CAL_NOISE_SD = 2e-5
CAL_SEED = 42

# Self-contained golden fixtures ------------------------------------- #

# Small explicit curve for interpolation cases.
G_CURVE_T = [1.0, 2.0, 3.0, 5.0]
G_CURVE_DF = [0.99, 0.975, 0.955, 0.90]

# Hull-White golden curves: 5 pillars, DF = exp(-z(t) * t).
HW_T = [1.0, 2.0, 3.0, 5.0, 10.0]


def _hw_pos_z(t: float) -> float:
    return 0.02 + 0.0015 * t


def _hw_neg_z(t: float) -> float:
    return -0.004 + 0.0012 * t


HW_POS_DF = [math.exp(-_hw_pos_z(t) * t) for t in HW_T]
HW_NEG_DF = [math.exp(-_hw_neg_z(t) * t) for t in HW_T]

VAS_BASE = dict(kappa=0.5, theta=0.03, sigma=0.01, r0=0.02)
VAS_NEG = dict(kappa=0.3, theta=0.01, sigma=0.008, r0=-0.005)

HW_A, HW_SIGMA = 0.1, 0.01

# Hull-White calibration golden: quotes generated from these "true" values
# (deliberately not one of the default optimiser starts).
HW_CAL_TRUE_A, HW_CAL_TRUE_SIGMA = 0.08, 0.012
HW_CAL_CAPLET_RESETS = [1.0, 2.0, 4.0]
HW_CAL_CAPLET_STRIKE = 0.025
HW_CAL_SWPT = dict(expiry=1.0, tenor_years=5, fixed_rate=0.028)

# Vasicek sigma-fixed calibration golden (sigma from "option prices").
VAS_CAL_SIGMA_FIXED = 0.02

# Bachelier goldens.
BACH_PAYER = dict(forward=0.03, strike=0.025, expiry=2.0, vol=0.0075, annuity=4.2)
BACH_ATM_NEG = dict(forward=-0.002, strike=-0.002, expiry=1.0, vol=0.005, annuity=0.98)

MC_PATHS_REF = 200_000
MC_PATHS_PORT = 50_000  # minimum paths a port must use for MC golden cases
MC_STEPS = 8


# --------------------------------------------------------------------- #
# CSV writers
# --------------------------------------------------------------------- #

def write_csvs() -> None:
    with open(HERE / "curve_quotes.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["curve_id", "instrument", "start", "maturity", "rate"])
        for cid, deps, swaps in (("EUR", EUR_DEPOSITS, EUR_SWAPS),
                                 ("USD", USD_DEPOSITS, USD_SWAPS)):
            for t, r in deps:
                w.writerow([cid, "deposit", 0.0, repr(t), repr(r)])
            for t, r in swaps:
                w.writerow([cid, "swap", 0.0, repr(t), repr(r)])

    with open(HERE / "ois_quotes.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["maturity", "rate"])
        for t, r in OIS_QUOTES:
            w.writerow([repr(t), repr(r)])

    true_model = Vasicek(**CAL_TRUE)
    rng = np.random.default_rng(CAL_SEED)
    noise = rng.normal(0.0, CAL_NOISE_SD, size=len(CAL_MATURITIES))
    with open(HERE / "zero_yields.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["maturity", "yield"])
        for t, eps in zip(CAL_MATURITIES, noise):
            w.writerow([repr(t), repr(true_model.zero_yield(t) + float(eps))])


# --------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------- #

def validate_repricing(curve: DiscountCurve, instruments, label: str) -> None:
    worst = max(abs(ins.residual(curve)) for ins in instruments)
    assert worst < 1e-9, f"{label}: repricing residual {worst} >= 1e-9"
    print(f"  [ok] {label}: max repricing residual {worst:.3e}")


def validate_vasicek_mc() -> None:
    m = Vasicek(**VAS_BASE)
    analytic = m.zcb_price(5.0)
    price, se = m.mc_zcb(5.0, MC_STEPS, MC_PATHS_REF, seed=7)
    assert abs(price - analytic) < 3.0 * se, (
        f"Vasicek MC ZCB {price} vs analytic {analytic}, se={se}"
    )
    print(f"  [ok] Vasicek MC ZCB within 3 SE (diff={abs(price-analytic):.2e}, se={se:.2e})")
    # Bond-option parity sanity: ZBC - ZBP = P(0,S) - K P(0,T).
    k = 0.95
    lhs = m.zbc(1.0, 3.0, k) - m.zbp(1.0, 3.0, k)
    rhs = m.zcb_price(3.0) - k * m.zcb_price(1.0)
    assert abs(lhs - rhs) < 1e-12, "Vasicek ZBC/ZBP parity broken"
    print("  [ok] Vasicek bond-option put-call parity")


def validate_hw(curve: DiscountCurve) -> None:
    hw = HullWhite(HW_A, HW_SIGMA, curve)
    for t in [1.0, 2.0, 5.0, 10.0]:
        assert abs(hw.zcb_price(0.0, t) - curve.df(t)) < 1e-12, "HW P(0,T) != DF"
    analytic = hw.caplet(1.0, 2.0, 0.025, 100.0)
    price, se = hw.mc_caplet(1.0, 2.0, 0.025, 100.0, MC_STEPS, MC_PATHS_REF, seed=11)
    assert abs(price - analytic) < 3.0 * se, (
        f"HW MC caplet {price} vs analytic {analytic}, se={se}"
    )
    print(f"  [ok] HW MC caplet within 3 SE (diff={abs(price-analytic):.2e}, se={se:.2e})")


def validate_bachelier() -> None:
    for q in (BACH_PAYER, BACH_ATM_NEG):
        p = bachelier_price(q["forward"], q["strike"], q["expiry"], q["vol"], q["annuity"], True)
        v = bachelier_implied_vol(p, q["forward"], q["strike"], q["expiry"], q["annuity"], True)
        assert abs(v - q["vol"]) < 1e-12, f"Bachelier round trip {v} vs {q['vol']}"
    # ATM closed form: price = A sigma sqrt(T) / sqrt(2 pi).
    q = BACH_ATM_NEG
    p = bachelier_price(q["forward"], q["strike"], q["expiry"], q["vol"], q["annuity"], True)
    assert abs(p - q["annuity"] * q["vol"] * math.sqrt(q["expiry"]) / math.sqrt(2 * math.pi)) < 1e-16
    print("  [ok] Bachelier price/implied-vol round trip")


def hw_calibration_quotes(curve: DiscountCurve):
    true = HullWhite(HW_CAL_TRUE_A, HW_CAL_TRUE_SIGMA, curve)
    caps = [
        CapletQuote(r, r + 1.0, HW_CAL_CAPLET_STRIKE,
                    true.caplet(r, r + 1.0, HW_CAL_CAPLET_STRIKE, 1.0))
        for r in HW_CAL_CAPLET_RESETS
    ]
    e = HW_CAL_SWPT["expiry"]
    pays = tuple(e + i for i in range(1, HW_CAL_SWPT["tenor_years"] + 1))
    swp = SwaptionQuote(e, pays, HW_CAL_SWPT["fixed_rate"],
                        true.jamshidian_swaption(e, pays, HW_CAL_SWPT["fixed_rate"], 1.0).value)
    return caps, [swp]


def validate_hw_calibration(curve: DiscountCurve) -> None:
    caps, swps = hw_calibration_quotes(curve)
    cal = calibrate_hullwhite(curve, caps, swps)
    assert cal.converged and cal.identified and not cal.at_bound, cal
    assert abs(cal.a - HW_CAL_TRUE_A) < 1e-9 and abs(cal.sigma - HW_CAL_TRUE_SIGMA) < 1e-9, cal
    print(f"  [ok] HW calibration recovers (a, sigma) to "
          f"({abs(cal.a - HW_CAL_TRUE_A):.1e}, {abs(cal.sigma - HW_CAL_TRUE_SIGMA):.1e})")


# --------------------------------------------------------------------- #
# Golden cases
# --------------------------------------------------------------------- #

def flat_curve_inputs(prefix_ts, prefix_dfs):
    d = {}
    for i, (t, p) in enumerate(zip(prefix_ts, prefix_dfs), start=1):
        d[f"t{i}"] = t
        d[f"df{i}"] = p
    return d


def build_golden(eur: DiscountCurve, usd: DiscountCurve, ois: DiscountCurve) -> dict:
    cases = []

    def add(name, inputs, expect, tol):
        cases.append({"name": name, "inputs": inputs, "expect": expect, "tol": tol})

    # ---- curve construction (self-contained) ------------------------- #
    g = DiscountCurve(G_CURVE_T, G_CURVE_DF)
    base = flat_curve_inputs(G_CURVE_T, G_CURVE_DF)
    add("curve_df_pillar", {**base, "t": 3.0}, {"df": g.df(3.0)}, 1e-14)
    add("curve_df_mid", {**base, "t": 2.5}, {"df": g.df(2.5)}, 1e-12)
    add("curve_df_extrap", {**base, "t": 7.0}, {"df": g.df(7.0)}, 1e-12)
    add("curve_zero_mid", {**base, "t": 2.5}, {"zero": g.zero_rate(2.5)}, 1e-12)
    add("curve_fwd_simple", {**base, "ta": 1.0, "tb": 3.0},
        {"fwd": g.fwd_rate(1.0, 3.0)}, 1e-12)

    # ---- bootstrap (quotes from curve_quotes.csv / ois_quotes.csv) --- #
    add("boot_eur_df_1y", {"curve_id": "EUR", "t": 1.0}, {"df": eur.df(1.0)}, 1e-8)
    add("boot_eur_df_10y", {"curve_id": "EUR", "t": 10.0}, {"df": eur.df(10.0)}, 1e-8)
    add("boot_eur_zero_30y", {"curve_id": "EUR", "t": 30.0},
        {"zero": eur.zero_rate(30.0)}, 1e-8)
    add("boot_usd_df_10y", {"curve_id": "USD", "t": 10.0}, {"df": usd.df(10.0)}, 1e-8)
    add("boot_usd_par_10y", {"curve_id": "USD", "t": 10.0},
        {"par": par_swap_rate(usd, [0.0] + [float(i) for i in range(1, 11)])}, 1e-10)
    add("boot_ois_df_10y", {"curve_id": "OIS", "t": 10.0}, {"df": ois.df(10.0)}, 1e-8)

    # ---- Vasicek closed forms ---------------------------------------- #
    vb = Vasicek(**VAS_BASE)
    vn = Vasicek(**VAS_NEG)
    add("vas_zcb_5y", {**VAS_BASE, "T": 5.0}, {"price": vb.zcb_price(5.0)}, 1e-10)
    add("vas_zcb_neg_r0", {**VAS_NEG, "T": 5.0}, {"price": vn.zcb_price(5.0)}, 1e-10)
    add("vas_rT_mean", {**VAS_BASE, "T": 2.0}, {"mean": vb.r_mean(2.0)}, 1e-12)
    add("vas_rT_var", {**VAS_BASE, "T": 2.0}, {"var": vb.r_var(2.0)}, 1e-14)
    add("vas_zbc", {**VAS_BASE, "expiry": 1.0, "bond_maturity": 3.0, "strike": 0.95},
        {"price": vb.zbc(1.0, 3.0, 0.95)}, 1e-10)
    add("vas_zbp", {**VAS_BASE, "expiry": 1.0, "bond_maturity": 3.0, "strike": 0.95},
        {"price": vb.zbp(1.0, 3.0, 0.95)}, 1e-10)
    # MC with a statistical band: expect = analytic, tol = 6 SE at the
    # minimum required path count (ports: >= MC_PATHS_PORT paths, any seed).
    _, se_port = vb.mc_zcb(5.0, MC_STEPS, MC_PATHS_PORT, seed=101)
    add("vas_mc_zcb", {**VAS_BASE, "T": 5.0, "min_paths": MC_PATHS_PORT},
        {"price": vb.zcb_price(5.0)}, 6.0 * se_port)

    # ---- Hull-White (self-contained 5-pillar curves) ----------------- #
    cp = DiscountCurve(HW_T, HW_POS_DF)
    cn = DiscountCurve(HW_T, HW_NEG_DF)
    hp = HullWhite(HW_A, HW_SIGMA, cp)
    hn = HullWhite(HW_A, HW_SIGMA, cn)
    hp0 = HullWhite(HW_A, 0.0, cp)
    pos = flat_curve_inputs(HW_T, HW_POS_DF)
    neg = flat_curve_inputs(HW_T, HW_NEG_DF)
    hw_par = {"a": HW_A, "sigma": HW_SIGMA}

    add("hw_zcb_t0", {**pos, **hw_par, "T": 5.0}, {"price": hp.zcb_price(0.0, 5.0)}, 1e-12)
    # P(t > 0, T; r) with an explicit r depends on the contractual FD stencil
    # for f^M(0,t): pinning it enforces the stencil across ports.
    add("hw_zcb_t1_explicit_r", {**pos, **hw_par, "t": 1.0, "T": 5.0, "r": 0.03},
        {"price": hp.zcb_price(1.0, 5.0, 0.03)}, 1e-10)
    add("hw_caplet", {**pos, **hw_par, "reset": 1.0, "pay": 2.0, "strike": 0.025,
                      "notional": 100.0},
        {"price": hp.caplet(1.0, 2.0, 0.025, 100.0)}, 1e-9)
    add("hw_floorlet", {**pos, **hw_par, "reset": 1.0, "pay": 2.0, "strike": 0.025,
                        "notional": 100.0},
        {"price": hp.floorlet(1.0, 2.0, 0.025, 100.0)}, 1e-9)
    # Schedule keys are s0..s3 (NOT t1..t3, which are the curve pillars).
    add("hw_cap_3y", {**pos, **hw_par, "s0": 0.5, "s1": 1.0, "s2": 2.0, "s3": 3.0,
                      "strike": 0.024, "notional": 100.0},
        {"price": hp.cap([0.5, 1.0, 2.0, 3.0], 0.024, 100.0)}, 1e-9)
    add("hw_caplet_sigma0", {**pos, "a": HW_A, "sigma": 0.0, "reset": 1.0, "pay": 2.0,
                             "strike": 0.02, "notional": 100.0},
        {"price": hp0.caplet(1.0, 2.0, 0.02, 100.0)}, 1e-12)
    # r* and the decomposition strikes depend on the FD stencil (the price
    # does not), so they are pinned too.
    sw = hp.jamshidian_swaption(1.0, [2.0, 3.0, 4.0, 5.0, 6.0], 0.028, 100.0)
    add("hw_swaption_payer", {**pos, **hw_par, "expiry": 1.0, "tenor_years": 5.0,
                              "fixed_rate": 0.028, "notional": 100.0},
        {"price": sw.value, "r_star": sw.r_star,
         **{f"strike_{i}": k for i, k in enumerate(sw.strikes, start=1)}}, 1e-9)
    swn = hn.jamshidian_swaption(1.0, [2.0, 3.0, 4.0], 0.0, 100.0)
    add("hw_swaption_neg_curve", {**neg, **hw_par, "expiry": 1.0, "tenor_years": 3.0,
                                  "fixed_rate": 0.0, "notional": 100.0},
        {"price": swn.value, "r_star": swn.r_star}, 1e-9)
    # Receiver on the same swap: pins the payer/receiver branch.
    swr = hp.jamshidian_swaption(1.0, [2.0, 3.0, 4.0, 5.0, 6.0], 0.028, 100.0, payer=False)
    add("hw_swaption_receiver", {**pos, **hw_par, "expiry": 1.0, "tenor_years": 5.0,
                                 "fixed_rate": 0.028, "notional": 100.0},
        {"price": swr.value}, 1e-9)
    _, se_cap = hp.mc_caplet(1.0, 2.0, 0.025, 100.0, MC_STEPS, MC_PATHS_PORT, seed=202)
    add("hw_mc_caplet", {**pos, **hw_par, "reset": 1.0, "pay": 2.0, "strike": 0.025,
                         "notional": 100.0, "min_paths": MC_PATHS_PORT},
        {"price": hp.caplet(1.0, 2.0, 0.025, 100.0)}, 6.0 * se_cap)

    # ---- Bachelier normal model ------------------------------------- #
    q = BACH_PAYER
    bp = bachelier_price(q["forward"], q["strike"], q["expiry"], q["vol"], q["annuity"], True)
    add("bach_price_payer", {**q}, {"price": bp}, 1e-14)
    add("bach_implied_vol_payer", {"forward": q["forward"], "strike": q["strike"],
                                   "expiry": q["expiry"], "annuity": q["annuity"], "price": bp},
        {"vol": bachelier_implied_vol(bp, q["forward"], q["strike"], q["expiry"],
                                      q["annuity"], True)}, 1e-12)
    q = BACH_ATM_NEG
    bp = bachelier_price(q["forward"], q["strike"], q["expiry"], q["vol"], q["annuity"], True)
    add("bach_price_atm_negative", {**q}, {"price": bp}, 1e-14)

    # ---- calibrations ------------------------------------------------ #
    caps, swps = hw_calibration_quotes(cp)
    hw_cal_inputs = {**pos, "strike": HW_CAL_CAPLET_STRIKE,
                     "swpt_expiry": HW_CAL_SWPT["expiry"],
                     "swpt_tenor_years": float(HW_CAL_SWPT["tenor_years"]),
                     "swpt_fixed_rate": HW_CAL_SWPT["fixed_rate"],
                     "swpt_price": swps[0].price}
    for i, c in enumerate(caps, start=1):
        hw_cal_inputs[f"cap_reset_{i}"] = c.reset
        hw_cal_inputs[f"cap_pay_{i}"] = c.pay
        hw_cal_inputs[f"cap_price_{i}"] = c.price
    hcal = calibrate_hullwhite(cp, caps, swps)
    add("hw_calib_recover", hw_cal_inputs, {"a": hcal.a, "sigma": hcal.sigma}, 1e-8)

    ts, ys = load_zero_yields(HERE / "zero_yields.csv")
    vcal = calibrate_vasicek(ts, ys, CAL_TRUE["r0"], sigma_fixed=VAS_CAL_SIGMA_FIXED)
    assert vcal.converged and vcal.identified, vcal
    add("vas_calib_sigma_fixed", {"r0": CAL_TRUE["r0"], "sigma_fixed": VAS_CAL_SIGMA_FIXED},
        {"kappa": vcal.kappa, "theta": vcal.theta}, 1e-6)

    return {"cases": cases}


# --------------------------------------------------------------------- #

def main() -> None:
    print("writing CSVs ...")
    write_csvs()

    print("bootstrapping + validating ...")
    eur = bootstrap(load_curve_quotes(HERE / "curve_quotes.csv", "EUR"))
    usd = bootstrap(load_curve_quotes(HERE / "curve_quotes.csv", "USD"))
    ois = bootstrap(load_ois_quotes(HERE / "ois_quotes.csv"))
    validate_repricing(eur, load_curve_quotes(HERE / "curve_quotes.csv", "EUR"), "EUR")
    validate_repricing(usd, load_curve_quotes(HERE / "curve_quotes.csv", "USD"), "USD")
    validate_repricing(ois, load_ois_quotes(HERE / "ois_quotes.csv"), "OIS")
    assert eur.df(1.0) > 1.0, "EUR negative short end must give DF > 1"
    validate_vasicek_mc()
    validate_hw(DiscountCurve(HW_T, HW_POS_DF))
    validate_bachelier()
    validate_hw_calibration(DiscountCurve(HW_T, HW_POS_DF))

    golden = build_golden(eur, usd, ois)
    out = HERE / "golden" / "golden.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as fh:
        json.dump(golden, fh, indent=1)
        fh.write("\n")
    print(f"wrote {out} with {len(golden['cases'])} cases")


if __name__ == "__main__":
    main()
