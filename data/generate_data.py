"""Regenerate the bundled sample data and cross-language golden values.

Deterministic: quotes are hard-coded; the only randomness (calibration
target noise) uses a fixed seed.  Before writing golden.json this script
VALIDATES the numbers:

* every bootstrapped curve reprices all of its input instruments to 1e-9;
* Vasicek closed forms are cross-checked against exact-transition Monte
  Carlo within 3 standard errors (ZCB and bond option via parity);
* the Hull-White caplet closed form is cross-checked against MC within 3 SE;
* HW P(0,T) is checked to match the market DF to 1e-12.

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
    DiscountCurve,
    HullWhite,
    Vasicek,
    bootstrap,
    load_curve_quotes,
    load_ois_quotes,
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

MC_PATHS_REF = 200_000
MC_PATHS_PORT = 50_000  # minimum paths a port must use for MC golden cases
MC_STEPS = 8


# --------------------------------------------------------------------- #
# CSV writers
# --------------------------------------------------------------------- #

def write_csvs() -> None:
    with open(HERE / "curve_quotes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["curve_id", "instrument", "start", "maturity", "rate"])
        for cid, deps, swaps in (("EUR", EUR_DEPOSITS, EUR_SWAPS),
                                 ("USD", USD_DEPOSITS, USD_SWAPS)):
            for t, r in deps:
                w.writerow([cid, "deposit", 0.0, repr(t), repr(r)])
            for t, r in swaps:
                w.writerow([cid, "swap", 0.0, repr(t), repr(r)])

    with open(HERE / "ois_quotes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["maturity", "rate"])
        for t, r in OIS_QUOTES:
            w.writerow([repr(t), repr(r)])

    true_model = Vasicek(**CAL_TRUE)
    rng = np.random.default_rng(CAL_SEED)
    noise = rng.normal(0.0, CAL_NOISE_SD, size=len(CAL_MATURITIES))
    with open(HERE / "zero_yields.csv", "w", newline="") as fh:
        w = csv.writer(fh)
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
    add("hw_caplet", {**pos, **hw_par, "reset": 1.0, "pay": 2.0, "strike": 0.025,
                      "notional": 100.0},
        {"price": hp.caplet(1.0, 2.0, 0.025, 100.0)}, 1e-9)
    add("hw_floorlet", {**pos, **hw_par, "reset": 1.0, "pay": 2.0, "strike": 0.025,
                        "notional": 100.0},
        {"price": hp.floorlet(1.0, 2.0, 0.025, 100.0)}, 1e-9)
    add("hw_cap_3y", {**pos, **hw_par, "t0": 0.5, "t1": 1.0, "t2": 2.0, "t3": 3.0,
                      "strike": 0.024, "notional": 100.0},
        {"price": hp.cap([0.5, 1.0, 2.0, 3.0], 0.024, 100.0)}, 1e-9)
    add("hw_caplet_sigma0", {**pos, "a": HW_A, "sigma": 0.0, "reset": 1.0, "pay": 2.0,
                             "strike": 0.02, "notional": 100.0},
        {"price": hp0.caplet(1.0, 2.0, 0.02, 100.0)}, 1e-12)
    sw = hp.jamshidian_swaption(1.0, [2.0, 3.0, 4.0, 5.0, 6.0], 0.028, 100.0)
    add("hw_swaption_payer", {**pos, **hw_par, "expiry": 1.0, "tenor_years": 5.0,
                              "fixed_rate": 0.028, "notional": 100.0},
        {"price": sw.value}, 1e-8)
    swn = hn.jamshidian_swaption(1.0, [2.0, 3.0, 4.0], 0.0, 100.0)
    add("hw_swaption_neg_curve", {**neg, **hw_par, "expiry": 1.0, "tenor_years": 3.0,
                                  "fixed_rate": 0.0, "notional": 100.0},
        {"price": swn.value}, 1e-8)
    _, se_cap = hp.mc_caplet(1.0, 2.0, 0.025, 100.0, MC_STEPS, MC_PATHS_PORT, seed=202)
    add("hw_mc_caplet", {**pos, **hw_par, "reset": 1.0, "pay": 2.0, "strike": 0.025,
                         "notional": 100.0, "min_paths": MC_PATHS_PORT},
        {"price": hp.caplet(1.0, 2.0, 0.025, 100.0)}, 6.0 * se_cap)

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

    golden = build_golden(eur, usd, ois)
    out = HERE / "golden" / "golden.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as fh:
        json.dump(golden, fh, indent=1)
        fh.write("\n")
    print(f"wrote {out} with {len(golden['cases'])} cases")


if __name__ == "__main__":
    main()
