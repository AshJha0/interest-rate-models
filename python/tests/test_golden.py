"""Golden-value suite: every case in data/golden/golden.json is recomputed
through the public API and must agree within its stored tolerance.

The dispatcher below is the executable definition of what each case name
means; ports implement the same mapping (see API_SPEC.md section 8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

from irm import (
    CapletQuote,
    DiscountCurve,
    HullWhite,
    SwaptionQuote,
    Vasicek,
    bachelier_implied_vol,
    bachelier_price,
    calibrate_hullwhite,
    calibrate_vasicek,
    load_zero_yields,
    par_swap_rate,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

with open(DATA_DIR / "golden" / "golden.json") as _fh:
    CASES = json.load(_fh)["cases"]


def _curve_from_inputs(inputs: Dict[str, float]) -> DiscountCurve:
    """Rebuild a curve from flat keys t1..tn / df1..dfn."""
    ts, dfs = [], []
    i = 1
    while f"df{i}" in inputs:
        ts.append(float(inputs[f"t{i}"]))
        dfs.append(float(inputs[f"df{i}"]))
        i += 1
    return DiscountCurve(ts, dfs)


def _vasicek_from_inputs(inputs: Dict[str, float]) -> Vasicek:
    return Vasicek(inputs["kappa"], inputs["theta"], inputs["sigma"], inputs["r0"])


def _hw_from_inputs(inputs: Dict[str, float]) -> HullWhite:
    return HullWhite(inputs["a"], inputs["sigma"], _curve_from_inputs(inputs))


def _boot_curve(curve_id: str):
    from irm import bootstrap, load_curve_quotes, load_ois_quotes

    if curve_id == "OIS":
        return bootstrap(load_ois_quotes(DATA_DIR / "ois_quotes.csv"))
    return bootstrap(load_curve_quotes(DATA_DIR / "curve_quotes.csv", curve_id))


_BOOT_CACHE: Dict[str, DiscountCurve] = {}


def _boot(curve_id: str) -> DiscountCurve:
    if curve_id not in _BOOT_CACHE:
        _BOOT_CACHE[curve_id] = _boot_curve(curve_id)
    return _BOOT_CACHE[curve_id]


def evaluate(name: str, inp: Dict[str, float]) -> Dict[str, float]:
    """Compute the expected quantities for one golden case."""
    if name.startswith("curve_"):
        c = _curve_from_inputs(inp)
        if name == "curve_df_pillar" or name == "curve_df_mid" or name == "curve_df_extrap":
            return {"df": c.df(inp["t"])}
        if name == "curve_zero_mid":
            return {"zero": c.zero_rate(inp["t"])}
        if name == "curve_fwd_simple":
            return {"fwd": c.fwd_rate(inp["ta"], inp["tb"])}
    if name.startswith("boot_"):
        c = _boot(str(inp["curve_id"]))
        if name.endswith("_par_10y"):
            times = [0.0] + [float(i) for i in range(1, 11)]
            return {"par": par_swap_rate(c, times)}
        if "zero" in name:
            return {"zero": c.zero_rate(inp["t"])}
        return {"df": c.df(inp["t"])}
    if name == "vas_calib_sigma_fixed":
        ts, ys = load_zero_yields(DATA_DIR / "zero_yields.csv")
        cal = calibrate_vasicek(ts, ys, inp["r0"], sigma_fixed=inp["sigma_fixed"])
        assert cal.converged and cal.identified and not cal.at_bound
        return {"kappa": cal.kappa, "theta": cal.theta}
    if name.startswith("vas_"):
        m = _vasicek_from_inputs(inp)
        if name == "vas_rT_mean":
            return {"mean": m.r_mean(inp["T"])}
        if name == "vas_rT_var":
            return {"var": m.r_var(inp["T"])}
        if name == "vas_zbc":
            return {"price": m.zbc(inp["expiry"], inp["bond_maturity"], inp["strike"])}
        if name == "vas_zbp":
            return {"price": m.zbp(inp["expiry"], inp["bond_maturity"], inp["strike"])}
        if name == "vas_mc_zcb":
            price, _ = m.mc_zcb(inp["T"], n_steps=8, n_paths=int(inp["min_paths"]), seed=901)
            return {"price": price}
        return {"price": m.zcb_price(inp["T"])}
    if name.startswith("bach_"):
        if name == "bach_implied_vol_payer":
            return {"vol": bachelier_implied_vol(
                inp["price"], inp["forward"], inp["strike"], inp["expiry"], inp["annuity"], True)}
        return {"price": bachelier_price(
            inp["forward"], inp["strike"], inp["expiry"], inp["vol"], inp["annuity"], True)}
    if name == "hw_calib_recover":
        curve = _curve_from_inputs(inp)
        caps = []
        i = 1
        while f"cap_price_{i}" in inp:
            caps.append(CapletQuote(inp[f"cap_reset_{i}"], inp[f"cap_pay_{i}"], inp["strike"],
                                    inp[f"cap_price_{i}"]))
            i += 1
        e = inp["swpt_expiry"]
        pays = tuple(e + k for k in range(1, int(round(inp["swpt_tenor_years"])) + 1))
        swps = [SwaptionQuote(e, pays, inp["swpt_fixed_rate"], inp["swpt_price"])]
        cal = calibrate_hullwhite(curve, caps, swps)
        assert cal.converged and cal.identified and not cal.at_bound
        return {"a": cal.a, "sigma": cal.sigma}
    if name.startswith("hw_"):
        hw = _hw_from_inputs(inp)
        if name == "hw_zcb_t0":
            return {"price": hw.zcb_price(0.0, inp["T"])}
        if name == "hw_zcb_t1_explicit_r":
            return {"price": hw.zcb_price(inp["t"], inp["T"], inp["r"])}
        if name in ("hw_caplet", "hw_caplet_sigma0"):
            return {"price": hw.caplet(inp["reset"], inp["pay"], inp["strike"], inp["notional"])}
        if name == "hw_floorlet":
            return {"price": hw.floorlet(inp["reset"], inp["pay"], inp["strike"], inp["notional"])}
        if name == "hw_cap_3y":
            sched = [inp["s0"], inp["s1"], inp["s2"], inp["s3"]]
            return {"price": hw.cap(sched, inp["strike"], inp["notional"])}
        if name in ("hw_swaption_payer", "hw_swaption_neg_curve", "hw_swaption_receiver"):
            expiry = inp["expiry"]
            n = int(round(inp["tenor_years"]))
            pay_times = [expiry + i for i in range(1, n + 1)]
            res = hw.jamshidian_swaption(expiry, pay_times, inp["fixed_rate"], inp["notional"],
                                         payer=(name != "hw_swaption_receiver"))
            out = {"price": res.value, "r_star": res.r_star}
            for i, k in enumerate(res.strikes, start=1):
                out[f"strike_{i}"] = k
            return out
        if name == "hw_mc_caplet":
            price, _ = hw.mc_caplet(
                inp["reset"], inp["pay"], inp["strike"], inp["notional"],
                n_steps=8, n_paths=int(inp["min_paths"]), seed=902,
            )
            return {"price": price}
    raise AssertionError(f"golden dispatcher has no rule for case {name!r}")


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_golden_case(case: dict) -> None:
    got = evaluate(case["name"], case["inputs"])
    for key, expected in case["expect"].items():
        assert key in got, f"{case['name']}: missing output {key}"
        assert got[key] == pytest.approx(expected, abs=case["tol"]), (
            f"{case['name']}.{key}: got {got[key]}, want {expected} +- {case['tol']}"
        )


def test_golden_has_expected_case_count() -> None:
    assert len(CASES) == 33
    names = [c["name"] for c in CASES]
    assert len(names) == len(set(names)), "duplicate golden case names"
