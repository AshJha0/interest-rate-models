"""End-to-end demo for the irm package.

Bootstraps the EUR (negative short end) and USD (normal) curves plus the
OIS curve, prints zero/forward tables, calibrates a Vasicek model to the
bundled zero yields (with and without sigma fixed), prices Hull-White caps
and a Jamshidian payer swaption, then calibrates Hull-White to
caplet/swaption prices and reports normal (bp) implied vols.
Run:  cd python && PYTHONPATH=src python3 demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from irm import (  # noqa: E402
    CapletQuote,
    DiscountCurve,
    HullWhite,
    SwaptionQuote,
    bachelier_implied_vol,
    bootstrap,
    calibrate_hullwhite,
    calibrate_vasicek,
    load_curve_quotes,
    load_ois_quotes,
    load_zero_yields,
    par_swap_rate,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def print_curve_table(name: str, curve: DiscountCurve) -> None:
    print(f"\n{name}  (ACT/365F, cont. comp. zeros, simple 1y-forwards)")
    print(f"  {'T':>5} {'DF':>10} {'zero %':>8} {'fwd(T,T+1) %':>13}")
    for t in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]:
        fwd = curve.fwd_rate(t, t + 1.0)
        print(
            f"  {t:5.2f} {curve.df(t):10.6f} {100*curve.zero_rate(t):8.4f} "
            f"{100*fwd:13.4f}"
        )


def main() -> int:
    print("=" * 64)
    print("irm demo — bootstrapping, Vasicek, Hull-White")
    print("=" * 64)

    # ---- 1. bootstrap both quote curves + OIS ------------------------ #
    eur = bootstrap(load_curve_quotes(DATA / "curve_quotes.csv", "EUR"))
    usd = bootstrap(load_curve_quotes(DATA / "curve_quotes.csv", "USD"))
    ois = bootstrap(load_ois_quotes(DATA / "ois_quotes.csv"))
    print_curve_table("EUR curve (negative short end)", eur)
    print_curve_table("USD curve (normal/inverted)", usd)
    print(f"\nOIS discount curve: DF(1y)={ois.df(1.0):.6f}  DF(10y)={ois.df(10.0):.6f}")
    print(f"Sanity: EUR DF(1y) = {eur.df(1.0):.6f} > 1 (negative rates)")

    # ---- 2. Vasicek calibration -------------------------------------- #
    ts, ys = load_zero_yields(DATA / "zero_yields.csv")
    cal = calibrate_vasicek(ts, ys, r0=0.03)
    print("\nVasicek calibration to data/zero_yields.csv (r0 = 3%), 3 starts:")
    print(
        f"  kappa={cal.kappa:.4f}  theta={cal.theta:.4f}  sigma={cal.sigma:.4f}"
        f"  rmse={cal.rmse:.2e}  converged={cal.converged}  at_bound={cal.at_bound}"
    )
    print(
        f"  identified={cal.identified}  spreads across equivalent starts: "
        f"kappa {cal.kappa_spread:.3f}, theta {cal.theta_spread:.4f}, sigma {cal.sigma_spread:.4f}"
    )
    cal = calibrate_vasicek(ts, ys, r0=0.03, sigma_fixed=0.02)
    print("  with sigma fixed at 0.02 (from option prices):")
    print(
        f"  kappa={cal.kappa:.4f}  theta={cal.theta:.4f}  rmse={cal.rmse:.2e}"
        f"  converged={cal.converged}  identified={cal.identified}"
    )
    model = cal.model()
    print(f"  {'T':>4} {'mkt yield %':>11} {'model %':>9}")
    for t, y in list(zip(ts, ys))[::3]:
        print(f"  {t:4.1f} {100*y:11.4f} {100*model.zero_yield(t):9.4f}")
    mc, se = model.mc_zcb(5.0, n_steps=8, n_paths=50_000, seed=42)
    print(
        f"  ZCB(5y): analytic {model.zcb_price(5.0):.6f}  "
        f"MC {mc:.6f} (se {se:.1e})"
    )

    # ---- 3. Hull-White caps on the USD curve ------------------------- #
    hw = HullWhite(a=0.1, sigma=0.01, curve=usd)
    atm = par_swap_rate(usd, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"\nHull-White (a=0.10, sigma=0.01) on USD curve; 5y par = {100*atm:.4f}%")
    print(f"  {'cap':>14} {'strike %':>9} {'PV (notional 100)':>18}")
    for label, strike in [("5y ATM", atm), ("5y ATM+100bp", atm + 0.01),
                          ("5y ATM-100bp", atm - 0.01)]:
        pv = hw.cap([1.0, 2.0, 3.0, 4.0, 5.0], strike, notional=100.0)
        print(f"  {label:>14} {100*strike:9.4f} {pv:18.6f}")
    caplet = hw.caplet(1.0, 2.0, atm, notional=100.0)
    mc_c, se_c = hw.mc_caplet(1.0, 2.0, atm, 100.0, n_steps=8, n_paths=50_000, seed=7)
    print(f"  caplet(1y->2y) analytic {caplet:.6f} | MC {mc_c:.6f} (se {se_c:.1e})")

    # ---- 4. Jamshidian swaption -------------------------------------- #
    expiry, pay_times = 1.0, [2.0, 3.0, 4.0, 5.0, 6.0]
    fwd_par = par_swap_rate(usd, [expiry] + pay_times)
    res = hw.jamshidian_swaption(expiry, pay_times, fwd_par, notional=100.0)
    print(f"\nJamshidian 1y-into-5y payer swaption (USD, strike = fwd par "
          f"{100*fwd_par:.4f}%):")
    print(f"  value = {res.value:.6f}   r* = {res.r_star:.6f}   "
          f"|residual| = {abs(res.residual):.1e}")
    eur_hw = HullWhite(a=0.05, sigma=0.008, curve=eur)
    eur_par = par_swap_rate(eur, [1.0, 2.0, 3.0, 4.0])
    eur_res = eur_hw.jamshidian_swaption(1.0, [2.0, 3.0, 4.0], eur_par, notional=100.0)
    print(f"  EUR (negative rates) 1y-into-3y @ {100*eur_par:.4f}%: "
          f"value = {eur_res.value:.6f}, r* = {eur_res.r_star:.6f}")

    # ---- 5. Hull-White calibration + Bachelier vols ------------------ #
    print("\nHull-White calibration to USD caplet/swaption prices (a, sigma free):")
    true = HullWhite(a=0.08, sigma=0.012, curve=usd)
    caplets = [
        CapletQuote(r, r + 1.0, atm, true.caplet(r, r + 1.0, atm, 1.0)) for r in (1.0, 2.0, 4.0)
    ]
    swaptions = [SwaptionQuote(expiry, tuple(pay_times), fwd_par,
                               true.jamshidian_swaption(expiry, pay_times, fwd_par, 1.0).value)]
    hcal = calibrate_hullwhite(usd, caplets, swaptions)
    print(
        f"  a={hcal.a:.6f}  sigma={hcal.sigma:.6f}  rmse={hcal.rmse:.1e}  "
        f"converged={hcal.converged}  identified={hcal.identified}  (true: 0.08, 0.012)"
    )
    print("  normal (bp) implied vols of the calibrated model:")
    for q in caplets:
        fwd = usd.fwd_rate(q.reset, q.pay)
        ann = (q.pay - q.reset) * usd.df(q.pay)
        vol = bachelier_implied_vol(q.price, fwd, q.strike, q.reset, ann)
        print(f"    caplet {q.reset:.0f}y->{q.pay:.0f}y  {1e4 * vol:6.1f} bp")
    ann = sum((b - a) * usd.df(b) for a, b in zip([expiry] + pay_times[:-1], pay_times))
    vol = bachelier_implied_vol(swaptions[0].price, fwd_par, fwd_par, expiry, ann)
    print(f"    1y-into-5y swaption   {1e4 * vol:6.1f} bp")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
