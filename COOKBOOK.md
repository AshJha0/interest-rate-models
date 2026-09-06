# COOKBOOK — irm recipes

Task-oriented recipes in all four languages. Python snippets run as-is
from the `python/` directory with `PYTHONPATH=src python3 <file>.py`
(paths to `../data/...` assume that working directory). The native
snippets follow the `API_SPEC.md` contract: C++ uses namespace `irm`
(snake_case, exceptions), Rust the `irm` crate (snake_case,
`Result<_, IrmError>` — snippets use `?` inside a function returning
`Result<(), IrmError>`), Java package `com.quant.irm` (camelCase methods,
`IllegalArgumentException`). C++ snippets show per-module headers under
`include/irm/` (`irm/irm.hpp` is the umbrella header). Every C++, Rust and
Java snippet below is extracted and compiled against the real library as
part of the repository's verification (see docs/ARCHITECTURE.md §7).

Numbers in the comments are golden values from
`data/golden/golden.json`, so you can eyeball correctness immediately.

---

## 1. How do I build a discount curve and query DF / zero / forward rates?

**Python**

```python
from irm import DiscountCurve

curve = DiscountCurve(times=[1.0, 2.0, 3.0, 5.0], dfs=[0.99, 0.975, 0.955, 0.9])
print(curve.df(2.5))            # 0.9649481851374196  (golden curve_df_mid)
print(curve.zero_rate(2.5))     # 0.014272349297139348 (golden curve_zero_mid)
print(curve.inst_forward(2.5))  # 0.02072613051711695  (constant on [2,3])
print(curve.fwd_rate(1.0, 3.0)) # 0.01832460732984298  (golden curve_fwd_simple)
print(curve.df(7.0))            # 0.8481675392670158   (flat-fwd extrapolation)
```

**C++**

```cpp
#include "irm/curve.hpp"
irm::DiscountCurve curve({1.0, 2.0, 3.0, 5.0}, {0.99, 0.975, 0.955, 0.9});
double df  = curve.df(2.5);          // 0.9649481851374196
double z   = curve.zero_rate(2.5);   // 0.0142723492971393
double fwd = curve.fwd_rate(1.0, 3.0);
```

**Rust**

```rust
use irm::DiscountCurve;
let curve = DiscountCurve::new(&[1.0, 2.0, 3.0, 5.0], &[0.99, 0.975, 0.955, 0.9])?;
let df  = curve.df(2.5)?;            // 0.9649481851374196
let z   = curve.zero_rate(2.5)?;
let fwd = curve.fwd_rate(1.0, 3.0)?;
```

**Java**

```java
import com.quant.irm.DiscountCurve;
DiscountCurve curve = new DiscountCurve(
    new double[]{1.0, 2.0, 3.0, 5.0}, new double[]{0.99, 0.975, 0.955, 0.9});
double df  = curve.df(2.5);          // 0.9649481851374196
double z   = curve.zeroRate(2.5);
double fwd = curve.fwdRate(1.0, 3.0);
```

Edge behavior (all languages): a query exactly at a pillar returns the
stored DF exactly; `t < 0`, NaN/inf, non-increasing pillar times, and
`DF <= 0` are rejected. `DF > 1` is legal (negative rates).

---

## 2. How do I bootstrap a curve from deposit and swap quotes?

**Python** (inline quotes; see recipe 3 for the CSV loaders)

```python
from irm import Deposit, Swap, bootstrap

instruments = [                      # strictly increasing pillars required
    Deposit(maturity=0.5, rate=-0.0040),
    Deposit(maturity=1.0, rate=-0.0030),
    Swap(maturity=2.0, rate=-0.0020),
    Swap(maturity=5.0, rate=0.0010),   # gap at 3y/4y: coupons interpolate
]
curve = bootstrap(instruments)
print(curve.df(1.0))   # 1.0030090270812433 — DF > 1, negative rates
for ins in instruments:                     # round-trip: repricing to ~1e-15
    assert abs(ins.residual(curve)) < 1e-9
```

**C++**

```cpp
#include "irm/bootstrap.hpp"
std::vector<irm::Instrument> ins = {
    irm::Deposit{0.5, -0.0040}, irm::Deposit{1.0, -0.0030},
    irm::Swap{2.0, -0.0020},    irm::Swap{5.0, 0.0010}};
irm::DiscountCurve curve = irm::bootstrap(ins);
double df1 = curve.df(1.0);              // 1.0030090270812433
```

**Rust**

```rust
use irm::{bootstrap, Instrument};
let ins = vec![
    Instrument::deposit(0.5, -0.0040)?, Instrument::deposit(1.0, -0.0030)?,
    Instrument::swap(2.0, -0.0020)?,    Instrument::swap(5.0, 0.0010)?];
let curve = bootstrap(&ins)?;
let df1 = curve.df(1.0)?;                // 1.0030090270812433
```

**Java**

```java
import com.quant.irm.*;
import java.util.List;
List<Instrument> ins = List.of(
    new Deposit(0.5, -0.0040), new Deposit(1.0, -0.0030),
    new Swap(2.0, -0.0020),    new Swap(5.0, 0.0010));
DiscountCurve curve = Bootstrap.bootstrap(ins);
double df1 = curve.df(1.0);              // 1.0030090270812433
```

---

## 3. How do I bootstrap the bundled EUR / USD / OIS curves from CSV?

**Python**

```python
from irm import bootstrap, load_curve_quotes, load_ois_quotes

eur = bootstrap(load_curve_quotes("../data/curve_quotes.csv", "EUR"))
usd = bootstrap(load_curve_quotes("../data/curve_quotes.csv", "USD"))
ois = bootstrap(load_ois_quotes("../data/ois_quotes.csv"))
print(eur.df(10.0))       # 0.9217932521073945 (golden boot_eur_df_10y)
print(usd.df(10.0))       # 0.6608868973765604 (golden boot_usd_df_10y)
print(ois.df(10.0))       # 0.6653290507800425 (golden boot_ois_df_10y)
print(eur.zero_rate(30.0))# 0.015515558481052624 (golden boot_eur_zero_30y)
```

**C++**

```cpp
#include "irm/quotes.hpp"
auto eur = irm::bootstrap(irm::load_curve_quotes("../data/curve_quotes.csv", "EUR"));
auto ois = irm::bootstrap(irm::load_ois_quotes("../data/ois_quotes.csv"));
double df10 = eur.df(10.0);              // 0.9217932521073945
```

**Rust**

```rust
use irm::{bootstrap, load_curve_quotes, load_ois_quotes};
use std::path::Path;
let eur = bootstrap(&load_curve_quotes(Path::new("../data/curve_quotes.csv"), "EUR")?)?;
let ois = bootstrap(&load_ois_quotes(Path::new("../data/ois_quotes.csv"))?)?;
let df10 = eur.df(10.0)?;                // 0.9217932521073945
```

**Java**

```java
import com.quant.irm.*;
import java.nio.file.Paths;
DiscountCurve eur = Bootstrap.bootstrap(
    Quotes.loadCurveQuotes(Paths.get("../data/curve_quotes.csv"), "EUR"));
DiscountCurve ois = Bootstrap.bootstrap(Quotes.loadOisQuotes(Paths.get("../data/ois_quotes.csv")));
double df10 = eur.df(10.0);              // 0.9217932521073945
```

The OIS curve is the discount curve a multi-curve setup would use; its
floating-leg residual is identical in form to the par swap's because
daily compounding is approximated as continuous (gap `O(r^2/365)`).

---

## 4. How do I detect crossed / arbitrageable quotes?

Bad quotes fail in one of two places: at instrument construction (the
quote *alone* implies `DF <= 0`, or the maturity is outside `[1e-6, 200]`
years) or at solve time (no positive DF can reprice it against its
neighbors). Both raise the language's standard error naming the problem.
A solver failure that is *not* a bracketing failure (a NaN residual, an
exhausted iteration budget) is reported as "solver failed at pillar" —
never relabelled as crossed quotes.

**Python**

```python
from irm import Deposit, FRA, Swap, bootstrap

try:
    Deposit(maturity=10.0, rate=-0.15)      # 1 + R*T = -0.5 <= 0
except ValueError as e:
    print("construction:", e)

try:                                        # 2y quote way above 1y: crossed
    bootstrap([Swap(1.0, 0.02), Swap(2.0, 5.0)])
except ValueError as e:
    print("solve:", e)                      # names pillar t=2.0

try:
    bootstrap([Swap(2.0, 0.02), Swap(2.0, 0.021)])   # duplicate pillar
except ValueError as e:
    print("ordering:", e)
```

**C++** — `std::invalid_argument` (construction/ordering) or
`std::domain_error` (no admissible DF / solver failure); **Rust** —
`Err(IrmError::InvalidInput(..))` at construction and
`Err(IrmError::Bootstrap(..))` at solve time (the message says "crossed"
or "solver failed"); **Java** — `IllegalArgumentException`. Same three
failure sites in every port:

```cpp
#include "irm/bootstrap.hpp"
try {
    irm::bootstrap({irm::Swap{1.0, 0.02}, irm::Swap{2.0, 5.0}});
} catch (const std::domain_error& e) {
    std::puts(e.what());                     // names pillar t=2.0, says "crossed"
}
```

```rust
use irm::{bootstrap, Instrument, IrmError};
match Instrument::deposit(10.0, -0.15) {
    Err(IrmError::InvalidInput(msg)) => eprintln!("construction: {msg}"),
    other => panic!("unexpected {other:?}"),
}
let crossed = [Instrument::swap(1.0, 0.02)?, Instrument::swap(2.0, 5.0)?];
match bootstrap(&crossed) {
    Err(IrmError::Bootstrap(msg)) => eprintln!("solve: {msg}"),   // "crossed/arbitrageable"
    other => panic!("unexpected {other:?}"),
}
```

```java
import com.quant.irm.*;
import java.util.List;
try { Bootstrap.bootstrap(List.of(new Swap(1.0, 0.02), new Swap(2.0, 5.0))); }
catch (IllegalArgumentException e) { System.out.println("solve: " + e.getMessage()); }
try { new Swap(1e12, 0.03); }                            // days typed as years
catch (IllegalArgumentException e) { System.out.println("construction: " + e.getMessage()); }
```

---

## 5. How do I price ZCBs and get the short-rate distribution under Vasicek?

**Python**

```python
from irm import Vasicek

v = Vasicek(kappa=0.5, theta=0.03, sigma=0.01, r0=0.02)
print(v.zcb_price(5.0))     # 0.8770621876849438 (golden vas_zcb_5y)
print(v.zero_yield(5.0))    # 0.026235475912595747
print(v.r_mean(2.0))        # 0.026321205588285577 (golden vas_rT_mean)
print(v.r_var(2.0))         # 8.646647167633873e-05 (golden vas_rT_var)

neg = Vasicek(kappa=0.3, theta=0.01, sigma=0.008, r0=-0.005)   # r0 < 0 fine
print(neg.zcb_price(5.0))   # 0.989399475473229 (golden vas_zcb_neg_r0)
```

**C++**

```cpp
#include "irm/vasicek.hpp"
irm::Vasicek v(0.5, 0.03, 0.01, 0.02);   // kappa, theta, sigma, r0
double p  = v.zcb_price(5.0);            // 0.8770621876849438
double m  = v.r_mean(2.0);               // 0.026321205588285577
double vr = v.r_var(2.0);                // 8.646647167633873e-05
```

**Rust**

```rust
use irm::Vasicek;
let v = Vasicek::new(0.5, 0.03, 0.01, 0.02)?;
let p  = v.zcb_price(5.0, 0.0, None)?;   // 0.8770621876849438  (maturity, t, r: None = r0)
let m  = v.r_mean(2.0, None)?;           // 0.026321205588285577
let vr = v.r_var(2.0)?;
```

**Java**

```java
import com.quant.irm.Vasicek;
Vasicek v = new Vasicek(0.5, 0.03, 0.01, 0.02);
double p  = v.zcbPrice(5.0);             // 0.8770621876849438
double m  = v.rMean(2.0);
double vr = v.rVar(2.0);
```

---

## 6. How do I price European options on zero-coupon bonds (and check parity)?

**Python**

```python
from irm import Vasicek

v = Vasicek(kappa=0.5, theta=0.03, sigma=0.01, r0=0.02)
zbc = v.zbc(expiry=1.0, bond_maturity=3.0, strike=0.95)
zbp = v.zbp(expiry=1.0, bond_maturity=3.0, strike=0.95)
print(zbc)   # 0.0033292795093556315 (golden vas_zbc)
print(zbp)   # 0.004148300214835365  (golden vas_zbp)
# put-call parity: ZBC - ZBP = P(0,S) - K P(0,T)
assert abs((zbc - zbp) - (v.zcb_price(3.0) - 0.95 * v.zcb_price(1.0))) < 1e-14
```

**C++**

```cpp
#include "irm/vasicek.hpp"
irm::Vasicek v(0.5, 0.03, 0.01, 0.02);
double zbc = v.zbc(1.0, 3.0, 0.95);      // 0.0033292795093556315
double zbp = v.zbp(1.0, 3.0, 0.95);      // 0.0041483002148353650
```

**Rust**

```rust
let v = irm::Vasicek::new(0.5, 0.03, 0.01, 0.02)?;
let zbc = v.zbc(1.0, 3.0, 0.95)?;        // 0.0033292795093556315
let zbp = v.zbp(1.0, 3.0, 0.95)?;
```

**Java**

```java
Vasicek v = new Vasicek(0.5, 0.03, 0.01, 0.02);
double zbc = v.zbc(1.0, 3.0, 0.95);      // 0.0033292795093556315
double zbp = v.zbp(1.0, 3.0, 0.95);
```

With `sigma = 0` (or `expiry = 0`, or `S = T`) the option returns
intrinsic on the forward bond, `max(P(0,S) - K*P(0,T), 0)` — no 0/0.

---

## 7. How do I calibrate Vasicek to market zero yields — and read the diagnostics?

The fit runs from three starts and reports whether they agree. On the
bundled data they do **not**: two minima fit within 2 % of each other in
rmse, so yields alone do not identify `(kappa, theta, sigma)`. Fix
`sigma` from option prices and the fit is unique.

**Python**

```python
from irm import calibrate_vasicek, load_zero_yields

ts, ys = load_zero_yields("../data/zero_yields.csv")
cal = calibrate_vasicek(ts, ys, r0=0.03)
print(cal.kappa, cal.theta, cal.sigma)   # 0.4114, 0.0688, 0.0805 (best of 3 starts)
print(cal.rmse, cal.converged, cal.at_bound)   # 1.70e-05, True, False
print(cal.identified, cal.sigma_spread)  # False, 0.058 -> yields do not pin sigma

cal = calibrate_vasicek(ts, ys, r0=0.03, sigma_fixed=0.02)   # sigma from options
print(cal.kappa, cal.theta, cal.identified)   # 0.8002, 0.0500, True
model = cal.model()                      # a Vasicek you can price with
print(model.zero_yield(10.0))

warm = calibrate_vasicek(ts, ys, r0=0.03, x0=(0.8, 0.05, 0.02))   # single warm start
assert warm.n_starts == 1 and not warm.identified    # identifiability not assessed
```

Non-convergence is *reported* (`converged=False`), never raised; a
solution on the domain wall sets `at_bound=True` and `converged=False`.
Price on a calibration only if `converged and (identified or sigma_fixed)`.

**C++**

```cpp
#include "irm/quotes.hpp"
#include "irm/vasicek.hpp"
auto [ts, ys] = irm::load_zero_yields("../data/zero_yields.csv");
irm::VasicekCalibration cal = irm::calibrate_vasicek(ts, ys, /*r0=*/0.03);
if (!cal.converged || !cal.identified) { /* report: on the bundled data identified == false */ }
cal = irm::calibrate_vasicek(ts, ys, 0.03, std::nullopt, 4000, /*sigma_fixed=*/0.02);
irm::Vasicek model = cal.model();         // kappa 0.8002, theta 0.0500, identified
```

**Rust**

```rust
use irm::{calibrate_vasicek, load_zero_yields};
use std::path::Path;
let (ts, ys) = load_zero_yields(Path::new("../data/zero_yields.csv"))?;
let cal = calibrate_vasicek(&ts, &ys, 0.03, None, 4000, None)?; // Err only on bad input
assert!(cal.converged && !cal.identified);                     // bundled data: not identified
let cal = calibrate_vasicek(&ts, &ys, 0.03, None, 4000, Some(0.02))?;
assert!(cal.converged && cal.identified);
let model = cal.model()?;
```

**Java**

```java
import com.quant.irm.*;
import java.nio.file.Paths;
double[][] data = Quotes.loadZeroYields(Paths.get("../data/zero_yields.csv"));
Vasicek.Calibration cal = Vasicek.calibrate(data[0], data[1], 0.03);
if (!cal.converged() || !cal.identified()) { /* report */ }
cal = Vasicek.calibrate(data[0], data[1], 0.03, null, 4000, 0.02);   // sigma fixed
Vasicek model = cal.model();
```

---

## 8. How do I run Monte Carlo and verify it against the closed form?

The simulation samples the exact OU transition of `(r, ∫r dt)`, so the
estimator is unbiased at any step count — use few steps, many paths.

**Python**

```python
from irm import Vasicek

v = Vasicek(kappa=0.5, theta=0.03, sigma=0.01, r0=0.02)
price, se = v.mc_zcb(maturity=5.0, n_steps=8, n_paths=50_000, seed=42)
analytic = v.zcb_price(5.0)              # 0.8770621876849438
assert abs(price - analytic) < 3.0 * se  # ~1.2e-4 SE at 50k paths
print(f"MC {price:.6f} +/- {se:.1e} vs analytic {analytic:.6f}")

v0 = Vasicek(kappa=0.5, theta=0.03, sigma=0.0, r0=0.02)   # sigma = 0
p0, se0 = v0.mc_zcb(5.0, 8, 1000, seed=1)
assert se0 < 1e-15 and abs(p0 - v0.zcb_price(5.0)) < 1e-12  # deterministic
```

**C++**

```cpp
irm::Vasicek v(0.5, 0.03, 0.01, 0.02);
auto [price, se] = v.mc_zcb(5.0, /*n_steps=*/8, /*n_paths=*/50000, /*seed=*/42);
// |price - 0.8770621876849438| < 3*se with any RNG — the scheme is exact
```

**Rust**

```rust
let v = irm::Vasicek::new(0.5, 0.03, 0.01, 0.02)?;
let (price, se) = v.mc_zcb(5.0, 8, 50_000, 42)?;
assert!((price - v.zcb_price(5.0, 0.0, None)?).abs() < 3.0 * se);
```

**Java**

```java
Vasicek v = new Vasicek(0.5, 0.03, 0.01, 0.02);
McEstimate mc = v.mcZcb(5.0, 8, 50000, 42L);   // mc.price(), mc.standardError()
```

Seeds fix a language's own stream, not cross-language equality — ports
agree *statistically* (golden bands are 6 SE), not path-by-path. A seed
reproduces a stream only for a pinned toolchain and dependency set:
libstdc++'s `std::normal_distribution`, Rust's `StdRng` (explicitly not
stable across `rand` versions) and `java.util.Random` all differ, so do
not archive seeds as if they were results.

---

## 9. How do I set up Hull-White on a bootstrapped curve and check the fit?

**Python**

```python
from irm import DiscountCurve, HullWhite

mkt = DiscountCurve(
    [1.0, 2.0, 3.0, 5.0, 10.0],
    [0.9787294774691476, 0.9550419621907147, 0.9291361457915193,
     0.8715343499971578, 0.7046880897187134])
hw = HullWhite(a=0.1, sigma=0.01, curve=mkt)

print(hw.zcb_price(0.0, 5.0))   # 0.8715343499971578 == mkt.df(5.0) exactly
print(hw.fwd0(0.0))             # r(0) = f(0,0) = 0.0215
print(hw.theta(0.5))            # ~0.0022 — fitted drift, inspection only
# theta(1.0) is ~15.0: the FD d f/dt spikes at pillars because piecewise-
# constant forwards jump there. Harmless: pricing never evaluates theta.
```

**C++**

```cpp
#include "irm/hullwhite.hpp"
irm::DiscountCurve mkt({1,2,3,5,10}, {0.9787294774691476, 0.9550419621907147,
                        0.9291361457915193, 0.8715343499971578, 0.7046880897187134});
irm::HullWhite hw(0.1, 0.01, mkt);
double p0 = hw.zcb_price(0.0, 5.0);      // == mkt.df(5.0) to 1e-12 (golden hw_zcb_t0)
```

**Rust**

```rust
let mkt = irm::DiscountCurve::new(&[1.0,2.0,3.0,5.0,10.0],
    &[0.9787294774691476, 0.9550419621907147, 0.9291361457915193,
      0.8715343499971578, 0.7046880897187134])?;
let hw = irm::HullWhite::new(0.1, 0.01, mkt.clone())?;   // takes the curve by value
let p0 = hw.zcb_price(0.0, 5.0, None)?;  // == mkt.df(5.0)?
```

**Java**

```java
DiscountCurve mkt = new DiscountCurve(
    new double[]{1,2,3,5,10},
    new double[]{0.9787294774691476, 0.9550419621907147, 0.9291361457915193,
                 0.8715343499971578, 0.7046880897187134});
HullWhite hw = new HullWhite(0.1, 0.01, mkt);
double p0 = hw.zcbPrice(0.0, 5.0);       // == mkt.df(5.0)
```

---

## 10. How do I price caplets, floorlets and caps under Hull-White?

**Python**

```python
from irm import DiscountCurve, HullWhite

mkt = DiscountCurve(
    [1.0, 2.0, 3.0, 5.0, 10.0],
    [0.9787294774691476, 0.9550419621907147, 0.9291361457915193,
     0.8715343499971578, 0.7046880897187134])
hw = HullWhite(a=0.1, sigma=0.01, curve=mkt)

cpl = hw.caplet(reset=1.0, pay=2.0, strike=0.025, notional=100.0)
flr = hw.floorlet(1.0, 2.0, 0.025, 100.0)
print(cpl)    # 0.3444276269860027 (golden hw_caplet)
print(flr)    # 0.3632810046194777 (golden hw_floorlet)
# cap/floor parity: caplet - floorlet = tau*(F - K)*DF(pay)*N
F = mkt.fwd_rate(1.0, 2.0)
assert abs((cpl - flr) - 100.0 * (F - 0.025) * mkt.df(2.0)) < 1e-10

cap = hw.cap([0.5, 1.0, 2.0, 3.0], strike=0.024, notional=100.0)
print(cap)    # 1.1428970000132672 (golden hw_cap_3y) — sum of 3 caplets
# uneven accruals are fine: caplets on [0.5,1], [1,2], [2,3]
assert abs(cap - sum(hw.caplet(a, b, 0.024, 100.0)
                     for a, b in [(0.5, 1.0), (1.0, 2.0), (2.0, 3.0)])) < 1e-12

hw0 = HullWhite(a=0.1, sigma=0.0, curve=mkt)   # sigma = 0: intrinsic
print(hw0.caplet(1.0, 2.0, 0.02, 100.0))       # 0.45866760346186464
```

**C++**

```cpp
irm::HullWhite hw(0.1, 0.01, mkt);
double cpl = hw.caplet(1.0, 2.0, 0.025, 100.0);    // 0.3444276269860027
double cap = hw.cap({0.5, 1.0, 2.0, 3.0}, 0.024, 100.0);  // 1.1428970000132672
```

**Rust**

```rust
let cpl = hw.caplet(1.0, 2.0, 0.025, 100.0)?;      // 0.3444276269860027
let cap = hw.cap(&[0.5, 1.0, 2.0, 3.0], 0.024, 100.0)?;  // 1.1428970000132672
```

**Java**

```java
double cpl = hw.caplet(1.0, 2.0, 0.025, 100.0);    // 0.3444276269860027
double cap = hw.cap(new double[]{0.5, 1.0, 2.0, 3.0}, 0.024, 100.0);  // 1.1428970000132672
```

A caplet is `(1 + K*tau)` ZCB *puts* struck at `1/(1 + K*tau)` — see
LEARN.md §5.2 for the replication; the strike must satisfy
`1 + K*tau > 0` (deep-negative strikes are rejected).

---

## 11. How do I price a European swaption with Jamshidian's decomposition?

**Python**

```python
from irm import DiscountCurve, HullWhite, par_swap_rate

mkt = DiscountCurve(
    [1.0, 2.0, 3.0, 5.0, 10.0],
    [0.9787294774691476, 0.9550419621907147, 0.9291361457915193,
     0.8715343499971578, 0.7046880897187134])
hw = HullWhite(a=0.1, sigma=0.01, curve=mkt)

# 1y expiry into a 5y annual swap, fixed rate 2.8%, notional 100
res = hw.jamshidian_swaption(expiry=1.0, pay_times=[2.0, 3.0, 4.0, 5.0, 6.0],
                             fixed_rate=0.028, notional=100.0, payer=True)
print(res.value)      # 2.4419483143080454 (golden hw_swaption_payer)
print(res.r_star)     # 0.0179465821156358
print(abs(res.residual) < 1e-10, res.strikes)  # decomposition diagnostics

# receiver via the same call; payer - receiver = value of the fwd swap
rec = hw.jamshidian_swaption(1.0, [2.0, 3.0, 4.0, 5.0, 6.0], 0.028,
                             notional=100.0, payer=False)
atm = par_swap_rate(mkt, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(f"fwd par = {atm:.4%}; payer-receiver = {res.value - rec.value:.6f}")
```

**C++**

```cpp
irm::JamshidianResult res = hw.jamshidian_swaption(
    1.0, {2.0, 3.0, 4.0, 5.0, 6.0}, 0.028, 100.0, /*payer=*/true);
// res.value == 2.4419483143080454, res.r_star == 0.0179465821, res.strikes[...]
```

**Rust**

```rust
let res = hw.jamshidian_swaption(1.0, &[2.0, 3.0, 4.0, 5.0, 6.0], 0.028, 100.0, true)?;
assert!(res.residual.abs() < 1e-10);     // enforced by the library, reported here
```

**Java**

```java
HullWhite.JamshidianResult res = hw.jamshidianSwaption(
    1.0, new double[]{2.0, 3.0, 4.0, 5.0, 6.0}, 0.028, 100.0, true);
double value = res.value();              // 2.4419483143080454
```

Works unchanged on negative-rate curves (`hw_swaption_neg_curve` prices a
0%-strike payer on the EUR-style curve; `r*` comes out negative) and for
negative fixed rates (`X = -0.5` still prices exactly). The residual
contract `|g(r*)| < 1e-10` and a negative slope at `r*` are enforced in
every port; `jamshidian_at_r_star(expiry, pay_times, X, r_star, ...)`
re-runs the decomposition from a known root (e.g. receiver from the
payer's `r*`) and raises if the root does not satisfy the contract.

---

## 12. How do I compute curve DV01 (parallel and keyed) by re-bootstrapping?

There is no dedicated DV01 function — the intended pattern is *bump the
quotes, re-bootstrap, reprice*, which works identically in every port.

**Python**

```python
from irm import Deposit, Swap, bootstrap, load_curve_quotes, par_swap_rate

quotes = load_curve_quotes("../data/curve_quotes.csv", "USD")
base = bootstrap(quotes)
par10 = par_swap_rate(base, [0.0] + list(range(1, 11)))   # 0.0425

def rebuilt(bump, pillar=None):
    """Re-bootstrap with +bump on one pillar (or all when pillar is None)."""
    bumped = [type(q)(maturity=q.maturity,
                      rate=q.rate + (bump if pillar in (None, q.pillar) else 0.0))
              for q in quotes]                       # Deposit/Swap have same fields
    return bootstrap(bumped)

def receiver_pv(curve, rate, T=10, notional=100.0):
    annuity = sum(curve.df(t) for t in range(1, T + 1))
    return notional * (rate * annuity - (1.0 - curve.df(T)))

pv0 = receiver_pv(base, par10)                       # 0.0 at par
dv01_par = receiver_pv(rebuilt(1e-4), par10) - pv0   # -0.0798 per 100 notional
dv01_10y = receiver_pv(rebuilt(1e-4, 10.0), par10) - pv0   # -0.0798 (all of it)
dv01_5y  = receiver_pv(rebuilt(1e-4, 5.0), par10) - pv0    # ~1e-14 (none)
print(dv01_par, dv01_10y, dv01_5y)
```

The keyed result — all sensitivity in the instrument's own bucket — is
the quote-space Jacobian localization discussed in LEARN.md §6, and it
depends on the *local* interpolation; expect smearing if you swap in
splines.

**C++ / Rust / Java** — same pattern: copy the instrument list with
bumped rates, call `bootstrap` again, reprice with `df`/`par_swap_rate`
(`parSwapRate` in Java). Bootstrapping the 14-instrument USD curve costs
about 0.8 ms in the Python reference and about 30 µs in the C++ port
(measured, see docs/ARCHITECTURE.md §8), so a full keyed ladder is a
loop, not a project. The localisation claim is a regression test in all
four suites (`dv01_localises_in_quote_space`).

---

## 13. How do I price with Hull-White Monte Carlo and check the caplet?

**Python**

```python
from irm import DiscountCurve, HullWhite

mkt = DiscountCurve(
    [1.0, 2.0, 3.0, 5.0, 10.0],
    [0.9787294774691476, 0.9550419621907147, 0.9291361457915193,
     0.8715343499971578, 0.7046880897187134])
hw = HullWhite(a=0.1, sigma=0.01, curve=mkt)
mc, se = hw.mc_caplet(reset=1.0, pay=2.0, strike=0.025, notional=100.0,
                      n_steps=8, n_paths=50_000, seed=7)
analytic = hw.caplet(1.0, 2.0, 0.025, 100.0)   # 0.3444276269860027
assert abs(mc - analytic) < 3.0 * se           # exact transitions: no bias
```

**C++**

```cpp
auto [mc, se] = hw.mc_caplet(1.0, 2.0, 0.025, 100.0, 8, 50000, 7);
```

**Rust**

```rust
let (mc, se) = hw.mc_caplet(1.0, 2.0, 0.025, 100.0, 8, 50_000, 7)?;
```

**Java**

```java
McEstimate mc = hw.mcCaplet(1.0, 2.0, 0.025, 100.0, 8, 50000, 7L);  // price(), standardError()
```

The pathwise discount factor is exact
(`D(0,T) = P^M(0,T) e^{-V(T) - ∫x}`), so the only error is statistical —
compare against the closed form with a 3-SE band, never with a fixed
tolerance. The step moments use a series for small `a·dt`, so many small
steps (`n_steps = 2000` with `a = 1e-3`) remain unbiased.

---

## 14. How do I calibrate Hull-White `(a, sigma)` to caplet / swaption prices?

Quotes are per unit notional; caplets as `(reset, pay, strike, price)`,
payer swaptions as `(expiry, pay_times, fixed_rate, price)`. The fit runs
three Nelder-Mead starts and reports the same diagnostics as the Vasicek
fit. The recipe below generates quotes from a known model so the answer
is checkable; in practice the prices come from `bachelier_price` applied
to broker bp vols (recipe 15).

**Python**

```python
from irm import CapletQuote, DiscountCurve, HullWhite, SwaptionQuote, calibrate_hullwhite

mkt = DiscountCurve(
    [1.0, 2.0, 3.0, 5.0, 10.0],
    [0.9787294774691476, 0.9550419621907147, 0.9291361457915193,
     0.8715343499971578, 0.7046880897187134])
truth = HullWhite(a=0.08, sigma=0.012, curve=mkt)
caplets = [CapletQuote(r, r + 1.0, 0.025, truth.caplet(r, r + 1.0, 0.025, 1.0))
           for r in (1.0, 2.0, 4.0)]
pays = (2.0, 3.0, 4.0, 5.0, 6.0)
swaptions = [SwaptionQuote(1.0, pays, 0.028,
                           truth.jamshidian_swaption(1.0, pays, 0.028, 1.0).value)]

cal = calibrate_hullwhite(mkt, caplets, swaptions)
print(cal.a, cal.sigma)                  # 0.08, 0.012 (golden hw_calib_recover, tol 1e-8)
print(cal.converged, cal.identified, cal.at_bound, cal.rmse)   # True, True, False, ~1e-13
hw = cal.model(mkt)

cal_a = calibrate_hullwhite(mkt, caplets, swaptions, a_fixed=0.08)   # fit sigma only (1-D)
one = calibrate_hullwhite(mkt, caplets[:1])          # one quote, two parameters:
assert not one.identified                             # a ridge of (a, sigma) pairs
```

**C++**

```cpp
#include "irm/hullwhite.hpp"
irm::HullWhite truth(0.08, 0.012, mkt);
std::vector<irm::CapletQuote> caplets;
for (double r : {1.0, 2.0, 4.0})
    caplets.emplace_back(r, r + 1.0, 0.025, truth.caplet(r, r + 1.0, 0.025, 1.0));
std::vector<double> pays = {2.0, 3.0, 4.0, 5.0, 6.0};
std::vector<irm::SwaptionQuote> swaptions = {irm::SwaptionQuote(
    1.0, pays, 0.028, truth.jamshidian_swaption(1.0, pays, 0.028, 1.0).value)};
irm::HullWhiteCalibration cal = irm::calibrate_hullwhite(mkt, caplets, swaptions);
// cal.a == 0.08, cal.sigma == 0.012 (1e-8), cal.converged && cal.identified
irm::HullWhiteCalibration cal_a = irm::calibrate_hullwhite(mkt, caplets, swaptions, /*a_fixed=*/0.08);
```

**Rust**

```rust
use irm::{calibrate_hullwhite, CapletQuote, HullWhite, SwaptionQuote};
let truth = HullWhite::new(0.08, 0.012, mkt.clone())?;
let mut caplets = Vec::new();
for r in [1.0, 2.0, 4.0] {
    caplets.push(CapletQuote::new(r, r + 1.0, 0.025, truth.caplet(r, r + 1.0, 0.025, 1.0)?)?);
}
let pays = [2.0, 3.0, 4.0, 5.0, 6.0];
let swaptions = [SwaptionQuote::new(
    1.0, &pays, 0.028, truth.jamshidian_swaption(1.0, &pays, 0.028, 1.0, true)?.value)?];
let cal = calibrate_hullwhite(&mkt, &caplets, &swaptions, None, None, 4000)?;
assert!(cal.converged && cal.identified && (cal.a - 0.08).abs() < 1e-8);
let cal_a = calibrate_hullwhite(&mkt, &caplets, &swaptions, Some(0.08), None, 4000)?;
```

**Java**

```java
import com.quant.irm.*;
import java.util.ArrayList;
import java.util.List;
HullWhite truth = new HullWhite(0.08, 0.012, mkt);
List<HullWhite.CapletQuote> caplets = new ArrayList<>();
for (double r : new double[]{1.0, 2.0, 4.0})
    caplets.add(new HullWhite.CapletQuote(r, r + 1.0, 0.025, truth.caplet(r, r + 1.0, 0.025, 1.0)));
double[] pays = {2.0, 3.0, 4.0, 5.0, 6.0};
List<HullWhite.SwaptionQuote> swaptions = List.of(new HullWhite.SwaptionQuote(
    1.0, pays, 0.028, truth.jamshidianSwaption(1.0, pays, 0.028, 1.0, true).value()));
HullWhite.Calibration cal = HullWhite.calibrate(mkt, caplets, swaptions, null, null, 4000);
// cal.a() == 0.08, cal.sigma() == 0.012, cal.converged() && cal.identified()
HullWhite.Calibration calA = HullWhite.calibrate(mkt, caplets, swaptions, 0.08, null, 4000);
```

---

## 15. How do I convert between prices and normal (bp) implied vols?

`bachelier_price(forward, strike, expiry, vol, annuity, payer)` and
`bachelier_implied_vol(price, forward, strike, expiry, annuity, payer)`.
The annuity is `tau * DF(pay)` for a caplet and `sum tau_i DF(u_i)` for a
swaption; `vol` is absolute (`0.0075` = 75 bp). Negative forwards and
strikes are fine.

**Python**

```python
from irm import DiscountCurve, HullWhite, bachelier_implied_vol, bachelier_price, par_swap_rate

mkt = DiscountCurve(
    [1.0, 2.0, 3.0, 5.0, 10.0],
    [0.9787294774691476, 0.9550419621907147, 0.9291361457915193,
     0.8715343499971578, 0.7046880897187134])
hw = HullWhite(a=0.1, sigma=0.01, curve=mkt)

# Model caplet -> bp vol
reset, pay, k = 1.0, 2.0, 0.025
fwd, ann = mkt.fwd_rate(reset, pay), (pay - reset) * mkt.df(pay)
price = hw.caplet(reset, pay, k, 1.0)                     # 0.003444276269860027 per unit notional
vol = bachelier_implied_vol(price, fwd, k, reset, ann)    # ~0.0095 -> 95 bp
assert abs(bachelier_price(fwd, k, reset, vol, ann) - price) < 1e-15

# Broker quote -> price (what calibrate_hullwhite wants)
pays = [2.0, 3.0, 4.0, 5.0, 6.0]
f_swap = par_swap_rate(mkt, [1.0] + pays)
a_swap = sum((b - a) * mkt.df(b) for a, b in zip([1.0] + pays[:-1], pays))
quote_price = bachelier_price(f_swap, f_swap, 1.0, 0.0080, a_swap)   # 80 bp ATM payer
print(bachelier_price(0.03, 0.025, 2.0, 0.0075, 4.2))     # 0.030210864350656882 (golden)
```

**C++**

```cpp
#include "irm/bachelier.hpp"
double p = irm::bachelier_price(0.03, 0.025, 2.0, 0.0075, 4.2, /*payer=*/true);   // 0.030210864350656882
double v = irm::bachelier_implied_vol(p, 0.03, 0.025, 2.0, 4.2, true);            // 0.0075
```

**Rust**

```rust
use irm::{bachelier_implied_vol, bachelier_price};
let p = bachelier_price(0.03, 0.025, 2.0, 0.0075, 4.2, true)?;   // 0.030210864350656882
let v = bachelier_implied_vol(p, 0.03, 0.025, 2.0, 4.2, true)?;  // 0.0075
```

**Java**

```java
double p = Bachelier.price(0.03, 0.025, 2.0, 0.0075, 4.2, true);      // 0.030210864350656882
double v = Bachelier.impliedVol(p, 0.03, 0.025, 2.0, 4.2, true);      // 0.0075
```

A price below intrinsic raises; a price at intrinsic (to 1e-15) returns
vol 0; deep in-the-money quotes whose time value is below double
precision round-trip in *price*, not in vol — which is the honest answer.
