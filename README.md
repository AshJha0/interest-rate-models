# Interest Rate Models (Bootstrapping, Vasicek, Hull-White)

Four parallel implementations (Python reference, C++, Rust, Java) of the
core fixed-income modeling stack: discount-curve construction from market
quotes, the Vasicek short-rate model, and a curve-fitted Hull-White
one-factor model with calibration to caplet/swaption prices — all sharing a
golden-value test suite that pins every language to identical numbers.

* **Discount curve**: pillar times + discount factors, log-linear
  interpolation in ln(DF) (piecewise-constant instantaneous forwards),
  flat-forward extrapolation, continuous zero rates, instantaneous and
  simple forwards, par-swap-rate helper. Negative rates are first-class:
  DF > 1 is legal.
* **Bootstrap**: sequential single-curve bootstrap from deposits, FRAs and
  annual par swaps plus an OIS variant, each pillar solved with a native
  Brent root find; strict pillar ordering, maturity bounds, and
  crossed/arbitrageable-quote detection (bracketing failure → error that
  says so; any other solver failure is reported as such, never relabelled).
* **Vasicek** `dr = kappa(theta - r)dt + sigma dW`: affine A/B closed-form
  ZCB prices and yields, exact mean/variance of `r_T`, Jamshidian (1989)
  ZCB call/put formula, exact-transition (bias-free) Monte Carlo of
  `(r, ∫r dt)`, multi-start least-squares calibration to zero yields via
  native Nelder-Mead with `sigma_fixed`, `at_bound` and `identified`
  diagnostics (non-convergence reported, never thrown).
* **Hull-White 1F** `dr = (theta(t) - a r)dt + sigma dW` fitted to a market
  curve: `theta(t)` from finite-difference forwards, affine ZCB via market
  DFs (exact curve reproduction at t = 0), ZBC/ZBP closed forms,
  caplets/floorlets as ZCB options, caps as caplet strips, Jamshidian
  swaption decomposition with an enforced `|g(r*)| < 1e-10` contract,
  exact-Gaussian MC (trinomial-tree-free), and `calibrate_hullwhite`
  fitting `(a, sigma)` to caplet / payer-swaption prices.
* **Bachelier (normal) model**: price and implied "bp" volatility for
  caplets/swaptions so model prices can be compared with broker quotes.
* **Shared kernels** (implemented natively in every port — no external
  numerics): Brent + bisection root finders (non-finite values and
  non-convergence are errors), Nelder-Mead simplex, normal CDF/PDF via
  `erfc`, exact Ornstein-Uhlenbeck step moments evaluated with `expm1` and
  a power series so they stay correct down to `a·dt ~ 1e-12`.

Theory: [LEARN.md](LEARN.md). Recipes: [COOKBOOK.md](COOKBOOK.md).
Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Language contract:
[API_SPEC.md](API_SPEC.md). Docs site: https://ashjha0.github.io/interest-rate-models/
(publishing guide: [docs/GITHUB_PAGES.md](docs/GITHUB_PAGES.md)).

## Feature matrix

| Feature | Python | C++ | Rust | Java |
|---|---|---|---|---|
| DiscountCurve (ln-DF interp, flat-fwd extrap) | yes | yes | yes | yes |
| Zero / inst-forward / simple forward / par rate | yes | yes | yes | yes |
| Bootstrap: deposits, FRAs, par swaps (maturity in [1e-6, 200] y) | yes | yes | yes | yes |
| OIS bootstrap variant | yes | yes | yes | yes |
| Crossed-quote detection vs. solver-failure reporting | yes | yes | yes | yes |
| Vasicek ZCB, yields, r_T mean/var | yes | yes | yes | yes |
| Vasicek ZBC/ZBP (Jamshidian 1989) | yes | yes | yes | yes |
| Vasicek exact-transition MC + SE | yes | yes | yes | yes |
| Vasicek calibration: 3 starts, `sigma_fixed`, `at_bound`, spreads, `identified` | yes | yes | yes | yes |
| Hull-White ZCB via A/B on market DFs | yes | yes | yes | yes |
| Hull-White theta(t), caplet/floorlet/cap | yes | yes | yes | yes |
| Jamshidian swaption (payer/receiver), `jamshidian_at_r_star` contract | yes | yes | yes | yes |
| Hull-White exact-Gaussian MC caplet | yes | yes | yes | yes |
| Hull-White calibration to caplet/swaption prices (`a_fixed` option) | yes | yes | yes | yes |
| Bachelier normal price / implied vol | yes | yes | yes | yes |
| Brent + bisection, Nelder-Mead (native) | yes | yes | yes | yes |
| Golden-value suite (33 cases, one test per case) | yes | yes | yes | yes |
| Error style | `ValueError` | `std::invalid_argument` / `std::domain_error` | `Result<_, IrmError>` (`InvalidInput`, `RootFind`, `Bootstrap`, `Data`) | `IllegalArgumentException` |

## Directory layout

```
interest-rate-models/
  README.md, LEARN.md, COOKBOOK.md, API_SPEC.md, LICENSE (MIT)
  docs/
    ARCHITECTURE.md          # design + Mermaid diagrams
    diagrams/*.mmd           # raw diagram sources
  data/
    curve_quotes.csv         # EUR (negative short end) + USD deposit/swap quotes
    ois_quotes.csv           # OIS par swap quotes 1M-30Y
    zero_yields.csv          # Vasicek calibration targets (seeded noise)
    generate_data.py         # regenerates all three (validates before writing)
    golden/golden.json       # 33 cross-language reference cases
  python/
    src/irm/                 # reference package (curve, bootstrap, vasicek, hullwhite, bachelier, ...)
    tests/                   # pytest suites incl. golden runner (220 tests)
    demo.py                  # end-to-end demo (output below)
  cpp/                       # C++17, CMake >= 3.16, GoogleTest (namespace irm) — 130 tests
  rust/                      # edition 2021 (crate irm)                        — 102 tests
  java/                      # Java 21, JUnit 4 (com.quant.irm)                — 162 tests
```

## Build, run, test

All scripts are invoked with `bash` so a fresh clone works regardless of
file modes.

### Python (3.11; numpy; pytest + scipy only as test cross-checks)

```bash
cd python
PYTHONPATH=src python3 demo.py           # run the demo
PYTHONPATH=src python3 -m pytest -q      # run the tests (220 passed)
```

### C++ (g++ 13, C++17, CMake >= 3.16, GoogleTest)

```bash
cd cpp
bash build.sh                            # cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
./build/demo                             # run the demo
ctest --test-dir build --output-on-failure   # run the tests (130 passed)
```

### Rust (1.95, edition 2021)

```bash
cd rust
cargo build --release                    # zero warnings
cargo run --release --bin demo           # run the demo
cargo test --release                     # run the tests (102 passed)
```

### Java (21, JUnit 4 + Hamcrest — no Maven/Gradle)

```bash
cd java
bash build.sh                            # javac -Xlint:all -Werror everything into out/
bash demo.sh                             # run the demo
bash test.sh                             # JUnitCore on all *Test classes (162 passed)
```

The scripts default to `/usr/share/java/junit4.jar` and
`/usr/share/java/hamcrest.jar` (Debian/Ubuntu `junit4` + `libhamcrest-java`
packages); override with `JUNIT=/path/junit4.jar HAMCREST=/path/hamcrest.jar
bash build.sh`.

## Demo output (Python reference, truncated)

```
================================================================
irm demo — bootstrapping, Vasicek, Hull-White
================================================================

EUR curve (negative short end)  (ACT/365F, cont. comp. zeros, simple 1y-forwards)
      T         DF   zero %  fwd(T,T+1) %
   0.25   1.001126  -0.4503       -0.2127
   0.50   1.002004  -0.4004       -0.1502
   1.00   1.003009  -0.3005       -0.1001
   2.00   1.004014  -0.2003        0.1001
   5.00   0.994995   0.1004        1.1678
  10.00   0.921793   0.8143        2.1027
  30.00   0.627842   1.5516        1.7648

USD curve (normal/inverted)  (ACT/365F, cont. comp. zeros, simple 1y-forwards)
      T         DF   zero %  fwd(T,T+1) %
   0.25   0.986850   5.2948        4.8622
   1.00   0.951475   4.9742        4.4865
   5.00   0.809103   4.2366        4.0785
  10.00   0.660887   4.1417        4.3247
  30.00   0.295184   4.0672        3.8994

OIS discount curve: DF(1y)=0.951928  DF(10y)=0.665329
Sanity: EUR DF(1y) = 1.003009 > 1 (negative rates)

Vasicek calibration to data/zero_yields.csv (r0 = 3%), 3 starts:
  kappa=0.4114  theta=0.0688  sigma=0.0805  rmse=1.70e-05  converged=True  at_bound=False
  identified=False  spreads across equivalent starts: kappa 0.386, theta 0.0188, sigma 0.0580
  with sigma fixed at 0.02 (from option prices):
  kappa=0.8002  theta=0.0500  rmse=1.74e-05  converged=True  identified=True
  ZCB(5y): analytic 0.798973  MC 0.799095 (se 1.6e-04)

Hull-White (a=0.10, sigma=0.01) on USD curve; 5y par = 4.3500%
             cap  strike %  PV (notional 100)
          5y ATM    4.3500           1.504467
    5y ATM+100bp    5.3500           0.460298
    5y ATM-100bp    3.3500           3.548314
  caplet(1y->2y) analytic 0.409407 | MC 0.408453 (se 2.4e-03)

Jamshidian 1y-into-5y payer swaption (USD, strike = fwd par 4.1306%):
  value = 1.324221   r* = 0.045456   |residual| = 0.0e+00
  EUR (negative rates) 1y-into-3y @ 0.1001%: value = 0.869091, r* = -0.001588

Hull-White calibration to USD caplet/swaption prices (a, sigma free):
  a=0.080000  sigma=0.012000  rmse=9.3e-14  converged=True  identified=True  (true: 0.08, 0.012)
  normal (bp) implied vols of the calibrated model:
    caplet 1y->2y   115.8 bp
    caplet 2y->3y   111.1 bp
    caplet 4y->5y   103.3 bp
    1y-into-5y swaption     99.8 bp

done.
```

The Vasicek block is deliberately honest: the bundled noisy yields have two
least-squares minima within 2 % of each other in rmse, so `identified` is
`False` until `sigma` is pinned from option prices.

## Golden values — how cross-language consistency is enforced

`data/golden/golden.json` stores exactly 33 named cases: flat scalar
inputs, one or more expected outputs, and an **absolute** tolerance
(`|got - expect| <= tol`). Every language's test suite loads this one file
and runs one test per case (parameterised in C++/Java/pytest, collect-all
in Rust), so a single bad case never masks another:

* **Curve cases** rebuild a `DiscountCurve` from the `t{i}/df{i}` input
  keys and check DF/zero/forward at and between pillars (tol 1e-12 to
  1e-14 — the interpolation formula is deterministic, so ports must match
  to rounding).
* **Bootstrap cases** re-bootstrap the EUR/USD/OIS curves from the bundled
  CSVs and check pinned DFs, zeros, and that the 10y USD par rate is
  recovered to 1e-10 (a repricing test in disguise).
* **Closed-form cases** (Vasicek ZCB/options/moments, Hull-White
  ZCB/caplet/cap/swaption incl. `r_star` and the decomposition strikes,
  Bachelier price/implied vol) are analytic and pinned to 1e-8 or tighter.
* **Calibration cases** pin the Vasicek fit with `sigma_fixed` (1e-6) and
  the Hull-White recovery of `(a, sigma)` from generated quotes (1e-8),
  which also exercises the Nelder-Mead kernel end-to-end.
* **Monte Carlo cases** store `tol = 6 x` the standard error measured at
  50,000 paths. Any *unbiased* implementation — any RNG, any step count,
  since the transition sampling is exact — passes with overwhelming
  probability; a biased discretization fails. Correctness is statistical,
  not bit-exact.

`data/generate_data.py` validates before writing: all three bootstrapped
curves must reprice every input quote to < 1e-9, both models' closed forms
must agree with their exact-transition MC within 3 SE at 200,000 paths,
Hull-White `P(0,T)` must match the market DFs to 1e-12, the Bachelier vol
must round-trip to 1e-12 and the Hull-White calibration must recover its
generating parameters to 1e-9.

## Real-world usage notes

What a practitioner gets, precisely:

* **Units and conventions.** Times are ACT/365F year fractions on a
  continuous axis (no calendars, business-day rolls or spot lag); rates
  are decimals with continuous compounding for zeros, simple compounding
  for deposits/FRAs/forwards, annual fixed legs with a short first stub;
  volatilities are absolute (normal): `sigma = 0.01` is 100 bp/√year.
  Prices are per unit notional unless a `notional` argument is given.
* **Parameter domains.** `kappa`/`a > 0`, `sigma >= 0`; instrument
  maturities and FRA ends in `[1e-6, 200]` years; option strikes `> 0` for
  ZCB options, `1 + K·tau > 0` for caplets; swaption fixed rates need
  `1 + X·tau_n > 0`. Calibration domains: Vasicek `kappa ∈ (1e-6, 50]`,
  `|theta| ≤ 5`, `sigma ∈ [0, 5]`; Hull-White `a ∈ (1e-6, 5]`,
  `sigma ∈ [0, 1]`.
* **What is validated.** NaN/inf in any public input, non-positive DFs,
  unsorted/duplicate pillars or schedules, empty inputs, out-of-range
  maturities, wrong-length calibration starts, non-representable results
  (exp overflow, DF underflow) — all raise the language's standard error.
  No function returns NaN or inf. Root finders reject non-finite function
  values and never return an unconverged root. Every loop is bounded.
* **What `converged` means.** For both calibrations `converged` is true
  only if the optimiser met its tolerances *and* the solution is not on a
  domain boundary (`at_bound`). `identified` reports whether the
  multi-start runs that fit equally well agree on the parameters; price
  on a calibration only when `converged and (identified or sigma/a fixed)`.
* **Numerical accuracy.** Closed forms agree across the four ports to
  ~1e-14 absolute; OU moments are accurate to ~1e-11 relative for any
  `a·dt`; MC estimators are unbiased at any step count with a reported
  sample standard error. Jamshidian prices are exact for any fixed rate
  with `1 + X·tau_n > 0` (verified against direct integration), and the
  `|g(r*)| < 1e-10` contract rejects inputs whose coupon-bond terms exceed
  double precision (roughly `X < -0.9`).
* **Deliberately out of scope.** Day-count conventions other than
  ACT/365F, holiday calendars, IMM dates and futures convexity, multi-curve
  projection/discounting splits and basis swaps (the OIS bootstrap builds
  the discount curve only), bid/ask, smile/skew (Gaussian models only),
  time-dependent `sigma(t)`/`a(t)`, Bermudan/American exercise, analytic
  Greeks (DV01 is a bump-and-reprice recipe in the COOKBOOK), and
  cross-language bit-identical random streams (MC agreement is
  statistical: seeds fix a stream only within one language, toolchain and
  dependency set — libstdc++ `std::normal_distribution`, Rust `StdRng`
  and `java.util.Random` all differ).
* **Verification status.** 614 automated tests across the four languages,
  33 shared golden cases, closed forms checked against direct numerical
  integration and 200k-path Monte Carlo, generator-validated data. The
  models are the standard production formulas; the conventions above are
  the simplifications you must map to before comparing with a desk system.

## References

* Vasicek, O. (1977). "An Equilibrium Characterization of the Term
  Structure." *Journal of Financial Economics* 5(2), 177–188.
  https://doi.org/10.1016/0304-405X(77)90016-2
* Hull, J. and White, A. (1990). "Pricing Interest-Rate-Derivative
  Securities." *The Review of Financial Studies* 3(4), 573–592.
  https://doi.org/10.1093/rfs/3.4.573
* Jamshidian, F. (1989). "An Exact Bond Option Formula." *The Journal of
  Finance* 44(1), 205–209. https://doi.org/10.1111/j.1540-6261.1989.tb02413.x
* Brigo, D. and Mercurio, F. (2006). *Interest Rate Models — Theory and
  Practice: With Smile, Inflation and Credit*, 2nd ed. Springer.
  https://doi.org/10.1007/978-3-540-34604-3 (§1.1–1.5 curves, §2.6 caplets,
  §3.2–3.3 Vasicek/Hull-White and the Jamshidian decomposition)
* Hagan, P. S. and West, G. (2006). "Interpolation Methods for Curve
  Construction." *Applied Mathematical Finance* 13(2), 89–129.
  https://doi.org/10.1080/13504860500396032
* Andersen, L. B. G. and Piterbarg, V. V. (2010). *Interest Rate Modeling*,
  vols. 1–2. Atlantic Financial Press. (curve construction, Gaussian
  models, vanilla calibration)
* Bachelier, L. (1900). "Théorie de la spéculation." *Annales scientifiques
  de l'École Normale Supérieure* 17, 21–86.
  https://doi.org/10.24033/asens.476 (the normal option model)
* Gillespie, D. T. (1996). "Exact numerical simulation of the
  Ornstein-Uhlenbeck process and its integral." *Physical Review E* 54(2),
  2084–2091. https://doi.org/10.1103/PhysRevE.54.2084
* Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*.
  Springer. https://doi.org/10.1007/978-0-387-21617-1 (§3.3 Gaussian
  short-rate simulation)
* Brent, R. P. (1973). *Algorithms for Minimization without Derivatives*.
  Prentice-Hall, ch. 4 (the `zbrent` root finder).
* Press, W. H., Teukolsky, S. A., Vetterling, W. T. and Flannery, B. P.
  (1992). *Numerical Recipes in C*, 2nd ed. Cambridge University Press,
  §9.3 (`zbrent`).
* Nelder, J. A. and Mead, R. (1965). "A Simplex Method for Function
  Minimization." *The Computer Journal* 7(4), 308–313.
  https://doi.org/10.1093/comjnl/7.4.308
* Lagarias, J. C., Reeds, J. A., Wright, M. H. and Wright, P. E. (1998).
  "Convergence Properties of the Nelder–Mead Simplex Method in Low
  Dimensions." *SIAM Journal on Optimization* 9(1), 112–147.
  https://doi.org/10.1137/S1052623496303470
* Cody, W. J. (1969). "Rational Chebyshev Approximations for the Error
  Function." *Mathematics of Computation* 23(107), 631–637.
  https://doi.org/10.1090/S0025-5718-1969-0247736-4 (the `erfc` used by the
  Rust and Java ports; netlib CALERF)
* Bianchetti, M. and Carlicchi, M. (2011). "Interest Rates After the
  Credit Crunch: Multiple Curve Vanilla Derivatives and SABR."
  SSRN 1783070. https://ssrn.com/abstract=1783070 (multi-curve context in
  LEARN.md)
* Schrimpf, A. and Sushko, V. (2019). "Beyond LIBOR: a primer on the new
  benchmark rates." *BIS Quarterly Review*, March 2019, 29–52.
  https://www.bis.org/publ/qtrpdf/r_qt1903e.htm

## License

MIT — see [LICENSE](LICENSE). Provided without warranty; the "Real-world
usage notes" above list exactly what is and is not covered before you
price anything real with it.
