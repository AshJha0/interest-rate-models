# Interest Rate Models (Bootstrapping, Vasicek, Hull-White)

Four parallel implementations (Python reference, C++, Rust, Java) of the
core fixed-income modeling stack: discount-curve construction from market
quotes, the Vasicek short-rate model, and a curve-fitted Hull-White
one-factor model — all sharing a golden-value test suite that pins every
language to identical numbers.

* **Discount curve**: pillar times + discount factors, log-linear
  interpolation in ln(DF) (piecewise-constant instantaneous forwards),
  flat-forward extrapolation, continuous zero rates, instantaneous and
  simple forwards, par-swap-rate helper. Negative rates are first-class:
  DF > 1 is legal.
* **Bootstrap**: sequential single-curve bootstrap from deposits, FRAs and
  annual par swaps plus an OIS variant, each pillar solved with a native
  Brent root find; strict pillar ordering, crossed/arbitrageable-quote
  detection (no admissible positive DF → error).
* **Vasicek** `dr = kappa(theta - r)dt + sigma dW`: affine A/B closed-form
  ZCB prices and yields, exact mean/variance of `r_T`, Jamshidian (1989)
  ZCB call/put formula, exact-transition (bias-free) Monte Carlo of
  `(r, ∫r dt)`, least-squares calibration to zero yields via native
  Nelder-Mead (non-convergence reported, never thrown).
* **Hull-White 1F** `dr = (theta(t) - a r)dt + sigma dW` fitted to a market
  curve: `theta(t)` from finite-difference forwards, affine ZCB via market
  DFs (exact curve reproduction at t = 0), ZBC/ZBP closed forms,
  caplets/floorlets as ZCB options, caps as caplet strips, Jamshidian
  swaption decomposition with a root find for `r*`, exact-Gaussian MC
  (trinomial-tree-free).
* **Shared kernels** (implemented natively in every port — no external
  numerics): Brent + bisection root finders, Nelder-Mead simplex, normal
  CDF via `erfc`, exact Ornstein-Uhlenbeck step moments.

Theory: [LEARN.md](LEARN.md). Recipes: [COOKBOOK.md](COOKBOOK.md).
Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Language contract:
[API_SPEC.md](API_SPEC.md).

## Feature matrix

| Feature | Python | C++ | Rust | Java |
|---|---|---|---|---|
| DiscountCurve (ln-DF interp, flat-fwd extrap) | yes | yes | yes | yes |
| Zero / inst-forward / simple forward / par rate | yes | yes | yes | yes |
| Bootstrap: deposits, FRAs, par swaps | yes | yes | yes | yes |
| OIS bootstrap variant | yes | yes | yes | yes |
| Crossed-quote / negative-implied-DF detection | yes | yes | yes | yes |
| Vasicek ZCB, yields, r_T mean/var | yes | yes | yes | yes |
| Vasicek ZBC/ZBP (Jamshidian 1989) | yes | yes | yes | yes |
| Vasicek exact-transition MC + SE | yes | yes | yes | yes |
| Vasicek calibration (native Nelder-Mead) | yes | yes | yes | yes |
| Hull-White ZCB via A/B on market DFs | yes | yes | yes | yes |
| Hull-White theta(t), caplet/floorlet/cap | yes | yes | yes | yes |
| Jamshidian swaption (payer/receiver) | yes | yes | yes | yes |
| Hull-White exact-Gaussian MC caplet | yes | yes | yes | yes |
| Brent + bisection, Nelder-Mead (native) | yes | yes | yes | yes |
| Golden-value suite (~25 cases) | yes | yes | yes | yes |
| Error style | `ValueError` | `std::invalid_argument` / `domain_error` | `Result<_, IrmError>` | `IllegalArgumentException` |

## Directory layout

```
04-interest-rate-models/
  README.md, LEARN.md, COOKBOOK.md, API_SPEC.md
  docs/
    ARCHITECTURE.md          # design + Mermaid diagrams
    diagrams/*.mmd           # raw diagram sources
  data/
    curve_quotes.csv         # EUR (negative short end) + USD deposit/swap quotes
    ois_quotes.csv           # OIS par swap quotes 1M-30Y
    zero_yields.csv          # Vasicek calibration targets (seeded noise)
    generate_data.py         # regenerates all three (validates before writing)
    golden/golden.json       # cross-language reference values
  python/
    src/irm/                 # reference package (curve, bootstrap, vasicek, hullwhite, ...)
    tests/                   # pytest suites incl. golden runner (96 tests)
    demo.py                  # end-to-end demo (output below)
  cpp/                       # C++17, CMake, GoogleTest    (namespace irm)
  rust/                      # edition 2021                (crate irm)
  java/                      # Java 21, JUnit 4            (com.quant.irm)
```

## Build, run, test

### Python (3.11; numpy, pytest; scipy only as a test cross-check)

```bash
cd python
PYTHONPATH=src python3 demo.py        # run the demo
PYTHONPATH=src pytest -q              # run the tests
```

### C++ (g++ 13, C++17, CMake ≥ 3.28, GoogleTest)

```bash
cd cpp
./build.sh                            # cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
./build/demo                          # run the demo
ctest --test-dir build --output-on-failure   # run the tests
```

### Rust (1.95, edition 2021)

```bash
cd rust
cargo build --release
cargo run --release --bin demo        # run the demo
cargo test                            # run the tests
```

### Java (21, JUnit 4 at /usr/share/java/junit4.jar — no Maven/Gradle)

```bash
cd java
./build.sh                            # javac everything into out/
./demo.sh                             # run the demo
./test.sh                             # JUnitCore on all *Test classes
```

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

Vasicek calibration to data/zero_yields.csv (r0 = 3%):
  kappa=0.7977  theta=0.0501  sigma=0.0225  rmse=1.73e-05  iters=302  converged=True
  ZCB(5y): analytic 0.798979  MC 0.799117 (se 1.8e-04)

Hull-White (a=0.10, sigma=0.01) on USD curve; 5y par = 4.3500%
             cap  strike %  PV (notional 100)
          5y ATM    4.3500           1.504467
    5y ATM+100bp    5.3500           0.460298
    5y ATM-100bp    3.3500           3.548314
  caplet(1y->2y) analytic 0.409407 | MC 0.408453 (se 2.4e-03)

Jamshidian 1y-into-5y payer swaption (USD, strike = fwd par 4.1306%):
  value = 1.324221   r* = 0.045456   |residual| = 0.0e+00
  EUR (negative rates) 1y-into-3y @ 0.1001%: value = 0.869091, r* = -0.001588

done.
```

## Golden values — how cross-language consistency is enforced

`data/golden/golden.json` stores ~25 named cases: flat scalar inputs, the
expected outputs, and an **absolute** tolerance (`|got - expect| <= tol`).
Every language's test suite loads this one file and asserts agreement:

* **Curve cases** rebuild a `DiscountCurve` from the `t{i}/df{i}` input
  keys and check DF/zero/forward at and between pillars (tol 1e-12 to
  1e-14 — the interpolation formula is deterministic, so ports must match
  to rounding).
* **Bootstrap cases** re-bootstrap the EUR/USD/OIS curves from the bundled
  CSVs and check pinned DFs, zeros, and that the 10y USD par rate is
  recovered to 1e-10 (a repricing test in disguise).
* **Closed-form cases** (Vasicek ZCB/options/moments, Hull-White
  ZCB/caplet/cap/swaption) are analytic and pinned to 1e-8 or tighter.
* **Monte Carlo cases** store `tol = 6 x` the standard error measured at
  50,000 paths. Any *unbiased* implementation — any RNG, any step count,
  since the transition sampling is exact — passes with overwhelming
  probability; a biased discretization fails. Correctness is statistical,
  not bit-exact.

`data/generate_data.py` validates before writing: all three bootstrapped
curves must reprice every input quote to < 1e-9, both models' closed forms
must agree with their exact-transition MC within 3 SE at 200,000 paths,
and Hull-White `P(0,T)` must match the market DFs to 1e-12.

## License

Educational use. No license restrictions; no warranty — do not price real
trades with this.
