# `irm` — API Specification (language-neutral contract)

This document is the normative contract for every port (Python reference,
C++, Rust, Java). A port implemented from this document alone — with **no
scipy or any external numerical library** — must reproduce every case in
`data/golden/golden.json` within the stored tolerances. All formulas below
are stated exactly as implemented.

## 0. Global conventions

* **Day count:** ACT/365F everywhere; every time is a year fraction
  `days/365` stored as a double. This is a documented simplification: real
  money-market deposits/FRAs use ACT/360 and swap fixed legs 30/360 —
  conversion is a data-preparation issue, not a model issue.
* **Rates:** decimals (`0.05` = 5%). Continuous compounding for zero
  rates, simple compounding for deposit/FRA/forward rates, annual fixed
  legs on swaps.
* **Negative rates are first-class:** discount factors above 1 are legal
  (only `DF > 0` is required); both models are Gaussian precisely so that
  `r < 0` works. No lognormal short-rate model is included for this reason.
* **Errors:** invalid input raises/returns the language's standard error
  (Python `ValueError`, C++ `std::invalid_argument` for bad inputs and
  `std::domain_error` for numerical failures, Rust `Result<_, IrmError>`
  with variants `InvalidInput` / `RootFind` / `Bootstrap` / `Data`, Java
  `IllegalArgumentException` for everything) with a descriptive message.
  NaN/inf anywhere in a public input is rejected at the boundary; no
  function returns NaN/inf silently (a value that would overflow, e.g. a
  ZCB price at an absurd short rate or a forward rate whose discount
  factor underflowed, is an error). Calibration **non-convergence is
  reported** via a `converged` flag on the result — never an
  exception/panic.
* **Loop bounds:** every iteration is bounded (Brent 100, bisection 200,
  Nelder-Mead `maxiter`, bracket expansions 24 / 40, schedule length 200).
* **Floating point:** IEEE-754 double everywhere. `1 - e^{-x}` is always
  evaluated as `-expm1(-x)`.

## 1. Discount curve

**State:** pillar times `t_1 < t_2 < ... < t_n` (all `> 0`, finite,
strictly increasing) and discount factors `P_i > 0` (finite). An implicit
node `(t = 0, DF = 1, ln DF = 0)` is prepended. Violations (empty input,
length mismatch, `t <= 0`, non-monotone/duplicate times, `DF <= 0`,
NaN/inf) are errors.

**Interpolation — log-linear in ln DF** (equivalently piecewise-constant
instantaneous forwards). For `t` in segment `[t_i, t_{i+1}]` (nodes include
the implicit zero node):

```
ln DF(t) = ln P_i + (ln P_{i+1} - ln P_i) * (t - t_i) / (t_{i+1} - t_i)
```

A query exactly at a pillar returns the pillar DF exactly.

**Extrapolation** beyond `t_n` is flat-forward: continue the last segment's
slope indefinitely. Queries at `t < 0` or non-finite `t` are errors.

**Derived quantities:**

* Zero rate (continuous): `z(t) = -ln DF(t) / t`; at `t = 0` return the
  limit, i.e. the first segment's instantaneous forward.
* Instantaneous forward: `f(t) = -d ln DF/dt` = minus the slope of the
  segment containing `t`; **at a pillar, the right-hand segment is used**;
  at/after `t_n`, the last segment.
* Simple forward: `F(t1,t2) = (DF(t1)/DF(t2) - 1) / (t2 - t1)`,
  requires `0 <= t1 < t2`; a result that is not representable (`DF(t2)`
  underflowed to 0 or the ratio overflowed) is an error, never `inf`.
* Par swap rate helper for times `[t_0, t_1, ..., t_m]` (start `t_0 >= 0`,
  strictly increasing), accruals `tau_i = t_i - t_{i-1}`:
  `par = (DF(t_0) - DF(t_m)) / sum_i tau_i DF(t_i)`.

## 2. Root finding (implemented natively)

**Brent's method** (Numerical-Recipes `zbrent` variant: bisection + secant
+ inverse quadratic interpolation). Signature-level contract:

* Inputs: `f, a, b, xtol = 1e-14, rtol = 4.44e-16, maxiter = 100`.
* If `f(a) == 0` return `a`; if `f(b) == 0` return `b`; if
  `sign(f(a)) == sign(f(b))` -> error "root not bracketed".
* Per-iteration convergence test: `|c - b|/2 <= 2*rtol*|b| + xtol/2`.
* Non-convergence after `maxiter` iterations is an error.
* A non-finite function value at any evaluation is an error ("function
  value not finite"); an unconverged or NaN root is never returned.

A plain **bisection** fallback with `xtol = 1e-12`, `maxiter = 200` is also
provided with the same contract (bracket required, `xtol > 0`, non-finite
values and exhausting `maxiter` are errors).

## 3. Bootstrap (single-curve framework)

Floating legs are projected off the same curve used for discounting, so a
swap floating leg PV telescopes to `DF(0) - DF(T)`. (Documented
simplification of the multi-curve/OIS-discounting reality; the OIS variant
below builds the discount curve one would use in such a setup.)

**Instruments and residual functions** (residual = PV mispricing, solved to
zero; `tau` = accrual):

| Instrument | Pillar | Residual |
|---|---|---|
| Deposit `(T, R)` (simple interest) | `T` | `DF(T)*(1 + R*T) - 1` |
| FRA `(T1, T2, R)` | `T2` | `DF(T1) - (1 + R*(T2-T1))*DF(T2)` |
| Par swap `(T, R)` (annual fixed) | `T` | `R * sum_i tau_i DF(s_i) - (1 - DF(T))` |
| OIS swap `(T, R)` | `T` | identical to par swap (see below) |

**Annual fixed schedule** for maturity `T`: `n = ceil(T - 1e-12)` payments
at `s_k = T - (n-1-k)`, `k = 0..n-1` (so `T=2 -> {1,2}`, `T=2.5 ->
{0.5,1.5,2.5}` with a short first stub, `T=0.25 -> {0.25}`); accruals
`tau_i = s_i - s_{i-1}` with `s_{-1} = 0`.

**OIS approximation:** daily compounding of the overnight rate is treated
as continuous compounding at the curve's short rate, under which each
compounded floating coupon pays `DF(s_{i-1})/DF(s_i) - 1` at `s_i` and the
leg PV telescopes to `1 - DF(T)`. The daily-vs-continuous gap is
`O(r^2/365)` — documented approximation.

**Instrument validation:** rates finite; every maturity / FRA end within
`[1e-6, 200]` years; deposit needs `1 + R*T > 0`, FRA needs `0 <= T1 < T2`
and `1 + R*(T2-T1) > 0` (otherwise the quote implies a non-positive DF ->
error at construction).

**Algorithm:** instruments must be supplied with **strictly increasing
pillar times** (duplicates/out-of-order -> error). Process in order; for
each instrument the unknown is `x = DF(pillar)`. Build a trial curve from
all previously solved pillars plus `(pillar, x)` — intermediate cashflow
dates interpolate on this trial curve (this is how missing pillar gaps are
handled) — and solve `residual(x) = 0` with Brent on the bracket
**`x in [1e-10, 100]`, `xtol = 1e-14`** (the per-pillar kernel is exposed
as `solve_pillar_df(residual, pillar, label)`). Error classification:

* a *bracketing* failure (no sign change on `[1e-10, 100]`) means no
  admissible positive DF reprices the quote (crossed/arbitrageable quotes,
  the "negative implied DF" case) -> error naming the pillar and saying
  "crossed/arbitrageable quotes";
* any other solver failure (non-finite residual, iteration budget) ->
  error naming the pillar and saying "solver failed", carrying the
  solver's message. It is never relabelled as crossed quotes.

**Acceptance:** the finished curve must reprice every input instrument
with `|residual| < 1e-9` (reference implementation achieves < 1e-15).

## 4. Vasicek model

`dr = kappa (theta - r) dt + sigma dW`, parameters validated as
`kappa > 0`, `sigma >= 0`, all finite. With `tau = T - t >= 0`:

```
B(tau) = (1 - exp(-kappa*tau)) / kappa          (= -expm1(-kappa tau)/kappa)
A(tau) = exp[ (theta - sigma^2/(2 kappa^2)) * (B(tau) - tau)
              - sigma^2 * B(tau)^2 / (4 kappa) ]
P(t,T) = A(tau) * exp(-B(tau) * r(t))          (r defaults to r0 at t=0)
zero yield y(t,T) = -ln P(t,T) / tau           (tau > 0 required)
E[r_T | r_t]   = theta + (r_t - theta) * exp(-kappa*tau)
Var[r_T | r_t] = sigma^2 * (1 - exp(-2 kappa tau)) / (2 kappa)
```

`P(t,T)` requires finite `t <= T` and a finite `r`; if `-B r > 700` the
price is not representable in double precision -> error (never `inf`).
Model parameters are immutable after construction in every port.

**European ZCB options** (Jamshidian 1989 closed form), expiry `T`, bond
maturity `S >= T`, strike `K > 0` (unit notional):

```
sigma_p = sigma * sqrt((1 - exp(-2 kappa T)) / (2 kappa)) * B(S - T)
h       = ln(P(0,S) / (K P(0,T))) / sigma_p + sigma_p / 2
ZBC     = P(0,S) N(h) - K P(0,T) N(h - sigma_p)
ZBP     = K P(0,T) N(sigma_p - h) - P(0,S) N(-h)
```

If `sigma_p == 0` (sigma = 0, T = 0, or S = T) the option collapses to
intrinsic on the forward bond: `ZBC = max(P(0,S) - K P(0,T), 0)`, `ZBP`
symmetric. `N(.)` is the standard normal CDF, computed as
`0.5 * erfc(-x / sqrt(2))` (implement `erfc` natively if the language's
standard library lacks it; absolute accuracy 1e-12 suffices).

## 5. Hull-White 1F model

`dr = (theta(t) - a r) dt + sigma dW`, fitted to a market curve;
`a > 0`, `sigma >= 0`.

**Market instantaneous forward — exact finite-difference contract.** With
step `h = 1e-5` (constant `FD_STEP`), on the market curve DFs:

```
t >= h :  f(0,t) = (ln DF(t-h) - ln DF(t+h)) / (2h)     (central)
t <  h :  f(0,t) = (ln DF(t)   - ln DF(t+h)) / h        (one-sided)
```

so `f(0,0) = -ln DF(h) / h`. Every port must use exactly this scheme. Note
which outputs depend on it: every pinned *price* (ZBC/ZBP, caplets, caps,
swaption values) is stencil-independent — they use market DFs only, and
in the swaption the strikes `K_i = P(T,u_i;r*)` are invariant to a shift
of `f^M(0,T)` because `r*` shifts by the same amount. What *does* depend
on the stencil is `P(t > 0, T; r)` with an explicit `r`, the reported
`r_star`, the decomposition strikes and `theta(t)`; the golden cases
`hw_zcb_t1_explicit_r` and `hw_swaption_payer` (`r_star`, `strike_i`)
pin those. (With piecewise-constant forwards the FD equals the segment
forward inside a segment and the average of the two adjacent forwards
when the stencil straddles a pillar.)

**Affine ZCB using market DFs:**

```
B(t,T) = (1 - exp(-a (T-t))) / a
A(t,T) = (P^M(0,T) / P^M(0,t))
         * exp[ B(t,T) f(0,t) - (sigma^2/(4a)) (1 - exp(-2 a t)) B(t,T)^2 ]
P(t,T) = A(t,T) exp(-B(t,T) r)
```

`r` defaults to `r(0) = f(0,0)`, under which `P(0,T)` equals the market DF
*exactly* (the `exp(B f(0,0))` and `exp(-B r0)` factors cancel — no FD
error enters at `t = 0`). `B(t,T)` requires finite `0 <= t <= T`; as for
Vasicek, `-B r > 700` is an error (not representable).

**theta(t) fitting** (reference/inspection function; pricing absorbs it
into `A` via the market DFs):

```
theta(t) = df(0,t)/dt + a f(0,t) + (sigma^2/(2a)) (1 - exp(-2 a t))
```

with `df/dt` approximated by the forward difference
`(f(0,t+h') - f(0,t))/h'`, `h' = 1e-4`.

**ZBC/ZBP:** identical formulas to section 4 with `a` in place of `kappa`
and market DFs in place of model DFs:
`sigma_p = sigma sqrt((1 - exp(-2 a T))/(2a)) B(T,S)`, same `h`, same
`sigma_p == 0` intrinsic degeneration, strike must be `> 0`.

**Caplet / floorlet** on the simple forward over `[T_reset, T_pay]`,
`tau = T_pay - T_reset > 0`, `T_reset >= 0`, strike `K` with `1 + K tau > 0`,
notional `N` (pays `N tau max(F - K, 0)` at `T_pay`). `T_reset`, `T_pay`,
`K` and `N` must be finite; `N` may be negative (a short position):

```
caplet   = N (1 + K tau) * ZBP(T_reset, T_pay, 1/(1 + K tau))
floorlet = N (1 + K tau) * ZBC(T_reset, T_pay, 1/(1 + K tau))
```

`sigma = 0` degenerates to discounted intrinsic on the curve forward:
`caplet = N DF(T_pay) tau max(F(T_reset,T_pay) - K, 0)`.

**Cap** over schedule `[s_0, s_1, ..., s_m]` (>= 2 finite, strictly
increasing times, `s_0 >= 0`; strike and notional finite): sum of the
caplets on consecutive pairs.

**Jamshidian European swaption**, expiry `T > 0`, fixed rate `X`, payment
times `T < u_1 < ... < u_m`, accruals `tau_i = u_i - u_{i-1}` (`u_0 = T`),
notional `N`:

1. Coupons `c_i = X tau_i` for `i < m`, `c_m = X tau_m + 1`. If
   `c_m <= 0` (i.e. `1 + X tau_m <= 0`) every coupon is non-positive,
   `g < 1` everywhere and no `r*` exists -> error ("degenerate coupon
   bond"). `X`, `N`, `T` and all `u_i` must be finite.
2. Solve for `r*` in `g(r) = sum_i c_i P(T, u_i; r) - 1 = 0`. `g - 1`
   changes sign exactly once: for `X >= 0` every coupon is positive and
   `g` is strictly decreasing (each `B > 0`); for `X < 0` with `c_m > 0`
   the coupon signs are `(-, ..., -, +)`, so `g' = -sum c_i B_i P_i` is an
   exponential sum with one sign change (at most one zero), `g` has a
   single minimum, `g -> +inf` as `r -> -inf` and `g -> 0^-` as
   `r -> +inf`, hence a unique crossing of 1 on the decreasing branch —
   the decomposition below stays exact. Bracket: start `[-1, 1]`; while
   `g(lo) <= 0` double `lo`, while `g(hi) >= 0` double `hi` (max 24
   doublings; `lo` is never pushed past `B(T,u_m)|lo| > 700`, the exp()
   overflow point; else error). Solve with Brent, `xtol = 1e-15`.
3. **Contract** (enforced identically in every port, via the public
   `jamshidian_at_r_star(T, u, X, r*, N, payer)`): require
   `|g(r*)| < 1e-10` and `g'(r*) = -sum_i c_i B(T,u_i) K_i < 0`; otherwise
   error (Python `ValueError`, C++ `std::domain_error`, Rust
   `IrmError::RootFind`, Java `IllegalArgumentException`). Extreme
   negative fixed rates (about `X < -0.9` on the golden curve) make the
   coupon-bond terms O(1e8+) so `|g(r*)|` cannot reach 1e-10 in double
   precision — such inputs are rejected by the contract rather than
   priced.
4. Decomposition strikes `K_i = P(T, u_i; r*)`.
5. `payer = N * sum_i c_i ZBP(T, u_i, K_i)`;
   `receiver = N * sum_i c_i ZBC(T, u_i, K_i)`.

Result carries `value`, `r_star`, the root residual, and the strikes.

## 6. Monte Carlo schemes (exact-transition, bias-free)

Both models reduce to a mean-zero OU process `dx = -a x dt + sigma dW`
(Vasicek: `x = r - theta`, `a = kappa`; Hull-White: `r(t) = x(t) +
alpha(t)`). Per uniform step `dt`, conditional on `x`, the pair
`(x', I = int x ds)` is bivariate Gaussian with **exact** moments
(`decay = exp(-a dt)`, `b = (1 - decay)/a`):

```
E[x']    = x * decay          Var[x'] = sigma^2 (1 - decay^2) / (2a)
E[I]     = x * b              Var[I]  = (sigma^2/a^2)(dt - 2b + (1 - decay^2)/(2a))
Cov[x',I]= (sigma^2/(2a^2)) (1 - decay)^2
```

**Numerical contract.** `1 - decay = -expm1(-a dt)`,
`1 - decay^2 = -expm1(-2 a dt)`. `Var[I]` cancels three O(dt) terms to an
O(a^2 dt^3) result and is evaluated by the shared kernel
`ou_integral_variance(a, sigma, dt)`: with `x = a dt`,

```
x <  1e-2 :  sigma^2 dt^3 (1/3 - x/4 + 7x^2/60 - x^3/24 + 31x^4/2520 - x^5/320)
x >= 1e-2 :  (sigma^2/a^2) (dt + 2 expm1(-x)/a + (-expm1(-2x))/(2a)), floored at 0
```

(relative error < 3e-15 below the threshold, < ~1e-11 above it; the plain
formula is wrong by orders of magnitude for `x < 1e-3`). `a`, `dt` must
be positive and finite, `sigma >= 0` finite.

Sample `x' = E[x'] + sd_x Z1`, then
`I = E[I] + (Cov/Var[x'])(x' - E[x']) + sqrt(Var[I] - Cov^2/Var[x']) Z2`
with independent standard normals `Z1, Z2` (two draws per step per path,
in that order). `sigma = 0` short-circuits to the deterministic decay
(standard error exactly 0).

* **Vasicek ZCB:** `price = mean over paths of exp(-(theta*T + sum I))`;
  report the standard error `stdev/sqrt(n_paths)` (sample stdev, ddof 1);
  for `sigma = 0` or `n_paths = 1` the standard error is exactly 0.
* **Hull-White caplet:** simulate `x` to `T_reset` accumulating `sum I`;
  with `alpha(t) = f(0,t) + (sigma^2/(2a^2))(1 - exp(-a t))^2` and

  ```
  V(t) = (sigma^2/(2a^2)) [ t - 2(1 - exp(-a t))/a + (1 - exp(-2 a t))/(2a) ]
       = ou_integral_variance(a, sigma, t) / 2        (V(0) = 0; same kernel)
  D(0,T_reset) = P^M(0,T_reset) * exp(-V(T_reset) - sum I)     (exact pathwise DF,
                 using  int_0^t alpha ds = -ln P^M(0,t) + V(t))
  ```

  then `r(T_reset) = x + alpha(T_reset)`, `P(T_reset,T_pay)` analytic from
  section 5, `F = (1/P - 1)/tau`, and the payoff (paid at `T_pay`, known at
  reset) is valued pathwise as
  `D(0,T_reset) * P(T_reset,T_pay) * N tau max(F - K, 0)`. Inputs are
  validated exactly as for the analytic caplet (finite `K`, `N`,
  `1 + K tau > 0`) plus `T_reset > 0`.

Because the transitions are exact, MC estimates are unbiased at **any**
step count; golden MC cases therefore only constrain statistical error.

## 7. Optimizer and calibrations

**Nelder-Mead** (native): coefficients reflection 1, expansion 2,
contraction 0.5, shrink 0.5; initial simplex = `x0` plus `initial_step`
(default 0.1) added to each coordinate; converged when function spread
`<= ftol` (1e-14) and simplex diameter `<= xtol` (1e-10); `maxiter`
default 5000. Result: `(x, fx, iterations, converged)`.

### 7.1 Vasicek calibration to zero yields

Fit `(kappa, theta, sigma)` to `>= 3` market zero yields `{(T_i, y_i)}`
with `r0` fixed, minimizing `sum_i (y_model(T_i) - y_i)^2`;
`initial_step = 0.05`, `maxiter` default 4000 (`>= 1`). Domain
`kappa in (1e-6, 50]`, `|theta| <= 5`, `sigma in [0, 5]`; out-of-domain
points get the penalty `1e6 * (1 + distance)`. Return `|sigma|`.

* **Starts.** `x0 = None`: three starts `(kappa, mean(y), sigma)` with
  `(kappa, sigma) in {(0.2, 0.005), (0.5, 0.01), (1.5, 0.02)}`, the lowest
  objective wins. `x0 = (kappa, theta, sigma)` (exactly 3 finite entries,
  else error): a single warm start.
* **`sigma_fixed`** (optional, finite `>= 0`): optimise `(kappa, theta)`
  only with `sigma` held; `x0[2]` is ignored.
* **Diagnostics** (all ports return the same fields):
  `rmse = sqrt(fx_best / n)`; `iterations` of the best start;
  `at_bound` = `kappa < 1e-5` or `kappa > 49.95` or `|theta| > 4.995` or
  (sigma free and `sigma < 1e-5` or `sigma > 4.995`);
  `converged` = optimizer converged at the best start **and** not
  `at_bound` (a penalty-boundary solution is never reported as converged);
  `n_starts`; `kappa_spread`, `theta_spread`, `sigma_spread` = max − min of
  each parameter over the *equivalent* starts, i.e. those with
  `rmse_start <= 1.05 rmse_best + 1e-12` (0 for a single start);
  `identified` = `n_starts >= 2` and `kappa_spread <= 1e-2`,
  `theta_spread <= 1e-3`, `sigma_spread <= 1e-3`; `sigma_fixed` flag.
  A single warm start reports `identified = false` (not assessed).
  Non-convergence is never an exception.

Yields identify Vasicek parameters weakly (they see `sigma` only through
convexity): on the bundled `zero_yields.csv` two minima,
`(0.798, 0.0501, 0.0225)` and `(0.411, 0.0688, 0.0805)`, fit within 2 % of
each other in rmse, so the default multi-start reports `identified =
false`. With `sigma_fixed = 0.02` all starts agree on
`(0.8002161, 0.0499880)` (golden `vas_calib_sigma_fixed`).

### 7.2 Hull-White calibration to caplet / swaption prices

`calibrate_hullwhite(curve, caplets, swaptions, a_fixed = None,
x0 = None, maxiter = 4000)`. Quotes are per unit notional:
`CapletQuote(reset, pay, strike, price)` (finite, `0 <= reset < pay`,
`1 + K tau > 0`, `price >= 0`) and *payer* `SwaptionQuote(expiry,
pay_times, fixed_rate, price)` (finite, `expiry > 0`, non-empty strictly
increasing schedule after expiry, `price >= 0`). At least one quote.
Objective `sum_j (model_j - price_j)^2` with the section-5 closed forms.
Nelder-Mead runs on `(a, 10 sigma)` with `initial_step = 0.05` (0.005 in
`sigma`); domain `a in (1e-6, 5]`, `sigma in [0, 1]`, same penalty form.

* Starts: `x0 = None` -> `(a, sigma) in {(0.03, 0.005), (0.1, 0.01),
  (0.5, 0.02)}`; `x0 = (a, sigma)` (2 finite entries) -> single start;
  `a_fixed` (finite `> 0`) -> 1-D in `sigma` starting at the three default
  sigmas (`x0[0]` ignored).
* Result `(a, sigma, rmse, iterations, converged, at_bound, n_starts,
  a_spread, sigma_spread, identified, a_fixed)`: `rmse` in price units;
  `at_bound` = (`a` free and `a < 1e-5` or `a > 4.995`) or `sigma < 1e-5`
  or `sigma > 0.999`; `converged` = optimizer converged and not at bound;
  equivalence uses `rmse_start <= 1.05 rmse_best + 1e-14`; `identified` =
  `n_starts >= 2` and `a_spread <= 1e-2`, `sigma_spread <= 1e-4`.
  A single quote with both parameters free is a ridge of solutions and
  reports `identified = false`.

Golden `hw_calib_recover` regenerates quotes from `(a, sigma) = (0.08,
0.012)` on the golden curve (3 caplets + 1 swaption) and requires the fit
to return them within 1e-8.

### 7.3 Bachelier (normal) model

For forward `F`, strike `K`, expiry `T > 0`, annuity `A > 0` (all finite)
and normal vol `sigma_N >= 0` (`d = (F - K)/(sigma_N sqrt(T))`):

```
payer    = A [ (F - K) N(d) + sigma_N sqrt(T) phi(d) ]
receiver = A [ (K - F) N(-d) + sigma_N sqrt(T) phi(d) ]
sigma_N = 0  ->  A max(+-(F - K), 0)
```

`bachelier_implied_vol(price, F, K, T, A, payer)`: `price` finite and
`>= intrinsic - 1e-15` (else error); `price <= intrinsic + 1e-15` returns
0; otherwise Brent (`xtol = 1e-14`) on `[0, hi]`, `hi` found by doubling
from 0.01 at most 40 times (else error "too large"). Negative forwards
and strikes are legal. Goldens: `bach_price_payer`, `bach_implied_vol_payer`,
`bach_price_atm_negative`.

## 8. Bundled data (normative — ports reproduce goldens from these tables)

`data/curve_quotes.csv` (`curve_id,instrument,start,maturity,rate`;
deposits and swaps have `start = 0`; O/N maturity is `1/365`, 1M is
`1/12`, then year fractions as shown):

| curve | deposits (T: rate) | par swaps (T: rate) |
|---|---|---|
| EUR | 1/365: -0.0050, 1/12: -0.0048, 0.25: -0.0045, 0.5: -0.0040, 1: -0.0030 | 2: -0.0020, 3: -0.0010, 4: 0.0000, 5: 0.0010, 7: 0.0040, 10: 0.0080, 15: 0.0120, 20: 0.0140, 30: 0.0150 |
| USD | 1/365: 0.0530, 1/12: 0.0531, 0.25: 0.0533, 0.5: 0.0528, 1: 0.0510 | 2: 0.0480, 3: 0.0455, 4: 0.0442, 5: 0.0435, 7: 0.0428, 10: 0.0425, 15: 0.0427, 20: 0.0428, 30: 0.0420 |

`data/ois_quotes.csv` (`maturity,rate`, all OIS par swaps): 1/12: 0.0528,
0.25: 0.0530, 0.5: 0.0524, 1: 0.0505, 2: 0.0473, 3: 0.0448, 5: 0.0428,
7: 0.0421, 10: 0.0418, 15: 0.0420, 20: 0.0421, 30: 0.0413.

`data/zero_yields.csv`: yields of the "true" Vasicek `(kappa=0.8,
theta=0.05, sigma=0.02, r0=0.03)` at maturities 0.5, 1, 2, 3, 4, 5, 7, 10,
15, 20 plus N(0, 2e-5) noise, numpy `default_rng(42)` (ports read the CSV;
they do not regenerate it).

## 9. Golden cases (`data/golden/golden.json`)

Schema: `{"cases": [{"name", "inputs": {flat scalar keys}, "expect":
{key: value, ...}, "tol"}]}` — exactly **33 cases** with unique names
(the runners assert the count). Comparison is **absolute** on every
`expect` key: `|got - expect| <= tol`. Where inputs contain
`t1/df1 ... tn/dfn`, rebuild a `DiscountCurve` from those pillars (collect
`df{i}` for i = 1,2,... until missing). Where inputs contain `curve_id`,
bootstrap that curve from the CSVs above (`"OIS"` -> `ois_quotes.csv`).
Vasicek cases carry `kappa/theta/sigma/r0`; Hull-White cases carry
`a/sigma` plus curve keys; cap schedules use `s0..s3` (never the pillar
keys `t{i}`). Runners must report every failing case (parameterised /
collect-all), not stop at the first.

| case | compute | tol |
|---|---|---|
| `curve_df_pillar` | `DF(t)` at a pillar | 1e-14 |
| `curve_df_mid` | `DF(t)` between pillars | 1e-12 |
| `curve_df_extrap` | `DF(t)` beyond last pillar | 1e-12 |
| `curve_zero_mid` | `zero(t)` | 1e-12 |
| `curve_fwd_simple` | `F(ta, tb)` | 1e-12 |
| `boot_eur_df_1y` | EUR `DF(1)` (must be > 1) | 1e-8 |
| `boot_eur_df_10y` | EUR `DF(10)` | 1e-8 |
| `boot_eur_zero_30y` | EUR `zero(30)` | 1e-8 |
| `boot_usd_df_10y` | USD `DF(10)` | 1e-8 |
| `boot_usd_par_10y` | USD par rate, times `[0,1,...,10]` | 1e-10 |
| `boot_ois_df_10y` | OIS `DF(10)` | 1e-8 |
| `vas_zcb_5y`, `vas_zcb_neg_r0` | `P(0,T)` | 1e-10 |
| `vas_rT_mean` / `vas_rT_var` | `E[r_T]` / `Var[r_T]` | 1e-12 / 1e-14 |
| `vas_zbc` / `vas_zbp` | ZBC/ZBP(expiry, bond_maturity, strike) | 1e-10 |
| `vas_mc_zcb` | MC ZCB, `>= min_paths` paths, any seed/steps | 6 SE band |
| `vas_calib_sigma_fixed` | `calibrate_vasicek(zero_yields.csv, r0, sigma_fixed)` -> `kappa`, `theta` (must be converged, identified, not at bound) | 1e-6 |
| `hw_zcb_t0` | `P(0,T)` == market DF | 1e-12 |
| `hw_zcb_t1_explicit_r` | `P(t, T; r)` with explicit `t > 0`, `r` (pins the FD stencil) | 1e-10 |
| `hw_caplet` / `hw_floorlet` | caplet/floorlet(reset, pay, strike, notional) | 1e-9 |
| `hw_cap_3y` | cap over `[s0,s1,s2,s3]` | 1e-9 |
| `hw_caplet_sigma0` | sigma = 0 caplet (discounted intrinsic) | 1e-12 |
| `hw_swaption_payer` | Jamshidian payer, expiry 1, annual payments `expiry+1 ... expiry+tenor_years`, given fixed_rate/notional -> `price`, `r_star`, `strike_1..5` | 1e-9 |
| `hw_swaption_receiver` | receiver on the same swap | 1e-9 |
| `hw_swaption_neg_curve` | payer on the negative-rate curve -> `price`, `r_star` | 1e-9 |
| `hw_mc_caplet` | MC caplet, `>= min_paths` paths, any seed/steps | 6 SE band |
| `hw_calib_recover` | `calibrate_hullwhite` on `cap_reset_i/cap_pay_i/cap_price_i` (strike `strike`) + one payer swaption `swpt_*` -> `a`, `sigma` | 1e-8 |
| `bach_price_payer` / `bach_price_atm_negative` | `bachelier_price(forward, strike, expiry, vol, annuity, payer)` | 1e-14 |
| `bach_implied_vol_payer` | `bachelier_implied_vol(price, forward, strike, expiry, annuity, payer)` -> `vol` | 1e-12 |

MC bands: `tol = 6 *` the standard error measured at `min_paths` (50000)
paths, so any correct unbiased implementation with its own RNG passes with
overwhelming probability; a biased scheme fails.

Regeneration (`data/generate_data.py`) validates before writing: all three
bootstrapped curves reprice every input to < 1e-9, Vasicek and Hull-White
closed forms agree with their exact-transition MC within 3 SE at 200000
paths, HW `P(0,T)` matches market DFs to 1e-12, the Bachelier implied vol
round-trips to 1e-12, and the Hull-White calibration recovers the
generating `(a, sigma)` to 1e-9. CSVs are written with LF line endings so
regeneration is byte-stable.
