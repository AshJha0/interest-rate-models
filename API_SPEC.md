# P04 `irm` — API Specification (language-neutral contract)

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
  (Python `ValueError`, C++ `std::invalid_argument`/`std::domain_error`,
  Rust `Result<_, IrmError>`, Java `IllegalArgumentException`) with a
  descriptive message. Calibration **non-convergence is reported** via a
  `converged` flag on the result — never an exception/panic.
* **Floating point:** IEEE-754 double everywhere.

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
  requires `0 <= t1 < t2`.
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

A plain **bisection** fallback with `xtol = 1e-12`, `maxiter = 200` is also
provided (same bracket contract).

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

**Instrument validation:** rates finite; deposit needs `1 + R*T > 0`, FRA
needs `0 <= T1 < T2` and `1 + R*(T2-T1) > 0` (otherwise the quote implies a
non-positive DF -> error at construction).

**Algorithm:** instruments must be supplied with **strictly increasing
pillar times** (duplicates/out-of-order -> error). Process in order; for
each instrument the unknown is `x = DF(pillar)`. Build a trial curve from
all previously solved pillars plus `(pillar, x)` — intermediate cashflow
dates interpolate on this trial curve (this is how missing pillar gaps are
handled) — and solve `residual(x) = 0` with Brent on the bracket
**`x in [1e-10, 100]`, `xtol = 1e-14`**. A bracketing failure means no
admissible positive DF reprices the quote (crossed/arbitrageable quotes,
the "negative implied DF" case) -> error naming the pillar.

**Acceptance:** the finished curve must reprice every input instrument
with `|residual| < 1e-9` (reference implementation achieves < 1e-15).

## 4. Vasicek model

`dr = kappa (theta - r) dt + sigma dW`, parameters validated as
`kappa > 0`, `sigma >= 0`, all finite. With `tau = T - t >= 0`:

```
B(tau) = (1 - exp(-kappa*tau)) / kappa
A(tau) = exp[ (theta - sigma^2/(2 kappa^2)) * (B(tau) - tau)
              - sigma^2 * B(tau)^2 / (4 kappa) ]
P(t,T) = A(tau) * exp(-B(tau) * r(t))          (r defaults to r0 at t=0)
zero yield y(t,T) = -ln P(t,T) / tau           (tau > 0 required)
E[r_T | r_t]   = theta + (r_t - theta) * exp(-kappa*tau)
Var[r_T | r_t] = sigma^2 * (1 - exp(-2 kappa tau)) / (2 kappa)
```

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

so `f(0,0) = -ln DF(h) / h`. Every port must use exactly this scheme —
golden values for `t > 0` prices depend on it. (With piecewise-constant
forwards the FD equals the segment forward inside a segment and the
average of the two adjacent forwards when the stencil straddles a pillar.)

**Affine ZCB using market DFs:**

```
B(t,T) = (1 - exp(-a (T-t))) / a
A(t,T) = (P^M(0,T) / P^M(0,t))
         * exp[ B(t,T) f(0,t) - (sigma^2/(4a)) (1 - exp(-2 a t)) B(t,T)^2 ]
P(t,T) = A(t,T) exp(-B(t,T) r)
```

`r` defaults to `r(0) = f(0,0)`, under which `P(0,T)` equals the market DF
*exactly* (the `exp(B f(0,0))` and `exp(-B r0)` factors cancel — no FD
error enters at `t = 0`).

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
`tau = T_pay - T_reset > 0`, strike `K` with `1 + K tau > 0`, notional `N`
(pays `N tau max(F - K, 0)` at `T_pay`):

```
caplet   = N (1 + K tau) * ZBP(T_reset, T_pay, 1/(1 + K tau))
floorlet = N (1 + K tau) * ZBC(T_reset, T_pay, 1/(1 + K tau))
```

`sigma = 0` degenerates to discounted intrinsic on the curve forward:
`caplet = N DF(T_pay) tau max(F(T_reset,T_pay) - K, 0)`.

**Cap** over schedule `[s_0, s_1, ..., s_m]` (>= 2 strictly increasing
times, `s_0 >= 0`): sum of the caplets on consecutive pairs.

**Jamshidian European swaption**, expiry `T > 0`, fixed rate `X`, payment
times `T < u_1 < ... < u_m`, accruals `tau_i = u_i - u_{i-1}` (`u_0 = T`),
notional `N`:

1. Coupons `c_i = X tau_i` for `i < m`, `c_m = X tau_m + 1`.
2. Solve for `r*` in `g(r) = sum_i c_i P(T, u_i; r) - 1 = 0`. `g` is
   strictly decreasing in `r` (each `B > 0`), so the root is unique.
   Bracket: start `[-1, 1]`; while `g(lo) <= 0` double `lo`, while
   `g(hi) >= 0` double `hi` (max 24 doublings, else error). Solve with
   Brent, `xtol = 1e-15`; require `|g(r*)| < 1e-10`.
3. Decomposition strikes `K_i = P(T, u_i; r*)`.
4. `payer = N * sum_i c_i ZBP(T, u_i, K_i)`;
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

Sample `x' = E[x'] + sd_x Z1`, then
`I = E[I] + (Cov/Var[x'])(x' - E[x']) + sqrt(Var[I] - Cov^2/Var[x']) Z2`
with independent standard normals `Z1, Z2` (two draws per step per path,
in that order). `sigma = 0` short-circuits to the deterministic decay
(standard error exactly 0).

* **Vasicek ZCB:** `price = mean over paths of exp(-(theta*T + sum I))`;
  report the standard error `stdev/sqrt(n_paths)` (sample stdev, ddof 1).
* **Hull-White caplet:** simulate `x` to `T_reset` accumulating `sum I`;
  with `alpha(t) = f(0,t) + (sigma^2/(2a^2))(1 - exp(-a t))^2` and

  ```
  V(t) = (sigma^2/(2a^2)) [ t - 2(1 - exp(-a t))/a + (1 - exp(-2 a t))/(2a) ]
  D(0,T_reset) = P^M(0,T_reset) * exp(-V(T_reset) - sum I)     (exact pathwise DF,
                 using  int_0^t alpha ds = -ln P^M(0,t) + V(t))
  ```

  then `r(T_reset) = x + alpha(T_reset)`, `P(T_reset,T_pay)` analytic from
  section 5, `F = (1/P - 1)/tau`, and the payoff (paid at `T_pay`, known at
  reset) is valued pathwise as
  `D(0,T_reset) * P(T_reset,T_pay) * N tau max(F - K, 0)`.

Because the transitions are exact, MC estimates are unbiased at **any**
step count; golden MC cases therefore only constrain statistical error.

## 7. Optimizer and Vasicek calibration

**Nelder-Mead** (native; P02 API pattern): coefficients reflection 1,
expansion 2, contraction 0.5, shrink 0.5; initial simplex = `x0` plus
`initial_step` (default 0.1) added to each coordinate; converged when
function spread `<= ftol` (1e-14) and simplex diameter `<= xtol` (1e-10);
`maxiter` default 5000. Result: `(x, fx, iterations, converged)`.

**Calibration:** fit `(kappa, theta, sigma)` to `>= 3` market zero yields
`{(T_i, y_i)}` with `r0` fixed, minimizing `sum_i (y_model(T_i) - y_i)^2`;
start `(0.5, mean(y), 0.01)`, `initial_step = 0.05`; out-of-domain points
(`kappa <= 1e-6`, `sigma < 0`, `kappa > 50`, `|theta| > 5`, `sigma > 5`)
get penalty `1e6 * (1 + distance)`. Report `rmse = sqrt(fx / n)` and the
optimizer's `converged` flag (never throw on non-convergence); return
`|sigma|`.

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
{key: value}, "tol"}]}`. Comparison is **absolute**: `|got - expect| <=
tol`. Where inputs contain `t1/df1 ... tn/dfn`, rebuild a
`DiscountCurve` from those pillars (collect `df{i}` for i = 1,2,... until
missing). Where inputs contain `curve_id`, bootstrap that curve from the
CSVs above (`"OIS"` -> `ois_quotes.csv`). Vasicek cases carry
`kappa/theta/sigma/r0`; Hull-White cases carry `a/sigma` plus curve keys.

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
| `hw_zcb_t0` | `P(0,T)` == market DF | 1e-12 |
| `hw_caplet` / `hw_floorlet` | caplet/floorlet(reset, pay, strike, notional) | 1e-9 |
| `hw_cap_3y` | cap over `[t0,t1,t2,t3]` | 1e-9 |
| `hw_caplet_sigma0` | sigma = 0 caplet (discounted intrinsic) | 1e-12 |
| `hw_swaption_payer` | Jamshidian payer, expiry 1, annual payments `expiry+1 ... expiry+tenor_years`, given fixed_rate/notional | 1e-8 |
| `hw_swaption_neg_curve` | same on the negative-rate curve | 1e-8 |
| `hw_mc_caplet` | MC caplet, `>= min_paths` paths, any seed/steps | 6 SE band |

MC bands: `tol = 6 *` the standard error measured at `min_paths` (50000)
paths, so any correct unbiased implementation with its own RNG passes with
overwhelming probability; a biased scheme fails.

Regeneration (`data/generate_data.py`) validates before writing: all three
bootstrapped curves reprice every input to < 1e-9, Vasicek and Hull-White
closed forms agree with their exact-transition MC within 3 SE at 200000
paths, and HW `P(0,T)` matches market DFs to 1e-12.
