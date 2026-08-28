# LEARN — Interest Rate Modeling: Curves, Vasicek, Hull-White

This document teaches the theory behind the `irm` package: what a discount
curve is and why interpolation choices matter, how bootstrapping turns
market quotes into a curve, why short-rate models mean-revert, the Vasicek
and Hull-White models with full derivations, curve risk (DV01), and where
this project's documented simplifications differ from desk practice. All
worked numbers reproduce entries in `data/golden/golden.json`, so you can
check every claim against running code in any of the four languages.

Notation follows `API_SPEC.md`: times are ACT/365F year fractions, rates
are decimals, `DF(t)` or `P(0,t)` is the discount factor, `P(t,T)` a
zero-coupon bond (ZCB) price seen from `t`.

---

## 1. The discount curve is the whole game

### 1.1 Intuition

Every linear interest-rate product — deposits, FRAs, swaps, bonds — is a
bundle of fixed or floating cashflows. If you know today's price of one
unit of currency delivered at every future date `t`, you can price the
bundle by summation. That price schedule is the **discount curve**
`t -> DF(t)`, and constructing it well is the foundational skill of any
rates desk. Options come later; the curve comes first.

Three equivalent descriptions of the same object:

$$
DF(t) = e^{-z(t)\,t}
\qquad
z(t) = -\frac{\ln DF(t)}{t}
\qquad
f(t) = -\frac{\partial \ln DF(t)}{\partial t}
$$

* `DF(t)` — discount factor: price today of 1 paid at `t`.
* `z(t)` — continuously compounded **zero rate**: the average rate to `t`.
* `f(t)` — **instantaneous forward**: the marginal rate at `t`. The zero
  rate is the running average of the forwards:
  `z(t) = (1/t) \int_0^t f(s) ds`.

The simply compounded forward over `[t_1, t_2]` — the rate a FRA locks
in — follows from a no-arbitrage replication (borrow at `t_1`, lend to
`t_2`):

$$
F(t_1, t_2) = \frac{1}{t_2 - t_1}\left(\frac{DF(t_1)}{DF(t_2)} - 1\right)
$$

**Negative rates are not a pathology.** EUR and JPY traded negative for
years; a negative zero rate simply means `DF(t) > 1` — you pay more than 1
today for 1 delivered later. The only hard constraint is `DF(t) > 0`
(a non-positive DF would let you buy future money for nothing). Everything
in this project — curve, bootstrap, both models — treats `r < 0` and
`DF > 1` as first-class. This is also why both short-rate models here are
Gaussian: a lognormal model (Black-Karasinski, BDT) structurally cannot
produce negative rates, which is exactly why the industry moved to
Gaussian and shifted-lognormal models after 2012.

### 1.2 Interpolation: the choice that hides in plain sight

Markets quote a few dozen instruments; the curve must answer queries at
*any* `t`. The interpolation scheme is therefore part of the model, and
different choices produce visibly different forward curves from identical
quotes. The classic survey is Hagan & West (2006). The main candidates:

| Scheme | Forwards look like | Pros | Cons |
|---|---|---|---|
| Linear in `z(t)` | piecewise hyperbolic, kinked | simple | forward curve has spikes at pillars; zigzags |
| **Linear in `ln DF(t)`** (this project) | **piecewise constant** | local, arb-free if DFs decrease, fast, stable deltas | forwards jump at pillars (step function) |
| Cubic spline in `z` or `ln DF` | smooth | pretty forwards | non-local: bumping one quote wiggles the whole curve; can imply negative forwards |
| Monotone convex (Hagan-West) | continuous, positive where inputs allow | designed for curves | more complex; still not C² |

Log-linear interpolation in `ln DF` is the workhorse because it is
**exactly the piecewise-constant instantaneous-forward interpolation**: on
a segment `[t_i, t_{i+1}]`,

$$
\ln DF(t) = \ln P_i + \frac{\ln P_{i+1} - \ln P_i}{t_{i+1} - t_i}(t - t_i)
\quad\Longleftrightarrow\quad
f(t) = f_i \equiv -\frac{\ln P_{i+1} - \ln P_i}{t_{i+1} - t_i}
$$

It is *local* (a quote bump only moves the curve between adjacent
pillars — see the DV01 section for why traders care), never produces
oscillation, and its one honest defect — a discontinuous forward curve —
is plainly visible rather than smoothed over. An implicit node
`(t=0, DF=1)` anchors the short end, and extrapolation beyond the last
pillar simply continues the last forward flat.

### 1.3 Worked example (golden: `curve_df_mid`, `curve_zero_mid`, `curve_fwd_simple`, `curve_df_extrap`)

Pillars `t = [1, 2, 3, 5]`, `DF = [0.99, 0.975, 0.955, 0.9]`. The four
segment forwards (including the implicit zero node) are

```
f0 on [0,1] = -ln(0.99)/1          = 0.0100503359
f1 on [1,2] = ln(0.99/0.975)/1     = 0.0152674721
f2 on [2,3] = ln(0.975/0.955)/1    = 0.0207261305
f3 on [3,5] = ln(0.955/0.900)/2    = 0.0296582886
```

* `DF(2.5)`: interpolate `ln DF` between `ln 0.975 = -0.02531781` and
  `ln 0.955 = -0.04604394` at the midpoint: `ln DF(2.5) = -0.03568087`,
  so `DF(2.5) = 0.9649481851374196`. Equivalently
  `0.975 · e^{-0.0207261 × 0.5}`.
* `z(2.5) = -\ln DF(2.5)/2.5 = 0.014272349297139348` — the
  forward-weighted average `(1·f0 + 1·f1 + 0.5·f2)/2.5`.
* `F(1,3) = (0.99/0.955 - 1)/2 = 0.01832460732984298` — note this is the
  *simple* forward; the continuous forward over the same span is
  `(f1 + f2)/2 = 0.0179968`, and the difference is exactly compounding.
* `DF(7) = 0.955 · e^{-f_3 · 4} = 0.8481675392670158` — flat-forward
  extrapolation carries `f3` past the last pillar.

Interpolation is exact at pillars by construction (`curve_df_pillar`, tol
1e-14): queries at a node return the stored DF, not an approximation.

---

## 2. Bootstrapping: from quotes to curve

### 2.1 The instruments and their pricing equations

A curve is not observed; it is *implied* by the prices of liquid
instruments. Each instrument contributes one equation in the unknown DFs
(residual = PV mispricing, driven to zero):

* **Deposit** `(T, R)` — invest 1, receive `1 + RT` (simple interest) at
  `T`:

  $$ DF(T)\,(1 + R\,T) - 1 = 0 \quad\Longrightarrow\quad DF(T) = \frac{1}{1 + R\,T} $$

* **FRA** `(T_1, T_2, R)` — lock the simple forward:

  $$ DF(T_1) - \bigl(1 + R\,(T_2 - T_1)\bigr)\,DF(T_2) = 0 $$

* **Par swap** `(T, R)` — annual fixed leg vs floating leg. In the
  **single-curve** framework the floating leg is projected off the same
  curve used for discounting, so each floating coupon over
  `[s_{i-1}, s_i]` has PV `DF(s_{i-1}) - DF(s_i)` and the leg
  **telescopes** to `1 - DF(T)`. Par means fixed PV = floating PV:

  $$ R \sum_i \tau_i\, DF(s_i) \;-\; \bigl(1 - DF(T)\bigr) = 0 $$

  with annual payment times `s_i` and accruals `\tau_i = s_i - s_{i-1}`
  (a non-integer maturity gets a short first stub: `T = 2.5` pays at
  `{0.5, 1.5, 2.5}`).

* **OIS swap** — annual fixed leg vs daily-compounded overnight leg.
  Approximating daily compounding by continuous compounding at the
  curve's short rate, each compounded coupon pays
  `DF(s_{i-1})/DF(s_i) - 1` and the floating leg again telescopes to
  `1 - DF(T)` — the residual is *identical in form* to the par swap. The
  daily-vs-continuous gap is `O(r^2/365)` (≈ 0.8 bp of rate at r = 5%,
  far below quote precision) and is a documented approximation.

### 2.2 The sequential algorithm

Sort instruments by pillar (maturity); pillars must be strictly
increasing — duplicates mean two instruments claim the same unknown and
are rejected. Then walk left to right; for each instrument the *only*
unknown is `x = DF(pillar)`, because every earlier cashflow date
interpolates on already-solved pillars (this is also how missing pillar
gaps are handled: a 4y swap's 3.5y coupon just interpolates). Solve
`residual(x) = 0` with Brent's method on the bracket `x ∈ [1e-10, 100]`
(`xtol = 1e-14`). The wide bracket is deliberate: `DF > 1` must be
reachable for negative rates.

**Failure is informative.** If no sign change exists on `(0, 100]`, then
*no positive discount factor can reprice the quote* — the inputs are
crossed/arbitrageable (e.g. a deposit with `1 + RT ≤ 0`, or an FRA quote
inconsistent with the surrounding curve implying a negative DF). The
bootstrap raises an error naming the pillar rather than silently
producing garbage. Detection at construction (`1 + RT > 0` checks) plus
detection at solve time (bracketing failure) covers both.

Because each equation is solved exactly, the finished curve **reprices
every input** — the round-trip test asserts `|residual| < 1e-9` for all
instruments (the reference achieves < 1e-15), and `boot_usd_par_10y`
recovers the 10y USD quote 0.0425 to 1e-10 from the bootstrapped curve.

### 2.3 Worked example: the EUR negative short end (golden: `boot_eur_df_1y`)

EUR quotes (from `data/curve_quotes.csv`): 1y deposit at −0.30%, 2y par
swap at −0.20%.

* 1y pillar: `DF(1) = 1/(1 + (−0.003)·1) = 1/0.997 = 1.0030090270812433`.
  Greater than 1 — the golden case asserts it.
* 2y pillar: the 2y swap pays fixed at `{1, 2}`; with `DF(1)` known the
  residual is linear in `DF(2)` and solvable by hand:

  $$
  DF(2) = \frac{1 - R\,DF(1)}{1 + R}
        = \frac{1 + 0.002 \times 1.003009}{0.998}
        = 1.0040140461
  $$

  Brent finds precisely this root. Note the mechanism: negative par rate
  → the fixed leg *receives* negative coupons → DF must exceed 1 for the
  legs to balance.

### 2.4 Single-curve vs the multi-curve/OIS reality

Pre-2008, LIBOR was treated as (nearly) risk-free and one curve did both
jobs: project floating coupons *and* discount them — the framework
implemented here, in which the floating leg telescopes. The 2008 crisis
blew out the basis between LIBOR tenors and OIS (3s6s basis, LIBOR-OIS
spread reached hundreds of bp), revealing that projection and discounting
are different questions:

* **Discounting** should use the rate actually earned on posted
  collateral — the overnight rate — so desks discount on an **OIS curve**
  (now €STR, SOFR, SONIA after the LIBOR transition).
* **Projection** of a floating index needs that index's own forward
  curve, bootstrapped from instruments referencing it, *given* the OIS
  discount curve. The floating leg no longer telescopes; the
  forward-basis spreads become market inputs.

This project implements the single-curve framework because it isolates
the bootstrapping mathematics, and separately provides the **OIS
bootstrap** (`ois_quotes.csv` → discount curve), which is exactly the
discount-curve leg of a modern multi-curve setup. What is *not*
implemented — and would be the next step on a desk — is projecting one
index off one curve while discounting on another, plus the basis-swap
instruments that pin the spread between them. In the post-LIBOR world of
compounded-in-arrears SOFR/€STR swaps, single-curve pricing off the OIS
curve is actually *correct* again for standard OIS — the multi-curve
machinery survives for term-rate and cross-currency basis.

---

## 3. Short-rate models: why mean reversion

A short-rate model posits dynamics for the instantaneous rate `r(t)` under
the risk-neutral measure and prices every bond as

$$
P(t,T) = \mathbb{E}^{\mathbb{Q}}_t\!\left[ e^{-\int_t^T r(s)\,ds} \right]
$$

Why must the drift pull `r` toward a level? Three reasons, in increasing
order of rigor:

1. **Economics.** Central banks react: high rates cool the economy and
   are cut; low rates are raised. Rates are policy-anchored in a way
   equity prices are not — a random walk is the wrong null model.
2. **Statistics.** Historical short rates show strong negative
   autocorrelation of changes at long horizons; variance of `r_{t+h}-r_t`
   grows far slower than linearly in `h`.
3. **Term-structure sanity.** Without mean reversion (`kappa -> 0`, i.e.
   the Merton/Ho-Lee limit), `Var[r_T] = sigma^2 T` grows without bound,
   long-dated bond-yield volatility does not decay, and long forward
   rates explode downward from the convexity term `-sigma^2 T^2/2`. Mean
   reversion caps the stationary variance at `sigma^2/(2\kappa)` and
   makes long yields converge to a constant — as observed.

Mean reversion also produces the single most important qualitative
feature of rate options: **volatility of a T-maturity bond yield decays
with maturity** (the `B(t,T)/(T-t)` factor below), so caps/swaptions on
long tenors are relatively cheaper than a random-walk model would say.

---

## 4. The Vasicek model

### 4.1 Dynamics and distribution

$$
dr = \kappa(\theta - r)\,dt + \sigma\,dW, \qquad \kappa > 0,\ \sigma \ge 0
$$

`theta` is the long-run level, `kappa` the reversion speed (`ln 2 /
kappa` is the half-life of a deviation), `sigma` the absolute
(normal/bps) volatility. Substituting `x = r - \theta` gives a mean-zero
Ornstein-Uhlenbeck process, solvable by the integrating factor
`e^{\kappa t}`:

$$
r_T = \theta + (r_t - \theta)e^{-\kappa(T-t)}
      + \sigma \int_t^T e^{-\kappa(T-s)}\,dW_s
$$

The Ito integral is Gaussian, so `r_T | r_t` is normal with

$$
\mathbb{E}[r_T] = \theta + (r_t-\theta)e^{-\kappa\tau},
\qquad
\mathrm{Var}[r_T] = \frac{\sigma^2}{2\kappa}\bigl(1 - e^{-2\kappa\tau}\bigr),
\qquad \tau = T - t
$$

Golden check (`vas_rT_mean`, `vas_rT_var`): with
`kappa=0.5, theta=0.03, sigma=0.01, r0=0.02, T=2`:
`E[r_2] = 0.03 - 0.01·e^{-1} = 0.026321205588285577`,
`Var[r_2] = 1e-4·(1-e^{-2})/1 = 8.646647167633873e-05`
(sd ≈ 93 bp). Being Gaussian, `r` goes negative with positive
probability — historically cited as a flaw, now correctly seen as a
feature.

### 4.2 ZCB prices: the affine machinery

Guess the exponential-affine form `P(t,T) = A(\tau)e^{-B(\tau) r}`. The
term-structure PDE

$$
\frac{\partial P}{\partial t} + \kappa(\theta - r)\frac{\partial P}{\partial r}
+ \tfrac{1}{2}\sigma^2 \frac{\partial^2 P}{\partial r^2} - rP = 0,
\qquad P(T,T) = 1
$$

separates, because every coefficient is affine in `r`, into two ODEs:
`B' = 1 - \kappa B` (Riccati, here linear) with `B(0)=0`, and
`(\ln A)' = -\theta\kappa B + \tfrac12 \sigma^2 B^2` with `A(0)=1`.
Solving:

$$
B(\tau) = \frac{1 - e^{-\kappa\tau}}{\kappa},
\qquad
A(\tau) = \exp\!\left[
  \Bigl(\theta - \frac{\sigma^2}{2\kappa^2}\Bigr)\bigl(B(\tau) - \tau\bigr)
  - \frac{\sigma^2 B(\tau)^2}{4\kappa}
\right]
$$

`B` is the bond's sensitivity `-\partial \ln P/\partial r` — a model
duration, saturating at `1/\kappa` for long maturities (mean reversion
caps duration risk). The `\sigma^2` terms in `A` are **convexity**:
volatility *raises* bond prices (Jensen's inequality on
`e^{-\int r}`), so yields are pulled down at the long end; the long-yield
asymptote is `y_\infty = \theta - \sigma^2/(2\kappa^2)`.

**Worked example** (`vas_zcb_5y`): `kappa=0.5, theta=0.03, sigma=0.01,
r0=0.02, T=5`:

```
B(5) = (1 - e^{-2.5})/0.5              = 1.8358300028
A(5) = exp[(0.03 - 0.0002)(1.83583 - 5) - 0.0001·1.83583²/2]
                                        = 0.9098634192
P(0,5) = A e^{-B·0.02}                  = 0.8770621876849438
y(0,5) = -ln P / 5                      = 0.0262355  (r0=2% pulled toward theta=3%)
```

The negative-rate case (`vas_zcb_neg_r0`, `r0 = -0.5%`) works through the
identical formulas and gives `P(0,5) = 0.989399475473229` — nothing
special happens at zero.

### 4.3 Options on ZCBs: the Jamshidian formula

Under the T-forward measure, `P(T,S)/P(T,T)` is a lognormal martingale
(Gaussian rates → lognormal bond prices), so a European call with expiry
`T`, bond maturity `S`, strike `K` prices by a Black-style formula
(Jamshidian 1989):

$$
\sigma_p = \sigma\sqrt{\frac{1 - e^{-2\kappa T}}{2\kappa}}\;B(S-T),
\qquad
h = \frac{\ln\bigl(P(0,S)/(K\,P(0,T))\bigr)}{\sigma_p} + \frac{\sigma_p}{2}
$$

$$
ZBC = P(0,S)\,N(h) - K\,P(0,T)\,N(h - \sigma_p),
\qquad
ZBP = K\,P(0,T)\,N(\sigma_p - h) - P(0,S)\,N(-h)
$$

`sigma_p` is exactly (std dev of `r_T`) x (duration `B(S-T)` of the
underlying bond at expiry) — the volatility of the log forward bond
price. When `sigma_p = 0` (`sigma=0`, `T=0`, or `S=T`) the option
degenerates to intrinsic on the forward bond,
`\max(\pm(P(0,S) - K P(0,T)), 0)`.

**Worked example** (`vas_zbc`/`vas_zbp`, same parameters, `T=1, S=3,
K=0.95`):

```
P(0,1) = 0.9781238661   P(0,3) = 0.9283986521
sigma_p = 0.01·sqrt((1-e^{-1})/1)·B(2) = 0.0100514767
h       = ln(0.92840/(0.95·0.97812))/sigma_p + sigma_p/2 = -0.0827024210
ZBC     = 0.0033292795093556   ZBP = 0.0041483002148354
```

Put-call parity `ZBC - ZBP = P(0,S) - K\,P(0,T) = -0.00081902` holds to
1e-15 — the test suite asserts it on a grid of strikes.

### 4.4 Calibration and the model's limits

`calibrate_vasicek` fits `(\kappa, \theta, \sigma)` to observed zero
yields by least squares with a native Nelder-Mead (out-of-domain
parameters get a smooth penalty; non-convergence is *reported* via a
`converged` flag, never raised; `sigma` is returned as `|sigma|` since
the objective is even in it). On the bundled `zero_yields.csv` —
generated from a true `(0.8, 0.05, 0.02)` model plus seeded noise — the
fit recovers `kappa=0.7977, theta=0.0501, sigma=0.0225` with RMSE
1.7e-5.

But note what Vasicek *cannot* do: with three parameters its yield curve
is a rigid one-hump/monotone family. It can never reproduce an arbitrary
market curve — today's EUR curve with a negative short end and a 30y
plateau is simply outside the family. If your product's value depends on
the *whole* curve being right (every swaption, every cap), you need a
model with a functional degree of freedom. That is Hull-White.

---

## 5. Hull-White: Vasicek fitted to the market curve

### 5.1 theta(t): buying an exact fit with a time-dependent drift

$$
dr = \bigl(\theta(t) - a\,r\bigr)\,dt + \sigma\,dW
$$

Replacing the constant `\kappa\theta` with a free function `\theta(t)`
adds exactly enough freedom to match *every* market discount factor. The
no-arbitrage condition (fit `P(0,T) = P^M(0,T)` for all `T`) forces

$$
\theta(t) = \frac{\partial f^M(0,t)}{\partial t} + a\,f^M(0,t)
          + \frac{\sigma^2}{2a}\bigl(1 - e^{-2at}\bigr)
$$

Read it term by term: the drift must (i) track the slope of the market
forward curve, (ii) offset the reversion pull `-a r` at the forward
level, and (iii) add back the convexity correction that Gaussian
volatility subtracts from forwards. In this implementation `f^M(0,t)`
comes from a *contractual* finite-difference on the curve
(`h = 1e-5`, central for `t >= h`, one-sided at the origin — every port
uses the identical stencil so golden values agree bit-for-bit), and
`theta(t)` itself is exposed only for inspection: **pricing never
evaluates it**, because the fit can be absorbed directly into the affine
`A` via market DFs:

$$
B(t,T) = \frac{1 - e^{-a(T-t)}}{a},\qquad
A(t,T) = \frac{P^M(0,T)}{P^M(0,t)}
\exp\!\left[B(t,T)f^M(0,t)
 - \frac{\sigma^2}{4a}\bigl(1-e^{-2at}\bigr)B(t,T)^2\right]
$$

$$
P(t,T) = A(t,T)\,e^{-B(t,T)\,r}
$$

At `t = 0` with `r = r(0) = f^M(0,0)`, the `e^{B f}` and `e^{-B r}`
factors cancel *algebraically*, so `P(0,T)` equals the market DF exactly
— no finite-difference error enters (`hw_zcb_t0`, tol 1e-12). The model
reproduces the curve by construction; `sigma` and `a` are then free to
match option prices.

### 5.2 Caps and floors via ZCB options

The ZBC/ZBP formulas are Vasicek's with `a` for `\kappa` and market DFs
in place of model DFs (`sigma_p` is identical — the *volatility*
structure of Hull-White is Vasicek's; only the drift changed).

The key replication: a **caplet** on the simple forward over
`[T_r, T_p]`, strike `K`, `\tau = T_p - T_r`, pays
`\tau\,\max(F(T_r) - K, 0)` at `T_p`. Discounting the payoff from `T_p`
to `T_r` with `P(T_r,T_p) = 1/(1+F\tau)` turns it into a payoff at reset:

$$
\frac{\tau (F - K)^+}{1 + F\tau}
= (1 + K\tau)\left(\frac{1}{1+K\tau} - P(T_r, T_p)\right)^{\!+}
$$

— i.e. the caplet is `(1 + K\tau)` **puts on the ZCB** `P(T_r,T_p)`
struck at `1/(1 + K\tau)`:

$$
\text{caplet} = N(1 + K\tau)\; ZBP\!\left(T_r,\ T_p,\ \tfrac{1}{1+K\tau}\right)
$$

and a floorlet is the same with ZBC. A **cap is a strip of caplets** over
consecutive schedule dates, each an independent closed form — the golden
suite checks that the sum of the three caplets equals `hw_cap_3y`
exactly.

**Worked example** (`hw_caplet`): golden curve, `a=0.1, sigma=0.01`,
reset 1y, pay 2y, `K=2.5%`, notional 100:

```
k_bond  = 1/(1 + 0.025)                       = 0.9756097561
sigma_p = 0.01·sqrt((1-e^{-0.2})/0.2)·B(1,2)  = 0.0090596889
ZBP(1, 2, k_bond)                             = 0.0033602695
caplet  = 100 · 1.025 · ZBP                   = 0.3444276269860027
floorlet (same strike, ZBC)                   = 0.3632810046194777
```

Sanity checks the tests enforce: caplet − floorlet = PV of the swaplet,
`100\,\tau(F-K)DF(T_p) = -0.0188534` with the curve forward
`F(1,2) = 0.0248026` (cap/floor parity); and with `sigma = 0` the caplet
collapses to discounted intrinsic — `hw_caplet_sigma0` at `K=2%` gives
`100 · DF(2) · (0.0248026 - 0.02) = 0.45866760346186464`.

### 5.3 The Jamshidian swaption decomposition

A payer swaption (expiry `T`, payments `u_1 < ... < u_m`, fixed rate `X`)
has exercise value `\max(1 - \sum_i c_i P(T,u_i),\, 0)` with coupons
`c_i = X\tau_i`, last coupon plus notional `c_m = X\tau_m + 1` — a **put
with strike 1 on a coupon bond**. Options on portfolios are normally
harder than portfolios of options, but in any one-factor model where all
bond prices move monotonically in the single state variable, Jamshidian's
trick makes them equal:

1. Because each `B(T,u_i) > 0`, the coupon-bond price
   `g(r) = \sum_i c_i P(T,u_i;r)` is **strictly decreasing** in `r`.
   Solve `g(r^*) = 1` — a 1-D Brent root find on an expanding bracket
   (start `[-1,1]`, double the failing side, max 24 doublings); the tests
   require `|g(r^*) - 1| < 1e-10`.
2. Set decomposition strikes `K_i = P(T, u_i; r^*)`. By monotonicity, the
   swaption is in the money **iff** `r_T > r^*`, and on that event *every*
   ZCB put `(K_i - P(T,u_i))^+` is in the money simultaneously. So the
   max of the sum equals the sum of the maxes:

$$
\text{payer} = N\sum_i c_i\, ZBP(T, u_i, K_i),
\qquad
\text{receiver} = N\sum_i c_i\, ZBC(T, u_i, K_i)
$$

**Worked example** (`hw_swaption_payer`): golden curve, `a=0.1,
sigma=0.01`, expiry 1y into a 5y annual swap at `X=2.8%`, notional 100:
the root find gives `r^* = 0.0179465821`, strikes
`K = (0.980461, 0.957922, 0.931269, 0.904989, 0.869951)`, and value
`2.4419483143080454`. The same code run on the negative-rate curve with
`X = 0` (`hw_swaption_neg_curve`) gives `1.3848608836` with `r^*` *below
zero* — Gaussian models don't care.

The decomposition is exact only in one-factor models — with two factors,
bonds of different maturities are no longer comonotone and the trick
fails; desks then use numeraire-measure integration or approximation.

### 5.4 Monte Carlo without a tree, without bias

Both models reduce to a mean-zero OU process
(`x = r - \theta` for Vasicek; `r(t) = x(t) + \alpha(t)` for Hull-White
with `\alpha(t) = f^M(0,t) + \tfrac{\sigma^2}{2a^2}(1-e^{-at})^2`). For
pricing you need the *pair* `(x_{t+dt}, \int_t^{t+dt} x\,ds)`, which is
jointly Gaussian with closed-form moments — so instead of an Euler scheme
with `O(dt)` bias, the simulation samples the exact transition
(conditional decomposition: draw `x'`, then the integral given `x'`).
Consequences worth internalizing:

* Estimates are unbiased at **any** step count — the golden MC cases pass
  with 8 steps as surely as with 800; only the payoff dates constrain the
  grid.
* The Hull-White pathwise discount factor uses the identity
  `\int_0^t \alpha\,ds = -\ln P^M(0,t) + V(t)` with closed-form `V(t)`,
  so even the discounting is exact — the MC caplet differs from the
  closed form by pure statistical noise (golden band: 6 standard errors
  at 50,000 paths, SE ≈ 1.2e-4 on a 0.877 price for `vas_mc_zcb`).
* `sigma = 0` short-circuits to the deterministic decay: MC equals the
  analytic price with SE exactly 0 (asserted in tests).

---

## 6. Curve risk: DV01 in practice

**DV01** (dollar value of a basis point; also PV01) is the PV change for
a 1 bp move in rates. For a par swap the fast mental model: a receiver
swap of maturity `T` has

$$
\text{DV01} \approx \text{annuity} \times 10^{-4} \times N,
\qquad \text{annuity} = \sum_i \tau_i\,DF(s_i)
$$

On the bootstrapped USD curve the 10y annuity is `7.9791`, so a 100
notional 10y par swap has DV01 ≈ `0.0798` — verified by actually bumping:
re-bootstrapping with **all quotes +1 bp** moves the receiver PV by
`-0.07975` (the tiny gap to the annuity approximation is convexity plus
discounting feedback).

Two flavors matter on a desk, and this project's design makes both cheap:

* **Parallel DV01**: bump every input quote, re-bootstrap, reprice.
* **Keyed (bucketed) DV01**: bump one quote at a time; the vector of
  sensitivities tells you *where* on the curve your risk lives, and is
  what you hedge with the quoted instruments themselves.

A beautiful and practically vital fact: **in quote space, par-instrument
deltas localize.** Bumping only the 10y USD quote by 1 bp changes the 10y
par swap PV by `-0.07978` — essentially the whole parallel DV01 — while
bumping only the 5y quote changes it by ~1e-14, i.e. *nothing*. Why:
after re-bootstrap, the 10y swap must still reprice its own (unchanged)
quote exactly, whatever happened at 5y. The Jacobian of par rates with
respect to quotes is the identity, so hedging a 10y swap takes exactly a
10y swap. This locality is a direct consequence of (a) bootstrapping
exactly and (b) the *local* interpolation scheme — a cubic spline would
smear the bump across the whole curve and produce nonzero (and
counterintuitive) deltas everywhere. Interpolation choice is a *risk*
choice, not just an aesthetic one.

Zero/forward-space bumps (bump `z(t)` pillars directly) are also common;
they are related to quote-space bumps by the same Jacobian and the test
`f(9.99) = 4.080\%` vs `f(10.01) = 4.234\%` shows the forward step at a
pillar you would be bumping across.

---

## 7. Market conventions: where this project simplifies desk practice

Each simplification is deliberate — the model content is unchanged, and
each is a data-preparation issue, but you must know them before comparing
numbers to a real system:

| Topic | This project | Desk practice |
|---|---|---|
| Day count | ACT/365F for *everything*; time = days/365 | Deposits/FRAs: ACT/360 (USD, EUR); swap fixed legs: 30/360 (USD) or ACT/365 (GBP); OIS: ACT/360. A 6M ACT/360 deposit accrues `~0.5069`, not `0.5` — rates differ by ~1.4% relative |
| Dates | Year fractions on a continuous axis | Actual calendars: business-day rolls (modified following), spot lag (T+2), holiday centers, end-of-month rules |
| Futures/FRA | Treated as simple forwards | Exchange futures expire on **IMM dates** (3rd Wednesday of Mar/Jun/Sep/Dec), are margined daily, and need a **convexity adjustment** (futures rate > forward rate because margin flows correlate with rates; size `~\tfrac12\sigma^2 T_1 T_2` in a Gaussian model — computable with the Hull-White machinery in this package) |
| Swap schedule | Annual fixed leg, stub-first | USD: semiannual fixed vs 3M float (legacy) or annual SOFR OIS; EUR: annual fixed vs 6M; schedules generated from date rolls |
| Curves | Single curve (+ separate OIS bootstrap) | Multi-curve: OIS discounting, per-index projection curves, basis swaps as inputs; CSA-dependent discounting (the "cheapest-to-deliver" collateral curve) |
| OIS floating leg | Continuous compounding approximation | Daily compounded in arrears with lockouts/lookbacks and payment lags |
| Quotes | Clean decimal rates | Bid/ask, quote conventions per market, turn-of-year effects, meeting-date jumps around central bank dates |

The models themselves (Gaussian short rate, affine bonds, Jamshidian)
are exactly the production formulas; Hull-White with these conventions
fixed is still a benchmark model on real desks for callable bonds,
Bermudans (via trees/PDE on the same dynamics) and quick swaption risk.

---

## 8. Common pitfalls and numerical issues

1. **theta(t) spikes at pillars — by design, not by bug.** With
   piecewise-constant forwards, `f^M(0,t)` *jumps* at each pillar, so the
   finite-difference `df/dt` in `theta(t)` is enormous there
   (`theta(1.0) ≈ 15.0` on the golden curve vs `theta(0.5) ≈ 0.0022`!).
   This is the interpolation's forward discontinuity showing up in a
   derivative. It is harmless for pricing — the implementation absorbs
   `theta` into `A(t,T)` via market DFs and never differentiates —
   but it is exactly why *smooth* interpolation matters if you ever
   discretize `theta(t)` on a tree/PDE grid.
2. **Finite-difference contracts must be bit-identical across ports.**
   `f^M(0,t)` is defined *by its stencil* (`h = 1e-5`, central/one-sided
   split). A port that "improves" it to an analytic slope will disagree
   with golden values at `t > 0` where the stencil straddles a pillar
   (it averages the two adjacent forwards). Treat numerical schemes in a
   contract as part of the model.
3. **Root-find brackets encode no-arbitrage.** The bootstrap bracket
   `DF ∈ [1e-10, 100]` fails exactly when quotes are crossed; the
   Jamshidian bracket expansion fails only for degenerate coupon bonds
   (e.g. absurd fixed rates). Report *which pillar* failed — a bare
   "no sign change" is undebuggable at 4:55pm.
4. **`sigma_p = 0` needs an explicit branch.** `sigma = 0`, `T = 0` and
   `S = T` all make the Black-style formula 0/0; the option must return
   forward intrinsic. Tests pin all three degenerations.
5. **Don't validate `DF <= 1`.** A tempting "sanity check" that breaks
   every negative-rate curve. The invariants are `DF > 0` and (per
   segment) whatever sign the forward implies — the tests assert `DF > 1`
   *is supported*.
6. **Nelder-Mead converges to wherever it converges.** The Vasicek
   least-squares surface is nearly flat in `kappa` x `sigma` for short
   maturity sets (yields depend on `sigma` only through convexity, which
   is tiny). Use domain penalties (not hard throws — the simplex must be
   able to wander), report `converged`, return `|sigma|`, and never trust
   a calibration you haven't repriced.
7. **Sample-vs-population standard errors.** MC SE uses `ddof = 1`; with
   50,000 paths it hardly matters, but golden bands are calibrated to the
   sample convention — a port using `ddof = 0` differs in the 5th digit
   of the SE, which is invisible until a 6-SE band test flakes.
8. **`erfc`, not `erf`, for the normal CDF.** `N(x) = \tfrac12
   \mathrm{erfc}(-x/\sqrt2)` keeps full relative accuracy in the far
   tails where `1 - \mathrm{erf}` cancels catastrophically — deep
   out-of-the-money options are exactly where sloppy CDFs bite.
9. **Time zero is special everywhere.** `z(0)` is a 0/0 resolved by the
   limit (first segment's forward); `f(0,0)` uses the one-sided stencil;
   the implicit `(0, DF=1)` node must not be user-suppliable. Off-by-one
   segment lookups at pillars (left vs right segment for `f`) are the
   classic silent bug — the spec pins "right segment at a pillar".

---

## 9. Interview-style Q&A

**Q1. Why does log-linear DF interpolation equal piecewise-constant
forwards, and why do traders like it?**
Because `f(t) = -d\ln DF/dt`, a linear `\ln DF` segment has constant
slope, i.e. constant forward. Traders like it because it is local (quote
bumps move only the adjacent segment, so risk localizes to hedgeable
instruments), it cannot oscillate, and any arbitrage (negative forward)
is immediately visible as an increasing `\ln DF` segment. Its cost: the
forward curve is a step function, which is ugly for products sensitive to
forward smoothness and makes `theta(t)` spike at pillars.

**Q2. Why does a swap's floating leg "telescope" to `1 - DF(T)`, and when
does that break?**
Each floating coupon set at `s_{i-1}`, paid at `s_i`, projected *and*
discounted off the same curve, has PV
`DF(s_i)\,\tau_i F(s_{i-1},s_i) = DF(s_{i-1}) - DF(s_i)`; summing
telescopes. It breaks the moment projection and discounting curves differ
(multi-curve: LIBOR/EURIBOR projected, OIS discounted) — then the
floating leg needs explicit forwards plus a basis, which is precisely the
post-2008 reality.

**Q3. Why did the market abandon single-curve pricing in 2008?**
The LIBOR-OIS spread exploded (bank credit/liquidity risk in LIBOR),
and tenor basis swaps (3M vs 6M) went from ~0 to tens of bp. A single
curve cannot reprice instruments on different indices simultaneously,
and discounting at LIBOR misprices collateralized trades whose collateral
earns the overnight rate. Hence: OIS discounting + per-index projection
curves.

**Q4. What goes wrong in a short-rate model *without* mean reversion?**
`Var[r_T] = \sigma^2 T` unbounded: long-yield volatility does not decay
(contradicting swaption vol surfaces), forward rates acquire an
ever-growing `-\tfrac12\sigma^2 T^2` convexity droop, and the model
assigns silly probabilities to extreme rates at long horizons. Mean
reversion bounds the stationary variance at `\sigma^2/(2\kappa)` and
gives bond-yield vols the observed maturity decay through
`B(\tau)/\tau`.

**Q5. Derive the Vasicek bond-price sensitivity and interpret `B`.**
From `P = A e^{-Br}`, `\partial P/\partial r = -BP`, so
`B(\tau) = (1-e^{-\kappa\tau})/\kappa` is the ZCB's semi-elasticity
(model duration). It grows like `\tau` for small `\tau` and saturates at
`1/\kappa`: shocks to today's short rate decay before they can affect
distant discounting, so long bonds carry *bounded* short-rate risk. This
is also why `\sigma_p` (bond-option vol) contains `B(S-T)`.

**Q6. Vasicek fits the market curve badly. Hull-White fixes this — at
what cost, and what does it *not* fix?**
`theta(t)` gives an exact fit to today's discount curve — that's a
functional degree of freedom, so t=0 bonds are perfect by construction.
It does not fix: single-factor dynamics (all rates perfectly correlated —
no decorrelation between 2y and 30y, so curve-shape options like CMS
spreads are mispriced), constant `\sigma` and `a` (cannot match the whole
swaption matrix — one column at best; time-dependent `\sigma(t)`, or
two factors G2++, is the next step), and Gaussian tails (no smile/skew).

**Q7. Walk through Jamshidian's swaption decomposition and state the
exact assumption it needs.**
Write the swaption payoff as an option on a coupon bond; find `r^*`
making the coupon bond worth par at expiry; strike each coupon's ZCB at
its own `K_i = P(T,u_i;r^*)`; then the coupon-bond option equals the sum
of ZCB options because all bonds cross their strikes at the *same* value
of the single state variable (comonotonicity). Needs: one-factor dynamics
with bond prices strictly monotone in the state — true for any
single-factor affine model with `B > 0`; false with two or more factors.

**Q8. Your bootstrap throws "root not bracketed" at the 4y pillar. What
do you check, in order?**
(1) The 4y quote itself — fat finger, wrong sign, percent-vs-decimal.
(2) Consistency with neighbors — a 4y swap rate far below 3y implies a
negative forward DF ratio; compute the implied `F(3,4)` and check
`1 + F\tau > 0`. (3) Instrument ordering/duplicates — two instruments on
one pillar. (4) Units of time — an ACT/360 accrual fed into an ACT/365
system shifts pillars. The bracket `[1e-10, 100]` is wide enough that a
genuine failure almost always means arbitrage-inconsistent inputs, not
solver trouble.

**Q9. Why is the exact-transition MC unbiased at any step size while
Euler is not, and why simulate the *pair* `(r, \int r\,dt)`?**
OU transitions are Gaussian with known conditional moments, so sampling
from the exact transition law introduces no discretization error —
Euler's `O(dt)` bias comes from freezing coefficients within a step.
Discounting needs `\int r\,ds`, and approximating that integral from
sampled endpoints (trapezoid) *reintroduces* bias; sampling the integral
jointly with the endpoint from their exact bivariate Gaussian (regression
of `I` on `x'`, plus independent residual) removes it entirely. The MC
then differs from closed forms by pure `1/\sqrt{n}` noise.

**Q10. How would you compute the DV01 of a 7y swap on this curve, and
what pattern do you expect across pillar buckets?**
Bump each input quote by 1 bp, re-bootstrap, reprice; assemble the vector
of PV changes. Expected pattern: essentially all sensitivity in the 7y
quote bucket and near-zero elsewhere, because the bootstrap forces every
par instrument to reprice its own quote exactly (quote-space Jacobian ≈
identity) — verified numerically here: bumping only the 5y quote moves a
10y par swap by ~1e-14, bumping the 10y quote reproduces the full
parallel DV01 of −0.0798 per 100 notional. With non-local interpolation
(splines) the sensitivity smears across buckets, which is a real reason
desks avoid splines for risk curves.

---

## 10. Further reading

* **Brigo & Mercurio**, *Interest Rate Models — Theory and Practice*
  (2nd ed., Springer 2006) — the reference for everything here: affine
  models ch. 3-4, Hull-White ch. 3.3, Jamshidian decomposition,
  multi-curve in the 2nd edition's later chapters.
* **Andersen & Piterbarg**, *Interest Rate Modeling* (3 vols, Atlantic
  Financial Press 2010) — the desk-level treatment: vol. 1 for curve
  construction and Gaussian models, vol. 2 for vanilla calibration.
* **Hagan & West**, "Interpolation Methods for Curve Construction",
  *Applied Mathematical Finance* 13(2), 2006 — the interpolation survey;
  source of the monotone-convex scheme.
* **Vasicek**, "An Equilibrium Characterization of the Term Structure",
  *JFE* 5, 1977. **Hull & White**, "Pricing Interest-Rate-Derivative
  Securities", *RFS* 3, 1990. **Jamshidian**, "An Exact Bond Option
  Formula", *Journal of Finance* 44, 1989 — the three primary sources.
* **Bianchetti & Carlicchi**, "Interest Rates After the Credit Crunch:
  Multiple Curve Vanilla Derivatives and SABR" (2011) — readable
  multi-curve overview.
* **Glasserman**, *Monte Carlo Methods in Financial Engineering*
  (Springer 2003) — ch. 3.3 for exact Gaussian simulation of OU/Vasicek.
* For the LIBOR transition and RFR conventions: **Schrimpf & Sushko**,
  "Beyond LIBOR: a primer on the new benchmark rates", *BIS Quarterly
  Review*, March 2019.
