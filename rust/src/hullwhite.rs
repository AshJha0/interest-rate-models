//! Hull-White one-factor model `dr = (theta(t) - a r) dt + sigma dW`.
//!
//! The model is fitted to an input [`DiscountCurve`] so that model ZCB
//! prices at `t = 0` reproduce the market discount factors *exactly*.
//! All formulas use market DFs `P^M(0, .)` directly:
//!
//! * `B(t,T) = (1 - e^{-a (T-t)}) / a`
//! * `A(t,T) = (P^M(0,T)/P^M(0,t)) * exp[B(t,T) f^M(0,t)
//!   - (sigma^2/(4a)) (1 - e^{-2 a t}) B(t,T)^2]`
//! * `P(t,T) = A(t,T) e^{-B(t,T) r(t)}`
//!
//! `f^M(0,t)` is the market instantaneous forward, computed by the *exact*
//! finite-difference scheme documented in `API_SPEC.md` (step `h = 1e-5`,
//! central for `t >= h`, one-sided at the short end) so that every
//! language port reproduces bit-comparable values.
//!
//! `theta(t)` fitting: matching the curve requires
//! `theta(t) = d f^M(0,t)/dt + a f^M(0,t) + (sigma^2/(2a)) (1 - e^{-2 a t})`
//! — the drift must (i) track the slope of the forward curve, (ii) offset
//! the pull `-a r` at the forward level, and (iii) add back the convexity
//! `sigma^2` term.  Pricing never needs `theta(t)` explicitly (it is
//! absorbed into `A` via the market DFs); the function is provided for
//! inspection and testing.
//!
//! Options: ZBP/ZBC closed forms, caplets/floorlets as ZCB puts/calls,
//! caps as caplet strips, and Jamshidian's decomposition for European
//! swaptions.  Monte Carlo simulates `r(t) = x(t) + alpha(t)` with exact
//! Gaussian OU moments (trinomial-tree-free),
//! `alpha(t) = f^M(0,t) + (sigma^2/(2a^2)) (1 - e^{-a t})^2`.
//!
//! Calibration: [`calibrate_hullwhite`] fits `(a, sigma)` (or sigma with
//! `a` fixed) to caplet / payer-swaption prices by least squares from
//! several Nelder-Mead starts; [`crate::bachelier`] converts prices to
//! normal (bp) implied volatilities.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::curve::{par_swap_rate, DiscountCurve};
use crate::error::{IrmError, Result};
use crate::mathutils::{norm_cdf, ou_integral_variance, ou_step_moments};
use crate::optimize::nelder_mead;
use crate::rootfind::{brentq, BRENT_MAXITER, BRENT_RTOL};

/// Finite-difference step for the market instantaneous forward `f(0,t)`.
pub const FD_STEP: f64 = 1e-5;
/// The coupon-bond residual `|g(r*)|` must be below this after the root find.
pub const JAMSHIDIAN_RESIDUAL_TOL: f64 = 1e-10;
/// Maximum bracket doublings when searching for r*.
const JAMSHIDIAN_MAX_DOUBLINGS: usize = 24;
/// Largest exponent passed to exp() before a ZCB price is declared not
/// representable (exp(709.78) is the double overflow point).
const MAX_EXP_ARG: f64 = 700.0;

/// Calibration domain `(a_lo, a_hi, sigma_hi)` (sigma_lo = 0); outside gets
/// the penalty `1e6 (1 + distance)`.
pub const HW_DOMAIN: (f64, f64, f64) = (1e-6, 5.0, 1.0);
/// Default multi-start `(a, sigma)` pairs (sigma-only starts use the sigmas).
pub const HW_CALIBRATION_STARTS: [(f64, f64); 3] = [(0.03, 0.005), (0.1, 0.01), (0.5, 0.02)];
const A_LO_BOUND: f64 = 1e-5;
const A_HI_BOUND: f64 = 4.995;
const SIGMA_LO_BOUND: f64 = 1e-5;
const SIGMA_HI_BOUND: f64 = 0.999;
const EQUIV_RMSE_FACTOR: f64 = 1.05;
const EQUIV_RMSE_FLOOR: f64 = 1e-14;
const IDENT_A: f64 = 1e-2;
const IDENT_SIGMA: f64 = 1e-4;
/// sigma is optimised as sigma * SIGMA_SCALE so one Nelder-Mead step fits both.
const SIGMA_SCALE: f64 = 10.0;

fn check_finite(name: &str, v: f64) -> Result<()> {
    if !v.is_finite() {
        return Err(IrmError::InvalidInput(format!(
            "{name} must be finite, got {v}"
        )));
    }
    Ok(())
}

/// Jamshidian swaption decomposition output.
#[derive(Debug, Clone, PartialEq)]
pub struct JamshidianResult {
    /// Swaption present value (notional included).
    pub value: f64,
    /// Critical short rate `r*` at which the coupon bond is at par.
    pub r_star: f64,
    /// Residual `g(r*)` of the root find (must be < 1e-10 in magnitude).
    pub residual: f64,
    /// Decomposition strikes `K_i = P(T, u_i; r*)`.
    pub strikes: Vec<f64>,
}

/// Hull-White 1F model fitted to a market discount curve.
#[derive(Debug, Clone)]
pub struct HullWhite {
    a: f64,
    sigma: f64,
    curve: DiscountCurve,
}

impl HullWhite {
    /// Validated constructor: `a > 0`, `sigma >= 0`, both finite.
    pub fn new(a: f64, sigma: f64, curve: DiscountCurve) -> Result<Self> {
        if !a.is_finite() || a <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "mean reversion a must be finite and > 0, got {a}"
            )));
        }
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "sigma must be finite and >= 0, got {sigma}"
            )));
        }
        Ok(Self { a, sigma, curve })
    }

    /// Mean-reversion speed.
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Short-rate volatility.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// The fitted market curve.
    pub fn curve(&self) -> &DiscountCurve {
        &self.curve
    }

    // ----------------------- market forward via FD -------------------- //

    /// Market instantaneous forward `f^M(0,t)` by finite difference.
    ///
    /// Exact scheme (`h = FD_STEP = 1e-5`), reproduced identically by all
    /// ports:
    ///
    /// * `t >= h`: central, `(ln DF(t-h) - ln DF(t+h)) / (2h)`
    /// * `t <  h`: one-sided, `(ln DF(t) - ln DF(t+h)) / h`
    ///
    /// With piecewise-constant forwards the FD equals the segment forward
    /// away from pillars and the average of the two adjacent forwards when
    /// the stencil straddles a pillar.
    pub fn fwd0(&self, t: f64) -> Result<f64> {
        if !t.is_finite() || t < 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "time must be finite and >= 0, got {t}"
            )));
        }
        let h = FD_STEP;
        if t >= h {
            Ok((self.curve.df(t - h)?.ln() - self.curve.df(t + h)?.ln()) / (2.0 * h))
        } else {
            Ok((self.curve.df(t)?.ln() - self.curve.df(t + h)?.ln()) / h)
        }
    }

    /// Fitted drift `theta(t)` (documentation/testing helper).
    ///
    /// `theta(t) = df/dt + a f + (sigma^2/(2a)) (1 - e^{-2 a t})` with
    /// `df/dt` approximated by the forward difference
    /// `(fwd0(t+h') - fwd0(t))/h'`, `h' = 1e-4` (see [`Self::theta_with_step`]).
    pub fn theta(&self, t: f64) -> Result<f64> {
        self.theta_with_step(t, 1e-4)
    }

    /// [`Self::theta`] with an explicit forward-difference step `h > 0`.
    pub fn theta_with_step(&self, t: f64, h: f64) -> Result<f64> {
        if !h.is_finite() || h <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "theta step h must be finite and > 0, got {h}"
            )));
        }
        let dfdt = (self.fwd0(t + h)? - self.fwd0(t)?) / h;
        Ok(dfdt
            + self.a * self.fwd0(t)?
            + (self.sigma * self.sigma / (2.0 * self.a))
                * (-(-2.0 * self.a * t).exp_m1()))
    }

    // ----------------------- affine ZCB ------------------------------- //

    /// `B(t,T) = (1 - e^{-a (T-t)}) / a` (`exp_m1` form), `0 <= t <= T` finite.
    pub fn b_factor(&self, t: f64, maturity: f64) -> Result<f64> {
        check_finite("t", t)?;
        check_finite("maturity", maturity)?;
        if t < 0.0 || maturity < t {
            return Err(IrmError::InvalidInput(format!(
                "need 0 <= t <= maturity, got t={t}, maturity={maturity}"
            )));
        }
        Ok(-(-self.a * (maturity - t)).exp_m1() / self.a)
    }

    /// `A(t,T)` built from market DFs and `f^M(0,t)` (see module docs).
    pub fn a_factor(&self, t: f64, maturity: f64) -> Result<f64> {
        let b = self.b_factor(t, maturity)?;
        let pm_t = self.curve.df(t)?;
        let pm_mat = self.curve.df(maturity)?;
        let conv = self.sigma * self.sigma / (4.0 * self.a)
            * (-(-2.0 * self.a * t).exp_m1())
            * b
            * b;
        Ok((pm_mat / pm_t) * (b * self.fwd0(t)? - conv).exp())
    }

    /// `P(t,T) = A(t,T) e^{-B(t,T) r}`.
    ///
    /// `r = None` defaults to `r(0) = f^M(0,0)` (so `P(0,T)` equals the
    /// market DF exactly: the `e^{B f}` and `e^{-B r}` terms cancel).
    /// Errors when `exp(-B r)` would overflow (result not representable).
    pub fn zcb_price(&self, t: f64, maturity: f64, r: Option<f64>) -> Result<f64> {
        let rr = match r {
            Some(v) => {
                if !v.is_finite() {
                    return Err(IrmError::InvalidInput(format!(
                        "short rate must be finite, got {v}"
                    )));
                }
                v
            }
            None => self.fwd0(0.0)?,
        };
        let expo = -self.b_factor(t, maturity)? * rr;
        if expo > MAX_EXP_ARG {
            return Err(IrmError::InvalidInput(format!(
                "ZCB price not representable: exp({expo}) overflows"
            )));
        }
        Ok(self.a_factor(t, maturity)? * expo.exp())
    }

    // ----------------------- ZCB options ------------------------------ //

    /// Std dev of `ln P` at expiry:
    /// `sigma_p = sigma sqrt((1 - e^{-2 a T})/(2a)) B(T,S)`.
    fn sigma_p(&self, expiry: f64, bond_maturity: f64) -> Result<f64> {
        Ok(self.sigma
            * ((-(-2.0 * self.a * expiry).exp_m1()) / (2.0 * self.a)).sqrt()
            * self.b_factor(expiry, bond_maturity)?)
    }

    fn zb_option(&self, expiry: f64, bond_maturity: f64, strike: f64, call: bool) -> Result<f64> {
        if !expiry.is_finite() || !bond_maturity.is_finite() || !strike.is_finite() {
            return Err(IrmError::InvalidInput("option inputs must be finite".into()));
        }
        if expiry < 0.0 || bond_maturity < expiry {
            return Err(IrmError::InvalidInput(format!(
                "need 0 <= expiry <= bond maturity, got {expiry}, {bond_maturity}"
            )));
        }
        if strike <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "strike must be > 0, got {strike}"
            )));
        }
        let p_s = self.curve.df(bond_maturity)?;
        let p_t = self.curve.df(expiry)?;
        let sp = self.sigma_p(expiry, bond_maturity)?;
        if sp == 0.0 {
            let mut intrinsic = p_s - strike * p_t;
            if !call {
                intrinsic = -intrinsic;
            }
            return Ok(intrinsic.max(0.0));
        }
        let h = (p_s / (p_t * strike)).ln() / sp + 0.5 * sp;
        if call {
            Ok(p_s * norm_cdf(h) - strike * p_t * norm_cdf(h - sp))
        } else {
            Ok(strike * p_t * norm_cdf(sp - h) - p_s * norm_cdf(-h))
        }
    }

    /// European call on a unit ZCB (uses market DFs — model-consistent).
    pub fn zbc(&self, expiry: f64, bond_maturity: f64, strike: f64) -> Result<f64> {
        self.zb_option(expiry, bond_maturity, strike, true)
    }

    /// European put on a unit ZCB.
    pub fn zbp(&self, expiry: f64, bond_maturity: f64, strike: f64) -> Result<f64> {
        self.zb_option(expiry, bond_maturity, strike, false)
    }

    // ----------------------- caplets / caps --------------------------- //

    /// Validate caplet-style inputs and return `tau`.
    fn check_caplet_inputs(
        what: &str,
        reset: f64,
        pay: f64,
        strike: f64,
        notional: f64,
    ) -> Result<f64> {
        check_finite("reset", reset)?;
        check_finite("pay", pay)?;
        check_finite("strike", strike)?;
        check_finite("notional", notional)?;
        let tau = pay - reset;
        if tau <= 0.0 || reset < 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "{what}: need 0 <= reset < pay, got {reset}, {pay}"
            )));
        }
        if 1.0 + strike * tau <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "{what}: strike implies non-positive 1 + K*tau ({strike})"
            )));
        }
        Ok(tau)
    }

    /// Caplet on the simple forward over `[reset, pay]`.
    ///
    /// Pays `notional * tau * max(F(reset,pay) - K, 0)` at `pay`.  A caplet
    /// is `(1 + K tau)` ZCB *puts* struck at `1/(1 + K tau)`:
    ///
    /// `caplet = notional (1 + K tau) ZBP(reset, pay, 1/(1 + K tau))`.
    ///
    /// With `sigma = 0` this degenerates to discounted intrinsic on the
    /// forward, `notional * DF(pay) * tau * max(F - K, 0)`.  All inputs must
    /// be finite (`notional` may be negative: a short position).
    pub fn caplet(&self, reset: f64, pay: f64, strike: f64, notional: f64) -> Result<f64> {
        let tau = Self::check_caplet_inputs("caplet", reset, pay, strike, notional)?;
        let k_bond = 1.0 / (1.0 + strike * tau);
        Ok(notional * (1.0 + strike * tau) * self.zbp(reset, pay, k_bond)?)
    }

    /// Floorlet: `notional (1 + K tau) ZBC(reset, pay, 1/(1 + K tau))`.
    pub fn floorlet(&self, reset: f64, pay: f64, strike: f64, notional: f64) -> Result<f64> {
        let tau = Self::check_caplet_inputs("floorlet", reset, pay, strike, notional)?;
        let k_bond = 1.0 / (1.0 + strike * tau);
        Ok(notional * (1.0 + strike * tau) * self.zbc(reset, pay, k_bond)?)
    }

    /// Cap = strip of caplets over consecutive schedule times.
    ///
    /// `schedule = [t0, t1, ..., tn]` (>= 2 finite, strictly increasing
    /// times, `t0 >= 0`) prices caplets on `[t0,t1], ..., [t_{n-1}, t_n]`.
    pub fn cap(&self, schedule: &[f64], strike: f64, notional: f64) -> Result<f64> {
        if schedule.len() < 2 {
            return Err(IrmError::InvalidInput(
                "cap schedule needs at least two times".into(),
            ));
        }
        if schedule.iter().any(|t| !t.is_finite()) {
            return Err(IrmError::InvalidInput(
                "cap schedule times must be finite".into(),
            ));
        }
        if schedule[0] < 0.0 || schedule.windows(2).any(|w| w[1] <= w[0]) {
            return Err(IrmError::InvalidInput(
                "cap schedule must be strictly increasing with t0 >= 0".into(),
            ));
        }
        check_finite("strike", strike)?;
        check_finite("notional", notional)?;
        let mut total = 0.0;
        for w in schedule.windows(2) {
            total += self.caplet(w[0], w[1], strike, notional)?;
        }
        Ok(total)
    }

    /// Par swap rate helper off the model's market curve.
    pub fn par_swap_rate(&self, times: &[f64]) -> Result<f64> {
        par_swap_rate(&self.curve, times)
    }

    // ----------------------- Jamshidian swaption ---------------------- //

    /// Validate a swaption schedule and return `(pay_times, coupons)`.
    fn swaption_coupons(
        expiry: f64,
        pay_times: &[f64],
        fixed_rate: f64,
        notional: f64,
    ) -> Result<Vec<f64>> {
        if !expiry.is_finite() || expiry <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "swaption expiry must be finite and > 0, got {expiry}"
            )));
        }
        if pay_times.is_empty() {
            return Err(IrmError::InvalidInput(
                "swaption needs at least one payment time".into(),
            ));
        }
        check_finite("fixed_rate", fixed_rate)?;
        check_finite("notional", notional)?;
        let mut prev = expiry;
        let mut coupons = Vec::with_capacity(pay_times.len());
        for &t in pay_times {
            if !t.is_finite() || t <= prev {
                return Err(IrmError::InvalidInput(
                    "payment times must be finite, strictly increasing and after expiry".into(),
                ));
            }
            coupons.push(fixed_rate * (t - prev));
            prev = t;
        }
        let last = coupons.len() - 1;
        coupons[last] += 1.0;
        if coupons[last] <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "jamshidian: degenerate coupon bond — 1 + X*tau_n must be > 0 (got {}); \
                 no r* exists",
                coupons[last]
            )));
        }
        Ok(coupons)
    }

    /// Jamshidian decomposition for a *given* critical rate `r*`.
    ///
    /// Enforces the contract of [`Self::jamshidian_swaption`] step 2: the
    /// residual `g(r*) = sum_i c_i P(T,u_i;r*) - 1` must satisfy
    /// `|g(r*)| < 1e-10` and the slope `g'(r*) = -sum_i c_i B_i K_i` must be
    /// negative (the crossing is on the decreasing branch of `g`);
    /// otherwise `IrmError::RootFind`.  Exposed so the contract is testable
    /// and so payer/receiver can share one root find.
    pub fn jamshidian_at_r_star(
        &self,
        expiry: f64,
        pay_times: &[f64],
        fixed_rate: f64,
        r_star: f64,
        notional: f64,
        payer: bool,
    ) -> Result<JamshidianResult> {
        let coupons = Self::swaption_coupons(expiry, pay_times, fixed_rate, notional)?;
        if !r_star.is_finite() {
            return Err(IrmError::InvalidInput(format!(
                "jamshidian: r* must be finite, got {r_star}"
            )));
        }
        let mut strikes = Vec::with_capacity(pay_times.len());
        for &t in pay_times {
            strikes.push(self.zcb_price(expiry, t, Some(r_star))?);
        }
        let residual = coupons
            .iter()
            .zip(strikes.iter())
            .map(|(c, k)| c * k)
            .sum::<f64>()
            - 1.0;
        if !(residual.abs() < JAMSHIDIAN_RESIDUAL_TOL) {
            return Err(IrmError::RootFind(format!(
                "jamshidian: r* residual {residual} exceeds {JAMSHIDIAN_RESIDUAL_TOL}"
            )));
        }
        let mut slope = 0.0;
        for ((c, &t), &k) in coupons.iter().zip(pay_times.iter()).zip(strikes.iter()) {
            slope -= c * self.b_factor(expiry, t)? * k;
        }
        if !(slope < 0.0) {
            return Err(IrmError::RootFind(format!(
                "jamshidian: coupon bond not decreasing at r* (slope {slope}); \
                 decomposition invalid"
            )));
        }
        let mut value = 0.0;
        for ((c, &t), &k) in coupons.iter().zip(pay_times.iter()).zip(strikes.iter()) {
            value += c * if payer {
                self.zbp(expiry, t, k)?
            } else {
                self.zbc(expiry, t, k)?
            };
        }
        Ok(JamshidianResult {
            value: notional * value,
            r_star,
            residual,
            strikes,
        })
    }

    /// European swaption via Jamshidian's decomposition.
    ///
    /// The underlying swap exchanges the fixed rate for the (single-curve)
    /// floating leg over `[expiry, pay_times]`; accruals are
    /// `tau_1 = t_1 - expiry`, `tau_i = t_i - t_{i-1}`.  A payer swaption's
    /// payoff at expiry is `max(1 - sum_i c_i P(T, t_i), 0)` with coupons
    /// `c_i = X tau_i` and `c_n = 1 + X tau_n` — a put with strike 1 on a
    /// coupon bond `g(r) = sum_i c_i P(T, t_i; r)`.
    ///
    /// Exactness needs `g(r) - 1` to change sign exactly once.  For `X >= 0`
    /// every coupon is positive and `g` is strictly decreasing (each
    /// `B > 0`).  For `X < 0` the coupon signs are `(-, ..., -, +)` when
    /// `1 + X tau_n > 0`: `g'` is an exponential sum with one sign change,
    /// so `g` has a single minimum, tends to `+inf` as `r -> -inf` and to
    /// `0^-` as `r -> +inf`, and still crosses 1 exactly once — the
    /// decomposition remains exact (verified against direct integration in
    /// the tests).  If `1 + X tau_n <= 0` every coupon is non-positive and
    /// no `r*` exists (`InvalidInput`, "degenerate").
    ///
    /// `r*` is found with the native Brent solver on an expanding bracket
    /// (start `[-1, 1]`, double the failing side, at most 24 doublings,
    /// never past the exp() overflow point), `xtol = 1e-15`; then
    /// `|g(r*)| < 1e-10` and `g'(r*) < 0` are enforced by
    /// [`Self::jamshidian_at_r_star`].  Setting `K_i = P(T, t_i; r*)`, the
    /// coupon-bond option splits *exactly* into ZCB options that are all
    /// exercised on the same event `{r_T > r*}`:
    ///
    /// * payer    = `sum_i c_i ZBP(T, t_i, K_i)`
    /// * receiver = `sum_i c_i ZBC(T, t_i, K_i)`
    pub fn jamshidian_swaption(
        &self,
        expiry: f64,
        pay_times: &[f64],
        fixed_rate: f64,
        notional: f64,
        payer: bool,
    ) -> Result<JamshidianResult> {
        let coupons = Self::swaption_coupons(expiry, pay_times, fixed_rate, notional)?;

        let bond_minus_one = |r: f64| -> Result<f64> {
            let mut s = 0.0;
            for (c, &t) in coupons.iter().zip(pay_times.iter()) {
                s += c * self.zcb_price(expiry, t, Some(r))?;
            }
            Ok(s - 1.0)
        };

        // Expanding bracket: g(lo) > 0 > g(hi) is needed.  g(lo) grows like
        // exp(B_n |lo|); refuse to evaluate past the exp() overflow point.
        let b_last = self.b_factor(expiry, pay_times[pay_times.len() - 1])?;
        let (mut lo, mut hi) = (-1.0_f64, 1.0_f64);
        let mut bracketed = false;
        for _ in 0..JAMSHIDIAN_MAX_DOUBLINGS {
            if b_last * (-lo) > MAX_EXP_ARG {
                break;
            }
            let g_lo = bond_minus_one(lo)?;
            let g_hi = bond_minus_one(hi)?;
            if g_lo > 0.0 && g_hi < 0.0 {
                bracketed = true;
                break;
            }
            if g_lo <= 0.0 {
                lo *= 2.0;
            }
            if g_hi >= 0.0 {
                hi *= 2.0;
            }
        }
        if !bracketed {
            return Err(IrmError::RootFind(
                "jamshidian: could not bracket r* (degenerate coupon bond — \
                 check fixed rate/schedule)"
                    .into(),
            ));
        }
        let r_star = brentq(bond_minus_one, lo, hi, 1e-15, BRENT_RTOL, BRENT_MAXITER)?;
        self.jamshidian_at_r_star(expiry, pay_times, fixed_rate, r_star, notional, payer)
    }

    // ----------------------- Monte Carlo ------------------------------ //

    /// Deterministic shift `alpha(t) = f^M(0,t)
    /// + (sigma^2/(2 a^2)) (1 - e^{-a t})^2` with `r(t) = x(t) + alpha(t)`.
    pub fn alpha(&self, t: f64) -> Result<f64> {
        let one_me = -(-self.a * t).exp_m1();
        Ok(self.fwd0(t)?
            + (self.sigma * self.sigma / (2.0 * self.a * self.a)) * one_me * one_me)
    }

    /// `V(t) = int_0^t (sigma^2/(2a^2))(1 - e^{-a s})^2 ds` so that
    /// `int_0^t alpha ds = -ln P^M(0,t) + V(t)`; closed form:
    ///
    /// `V(t) = (sigma^2/(2a^2)) [t - 2(1-e^{-a t})/a + (1-e^{-2 a t})/(2a)]`,
    /// which is half the integrated-OU variance kernel
    /// ([`ou_integral_variance`]), evaluated without cancellation.  `V(0) = 0`.
    pub fn variance_integral(&self, t: f64) -> Result<f64> {
        if t == 0.0 {
            return Ok(0.0);
        }
        Ok(0.5 * ou_integral_variance(self.a, self.sigma, t)?)
    }

    /// Unbiased MC caplet price and standard error.
    ///
    /// Simulates `x` to the reset date with exact OU joint moments of
    /// `(x, int x dt)`; the pathwise discount factor is exact:
    ///
    /// `D(0,T) = P^M(0,T) exp(-V(T) - int_0^T x dt)`.
    ///
    /// At reset, `P(T, pay)` is computed analytically from
    /// `r(T) = x(T) + alpha(T)`; the payoff (paid at `pay`, known at `T`)
    /// is valued as `D(0,T) P(T,pay) * notional * tau * max(F-K,0)`.
    /// No discretisation bias at any step count (`sigma = 0` gives the
    /// deterministic intrinsic and a standard error of exactly 0).  Same
    /// validation as [`Self::caplet`] plus `reset > 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn mc_caplet(
        &self,
        reset: f64,
        pay: f64,
        strike: f64,
        notional: f64,
        n_steps: usize,
        n_paths: usize,
        seed: u64,
    ) -> Result<(f64, f64)> {
        let tau = Self::check_caplet_inputs("mc_caplet", reset, pay, strike, notional)?;
        if reset <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "mc_caplet: need 0 < reset < pay, got {reset}, {pay}"
            )));
        }
        if n_steps < 1 || n_paths < 1 {
            return Err(IrmError::InvalidInput(
                "n_steps and n_paths must be >= 1".into(),
            ));
        }
        let dt = reset / n_steps as f64;
        let m = ou_step_moments(self.a, self.sigma, dt)?;
        let mut rng = StdRng::seed_from_u64(seed);
        let df_reset = self.curve.df(reset)?;
        let v_reset = self.variance_integral(reset)?;
        let alpha_reset = self.alpha(reset)?;
        let b = self.b_factor(reset, pay)?;
        let a_f = self.a_factor(reset, pay)?;
        let mut pv = Vec::with_capacity(n_paths);
        let (sd_x, beta, sd_resid) = if m.var_x > 0.0 {
            (
                m.var_x.sqrt(),
                m.cov / m.var_x,
                (m.var_i - m.cov * m.cov / m.var_x).max(0.0).sqrt(),
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        for _ in 0..n_paths {
            let mut x = 0.0_f64;
            let mut integral_x = 0.0_f64;
            if m.var_x > 0.0 {
                for _ in 0..n_steps {
                    let z1: f64 = rng.sample(StandardNormal);
                    let z2: f64 = rng.sample(StandardNormal);
                    let mean_x = x * m.decay;
                    let x_new = mean_x + sd_x * z1;
                    integral_x += x * m.b + beta * (x_new - mean_x) + sd_resid * z2;
                    x = x_new;
                }
            }
            // sigma = 0: x stays 0, integral stays 0 (deterministic replay).
            let disc = df_reset * (-v_reset - integral_x).exp();
            let r_reset = x + alpha_reset;
            let p_reset_pay = a_f * (-b * r_reset).exp();
            let fwd = (1.0 / p_reset_pay - 1.0) / tau;
            let payoff = notional * tau * (fwd - strike).max(0.0);
            pv.push(disc * p_reset_pay * payoff);
        }
        let price = pv.iter().sum::<f64>() / n_paths as f64;
        // sigma = 0 is fully deterministic: standard error exactly 0.
        let se = if n_paths > 1 && self.sigma > 0.0 {
            let var = pv.iter().map(|v| (v - price) * (v - price)).sum::<f64>()
                / (n_paths as f64 - 1.0);
            (var / n_paths as f64).sqrt()
        } else {
            0.0
        };
        Ok((price, se))
    }
}

// --------------------------------------------------------------------- //
// Calibration of (a, sigma) to caplet / swaption prices
// --------------------------------------------------------------------- //

/// Market caplet price per unit notional on `[reset, pay]` at `strike`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CapletQuote {
    /// Reset (fixing) time.
    pub reset: f64,
    /// Payment time.
    pub pay: f64,
    /// Strike rate.
    pub strike: f64,
    /// Market price per unit notional.
    pub price: f64,
}

impl CapletQuote {
    /// Validated constructor: finite fields, `0 <= reset < pay`,
    /// `1 + K tau > 0`, `price >= 0`.
    pub fn new(reset: f64, pay: f64, strike: f64, price: f64) -> Result<Self> {
        check_finite("reset", reset)?;
        check_finite("pay", pay)?;
        check_finite("strike", strike)?;
        check_finite("price", price)?;
        if reset < 0.0 || pay <= reset {
            return Err(IrmError::InvalidInput(format!(
                "caplet quote needs 0 <= reset < pay, got {reset}, {pay}"
            )));
        }
        if 1.0 + strike * (pay - reset) <= 0.0 {
            return Err(IrmError::InvalidInput(
                "caplet quote strike implies non-positive 1 + K*tau".into(),
            ));
        }
        if price < 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "caplet quote price must be >= 0, got {price}"
            )));
        }
        Ok(Self { reset, pay, strike, price })
    }
}

/// Market *payer* swaption price per unit notional.
#[derive(Debug, Clone, PartialEq)]
pub struct SwaptionQuote {
    /// Option expiry (swap start).
    pub expiry: f64,
    /// Fixed-leg payment times, strictly increasing and after expiry.
    pub pay_times: Vec<f64>,
    /// Fixed rate of the underlying swap.
    pub fixed_rate: f64,
    /// Market price per unit notional.
    pub price: f64,
}

impl SwaptionQuote {
    /// Validated constructor: finite fields, `expiry > 0`, a non-empty
    /// strictly increasing schedule after expiry, `price >= 0`.
    pub fn new(expiry: f64, pay_times: &[f64], fixed_rate: f64, price: f64) -> Result<Self> {
        check_finite("expiry", expiry)?;
        check_finite("fixed_rate", fixed_rate)?;
        check_finite("price", price)?;
        if expiry <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "swaption quote expiry must be > 0, got {expiry}"
            )));
        }
        if pay_times.is_empty() {
            return Err(IrmError::InvalidInput(
                "swaption quote needs at least one payment time".into(),
            ));
        }
        let mut prev = expiry;
        for &t in pay_times {
            if !t.is_finite() || t <= prev {
                return Err(IrmError::InvalidInput(
                    "swaption quote payment times must be finite, strictly increasing and \
                     after expiry"
                        .into(),
                ));
            }
            prev = t;
        }
        if price < 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "swaption quote price must be >= 0, got {price}"
            )));
        }
        Ok(Self {
            expiry,
            pay_times: pay_times.to_vec(),
            fixed_rate,
            price,
        })
    }
}

/// Outcome of [`calibrate_hullwhite`] (same diagnostics as
/// [`crate::vasicek::VasicekCalibration`]).
///
/// `rmse` is in price units per unit notional.  `converged` means the
/// optimiser converged at the best start *and* the solution is not at a
/// domain bound (`a < 1e-5` or `> 4.995`, `sigma < 1e-5` or `> 0.999`).
/// `identified` requires `n_starts >= 2` and `a_spread <= 1e-2`,
/// `sigma_spread <= 1e-4` over the starts whose rmse is within 5 % of the
/// best.  `a_fixed` is `true` when `a` was supplied, not fitted.
#[derive(Debug, Clone, PartialEq)]
pub struct HullWhiteCalibration {
    /// Fitted (or fixed) mean reversion.
    pub a: f64,
    /// Fitted volatility.
    pub sigma: f64,
    /// Root-mean-square price error per unit notional.
    pub rmse: f64,
    /// Optimiser iterations of the best start.
    pub iterations: usize,
    /// Optimiser converged and the solution is not at a bound.
    pub converged: bool,
    /// Solution at the edge of the domain.
    pub at_bound: bool,
    /// Number of starts run.
    pub n_starts: usize,
    /// Spread of `a` over equivalent starts.
    pub a_spread: f64,
    /// Spread of sigma over equivalent starts.
    pub sigma_spread: f64,
    /// Equivalent starts agree (multi-start only).
    pub identified: bool,
    /// `a` was held fixed at the supplied value.
    pub a_fixed: bool,
}

impl HullWhiteCalibration {
    /// Build the calibrated model on `curve`.
    pub fn model(&self, curve: DiscountCurve) -> Result<HullWhite> {
        HullWhite::new(self.a, self.sigma, curve)
    }
}

struct HwStartResult {
    fx: f64,
    a: f64,
    sigma: f64,
    iterations: usize,
    converged: bool,
}

/// Fit `(a, sigma)` to caplet and payer-swaption prices by least squares.
///
/// Minimises `sum_j (model_price_j - market_price_j)^2` over all quotes
/// (prices per unit notional) with Nelder-Mead on `(a, 10 sigma)`
/// (`initial_step = 0.05`, i.e. 0.005 in sigma); out-of-domain points
/// (`a <= 1e-6`, `a > 5`, `sigma < 0`, `sigma > 1`) get the penalty
/// `1e6 (1 + distance)`.
///
/// * `x0 = None`: three starts from [`HW_CALIBRATION_STARTS`], best
///   objective returned with spreads over equivalent starts.
/// * `x0 = Some([a, sigma])`: single warm start.
/// * `a_fixed = Some(a)`: fit sigma only (1-D), starting at the sigma values
///   of the default starts; `x0[0]` is ignored.
///
/// At least one quote is required; with both parameters free at least two
/// quotes of different expiry/tenor are needed for identifiability (a single
/// quote fits sigma for any `a` — `identified` will be `false`).
/// Non-convergence is reported, never an error.  Quotes are trusted as
/// given: convert broker normal vols with [`crate::bachelier::bachelier_price`]
/// first.  Errors: no quotes, non-finite `x0`, non-positive `a_fixed`,
/// `maxiter < 1`, or a quote the model cannot price (propagated).
pub fn calibrate_hullwhite(
    curve: &DiscountCurve,
    caplets: &[CapletQuote],
    swaptions: &[SwaptionQuote],
    a_fixed: Option<f64>,
    x0: Option<[f64; 2]>,
    maxiter: usize,
) -> Result<HullWhiteCalibration> {
    if caplets.is_empty() && swaptions.is_empty() {
        return Err(IrmError::InvalidInput(
            "calibrate_hullwhite needs at least one caplet or swaption quote".into(),
        ));
    }
    if maxiter < 1 {
        return Err(IrmError::InvalidInput("maxiter must be >= 1".into()));
    }
    if let Some(start) = x0 {
        if start.iter().any(|v| !v.is_finite()) {
            return Err(IrmError::InvalidInput("x0 must be finite".into()));
        }
    }
    if let Some(a) = a_fixed {
        if !a.is_finite() || a <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "a_fixed must be finite and > 0, got {a}"
            )));
        }
    }
    let (a_lo, a_hi, s_hi) = HW_DOMAIN;
    let fixed = a_fixed.is_some();
    let n_quotes = (caplets.len() + swaptions.len()) as f64;

    // The objective cannot fail for in-domain parameters and validated
    // quotes; a pricing error would be a bug, but never panic: it is
    // recorded and re-raised after the optimiser returns.
    let mut pricing_error: Option<IrmError> = None;
    let mut sse = |a: f64, sigma: f64| -> f64 {
        if a <= a_lo || a > a_hi || sigma < 0.0 || sigma > s_hi {
            return 1e6
                * (1.0
                    + (a_lo - a).max(0.0)
                    + (a - a_hi).max(0.0)
                    + (-sigma).max(0.0)
                    + (sigma - s_hi).max(0.0));
        }
        let inner = || -> Result<f64> {
            let hw = HullWhite::new(a, sigma, curve.clone())?;
            let mut total = 0.0;
            for q in caplets {
                let d = hw.caplet(q.reset, q.pay, q.strike, 1.0)? - q.price;
                total += d * d;
            }
            for q in swaptions {
                let d = hw
                    .jamshidian_swaption(q.expiry, &q.pay_times, q.fixed_rate, 1.0, true)?
                    .value
                    - q.price;
                total += d * d;
            }
            Ok(total)
        };
        match inner() {
            Ok(v) => v,
            Err(e) => {
                pricing_error = Some(e);
                f64::INFINITY
            }
        }
    };
    // Nelder-Mead uses one initial_step (0.05) for every coordinate, so the
    // optimiser works on sigma * SIGMA_SCALE (step 0.005 in sigma units).
    let starts: Vec<Vec<f64>> = match (fixed, x0) {
        (true, Some(st)) => vec![vec![st[1] * SIGMA_SCALE]],
        (true, None) => HW_CALIBRATION_STARTS
            .iter()
            .map(|&(_, s)| vec![s * SIGMA_SCALE])
            .collect(),
        (false, Some(st)) => vec![vec![st[0], st[1] * SIGMA_SCALE]],
        (false, None) => HW_CALIBRATION_STARTS
            .iter()
            .map(|&(a, s)| vec![a, s * SIGMA_SCALE])
            .collect(),
    };

    let mut results: Vec<HwStartResult> = Vec::with_capacity(starts.len());
    for st in &starts {
        let res = nelder_mead(
            |p: &[f64]| match a_fixed {
                Some(a) => sse(a, p[0] / SIGMA_SCALE),
                None => sse(p[0], p[1] / SIGMA_SCALE),
            },
            st,
            0.05,
            1e-10,
            1e-14,
            maxiter,
        )?;
        let (a_val, s_val) = match a_fixed {
            Some(a) => (a, res.x[0].abs() / SIGMA_SCALE),
            None => (res.x[0], res.x[1].abs() / SIGMA_SCALE),
        };
        results.push(HwStartResult {
            fx: res.fx,
            a: a_val,
            sigma: s_val,
            iterations: res.iterations,
            converged: res.converged,
        });
    }
    if let Some(e) = pricing_error {
        return Err(e);
    }
    let mut best = 0;
    for i in 1..results.len() {
        if results[i].fx < results[best].fx {
            best = i;
        }
    }
    let b = &results[best];
    let rmse = (b.fx.max(0.0) / n_quotes).sqrt();
    let at_bound = (!fixed && (b.a < A_LO_BOUND || b.a > A_HI_BOUND))
        || b.sigma < SIGMA_LO_BOUND
        || b.sigma > SIGMA_HI_BOUND;
    let (mut a_min, mut a_max) = (b.a, b.a);
    let (mut s_min, mut s_max) = (b.sigma, b.sigma);
    for r in &results {
        let r_rmse = (r.fx.max(0.0) / n_quotes).sqrt();
        if r_rmse <= EQUIV_RMSE_FACTOR * rmse + EQUIV_RMSE_FLOOR {
            a_min = a_min.min(r.a);
            a_max = a_max.max(r.a);
            s_min = s_min.min(r.sigma);
            s_max = s_max.max(r.sigma);
        }
    }
    let a_spread = a_max - a_min;
    let sigma_spread = s_max - s_min;
    let identified = results.len() >= 2 && a_spread <= IDENT_A && sigma_spread <= IDENT_SIGMA;
    Ok(HullWhiteCalibration {
        a: b.a,
        sigma: b.sigma,
        rmse,
        iterations: b.iterations,
        converged: b.converged && !at_bound,
        at_bound,
        n_starts: results.len(),
        a_spread,
        sigma_spread,
        identified,
        a_fixed: fixed,
    })
}
