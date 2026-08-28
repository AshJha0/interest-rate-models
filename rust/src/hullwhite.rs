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

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::curve::{par_swap_rate, DiscountCurve};
use crate::error::{IrmError, Result};
use crate::mathutils::{norm_cdf, ou_step_moments};
use crate::rootfind::{brentq, BRENT_MAXITER, BRENT_RTOL};

/// Finite-difference step for the market instantaneous forward `f(0,t)`.
pub const FD_STEP: f64 = 1e-5;

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
    /// `(fwd0(t+h') - fwd0(t))/h'`, `h' = 1e-4`.
    pub fn theta(&self, t: f64) -> Result<f64> {
        let h = 1e-4;
        let dfdt = (self.fwd0(t + h)? - self.fwd0(t)?) / h;
        Ok(dfdt
            + self.a * self.fwd0(t)?
            + (self.sigma * self.sigma / (2.0 * self.a))
                * (1.0 - (-2.0 * self.a * t).exp()))
    }

    // ----------------------- affine ZCB ------------------------------- //

    /// `B(t,T) = (1 - e^{-a (T-t)}) / a`.
    pub fn b_factor(&self, t: f64, maturity: f64) -> Result<f64> {
        if maturity < t {
            return Err(IrmError::InvalidInput(format!(
                "maturity {maturity} before t {t}"
            )));
        }
        Ok((1.0 - (-self.a * (maturity - t)).exp()) / self.a)
    }

    /// `A(t,T)` built from market DFs and `f^M(0,t)` (see module docs).
    pub fn a_factor(&self, t: f64, maturity: f64) -> Result<f64> {
        let b = self.b_factor(t, maturity)?;
        let pm_t = self.curve.df(t)?;
        let pm_mat = self.curve.df(maturity)?;
        let conv = self.sigma * self.sigma / (4.0 * self.a)
            * (1.0 - (-2.0 * self.a * t).exp())
            * b
            * b;
        Ok((pm_mat / pm_t) * (b * self.fwd0(t)? - conv).exp())
    }

    /// `P(t,T) = A(t,T) e^{-B(t,T) r}`.
    ///
    /// `r = None` defaults to `r(0) = f^M(0,0)` (so `P(0,T)` equals the
    /// market DF exactly: the `e^{B f}` and `e^{-B r}` terms cancel).
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
        Ok(self.a_factor(t, maturity)? * (-self.b_factor(t, maturity)? * rr).exp())
    }

    // ----------------------- ZCB options ------------------------------ //

    /// Std dev of `ln P` at expiry:
    /// `sigma_p = sigma sqrt((1 - e^{-2 a T})/(2a)) B(T,S)`.
    fn sigma_p(&self, expiry: f64, bond_maturity: f64) -> Result<f64> {
        Ok(self.sigma
            * ((1.0 - (-2.0 * self.a * expiry).exp()) / (2.0 * self.a)).sqrt()
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

    /// Caplet on the simple forward over `[reset, pay]`.
    ///
    /// Pays `notional * tau * max(F(reset,pay) - K, 0)` at `pay`.  A caplet
    /// is `(1 + K tau)` ZCB *puts* struck at `1/(1 + K tau)`:
    ///
    /// `caplet = notional (1 + K tau) ZBP(reset, pay, 1/(1 + K tau))`.
    ///
    /// With `sigma = 0` this degenerates to discounted intrinsic on the
    /// forward, `notional * DF(pay) * tau * max(F - K, 0)`.
    pub fn caplet(&self, reset: f64, pay: f64, strike: f64, notional: f64) -> Result<f64> {
        let tau = pay - reset;
        if tau <= 0.0 || reset < 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "need 0 <= reset < pay, got {reset}, {pay}"
            )));
        }
        if !strike.is_finite() || 1.0 + strike * tau <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "caplet strike implies non-positive 1 + K*tau ({strike})"
            )));
        }
        let k_bond = 1.0 / (1.0 + strike * tau);
        Ok(notional * (1.0 + strike * tau) * self.zbp(reset, pay, k_bond)?)
    }

    /// Floorlet: `notional (1 + K tau) ZBC(reset, pay, 1/(1 + K tau))`.
    pub fn floorlet(&self, reset: f64, pay: f64, strike: f64, notional: f64) -> Result<f64> {
        let tau = pay - reset;
        if tau <= 0.0 || reset < 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "need 0 <= reset < pay, got {reset}, {pay}"
            )));
        }
        if !strike.is_finite() || 1.0 + strike * tau <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "floorlet strike implies non-positive 1 + K*tau ({strike})"
            )));
        }
        let k_bond = 1.0 / (1.0 + strike * tau);
        Ok(notional * (1.0 + strike * tau) * self.zbc(reset, pay, k_bond)?)
    }

    /// Cap = strip of caplets over consecutive schedule times.
    ///
    /// `schedule = [t0, t1, ..., tn]` prices caplets on
    /// `[t0,t1], ..., [t_{n-1}, t_n]` (first reset `t0 >= 0`).
    pub fn cap(&self, schedule: &[f64], strike: f64, notional: f64) -> Result<f64> {
        if schedule.len() < 2 {
            return Err(IrmError::InvalidInput(
                "cap schedule needs at least two times".into(),
            ));
        }
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

    /// European swaption via Jamshidian's decomposition.
    ///
    /// The underlying swap exchanges the fixed rate for the (single-curve)
    /// floating leg over `[expiry, pay_times]`; accruals are
    /// `tau_1 = t_1 - expiry`, `tau_i = t_i - t_{i-1}`.  A payer swaption's
    /// payoff at expiry is `max(1 - sum_i c_i P(T, t_i), 0)` with coupons
    /// `c_i = X tau_i` and `c_n = 1 + X tau_n` — a put with strike 1 on a
    /// coupon bond.
    ///
    /// Because `P(T, t; r)` is strictly decreasing in `r` (`B > 0`), the
    /// coupon bond price is monotone in `r`, so there is a unique `r*` with
    /// `sum_i c_i P(T, t_i; r*) = 1` (found with the native Brent solver on
    /// an expanding bracket).  Setting `K_i = P(T, t_i; r*)`, the
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
        if !expiry.is_finite() || expiry <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "swaption expiry must be > 0, got {expiry}"
            )));
        }
        if pay_times.is_empty() {
            return Err(IrmError::InvalidInput(
                "swaption needs at least one payment time".into(),
            ));
        }
        if !fixed_rate.is_finite() {
            return Err(IrmError::InvalidInput("fixed rate must be finite".into()));
        }
        let mut prev = expiry;
        let mut taus = Vec::with_capacity(pay_times.len());
        for &t in pay_times {
            if t <= prev {
                return Err(IrmError::InvalidInput(
                    "payment times must be strictly increasing and after expiry".into(),
                ));
            }
            taus.push(t - prev);
            prev = t;
        }
        let mut coupons: Vec<f64> = taus.iter().map(|tau| fixed_rate * tau).collect();
        *coupons.last_mut().expect("non-empty") += 1.0;

        let bond_minus_one = |r: f64| -> Result<f64> {
            let mut s = 0.0;
            for (c, &t) in coupons.iter().zip(pay_times.iter()) {
                s += c * self.zcb_price(expiry, t, Some(r))?;
            }
            Ok(s - 1.0)
        };

        // Expanding bracket: g is strictly decreasing in r.
        let (mut lo, mut hi) = (-1.0_f64, 1.0_f64);
        let mut bracketed = false;
        for _ in 0..24 {
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
        let residual = bond_minus_one(r_star)?;
        if residual.abs() >= 1e-10 {
            return Err(IrmError::RootFind(format!(
                "jamshidian: r* residual {residual} exceeds 1e-10"
            )));
        }
        let mut strikes = Vec::with_capacity(pay_times.len());
        for &t in pay_times {
            strikes.push(self.zcb_price(expiry, t, Some(r_star))?);
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

    // ----------------------- Monte Carlo ------------------------------ //

    /// Deterministic shift `alpha(t) = f^M(0,t)
    /// + (sigma^2/(2 a^2)) (1 - e^{-a t})^2` with `r(t) = x(t) + alpha(t)`.
    pub fn alpha(&self, t: f64) -> Result<f64> {
        let one_me = 1.0 - (-self.a * t).exp();
        Ok(self.fwd0(t)?
            + (self.sigma * self.sigma / (2.0 * self.a * self.a)) * one_me * one_me)
    }

    /// `V(t) = int_0^t (sigma^2/(2a^2))(1 - e^{-a s})^2 ds` so that
    /// `int_0^t alpha ds = -ln P^M(0,t) + V(t)`; closed form:
    ///
    /// `V(t) = (sigma^2/(2a^2)) [t - 2(1-e^{-a t})/a + (1-e^{-2 a t})/(2a)]`.
    fn variance_integral(&self, t: f64) -> f64 {
        let a = self.a;
        (self.sigma * self.sigma / (2.0 * a * a))
            * (t - 2.0 * (1.0 - (-a * t).exp()) / a + (1.0 - (-2.0 * a * t).exp()) / (2.0 * a))
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
    /// deterministic intrinsic).
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
        let tau = pay - reset;
        if tau <= 0.0 || reset <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "need 0 < reset < pay, got {reset}, {pay}"
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
        let v_reset = self.variance_integral(reset);
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
