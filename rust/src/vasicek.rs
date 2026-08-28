//! Vasicek short-rate model `dr = kappa (theta - r) dt + sigma dW`.
//!
//! Gaussian model: rates can go negative, which is a *feature* here
//! (EUR/JPY style curves) and the reason no lognormal short-rate model is
//! included in this project.
//!
//! Closed forms implemented (affine term structure):
//!
//! * `B(t,T) = (1 - e^{-kappa (T-t)}) / kappa`
//! * `A(t,T) = exp[(theta - sigma^2/(2 kappa^2)) (B(t,T) - (T-t))
//!   - sigma^2 B(t,T)^2 / (4 kappa)]`
//! * `P(t,T) = A(t,T) e^{-B(t,T) r(t)}`
//! * `r_T | r_t ~ N(theta + (r_t - theta) e^{-kappa (T-t)},
//!   sigma^2 (1 - e^{-2 kappa (T-t)}) / (2 kappa))`
//! * Jamshidian (1989) bond-option formula for European calls/puts on ZCBs.
//!
//! Monte Carlo uses the *exact* OU transition of the pair
//! `(r, int r dt)` (see [`crate::mathutils`]), so the ZCB estimator is
//! unbiased at any step count.  Calibration fits `(kappa, theta, sigma)`
//! to market zero yields by least squares with the native Nelder-Mead from
//! [`crate::optimize`].

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::error::{IrmError, Result};
use crate::mathutils::{norm_cdf, ou_step_moments};
use crate::optimize::nelder_mead;

/// Vasicek model with parameters `kappa > 0`, `sigma >= 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vasicek {
    kappa: f64,
    theta: f64,
    sigma: f64,
    r0: f64,
}

impl Vasicek {
    /// Validated constructor: all parameters finite, `kappa > 0`,
    /// `sigma >= 0`.
    pub fn new(kappa: f64, theta: f64, sigma: f64, r0: f64) -> Result<Self> {
        for (name, v) in [("kappa", kappa), ("theta", theta), ("sigma", sigma), ("r0", r0)] {
            if !v.is_finite() {
                return Err(IrmError::InvalidInput(format!(
                    "{name} must be finite, got {v}"
                )));
            }
        }
        if kappa <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "kappa must be > 0, got {kappa}"
            )));
        }
        if sigma < 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "sigma must be >= 0, got {sigma}"
            )));
        }
        Ok(Self { kappa, theta, sigma, r0 })
    }

    /// Mean-reversion speed.
    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Long-run mean level.
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Short-rate volatility.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Initial short rate.
    pub fn r0(&self) -> f64 {
        self.r0
    }

    // ----------------------------- affine functions ------------------- //

    /// `B(t,T)` with `tau = T - t`: duration of the ZCB w.r.t. `r`.
    ///
    /// `B = (1 - e^{-kappa tau})/kappa`; it is the sensitivity
    /// `-d ln P / d r` and appears in every option/moment formula.
    pub fn b_factor(&self, tau: f64) -> Result<f64> {
        if !tau.is_finite() || tau < 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "tau must be finite and >= 0, got {tau}"
            )));
        }
        Ok((1.0 - (-self.kappa * tau).exp()) / self.kappa)
    }

    /// `A(t,T)` with `tau = T - t` (depends on `tau` only):
    ///
    /// `A = exp[(theta - sigma^2/(2 kappa^2)) (B - tau) - sigma^2 B^2/(4 kappa)]`.
    ///
    /// The first term is the risk-adjusted drift contribution of the mean
    /// level; the second is the convexity from rate volatility.
    pub fn a_factor(&self, tau: f64) -> Result<f64> {
        let b = self.b_factor(tau)?;
        let (k, s) = (self.kappa, self.sigma);
        Ok(((self.theta - s * s / (2.0 * k * k)) * (b - tau) - s * s * b * b / (4.0 * k)).exp())
    }

    // ----------------------------- ZCB / yields ----------------------- //

    /// Zero-coupon bond price `P(t, maturity) = A e^{-B r(t)}`.
    ///
    /// `r = None` defaults to `r0` (only sensible for `t = 0`).
    pub fn zcb_price(&self, maturity: f64, t: f64, r: Option<f64>) -> Result<f64> {
        if maturity < t {
            return Err(IrmError::InvalidInput(format!(
                "maturity {maturity} before valuation time {t}"
            )));
        }
        let rr = r.unwrap_or(self.r0);
        let tau = maturity - t;
        Ok(self.a_factor(tau)? * (-self.b_factor(tau)? * rr).exp())
    }

    /// Continuously compounded zero yield `-ln P(t,T) / (T - t)`.
    pub fn zero_yield(&self, maturity: f64, t: f64, r: Option<f64>) -> Result<f64> {
        let tau = maturity - t;
        if tau <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "need maturity > t, got tau={tau}"
            )));
        }
        Ok(-self.zcb_price(maturity, t, r)?.ln() / tau)
    }

    // ----------------------------- r_T distribution ------------------- //

    /// `E[r_T | r_0] = theta + (r_0 - theta) e^{-kappa T}` (`r = None`
    /// starts from `r0`).
    pub fn r_mean(&self, horizon: f64, r: Option<f64>) -> Result<f64> {
        if !horizon.is_finite() || horizon < 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "horizon must be >= 0, got {horizon}"
            )));
        }
        let rr = r.unwrap_or(self.r0);
        Ok(self.theta + (rr - self.theta) * (-self.kappa * horizon).exp())
    }

    /// `Var[r_T | r_0] = sigma^2 (1 - e^{-2 kappa T}) / (2 kappa)`.
    pub fn r_var(&self, horizon: f64) -> Result<f64> {
        if !horizon.is_finite() || horizon < 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "horizon must be >= 0, got {horizon}"
            )));
        }
        Ok(self.sigma * self.sigma * (1.0 - (-2.0 * self.kappa * horizon).exp())
            / (2.0 * self.kappa))
    }

    // ----------------------------- bond options ----------------------- //

    /// Std dev of `ln P(T,S)` at option expiry `T` for bond maturity `S`:
    /// `sigma_p = sigma sqrt((1 - e^{-2 kappa T})/(2 kappa)) B(S - T)`.
    fn sigma_p(&self, expiry: f64, bond_maturity: f64) -> Result<f64> {
        Ok(self.sigma
            * ((1.0 - (-2.0 * self.kappa * expiry).exp()) / (2.0 * self.kappa)).sqrt()
            * self.b_factor(bond_maturity - expiry)?)
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
        let p_s = self.zcb_price(bond_maturity, 0.0, None)?;
        let p_t = self.zcb_price(expiry, 0.0, None)?;
        let sp = self.sigma_p(expiry, bond_maturity)?;
        if sp == 0.0 {
            // sigma = 0, expiry = 0, or S = T: option collapses to intrinsic
            // on the forward bond price.
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

    /// European call on a unit ZCB (Jamshidian 1989 closed form):
    /// `ZBC = P(0,S) N(h) - K P(0,T) N(h - sigma_p)` with
    /// `h = ln(P(0,S)/(K P(0,T)))/sigma_p + sigma_p/2`.
    pub fn zbc(&self, expiry: f64, bond_maturity: f64, strike: f64) -> Result<f64> {
        self.zb_option(expiry, bond_maturity, strike, true)
    }

    /// European put on a unit ZCB:
    /// `ZBP = K P(0,T) N(sigma_p - h) - P(0,S) N(-h)`.
    pub fn zbp(&self, expiry: f64, bond_maturity: f64, strike: f64) -> Result<f64> {
        self.zb_option(expiry, bond_maturity, strike, false)
    }

    // ----------------------------- Monte Carlo ------------------------ //

    /// Exact-transition simulation of `(r_T, int_0^T r dt)` per path.
    ///
    /// Writes `r = theta + x` with `x` a mean-zero OU process and samples
    /// `(x_next, int x dt)` per step from their exact joint Gaussian
    /// (conditional decomposition), so both outputs are free of
    /// discretisation bias.  Returns `(r_T paths, integral paths)`.
    pub fn simulate(
        &self,
        horizon: f64,
        n_steps: usize,
        n_paths: usize,
        seed: u64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        if !horizon.is_finite() || horizon <= 0.0 {
            return Err(IrmError::InvalidInput(format!(
                "horizon must be finite and > 0, got {horizon}"
            )));
        }
        if n_steps < 1 || n_paths < 1 {
            return Err(IrmError::InvalidInput(
                "n_steps and n_paths must be >= 1".into(),
            ));
        }
        let dt = horizon / n_steps as f64;
        let m = ou_step_moments(self.kappa, self.sigma, dt)?;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut r_end = Vec::with_capacity(n_paths);
        let mut integral_r = Vec::with_capacity(n_paths);
        let x0 = self.r0 - self.theta;
        if m.var_x > 0.0 {
            let sd_x = m.var_x.sqrt();
            let beta = m.cov / m.var_x; // regression of the integral on x_next
            let sd_resid = (m.var_i - m.cov * m.cov / m.var_x).max(0.0).sqrt();
            for _ in 0..n_paths {
                let mut x = x0;
                let mut integral_x = 0.0;
                for _ in 0..n_steps {
                    let z1: f64 = rng.sample(StandardNormal);
                    let z2: f64 = rng.sample(StandardNormal);
                    let mean_x = x * m.decay;
                    let x_new = mean_x + sd_x * z1;
                    integral_x += x * m.b + beta * (x_new - mean_x) + sd_resid * z2;
                    x = x_new;
                }
                r_end.push(self.theta + x);
                integral_r.push(self.theta * horizon + integral_x);
            }
        } else {
            // sigma = 0: fully deterministic decay.
            let mut x = x0;
            let mut integral_x = 0.0;
            for _ in 0..n_steps {
                integral_x += x * m.b;
                x *= m.decay;
            }
            r_end.resize(n_paths, self.theta + x);
            integral_r.resize(n_paths, self.theta * horizon + integral_x);
        }
        Ok((r_end, integral_r))
    }

    /// MC ZCB price `E[e^{-int r}]` and its standard error
    /// (`stdev/sqrt(n_paths)`, sample stdev with ddof 1).
    pub fn mc_zcb(
        &self,
        maturity: f64,
        n_steps: usize,
        n_paths: usize,
        seed: u64,
    ) -> Result<(f64, f64)> {
        let (_, integral) = self.simulate(maturity, n_steps, n_paths, seed)?;
        let disc: Vec<f64> = integral.iter().map(|i| (-i).exp()).collect();
        let price = disc.iter().sum::<f64>() / n_paths as f64;
        // sigma = 0 is fully deterministic: standard error exactly 0.
        let se = if n_paths > 1 && self.sigma > 0.0 {
            let var = disc.iter().map(|d| (d - price) * (d - price)).sum::<f64>()
                / (n_paths as f64 - 1.0);
            (var / n_paths as f64).sqrt()
        } else {
            0.0
        };
        Ok((price, se))
    }
}

/// Least-squares calibration outcome (non-convergence reported, not raised).
#[derive(Debug, Clone, PartialEq)]
pub struct VasicekCalibration {
    /// Fitted mean-reversion speed.
    pub kappa: f64,
    /// Fitted long-run mean.
    pub theta: f64,
    /// Fitted volatility (returned as `|sigma|`).
    pub sigma: f64,
    /// Fixed initial short rate.
    pub r0: f64,
    /// Root-mean-square yield error at the optimum.
    pub rmse: f64,
    /// Optimiser iterations.
    pub iterations: usize,
    /// Optimiser convergence flag (reported, never an error).
    pub converged: bool,
}

impl VasicekCalibration {
    /// Build the calibrated model.
    pub fn model(&self) -> Result<Vasicek> {
        Vasicek::new(self.kappa, self.theta, self.sigma, self.r0)
    }
}

/// Fit `(kappa, theta, sigma)` to market zero yields by least squares.
///
/// Minimises `sum_i (y_model(T_i) - y_i)^2` with Nelder-Mead; invalid
/// regions (`kappa <= 1e-6`, `sigma < 0`, `kappa > 50`, `|theta| > 5`,
/// `sigma > 5`) are handled with a smooth penalty so the simplex is pushed
/// back inside the domain.  `r0` is held fixed; start is
/// `(0.5, mean(y), 0.01)` with `initial_step = 0.05` unless `x0` is given.
pub fn calibrate_vasicek(
    maturities: &[f64],
    yields: &[f64],
    r0: f64,
    x0: Option<[f64; 3]>,
    maxiter: usize,
) -> Result<VasicekCalibration> {
    if maturities.len() != yields.len() || maturities.len() < 3 {
        return Err(IrmError::InvalidInput(
            "need at least 3 (maturity, yield) pairs of equal length".into(),
        ));
    }
    if maturities.iter().any(|t| !t.is_finite() || *t <= 0.0) {
        return Err(IrmError::InvalidInput(
            "maturities must be finite and > 0".into(),
        ));
    }
    if yields.iter().any(|y| !y.is_finite()) {
        return Err(IrmError::InvalidInput("yields must be finite".into()));
    }
    if !r0.is_finite() {
        return Err(IrmError::InvalidInput("r0 must be finite".into()));
    }

    let objective = |p: &[f64]| -> f64 {
        let (kappa, theta, sigma) = (p[0], p[1], p[2]);
        if kappa <= 1e-6 || sigma < 0.0 || kappa > 50.0 || theta.abs() > 5.0 || sigma > 5.0 {
            // Smooth penalty pointing back toward the feasible region.
            return 1e6
                * (1.0
                    + (1e-6 - kappa).max(0.0)
                    + (-sigma).max(0.0)
                    + (kappa - 50.0).max(0.0)
                    + (theta.abs() - 5.0).max(0.0)
                    + (sigma - 5.0).max(0.0));
        }
        // Parameters are in-domain here, so model construction and yield
        // evaluation (t > 0, validated above) cannot fail.
        let model = Vasicek::new(kappa, theta, sigma, r0).expect("in-domain parameters");
        maturities
            .iter()
            .zip(yields.iter())
            .map(|(&t, &y)| {
                let e = model.zero_yield(t, 0.0, None).expect("t > 0 validated") - y;
                e * e
            })
            .sum()
    };

    let start = x0.map(|a| a.to_vec()).unwrap_or_else(|| {
        vec![0.5, yields.iter().sum::<f64>() / yields.len() as f64, 0.01]
    });
    let res = nelder_mead(objective, &start, 0.05, 1e-10, 1e-14, maxiter)?;
    let rmse = (res.fx.max(0.0) / maturities.len() as f64).sqrt();
    Ok(VasicekCalibration {
        kappa: res.x[0],
        theta: res.x[1],
        sigma: res.x[2].abs(),
        r0,
        rmse,
        iterations: res.iterations,
        converged: res.converged,
    })
}
