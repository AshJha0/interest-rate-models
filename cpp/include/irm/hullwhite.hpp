#ifndef IRM_HULLWHITE_HPP
#define IRM_HULLWHITE_HPP

/// \file hullwhite.hpp
/// \brief Hull-White one-factor model `dr = (theta(t) - a r) dt + sigma dW`.
///
/// The model is fitted to an input DiscountCurve so that model ZCB prices
/// at `t = 0` reproduce the market discount factors *exactly*.  All
/// formulas use market DFs `P^M(0, .)` directly:
///
///  * `B(t,T) = (1 - e^{-a (T-t)}) / a`
///  * `A(t,T) = (P^M(0,T)/P^M(0,t))
///              * exp[B(t,T) f^M(0,t)
///                    - (sigma^2/(4a)) (1 - e^{-2 a t}) B(t,T)^2]`
///  * `P(t,T) = A(t,T) e^{-B(t,T) r(t)}`
///
/// `f^M(0,t)` is the market instantaneous forward, computed by the *exact*
/// finite-difference scheme documented in API_SPEC.md (step `h = 1e-5`,
/// central for `t >= h`, one-sided at the short end) so that every language
/// port reproduces bit-comparable values.
///
/// Options: ZBP/ZBC closed forms, caplets/floorlets as ZCB puts/calls, caps
/// as caplet strips, and Jamshidian's decomposition for European swaptions.
/// Monte Carlo simulates `r(t) = x(t) + alpha(t)` with exact Gaussian OU
/// moments (trinomial-tree-free),
/// `alpha(t) = f^M(0,t) + (sigma^2/(2a^2)) (1 - e^{-a t})^2`.
///
/// Calibration: calibrate_hullwhite() fits `(a, sigma)` (or sigma with `a`
/// fixed) to caplet / payer-swaption prices by least squares from several
/// Nelder-Mead starts; bachelier.hpp converts prices to normal (bp) vols.

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "irm/curve.hpp"

namespace irm {

/// Finite-difference step for the market instantaneous forward f(0,t).
constexpr double kFdStep = 1e-5;
/// The coupon-bond residual `|g(r*)|` must be below this after the root find.
constexpr double kJamshidianResidualTol = 1e-10;

/// Jamshidian swaption decomposition output.
struct JamshidianResult {
    double value;                  ///< notional * option value
    double r_star;                 ///< critical short rate at expiry
    double residual;               ///< coupon-bond residual at r_star
    std::vector<double> strikes;   ///< decomposition ZCB strikes K_i
};

/// Hull-White 1F model fitted to a market discount curve.
class HullWhite {
public:
    /// \throws std::invalid_argument unless a > 0 and sigma >= 0 (finite).
    HullWhite(double a, double sigma, DiscountCurve curve);

    double a() const { return a_; }
    double sigma() const { return sigma_; }
    const DiscountCurve& curve() const { return curve_; }

    /// Market instantaneous forward `f^M(0,t)` by finite difference.
    ///
    /// Exact scheme (h = kFdStep = 1e-5), reproduced identically by ports:
    ///  * `t >= h`: central,  `(ln DF(t-h) - ln DF(t+h)) / (2h)`
    ///  * `t <  h`: one-sided, `(ln DF(t) - ln DF(t+h)) / h`
    ///
    /// With piecewise-constant forwards the FD equals the segment forward
    /// away from pillars and the average of the two adjacent forwards when
    /// the stencil straddles a pillar.
    double fwd0(double t) const;

    /// Fitted drift `theta(t)` (documentation/testing helper):
    /// `theta(t) = df/dt + a f + (sigma^2/(2a)) (1 - e^{-2 a t})` with
    /// `df/dt` approximated by the forward difference
    /// `(fwd0(t+h) - fwd0(t))/h`.
    /// \throws std::invalid_argument unless h is finite and > 0.
    double theta(double t, double h = 1e-4) const;

    /// `B(t,T) = (1 - e^{-a (T-t)}) / a` (`expm1` form).
    /// \throws std::invalid_argument unless t and maturity are finite and
    ///   `0 <= t <= maturity`.
    double b_factor(double t, double maturity) const;

    /// `A(t,T)` built from market DFs and `f^M(0,t)` (see file docs).
    double a_factor(double t, double maturity) const;

    /// `P(t,T) = A(t,T) e^{-B(t,T) r}`.
    ///
    /// \p r defaults to `r(0) = f^M(0,0)`, under which `P(0,T)` equals the
    /// market DF exactly: the `e^{B f}` and `e^{-B r}` terms cancel.
    /// \throws std::invalid_argument on a non-finite r or when `exp(-B r)`
    ///   would overflow (result not representable).
    double zcb_price(double t, double maturity,
                     std::optional<double> r = std::nullopt) const;

    /// European call on a unit ZCB (uses market DFs — model-consistent).
    double zbc(double expiry, double bond_maturity, double strike) const;

    /// European put on a unit ZCB.
    double zbp(double expiry, double bond_maturity, double strike) const;

    /// Caplet on the simple forward over `[reset, pay]`.
    ///
    /// Pays `notional * tau * max(F(reset,pay) - K, 0)` at `pay`.  A caplet
    /// is `(1 + K tau)` ZCB *puts* struck at `1/(1 + K tau)`:
    /// `caplet = notional (1 + K tau) ZBP(reset, pay, 1/(1 + K tau))`.
    /// With `sigma = 0` this degenerates to discounted intrinsic on the
    /// forward, `notional * DF(pay) * tau * max(F - K, 0)`.
    /// \throws std::invalid_argument unless reset, pay, strike and notional
    ///   are finite, `0 <= reset < pay` and `1 + K tau > 0` (notional may be
    ///   negative: a short position).
    double caplet(double reset, double pay, double strike,
                  double notional = 1.0) const;

    /// Floorlet: `notional (1 + K tau) ZBC(reset, pay, 1/(1 + K tau))`.
    double floorlet(double reset, double pay, double strike,
                    double notional = 1.0) const;

    /// Cap = strip of caplets over consecutive schedule times.
    /// `schedule = [t0, t1, ..., tn]` (>= 2 finite, strictly increasing
    /// times, t0 >= 0) prices caplets on `[t0,t1], ..., [t_{n-1}, t_n]`.
    /// \throws std::invalid_argument on a bad schedule or non-finite
    ///   strike/notional.
    double cap(const std::vector<double>& schedule, double strike,
               double notional = 1.0) const;

    /// Par swap rate helper off the model's market curve.
    double par_swap_rate(const std::vector<double>& times) const;

    /// European swaption via Jamshidian's decomposition.
    ///
    /// The underlying swap exchanges the fixed rate for the (single-curve)
    /// floating leg over `[expiry, pay_times]`; accruals are
    /// `tau_1 = t_1 - expiry`, `tau_i = t_i - t_{i-1}`.  A payer swaption's
    /// payoff at expiry is `max(1 - sum_i c_i P(T, t_i), 0)` with coupons
    /// `c_i = X tau_i` and `c_n = 1 + X tau_n` — a put with strike 1 on a
    /// coupon bond.
    ///
    /// Exactness needs `g(r) - 1` (`g` the coupon bond) to change sign
    /// exactly once.  For `X >= 0` every coupon is positive and `g` is
    /// strictly decreasing (each `B > 0`).  For `X < 0` the coupon signs are
    /// `(-, ..., -, +)` when `1 + X tau_n > 0`: `g'` is an exponential sum
    /// with one sign change, so `g` has a single minimum, tends to +inf as
    /// r -> -inf and to 0^- as r -> +inf, and still crosses 1 exactly once —
    /// the decomposition remains exact (verified against direct integration
    /// in the tests).  If `1 + X tau_n <= 0` every coupon is non-positive
    /// and no r* exists (std::invalid_argument, "degenerate").
    ///
    /// `r*` is found with the native Brent solver on an expanding bracket
    /// (start `[-1, 1]`, double the failing side, at most 24 doublings,
    /// never past the exp() overflow point), `xtol = 1e-15`; then
    /// `|g(r*)| < 1e-10` and `g'(r*) < 0` are enforced by
    /// jamshidian_at_r_star().  Setting `K_i = P(T, t_i; r*)`, the
    /// coupon-bond option splits *exactly* into ZCB options that are all
    /// exercised on the same event {r_T > r*}:
    ///
    ///  * payer    = `sum_i c_i ZBP(T, t_i, K_i)`
    ///  * receiver = `sum_i c_i ZBC(T, t_i, K_i)`
    /// \throws std::invalid_argument on invalid inputs / degenerate coupons.
    /// \throws std::domain_error if r* cannot be bracketed or the residual /
    ///   slope contract fails.
    JamshidianResult jamshidian_swaption(double expiry,
                                         const std::vector<double>& pay_times,
                                         double fixed_rate,
                                         double notional = 1.0,
                                         bool payer = true) const;

    /// Jamshidian decomposition for a *given* critical rate `r*`.
    ///
    /// Enforces the contract of jamshidian_swaption() step 2: the residual
    /// `g(r*) = sum_i c_i P(T,u_i;r*) - 1` must satisfy `|g(r*)| < 1e-10`
    /// and the slope `g'(r*) = -sum_i c_i B_i K_i` must be negative.
    /// Exposed so the contract is testable and so payer/receiver can share
    /// one root find.
    /// \throws std::invalid_argument on invalid inputs or a non-finite r*.
    /// \throws std::domain_error if the residual or slope contract fails.
    JamshidianResult jamshidian_at_r_star(double expiry,
                                          const std::vector<double>& pay_times,
                                          double fixed_rate, double r_star,
                                          double notional = 1.0,
                                          bool payer = true) const;

    /// Deterministic shift `alpha(t) = f^M(0,t)
    /// + (sigma^2/(2 a^2)) (1 - e^{-a t})^2` with `r(t) = x(t) + alpha(t)`.
    double alpha(double t) const;

    /// Unbiased MC caplet price and standard error.
    ///
    /// Simulates `x` to the reset date with exact OU joint moments of
    /// `(x, int x dt)`; the pathwise discount factor is exact:
    /// `D(0,T) = P^M(0,T) exp(-V(T) - int_0^T x dt)`.  At reset,
    /// `P(T, pay)` is computed analytically from `r(T) = x(T) + alpha(T)`;
    /// the payoff (paid at `pay`, known at `T`) is valued as
    /// `D(0,T) P(T,pay) * notional * tau * max(F-K,0)`.  No discretisation
    /// bias at any step count (sigma = 0 gives the deterministic intrinsic
    /// and a standard error of exactly 0).  Same validation as caplet()
    /// plus `reset > 0`.
    std::pair<double, double> mc_caplet(double reset, double pay, double strike,
                                        double notional, int n_steps,
                                        int n_paths, std::uint64_t seed) const;

    /// `V(t) = int_0^t (sigma^2/(2a^2))(1 - e^{-a s})^2 ds` so that
    /// `int_0^t alpha ds = -ln P^M(0,t) + V(t)`; closed form
    /// `V(t) = (sigma^2/(2a^2)) [t - 2(1-e^{-a t})/a + (1-e^{-2 a t})/(2a)]`,
    /// which is half the integrated-OU variance kernel
    /// (ou_integral_variance()), evaluated without cancellation.  V(0) = 0.
    double variance_integral(double t) const;

private:
    double sigma_p(double expiry, double bond_maturity) const;
    double zb_option(double expiry, double bond_maturity, double strike,
                     bool call) const;
    /// Validate caplet-style inputs and return tau.
    static double check_caplet_inputs(const char* what, double reset, double pay,
                                      double strike, double notional);
    /// Validate a swaption schedule and return its coupons.
    void swaption_coupons(double expiry, const std::vector<double>& pay_times,
                          double fixed_rate, double notional,
                          std::vector<double>& coupons) const;

    double a_;
    double sigma_;
    DiscountCurve curve_;
};

// ---- calibration of (a, sigma) to caplet / swaption prices --------------

/// Market caplet price per unit notional on `[reset, pay]` at `strike`.
struct CapletQuote {
    double reset;
    double pay;
    double strike;
    double price;
    /// \throws std::invalid_argument on non-finite fields, `pay <= reset`,
    ///   `reset < 0`, `1 + K tau <= 0` or a negative price.
    CapletQuote(double reset, double pay, double strike, double price);
};

/// Market *payer* swaption price per unit notional.
struct SwaptionQuote {
    double expiry;
    std::vector<double> pay_times;
    double fixed_rate;
    double price;
    /// \throws std::invalid_argument on non-finite fields, `expiry <= 0`, an
    ///   empty / non-increasing schedule or a negative price.
    SwaptionQuote(double expiry, std::vector<double> pay_times, double fixed_rate,
                  double price);
};

/// Calibration domain: `a in (1e-6, 5]`, `sigma in [0, 1]`; outside gets the
/// penalty `1e6 (1 + distance)`.
constexpr double kHwALo = 1e-6;
constexpr double kHwAHi = 5.0;
constexpr double kHwSigmaHi = 1.0;

/// Outcome of calibrate_hullwhite() (same diagnostics as VasicekCalibration).
///
/// `rmse` is in price units per unit notional.  `converged` means the
/// optimiser converged at the best start *and* the solution is not at a
/// domain bound (`a < 1e-5` or `> 4.995`, `sigma < 1e-5` or `> 0.999`).
/// `identified` requires `n_starts >= 2` and `a_spread <= 1e-2`,
/// `sigma_spread <= 1e-4` over the starts whose rmse is within 5 % of the
/// best.  `a_fixed` is true when `a` was supplied, not fitted.
struct HullWhiteCalibration {
    double a;
    double sigma;
    double rmse;
    int iterations;
    bool converged;
    bool at_bound;
    int n_starts;
    double a_spread;
    double sigma_spread;
    bool identified;
    bool a_fixed;

    /// Build the calibrated model on \p curve.
    HullWhite model(DiscountCurve curve) const { return HullWhite(a, sigma, std::move(curve)); }
};

/// Fit `(a, sigma)` to caplet and payer-swaption prices by least squares.
///
/// Minimises `sum_j (model_price_j - market_price_j)^2` over all quotes
/// (prices per unit notional) with Nelder-Mead on `(a, 10 sigma)`
/// (`initial_step = 0.05`, i.e. 0.005 in sigma); out-of-domain points get
/// the penalty `1e6 (1 + distance)`.
///  * `x0` empty: three starts `(a, sigma) in {(0.03, 0.005), (0.1, 0.01),
///    (0.5, 0.02)}`, best objective returned with spreads over equivalent
///    starts.
///  * `x0 = {a, sigma}`: single warm start.
///  * `a_fixed`: fit sigma only (1-D), starting at the sigma values of the
///    default starts; `x0[0]` is ignored.
///
/// At least one quote is required; with both parameters free at least two
/// quotes of different expiry/tenor are needed for identifiability (a single
/// quote fits sigma for any `a` — `identified` will be false).
/// Non-convergence is reported, never thrown.  Quotes are trusted as given:
/// convert broker normal vols with bachelier_price() first.
/// \throws std::invalid_argument on no quotes, a wrong-length / non-finite
///   x0, non-positive a_fixed, or maxiter < 1.
HullWhiteCalibration calibrate_hullwhite(
    const DiscountCurve& curve, const std::vector<CapletQuote>& caplets,
    const std::vector<SwaptionQuote>& swaptions = {},
    std::optional<double> a_fixed = std::nullopt,
    std::optional<std::vector<double>> x0 = std::nullopt, int maxiter = 4000);

}  // namespace irm

#endif  // IRM_HULLWHITE_HPP
