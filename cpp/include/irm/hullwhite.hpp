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

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "irm/curve.hpp"

namespace irm {

/// Finite-difference step for the market instantaneous forward f(0,t).
constexpr double kFdStep = 1e-5;

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
    double theta(double t, double h = 1e-4) const;

    /// `B(t,T) = (1 - e^{-a (T-t)}) / a`.
    double b_factor(double t, double maturity) const;

    /// `A(t,T)` built from market DFs and `f^M(0,t)` (see file docs).
    double a_factor(double t, double maturity) const;

    /// `P(t,T) = A(t,T) e^{-B(t,T) r}`.
    ///
    /// \p r defaults to `r(0) = f^M(0,0)`, under which `P(0,T)` equals the
    /// market DF exactly: the `e^{B f}` and `e^{-B r}` terms cancel.
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
    double caplet(double reset, double pay, double strike,
                  double notional = 1.0) const;

    /// Floorlet: `notional (1 + K tau) ZBC(reset, pay, 1/(1 + K tau))`.
    double floorlet(double reset, double pay, double strike,
                    double notional = 1.0) const;

    /// Cap = strip of caplets over consecutive schedule times.
    /// `schedule = [t0, t1, ..., tn]` prices caplets on
    /// `[t0,t1], ..., [t_{n-1}, t_n]` (first reset t0 >= 0).
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
    /// Because P(T, t; r) is strictly decreasing in r (B > 0), the coupon
    /// bond price is monotone in r, so there is a unique `r*` with
    /// `sum_i c_i P(T, t_i; r*) = 1` (found with the native Brent solver on
    /// an expanding bracket).  Setting `K_i = P(T, t_i; r*)`, the
    /// coupon-bond option splits *exactly* into ZCB options that are all
    /// exercised on the same event {r_T > r*}:
    ///
    ///  * payer    = `sum_i c_i ZBP(T, t_i, K_i)`
    ///  * receiver = `sum_i c_i ZBC(T, t_i, K_i)`
    JamshidianResult jamshidian_swaption(double expiry,
                                         const std::vector<double>& pay_times,
                                         double fixed_rate,
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
    /// bias at any step count (sigma = 0 gives the deterministic intrinsic).
    std::pair<double, double> mc_caplet(double reset, double pay, double strike,
                                        double notional, int n_steps,
                                        int n_paths, std::uint64_t seed) const;

private:
    double sigma_p(double expiry, double bond_maturity) const;
    double zb_option(double expiry, double bond_maturity, double strike,
                     bool call) const;
    /// `V(t) = int_0^t (sigma^2/(2a^2))(1 - e^{-a s})^2 ds` so that
    /// `int_0^t alpha ds = -ln P^M(0,t) + V(t)`; closed form:
    /// `V(t) = (sigma^2/(2a^2)) [t - 2(1-e^{-a t})/a + (1-e^{-2 a t})/(2a)]`.
    double variance_integral(double t) const;

    double a_;
    double sigma_;
    DiscountCurve curve_;
};

}  // namespace irm

#endif  // IRM_HULLWHITE_HPP
