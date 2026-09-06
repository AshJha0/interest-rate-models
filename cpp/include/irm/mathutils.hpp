#ifndef IRM_MATHUTILS_HPP
#define IRM_MATHUTILS_HPP

/// \file mathutils.hpp
/// \brief Small numerical utilities shared by the models: the standard
/// normal CDF/PDF (via `std::erfc`) and the exact Ornstein-Uhlenbeck
/// transition moments used by both Monte Carlo engines, plus the
/// cancellation-free integrated-OU variance kernel shared with the
/// Hull-White pathwise discount factor.

#include <cmath>

namespace irm {

/// Standard normal CDF `N(x) = 0.5 * erfc(-x / sqrt(2))` (tail-accurate).
inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

/// Standard normal density `phi(x) = exp(-x^2/2) / sqrt(2 pi)`.
inline double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

/// Below this value of `x = a * dt` the integrated-OU variance uses its
/// power series instead of the closed form.
constexpr double kOuSeriesThreshold = 1e-2;

/// `Var[int_0^dt x(s) ds | x(0)]` for `dx = -a x dt + sigma dW`.
///
/// Closed form `(sigma^2/a^2)(dt - 2b + (1 - e^{-2 a dt})/(2a))` with
/// `b = (1 - e^{-a dt})/a`.  The three O(dt) terms cancel to an O(a^2 dt^3)
/// result, which loses all precision for small `a*dt` (80x too large at
/// `a*dt = 1e-4` in plain double arithmetic).  With `x = a*dt`:
///  * `x < kOuSeriesThreshold`: series
///    `sigma^2 dt^3 (1/3 - x/4 + 7x^2/60 - x^3/24 + 31x^4/2520 - x^5/320)`
///    (relative truncation error < 3e-15 at the threshold);
///  * otherwise the closed form written with `expm1` (relative rounding
///    error below ~1e-11 for `x >= 1e-2`).
/// This is also `2 V(t)` of the Hull-White model with `dt = t`.
/// \throws std::invalid_argument unless a > 0, dt > 0 (finite) and sigma finite.
double ou_integral_variance(double a, double sigma, double dt);

/// Exact one-step transition moments of a mean-zero OU process.
///
/// For `dx = -a x dt + sigma dW` and a step of length `dt`, conditional on
/// the current state `x` (writing `b = (1 - decay)/a`, `decay = e^{-a dt}`):
///  * `x_next ~ N(x * decay, var_x)` with `var_x = sigma^2 (1 - decay^2)/(2a)`
///  * `I = int x ds ~ N(x * b, var_i)` with
///    `var_i = (sigma^2/a^2)(dt - 2b + (1 - decay^2)/(2a))`
///  * `Cov(x_next, I) = (sigma^2/(2a^2)) (1 - decay)^2`
///
/// `1 - decay` terms use `expm1` and `var_i` uses ou_integral_variance(), so
/// the moments stay accurate down to `a*dt ~ 1e-12`.
struct OUStepMoments {
    double decay;  ///< exp(-a dt)
    double b;      ///< (1 - decay) / a
    double var_x;  ///< variance of x_next given x
    double var_i;  ///< variance of the step integral given x
    double cov;    ///< covariance of (x_next, integral)
};

/// Compute the moments for mean reversion `a > 0`, `sigma >= 0`, step `dt > 0`.
/// \throws std::invalid_argument otherwise (including non-finite inputs).
OUStepMoments ou_step_moments(double a, double sigma, double dt);

}  // namespace irm

#endif  // IRM_MATHUTILS_HPP
