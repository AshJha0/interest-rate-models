#ifndef IRM_MATHUTILS_HPP
#define IRM_MATHUTILS_HPP

/// \file mathutils.hpp
/// \brief Small numerical utilities shared by the models: the standard
/// normal CDF and the exact Ornstein-Uhlenbeck transition moments used by
/// both the Vasicek and the Hull-White Monte Carlo engines.

#include <cmath>

namespace irm {

/// Standard normal CDF \f$ N(x) = \tfrac12 \operatorname{erfc}(-x/\sqrt2) \f$.
///
/// `erfc` is used instead of `erf` for better relative accuracy in the far
/// tails (large negative \p x).
inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

/// Exact one-step transition moments of a mean-zero OU process.
///
/// For `dx = -a x dt + sigma dW` and a step of length `dt`, conditional on
/// the current state `x` (writing `b = (1 - decay)/a`):
///
///   * `x_next ~ N(x * decay, var_x)` with `decay = exp(-a dt)`
///   * `I = int_t^{t+dt} x(s) ds ~ N(x * b, var_i)`
///   * `Cov(x_next, I) = cov`
///
/// where
///
///   * `var_x = sigma^2 (1 - decay^2) / (2a)`
///   * `var_i = (sigma^2/a^2) (dt - 2 b + (1 - decay^2)/(2a))`
///   * `cov   = (sigma^2/(2 a^2)) (1 - decay)^2`
///
/// Sampling the pair `(x_next, I)` from this bivariate Gaussian gives an
/// *exact* (bias-free) discretisation of both the short rate and its time
/// integral, which is what makes the MC zero-coupon-bond estimators
/// unbiased at any step size.
struct OUStepMoments {
    double decay;  ///< exp(-a dt)
    double b;      ///< (1 - decay) / a
    double var_x;  ///< variance of x_next given x
    double var_i;  ///< variance of the step integral given x
    double cov;    ///< covariance of (x_next, integral)
};

/// Compute OUStepMoments for mean-reversion `a > 0`, `dt > 0`.
/// \throws std::invalid_argument on non-positive `a` or `dt`.
OUStepMoments ou_step_moments(double a, double sigma, double dt);

}  // namespace irm

#endif  // IRM_MATHUTILS_HPP
