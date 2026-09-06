#ifndef IRM_ROOTFIND_HPP
#define IRM_ROOTFIND_HPP

/// \file rootfind.hpp
/// \brief Native 1-D root finders: Brent's method and plain bisection.
///
/// Implemented from scratch (no external numerical library) per project
/// rules.  Brent's method combines bisection, secant and inverse quadratic
/// interpolation and is guaranteed to converge for a continuous function
/// with a sign change on the bracket, with superlinear speed on
/// well-behaved functions.

#include <functional>

namespace irm {

/// Machine-epsilon-scale relative tolerance (2 * eps, classic zbrent).
constexpr double kBrentRtol = 4.44e-16;

/// Find a root of \p f in `[a, b]` using Brent's method
/// (Numerical-Recipes `zbrent` variant).
///
/// `f(a)` and `f(b)` must have opposite signs (or one endpoint be an exact
/// root).  Per-iteration convergence test on the abscissa is
/// `|c - b|/2 <= 2*rtol*|b| + xtol/2`.
///
/// \throws std::invalid_argument on a non-finite bracket, non-positive
///   xtol, or an unbracketed root.
/// \throws std::domain_error if no convergence after \p maxiter iterations
///   or if \p f returns a non-finite value anywhere (an unconverged or NaN
///   root is never returned).
double brentq(const std::function<double(double)>& f, double a, double b,
              double xtol = 1e-14, double rtol = kBrentRtol, int maxiter = 100);

/// Plain bisection on `[a, b]`; robust fallback with linear convergence.
/// Same bracket contract as brentq(): unbracketed root / bad inputs are
/// std::invalid_argument, non-finite values and exhausting \p maxiter are
/// std::domain_error.
double bisect(const std::function<double(double)>& f, double a, double b,
              double xtol = 1e-12, int maxiter = 200);

}  // namespace irm

#endif  // IRM_ROOTFIND_HPP
