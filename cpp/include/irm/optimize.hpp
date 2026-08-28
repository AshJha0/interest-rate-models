#ifndef IRM_OPTIMIZE_HPP
#define IRM_OPTIMIZE_HPP

/// \file optimize.hpp
/// \brief Native Nelder-Mead simplex minimiser (P02-style API).
///
/// Used by the Vasicek least-squares calibration.  Standard coefficients:
/// reflection 1, expansion 2, contraction 0.5, shrink 0.5.  Non-convergence
/// is *reported* via `converged = false` in the result, never thrown —
/// callers decide how to react (per the shared conventions: report, don't
/// crash).

#include <functional>
#include <vector>

namespace irm {

/// Outcome of a Nelder-Mead minimisation.
struct NelderMeadResult {
    std::vector<double> x;  ///< best vertex found
    double fx;              ///< objective value at x
    int iterations;         ///< iterations performed
    bool converged;         ///< tolerance criteria met before maxiter
};

/// Minimise \p f starting from \p x0.
///
/// The initial simplex is `x0` plus \p initial_step added to each
/// coordinate in turn.  Converged when both the simplex diameter is below
/// \p xtol and the function-value spread below \p ftol.
/// \throws std::invalid_argument on an empty or non-finite \p x0.
NelderMeadResult nelder_mead(
    const std::function<double(const std::vector<double>&)>& f,
    const std::vector<double>& x0, double initial_step = 0.1,
    double xtol = 1e-10, double ftol = 1e-14, int maxiter = 5000);

}  // namespace irm

#endif  // IRM_OPTIMIZE_HPP
