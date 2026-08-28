#include "irm/mathutils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace irm {

OUStepMoments ou_step_moments(double a, double sigma, double dt) {
    if (!(a > 0.0)) {
        throw std::invalid_argument("ou_step_moments: mean reversion must be positive, got " +
                                    std::to_string(a));
    }
    if (!(dt > 0.0)) {
        throw std::invalid_argument("ou_step_moments: time step must be positive, got " +
                                    std::to_string(dt));
    }
    const double decay = std::exp(-a * dt);
    const double b = (1.0 - decay) / a;
    const double s2 = sigma * sigma;
    double var_x = s2 * (1.0 - decay * decay) / (2.0 * a);
    double var_i = (s2 / (a * a)) * (dt - 2.0 * b + (1.0 - decay * decay) / (2.0 * a));
    const double cov = (s2 / (2.0 * a * a)) * (1.0 - decay) * (1.0 - decay);
    // Guard tiny negative values from floating-point cancellation.
    var_x = std::max(var_x, 0.0);
    var_i = std::max(var_i, 0.0);
    return OUStepMoments{decay, b, var_x, var_i, cov};
}

}  // namespace irm
