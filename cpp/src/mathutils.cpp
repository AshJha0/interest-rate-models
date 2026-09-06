#include "irm/mathutils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace irm {

namespace {

// Coefficients of  x - 2(1 - e^{-x}) + (1 - e^{-2x})/2  =  x^3 * sum_k c_k x^k,
// c_k = (-1)^(k+3) (2 - 2^(k+2)) / (k+3)!  for k = 0, 1, ...
constexpr double kOuSeries[] = {1.0 / 3.0,   -1.0 / 4.0,     7.0 / 60.0,
                                -1.0 / 24.0, 31.0 / 2520.0, -1.0 / 320.0};

void check_ou_inputs(double a, double sigma, double dt) {
    if (!(a > 0.0) || !std::isfinite(a)) {
        throw std::invalid_argument(
            "ou_step_moments: mean reversion must be positive and finite, got " +
            std::to_string(a));
    }
    if (!(dt > 0.0) || !std::isfinite(dt)) {
        throw std::invalid_argument("ou_step_moments: time step must be positive and finite, got " +
                                    std::to_string(dt));
    }
    if (!std::isfinite(sigma)) {
        throw std::invalid_argument("ou_step_moments: sigma must be finite, got " +
                                    std::to_string(sigma));
    }
}

}  // namespace

double ou_integral_variance(double a, double sigma, double dt) {
    check_ou_inputs(a, sigma, dt);
    const double x = a * dt;
    const double s2 = sigma * sigma;
    if (x < kOuSeriesThreshold) {
        double poly = 0.0;
        for (int k = 5; k >= 0; --k) poly = poly * x + kOuSeries[k];
        return s2 * dt * dt * dt * poly;
    }
    const double bracket = dt + 2.0 * std::expm1(-x) / a + (-std::expm1(-2.0 * x)) / (2.0 * a);
    return std::max((s2 / (a * a)) * bracket, 0.0);
}

OUStepMoments ou_step_moments(double a, double sigma, double dt) {
    check_ou_inputs(a, sigma, dt);
    if (sigma < 0.0) {
        throw std::invalid_argument("ou_step_moments: sigma must be >= 0, got " +
                                    std::to_string(sigma));
    }
    const double x = a * dt;
    const double decay = std::exp(-x);
    const double one_minus_decay = -std::expm1(-x);
    const double b = one_minus_decay / a;
    const double s2 = sigma * sigma;
    const double var_x = s2 * (-std::expm1(-2.0 * x)) / (2.0 * a);
    const double var_i = ou_integral_variance(a, sigma, dt);
    const double cov = (s2 / (2.0 * a * a)) * one_minus_decay * one_minus_decay;
    return OUStepMoments{decay, b, var_x, var_i, cov};
}

}  // namespace irm
