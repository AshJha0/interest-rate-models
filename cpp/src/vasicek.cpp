#include "irm/vasicek.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

#include "irm/mathutils.hpp"
#include "irm/optimize.hpp"

namespace irm {

Vasicek::Vasicek(double kappa, double theta, double sigma, double r0)
    : kappa_(kappa), theta_(theta), sigma_(sigma), r0_(r0) {
    const struct { const char* name; double v; } params[] = {
        {"kappa", kappa}, {"theta", theta}, {"sigma", sigma}, {"r0", r0}};
    for (const auto& p : params) {
        if (!std::isfinite(p.v)) {
            throw std::invalid_argument(std::string(p.name) + " must be finite, got " +
                                        std::to_string(p.v));
        }
    }
    if (kappa <= 0.0) {
        throw std::invalid_argument("kappa must be > 0, got " + std::to_string(kappa));
    }
    if (sigma < 0.0) {
        throw std::invalid_argument("sigma must be >= 0, got " + std::to_string(sigma));
    }
}

double Vasicek::b_factor(double tau) const {
    if (!std::isfinite(tau) || tau < 0.0) {
        throw std::invalid_argument("tau must be finite and >= 0, got " + std::to_string(tau));
    }
    return (1.0 - std::exp(-kappa_ * tau)) / kappa_;
}

double Vasicek::a_factor(double tau) const {
    const double b = b_factor(tau);
    const double k = kappa_, s = sigma_;
    return std::exp((theta_ - s * s / (2.0 * k * k)) * (b - tau) - s * s * b * b / (4.0 * k));
}

double Vasicek::zcb_price(double maturity, double t, std::optional<double> r) const {
    if (maturity < t) {
        throw std::invalid_argument("maturity " + std::to_string(maturity) +
                                    " before valuation time " + std::to_string(t));
    }
    const double rr = r.value_or(r0_);
    const double tau = maturity - t;
    return a_factor(tau) * std::exp(-b_factor(tau) * rr);
}

double Vasicek::zero_yield(double maturity, double t, std::optional<double> r) const {
    const double tau = maturity - t;
    if (tau <= 0.0) {
        throw std::invalid_argument("need maturity > t, got tau=" + std::to_string(tau));
    }
    return -std::log(zcb_price(maturity, t, r)) / tau;
}

double Vasicek::r_mean(double horizon, std::optional<double> r) const {
    if (!std::isfinite(horizon) || horizon < 0.0) {
        throw std::invalid_argument("horizon must be >= 0, got " + std::to_string(horizon));
    }
    const double rr = r.value_or(r0_);
    return theta_ + (rr - theta_) * std::exp(-kappa_ * horizon);
}

double Vasicek::r_var(double horizon) const {
    if (!std::isfinite(horizon) || horizon < 0.0) {
        throw std::invalid_argument("horizon must be >= 0, got " + std::to_string(horizon));
    }
    return sigma_ * sigma_ * (1.0 - std::exp(-2.0 * kappa_ * horizon)) / (2.0 * kappa_);
}

double Vasicek::sigma_p(double expiry, double bond_maturity) const {
    // Std dev of ln P(T,S) at option expiry T for bond maturity S.
    return sigma_ * std::sqrt((1.0 - std::exp(-2.0 * kappa_ * expiry)) / (2.0 * kappa_)) *
           b_factor(bond_maturity - expiry);
}

double Vasicek::zb_option(double expiry, double bond_maturity, double strike, bool call) const {
    if (!(std::isfinite(expiry) && std::isfinite(bond_maturity) && std::isfinite(strike))) {
        throw std::invalid_argument("option inputs must be finite");
    }
    if (expiry < 0.0 || bond_maturity < expiry) {
        throw std::invalid_argument("need 0 <= expiry <= bond maturity, got " +
                                    std::to_string(expiry) + ", " + std::to_string(bond_maturity));
    }
    if (strike <= 0.0) {
        throw std::invalid_argument("strike must be > 0, got " + std::to_string(strike));
    }
    const double p_s = zcb_price(bond_maturity);
    const double p_t = zcb_price(expiry);
    const double sp = sigma_p(expiry, bond_maturity);
    if (sp == 0.0) {
        // sigma = 0, expiry = 0, or S = T: option collapses to intrinsic on
        // the forward bond price.
        double intrinsic = p_s - strike * p_t;
        if (!call) intrinsic = -intrinsic;
        return std::max(intrinsic, 0.0);
    }
    const double h = std::log(p_s / (p_t * strike)) / sp + 0.5 * sp;
    if (call) {
        return p_s * norm_cdf(h) - strike * p_t * norm_cdf(h - sp);
    }
    return strike * p_t * norm_cdf(sp - h) - p_s * norm_cdf(-h);
}

double Vasicek::zbc(double expiry, double bond_maturity, double strike) const {
    return zb_option(expiry, bond_maturity, strike, /*call=*/true);
}

double Vasicek::zbp(double expiry, double bond_maturity, double strike) const {
    return zb_option(expiry, bond_maturity, strike, /*call=*/false);
}

std::pair<std::vector<double>, std::vector<double>> Vasicek::simulate(
    double horizon, int n_steps, int n_paths, std::uint64_t seed) const {
    if (!(std::isfinite(horizon)) || horizon <= 0.0) {
        throw std::invalid_argument("horizon must be finite and > 0, got " +
                                    std::to_string(horizon));
    }
    if (n_steps < 1 || n_paths < 1) {
        throw std::invalid_argument("n_steps and n_paths must be >= 1");
    }
    const double dt = horizon / n_steps;
    const OUStepMoments m = ou_step_moments(kappa_, sigma_, dt);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<double> r_end(static_cast<std::size_t>(n_paths));
    std::vector<double> integral_r(static_cast<std::size_t>(n_paths));
    const double x0 = r0_ - theta_;
    if (m.var_x > 0.0) {
        const double sd_x = std::sqrt(m.var_x);
        const double beta = m.cov / m.var_x;  // regression of the integral on x_next
        const double resid_var = std::max(m.var_i - m.cov * m.cov / m.var_x, 0.0);
        const double sd_resid = std::sqrt(resid_var);
        for (int p = 0; p < n_paths; ++p) {
            double x = x0;
            double integral_x = 0.0;
            for (int s = 0; s < n_steps; ++s) {
                const double z1 = normal(rng);
                const double z2 = normal(rng);
                const double mean_x = x * m.decay;
                const double mean_i = x * m.b;
                const double x_new = mean_x + sd_x * z1;
                integral_x += mean_i + beta * (x_new - mean_x) + sd_resid * z2;
                x = x_new;
            }
            r_end[static_cast<std::size_t>(p)] = theta_ + x;
            integral_r[static_cast<std::size_t>(p)] = theta_ * horizon + integral_x;
        }
    } else {  // sigma = 0: fully deterministic decay
        double x = x0;
        double integral_x = 0.0;
        for (int s = 0; s < n_steps; ++s) {
            integral_x += x * m.b;
            x = x * m.decay;
        }
        std::fill(r_end.begin(), r_end.end(), theta_ + x);
        std::fill(integral_r.begin(), integral_r.end(), theta_ * horizon + integral_x);
    }
    return {std::move(r_end), std::move(integral_r)};
}

std::pair<double, double> Vasicek::mc_zcb(double maturity, int n_steps, int n_paths,
                                          std::uint64_t seed) const {
    const auto sim = simulate(maturity, n_steps, n_paths, seed);
    const std::vector<double>& integral = sim.second;
    std::vector<double> disc(integral.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < integral.size(); ++i) {
        disc[i] = std::exp(-integral[i]);
        sum += disc[i];
    }
    const double price = sum / static_cast<double>(disc.size());
    double se = 0.0;
    // sigma = 0: deterministic paths, standard error exactly 0 (per spec) —
    // avoids reporting summation-rounding noise as spread.
    if (n_paths > 1 && sigma_ > 0.0) {
        double ss = 0.0;
        for (double d : disc) ss += (d - price) * (d - price);
        const double stdev = std::sqrt(ss / static_cast<double>(n_paths - 1));
        se = stdev / std::sqrt(static_cast<double>(n_paths));
    }
    return {price, se};
}

VasicekCalibration calibrate_vasicek(const std::vector<double>& maturities,
                                     const std::vector<double>& yields, double r0,
                                     std::optional<std::vector<double>> x0, int maxiter) {
    if (maturities.size() != yields.size() || maturities.size() < 3) {
        throw std::invalid_argument("need at least 3 (maturity, yield) pairs of equal length");
    }
    for (double t : maturities) {
        if (!std::isfinite(t) || t <= 0.0) {
            throw std::invalid_argument("maturities must be finite and > 0");
        }
    }
    for (double y : yields) {
        if (!std::isfinite(y)) {
            throw std::invalid_argument("yields must be finite");
        }
    }
    if (!std::isfinite(r0)) {
        throw std::invalid_argument("r0 must be finite");
    }

    const auto objective = [&](const std::vector<double>& p) {
        const double kappa = p[0], theta = p[1], sigma = p[2];
        if (kappa <= 1e-6 || sigma < 0.0 || kappa > 50.0 || std::fabs(theta) > 5.0 ||
            sigma > 5.0) {
            // Smooth penalty pointing back toward the feasible region.
            return 1e6 * (1.0 + std::max(0.0, 1e-6 - kappa) + std::max(0.0, -sigma) +
                          std::max(0.0, kappa - 50.0) + std::max(0.0, std::fabs(theta) - 5.0) +
                          std::max(0.0, sigma - 5.0));
        }
        const Vasicek model(kappa, theta, sigma, r0);
        double sum = 0.0;
        for (std::size_t i = 0; i < maturities.size(); ++i) {
            const double diff = model.zero_yield(maturities[i]) - yields[i];
            sum += diff * diff;
        }
        return sum;
    };

    std::vector<double> start;
    if (x0.has_value()) {
        start = *x0;
    } else {
        double mean_y = 0.0;
        for (double y : yields) mean_y += y;
        mean_y /= static_cast<double>(yields.size());
        start = {0.5, mean_y, 0.01};
    }
    const NelderMeadResult res = nelder_mead(objective, start, /*initial_step=*/0.05,
                                             /*xtol=*/1e-10, /*ftol=*/1e-14, maxiter);
    const double rmse =
        std::sqrt(std::max(res.fx, 0.0) / static_cast<double>(maturities.size()));
    return VasicekCalibration{res.x[0],           res.x[1],       std::fabs(res.x[2]), r0,
                              rmse,               res.iterations, res.converged};
}

}  // namespace irm
