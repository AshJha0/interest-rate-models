#include "irm/vasicek.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

#include "irm/mathutils.hpp"
#include "irm/optimize.hpp"

namespace irm {

namespace {

/// Largest exponent passed to exp() before a ZCB price is declared not
/// representable (exp(709.78) is the double overflow point).
constexpr double kMaxExpArg = 700.0;

// at_bound thresholds (documented in API_SPEC section 7).
constexpr double kKappaLoBound = 1e-5;   // 10x the penalty floor 1e-6
constexpr double kKappaHiBound = 49.95;  // within 1e-3 of the range from 50
constexpr double kThetaBound = 4.995;
constexpr double kSigmaLoBound = 1e-5;
constexpr double kSigmaHiBound = 4.995;
// Starts whose rmse is within 5 % of the best are "equivalent fits"; their
// parameter spreads must be below these to call the solution identified.
constexpr double kEquivRmseFactor = 1.05;
constexpr double kEquivRmseFloor = 1e-12;
constexpr double kIdentKappa = 1e-2;
constexpr double kIdentTheta = 1e-3;
constexpr double kIdentSigma = 1e-3;
// Default multi-start (kappa, sigma) pairs; theta starts at the mean yield.
constexpr double kStartKappa[] = {0.2, 0.5, 1.5};
constexpr double kStartSigma[] = {0.005, 0.01, 0.02};

struct StartResult {
    double fx;
    double kappa;
    double theta;
    double sigma;
    int iterations;
    bool converged;
};

}  // namespace

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
    return -std::expm1(-kappa_ * tau) / kappa_;
}

double Vasicek::a_factor(double tau) const {
    const double b = b_factor(tau);
    const double k = kappa_, s = sigma_;
    return std::exp((theta_ - s * s / (2.0 * k * k)) * (b - tau) - s * s * b * b / (4.0 * k));
}

double Vasicek::zcb_price(double maturity, double t, std::optional<double> r) const {
    if (!(std::isfinite(maturity) && std::isfinite(t))) {
        throw std::invalid_argument("maturity and t must be finite");
    }
    if (maturity < t) {
        throw std::invalid_argument("maturity " + std::to_string(maturity) +
                                    " before valuation time " + std::to_string(t));
    }
    const double rr = r.value_or(r0_);
    if (!std::isfinite(rr)) {
        throw std::invalid_argument("short rate must be finite, got " + std::to_string(rr));
    }
    const double tau = maturity - t;
    const double expo = -b_factor(tau) * rr;
    if (expo > kMaxExpArg) {
        throw std::invalid_argument("ZCB price not representable: exp(" + std::to_string(expo) +
                                    ") overflows");
    }
    return a_factor(tau) * std::exp(expo);
}

double Vasicek::zero_yield(double maturity, double t, std::optional<double> r) const {
    const double tau = maturity - t;
    if (!std::isfinite(tau) || tau <= 0.0) {
        throw std::invalid_argument("need maturity > t, got tau=" + std::to_string(tau));
    }
    return -std::log(zcb_price(maturity, t, r)) / tau;
}

double Vasicek::r_mean(double horizon, std::optional<double> r) const {
    if (!std::isfinite(horizon) || horizon < 0.0) {
        throw std::invalid_argument("horizon must be >= 0, got " + std::to_string(horizon));
    }
    const double rr = r.value_or(r0_);
    if (!std::isfinite(rr)) {
        throw std::invalid_argument("short rate must be finite, got " + std::to_string(rr));
    }
    return theta_ + (rr - theta_) * std::exp(-kappa_ * horizon);
}

double Vasicek::r_var(double horizon) const {
    if (!std::isfinite(horizon) || horizon < 0.0) {
        throw std::invalid_argument("horizon must be >= 0, got " + std::to_string(horizon));
    }
    return sigma_ * sigma_ * (-std::expm1(-2.0 * kappa_ * horizon)) / (2.0 * kappa_);
}

double Vasicek::sigma_p(double expiry, double bond_maturity) const {
    // Std dev of ln P(T,S) at option expiry T for bond maturity S.
    return sigma_ * std::sqrt((-std::expm1(-2.0 * kappa_ * expiry)) / (2.0 * kappa_)) *
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
                                     std::optional<std::vector<double>> x0, int maxiter,
                                     std::optional<double> sigma_fixed) {
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
    if (maxiter < 1) {
        throw std::invalid_argument("maxiter must be >= 1");
    }
    if (x0.has_value()) {
        if (x0->size() != 3) {
            throw std::invalid_argument("x0 must have 3 entries (kappa, theta, sigma)");
        }
        for (double v : *x0) {
            if (!std::isfinite(v)) throw std::invalid_argument("x0 must be finite");
        }
    }
    if (sigma_fixed.has_value() && (!std::isfinite(*sigma_fixed) || *sigma_fixed < 0.0)) {
        throw std::invalid_argument("sigma_fixed must be finite and >= 0, got " +
                                    std::to_string(*sigma_fixed));
    }
    const bool fixed = sigma_fixed.has_value();

    const auto sse = [&](double kappa, double theta, double sigma) {
        if (kappa <= kVasicekKappaLo || sigma < 0.0 || kappa > kVasicekKappaHi ||
            std::fabs(theta) > kVasicekThetaMax || sigma > kVasicekSigmaHi) {
            // Smooth penalty pointing back toward the feasible region.
            return 1e6 * (1.0 + std::max(0.0, kVasicekKappaLo - kappa) + std::max(0.0, -sigma) +
                          std::max(0.0, kappa - kVasicekKappaHi) +
                          std::max(0.0, std::fabs(theta) - kVasicekThetaMax) +
                          std::max(0.0, sigma - kVasicekSigmaHi));
        }
        const Vasicek model(kappa, theta, sigma, r0);
        double sum = 0.0;
        for (std::size_t i = 0; i < maturities.size(); ++i) {
            const double diff = model.zero_yield(maturities[i]) - yields[i];
            sum += diff * diff;
        }
        return sum;
    };
    const auto objective = [&](const std::vector<double>& p) {
        return fixed ? sse(p[0], p[1], *sigma_fixed) : sse(p[0], p[1], p[2]);
    };

    double mean_y = 0.0;
    for (double y : yields) mean_y += y;
    mean_y /= static_cast<double>(yields.size());
    std::vector<std::vector<double>> starts;
    if (x0.has_value()) {
        starts.push_back(*x0);
    } else {
        for (int k = 0; k < 3; ++k) starts.push_back({kStartKappa[k], mean_y, kStartSigma[k]});
    }

    std::vector<StartResult> results;
    for (const auto& st : starts) {
        const std::vector<double> p0 = fixed ? std::vector<double>{st[0], st[1]} : st;
        const NelderMeadResult res = nelder_mead(objective, p0, /*initial_step=*/0.05,
                                                 /*xtol=*/1e-10, /*ftol=*/1e-14, maxiter);
        const double sigma = fixed ? *sigma_fixed : std::fabs(res.x[2]);
        results.push_back({res.fx, res.x[0], res.x[1], sigma, res.iterations, res.converged});
    }
    std::size_t best = 0;
    for (std::size_t i = 1; i < results.size(); ++i) {
        if (results[i].fx < results[best].fx) best = i;
    }
    const StartResult& b = results[best];
    const double n = static_cast<double>(maturities.size());
    const double rmse = std::sqrt(std::max(b.fx, 0.0) / n);
    const bool at_bound =
        b.kappa < kKappaLoBound || b.kappa > kKappaHiBound || std::fabs(b.theta) > kThetaBound ||
        (!fixed && (b.sigma < kSigmaLoBound || b.sigma > kSigmaHiBound));

    double k_lo = b.kappa, k_hi = b.kappa, t_lo = b.theta, t_hi = b.theta;
    double s_lo = b.sigma, s_hi = b.sigma;
    for (const StartResult& r : results) {
        const double r_rmse = std::sqrt(std::max(r.fx, 0.0) / n);
        if (r_rmse <= kEquivRmseFactor * rmse + kEquivRmseFloor) {
            k_lo = std::min(k_lo, r.kappa);
            k_hi = std::max(k_hi, r.kappa);
            t_lo = std::min(t_lo, r.theta);
            t_hi = std::max(t_hi, r.theta);
            s_lo = std::min(s_lo, r.sigma);
            s_hi = std::max(s_hi, r.sigma);
        }
    }
    const double kappa_spread = k_hi - k_lo;
    const double theta_spread = t_hi - t_lo;
    const double sigma_spread = s_hi - s_lo;
    const bool identified = results.size() >= 2 && kappa_spread <= kIdentKappa &&
                            theta_spread <= kIdentTheta && sigma_spread <= kIdentSigma;
    return VasicekCalibration{b.kappa,
                              b.theta,
                              b.sigma,
                              r0,
                              rmse,
                              b.iterations,
                              b.converged && !at_bound,
                              at_bound,
                              static_cast<int>(results.size()),
                              kappa_spread,
                              theta_spread,
                              sigma_spread,
                              identified,
                              fixed};
}

}  // namespace irm
