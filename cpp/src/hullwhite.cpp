#include "irm/hullwhite.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include "irm/mathutils.hpp"
#include "irm/rootfind.hpp"

namespace irm {

HullWhite::HullWhite(double a, double sigma, DiscountCurve curve)
    : a_(a), sigma_(sigma), curve_(std::move(curve)) {
    if (!std::isfinite(a) || a <= 0.0) {
        throw std::invalid_argument("mean reversion a must be finite and > 0, got " +
                                    std::to_string(a));
    }
    if (!std::isfinite(sigma) || sigma < 0.0) {
        throw std::invalid_argument("sigma must be finite and >= 0, got " +
                                    std::to_string(sigma));
    }
}

double HullWhite::fwd0(double t) const {
    if (!std::isfinite(t) || t < 0.0) {
        throw std::invalid_argument("time must be finite and >= 0, got " + std::to_string(t));
    }
    const double h = kFdStep;
    if (t >= h) {
        return (std::log(curve_.df(t - h)) - std::log(curve_.df(t + h))) / (2.0 * h);
    }
    return (std::log(curve_.df(t)) - std::log(curve_.df(t + h))) / h;
}

double HullWhite::theta(double t, double h) const {
    const double dfdt = (fwd0(t + h) - fwd0(t)) / h;
    return dfdt + a_ * fwd0(t) +
           (sigma_ * sigma_ / (2.0 * a_)) * (1.0 - std::exp(-2.0 * a_ * t));
}

double HullWhite::b_factor(double t, double maturity) const {
    if (maturity < t) {
        throw std::invalid_argument("maturity " + std::to_string(maturity) + " before t " +
                                    std::to_string(t));
    }
    return (1.0 - std::exp(-a_ * (maturity - t))) / a_;
}

double HullWhite::a_factor(double t, double maturity) const {
    const double b = b_factor(t, maturity);
    const double pm_t = curve_.df(t);
    const double pm_mat = curve_.df(maturity);
    const double conv =
        sigma_ * sigma_ / (4.0 * a_) * (1.0 - std::exp(-2.0 * a_ * t)) * b * b;
    return (pm_mat / pm_t) * std::exp(b * fwd0(t) - conv);
}

double HullWhite::zcb_price(double t, double maturity, std::optional<double> r) const {
    const double rr = r.has_value() ? *r : fwd0(0.0);
    if (!std::isfinite(rr)) {
        throw std::invalid_argument("short rate must be finite, got " + std::to_string(rr));
    }
    return a_factor(t, maturity) * std::exp(-b_factor(t, maturity) * rr);
}

double HullWhite::sigma_p(double expiry, double bond_maturity) const {
    // Std dev of ln P at expiry.
    return sigma_ * std::sqrt((1.0 - std::exp(-2.0 * a_ * expiry)) / (2.0 * a_)) *
           b_factor(expiry, bond_maturity);
}

double HullWhite::zb_option(double expiry, double bond_maturity, double strike,
                            bool call) const {
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
    const double p_s = curve_.df(bond_maturity);
    const double p_t = curve_.df(expiry);
    const double sp = sigma_p(expiry, bond_maturity);
    if (sp == 0.0) {
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

double HullWhite::zbc(double expiry, double bond_maturity, double strike) const {
    return zb_option(expiry, bond_maturity, strike, /*call=*/true);
}

double HullWhite::zbp(double expiry, double bond_maturity, double strike) const {
    return zb_option(expiry, bond_maturity, strike, /*call=*/false);
}

double HullWhite::caplet(double reset, double pay, double strike, double notional) const {
    const double tau = pay - reset;
    if (tau <= 0.0 || reset < 0.0) {
        throw std::invalid_argument("need 0 <= reset < pay, got " + std::to_string(reset) +
                                    ", " + std::to_string(pay));
    }
    if (!std::isfinite(strike) || 1.0 + strike * tau <= 0.0) {
        throw std::invalid_argument("caplet strike implies non-positive 1 + K*tau (" +
                                    std::to_string(strike) + ")");
    }
    const double k_bond = 1.0 / (1.0 + strike * tau);
    return notional * (1.0 + strike * tau) * zbp(reset, pay, k_bond);
}

double HullWhite::floorlet(double reset, double pay, double strike, double notional) const {
    const double tau = pay - reset;
    if (tau <= 0.0 || reset < 0.0) {
        throw std::invalid_argument("need 0 <= reset < pay, got " + std::to_string(reset) +
                                    ", " + std::to_string(pay));
    }
    if (!std::isfinite(strike) || 1.0 + strike * tau <= 0.0) {
        throw std::invalid_argument("floorlet strike implies non-positive 1 + K*tau (" +
                                    std::to_string(strike) + ")");
    }
    const double k_bond = 1.0 / (1.0 + strike * tau);
    return notional * (1.0 + strike * tau) * zbc(reset, pay, k_bond);
}

double HullWhite::cap(const std::vector<double>& schedule, double strike,
                      double notional) const {
    if (schedule.size() < 2) {
        throw std::invalid_argument("cap schedule needs at least two times");
    }
    double total = 0.0;
    for (std::size_t i = 0; i + 1 < schedule.size(); ++i) {
        total += caplet(schedule[i], schedule[i + 1], strike, notional);
    }
    return total;
}

double HullWhite::par_swap_rate(const std::vector<double>& times) const {
    return irm::par_swap_rate(curve_, times);
}

JamshidianResult HullWhite::jamshidian_swaption(double expiry,
                                                const std::vector<double>& pay_times,
                                                double fixed_rate, double notional,
                                                bool payer) const {
    if (!std::isfinite(expiry) || expiry <= 0.0) {
        throw std::invalid_argument("swaption expiry must be > 0, got " +
                                    std::to_string(expiry));
    }
    if (pay_times.empty()) {
        throw std::invalid_argument("swaption needs at least one payment time");
    }
    if (!std::isfinite(fixed_rate)) {
        throw std::invalid_argument("fixed rate must be finite");
    }
    double prev = expiry;
    std::vector<double> taus;
    taus.reserve(pay_times.size());
    for (double t : pay_times) {
        if (t <= prev) {
            throw std::invalid_argument(
                "payment times must be strictly increasing and after expiry");
        }
        taus.push_back(t - prev);
        prev = t;
    }
    std::vector<double> coupons;
    coupons.reserve(taus.size());
    for (double tau : taus) coupons.push_back(fixed_rate * tau);
    coupons.back() += 1.0;

    // g(r) = sum_i c_i P(T, u_i; r) - 1, strictly decreasing in r (B > 0).
    const auto bond_minus_one = [&](double r) {
        double sum = 0.0;
        for (std::size_t i = 0; i < coupons.size(); ++i) {
            sum += coupons[i] * zcb_price(expiry, pay_times[i], r);
        }
        return sum - 1.0;
    };

    // Expanding bracket: double the endpoints until g(lo) > 0 > g(hi).
    double lo = -1.0, hi = 1.0;
    bool bracketed = false;
    for (int i = 0; i < 24; ++i) {
        if (bond_minus_one(lo) > 0.0 && bond_minus_one(hi) < 0.0) {
            bracketed = true;
            break;
        }
        if (bond_minus_one(lo) <= 0.0) lo *= 2.0;
        if (bond_minus_one(hi) >= 0.0) hi *= 2.0;
    }
    if (!bracketed) {
        throw std::domain_error(
            "jamshidian: could not bracket r* (degenerate coupon bond — check fixed "
            "rate/schedule)");
    }
    const double r_star = brentq(bond_minus_one, lo, hi, /*xtol=*/1e-15);
    const double residual = bond_minus_one(r_star);

    std::vector<double> strikes;
    strikes.reserve(pay_times.size());
    for (double t : pay_times) strikes.push_back(zcb_price(expiry, t, r_star));

    double value = 0.0;
    for (std::size_t i = 0; i < coupons.size(); ++i) {
        value += coupons[i] * (payer ? zbp(expiry, pay_times[i], strikes[i])
                                     : zbc(expiry, pay_times[i], strikes[i]));
    }
    return JamshidianResult{notional * value, r_star, residual, std::move(strikes)};
}

double HullWhite::alpha(double t) const {
    const double one_m = 1.0 - std::exp(-a_ * t);
    return fwd0(t) + (sigma_ * sigma_ / (2.0 * a_ * a_)) * one_m * one_m;
}

double HullWhite::variance_integral(double t) const {
    const double a = a_;
    return (sigma_ * sigma_ / (2.0 * a * a)) *
           (t - 2.0 * (1.0 - std::exp(-a * t)) / a +
            (1.0 - std::exp(-2.0 * a * t)) / (2.0 * a));
}

std::pair<double, double> HullWhite::mc_caplet(double reset, double pay, double strike,
                                               double notional, int n_steps, int n_paths,
                                               std::uint64_t seed) const {
    const double tau = pay - reset;
    if (tau <= 0.0 || reset <= 0.0) {
        throw std::invalid_argument("need 0 < reset < pay, got " + std::to_string(reset) +
                                    ", " + std::to_string(pay));
    }
    if (n_steps < 1 || n_paths < 1) {
        throw std::invalid_argument("n_steps and n_paths must be >= 1");
    }
    const double dt = reset / n_steps;
    const OUStepMoments m = ou_step_moments(a_, sigma_, dt);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    // Path-independent pieces of the valuation.
    const double df_reset = curve_.df(reset);
    const double v_reset = variance_integral(reset);
    const double alpha_reset = alpha(reset);
    const double b = b_factor(reset, pay);
    const double a_f = a_factor(reset, pay);

    const double sd_x = (m.var_x > 0.0) ? std::sqrt(m.var_x) : 0.0;
    const double beta = (m.var_x > 0.0) ? m.cov / m.var_x : 0.0;
    const double sd_resid =
        (m.var_x > 0.0) ? std::sqrt(std::max(m.var_i - m.cov * m.cov / m.var_x, 0.0)) : 0.0;

    std::vector<double> pv(static_cast<std::size_t>(n_paths));
    for (int p = 0; p < n_paths; ++p) {
        double x = 0.0;
        double integral_x = 0.0;
        if (m.var_x > 0.0) {
            for (int s = 0; s < n_steps; ++s) {
                const double z1 = normal(rng);
                const double z2 = normal(rng);
                const double mean_x = x * m.decay;
                const double mean_i = x * m.b;
                const double x_new = mean_x + sd_x * z1;
                integral_x += mean_i + beta * (x_new - mean_x) + sd_resid * z2;
                x = x_new;
            }
        }
        // sigma = 0: x stays 0, integral stays 0 (deterministic curve replay).
        const double disc = df_reset * std::exp(-v_reset - integral_x);
        const double r_reset = x + alpha_reset;
        const double p_reset_pay = a_f * std::exp(-b * r_reset);
        const double fwd = (1.0 / p_reset_pay - 1.0) / tau;
        const double payoff = notional * tau * std::max(fwd - strike, 0.0);
        pv[static_cast<std::size_t>(p)] = disc * p_reset_pay * payoff;
    }

    double sum = 0.0;
    for (double v : pv) sum += v;
    const double price = sum / static_cast<double>(n_paths);
    double se = 0.0;
    // sigma = 0: deterministic paths, standard error exactly 0 (per spec).
    if (n_paths > 1 && sigma_ > 0.0) {
        double ss = 0.0;
        for (double v : pv) ss += (v - price) * (v - price);
        const double stdev = std::sqrt(ss / static_cast<double>(n_paths - 1));
        se = stdev / std::sqrt(static_cast<double>(n_paths));
    }
    return {price, se};
}

}  // namespace irm
