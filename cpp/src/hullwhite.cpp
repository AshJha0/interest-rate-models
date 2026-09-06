#include "irm/hullwhite.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include "irm/mathutils.hpp"
#include "irm/optimize.hpp"
#include "irm/rootfind.hpp"

namespace irm {

namespace {

/// Maximum bracket doublings when searching for r*.
constexpr int kJamshidianMaxDoublings = 24;
/// Largest exponent passed to exp() before a ZCB price is declared not
/// representable (exp(709.78) is the double overflow point).
constexpr double kMaxExpArg = 700.0;

// Calibration diagnostics thresholds (documented in API_SPEC section 7).
constexpr double kALoBound = 1e-5;
constexpr double kAHiBound = 4.995;
constexpr double kSigmaLoBound = 1e-5;
constexpr double kSigmaHiBound = 0.999;
constexpr double kEquivRmseFactor = 1.05;
constexpr double kEquivRmseFloor = 1e-14;
constexpr double kIdentA = 1e-2;
constexpr double kIdentSigma = 1e-4;
/// sigma is optimised as sigma * kSigmaScale so one Nelder-Mead step fits both.
constexpr double kSigmaScale = 10.0;
constexpr double kStartA[] = {0.03, 0.1, 0.5};
constexpr double kStartSigma[] = {0.005, 0.01, 0.02};

void check_finite(const char* name, double v) {
    if (!std::isfinite(v)) {
        throw std::invalid_argument(std::string(name) + " must be finite, got " +
                                    std::to_string(v));
    }
}

struct StartResult {
    double fx;
    double a;
    double sigma;
    int iterations;
    bool converged;
};

}  // namespace

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
    if (!std::isfinite(h) || h <= 0.0) {
        throw std::invalid_argument("theta step h must be finite and > 0, got " +
                                    std::to_string(h));
    }
    const double dfdt = (fwd0(t + h) - fwd0(t)) / h;
    return dfdt + a_ * fwd0(t) +
           (sigma_ * sigma_ / (2.0 * a_)) * (-std::expm1(-2.0 * a_ * t));
}

double HullWhite::b_factor(double t, double maturity) const {
    check_finite("t", t);
    check_finite("maturity", maturity);
    if (t < 0.0 || maturity < t) {
        throw std::invalid_argument("need 0 <= t <= maturity, got t=" + std::to_string(t) +
                                    ", maturity=" + std::to_string(maturity));
    }
    return -std::expm1(-a_ * (maturity - t)) / a_;
}

double HullWhite::a_factor(double t, double maturity) const {
    const double b = b_factor(t, maturity);
    const double pm_t = curve_.df(t);
    const double pm_mat = curve_.df(maturity);
    const double conv =
        sigma_ * sigma_ / (4.0 * a_) * (-std::expm1(-2.0 * a_ * t)) * b * b;
    return (pm_mat / pm_t) * std::exp(b * fwd0(t) - conv);
}

double HullWhite::zcb_price(double t, double maturity, std::optional<double> r) const {
    const double rr = r.has_value() ? *r : fwd0(0.0);
    if (!std::isfinite(rr)) {
        throw std::invalid_argument("short rate must be finite, got " + std::to_string(rr));
    }
    const double expo = -b_factor(t, maturity) * rr;
    if (expo > kMaxExpArg) {
        throw std::invalid_argument("ZCB price not representable: exp(" + std::to_string(expo) +
                                    ") overflows");
    }
    return a_factor(t, maturity) * std::exp(expo);
}

double HullWhite::sigma_p(double expiry, double bond_maturity) const {
    // Std dev of ln P at expiry.
    return sigma_ * std::sqrt((-std::expm1(-2.0 * a_ * expiry)) / (2.0 * a_)) *
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

double HullWhite::check_caplet_inputs(const char* what, double reset, double pay,
                                      double strike, double notional) {
    check_finite("reset", reset);
    check_finite("pay", pay);
    check_finite("strike", strike);
    check_finite("notional", notional);
    const double tau = pay - reset;
    if (tau <= 0.0 || reset < 0.0) {
        throw std::invalid_argument(std::string(what) + ": need 0 <= reset < pay, got " +
                                    std::to_string(reset) + ", " + std::to_string(pay));
    }
    if (1.0 + strike * tau <= 0.0) {
        throw std::invalid_argument(std::string(what) +
                                    ": strike implies non-positive 1 + K*tau (" +
                                    std::to_string(strike) + ")");
    }
    return tau;
}

double HullWhite::caplet(double reset, double pay, double strike, double notional) const {
    const double tau = check_caplet_inputs("caplet", reset, pay, strike, notional);
    const double k_bond = 1.0 / (1.0 + strike * tau);
    return notional * (1.0 + strike * tau) * zbp(reset, pay, k_bond);
}

double HullWhite::floorlet(double reset, double pay, double strike, double notional) const {
    const double tau = check_caplet_inputs("floorlet", reset, pay, strike, notional);
    const double k_bond = 1.0 / (1.0 + strike * tau);
    return notional * (1.0 + strike * tau) * zbc(reset, pay, k_bond);
}

double HullWhite::cap(const std::vector<double>& schedule, double strike,
                      double notional) const {
    if (schedule.size() < 2) {
        throw std::invalid_argument("cap schedule needs at least two times");
    }
    for (double t : schedule) {
        if (!std::isfinite(t)) throw std::invalid_argument("cap schedule times must be finite");
    }
    if (schedule.front() < 0.0) {
        throw std::invalid_argument("cap schedule must be strictly increasing with t0 >= 0");
    }
    for (std::size_t i = 0; i + 1 < schedule.size(); ++i) {
        if (schedule[i + 1] <= schedule[i]) {
            throw std::invalid_argument("cap schedule must be strictly increasing with t0 >= 0");
        }
    }
    check_finite("strike", strike);
    check_finite("notional", notional);
    double total = 0.0;
    for (std::size_t i = 0; i + 1 < schedule.size(); ++i) {
        total += caplet(schedule[i], schedule[i + 1], strike, notional);
    }
    return total;
}

double HullWhite::par_swap_rate(const std::vector<double>& times) const {
    return irm::par_swap_rate(curve_, times);
}

void HullWhite::swaption_coupons(double expiry, const std::vector<double>& pay_times,
                                 double fixed_rate, double notional,
                                 std::vector<double>& coupons) const {
    if (!std::isfinite(expiry) || expiry <= 0.0) {
        throw std::invalid_argument("swaption expiry must be finite and > 0, got " +
                                    std::to_string(expiry));
    }
    if (pay_times.empty()) {
        throw std::invalid_argument("swaption needs at least one payment time");
    }
    check_finite("fixed_rate", fixed_rate);
    check_finite("notional", notional);
    double prev = expiry;
    coupons.clear();
    coupons.reserve(pay_times.size());
    for (double t : pay_times) {
        if (!std::isfinite(t) || t <= prev) {
            throw std::invalid_argument(
                "payment times must be finite, strictly increasing and after expiry");
        }
        coupons.push_back(fixed_rate * (t - prev));
        prev = t;
    }
    coupons.back() += 1.0;
    if (coupons.back() <= 0.0) {
        throw std::invalid_argument(
            "jamshidian: degenerate coupon bond — 1 + X*tau_n must be > 0 (got " +
            std::to_string(coupons.back()) + "); no r* exists");
    }
}

JamshidianResult HullWhite::jamshidian_at_r_star(double expiry,
                                                 const std::vector<double>& pay_times,
                                                 double fixed_rate, double r_star,
                                                 double notional, bool payer) const {
    std::vector<double> coupons;
    swaption_coupons(expiry, pay_times, fixed_rate, notional, coupons);
    if (!std::isfinite(r_star)) {
        throw std::invalid_argument("jamshidian: r* must be finite, got " +
                                    std::to_string(r_star));
    }
    std::vector<double> strikes;
    strikes.reserve(pay_times.size());
    for (double t : pay_times) strikes.push_back(zcb_price(expiry, t, r_star));
    double residual = 0.0;
    for (std::size_t i = 0; i < coupons.size(); ++i) residual += coupons[i] * strikes[i];
    residual -= 1.0;
    if (!(std::fabs(residual) < kJamshidianResidualTol)) {
        throw std::domain_error("jamshidian: r* residual " + std::to_string(residual) +
                                " exceeds " + std::to_string(kJamshidianResidualTol));
    }
    double slope = 0.0;
    for (std::size_t i = 0; i < coupons.size(); ++i) {
        slope -= coupons[i] * b_factor(expiry, pay_times[i]) * strikes[i];
    }
    if (!(slope < 0.0)) {
        throw std::domain_error("jamshidian: coupon bond not decreasing at r* (slope " +
                                std::to_string(slope) + "); decomposition invalid");
    }
    double value = 0.0;
    for (std::size_t i = 0; i < coupons.size(); ++i) {
        value += coupons[i] * (payer ? zbp(expiry, pay_times[i], strikes[i])
                                     : zbc(expiry, pay_times[i], strikes[i]));
    }
    return JamshidianResult{notional * value, r_star, residual, std::move(strikes)};
}

JamshidianResult HullWhite::jamshidian_swaption(double expiry,
                                                const std::vector<double>& pay_times,
                                                double fixed_rate, double notional,
                                                bool payer) const {
    std::vector<double> coupons;
    swaption_coupons(expiry, pay_times, fixed_rate, notional, coupons);

    // g(r) = sum_i c_i P(T, u_i; r) - 1; g - 1 has a single sign change.
    const auto bond_minus_one = [&](double r) {
        double sum = 0.0;
        for (std::size_t i = 0; i < coupons.size(); ++i) {
            sum += coupons[i] * zcb_price(expiry, pay_times[i], r);
        }
        return sum - 1.0;
    };

    // Expanding bracket: g(lo) > 0 > g(hi) is needed.  g(lo) grows like
    // exp(B_n |lo|); refuse to evaluate past the exp() overflow point.
    const double b_last = b_factor(expiry, pay_times.back());
    double lo = -1.0, hi = 1.0;
    bool bracketed = false;
    for (int i = 0; i < kJamshidianMaxDoublings; ++i) {
        if (b_last * (-lo) > kMaxExpArg) break;
        const double g_lo = bond_minus_one(lo);
        const double g_hi = bond_minus_one(hi);
        if (g_lo > 0.0 && g_hi < 0.0) {
            bracketed = true;
            break;
        }
        if (g_lo <= 0.0) lo *= 2.0;
        if (g_hi >= 0.0) hi *= 2.0;
    }
    if (!bracketed) {
        throw std::domain_error(
            "jamshidian: could not bracket r* (degenerate coupon bond — check fixed "
            "rate/schedule)");
    }
    const double r_star = brentq(bond_minus_one, lo, hi, /*xtol=*/1e-15);
    return jamshidian_at_r_star(expiry, pay_times, fixed_rate, r_star, notional, payer);
}

double HullWhite::alpha(double t) const {
    const double one_m = -std::expm1(-a_ * t);
    return fwd0(t) + (sigma_ * sigma_ / (2.0 * a_ * a_)) * one_m * one_m;
}

double HullWhite::variance_integral(double t) const {
    if (t == 0.0) return 0.0;
    return 0.5 * ou_integral_variance(a_, sigma_, t);
}

std::pair<double, double> HullWhite::mc_caplet(double reset, double pay, double strike,
                                               double notional, int n_steps, int n_paths,
                                               std::uint64_t seed) const {
    const double tau = check_caplet_inputs("mc_caplet", reset, pay, strike, notional);
    if (reset <= 0.0) {
        throw std::invalid_argument("mc_caplet: need 0 < reset < pay, got " +
                                    std::to_string(reset) + ", " + std::to_string(pay));
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

// ---- calibration ----------------------------------------------------------

CapletQuote::CapletQuote(double reset_, double pay_, double strike_, double price_)
    : reset(reset_), pay(pay_), strike(strike_), price(price_) {
    check_finite("reset", reset);
    check_finite("pay", pay);
    check_finite("strike", strike);
    check_finite("price", price);
    if (reset < 0.0 || pay <= reset) {
        throw std::invalid_argument("caplet quote needs 0 <= reset < pay, got " +
                                    std::to_string(reset) + ", " + std::to_string(pay));
    }
    if (1.0 + strike * (pay - reset) <= 0.0) {
        throw std::invalid_argument("caplet quote strike implies non-positive 1 + K*tau");
    }
    if (price < 0.0) {
        throw std::invalid_argument("caplet quote price must be >= 0, got " +
                                    std::to_string(price));
    }
}

SwaptionQuote::SwaptionQuote(double expiry_, std::vector<double> pay_times_, double fixed_rate_,
                             double price_)
    : expiry(expiry_), pay_times(std::move(pay_times_)), fixed_rate(fixed_rate_), price(price_) {
    check_finite("expiry", expiry);
    check_finite("fixed_rate", fixed_rate);
    check_finite("price", price);
    if (expiry <= 0.0) {
        throw std::invalid_argument("swaption quote expiry must be > 0, got " +
                                    std::to_string(expiry));
    }
    if (pay_times.empty()) {
        throw std::invalid_argument("swaption quote needs at least one payment time");
    }
    double prev = expiry;
    for (double t : pay_times) {
        if (!std::isfinite(t) || t <= prev) {
            throw std::invalid_argument(
                "swaption quote payment times must be finite, strictly increasing and after "
                "expiry");
        }
        prev = t;
    }
    if (price < 0.0) {
        throw std::invalid_argument("swaption quote price must be >= 0, got " +
                                    std::to_string(price));
    }
}

HullWhiteCalibration calibrate_hullwhite(const DiscountCurve& curve,
                                         const std::vector<CapletQuote>& caplets,
                                         const std::vector<SwaptionQuote>& swaptions,
                                         std::optional<double> a_fixed,
                                         std::optional<std::vector<double>> x0, int maxiter) {
    if (caplets.empty() && swaptions.empty()) {
        throw std::invalid_argument(
            "calibrate_hullwhite needs at least one caplet or swaption quote");
    }
    if (maxiter < 1) {
        throw std::invalid_argument("maxiter must be >= 1");
    }
    if (x0.has_value()) {
        if (x0->size() != 2) throw std::invalid_argument("x0 must have 2 entries (a, sigma)");
        for (double v : *x0) {
            if (!std::isfinite(v)) throw std::invalid_argument("x0 must be finite");
        }
    }
    if (a_fixed.has_value() && (!std::isfinite(*a_fixed) || *a_fixed <= 0.0)) {
        throw std::invalid_argument("a_fixed must be finite and > 0, got " +
                                    std::to_string(*a_fixed));
    }
    const bool fixed = a_fixed.has_value();
    const double n_quotes = static_cast<double>(caplets.size() + swaptions.size());

    const auto sse = [&](double a, double sigma) {
        if (a <= kHwALo || a > kHwAHi || sigma < 0.0 || sigma > kHwSigmaHi) {
            return 1e6 * (1.0 + std::max(0.0, kHwALo - a) + std::max(0.0, a - kHwAHi) +
                          std::max(0.0, -sigma) + std::max(0.0, sigma - kHwSigmaHi));
        }
        const HullWhite hw(a, sigma, curve);
        double total = 0.0;
        for (const CapletQuote& q : caplets) {
            const double d = hw.caplet(q.reset, q.pay, q.strike, 1.0) - q.price;
            total += d * d;
        }
        for (const SwaptionQuote& q : swaptions) {
            const double d =
                hw.jamshidian_swaption(q.expiry, q.pay_times, q.fixed_rate, 1.0, true).value -
                q.price;
            total += d * d;
        }
        return total;
    };
    // Nelder-Mead uses one initial_step (0.05) for every coordinate, so the
    // optimiser works on sigma * kSigmaScale (step 0.005 in sigma units).
    const auto objective = [&](const std::vector<double>& p) {
        return fixed ? sse(*a_fixed, p[0] / kSigmaScale) : sse(p[0], p[1] / kSigmaScale);
    };

    std::vector<std::vector<double>> starts;
    if (fixed) {
        if (x0.has_value()) {
            starts.push_back({(*x0)[1] * kSigmaScale});
        } else {
            for (int k = 0; k < 3; ++k) starts.push_back({kStartSigma[k] * kSigmaScale});
        }
    } else {
        if (x0.has_value()) {
            starts.push_back({(*x0)[0], (*x0)[1] * kSigmaScale});
        } else {
            for (int k = 0; k < 3; ++k) starts.push_back({kStartA[k], kStartSigma[k] * kSigmaScale});
        }
    }

    std::vector<StartResult> results;
    for (const auto& st : starts) {
        const NelderMeadResult res = nelder_mead(objective, st, /*initial_step=*/0.05,
                                                 /*xtol=*/1e-10, /*ftol=*/1e-14, maxiter);
        const double a_val = fixed ? *a_fixed : res.x[0];
        const double s_val = fixed ? std::fabs(res.x[0]) / kSigmaScale
                                   : std::fabs(res.x[1]) / kSigmaScale;
        results.push_back({res.fx, a_val, s_val, res.iterations, res.converged});
    }
    std::size_t best = 0;
    for (std::size_t i = 1; i < results.size(); ++i) {
        if (results[i].fx < results[best].fx) best = i;
    }
    const StartResult& b = results[best];
    const double rmse = std::sqrt(std::max(b.fx, 0.0) / n_quotes);
    const bool at_bound = (!fixed && (b.a < kALoBound || b.a > kAHiBound)) ||
                          b.sigma < kSigmaLoBound || b.sigma > kSigmaHiBound;
    double a_lo = b.a, a_hi = b.a, s_lo = b.sigma, s_hi = b.sigma;
    for (const StartResult& r : results) {
        const double r_rmse = std::sqrt(std::max(r.fx, 0.0) / n_quotes);
        if (r_rmse <= kEquivRmseFactor * rmse + kEquivRmseFloor) {
            a_lo = std::min(a_lo, r.a);
            a_hi = std::max(a_hi, r.a);
            s_lo = std::min(s_lo, r.sigma);
            s_hi = std::max(s_hi, r.sigma);
        }
    }
    const double a_spread = a_hi - a_lo;
    const double sigma_spread = s_hi - s_lo;
    const bool identified =
        results.size() >= 2 && a_spread <= kIdentA && sigma_spread <= kIdentSigma;
    return HullWhiteCalibration{b.a,
                                b.sigma,
                                rmse,
                                b.iterations,
                                b.converged && !at_bound,
                                at_bound,
                                static_cast<int>(results.size()),
                                a_spread,
                                sigma_spread,
                                identified,
                                fixed};
}

}  // namespace irm
