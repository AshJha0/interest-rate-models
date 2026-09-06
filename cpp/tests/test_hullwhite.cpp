// Tests for the Hull-White model: curve fit at t=0, the FD forward scheme,
// bond-option parity, caplet/cap decomposition, sigma=0 degeneration,
// Jamshidian swaptions and the exact-transition Monte Carlo.

#include "irm/hullwhite.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "irm/curve.hpp"
#include "irm/mathutils.hpp"

namespace {

irm::DiscountCurve market_curve() {
    // Same pillars as the golden HW cases (a smooth positive-rate curve).
    return irm::DiscountCurve(
        {1.0, 2.0, 3.0, 5.0, 10.0},
        {0.9787294774691476, 0.9550419621907147, 0.9291361457915193, 0.8715343499971578,
         0.7046880897187134});
}

irm::DiscountCurve negative_curve() {
    return irm::DiscountCurve(
        {1.0, 2.0, 3.0, 5.0, 10.0},
        {1.0028039236612292, 1.0032051254657053, 1.0012007202880864, 0.9900498337491681,
         0.9231163463866358});
}

TEST(HullWhite, ParameterValidation) {
    EXPECT_THROW(irm::HullWhite(0.0, 0.01, market_curve()), std::invalid_argument);
    EXPECT_THROW(irm::HullWhite(-0.1, 0.01, market_curve()), std::invalid_argument);
    EXPECT_THROW(irm::HullWhite(0.1, -0.01, market_curve()), std::invalid_argument);
    EXPECT_NO_THROW(irm::HullWhite(0.1, 0.0, market_curve()));
}

TEST(HullWhite, Fwd0MatchesSegmentForwardInsideSegments) {
    const irm::HullWhite hw(0.1, 0.01, market_curve());
    const auto& c = hw.curve();
    // Away from pillars the FD equals the piecewise-constant segment forward.
    for (double t : {0.5, 1.5, 2.5, 4.0, 7.0}) {
        EXPECT_NEAR(hw.fwd0(t), c.inst_forward(t), 1e-9) << "t=" << t;
    }
    // Straddling a pillar: average of adjacent segment forwards.
    const double avg = 0.5 * (c.inst_forward(0.5) + c.inst_forward(1.5));
    EXPECT_NEAR(hw.fwd0(1.0), avg, 1e-9);
    // Short-end one-sided scheme: f(0,0) = -ln DF(h)/h.
    EXPECT_NEAR(hw.fwd0(0.0), -std::log(c.df(irm::kFdStep)) / irm::kFdStep, 1e-12);
    EXPECT_THROW(hw.fwd0(-1.0), std::invalid_argument);
}

TEST(HullWhite, ZcbAtTimeZeroReproducesMarketDfsExactly) {
    const irm::HullWhite hw(0.1, 0.01, market_curve());
    // Property loop over pillars and between: P(0,T) == market DF to 1e-12.
    for (double t : {0.5, 1.0, 2.0, 2.5, 3.0, 5.0, 7.5, 10.0}) {
        EXPECT_NEAR(hw.zcb_price(0.0, t), hw.curve().df(t), 1e-12) << "T=" << t;
    }
}

TEST(HullWhite, ThetaConsistentWithFlatCurve) {
    // Flat 3% curve: f = 0.03 everywhere, df/dt = 0, so
    // theta(t) = a f + (sigma^2/(2a))(1 - e^{-2at}).
    std::vector<double> ts, ps;
    for (double t = 0.5; t <= 20.0; t += 0.5) {
        ts.push_back(t);
        ps.push_back(std::exp(-0.03 * t));
    }
    const irm::HullWhite hw(0.1, 0.01, irm::DiscountCurve(ts, ps));
    const double t = 2.0;
    const double expect =
        0.1 * 0.03 + (0.01 * 0.01 / (2.0 * 0.1)) * (1.0 - std::exp(-2.0 * 0.1 * t));
    EXPECT_NEAR(hw.theta(t), expect, 1e-6);
}

TEST(HullWhite, BondOptionPutCallParityOnMarketDfs) {
    const irm::HullWhite hw(0.1, 0.01, market_curve());
    for (double k = 0.9; k <= 1.02; k += 0.02) {
        const double lhs = hw.zbc(1.0, 2.0, k) - hw.zbp(1.0, 2.0, k);
        const double rhs = hw.curve().df(2.0) - k * hw.curve().df(1.0);
        EXPECT_NEAR(lhs, rhs, 1e-14) << "strike " << k;
    }
    EXPECT_THROW(hw.zbc(1.0, 2.0, 0.0), std::invalid_argument);
    EXPECT_THROW(hw.zbp(2.0, 1.0, 0.9), std::invalid_argument);
}

TEST(HullWhite, CapEqualsSumOfCaplets) {
    const irm::HullWhite hw(0.1, 0.01, market_curve());
    const std::vector<double> sched = {0.5, 1.0, 2.0, 3.0};
    double sum = 0.0;
    for (std::size_t i = 0; i + 1 < sched.size(); ++i) {
        sum += hw.caplet(sched[i], sched[i + 1], 0.024, 100.0);
    }
    EXPECT_NEAR(hw.cap(sched, 0.024, 100.0), sum, 1e-12);
    EXPECT_THROW(hw.cap({1.0}, 0.02), std::invalid_argument);
}

TEST(HullWhite, CapletFloorletParityAndValidation) {
    const irm::HullWhite hw(0.1, 0.01, market_curve());
    // caplet - floorlet = PV(payer FRA at K): DF(pay) tau (F - K).
    const double tau = 1.0, k = 0.025;
    const double f = hw.curve().fwd_rate(1.0, 2.0);
    const double lhs = hw.caplet(1.0, 2.0, k, 1.0) - hw.floorlet(1.0, 2.0, k, 1.0);
    EXPECT_NEAR(lhs, hw.curve().df(2.0) * tau * (f - k), 1e-13);
    EXPECT_THROW(hw.caplet(2.0, 1.0, 0.02), std::invalid_argument);
    EXPECT_THROW(hw.caplet(-1.0, 1.0, 0.02), std::invalid_argument);
    EXPECT_THROW(hw.caplet(1.0, 2.0, -1.5), std::invalid_argument);  // 1 + K tau <= 0
}

TEST(HullWhite, SigmaZeroCapletIsDiscountedIntrinsic) {
    const irm::HullWhite det(0.1, 0.0, market_curve());
    const double f = det.curve().fwd_rate(1.0, 2.0);
    const double k = 0.02;
    const double expect = det.curve().df(2.0) * 1.0 * std::max(f - k, 0.0);
    EXPECT_NEAR(det.caplet(1.0, 2.0, k, 1.0), expect, 1e-14);
    // OTM caplet is exactly zero when sigma = 0.
    EXPECT_EQ(det.caplet(1.0, 2.0, f + 0.01, 1.0), 0.0);
}

TEST(HullWhite, JamshidianSwaptionProperties) {
    const irm::HullWhite hw(0.1, 0.01, market_curve());
    const double expiry = 1.0;
    const std::vector<double> pay = {2.0, 3.0, 4.0, 5.0, 6.0};
    const double fixed = 0.028;
    const auto res = hw.jamshidian_swaption(expiry, pay, fixed, 100.0);
    // Root residual < 1e-10 as required by the spec.
    EXPECT_LT(std::fabs(res.residual), 1e-10);
    ASSERT_EQ(res.strikes.size(), pay.size());
    // Decomposition strikes are ZCB prices at r*.
    for (std::size_t i = 0; i < pay.size(); ++i) {
        EXPECT_NEAR(res.strikes[i], hw.zcb_price(expiry, pay[i], res.r_star), 1e-15);
    }
    // Payer swaption >= max(forward payer swap PV, 0) (intrinsic bound).
    std::vector<double> par_times = {expiry};
    par_times.insert(par_times.end(), pay.begin(), pay.end());
    double annuity = 0.0, prev = expiry;
    for (double t : pay) {
        annuity += (t - prev) * hw.curve().df(t);
        prev = t;
    }
    const double swap_pv =
        100.0 * ((hw.curve().df(expiry) - hw.curve().df(pay.back())) - fixed * annuity);
    EXPECT_GE(res.value, std::max(swap_pv, 0.0) - 1e-12);
    EXPECT_GT(res.value, 0.0);
    // Payer/receiver parity: payer - receiver = swap PV.
    const auto rec = hw.jamshidian_swaption(expiry, pay, fixed, 100.0, /*payer=*/false);
    EXPECT_NEAR(res.value - rec.value, swap_pv, 1e-9);
}

TEST(HullWhite, JamshidianOnNegativeCurve) {
    const irm::HullWhite hw(0.1, 0.01, negative_curve());
    const auto res = hw.jamshidian_swaption(1.0, {2.0, 3.0, 4.0}, 0.0, 100.0);
    EXPECT_LT(std::fabs(res.residual), 1e-10);
    EXPECT_GT(res.value, 0.0);
    // Negative-rate environment: r* can be negative — just require finite.
    EXPECT_TRUE(std::isfinite(res.r_star));
}

TEST(HullWhite, JamshidianValidation) {
    const irm::HullWhite hw(0.1, 0.01, market_curve());
    EXPECT_THROW(hw.jamshidian_swaption(0.0, {2.0}, 0.02), std::invalid_argument);
    EXPECT_THROW(hw.jamshidian_swaption(1.0, {}, 0.02), std::invalid_argument);
    EXPECT_THROW(hw.jamshidian_swaption(1.0, {0.5}, 0.02), std::invalid_argument);
    EXPECT_THROW(hw.jamshidian_swaption(1.0, {2.0, 2.0}, 0.02), std::invalid_argument);
    EXPECT_THROW(hw.jamshidian_swaption(1.0, {2.0, INFINITY}, 0.02), std::invalid_argument);
}

// ---- validation of notional / strike / schedule (MAJOR-5) -----------------

TEST(HullWhite, NonFiniteNotionalAndMcStrikeRejected) {
    const irm::HullWhite hw(0.1, 0.01, market_curve());
    const double nan = std::nan("");
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_THROW(hw.caplet(1.0, 2.0, 0.025, nan), std::invalid_argument);
    EXPECT_THROW(hw.floorlet(1.0, 2.0, 0.025, inf), std::invalid_argument);
    EXPECT_THROW(hw.cap({1.0, 2.0, 3.0}, 0.02, inf), std::invalid_argument);
    EXPECT_THROW(hw.jamshidian_swaption(1.0, {2.0, 3.0}, 0.02, nan), std::invalid_argument);
    EXPECT_THROW(hw.mc_caplet(1.0, 2.0, nan, 100.0, 8, 100, 1), std::invalid_argument);
    EXPECT_THROW(hw.mc_caplet(1.0, 2.0, -1.5, 100.0, 8, 100, 1), std::invalid_argument);
    EXPECT_THROW(hw.mc_caplet(1.0, 2.0, 0.02, inf, 8, 100, 1), std::invalid_argument);
    EXPECT_THROW(hw.caplet(nan, 2.0, 0.02), std::invalid_argument);
    EXPECT_THROW(hw.b_factor(nan, 5.0), std::invalid_argument);
    EXPECT_THROW(hw.b_factor(-1.0, 5.0), std::invalid_argument);
    EXPECT_THROW(hw.theta(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(hw.zcb_price(1.0, 5.0, -1e6), std::invalid_argument);  // exp overflow
    // A negative notional is a short position, not an error.
    EXPECT_NEAR(hw.caplet(1.0, 2.0, 0.025, -100.0), -hw.caplet(1.0, 2.0, 0.025, 100.0), 1e-15);
}

TEST(HullWhite, CapScheduleNonIncreasingRejected) {
    const irm::HullWhite hw(0.1, 0.01, market_curve());
    EXPECT_THROW(hw.cap({1.0, 2.0, 2.0, 3.0}, 0.02), std::invalid_argument);
    EXPECT_THROW(hw.cap({1.0, std::nan(""), 3.0}, 0.02), std::invalid_argument);
    EXPECT_THROW(hw.cap({-1.0, 1.0}, 0.02), std::invalid_argument);
    EXPECT_THROW(hw.cap({1.0, 2.0}, std::nan("")), std::invalid_argument);
}

// ---- Jamshidian contract (MAJOR-6, MINOR-4) ---------------------------------

TEST(HullWhite, JamshidianResidualContractEnforced) {
    const irm::HullWhite hw(0.1, 0.01, market_curve());
    const std::vector<double> pays = {2.0, 3.0, 4.0, 5.0, 6.0};
    const auto res = hw.jamshidian_swaption(1.0, pays, 0.028, 100.0);
    const auto again = hw.jamshidian_at_r_star(1.0, pays, 0.028, res.r_star, 100.0);
    EXPECT_NEAR(again.value, res.value, 1e-14);
    EXPECT_EQ(again.residual, res.residual);
    EXPECT_LT(std::fabs(again.residual), irm::kJamshidianResidualTol);
    // A wrong r* violates |g(r*)| < 1e-10 -> domain_error.
    EXPECT_THROW(hw.jamshidian_at_r_star(1.0, pays, 0.028, -1.0, 100.0), std::domain_error);
    EXPECT_THROW(hw.jamshidian_at_r_star(1.0, pays, 0.028, res.r_star + 1e-6, 100.0),
                 std::domain_error);
    EXPECT_THROW(hw.jamshidian_at_r_star(1.0, pays, 0.028, std::nan("")), std::invalid_argument);
    const auto rec = hw.jamshidian_at_r_star(1.0, pays, 0.028, res.r_star, 100.0, false);
    EXPECT_NEAR(rec.value, hw.jamshidian_swaption(1.0, pays, 0.028, 100.0, false).value, 1e-14);
}

/// Payer swaption per unit notional by direct integration of the payoff
/// (1 - g(r_T))^+ under the T-forward Gaussian law of r_T (Simpson from the
/// payoff kink to mu + 12 sd) — independent of the decomposition.
double swaption_by_integration(const irm::HullWhite& hw, double expiry,
                               const std::vector<double>& pays, double x, int n = 2000) {
    const irm::DiscountCurve& c = hw.curve();
    const double v = hw.sigma() * hw.sigma() * (-std::expm1(-2.0 * hw.a() * expiry)) /
                     (2.0 * hw.a());
    const double b1 = hw.b_factor(expiry, pays[0]);
    const double a1 = hw.a_factor(expiry, pays[0]);
    const double mu = (std::log(a1 * c.df(expiry) / c.df(pays[0])) + b1 * b1 * v / 2.0) / b1;
    std::vector<double> coupons;
    double prev = expiry;
    for (double t : pays) {
        coupons.push_back(x * (t - prev));
        prev = t;
    }
    coupons.back() += 1.0;
    const double sd = std::sqrt(v);
    const auto payoff = [&](double r) {
        double g = 0.0;
        for (std::size_t i = 0; i < pays.size(); ++i) g += coupons[i] * hw.zcb_price(expiry, pays[i], r);
        return std::max(1.0 - g, 0.0);
    };
    double lo = mu - 12.0 * sd, hi = mu + 12.0 * sd;
    while (lo < mu - 400.0 * sd || payoff(lo) > 0.0) lo -= 12.0 * sd;
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (payoff(mid) > 0.0) hi = mid; else lo = mid;
    }
    const double kink = hi, top = mu + 12.0 * sd;
    const double h = (top - kink) / n;
    double total = 0.0;
    for (int i = 0; i <= n; ++i) {
        const double r = kink + i * h;
        const double w = std::exp(-0.5 * ((r - mu) / sd) * ((r - mu) / sd)) /
                         (sd * std::sqrt(2.0 * M_PI));
        const double coef = (i == 0 || i == n) ? 1.0 : ((i % 2 == 1) ? 4.0 : 2.0);
        total += coef * payoff(r) * w;
    }
    return c.df(expiry) * total * h / 3.0;
}

TEST(HullWhite, JamshidianNegativeFixedRateStaysExact) {
    // For X < 0 the coupon signs are (-, ..., -, +): g is not monotone but
    // g - 1 still has a single crossing, so the decomposition is exact;
    // 1 + X tau_n <= 0 is the genuinely degenerate case.
    const irm::HullWhite hw(0.1, 0.01, market_curve());
    const std::vector<double> pays = {2.0, 3.0, 4.0, 5.0, 6.0};
    for (double x : {-0.005, -0.2, -0.5}) {
        const auto res = hw.jamshidian_swaption(1.0, pays, x, 1.0);
        EXPECT_GT(res.value, 0.0);
        EXPECT_LT(std::fabs(res.residual), 1e-10);
        EXPECT_NEAR(res.value / swaption_by_integration(hw, 1.0, pays, x), 1.0, 1e-9) << "X=" << x;
    }
    EXPECT_LT(hw.jamshidian_swaption(1.0, pays, -0.005).r_star, 0.0);
    EXPECT_THROW(hw.jamshidian_swaption(1.0, pays, -1.0), std::invalid_argument);
    EXPECT_THROW(hw.jamshidian_swaption(1.0, pays, -1.5), std::invalid_argument);
    const auto ok = hw.jamshidian_swaption(1.0, pays, -0.8);
    EXPECT_LT(std::fabs(ok.residual), 1e-10);
    // Beyond about -0.95 the coupon-bond terms are O(1e8+) and |g(r*)| cannot
    // beat 1e-10 in double precision: the contract rejects it honestly.
    EXPECT_THROW(hw.jamshidian_swaption(1.0, pays, -0.99), std::domain_error);
    // Positive fixed rate agrees with integration too.
    const auto pos = hw.jamshidian_swaption(1.0, pays, 0.028);
    EXPECT_NEAR(pos.value / swaption_by_integration(hw, 1.0, pays, 0.028), 1.0, 1e-9);
}

// ---- Monte Carlo determinism / small mean reversion --------------------------

TEST(HullWhite, McCapletSameSeedAndSinglePath) {
    const irm::HullWhite hw(0.1, 0.01, market_curve());
    const auto a = hw.mc_caplet(1.0, 2.0, 0.025, 100.0, 4, 500, 3);
    const auto b = hw.mc_caplet(1.0, 2.0, 0.025, 100.0, 4, 500, 3);
    const auto c = hw.mc_caplet(1.0, 2.0, 0.025, 100.0, 4, 500, 4);
    EXPECT_EQ(a.first, b.first);
    EXPECT_EQ(a.second, b.second);
    EXPECT_NE(a.first, c.first);
    const auto single = hw.mc_caplet(1.0, 2.0, 0.02, 100.0, 1, 1, 1);
    EXPECT_TRUE(std::isfinite(single.first));
    EXPECT_GE(single.first, 0.0);
    EXPECT_EQ(single.second, 0.0);
}

TEST(HullWhite, McCapletTinyMeanReversionUnbiased) {
    // a*dt = 5e-7 per step exercises the series branch of var_i and V(t).
    const irm::HullWhite slow(1e-4, 0.01, market_curve());
    const double analytic = slow.caplet(1.0, 2.0, 0.025, 100.0);
    const auto [price, se] = slow.mc_caplet(1.0, 2.0, 0.025, 100.0, 200, 20000, 9);
    EXPECT_NEAR(price, analytic, 3.0 * se);
}

TEST(HullWhite, VarianceIntegralUsesCancellationFreeKernel) {
    const irm::HullWhite slow(1e-5, 0.02, market_curve());
    EXPECT_NEAR(slow.variance_integral(1.0), 0.5 * irm::ou_integral_variance(1e-5, 0.02, 1.0),
                1e-20);
    EXPECT_NEAR(slow.variance_integral(1.0) / (0.02 * 0.02 / 6.0), 1.0, 1e-5);
    EXPECT_EQ(slow.variance_integral(0.0), 0.0);
}

// ---- calibration of (a, sigma) (MAJOR-7) --------------------------------------

void make_quotes(const irm::HullWhite& truth, std::vector<irm::CapletQuote>& caps,
                 std::vector<irm::SwaptionQuote>& swps) {
    for (double r : {1.0, 2.0, 4.0}) {
        caps.emplace_back(r, r + 1.0, 0.025, truth.caplet(r, r + 1.0, 0.025, 1.0));
    }
    const std::vector<double> pays = {2.0, 3.0, 4.0, 5.0, 6.0};
    swps.emplace_back(1.0, pays, 0.028, truth.jamshidian_swaption(1.0, pays, 0.028, 1.0).value);
}

TEST(HullWhiteCalibration, RecoversAAndSigma) {
    const irm::HullWhite truth(0.08, 0.012, market_curve());
    std::vector<irm::CapletQuote> caps;
    std::vector<irm::SwaptionQuote> swps;
    make_quotes(truth, caps, swps);
    const auto cal = irm::calibrate_hullwhite(market_curve(), caps, swps);
    EXPECT_TRUE(cal.converged);
    EXPECT_TRUE(cal.identified);
    EXPECT_FALSE(cal.at_bound);
    EXPECT_FALSE(cal.a_fixed);
    EXPECT_EQ(cal.n_starts, 3);
    EXPECT_NEAR(cal.a, 0.08, 1e-8);
    EXPECT_NEAR(cal.sigma, 0.012, 1e-9);
    EXPECT_LT(cal.rmse, 1e-10);
    const irm::HullWhite model = cal.model(market_curve());
    for (const auto& q : caps) {
        EXPECT_NEAR(model.caplet(q.reset, q.pay, q.strike, 1.0), q.price, 1e-10);
    }
}

TEST(HullWhiteCalibration, AFixedAndWarmStart) {
    const irm::HullWhite truth(0.08, 0.012, market_curve());
    std::vector<irm::CapletQuote> caps;
    std::vector<irm::SwaptionQuote> swps;
    make_quotes(truth, caps, swps);
    const auto cal = irm::calibrate_hullwhite(market_curve(), caps, swps, 0.08);
    EXPECT_TRUE(cal.a_fixed);
    EXPECT_EQ(cal.a, 0.08);
    EXPECT_TRUE(cal.converged);
    EXPECT_TRUE(cal.identified);
    EXPECT_NEAR(cal.sigma, 0.012, 1e-9);
    const auto warm = irm::calibrate_hullwhite(market_curve(), caps, swps, std::nullopt,
                                               std::vector<double>{0.2, 0.02});
    EXPECT_EQ(warm.n_starts, 1);
    EXPECT_FALSE(warm.identified);
    EXPECT_TRUE(warm.converged);
    EXPECT_NEAR(warm.a, 0.08, 1e-6);
    EXPECT_NEAR(warm.sigma, 0.012, 1e-8);
    const auto only_caps = irm::calibrate_hullwhite(market_curve(), caps);
    EXPECT_NEAR(only_caps.sigma, 0.012, 1e-8);
    const auto only_swp = irm::calibrate_hullwhite(market_curve(), {}, swps, 0.08);
    EXPECT_NEAR(only_swp.sigma, 0.012, 1e-8);
}

TEST(HullWhiteCalibration, SingleQuoteNotIdentified) {
    const irm::HullWhite truth(0.08, 0.012, market_curve());
    std::vector<irm::CapletQuote> caps;
    std::vector<irm::SwaptionQuote> swps;
    make_quotes(truth, caps, swps);
    caps.erase(caps.begin() + 1, caps.end());
    const auto cal = irm::calibrate_hullwhite(market_curve(), caps);
    EXPECT_TRUE(cal.converged);
    EXPECT_FALSE(cal.identified);  // one price, two parameters: a ridge
    EXPECT_GT(cal.a_spread, 1e-2);
    EXPECT_LT(cal.rmse, 1e-9);
}

TEST(HullWhiteCalibration, NonConvergenceAndValidation) {
    const irm::HullWhite truth(0.08, 0.012, market_curve());
    std::vector<irm::CapletQuote> caps;
    std::vector<irm::SwaptionQuote> swps;
    make_quotes(truth, caps, swps);
    const auto res = irm::calibrate_hullwhite(market_curve(), caps, swps, std::nullopt,
                                              std::nullopt, /*maxiter=*/2);
    EXPECT_FALSE(res.converged);
    EXPECT_TRUE(std::isfinite(res.rmse));
    EXPECT_THROW(irm::calibrate_hullwhite(market_curve(), {}), std::invalid_argument);
    EXPECT_THROW(irm::calibrate_hullwhite(market_curve(), caps, swps, 0.0), std::invalid_argument);
    EXPECT_THROW(irm::calibrate_hullwhite(market_curve(), caps, swps, std::nullopt,
                                          std::vector<double>{0.1}),
                 std::invalid_argument);
    EXPECT_THROW(irm::calibrate_hullwhite(market_curve(), caps, swps, std::nullopt,
                                          std::vector<double>{std::nan(""), 0.01}),
                 std::invalid_argument);
    EXPECT_THROW(irm::CapletQuote(1.0, 2.0, 0.02, std::nan("")), std::invalid_argument);
    EXPECT_THROW(irm::CapletQuote(2.0, 1.0, 0.02, 0.001), std::invalid_argument);
    EXPECT_THROW(irm::CapletQuote(1.0, 2.0, 0.02, -0.001), std::invalid_argument);
    EXPECT_THROW(irm::SwaptionQuote(1.0, {0.5, 2.0}, 0.02, 0.001), std::invalid_argument);
    EXPECT_THROW(irm::SwaptionQuote(1.0, {}, 0.02, 0.001), std::invalid_argument);
    EXPECT_THROW(irm::SwaptionQuote(INFINITY, {2.0}, 0.02, 0.001), std::invalid_argument);
}

TEST(HullWhite, McCapletWithin3StandardErrors) {
    const irm::HullWhite hw(0.1, 0.01, market_curve());
    const double analytic = hw.caplet(1.0, 2.0, 0.025, 100.0);
    const auto [price, se] =
        hw.mc_caplet(1.0, 2.0, 0.025, 100.0, /*n_steps=*/8, /*n_paths=*/50000, /*seed=*/77);
    EXPECT_GT(se, 0.0);
    EXPECT_NEAR(price, analytic, 3.0 * se);
}

TEST(HullWhite, McCapletSigmaZeroIsExactIntrinsic) {
    const irm::HullWhite det(0.1, 0.0, market_curve());
    const auto [price, se] = det.mc_caplet(1.0, 2.0, 0.02, 100.0, 4, 100, 3);
    EXPECT_EQ(se, 0.0);
    EXPECT_NEAR(price, det.caplet(1.0, 2.0, 0.02, 100.0), 1e-12);
}

TEST(HullWhite, McCapletValidation) {
    const irm::HullWhite hw(0.1, 0.01, market_curve());
    EXPECT_THROW(hw.mc_caplet(0.0, 1.0, 0.02, 1.0, 8, 10, 1), std::invalid_argument);
    EXPECT_THROW(hw.mc_caplet(2.0, 1.0, 0.02, 1.0, 8, 10, 1), std::invalid_argument);
    EXPECT_THROW(hw.mc_caplet(1.0, 2.0, 0.02, 1.0, 0, 10, 1), std::invalid_argument);
}

}  // namespace
