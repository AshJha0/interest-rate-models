// Tests for the Hull-White model: curve fit at t=0, the FD forward scheme,
// bond-option parity, caplet/cap decomposition, sigma=0 degeneration,
// Jamshidian swaptions and the exact-transition Monte Carlo.

#include "irm/hullwhite.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "irm/curve.hpp"

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
