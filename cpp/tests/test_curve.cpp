// Tests for the discount curve: construction validation, interpolation
// exactness, extrapolation, derived rates, negative-rate support.

#include "irm/curve.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {

irm::DiscountCurve sample_curve() {
    return irm::DiscountCurve({1.0, 2.0, 3.0, 5.0}, {0.99, 0.975, 0.955, 0.9});
}

TEST(Curve, ConstructionValidation) {
    EXPECT_THROW(irm::DiscountCurve({}, {}), std::invalid_argument);
    EXPECT_THROW(irm::DiscountCurve({1.0, 2.0}, {0.99}), std::invalid_argument);
    EXPECT_THROW(irm::DiscountCurve({0.0}, {1.0}), std::invalid_argument);   // t <= 0
    EXPECT_THROW(irm::DiscountCurve({-1.0}, {1.0}), std::invalid_argument);  // t < 0
    EXPECT_THROW(irm::DiscountCurve({1.0, 1.0}, {0.99, 0.98}), std::invalid_argument);
    EXPECT_THROW(irm::DiscountCurve({2.0, 1.0}, {0.99, 0.98}), std::invalid_argument);
    EXPECT_THROW(irm::DiscountCurve({1.0}, {0.0}), std::invalid_argument);   // DF <= 0
    EXPECT_THROW(irm::DiscountCurve({1.0}, {-0.5}), std::invalid_argument);
    EXPECT_THROW(irm::DiscountCurve({1.0}, {std::nan("")}), std::invalid_argument);
    EXPECT_THROW(irm::DiscountCurve({std::numeric_limits<double>::infinity()}, {0.9}),
                 std::invalid_argument);
}

TEST(Curve, InterpolationExactAtPillars) {
    const auto c = sample_curve();
    const std::vector<double> ts = {1.0, 2.0, 3.0, 5.0};
    const std::vector<double> ps = {0.99, 0.975, 0.955, 0.9};
    for (std::size_t i = 0; i < ts.size(); ++i) {
        EXPECT_DOUBLE_EQ(c.df(ts[i]), ps[i]) << "pillar " << ts[i];
    }
    EXPECT_DOUBLE_EQ(c.df(0.0), 1.0);
}

TEST(Curve, LogLinearMidpointAndQueryErrors) {
    const auto c = sample_curve();
    // ln DF linear between pillars.
    const double expect_mid = std::exp(0.5 * (std::log(0.975) + std::log(0.955)));
    EXPECT_NEAR(c.df(2.5), expect_mid, 1e-15);
    EXPECT_THROW(c.df(-0.1), std::invalid_argument);
    EXPECT_THROW(c.df(std::nan("")), std::invalid_argument);
    EXPECT_THROW(c.zero_rate(-1.0), std::invalid_argument);
    EXPECT_THROW(c.inst_forward(-1.0), std::invalid_argument);
}

TEST(Curve, FlatForwardExtrapolation) {
    const auto c = sample_curve();
    // Last segment forward carried on: DF(7) = DF(5) * exp(-2 * f_last).
    const double f_last = std::log(0.955 / 0.9) / 2.0;
    EXPECT_NEAR(c.df(7.0), 0.9 * std::exp(-2.0 * f_last), 1e-15);
    EXPECT_NEAR(c.inst_forward(10.0), f_last, 1e-15);
}

TEST(Curve, ZeroRateDefinitionAndLimit) {
    const auto c = sample_curve();
    EXPECT_NEAR(c.zero_rate(2.5), -std::log(c.df(2.5)) / 2.5, 1e-16);
    // t = 0 returns the first segment's instantaneous forward.
    EXPECT_NEAR(c.zero_rate(0.0), -std::log(0.99) / 1.0, 1e-15);
    EXPECT_NEAR(c.zero_rate(0.0), c.inst_forward(0.0), 1e-15);
}

TEST(Curve, InstForwardUsesRightSegmentAtPillar) {
    const auto c = sample_curve();
    // At t = 2 the segment [2, 3] applies.
    const double f_right = std::log(0.975 / 0.955) / 1.0;
    EXPECT_NEAR(c.inst_forward(2.0), f_right, 1e-15);
}

TEST(Curve, ForwardRateConsistencyWithDfs) {
    const auto c = sample_curve();
    // Property-style: F(t1,t2) matches its DF definition across a grid.
    for (double t1 = 0.0; t1 < 6.0; t1 += 0.7) {
        for (double dt = 0.3; dt <= 2.1; dt += 0.6) {
            const double t2 = t1 + dt;
            const double expect = (c.df(t1) / c.df(t2) - 1.0) / (t2 - t1);
            EXPECT_NEAR(c.fwd_rate(t1, t2), expect, 1e-14);
        }
    }
    EXPECT_THROW(c.fwd_rate(2.0, 2.0), std::invalid_argument);
    EXPECT_THROW(c.fwd_rate(-1.0, 2.0), std::invalid_argument);
    EXPECT_THROW(c.fwd_rate(3.0, 2.0), std::invalid_argument);
}

TEST(Curve, NegativeRatesDfAboveOneSupported) {
    // EUR/JPY style: DF > 1 legal, zero rates negative.
    const irm::DiscountCurve c({1.0, 2.0}, {1.005, 1.006});
    EXPECT_GT(c.df(1.0), 1.0);
    EXPECT_GT(c.df(1.5), 1.0);
    EXPECT_LT(c.zero_rate(1.0), 0.0);
    EXPECT_LT(c.inst_forward(0.5), 0.0);
}

TEST(Curve, DfPositiveAndDecreasingForPositiveRates) {
    const auto c = sample_curve();
    double prev = c.df(0.0);
    for (double t = 0.25; t <= 8.0; t += 0.25) {
        const double d = c.df(t);
        EXPECT_GT(d, 0.0);
        EXPECT_LT(d, prev);
        prev = d;
    }
}

TEST(Curve, ParSwapRate) {
    const auto c = sample_curve();
    // Definition check.
    const std::vector<double> times = {0.0, 1.0, 2.0, 3.0};
    double annuity = 0.0;
    double prev = 0.0;
    for (std::size_t i = 1; i < times.size(); ++i) {
        annuity += (times[i] - prev) * c.df(times[i]);
        prev = times[i];
    }
    EXPECT_NEAR(irm::par_swap_rate(c, times), (1.0 - c.df(3.0)) / annuity, 1e-15);
    EXPECT_THROW(irm::par_swap_rate(c, {1.0}), std::invalid_argument);
    EXPECT_THROW(irm::par_swap_rate(c, {-1.0, 1.0}), std::invalid_argument);
    EXPECT_THROW(irm::par_swap_rate(c, {0.0, 2.0, 1.0}), std::invalid_argument);
}

TEST(Curve, FwdRateUnderflowIsStandardError) {
    const irm::DiscountCurve c({1.0}, {0.95});
    EXPECT_THROW(c.fwd_rate(0.0, 1e6), std::invalid_argument);  // DF(1e6) underflows to 0
    EXPECT_TRUE(std::isfinite(c.fwd_rate(0.0, 500.0)));
}

}  // namespace
