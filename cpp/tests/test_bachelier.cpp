// Tests for the Bachelier (normal) model: price, implied vol, parity.

#include "irm/bachelier.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

namespace {

TEST(Bachelier, AtmClosedForm) {
    // ATM: price = A sigma sqrt(T) / sqrt(2 pi).
    const double a = 0.98, vol = 0.005, t = 1.0;
    const double p = irm::bachelier_price(-0.002, -0.002, t, vol, a, true);
    EXPECT_NEAR(p, a * vol * std::sqrt(t) / std::sqrt(2.0 * M_PI), 1e-17);
    EXPECT_NEAR(irm::bachelier_price(-0.002, -0.002, t, vol, a, false), p, 1e-17);
    EXPECT_NEAR(irm::bachelier_implied_vol(p, -0.002, -0.002, t, a, true), vol, 1e-14);
}

TEST(Bachelier, ImpliedVolRoundTripGrid) {
    const double t = 2.5, annuity = 3.7;
    for (double forward : {-0.01, 0.0, 0.02, 0.05}) {
        for (double strike : {-0.02, 0.0, 0.03}) {
            for (double vol : {0.001, 0.0075, 0.02}) {
                for (bool payer : {true, false}) {
                    const double p = irm::bachelier_price(forward, strike, t, vol, annuity, payer);
                    const double iv = irm::bachelier_implied_vol(p, forward, strike, t, annuity, payer);
                    const double d = std::fabs(forward - strike) / (vol * std::sqrt(t));
                    if (d < 4.0) {
                        EXPECT_NEAR(iv, vol, 1e-12) << forward << " " << strike << " " << vol;
                    } else {
                        // Time value below rounding: only the price round-trips.
                        EXPECT_GE(iv, 0.0);
                        EXPECT_LE(iv, vol * 1.001);
                    }
                    EXPECT_NEAR(irm::bachelier_price(forward, strike, t, iv, annuity, payer), p, 1e-15);
                }
            }
        }
    }
}

TEST(Bachelier, PayerReceiverParityAndMonotonicity) {
    const double f = 0.03, k = 0.025, t = 2.0, a = 4.2;
    for (double vol : {0.0, 0.002, 0.0075, 0.03}) {
        const double payer = irm::bachelier_price(f, k, t, vol, a, true);
        const double receiver = irm::bachelier_price(f, k, t, vol, a, false);
        EXPECT_NEAR(payer - receiver, a * (f - k), 1e-15);
    }
    double prev = -1.0;
    for (double vol : {0.001, 0.002, 0.005, 0.01, 0.02}) {
        const double p = irm::bachelier_price(f, k, t, vol, a, true);
        EXPECT_GT(p, prev);
        prev = p;
    }
}

TEST(Bachelier, IntrinsicAndBelowIntrinsic) {
    const double f = 0.03, k = 0.025, t = 2.0, a = 4.2;
    const double intrinsic = a * (f - k);
    EXPECT_EQ(irm::bachelier_price(f, k, t, 0.0, a, true), intrinsic);
    EXPECT_EQ(irm::bachelier_price(f, k, t, 0.0, a, false), 0.0);
    EXPECT_EQ(irm::bachelier_implied_vol(intrinsic, f, k, t, a, true), 0.0);
    EXPECT_EQ(irm::bachelier_implied_vol(0.0, f, k, t, a, false), 0.0);
    EXPECT_THROW(irm::bachelier_implied_vol(intrinsic - 1e-6, f, k, t, a, true),
                 std::invalid_argument);
    EXPECT_THROW(irm::bachelier_implied_vol(-1e-9, f, k, t, a, false), std::invalid_argument);
}

TEST(Bachelier, Validation) {
    const double nan = std::nan("");
    EXPECT_THROW(irm::bachelier_price(nan, 0.02, 1.0, 0.01, 1.0), std::invalid_argument);
    EXPECT_THROW(irm::bachelier_price(0.02, 0.02, 0.0, 0.01, 1.0), std::invalid_argument);
    EXPECT_THROW(irm::bachelier_price(0.02, 0.02, 1.0, -0.01, 1.0), std::invalid_argument);
    EXPECT_THROW(irm::bachelier_price(0.02, 0.02, 1.0, 0.01, 0.0), std::invalid_argument);
    EXPECT_THROW(irm::bachelier_implied_vol(nan, 0.02, 0.02, 1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(irm::bachelier_implied_vol(0.01, 0.02, 0.02,
                                            std::numeric_limits<double>::infinity(), 1.0),
                 std::invalid_argument);
    EXPECT_THROW(irm::bachelier_implied_vol(1e300, 0.02, 0.02, 1.0, 1.0), std::invalid_argument);
}

}  // namespace
