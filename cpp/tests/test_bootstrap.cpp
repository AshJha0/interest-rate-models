// Tests for the sequential bootstrap: round-trip repricing of every input
// instrument, schedule generation, validation and arbitrage detection.

#include "irm/bootstrap.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "irm/quotes.hpp"

namespace {

const std::string kDataDir = IRM_DATA_DIR;

TEST(AnnualSchedule, KnownCases) {
    EXPECT_EQ(irm::annual_schedule(2.0), (std::vector<double>{1.0, 2.0}));
    EXPECT_EQ(irm::annual_schedule(0.25), (std::vector<double>{0.25}));
    const auto stub = irm::annual_schedule(2.5);
    ASSERT_EQ(stub.size(), 3u);
    EXPECT_NEAR(stub[0], 0.5, 1e-15);
    EXPECT_NEAR(stub[1], 1.5, 1e-15);
    EXPECT_NEAR(stub[2], 2.5, 1e-15);
    EXPECT_THROW(irm::annual_schedule(0.0), std::invalid_argument);
    EXPECT_THROW(irm::annual_schedule(-1.0), std::invalid_argument);
}

TEST(Instruments, ConstructionValidation) {
    EXPECT_THROW(irm::Deposit(0.0, 0.05), std::invalid_argument);
    EXPECT_THROW(irm::Deposit(1.0, std::nan("")), std::invalid_argument);
    // 1 + R*T <= 0 implies a non-positive DF.
    EXPECT_THROW(irm::Deposit(2.0, -0.5), std::invalid_argument);
    EXPECT_THROW(irm::FRA(-0.5, 1.0, 0.02), std::invalid_argument);
    EXPECT_THROW(irm::FRA(1.0, 1.0, 0.02), std::invalid_argument);
    EXPECT_THROW(irm::FRA(1.0, 2.0, -1.5), std::invalid_argument);
    EXPECT_THROW(irm::Swap(-2.0, 0.02), std::invalid_argument);
    EXPECT_THROW(irm::OISSwap(2.0, std::nan("")), std::invalid_argument);
}

TEST(Bootstrap, EmptyAndOrderingViolationsThrow) {
    EXPECT_THROW(irm::bootstrap({}), std::invalid_argument);
    // Duplicate pillar.
    EXPECT_THROW(irm::bootstrap({irm::Deposit(1.0, 0.02), irm::Deposit(1.0, 0.021)}),
                 std::invalid_argument);
    // Out of order.
    EXPECT_THROW(irm::bootstrap({irm::Deposit(2.0, 0.02), irm::Deposit(1.0, 0.02)}),
                 std::invalid_argument);
}

TEST(Bootstrap, SingleDepositExactDf) {
    const auto curve = irm::bootstrap({irm::Deposit(1.0, 0.05)});
    EXPECT_NEAR(curve.df(1.0), 1.0 / 1.05, 1e-14);
}

TEST(Bootstrap, EurCurveRoundTripAndNegativeShortEnd) {
    const auto quotes = irm::load_curve_quotes(kDataDir + "/curve_quotes.csv", "EUR");
    const auto curve = irm::bootstrap(quotes);
    // Round trip: every instrument reprices to < 1e-9 (reference < 1e-15).
    for (const auto& ins : quotes) {
        EXPECT_LT(std::fabs(irm::instrument_residual(ins, curve)), 1e-9)
            << "pillar " << irm::instrument_pillar(ins);
    }
    // Negative rates at the short end: DF > 1 (assert supported!).
    EXPECT_GT(curve.df(1.0), 1.0);
    EXPECT_LT(curve.zero_rate(0.5), 0.0);
}

TEST(Bootstrap, UsdCurveRoundTripAndParConsistency) {
    const auto quotes = irm::load_curve_quotes(kDataDir + "/curve_quotes.csv", "USD");
    const auto curve = irm::bootstrap(quotes);
    for (const auto& ins : quotes) {
        EXPECT_LT(std::fabs(irm::instrument_residual(ins, curve)), 1e-9);
    }
    // The par-swap helper must reproduce the quoted 10y par rate.
    std::vector<double> times = {0.0};
    for (int i = 1; i <= 10; ++i) times.push_back(static_cast<double>(i));
    EXPECT_NEAR(irm::par_swap_rate(curve, times), 0.0425, 1e-10);
    // Positive-rate curve: DFs strictly decreasing across pillars.
    double prev = 1.0;
    for (double t : curve.times()) {
        EXPECT_LT(curve.df(t), prev);
        prev = curve.df(t);
    }
}

TEST(Bootstrap, OisCurveRoundTrip) {
    const auto quotes = irm::load_ois_quotes(kDataDir + "/ois_quotes.csv");
    const auto curve = irm::bootstrap(quotes);
    for (const auto& ins : quotes) {
        EXPECT_LT(std::fabs(irm::instrument_residual(ins, curve)), 1e-9);
    }
}

TEST(Bootstrap, GapPillarsHandledByInterpolation) {
    // Swap-only long end with missing intermediate pillars: the 5y swap's
    // 1..4y coupons interpolate on the trial curve.
    const auto curve = irm::bootstrap({irm::Deposit(1.0, 0.03), irm::Swap(5.0, 0.035)});
    const irm::Swap five(5.0, 0.035);
    EXPECT_LT(std::fabs(five.residual(curve)), 1e-9);
    ASSERT_EQ(curve.times().size(), 2u);
}

TEST(Bootstrap, CrossedQuotesDetected) {
    // A 300% 2y par rate cannot be repriced by any positive DF(2): the
    // implied DF is negative -> domain_error naming the pillar.
    try {
        irm::bootstrap({irm::Deposit(1.0, 0.05), irm::Swap(2.0, 3.0)});
        FAIL() << "expected domain_error";
    } catch (const std::domain_error& exc) {
        EXPECT_NE(std::string(exc.what()).find("t=2.0"), std::string::npos);
        EXPECT_NE(std::string(exc.what()).find("crossed"), std::string::npos);
    }
}

TEST(Bootstrap, NonConvergenceMessageNotCrossedQuotes) {
    // A solver failure that is not a bracketing failure must not be
    // relabelled as crossed quotes (MINOR-10).
    try {
        irm::solve_pillar_df([](double x) { return (x > 1.0) ? INFINITY : -1.0; }, 2.0, "test");
        FAIL() << "expected domain_error";
    } catch (const std::domain_error& exc) {
        const std::string msg = exc.what();
        EXPECT_NE(msg.find("solver failed"), std::string::npos);
        EXPECT_NE(msg.find("not finite"), std::string::npos);
        EXPECT_EQ(msg.find("crossed"), std::string::npos);
    }
    try {
        irm::solve_pillar_df([](double x) { return 1.0 + x; }, 2.0, "test");
        FAIL() << "expected domain_error";
    } catch (const std::domain_error& exc) {
        EXPECT_NE(std::string(exc.what()).find("crossed"), std::string::npos);
    }
    EXPECT_NEAR(irm::solve_pillar_df([](double x) { return x - 0.9; }, 2.0, "test"), 0.9, 1e-13);
}

TEST(Instruments, MaturityOutOfRangeRejected) {
    // 1e12 used to be UB (float -> int) and a multi-GB allocation.
    for (double bad : {1e12, 1e9, irm::kMaxMaturity * (1.0 + 1e-9), 1e-13,
                       irm::kMinMaturity / 2.0, std::numeric_limits<double>::infinity()}) {
        EXPECT_THROW(irm::Swap(bad, 0.03), std::invalid_argument) << bad;
        EXPECT_THROW(irm::OISSwap(bad, 0.03), std::invalid_argument) << bad;
        EXPECT_THROW(irm::Deposit(bad, 0.03), std::invalid_argument) << bad;
        EXPECT_THROW(irm::FRA(0.0, bad, 0.03), std::invalid_argument) << bad;
        EXPECT_THROW(irm::annual_schedule(bad), std::invalid_argument) << bad;
    }
    EXPECT_EQ(irm::annual_schedule(irm::kMaxMaturity).size(), 200u);
    EXPECT_EQ(irm::annual_schedule(irm::kMinMaturity), (std::vector<double>{irm::kMinMaturity}));
    EXPECT_EQ(irm::annual_schedule(199.5).size(), 200u);
}

TEST(Bootstrap, FraStripReprices) {
    const std::vector<irm::Instrument> ins = {irm::Deposit(0.5, 0.030), irm::FRA(0.5, 1.0, 0.032),
                                              irm::FRA(1.0, 1.5, 0.034), irm::Swap(3.0, 0.033)};
    const auto curve = irm::bootstrap(ins);
    for (const auto& i : ins) EXPECT_LT(std::fabs(irm::instrument_residual(i, curve)), 1e-9);
    EXPECT_NEAR(curve.fwd_rate(0.5, 1.0), 0.032, 1e-12);
}

TEST(Bootstrap, OisSubYearMaturity) {
    const auto c = irm::bootstrap({irm::OISSwap(1.0 / 12.0, 0.03)});
    EXPECT_NEAR(c.df(1.0 / 12.0), 1.0 / (1.0 + 0.03 / 12.0), 1e-12);
}

TEST(Bootstrap, Dv01LocalisesInQuoteSpace) {
    const auto quotes = irm::load_curve_quotes(kDataDir + "/curve_quotes.csv", "USD");
    std::vector<double> times = {0.0};
    for (int i = 1; i <= 10; ++i) times.push_back(static_cast<double>(i));
    const auto bumped = [&](double pillar) {
        std::vector<irm::Instrument> out;
        for (const auto& q : quotes) {
            const double bump = (irm::instrument_pillar(q) == pillar) ? 1e-4 : 0.0;
            if (std::holds_alternative<irm::Deposit>(q)) {
                const auto& d = std::get<irm::Deposit>(q);
                out.emplace_back(irm::Deposit(d.maturity, d.rate + bump));
            } else {
                const auto& s = std::get<irm::Swap>(q);
                out.emplace_back(irm::Swap(s.maturity, s.rate + bump));
            }
        }
        return irm::bootstrap(out);
    };
    EXPECT_NEAR(irm::par_swap_rate(irm::bootstrap(quotes), times), 0.0425, 1e-12);
    EXPECT_NEAR(irm::par_swap_rate(bumped(5.0), times), 0.0425, 1e-12);
    EXPECT_NEAR(irm::par_swap_rate(bumped(10.0), times) - 0.0425, 1e-4, 1e-12);
}

TEST(Quotes, LoadersValidate) {
    EXPECT_THROW(irm::load_curve_quotes(kDataDir + "/curve_quotes.csv", "GBP"),
                 std::invalid_argument);
    EXPECT_THROW(irm::load_curve_quotes(kDataDir + "/does_not_exist.csv", "EUR"),
                 std::invalid_argument);
    const auto ys = irm::load_zero_yields(kDataDir + "/zero_yields.csv");
    EXPECT_EQ(ys.first.size(), 10u);
    EXPECT_EQ(ys.first.size(), ys.second.size());
}

}  // namespace
