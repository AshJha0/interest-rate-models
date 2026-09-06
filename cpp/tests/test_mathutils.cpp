// Tests for the shared kernels: normal CDF/PDF and the exact OU step
// moments.  Reference values for the integrated-OU variance come from a
// 60-digit evaluation of the closed form (python/tests/test_mathutils.py
// derives them with `decimal`; the same numbers are pinned here).

#include "irm/mathutils.hpp"

#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

namespace {

struct OuCase {
    double a, sigma, dt, var_i_ref, var_x_ref;
};

// var_i / var_x evaluated with 60-digit decimal arithmetic.
const OuCase kCases[] = {
    {0.001, 0.01, 0.01, 3.3333083334499993e-11, 9.9999000006666625e-07},
    {0.0001, 0.01, 0.001, 3.3333330833333452e-14, 9.9999990000000661e-08},
    {1e-06, 0.02, 0.125, 2.6041664225260560e-07, 4.9999993750000522e-05},
    {0.01, 0.01, 1.0, 3.3084495845603743e-05, 9.9006633466223494e-05},
    {0.05, 0.02, 0.01, 1.3328334499791697e-10, 3.9980006665000334e-06},
    {0.1, 0.01, 0.125, 6.4497359652182101e-08, 1.2345043985833665e-05},
    {0.8, 0.02, 2.5, 5.9493183361629550e-04, 2.4542109027781648e-04},
    {5.0, 0.01, 1.0, 2.8107625552266318e-06, 9.9995460007023755e-06},
};

TEST(OuMoments, SmallADtAccurate) {
    for (const OuCase& c : kCases) {
        const irm::OUStepMoments m = irm::ou_step_moments(c.a, c.sigma, c.dt);
        EXPECT_NEAR(m.var_i / c.var_i_ref, 1.0, 1e-9) << "a=" << c.a << " dt=" << c.dt;
        EXPECT_NEAR(m.var_x / c.var_x_ref, 1.0, 1e-12) << "a=" << c.a << " dt=" << c.dt;
        EXPECT_NEAR(m.b, -std::expm1(-c.a * c.dt) / c.a, 1e-14 * m.b);
        // Cauchy-Schwarz: the (x', I) covariance matrix must be PSD.
        EXPECT_LE(m.cov * m.cov, m.var_x * m.var_i * (1.0 + 1e-12));
        const double x = c.a * c.dt;
        if (x < 1e-3) {
            const double series = c.sigma * c.sigma * c.dt * c.dt * c.dt *
                                  (1.0 / 3.0 - x / 4.0 + 7.0 * x * x / 60.0 - x * x * x / 24.0);
            EXPECT_NEAR(m.var_i / series, 1.0, 1e-9);
        }
    }
}

TEST(OuMoments, NaiveFormulaIsWrongForTinyADt) {
    // Documents the bug the series fixes: plain evaluation is off by orders
    // of magnitude at a*dt = 1e-7 (80x per the review measurement).
    const double a = 1e-4, sigma = 0.01, dt = 1e-3;
    const double decay = std::exp(-a * dt);
    const double b = (1.0 - decay) / a;
    const double naive = (sigma * sigma / (a * a)) *
                         (dt - 2.0 * b + (1.0 - decay * decay) / (2.0 * a));
    const double ref = 3.3333330833333452e-14;
    EXPECT_GT(std::fabs(naive - ref) / ref, 10.0);
    EXPECT_NEAR(irm::ou_integral_variance(a, sigma, dt) / ref, 1.0, 1e-12);
}

TEST(OuMoments, SeriesAndClosedFormAgreeAtThreshold) {
    const double x = irm::kOuSeriesThreshold;
    const double below = irm::ou_integral_variance(x * (1.0 - 1e-12), 0.01, 1.0);
    const double above = irm::ou_integral_variance(x * (1.0 + 1e-12), 0.01, 1.0);
    EXPECT_NEAR(below / above, 1.0, 1e-9);
}

TEST(OuMoments, Validation) {
    EXPECT_THROW(irm::ou_step_moments(0.0, 0.01, 0.1), std::invalid_argument);
    EXPECT_THROW(irm::ou_step_moments(0.1, 0.01, 0.0), std::invalid_argument);
    EXPECT_THROW(irm::ou_step_moments(std::nan(""), 0.01, 0.1), std::invalid_argument);
    EXPECT_THROW(irm::ou_step_moments(0.1, -0.01, 0.1), std::invalid_argument);
    EXPECT_THROW(irm::ou_integral_variance(0.1, INFINITY, 0.1), std::invalid_argument);
    const irm::OUStepMoments m = irm::ou_step_moments(0.3, 0.0, 0.5);
    EXPECT_EQ(m.var_x, 0.0);
    EXPECT_EQ(m.var_i, 0.0);
    EXPECT_EQ(m.cov, 0.0);
    EXPECT_NEAR(m.decay, std::exp(-0.15), 1e-16);
}

TEST(Normal, CdfAndPdf) {
    EXPECT_EQ(irm::norm_cdf(0.0), 0.5);
    EXPECT_NEAR(irm::norm_cdf(1.0), 0.8413447460685429, 1e-15);
    EXPECT_NEAR(irm::norm_cdf(-8.0) / 6.220960574271786e-16, 1.0, 1e-10);
    EXPECT_NEAR(irm::norm_pdf(0.0), 1.0 / std::sqrt(2.0 * M_PI), 1e-16);
    EXPECT_NEAR(irm::norm_pdf(2.0), std::exp(-2.0) / std::sqrt(2.0 * M_PI), 1e-16);
}

}  // namespace
