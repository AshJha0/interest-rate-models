// Tests for the native Nelder-Mead minimiser.

#include "irm/optimize.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {

TEST(NelderMead, MinimisesShiftedQuadratic) {
    const auto f = [](const std::vector<double>& p) {
        const double dx = p[0] - 1.5, dy = p[1] + 2.0;
        return dx * dx + 3.0 * dy * dy;
    };
    const irm::NelderMeadResult res = irm::nelder_mead(f, {0.0, 0.0});
    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.x[0], 1.5, 1e-7);
    EXPECT_NEAR(res.x[1], -2.0, 1e-7);
    EXPECT_NEAR(res.fx, 0.0, 1e-12);
}

TEST(NelderMead, MinimisesRosenbrock) {
    const auto f = [](const std::vector<double>& p) {
        const double a = 1.0 - p[0];
        const double b = p[1] - p[0] * p[0];
        return a * a + 100.0 * b * b;
    };
    const irm::NelderMeadResult res = irm::nelder_mead(f, {-1.2, 1.0});
    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.x[0], 1.0, 1e-6);
    EXPECT_NEAR(res.x[1], 1.0, 1e-6);
}

TEST(NelderMead, NonConvergenceReportedNotThrown) {
    const auto f = [](const std::vector<double>& p) {
        const double a = 1.0 - p[0];
        const double b = p[1] - p[0] * p[0];
        return a * a + 100.0 * b * b;
    };
    const irm::NelderMeadResult res =
        irm::nelder_mead(f, {-1.2, 1.0}, 0.1, 1e-10, 1e-14, /*maxiter=*/5);
    EXPECT_FALSE(res.converged);
    EXPECT_EQ(res.iterations, 5);
}

TEST(NelderMead, InvalidStartThrows) {
    const auto f = [](const std::vector<double>&) { return 0.0; };
    EXPECT_THROW(irm::nelder_mead(f, {}), std::invalid_argument);
    EXPECT_THROW(irm::nelder_mead(f, {std::nan("")}), std::invalid_argument);
}

}  // namespace
