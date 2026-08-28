// Tests for the native Brent and bisection root finders.

#include "irm/rootfind.hpp"

#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

namespace {

TEST(Brentq, FindsSimpleRoots) {
    const double r = irm::brentq([](double x) { return x * x - 2.0; }, 0.0, 2.0);
    EXPECT_NEAR(r, std::sqrt(2.0), 1e-13);
    const double c = irm::brentq([](double x) { return std::cos(x); }, 0.0, 3.0);
    EXPECT_NEAR(c, M_PI / 2.0, 1e-13);
}

TEST(Brentq, ExactEndpointRootsReturnedDirectly) {
    EXPECT_EQ(irm::brentq([](double x) { return x; }, 0.0, 1.0), 0.0);
    EXPECT_EQ(irm::brentq([](double x) { return x - 1.0; }, 0.0, 1.0), 1.0);
}

TEST(Brentq, UnbracketedRootThrows) {
    EXPECT_THROW(irm::brentq([](double x) { return x * x + 1.0; }, -1.0, 1.0),
                 std::invalid_argument);
}

TEST(Brentq, InvalidInputsThrow) {
    EXPECT_THROW(irm::brentq([](double x) { return x; }, -1.0,
                             std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
    EXPECT_THROW(irm::brentq([](double x) { return x; }, -1.0, 1.0, /*xtol=*/0.0),
                 std::invalid_argument);
}

TEST(Brentq, NonConvergenceThrowsDomainError) {
    // One iteration cannot converge on a wide bracket for a hard function.
    EXPECT_THROW(irm::brentq([](double x) { return std::tanh(1e3 * (x - 0.123456)); }, -10.0,
                             10.0, 1e-14, irm::kBrentRtol, /*maxiter=*/1),
                 std::domain_error);
}

TEST(Brentq, PropertyGridOfShiftedExponentials) {
    // Property-style check: the root of exp(x) - exp(a) is recovered
    // across a grid of shifts a.
    for (double a = -2.0; a <= 2.0; a += 0.25) {
        const double r = irm::brentq(
            [a](double x) { return std::exp(x) - std::exp(a); }, a - 1.7, a + 2.3);
        EXPECT_NEAR(r, a, 1e-12) << "a=" << a;
    }
}

TEST(Bisect, MatchesBrentOnSimpleRoot) {
    const double r = irm::bisect([](double x) { return x * x * x - 8.0; }, 0.0, 5.0);
    EXPECT_NEAR(r, 2.0, 1e-11);
}

TEST(Bisect, UnbracketedRootThrows) {
    EXPECT_THROW(irm::bisect([](double x) { return x * x + 1.0; }, -1.0, 1.0),
                 std::invalid_argument);
}

TEST(Bisect, ReversedBracketAccepted) {
    const double r = irm::bisect([](double x) { return x - 0.5; }, 1.0, 0.0);
    EXPECT_NEAR(r, 0.5, 1e-11);
}

}  // namespace
