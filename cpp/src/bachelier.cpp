#include "irm/bachelier.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "irm/mathutils.hpp"
#include "irm/rootfind.hpp"

namespace irm {

namespace {

constexpr double kVolBracketStart = 0.01;
constexpr int kVolBracketDoublings = 40;

void check_common(double forward, double strike, double expiry, double annuity) {
    if (!(std::isfinite(forward) && std::isfinite(strike))) {
        throw std::invalid_argument("bachelier: forward and strike must be finite");
    }
    if (!std::isfinite(expiry) || expiry <= 0.0) {
        throw std::invalid_argument("bachelier: expiry must be finite and > 0, got " +
                                    std::to_string(expiry));
    }
    if (!std::isfinite(annuity) || annuity <= 0.0) {
        throw std::invalid_argument("bachelier: annuity must be finite and > 0, got " +
                                    std::to_string(annuity));
    }
}

}  // namespace

double bachelier_price(double forward, double strike, double expiry, double vol,
                       double annuity, bool payer) {
    check_common(forward, strike, expiry, annuity);
    if (!std::isfinite(vol) || vol < 0.0) {
        throw std::invalid_argument("bachelier: vol must be finite and >= 0, got " +
                                    std::to_string(vol));
    }
    const double moneyness = payer ? forward - strike : strike - forward;
    if (vol == 0.0) {
        return annuity * std::max(moneyness, 0.0);
    }
    const double sd = vol * std::sqrt(expiry);
    const double d = moneyness / sd;
    return annuity * (moneyness * norm_cdf(d) + sd * norm_pdf(d));
}

double bachelier_implied_vol(double price, double forward, double strike, double expiry,
                             double annuity, bool payer) {
    check_common(forward, strike, expiry, annuity);
    if (!std::isfinite(price)) {
        throw std::invalid_argument("bachelier: price must be finite, got " +
                                    std::to_string(price));
    }
    const double intrinsic = bachelier_price(forward, strike, expiry, 0.0, annuity, payer);
    if (price < intrinsic - kBachelierIntrinsicTol) {
        throw std::invalid_argument("bachelier: price " + std::to_string(price) +
                                    " below intrinsic " + std::to_string(intrinsic) +
                                    " — no implied vol");
    }
    if (price <= intrinsic + kBachelierIntrinsicTol) {
        return 0.0;
    }
    const auto diff = [&](double v) {
        return bachelier_price(forward, strike, expiry, v, annuity, payer) - price;
    };
    double hi = kVolBracketStart;
    bool bracketed = false;
    for (int i = 0; i < kVolBracketDoublings; ++i) {
        if (diff(hi) > 0.0) {
            bracketed = true;
            break;
        }
        hi *= 2.0;
    }
    if (!bracketed) {
        throw std::invalid_argument("bachelier: price " + std::to_string(price) +
                                    " too large to imply a volatility");
    }
    return brentq(diff, 0.0, hi, kBachelierImpliedVolXtol);
}

}  // namespace irm
