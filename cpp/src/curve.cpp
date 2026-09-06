#include "irm/curve.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace irm {

DiscountCurve::DiscountCurve(std::vector<double> times, std::vector<double> dfs)
    : times_(std::move(times)), dfs_(std::move(dfs)) {
    if (times_.empty()) {
        throw std::invalid_argument("curve needs at least one pillar");
    }
    if (times_.size() != dfs_.size()) {
        throw std::invalid_argument("times and dfs length mismatch: " +
                                    std::to_string(times_.size()) + " vs " +
                                    std::to_string(dfs_.size()));
    }
    double prev = 0.0;
    for (double t : times_) {
        if (!std::isfinite(t)) {
            throw std::invalid_argument("non-finite pillar time " + std::to_string(t));
        }
        if (t <= prev) {
            throw std::invalid_argument(
                "pillar times must be strictly increasing and positive; offending time " +
                std::to_string(t) + " after " + std::to_string(prev));
        }
        prev = t;
    }
    for (std::size_t i = 0; i < times_.size(); ++i) {
        const double p = dfs_[i];
        if (!std::isfinite(p) || p <= 0.0) {
            throw std::invalid_argument("discount factor at t=" + std::to_string(times_[i]) +
                                        " must be finite and > 0, got " + std::to_string(p));
        }
    }
    // Internal nodes include the implicit (0, ln 1 = 0).
    t_.reserve(times_.size() + 1);
    lnp_.reserve(times_.size() + 1);
    t_.push_back(0.0);
    lnp_.push_back(0.0);
    for (std::size_t i = 0; i < times_.size(); ++i) {
        t_.push_back(times_[i]);
        lnp_.push_back(std::log(dfs_[i]));
    }
}

std::size_t DiscountCurve::segment(double t) const {
    // Index i of the segment [t_i, t_{i+1}] used for time t.  Times at or
    // beyond the last pillar map to the last segment (flat-forward
    // extrapolation); a time exactly at an interior node maps to the
    // segment starting at that node.
    const std::size_t n_seg = t_.size() - 1;
    const auto it = std::upper_bound(t_.begin(), t_.end(), t);
    std::size_t i = (it == t_.begin()) ? 0 : static_cast<std::size_t>(it - t_.begin()) - 1;
    if (i > n_seg - 1) i = n_seg - 1;
    return i;
}

double DiscountCurve::slope(std::size_t i) const {
    // d ln(DF)/dt on segment i (equals minus the constant forward).
    return (lnp_[i + 1] - lnp_[i]) / (t_[i + 1] - t_[i]);
}

double DiscountCurve::ln_df(double t) const {
    if (!std::isfinite(t) || t < 0.0) {
        throw std::invalid_argument("time must be finite and >= 0, got " + std::to_string(t));
    }
    if (t == 0.0) return 0.0;
    const std::size_t i = segment(t);
    return lnp_[i] + slope(i) * (t - t_[i]);
}

double DiscountCurve::df(double t) const { return std::exp(ln_df(t)); }

double DiscountCurve::zero_rate(double t) const {
    if (!std::isfinite(t) || t < 0.0) {
        throw std::invalid_argument("time must be finite and >= 0, got " + std::to_string(t));
    }
    if (t == 0.0) return -slope(0);
    return -ln_df(t) / t;
}

double DiscountCurve::inst_forward(double t) const {
    if (!std::isfinite(t) || t < 0.0) {
        throw std::invalid_argument("time must be finite and >= 0, got " + std::to_string(t));
    }
    return -slope(segment(t));
}

double DiscountCurve::fwd_rate(double t1, double t2) const {
    if (!(std::isfinite(t1) && std::isfinite(t2))) {
        throw std::invalid_argument("forward-rate times must be finite");
    }
    if (t1 < 0.0 || t2 <= t1) {
        throw std::invalid_argument("need 0 <= t1 < t2 for a forward rate, got t1=" +
                                    std::to_string(t1) + ", t2=" + std::to_string(t2));
    }
    const double df1 = df(t1);
    const double df2 = df(t2);
    if (df2 == 0.0) {
        throw std::invalid_argument("forward rate over [" + std::to_string(t1) + ", " +
                                    std::to_string(t2) + "] not representable: DF(t2) underflowed to 0");
    }
    const double fwd = (df1 / df2 - 1.0) / (t2 - t1);
    if (!std::isfinite(fwd)) {
        throw std::invalid_argument("forward rate over [" + std::to_string(t1) + ", " +
                                    std::to_string(t2) + "] not representable: " +
                                    std::to_string(fwd));
    }
    return fwd;
}

double par_swap_rate(const DiscountCurve& curve, const std::vector<double>& times) {
    if (times.size() < 2) {
        throw std::invalid_argument("par_swap_rate needs a start and at least one payment");
    }
    double prev = times.front();
    if (prev < 0.0) {
        throw std::invalid_argument("swap start must be >= 0, got " + std::to_string(prev));
    }
    double annuity = 0.0;
    for (std::size_t i = 1; i < times.size(); ++i) {
        const double t = times[i];
        if (t <= prev) {
            throw std::invalid_argument("swap schedule times must be strictly increasing");
        }
        annuity += (t - prev) * curve.df(t);
        prev = t;
    }
    return (curve.df(times.front()) - curve.df(times.back())) / annuity;
}

}  // namespace irm
