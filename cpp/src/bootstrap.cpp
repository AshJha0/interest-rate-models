#include "irm/bootstrap.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "irm/rootfind.hpp"

namespace irm {

namespace {

constexpr double kDfLo = 1e-10;
constexpr double kDfHi = 100.0;

void check_rate(double rate) {
    if (!std::isfinite(rate)) {
        throw std::invalid_argument("quote rate must be finite, got " + std::to_string(rate));
    }
}

/// Shared fixed-vs-telescoped-floating residual for Swap and OISSwap:
/// `R * annuity - (1 - DF(T))` over the annual fixed schedule.
double swap_residual(double maturity, double rate, const DiscountCurve& curve) {
    double annuity = 0.0;
    double prev = 0.0;
    for (double t : annual_schedule(maturity)) {
        annuity += (t - prev) * curve.df(t);
        prev = t;
    }
    return rate * annuity - (1.0 - curve.df(maturity));
}

}  // namespace

std::vector<double> annual_schedule(double maturity) {
    if (!std::isfinite(maturity) || maturity <= 0.0) {
        throw std::invalid_argument("maturity must be finite and > 0, got " +
                                    std::to_string(maturity));
    }
    const int n = static_cast<int>(std::ceil(maturity - 1e-12));
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        out.push_back(maturity - static_cast<double>(n - 1 - k));
    }
    return out;
}

Deposit::Deposit(double maturity_, double rate_) : maturity(maturity_), rate(rate_) {
    check_rate(rate);
    if (!std::isfinite(maturity) || maturity <= 0.0) {
        throw std::invalid_argument("deposit maturity must be > 0, got " +
                                    std::to_string(maturity));
    }
    if (1.0 + rate * maturity <= 0.0) {
        throw std::invalid_argument("deposit quote implies non-positive discount factor (1 + R*T = " +
                                    std::to_string(1.0 + rate * maturity) + ")");
    }
}

double Deposit::residual(const DiscountCurve& curve) const {
    return curve.df(maturity) * (1.0 + rate * maturity) - 1.0;
}

FRA::FRA(double start_, double end_, double rate_) : start(start_), end(end_), rate(rate_) {
    check_rate(rate);
    if (!(std::isfinite(start) && std::isfinite(end))) {
        throw std::invalid_argument("FRA times must be finite");
    }
    if (start < 0.0 || end <= start) {
        throw std::invalid_argument("FRA needs 0 <= start < end, got start=" +
                                    std::to_string(start) + ", end=" + std::to_string(end));
    }
    if (1.0 + rate * (end - start) <= 0.0) {
        throw std::invalid_argument("FRA quote implies non-positive discount factor");
    }
}

double FRA::residual(const DiscountCurve& curve) const {
    const double tau = end - start;
    return curve.df(start) - (1.0 + rate * tau) * curve.df(end);
}

Swap::Swap(double maturity_, double rate_) : maturity(maturity_), rate(rate_) {
    check_rate(rate);
    annual_schedule(maturity);  // validates maturity
}

double Swap::residual(const DiscountCurve& curve) const {
    return swap_residual(maturity, rate, curve);
}

OISSwap::OISSwap(double maturity_, double rate_) : maturity(maturity_), rate(rate_) {
    check_rate(rate);
    annual_schedule(maturity);
}

double OISSwap::residual(const DiscountCurve& curve) const {
    return swap_residual(maturity, rate, curve);
}

double instrument_pillar(const Instrument& ins) {
    return std::visit([](const auto& i) { return i.pillar(); }, ins);
}

double instrument_residual(const Instrument& ins, const DiscountCurve& curve) {
    return std::visit([&curve](const auto& i) { return i.residual(curve); }, ins);
}

double instrument_rate(const Instrument& ins) {
    return std::visit([](const auto& i) { return i.rate; }, ins);
}

std::string instrument_name(const Instrument& ins) {
    struct Namer {
        std::string operator()(const Deposit&) const { return "Deposit"; }
        std::string operator()(const FRA&) const { return "FRA"; }
        std::string operator()(const Swap&) const { return "Swap"; }
        std::string operator()(const OISSwap&) const { return "OISSwap"; }
    };
    return std::visit(Namer{}, ins);
}

DiscountCurve bootstrap(const std::vector<Instrument>& instruments) {
    if (instruments.empty()) {
        throw std::invalid_argument("bootstrap needs at least one instrument");
    }
    double prev_pillar = 0.0;
    for (const Instrument& ins : instruments) {
        const double p = instrument_pillar(ins);
        if (p <= prev_pillar) {
            throw std::invalid_argument(
                "instrument pillars must be strictly increasing (duplicate or out-of-order "
                "pillar at t=" + std::to_string(p) + ")");
        }
        prev_pillar = p;
    }

    std::vector<double> times;
    std::vector<double> dfs;
    times.reserve(instruments.size());
    dfs.reserve(instruments.size());
    for (const Instrument& ins : instruments) {
        const double pillar = instrument_pillar(ins);
        // Trial curve = all solved pillars plus (pillar, x); intermediate
        // cashflow dates interpolate on it (how missing pillar gaps are
        // handled).
        const auto objective = [&](double x) {
            std::vector<double> trial_t = times;
            std::vector<double> trial_p = dfs;
            trial_t.push_back(pillar);
            trial_p.push_back(x);
            return instrument_residual(ins, DiscountCurve(std::move(trial_t), std::move(trial_p)));
        };
        double df;
        try {
            df = brentq(objective, kDfLo, kDfHi, 1e-14);
        } catch (const std::exception& exc) {
            throw std::domain_error("bootstrap failed at pillar t=" + std::to_string(pillar) +
                                    " (" + instrument_name(ins) +
                                    ", rate=" + std::to_string(instrument_rate(ins)) +
                                    "): no admissible positive discount factor — "
                                    "crossed/arbitrageable quotes? [" + exc.what() + "]");
        }
        times.push_back(pillar);
        dfs.push_back(df);
    }
    return DiscountCurve(std::move(times), std::move(dfs));
}

}  // namespace irm
