#ifndef IRM_BOOTSTRAP_HPP
#define IRM_BOOTSTRAP_HPP

/// \file bootstrap.hpp
/// \brief Sequential single-curve bootstrap from deposits, FRAs and par swaps.
///
/// Framework
/// ---------
/// Single-curve: the floating leg of a swap is projected off the same curve
/// it is discounted on, so its PV telescopes to `DF(0) - DF(T)`.  (Real
/// desks use multi-curve setups — OIS discounting plus tenor-specific
/// projection curves; the OIS variant below bootstraps such a discount
/// curve, but projection basis is out of scope and documented as a
/// simplification.)
///
/// Algorithm
/// ---------
/// Instruments must be supplied with strictly increasing pillar (maturity)
/// times.  For each instrument in order, the discount factor at its pillar
/// is the single unknown `x`; all earlier dates interpolate on the
/// already-solved pillars (plus the trial pillar), so each pillar is found
/// by a 1-D root find of the instrument's *residual* (PV mispricing):
///
///  * Deposit (simple interest): `residual = DF(T) (1 + R T) - 1`
///  * FRA:                       `residual = DF(T1) - (1 + R (T2-T1)) DF(T2)`
///  * Par swap / OIS swap:       `residual = R * annuity - (1 - DF(Tn))`
///    with `annuity = sum_i tau_i DF(t_i)` over the annual fixed schedule.
///
/// The root is found with the native Brent solver on the bracket
/// `DF in [1e-10, 100]` with `xtol = 1e-14`.  A bracket failure means no
/// positive discount factor can reprice the quote — i.e. crossed /
/// arbitrageable quotes (the "negative implied DF" case) — and throws.

#include <string>
#include <variant>
#include <vector>

#include "irm/curve.hpp"

namespace irm {

/// Annual fixed-leg payment times ending exactly at \p maturity.
///
/// `n = ceil(maturity - 1e-12)` payments at `maturity - (n-1), ..., maturity`;
/// e.g. 2.0 -> {1.0, 2.0}; 2.5 -> {0.5, 1.5, 2.5}; 0.25 -> {0.25}.  A short
/// first stub (< 1y) is created for non-integer maturities.
/// \throws std::invalid_argument for non-finite or non-positive maturity.
std::vector<double> annual_schedule(double maturity);

/// Cash deposit paying simple interest `rate` at `maturity`.
struct Deposit {
    double maturity;
    double rate;

    /// \throws std::invalid_argument on non-finite rate, maturity <= 0, or a
    ///   quote implying a non-positive discount factor (1 + R*T <= 0).
    Deposit(double maturity, double rate);

    double pillar() const { return maturity; }
    /// Mispricing of investing 1 at the simple rate: DF(T)(1+RT) - 1.
    double residual(const DiscountCurve& curve) const;
};

/// Forward rate agreement fixing the simple forward over [start, end].
struct FRA {
    double start;
    double end;
    double rate;

    /// \throws std::invalid_argument unless 0 <= start < end, rate finite,
    ///   and 1 + R*(end-start) > 0.
    FRA(double start, double end, double rate);

    double pillar() const { return end; }
    /// DF-linear mispricing: DF(T1) - (1 + R tau) DF(T2).
    double residual(const DiscountCurve& curve) const;
};

/// Par interest-rate swap: annual fixed leg vs single-curve floating leg.
/// Floating leg PV telescopes to `1 - DF(T)` because projection and
/// discounting share the curve.
struct Swap {
    double maturity;
    double rate;

    Swap(double maturity, double rate);

    double pillar() const { return maturity; }
    /// R * annuity - (1 - DF(T)), the fixed-minus-floating PV.
    double residual(const DiscountCurve& curve) const;
};

/// OIS par swap: annual fixed leg vs daily-compounded overnight leg.
///
/// Daily compounding of the overnight rate is approximated as continuous
/// compounding at the curve's short rate, under which the compounded
/// floating coupon over `[t_{i-1}, t_i]` pays `DF(t_{i-1})/DF(t_i) - 1` and
/// the floating leg PV telescopes to `1 - DF(T)` — identical in form to
/// Swap.  The daily-vs-continuous compounding gap is O(rate^2 / 365) and is
/// a documented approximation.
struct OISSwap {
    double maturity;
    double rate;

    OISSwap(double maturity, double rate);

    double pillar() const { return maturity; }
    double residual(const DiscountCurve& curve) const;
};

/// Value-semantic bootstrap instrument.
using Instrument = std::variant<Deposit, FRA, Swap, OISSwap>;

/// Pillar (maturity) time of any instrument.
double instrument_pillar(const Instrument& ins);
/// PV mispricing of any instrument on \p curve.
double instrument_residual(const Instrument& ins, const DiscountCurve& curve);
/// Quoted rate of any instrument.
double instrument_rate(const Instrument& ins);
/// Human-readable type name ("Deposit", "FRA", "Swap", "OISSwap").
std::string instrument_name(const Instrument& ins);

/// Sequentially bootstrap a discount curve from sorted instruments.
///
/// Instruments must be ordered with strictly increasing pillar times
/// (duplicates rejected).  Each pillar DF is solved with Brent's method on
/// `[1e-10, 100]` (DF > 1 allowed: negative rates).
/// \throws std::invalid_argument on an empty list or ordering violations.
/// \throws std::domain_error when no positive DF can reprice a quote
///   (crossed/arbitrageable inputs), naming the offending pillar.
DiscountCurve bootstrap(const std::vector<Instrument>& instruments);

}  // namespace irm

#endif  // IRM_BOOTSTRAP_HPP
