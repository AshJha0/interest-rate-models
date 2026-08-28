#ifndef IRM_CURVE_HPP
#define IRM_CURVE_HPP

/// \file curve.hpp
/// \brief Discount curve with log-linear interpolation in ln(DF).
///
/// Conventions (see API_SPEC.md):
///
///  * Day count is ACT/365F everywhere; all times are year fractions
///    `days/365` (a documented simplification versus the ACT/360
///    money-market convention used for real deposits/FRAs).
///  * Rates are decimals (0.05 = 5%).
///  * Interpolation is *linear in ln(DF)* between pillars, with an implicit
///    node `(t=0, DF=1)`.  Linear-in-ln(DF) is exactly the
///    piecewise-constant instantaneous-forward interpolation: on a segment
///    `[t_i, t_{i+1}]` the instantaneous forward is the constant
///    `f_i = -(ln DF_{i+1} - ln DF_i) / (t_{i+1} - t_i)`.
///  * Extrapolation beyond the last pillar is flat-forward: the last
///    segment's constant forward is carried on indefinitely.
///  * Negative rates are fully supported: DF > 1 is legal (EUR/JPY style);
///    the only requirement is DF > 0.

#include <cstddef>
#include <vector>

namespace irm {

/// Immutable discount curve defined by pillar times and discount factors.
class DiscountCurve {
public:
    /// \param times Strictly increasing pillar times in years (ACT/365F),
    ///   all > 0 and finite.
    /// \param dfs Discount factors at the pillars, all finite and > 0.
    ///   Values > 1 are allowed (negative rates).
    /// \throws std::invalid_argument on empty input, length mismatch,
    ///   non-monotone/duplicate/non-positive times, or DF <= 0 / non-finite.
    DiscountCurve(std::vector<double> times, std::vector<double> dfs);

    /// Discount factor DF(t); DF(0) = 1, flat-forward beyond last pillar.
    /// \throws std::invalid_argument for t < 0 or non-finite t.
    double df(double t) const;

    /// Continuously compounded zero rate `z(t) = -ln DF(t) / t`.
    /// At `t = 0` the limit is returned, i.e. the instantaneous forward of
    /// the first segment.
    double zero_rate(double t) const;

    /// Instantaneous forward `f(t) = -d ln DF / dt` (piecewise constant).
    /// At a pillar the *right* segment's forward is returned; at/after the
    /// last pillar the last segment's forward (the flat extrapolation level).
    double inst_forward(double t) const;

    /// Simply compounded forward rate over `[t1, t2]`:
    /// `F(t1, t2) = (DF(t1)/DF(t2) - 1) / (t2 - t1)`; requires 0 <= t1 < t2.
    double fwd_rate(double t1, double t2) const;

    /// Pillar times (without the implicit t = 0 node).
    const std::vector<double>& times() const { return times_; }
    /// Pillar discount factors.
    const std::vector<double>& dfs() const { return dfs_; }

private:
    std::size_t segment(double t) const;
    double slope(std::size_t i) const;
    double ln_df(double t) const;

    std::vector<double> times_;  ///< user pillars
    std::vector<double> dfs_;    ///< user DFs
    std::vector<double> t_;      ///< nodes incl. implicit (0, ln 1 = 0)
    std::vector<double> lnp_;    ///< ln DF at the nodes
};

/// Forward-starting par swap rate off a single curve.
///
/// `times = [t0, t1, ..., tn]` gives the swap start `t0 >= 0` and the fixed
/// payment times.  With accruals `tau_i = t_i - t_{i-1}` and the
/// single-curve identity (floating leg PV = DF(t0) - DF(tn)):
///
///   `par = (DF(t0) - DF(tn)) / sum_i tau_i DF(t_i)`.
///
/// \throws std::invalid_argument on fewer than two times, negative start,
///   or a non-increasing schedule.
double par_swap_rate(const DiscountCurve& curve, const std::vector<double>& times);

}  // namespace irm

#endif  // IRM_CURVE_HPP
