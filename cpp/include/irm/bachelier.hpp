#ifndef IRM_BACHELIER_HPP
#define IRM_BACHELIER_HPP

/// \file bachelier.hpp
/// \brief Bachelier (normal) model for caplets and swaptions: price and
/// implied volatility.
///
/// Rates desks quote cap/floor and swaption prices as *normal* ("bp")
/// volatilities: the volatility `sigma_N` such that the Bachelier formula
/// reproduces the price.  For a payer option on a forward `F` with strike
/// `K`, expiry `T` and annuity `A` (`A = tau DF(T_pay)` for a caplet,
/// `sum_i tau_i DF(u_i)` for a swaption):
///
///   `price = A [ (F - K) N(d) + sigma_N sqrt(T) phi(d) ]`,
///   `d = (F - K) / (sigma_N sqrt(T))`
///
/// and the receiver is `A [ (K - F) N(-d) + sigma_N sqrt(T) phi(d) ]`.
/// With `sigma_N = 0` both collapse to discounted intrinsic.  The price is
/// strictly increasing in `sigma_N`, so the implied volatility is unique and
/// found with the native Brent solver on an expanding bracket.  Negative
/// forwards and strikes are first-class.

namespace irm {

/// Absolute price tolerance for deciding a quote is at intrinsic (vol 0).
constexpr double kBachelierIntrinsicTol = 1e-15;
/// Root tolerance on the implied volatility.
constexpr double kBachelierImpliedVolXtol = 1e-14;

/// Normal-model option price (per unit notional, annuity included).
/// `vol` is the absolute (normal) volatility in rate units per sqrt(year):
/// 0.0075 means 75 bp.  `vol = 0` returns discounted intrinsic.
/// \throws std::invalid_argument unless forward/strike finite, expiry > 0,
///   annuity > 0 and vol >= 0 (all finite).
double bachelier_price(double forward, double strike, double expiry, double vol,
                       double annuity, bool payer = true);

/// Normal implied volatility of a payer/receiver price.
///
/// `price` must be finite and at least the discounted intrinsic
/// `annuity * max(+-(F - K), 0)`; a price within kBachelierIntrinsicTol of
/// intrinsic returns 0.  Otherwise Brent's method solves
/// `bachelier_price(vol) = price` on `[0, hi]` with `hi` found by doubling
/// from 0.01 (at most 40 doublings).  Root tolerance 1e-14 on the vol.
/// \throws std::invalid_argument on bad inputs, a price below intrinsic, or
///   a price too large to imply a volatility.
double bachelier_implied_vol(double price, double forward, double strike, double expiry,
                             double annuity, bool payer = true);

}  // namespace irm

#endif  // IRM_BACHELIER_HPP
