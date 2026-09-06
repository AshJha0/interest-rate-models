#ifndef IRM_VASICEK_HPP
#define IRM_VASICEK_HPP

/// \file vasicek.hpp
/// \brief Vasicek short-rate model `dr = kappa (theta - r) dt + sigma dW`.
///
/// Gaussian model: rates can go negative, which is a *feature* here
/// (EUR/JPY style curves) and the reason no lognormal short-rate model is
/// included in this project.
///
/// Closed forms implemented (affine term structure):
///
///  * `B(t,T) = (1 - e^{-kappa (T-t)}) / kappa`
///  * `A(t,T) = exp[(theta - sigma^2/(2 kappa^2)) (B(t,T) - (T-t))
///                 - sigma^2 B(t,T)^2 / (4 kappa)]`
///  * `P(t,T) = A(t,T) e^{-B(t,T) r(t)}`
///  * `r_T | r_t ~ N(theta + (r_t - theta) e^{-kappa (T-t)},
///                   sigma^2 (1 - e^{-2 kappa (T-t)}) / (2 kappa))`
///  * Jamshidian (1989) bond-option formula for European calls/puts on ZCBs.
///
/// Monte Carlo uses the *exact* OU transition of the pair `(r, int r dt)`
/// (see mathutils.hpp), so the ZCB estimator is unbiased at any step count.
/// Calibration fits `(kappa, theta, sigma)` to market zero yields by least
/// squares with the native Nelder-Mead from optimize.hpp.

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace irm {

/// Vasicek model with parameters `kappa > 0`, `sigma >= 0`.
class Vasicek {
public:
    /// \throws std::invalid_argument on non-finite parameters, kappa <= 0,
    ///   or sigma < 0.
    Vasicek(double kappa, double theta, double sigma, double r0);

    double kappa() const { return kappa_; }
    double theta() const { return theta_; }
    double sigma() const { return sigma_; }
    double r0() const { return r0_; }

    /// `B(t,T)` with `tau = T - t`: duration of the ZCB w.r.t. r.
    /// `B = (1 - e^{-kappa tau})/kappa` (evaluated with `expm1`); it is the sensitivity
    /// `-d ln P / d r` and appears in every option/moment formula.
    double b_factor(double tau) const;

    /// `A(t,T)` with `tau = T - t` (depends on tau only):
    /// `A = exp[(theta - sigma^2/(2 kappa^2)) (B - tau) - sigma^2 B^2/(4 kappa)]`.
    /// The first term is the risk-adjusted drift contribution of the mean
    /// level; the second is the convexity from rate volatility.
    double a_factor(double tau) const;

    /// Zero-coupon bond price `P(t, maturity) = A e^{-B r(t)}`.
    /// \p r defaults to `r0` (only sensible for t = 0).
    /// \throws std::invalid_argument on non-finite inputs, maturity < t, or
    ///   when `exp(-B r)` would overflow (result not representable).
    double zcb_price(double maturity, double t = 0.0,
                     std::optional<double> r = std::nullopt) const;

    /// Continuously compounded zero yield `-ln P(t,T) / (T - t)` (tau > 0).
    double zero_yield(double maturity, double t = 0.0,
                      std::optional<double> r = std::nullopt) const;

    /// `E[r_T | r_0] = theta + (r_0 - theta) e^{-kappa T}`.
    double r_mean(double horizon, std::optional<double> r = std::nullopt) const;

    /// `Var[r_T | r_0] = sigma^2 (1 - e^{-2 kappa T}) / (2 kappa)`.
    double r_var(double horizon) const;

    /// European call on a unit ZCB (Jamshidian 1989 closed form):
    /// `ZBC = P(0,S) N(h) - K P(0,T) N(h - sigma_p)` with
    /// `h = ln(P(0,S)/(K P(0,T)))/sigma_p + sigma_p/2`.
    double zbc(double expiry, double bond_maturity, double strike) const;

    /// European put on a unit ZCB:
    /// `ZBP = K P(0,T) N(sigma_p - h) - P(0,S) N(-h)`.
    double zbp(double expiry, double bond_maturity, double strike) const;

    /// Exact-transition simulation of `(r_T, int_0^T r dt)`.
    ///
    /// Writes `r = theta + x` with `x` a mean-zero OU process and samples
    /// `(x_next, int x dt)` per step from their exact joint Gaussian
    /// (conditional decomposition), so both outputs are free of
    /// discretisation bias.  Two standard-normal draws per step per path.
    /// \returns pair of per-path vectors (r at horizon, integral of r).
    std::pair<std::vector<double>, std::vector<double>> simulate(
        double horizon, int n_steps, int n_paths, std::uint64_t seed) const;

    /// MC ZCB price `E[e^{-int r}]` and its standard error
    /// (sample stdev with ddof 1 over sqrt(n_paths); exactly 0 for sigma = 0
    /// or a single path).
    std::pair<double, double> mc_zcb(double maturity, int n_steps, int n_paths,
                                     std::uint64_t seed) const;

private:
    double sigma_p(double expiry, double bond_maturity) const;
    double zb_option(double expiry, double bond_maturity, double strike,
                     bool call) const;

    double kappa_;
    double theta_;
    double sigma_;
    double r0_;
};

/// Least-squares calibration outcome (non-convergence reported, not thrown).
/// Calibration domain: `kappa in (1e-6, 50]`, `|theta| <= 5`, `sigma in [0, 5]`;
/// points outside get the penalty `1e6 (1 + distance)`.
constexpr double kVasicekKappaLo = 1e-6;
constexpr double kVasicekKappaHi = 50.0;
constexpr double kVasicekThetaMax = 5.0;
constexpr double kVasicekSigmaHi = 5.0;

/// Least-squares calibration outcome (non-convergence reported, not thrown).
///
///  * `converged`: the optimiser met its tolerances at the reported solution
///    **and** the solution is not at a domain bound.
///  * `at_bound`: the solution sits at the edge of the calibration domain
///    (`kappa < 1e-5` or `> 49.95`, `|theta| > 4.995`, `sigma < 1e-5` or
///    `> 4.995`) — a penalty-boundary artefact, not a fit.
///  * `n_starts`: Nelder-Mead starts run (3 by default; 1 when `x0` given).
///  * `kappa_spread` / `theta_spread` / `sigma_spread`: max minus min of each
///    parameter over the starts whose rmse is within 5 % of the best (0 for
///    a single start).
///  * `identified`: `n_starts >= 2` and all equivalent starts agree
///    (`kappa_spread <= 1e-2`, `theta_spread <= 1e-3`, `sigma_spread <= 1e-3`).
///    With a single start identifiability is not assessed and this is false.
///  * `sigma_fixed`: true when sigma was supplied, not fitted.
///
/// Price on the result only if `converged && (identified || sigma_fixed)`.
struct VasicekCalibration {
    double kappa;
    double theta;
    double sigma;
    double r0;
    double rmse;          ///< sqrt(objective / n)
    int iterations;       ///< optimizer iterations of the best start
    bool converged;       ///< optimizer converged and not at a bound
    bool at_bound;        ///< solution at the edge of the domain
    int n_starts;         ///< number of starts run
    double kappa_spread;  ///< spread over equivalent starts
    double theta_spread;
    double sigma_spread;
    bool identified;      ///< equivalent starts agree (multi-start only)
    bool sigma_fixed;     ///< sigma held fixed at the supplied value

    /// Build the calibrated model.
    Vasicek model() const { return Vasicek(kappa, theta, sigma, r0); }
};

/// Fit `(kappa, theta, sigma)` to market zero yields by least squares.
///
/// Minimises `sum_i (y_model(T_i) - y_i)^2` with Nelder-Mead
/// (`initial_step = 0.05`); out-of-domain points get a smooth penalty so
/// the simplex is pushed back inside the domain.  `r0` is held fixed.
///  * `x0` empty (default): three starts `(kappa, mean(y), sigma)` for
///    `(kappa, sigma) in {(0.2, 0.005), (0.5, 0.01), (1.5, 0.02)}`; the lowest
///    objective is returned with the parameter spreads across equivalent
///    starts.
///  * `x0 = {kappa, theta, sigma}`: a single warm start (must have exactly 3
///    finite entries; identifiability is then not assessed).
///  * `sigma_fixed`: fit only `(kappa, theta)` with sigma held at the given
///    value (>= 0) — the practitioner workflow; `x0[2]` is ignored.
///
/// Yields identify the parameters weakly (the bundled data has two minima
/// with rmse within 2 %); always check `converged`, `at_bound`, `identified`.
/// \throws std::invalid_argument on fewer than 3 points, length mismatch,
///   non-finite/non-positive inputs, a wrong-length or non-finite x0,
///   negative sigma_fixed, or maxiter < 1.
VasicekCalibration calibrate_vasicek(
    const std::vector<double>& maturities, const std::vector<double>& yields,
    double r0, std::optional<std::vector<double>> x0 = std::nullopt,
    int maxiter = 4000, std::optional<double> sigma_fixed = std::nullopt);

}  // namespace irm

#endif  // IRM_VASICEK_HPP
