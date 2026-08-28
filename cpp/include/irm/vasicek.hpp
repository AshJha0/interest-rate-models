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
    /// `B = (1 - e^{-kappa tau})/kappa`; it is the sensitivity
    /// `-d ln P / d r` and appears in every option/moment formula.
    double b_factor(double tau) const;

    /// `A(t,T)` with `tau = T - t` (depends on tau only):
    /// `A = exp[(theta - sigma^2/(2 kappa^2)) (B - tau) - sigma^2 B^2/(4 kappa)]`.
    /// The first term is the risk-adjusted drift contribution of the mean
    /// level; the second is the convexity from rate volatility.
    double a_factor(double tau) const;

    /// Zero-coupon bond price `P(t, maturity) = A e^{-B r(t)}`.
    /// \p r defaults to `r0` (only sensible for t = 0).
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
    /// (sample stdev with ddof 1 over sqrt(n_paths)).
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
struct VasicekCalibration {
    double kappa;
    double theta;
    double sigma;
    double r0;
    double rmse;      ///< sqrt(objective / n)
    int iterations;   ///< optimizer iterations
    bool converged;   ///< optimizer convergence flag

    /// Build the calibrated model.
    Vasicek model() const { return Vasicek(kappa, theta, sigma, r0); }
};

/// Fit `(kappa, theta, sigma)` to market zero yields by least squares.
///
/// Minimises `sum_i (y_model(T_i) - y_i)^2` with Nelder-Mead; invalid
/// regions (kappa <= 1e-6, sigma < 0, kappa > 50, |theta| > 5, sigma > 5)
/// are handled with a smooth penalty `1e6 * (1 + distance)` pushing the
/// simplex back inside the domain.  `r0` is held fixed; returned sigma is
/// `|sigma|`.
/// \param x0 optional start; defaults to (0.5, mean(yields), 0.01) with
///   initial simplex step 0.05.
/// \throws std::invalid_argument on fewer than 3 pairs, length mismatch, or
///   non-finite/non-positive inputs.
VasicekCalibration calibrate_vasicek(
    const std::vector<double>& maturities, const std::vector<double>& yields,
    double r0, std::optional<std::vector<double>> x0 = std::nullopt,
    int maxiter = 4000);

}  // namespace irm

#endif  // IRM_VASICEK_HPP
