// End-to-end demo for the irm C++ library.
//
// Bootstraps the EUR (negative short end) and USD (normal) curves plus the
// OIS curve, prints zero/forward tables, calibrates a Vasicek model to the
// bundled zero yields, and prices Hull-White caps and a Jamshidian payer
// swaption.  Mirrors python/demo.py.

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "irm/irm.hpp"

namespace {

const std::string kDataDir = IRM_DATA_DIR;

void print_curve_table(const char* name, const irm::DiscountCurve& curve) {
    std::printf("\n%s  (ACT/365F, cont. comp. zeros, simple 1y-forwards)\n", name);
    std::printf("  %5s %10s %8s %13s\n", "T", "DF", "zero %", "fwd(T,T+1) %");
    for (double t : {0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0}) {
        const double fwd = curve.fwd_rate(t, t + 1.0);
        std::printf("  %5.2f %10.6f %8.4f %13.4f\n", t, curve.df(t),
                    100.0 * curve.zero_rate(t), 100.0 * fwd);
    }
}

}  // namespace

int main() {
    try {
        std::printf("================================================================\n");
        std::printf("irm demo (C++) — bootstrapping, Vasicek, Hull-White\n");
        std::printf("================================================================\n");

        // ---- 1. bootstrap both quote curves + OIS ---------------------- //
        const auto eur = irm::bootstrap(
            irm::load_curve_quotes(kDataDir + "/curve_quotes.csv", "EUR"));
        const auto usd = irm::bootstrap(
            irm::load_curve_quotes(kDataDir + "/curve_quotes.csv", "USD"));
        const auto ois = irm::bootstrap(irm::load_ois_quotes(kDataDir + "/ois_quotes.csv"));
        print_curve_table("EUR curve (negative short end)", eur);
        print_curve_table("USD curve (normal/inverted)", usd);
        std::printf("\nOIS discount curve: DF(1y)=%.6f  DF(10y)=%.6f\n", ois.df(1.0),
                    ois.df(10.0));
        std::printf("Sanity: EUR DF(1y) = %.6f > 1 (negative rates)\n", eur.df(1.0));

        // ---- 2. Vasicek calibration ------------------------------------ //
        const auto [ts, ys] = irm::load_zero_yields(kDataDir + "/zero_yields.csv");
        const irm::VasicekCalibration cal = irm::calibrate_vasicek(ts, ys, /*r0=*/0.03);
        std::printf("\nVasicek calibration to data/zero_yields.csv (r0 = 3%%):\n");
        std::printf("  kappa=%.4f  theta=%.4f  sigma=%.4f  rmse=%.2e  iters=%d  converged=%s\n",
                    cal.kappa, cal.theta, cal.sigma, cal.rmse, cal.iterations,
                    cal.converged ? "true" : "false");
        const irm::Vasicek model = cal.model();
        std::printf("  %4s %11s %9s\n", "T", "mkt yield %", "model %");
        for (std::size_t i = 0; i < ts.size(); i += 3) {
            std::printf("  %4.1f %11.4f %9.4f\n", ts[i], 100.0 * ys[i],
                        100.0 * model.zero_yield(ts[i]));
        }
        const auto [mc, se] = model.mc_zcb(5.0, /*n_steps=*/8, /*n_paths=*/50000, /*seed=*/42);
        std::printf("  ZCB(5y): analytic %.6f  MC %.6f (se %.1e)\n", model.zcb_price(5.0), mc,
                    se);

        // ---- 3. Hull-White caps on the USD curve ----------------------- //
        const irm::HullWhite hw(/*a=*/0.1, /*sigma=*/0.01, usd);
        const double atm = irm::par_swap_rate(usd, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
        std::printf("\nHull-White (a=0.10, sigma=0.01) on USD curve; 5y par = %.4f%%\n",
                    100.0 * atm);
        std::printf("  %14s %9s %18s\n", "cap", "strike %", "PV (notional 100)");
        const struct { const char* label; double strike; } caps[] = {
            {"5y ATM", atm}, {"5y ATM+100bp", atm + 0.01}, {"5y ATM-100bp", atm - 0.01}};
        for (const auto& c : caps) {
            const double pv = hw.cap({1.0, 2.0, 3.0, 4.0, 5.0}, c.strike, /*notional=*/100.0);
            std::printf("  %14s %9.4f %18.6f\n", c.label, 100.0 * c.strike, pv);
        }
        const double caplet = hw.caplet(1.0, 2.0, atm, /*notional=*/100.0);
        const auto [mc_c, se_c] =
            hw.mc_caplet(1.0, 2.0, atm, 100.0, /*n_steps=*/8, /*n_paths=*/50000, /*seed=*/7);
        std::printf("  caplet(1y->2y) analytic %.6f | MC %.6f (se %.1e)\n", caplet, mc_c, se_c);

        // ---- 4. Jamshidian swaption ------------------------------------ //
        const double expiry = 1.0;
        const std::vector<double> pay_times = {2.0, 3.0, 4.0, 5.0, 6.0};
        std::vector<double> par_times = {expiry};
        par_times.insert(par_times.end(), pay_times.begin(), pay_times.end());
        const double fwd_par = irm::par_swap_rate(usd, par_times);
        const irm::JamshidianResult res =
            hw.jamshidian_swaption(expiry, pay_times, fwd_par, /*notional=*/100.0);
        std::printf("\nJamshidian 1y-into-5y payer swaption (USD, strike = fwd par %.4f%%):\n",
                    100.0 * fwd_par);
        std::printf("  value = %.6f   r* = %.6f   |residual| = %.1e\n", res.value, res.r_star,
                    std::fabs(res.residual));
        const irm::HullWhite eur_hw(/*a=*/0.05, /*sigma=*/0.008, eur);
        const double eur_par = irm::par_swap_rate(eur, {1.0, 2.0, 3.0, 4.0});
        const irm::JamshidianResult eur_res =
            eur_hw.jamshidian_swaption(1.0, {2.0, 3.0, 4.0}, eur_par, /*notional=*/100.0);
        std::printf("  EUR (negative rates) 1y-into-3y @ %.4f%%: value = %.6f, r* = %.6f\n",
                    100.0 * eur_par, eur_res.value, eur_res.r_star);

        std::printf("\ndone.\n");
        return 0;
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "demo failed: %s\n", exc.what());
        return 1;
    }
}
