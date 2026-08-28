package com.quant.irm;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * End-to-end demo for the irm package.
 *
 * <p>Bootstraps the EUR (negative short end) and USD (normal) curves plus
 * the OIS curve, prints zero/forward tables, calibrates a Vasicek model to
 * the bundled zero yields, and prices Hull-White caps and a Jamshidian
 * payer swaption. Run via {@code ./demo.sh} (after {@code ./build.sh}).</p>
 */
public final class Demo {

    private Demo() {
    }

    private static void printCurveTable(String name, DiscountCurve curve) {
        System.out.println();
        System.out.println(name + "  (ACT/365F, cont. comp. zeros, simple 1y-forwards)");
        System.out.printf("  %5s %10s %8s %13s%n", "T", "DF", "zero %", "fwd(T,T+1) %");
        double[] grid = {0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0};
        for (double t : grid) {
            double fwd = curve.fwdRate(t, t + 1.0);
            System.out.printf("  %5.2f %10.6f %8.4f %13.4f%n",
                    t, curve.df(t), 100.0 * curve.zeroRate(t), 100.0 * fwd);
        }
    }

    /**
     * Runs the demo report.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        Path data = Paths.get("..", "data");

        System.out.println("=".repeat(64));
        System.out.println("irm demo -- bootstrapping, Vasicek, Hull-White (Java port)");
        System.out.println("=".repeat(64));

        // ---- 1. bootstrap both quote curves + OIS ---------------------
        DiscountCurve eur = Bootstrap.bootstrap(
                Quotes.loadCurveQuotes(data.resolve("curve_quotes.csv"), "EUR"));
        DiscountCurve usd = Bootstrap.bootstrap(
                Quotes.loadCurveQuotes(data.resolve("curve_quotes.csv"), "USD"));
        DiscountCurve ois = Bootstrap.bootstrap(
                Quotes.loadOisQuotes(data.resolve("ois_quotes.csv")));
        printCurveTable("EUR curve (negative short end)", eur);
        printCurveTable("USD curve (normal/inverted)", usd);
        System.out.printf("%nOIS discount curve: DF(1y)=%.6f  DF(10y)=%.6f%n",
                ois.df(1.0), ois.df(10.0));
        System.out.printf("Sanity: EUR DF(1y) = %.6f > 1 (negative rates)%n", eur.df(1.0));

        // ---- 2. Vasicek calibration -----------------------------------
        double[][] zy = Quotes.loadZeroYields(data.resolve("zero_yields.csv"));
        double[] ts = zy[0];
        double[] ys = zy[1];
        Vasicek.Calibration cal = Vasicek.calibrate(ts, ys, 0.03);
        System.out.println();
        System.out.println("Vasicek calibration to data/zero_yields.csv (r0 = 3%):");
        System.out.printf(
                "  kappa=%.4f  theta=%.4f  sigma=%.4f  rmse=%.2e  iters=%d  converged=%b%n",
                cal.kappa(), cal.theta(), cal.sigma(), cal.rmse(),
                cal.iterations(), cal.converged());
        Vasicek model = cal.model();
        System.out.printf("  %4s %11s %9s%n", "T", "mkt yield %", "model %");
        for (int i = 0; i < ts.length; i += 3) {
            System.out.printf("  %4.1f %11.4f %9.4f%n",
                    ts[i], 100.0 * ys[i], 100.0 * model.zeroYield(ts[i]));
        }
        McEstimate mc = model.mcZcb(5.0, 8, 50_000, 42L);
        System.out.printf("  ZCB(5y): analytic %.6f  MC %.6f (se %.1e)%n",
                model.zcbPrice(5.0), mc.price(), mc.standardError());

        // ---- 3. Hull-White caps on the USD curve ----------------------
        HullWhite hw = new HullWhite(0.1, 0.01, usd);
        double atm = DiscountCurve.parSwapRate(usd, new double[] {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
        System.out.printf("%nHull-White (a=0.10, sigma=0.01) on USD curve; 5y par = %.4f%%%n",
                100.0 * atm);
        System.out.printf("  %14s %9s %18s%n", "cap", "strike %", "PV (notional 100)");
        String[] labels = {"5y ATM", "5y ATM+100bp", "5y ATM-100bp"};
        double[] strikes = {atm, atm + 0.01, atm - 0.01};
        double[] capSchedule = {1.0, 2.0, 3.0, 4.0, 5.0};
        for (int i = 0; i < labels.length; i++) {
            double pv = hw.cap(capSchedule, strikes[i], 100.0);
            System.out.printf("  %14s %9.4f %18.6f%n", labels[i], 100.0 * strikes[i], pv);
        }
        double caplet = hw.caplet(1.0, 2.0, atm, 100.0);
        McEstimate mcC = hw.mcCaplet(1.0, 2.0, atm, 100.0, 8, 50_000, 7L);
        System.out.printf("  caplet(1y->2y) analytic %.6f | MC %.6f (se %.1e)%n",
                caplet, mcC.price(), mcC.standardError());

        // ---- 4. Jamshidian swaption -----------------------------------
        double expiry = 1.0;
        double[] payTimes = {2.0, 3.0, 4.0, 5.0, 6.0};
        double fwdPar = DiscountCurve.parSwapRate(
                usd, new double[] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        HullWhite.JamshidianResult res =
                hw.jamshidianSwaption(expiry, payTimes, fwdPar, 100.0, true);
        System.out.printf(
                "%nJamshidian 1y-into-5y payer swaption (USD, strike = fwd par %.4f%%):%n",
                100.0 * fwdPar);
        System.out.printf("  value = %.6f   r* = %.6f   |residual| = %.1e%n",
                res.value(), res.rStar(), Math.abs(res.residual()));
        HullWhite eurHw = new HullWhite(0.05, 0.008, eur);
        double eurPar = DiscountCurve.parSwapRate(eur, new double[] {1.0, 2.0, 3.0, 4.0});
        HullWhite.JamshidianResult eurRes = eurHw.jamshidianSwaption(
                1.0, new double[] {2.0, 3.0, 4.0}, eurPar, 100.0, true);
        System.out.printf("  EUR (negative rates) 1y-into-3y @ %.4f%%: value = %.6f, r* = %.6f%n",
                100.0 * eurPar, eurRes.value(), eurRes.rStar());

        System.out.println();
        System.out.println("done.");
    }
}
