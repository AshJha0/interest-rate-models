package com.quant.irm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.nio.file.Paths;
import org.junit.Test;

/** Vasicek closed forms, options, Monte Carlo and calibration. */
public class VasicekTest {

    private static Vasicek model() {
        return new Vasicek(0.5, 0.03, 0.01, 0.02);
    }

    @Test
    public void zcbPriceMatchesAffineFormulaByHand() {
        Vasicek m = model();
        double tau = 5.0;
        double b = (1.0 - Math.exp(-0.5 * tau)) / 0.5;
        double a = Math.exp((0.03 - 0.0001 / (2.0 * 0.25)) * (b - tau)
                - 0.0001 * b * b / (4.0 * 0.5));
        assertEquals(a * Math.exp(-b * 0.02), m.zcbPrice(5.0), 1e-15);
    }

    @Test
    public void yieldCurveConvergesToLongRunLevel() {
        Vasicek m = model();
        // The infinite-maturity yield is theta - sigma^2/(2 kappa^2).
        double yInf = 0.03 - 0.0001 / (2.0 * 0.25);
        assertEquals(yInf, m.zeroYield(2000.0), 1e-4);
    }

    @Test
    public void rDistributionMomentsMatchClosedForm() {
        Vasicek m = model();
        assertEquals(0.03 + (0.02 - 0.03) * Math.exp(-1.0), m.rMean(2.0), 1e-15);
        assertEquals(0.0001 * (1.0 - Math.exp(-2.0)) / 1.0, m.rVar(2.0), 1e-18);
        // Long-horizon variance tends to sigma^2/(2 kappa).
        assertEquals(0.0001 / 1.0, m.rVar(1e6), 1e-15);
    }

    @Test
    public void bondOptionPutCallParityHoldsOnAGridOfStrikes() {
        Vasicek m = model();
        double pT = m.zcbPrice(1.0);
        double pS = m.zcbPrice(3.0);
        for (double k = 0.85; k <= 1.05; k += 0.02) {
            double call = m.zbc(1.0, 3.0, k);
            double put = m.zbp(1.0, 3.0, k);
            // ZBC - ZBP = P(0,S) - K P(0,T)
            assertEquals("parity at K=" + k, pS - k * pT, call - put, 1e-14);
        }
    }

    @Test
    public void sigmaZeroOptionCollapsesToForwardIntrinsic() {
        Vasicek m = new Vasicek(0.5, 0.03, 0.0, 0.02);
        double pT = m.zcbPrice(1.0);
        double pS = m.zcbPrice(3.0);
        double kItm = 0.9;
        assertEquals(Math.max(pS - kItm * pT, 0.0), m.zbc(1.0, 3.0, kItm), 1e-15);
        double kOtm = 1.2;
        assertEquals(0.0, m.zbc(1.0, 3.0, kOtm), 0.0);
        assertEquals(Math.max(kOtm * pT - pS, 0.0), m.zbp(1.0, 3.0, kOtm), 1e-15);
    }

    @Test
    public void zeroExpiryOptionIsIntrinsic() {
        Vasicek m = model();
        double pS = m.zcbPrice(3.0);
        assertEquals(Math.max(pS - 0.9, 0.0), m.zbc(0.0, 3.0, 0.9), 1e-15);
    }

    @Test
    public void mcZcbWithinThreeStandardErrorsOfAnalytic() {
        Vasicek m = model();
        McEstimate est = m.mcZcb(5.0, 8, 40_000, 12345L);
        double analytic = m.zcbPrice(5.0);
        assertTrue("SE must be positive", est.standardError() > 0.0);
        assertTrue("MC " + est.price() + " vs analytic " + analytic
                        + " (3 SE = " + 3.0 * est.standardError() + ")",
                Math.abs(est.price() - analytic) < 3.0 * est.standardError());
    }

    @Test
    public void sigmaZeroMcIsDeterministicWithZeroStandardError() {
        Vasicek m = new Vasicek(0.5, 0.03, 0.0, 0.02);
        McEstimate est = m.mcZcb(5.0, 4, 100, 1L);
        assertEquals(m.zcbPrice(5.0), est.price(), 1e-12);
        assertEquals(0.0, est.standardError(), 0.0);
    }

    @Test
    public void negativeRatesAreSupportedThroughout() {
        Vasicek m = new Vasicek(0.3, 0.01, 0.008, -0.005);
        assertTrue("ZCB with negative r0 can exceed 1", m.zcbPrice(0.5) > 1.0);
        assertTrue(m.rMean(0.1) < 0.0);
        assertTrue(m.zbc(1.0, 2.0, 0.95) >= 0.0);
    }

    private static double[][] bundledYields() {
        return Quotes.loadZeroYields(Paths.get("..", "data").resolve("zero_yields.csv"));
    }

    @Test
    public void calibrationRecoversCleanParameters() {
        Vasicek truth = new Vasicek(0.8, 0.05, 0.02, 0.03);
        double[] ts = {0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0};
        double[] ys = new double[ts.length];
        for (int i = 0; i < ts.length; i++) {
            ys[i] = truth.zeroYield(ts[i]);
        }
        Vasicek.Calibration cal = Vasicek.calibrate(ts, ys, 0.03);
        assertTrue(cal.converged());
        assertFalse(cal.atBound());
        assertEquals(3, cal.nStarts());
        assertFalse(cal.sigmaFixed());
        assertEquals(0.8, cal.kappa(), 1e-3);
        assertEquals(0.05, cal.theta(), 1e-4);
        assertEquals(0.02, cal.sigma(), 1e-3);
        assertTrue(cal.rmse() < 1e-8);
    }

    @Test
    public void calibrationMultistartReportsNonIdentifiability() {
        // The bundled noisy yields have two least-squares minima with rmse within
        // 2 % of each other, at (0.80, 0.050, 0.023) and (0.41, 0.069, 0.080):
        // the diagnostics must say so instead of claiming a unique fit.
        double[][] zy = bundledYields();
        Vasicek.Calibration cal = Vasicek.calibrate(zy[0], zy[1], 0.03);
        assertTrue(cal.converged());
        assertFalse(cal.atBound());
        assertTrue(cal.rmse() < 2e-5);
        assertEquals(3, cal.nStarts());
        assertFalse(cal.identified());
        assertTrue(cal.sigmaSpread() > 0.01);
        assertTrue(cal.kappaSpread() > 0.1);
        assertTrue(cal.theta() >= 0.04 && cal.theta() <= 0.07);
        Vasicek m = cal.model();
        for (int i = 0; i < zy[0].length; i++) {
            assertEquals(zy[1][i], m.zeroYield(zy[0][i]), 5e-4);
        }
        double meanY = 0.0;
        for (double y : zy[1]) {
            meanY += y;
        }
        meanY /= zy[1].length;
        Vasicek.Calibration a = Vasicek.calibrate(
                zy[0], zy[1], 0.03, new double[] {0.5, meanY, 0.01}, 4000, null);
        Vasicek.Calibration b = Vasicek.calibrate(
                zy[0], zy[1], 0.03, new double[] {0.5, 0.05, 0.05}, 4000, null);
        assertEquals(1, a.nStarts());
        assertTrue(a.rmse() < 2e-5 && b.rmse() < 2e-5);
        assertTrue(Math.abs(a.rmse() - b.rmse()) < 5e-6);
        assertTrue(Math.abs(a.sigma() - b.sigma()) > 0.01);
        assertFalse(a.identified()); // single start: identifiability not assessed
    }

    @Test
    public void calibrationSigmaFixedIdentifiesKappaTheta() {
        double[][] zy = bundledYields();
        Vasicek.Calibration cal = Vasicek.calibrate(zy[0], zy[1], 0.03, null, 4000, 0.02);
        assertTrue(cal.sigmaFixed());
        assertEquals(0.02, cal.sigma(), 0.0);
        assertTrue(cal.converged() && cal.identified() && !cal.atBound());
        assertEquals(0.8002161, cal.kappa(), 1e-6);
        assertEquals(0.0499880, cal.theta(), 1e-6);
        assertTrue(cal.kappaSpread() < 1e-6);
        assertEquals(0.0, cal.sigmaSpread(), 0.0);
        // sigmaFixed with a warm start ignores x0[2].
        Vasicek.Calibration warm = Vasicek.calibrate(
                zy[0], zy[1], 0.03, new double[] {0.3, 0.04, 99.0}, 4000, 0.02);
        assertEquals(cal.kappa(), warm.kappa(), 1e-6);
        assertEquals(0.02, warm.sigma(), 0.0);
        assertThrows(IllegalArgumentException.class,
                () -> Vasicek.calibrate(zy[0], zy[1], 0.03, null, 4000, -0.01));
    }

    @Test
    public void calibrationPenaltyStartNotReportedConverged() {
        double[][] zy = bundledYields();
        Vasicek.Calibration cal = Vasicek.calibrate(
                zy[0], zy[1], 0.03, new double[] {-1.0, 0.0, -1.0}, 4000, null);
        assertTrue(cal.atBound());
        assertFalse(cal.converged()); // stopped at the penalty wall
        assertTrue(cal.kappa() <= 1e-5);
        assertTrue(Double.isFinite(cal.rmse()));
    }

    @Test
    public void calibrationFlatYieldsSigmaAtZeroBound() {
        // Flat yields equal to r0 are fitted exactly by sigma = 0, theta = r0 for
        // any kappa: sigma sits on its bound and kappa is not identified.
        double[] ts = {1.0, 2.0, 3.0, 5.0, 10.0};
        double[] ys = {0.03, 0.03, 0.03, 0.03, 0.03};
        Vasicek.Calibration cal = Vasicek.calibrate(ts, ys, 0.03);
        assertTrue(cal.atBound());
        assertFalse(cal.converged());
        assertTrue(cal.sigma() < 1e-5);
        assertEquals(0.03, cal.theta(), 1e-8);
        assertFalse(cal.identified());
    }

    @Test
    public void calibrationNonConvergenceReportedNotThrown() {
        Vasicek truth = new Vasicek(0.8, 0.05, 0.02, 0.03);
        double[] ts = {1.0, 2.0, 5.0, 10.0};
        double[] ys = new double[ts.length];
        for (int i = 0; i < ts.length; i++) {
            ys[i] = truth.zeroYield(ts[i]);
        }
        Vasicek.Calibration cal = Vasicek.calibrate(ts, ys, 0.03, null, 2, null);
        assertFalse(cal.converged());
        assertEquals(2, cal.iterations());
        assertTrue(Double.isFinite(cal.rmse()));
    }

    @Test
    public void calibrationInputValidation() {
        double[] ts = {1.0, 2.0};
        double[] ys = {0.03, 0.032};
        assertThrows(IllegalArgumentException.class,
                () -> Vasicek.calibrate(ts, ys, 0.03)); // < 3 points
        assertThrows(IllegalArgumentException.class,
                () -> Vasicek.calibrate(new double[] {1.0, 2.0, -3.0},
                        new double[] {0.03, 0.03, 0.03}, 0.03));
        assertThrows(IllegalArgumentException.class,
                () -> Vasicek.calibrate(new double[] {1.0, 2.0, 3.0},
                        new double[] {0.03, Double.NaN, 0.03}, 0.03));
        assertThrows(IllegalArgumentException.class,
                () -> Vasicek.calibrate(new double[] {1.0, 2.0, 3.0},
                        new double[] {0.03, 0.03, 0.03}, Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class,
                () -> Vasicek.calibrate(new double[] {1.0, 2.0, 3.0},
                        new double[] {0.03, 0.03, 0.03}, 0.03, null, 0, null));
    }

    @Test
    public void calibrationX0WrongLengthOrNonFiniteRejected() {
        double[] ts = {1.0, 2.0, 3.0};
        double[] ys = {0.02, 0.021, 0.022};
        // A 2-entry x0 used to throw ArrayIndexOutOfBoundsException.
        assertThrows(IllegalArgumentException.class,
                () -> Vasicek.calibrate(ts, ys, 0.02, new double[] {0.5, 0.03}, 4000, null));
        assertThrows(IllegalArgumentException.class,
                () -> Vasicek.calibrate(ts, ys, 0.02, new double[] {0.5, 0.03, 0.01, 0.0},
                        4000, null));
        assertThrows(IllegalArgumentException.class,
                () -> Vasicek.calibrate(ts, ys, 0.02, new double[] {0.5, Double.NaN, 0.01},
                        4000, null));
    }

    @Test
    public void mcZcbManySmallStepsUnbiased() {
        // kappa*dt = 2.5e-6 per step: the naive var_i would be wrong by orders of
        // magnitude; with the series the estimator stays unbiased.
        Vasicek m = new Vasicek(1e-3, 0.03, 0.01, 0.02);
        double analytic = m.zcbPrice(5.0);
        McEstimate est = m.mcZcb(5.0, 2000, 10_000, 77L);
        assertTrue(est.standardError() > 0.0);
        assertTrue(Math.abs(est.price() - analytic) < 3.0 * est.standardError());
    }

    @Test
    public void sameSeedSameResultAndSinglePath() {
        Vasicek m = model();
        McEstimate a = m.mcZcb(3.0, 4, 2000, 11L);
        McEstimate b = m.mcZcb(3.0, 4, 2000, 11L);
        McEstimate c = m.mcZcb(3.0, 4, 2000, 12L);
        assertEquals(a, b);
        assertTrue(a.price() != c.price());
        McEstimate single = m.mcZcb(1.0, 1, 1, 5L);
        assertTrue(Double.isFinite(single.price()) && single.price() > 0.0);
        assertEquals(0.0, single.standardError(), 0.0);
    }

    @Test
    public void rMeanWithExplicitShortRate() {
        Vasicek m = model();
        assertEquals(m.rMean(2.0), m.rMean(2.0, 0.02), 0.0);
        assertEquals(0.03 + (0.05 - 0.03) * Math.exp(-1.0), m.rMean(2.0, 0.05), 1e-15);
        assertThrows(IllegalArgumentException.class, () -> m.rMean(2.0, Double.NaN));
    }

    @Test
    public void nonFiniteAndOverflowRejected() {
        Vasicek m = model();
        assertThrows(IllegalArgumentException.class, () -> m.zcbPrice(Double.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> m.zcbPrice(5.0, 0.0, Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class,
                () -> m.zeroYield(Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> m.zcbPrice(5.0, 0.0, -1e6));
    }

    @Test
    public void invalidModelParametersAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Vasicek(0.0, 0.03, 0.01, 0.02));
        assertThrows(IllegalArgumentException.class, () -> new Vasicek(-0.5, 0.03, 0.01, 0.02));
        assertThrows(IllegalArgumentException.class, () -> new Vasicek(0.5, 0.03, -0.01, 0.02));
        assertThrows(IllegalArgumentException.class,
                () -> new Vasicek(0.5, Double.NaN, 0.01, 0.02));
        Vasicek m = model();
        assertThrows(IllegalArgumentException.class, () -> m.zcbPrice(1.0, 2.0, 0.02));
        assertThrows(IllegalArgumentException.class, () -> m.zeroYield(0.0));
        assertThrows(IllegalArgumentException.class, () -> m.zbc(1.0, 3.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> m.zbc(1.0, 3.0, -0.5));
        assertThrows(IllegalArgumentException.class, () -> m.zbc(2.0, 1.0, 0.9));
        assertThrows(IllegalArgumentException.class, () -> m.rMean(-1.0));
        assertThrows(IllegalArgumentException.class, () -> m.simulate(0.0, 8, 100, 1L));
        assertThrows(IllegalArgumentException.class, () -> m.simulate(1.0, 0, 100, 1L));
        assertThrows(IllegalArgumentException.class, () -> m.simulate(1.0, 8, 0, 1L));
    }
}
