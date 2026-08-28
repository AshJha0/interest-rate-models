package com.quant.irm;

import static org.junit.Assert.assertEquals;
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

    @Test
    public void calibrationRecoversGeneratingParameters() {
        // data/zero_yields.csv was generated from (0.8, 0.05, 0.02, r0=0.03)
        // plus N(0, 2e-5) noise.
        double[][] zy = Quotes.loadZeroYields(
                Paths.get("..", "data").resolve("zero_yields.csv"));
        Vasicek.Calibration cal = Vasicek.calibrate(zy[0], zy[1], 0.03);
        assertTrue("calibration should converge", cal.converged());
        assertEquals(0.8, cal.kappa(), 0.15);
        assertEquals(0.05, cal.theta(), 0.005);
        assertTrue("rmse should be at noise level, got " + cal.rmse(),
                cal.rmse() < 1e-4);
        // The calibrated model reprices the target yields closely.
        Vasicek m = cal.model();
        for (int i = 0; i < zy[0].length; i++) {
            assertEquals(zy[1][i], m.zeroYield(zy[0][i]), 5e-4);
        }
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
