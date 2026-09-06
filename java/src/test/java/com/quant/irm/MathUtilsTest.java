package com.quant.irm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Shared kernels: normal CDF/PDF and the exact OU step moments. Reference
 * values for the integrated-OU variance come from a 60-digit evaluation of
 * the closed form (python/tests/test_mathutils.py derives them with
 * {@code decimal}; the same numbers are pinned here).
 */
public class MathUtilsTest {

    /** {a, sigma, dt, var_i, var_x} with the variances at 60-digit precision. */
    private static final double[][] CASES = {
        {0.001, 0.01, 0.01, 3.3333083334499993e-11, 9.9999000006666625e-07},
        {0.0001, 0.01, 0.001, 3.3333330833333452e-14, 9.9999990000000661e-08},
        {1e-06, 0.02, 0.125, 2.6041664225260560e-07, 4.9999993750000522e-05},
        {0.01, 0.01, 1.0, 3.3084495845603743e-05, 9.9006633466223494e-05},
        {0.05, 0.02, 0.01, 1.3328334499791697e-10, 3.9980006665000334e-06},
        {0.1, 0.01, 0.125, 6.4497359652182101e-08, 1.2345043985833665e-05},
        {0.8, 0.02, 2.5, 5.9493183361629550e-04, 2.4542109027781648e-04},
        {5.0, 0.01, 1.0, 2.8107625552266318e-06, 9.9995460007023755e-06},
    };

    @Test
    public void ouMomentsSmallADtAccurate() {
        for (double[] c : CASES) {
            double a = c[0];
            double sigma = c[1];
            double dt = c[2];
            MathUtils.OuStepMoments m = MathUtils.ouStepMoments(a, sigma, dt);
            assertEquals("var_i a=" + a + " dt=" + dt, 1.0, m.varI() / c[3], 1e-9);
            assertEquals("var_x a=" + a + " dt=" + dt, 1.0, m.varX() / c[4], 1e-12);
            assertEquals(-Math.expm1(-a * dt) / a, m.b(), 1e-14 * m.b());
            // Cauchy-Schwarz: the (x', I) covariance matrix must be PSD.
            assertTrue(m.cov() * m.cov() <= m.varX() * m.varI() * (1.0 + 1e-12));
            double x = a * dt;
            if (x < 1e-3) {
                double series = sigma * sigma * dt * dt * dt
                        * (1.0 / 3.0 - x / 4.0 + 7.0 * x * x / 60.0 - x * x * x / 24.0);
                assertEquals(1.0, m.varI() / series, 1e-9);
            }
        }
    }

    @Test
    public void naiveFormulaIsWrongForTinyADt() {
        // Documents the bug the series fixes: plain evaluation is off by orders
        // of magnitude at a*dt = 1e-7 (80x per the review measurement).
        double a = 1e-4;
        double sigma = 0.01;
        double dt = 1e-3;
        double decay = Math.exp(-a * dt);
        double b = (1.0 - decay) / a;
        double naive = (sigma * sigma / (a * a))
                * (dt - 2.0 * b + (1.0 - decay * decay) / (2.0 * a));
        double ref = 3.3333330833333452e-14;
        assertTrue(Math.abs(naive - ref) / ref > 10.0);
        assertEquals(1.0, MathUtils.ouIntegralVariance(a, sigma, dt) / ref, 1e-12);
    }

    @Test
    public void seriesAndClosedFormAgreeAtThreshold() {
        double x = MathUtils.OU_SERIES_THRESHOLD;
        double below = MathUtils.ouIntegralVariance(x * (1.0 - 1e-12), 0.01, 1.0);
        double above = MathUtils.ouIntegralVariance(x * (1.0 + 1e-12), 0.01, 1.0);
        assertEquals(1.0, below / above, 1e-9);
    }

    @Test
    public void ouMomentsValidation() {
        assertThrows(IllegalArgumentException.class, () -> MathUtils.ouStepMoments(0.0, 0.01, 0.1));
        assertThrows(IllegalArgumentException.class, () -> MathUtils.ouStepMoments(0.1, 0.01, 0.0));
        assertThrows(IllegalArgumentException.class,
                () -> MathUtils.ouStepMoments(Double.NaN, 0.01, 0.1));
        assertThrows(IllegalArgumentException.class, () -> MathUtils.ouStepMoments(0.1, -0.01, 0.1));
        assertThrows(IllegalArgumentException.class,
                () -> MathUtils.ouIntegralVariance(0.1, Double.POSITIVE_INFINITY, 0.1));
        MathUtils.OuStepMoments m = MathUtils.ouStepMoments(0.3, 0.0, 0.5);
        assertEquals(0.0, m.varX(), 0.0);
        assertEquals(0.0, m.varI(), 0.0);
        assertEquals(0.0, m.cov(), 0.0);
        assertEquals(Math.exp(-0.15), m.decay(), 1e-16);
    }

    @Test
    public void normalCdfAndPdf() {
        assertEquals(0.5, MathUtils.normCdf(0.0), 0.0);
        assertEquals(0.8413447460685429, MathUtils.normCdf(1.0), 1e-15);
        assertEquals(1.0, MathUtils.normCdf(-8.0) / 6.220960574271786e-16, 1e-10);
        assertEquals(1.0 / Math.sqrt(2.0 * Math.PI), MathUtils.normPdf(0.0), 1e-16);
        assertEquals(Math.exp(-2.0) / Math.sqrt(2.0 * Math.PI), MathUtils.normPdf(2.0), 1e-16);
    }
}
