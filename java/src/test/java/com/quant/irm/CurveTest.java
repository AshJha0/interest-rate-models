package com.quant.irm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Discount-curve construction, interpolation, derived rates and validation. */
public class CurveTest {

    private static final double[] TIMES = {1.0, 2.0, 3.0, 5.0};
    private static final double[] DFS = {0.99, 0.975, 0.955, 0.9};

    private static DiscountCurve curve() {
        return new DiscountCurve(TIMES, DFS);
    }

    @Test
    public void interpolationIsExactAtEveryPillar() {
        DiscountCurve c = curve();
        for (int i = 0; i < TIMES.length; i++) {
            assertEquals("pillar " + TIMES[i], DFS[i], c.df(TIMES[i]), 0.0);
        }
        assertEquals(1.0, c.df(0.0), 0.0);
    }

    @Test
    public void logLinearInterpolationBetweenPillars() {
        DiscountCurve c = curve();
        // ln DF(2.5) = (ln 0.975 + ln 0.955)/2 for the midpoint of [2,3].
        double expected = Math.exp(0.5 * (Math.log(0.975) + Math.log(0.955)));
        assertEquals(expected, c.df(2.5), 1e-15);
    }

    @Test
    public void flatForwardExtrapolationContinuesLastSlope() {
        DiscountCurve c = curve();
        double lastFwd = -(Math.log(0.9) - Math.log(0.955)) / 2.0;
        assertEquals(0.9 * Math.exp(-lastFwd * 2.0), c.df(7.0), 1e-14);
        assertEquals(lastFwd, c.instForward(100.0), 1e-15);
    }

    @Test
    public void dfPositiveAndDecreasingForPositiveRateCurve() {
        DiscountCurve c = curve();
        double prev = Double.MAX_VALUE;
        for (double t = 0.1; t <= 8.0; t += 0.1) {
            double df = c.df(t);
            assertTrue("DF must be positive at t=" + t, df > 0.0);
            assertTrue("DF must decrease at t=" + t, df < prev);
            prev = df;
        }
    }

    @Test
    public void negativeRatesGiveDfAboveOneAndAreSupported() {
        // EUR-style short end: DF(1) > 1 is legal.
        DiscountCurve c = new DiscountCurve(
                new double[] {1.0, 2.0}, new double[] {1.003, 1.004});
        assertTrue(c.df(1.0) > 1.0);
        assertTrue(c.df(0.5) > 1.0);
        assertTrue(c.zeroRate(1.0) < 0.0);
    }

    @Test
    public void zeroRateMatchesDefinitionAndLimitAtZero() {
        DiscountCurve c = curve();
        assertEquals(-Math.log(c.df(2.5)) / 2.5, c.zeroRate(2.5), 1e-16);
        // At t = 0 the first segment's instantaneous forward is returned.
        assertEquals(-Math.log(0.99), c.zeroRate(0.0), 1e-15);
    }

    @Test
    public void forwardRateConsistentWithDiscountFactors() {
        DiscountCurve c = curve();
        // Property-style grid check: F(t1,t2) from DFs at many pairs.
        for (double t1 = 0.0; t1 < 6.0; t1 += 0.7) {
            double t2 = t1 + 0.9;
            double expected = (c.df(t1) / c.df(t2) - 1.0) / (t2 - t1);
            assertEquals(expected, c.fwdRate(t1, t2), 1e-15);
        }
    }

    @Test
    public void instForwardUsesRightSegmentAtPillar() {
        DiscountCurve c = curve();
        double fwdSeg23 = -(Math.log(0.955) - Math.log(0.975)) / 1.0;
        assertEquals(fwdSeg23, c.instForward(2.0), 1e-15);
    }

    @Test
    public void parSwapRateMatchesManualComputation() {
        DiscountCurve c = curve();
        double[] times = {0.0, 1.0, 2.0, 3.0};
        double annuity = c.df(1.0) + c.df(2.0) + c.df(3.0);
        assertEquals((1.0 - c.df(3.0)) / annuity,
                DiscountCurve.parSwapRate(c, times), 1e-15);
    }

    @Test
    public void invalidConstructionIsRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> new DiscountCurve(new double[] {}, new double[] {}));
        assertThrows(IllegalArgumentException.class,
                () -> new DiscountCurve(new double[] {1.0, 2.0}, new double[] {0.99}));
        assertThrows(IllegalArgumentException.class, // duplicate pillar
                () -> new DiscountCurve(new double[] {1.0, 1.0}, new double[] {0.99, 0.98}));
        assertThrows(IllegalArgumentException.class, // non-monotone
                () -> new DiscountCurve(new double[] {2.0, 1.0}, new double[] {0.99, 0.98}));
        assertThrows(IllegalArgumentException.class, // t <= 0
                () -> new DiscountCurve(new double[] {0.0, 1.0}, new double[] {1.0, 0.99}));
        assertThrows(IllegalArgumentException.class, // DF <= 0
                () -> new DiscountCurve(new double[] {1.0}, new double[] {-0.5}));
        assertThrows(IllegalArgumentException.class, // NaN DF
                () -> new DiscountCurve(new double[] {1.0}, new double[] {Double.NaN}));
        assertThrows(IllegalArgumentException.class, // non-finite time
                () -> new DiscountCurve(
                        new double[] {Double.POSITIVE_INFINITY}, new double[] {0.9}));
    }

    @Test
    public void invalidQueriesAreRejected() {
        DiscountCurve c = curve();
        assertThrows(IllegalArgumentException.class, () -> c.df(-0.5));
        assertThrows(IllegalArgumentException.class, () -> c.df(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> c.zeroRate(-1.0));
        assertThrows(IllegalArgumentException.class, () -> c.fwdRate(2.0, 2.0));
        assertThrows(IllegalArgumentException.class, () -> c.fwdRate(-1.0, 2.0));
        assertThrows(IllegalArgumentException.class,
                () -> DiscountCurve.parSwapRate(c, new double[] {0.0}));
        assertThrows(IllegalArgumentException.class,
                () -> DiscountCurve.parSwapRate(c, new double[] {0.0, 2.0, 1.0}));
    }
}
