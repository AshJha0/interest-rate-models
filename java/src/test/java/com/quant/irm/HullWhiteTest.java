package com.quant.irm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Hull-White fit-to-curve properties, options, swaptions and Monte Carlo. */
public class HullWhiteTest {

    /** Positive-rate fixture curve (same pillars as the golden HW cases). */
    private static DiscountCurve usdLikeCurve() {
        return new DiscountCurve(
                new double[] {1.0, 2.0, 3.0, 5.0, 10.0},
                new double[] {0.9787294774691476, 0.9550419621907147, 0.9291361457915193,
                        0.8715343499971578, 0.7046880897187134});
    }

    /** Negative-rate fixture curve (EUR style, DF > 1 at the short end). */
    private static DiscountCurve negativeCurve() {
        return new DiscountCurve(
                new double[] {1.0, 2.0, 3.0, 5.0, 10.0},
                new double[] {1.0028039236612292, 1.0032051254657053, 1.0012007202880864,
                        0.9900498337491681, 0.9231163463866358});
    }

    private static HullWhite model() {
        return new HullWhite(0.1, 0.01, usdLikeCurve());
    }

    @Test
    public void zcbAtTimeZeroReproducesMarketDfsExactly() {
        HullWhite hw = model();
        DiscountCurve c = usdLikeCurve();
        for (double t : new double[] {0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0}) {
            assertEquals("P(0," + t + ")", c.df(t), hw.zcbPrice(0.0, t), 1e-12);
        }
    }

    @Test
    public void fd0ForwardMatchesSegmentForwardInsideSegments() {
        HullWhite hw = model();
        DiscountCurve c = usdLikeCurve();
        // Away from pillars the FD equals the piecewise-constant forward.
        assertEquals(c.instForward(1.5), hw.fwd0(1.5), 1e-9);
        assertEquals(c.instForward(4.0), hw.fwd0(4.0), 1e-9);
        // At a pillar the central stencil averages the adjacent forwards.
        double avg = 0.5 * (c.instForward(1.9999) + c.instForward(2.0001));
        assertEquals(avg, hw.fwd0(2.0), 1e-9);
    }

    @Test
    public void thetaContainsForwardSlopeMeanReversionAndConvexityTerms() {
        HullWhite hw = model();
        double t = 1.5;
        // Inside a segment the forward is locally constant: theta ~ a*f + conv.
        double expected = hw.a() * hw.fwd0(t)
                + (hw.sigma() * hw.sigma() / (2.0 * hw.a()))
                * (1.0 - Math.exp(-2.0 * hw.a() * t));
        assertEquals(expected, hw.theta(t), 1e-6);
    }

    @Test
    public void zcbOptionPutCallParityHoldsOnAGridOfStrikes() {
        HullWhite hw = model();
        DiscountCurve c = usdLikeCurve();
        double pT = c.df(1.0);
        double pS = c.df(2.0);
        for (double k = 0.9; k <= 1.02; k += 0.01) {
            // ZBC - ZBP = P(0,S) - K P(0,T)
            assertEquals("parity at K=" + k,
                    pS - k * pT, hw.zbc(1.0, 2.0, k) - hw.zbp(1.0, 2.0, k), 1e-14);
        }
    }

    @Test
    public void capletDecompositionEqualsCap() {
        HullWhite hw = model();
        double[] sched = {0.5, 1.0, 2.0, 3.0};
        double strike = 0.024;
        double sum = hw.caplet(0.5, 1.0, strike, 100.0)
                + hw.caplet(1.0, 2.0, strike, 100.0)
                + hw.caplet(2.0, 3.0, strike, 100.0);
        assertEquals(sum, hw.cap(sched, strike, 100.0), 1e-12);
    }

    @Test
    public void capletFloorletParityMatchesForwardSwaplet() {
        HullWhite hw = model();
        DiscountCurve c = usdLikeCurve();
        double strike = 0.025;
        double tau = 1.0;
        // caplet - floorlet = tau DF(pay) (F - K) (unit notional)
        double lhs = hw.caplet(1.0, 2.0, strike, 1.0) - hw.floorlet(1.0, 2.0, strike, 1.0);
        double rhs = tau * c.df(2.0) * (c.fwdRate(1.0, 2.0) - strike);
        assertEquals(rhs, lhs, 1e-14);
    }

    @Test
    public void sigmaZeroCapletIsDiscountedIntrinsic() {
        DiscountCurve c = usdLikeCurve();
        HullWhite hw0 = new HullWhite(0.1, 0.0, c);
        double strike = 0.02;
        double expected = 100.0 * c.df(2.0) * 1.0
                * Math.max(c.fwdRate(1.0, 2.0) - strike, 0.0);
        assertEquals(expected, hw0.caplet(1.0, 2.0, strike, 100.0), 1e-12);
        // OTM sigma=0 caplet is exactly zero.
        assertEquals(0.0, hw0.caplet(1.0, 2.0, 0.5, 100.0), 0.0);
    }

    @Test
    public void jamshidianRootResidualIsTiny() {
        HullWhite hw = model();
        HullWhite.JamshidianResult res = hw.jamshidianSwaption(
                1.0, new double[] {2.0, 3.0, 4.0, 5.0, 6.0}, 0.028, 100.0, true);
        assertTrue("residual " + res.residual(), Math.abs(res.residual()) < 1e-10);
        assertEquals(5, res.strikes().length);
        for (double k : res.strikes()) {
            assertTrue("strikes must be positive", k > 0.0);
        }
    }

    @Test
    public void swaptionValueIsAtLeastIntrinsic() {
        HullWhite hw = model();
        DiscountCurve c = usdLikeCurve();
        double expiry = 1.0;
        double[] payTimes = {2.0, 3.0, 4.0, 5.0, 6.0};
        double fixedRate = 0.028;
        HullWhite.JamshidianResult res =
                hw.jamshidianSwaption(expiry, payTimes, fixedRate, 100.0, true);
        // Intrinsic = PV(floating) - PV(fixed) of the forward swap, floored.
        double annuity = 0.0;
        double prev = expiry;
        for (double t : payTimes) {
            annuity += (t - prev) * c.df(t);
            prev = t;
        }
        double intrinsic = Math.max(
                (c.df(expiry) - c.df(payTimes[payTimes.length - 1])) - fixedRate * annuity, 0.0);
        assertTrue("swaption " + res.value() + " >= intrinsic " + 100.0 * intrinsic,
                res.value() >= 100.0 * intrinsic - 1e-12);
    }

    @Test
    public void payerReceiverParityMatchesForwardSwapValue() {
        HullWhite hw = model();
        DiscountCurve c = usdLikeCurve();
        double expiry = 1.0;
        double[] payTimes = {2.0, 3.0, 4.0};
        double x = 0.03;
        double payer = hw.jamshidianSwaption(expiry, payTimes, x, 1.0, true).value();
        double receiver = hw.jamshidianSwaption(expiry, payTimes, x, 1.0, false).value();
        double annuity = 0.0;
        double prev = expiry;
        for (double t : payTimes) {
            annuity += (t - prev) * c.df(t);
            prev = t;
        }
        double fwdSwap = (c.df(expiry) - c.df(payTimes[2])) - x * annuity;
        assertEquals("payer - receiver = forward swap PV", fwdSwap, payer - receiver, 1e-12);
    }

    @Test
    public void negativeRateCurveIsFullySupported() {
        DiscountCurve neg = negativeCurve();
        HullWhite hw = new HullWhite(0.1, 0.01, neg);
        assertTrue("DF(1) > 1 on the negative curve", neg.df(1.0) > 1.0);
        for (double t : new double[] {1.0, 3.0, 5.0}) {
            assertEquals(neg.df(t), hw.zcbPrice(0.0, t), 1e-12);
        }
        HullWhite.JamshidianResult res = hw.jamshidianSwaption(
                1.0, new double[] {2.0, 3.0, 4.0}, 0.0, 100.0, true);
        assertTrue(res.value() > 0.0);
        assertTrue(Math.abs(res.residual()) < 1e-10);
        assertTrue("r* should be negative on this curve", res.rStar() < 0.0);
    }

    @Test
    public void mcCapletWithinThreeStandardErrorsOfAnalytic() {
        HullWhite hw = model();
        double analytic = hw.caplet(1.0, 2.0, 0.025, 100.0);
        McEstimate est = hw.mcCaplet(1.0, 2.0, 0.025, 100.0, 8, 40_000, 777L);
        assertTrue("SE must be positive", est.standardError() > 0.0);
        assertTrue("MC " + est.price() + " vs analytic " + analytic
                        + " (3 SE = " + 3.0 * est.standardError() + ")",
                Math.abs(est.price() - analytic) < 3.0 * est.standardError());
    }

    @Test
    public void sigmaZeroMcCapletIsDeterministicIntrinsic() {
        HullWhite hw0 = new HullWhite(0.1, 0.0, usdLikeCurve());
        McEstimate est = hw0.mcCaplet(1.0, 2.0, 0.02, 100.0, 4, 50, 1L);
        assertEquals(hw0.caplet(1.0, 2.0, 0.02, 100.0), est.price(), 1e-10);
        assertEquals(0.0, est.standardError(), 0.0);
    }

    @Test
    public void invalidInputsAreRejected() {
        DiscountCurve c = usdLikeCurve();
        assertThrows(IllegalArgumentException.class, () -> new HullWhite(0.0, 0.01, c));
        assertThrows(IllegalArgumentException.class, () -> new HullWhite(0.1, -0.01, c));
        assertThrows(IllegalArgumentException.class, () -> new HullWhite(0.1, 0.01, null));
        HullWhite hw = model();
        assertThrows(IllegalArgumentException.class, () -> hw.fwd0(-1.0));
        assertThrows(IllegalArgumentException.class, () -> hw.zbc(1.0, 2.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> hw.zbc(2.0, 1.0, 0.9));
        assertThrows(IllegalArgumentException.class, () -> hw.caplet(2.0, 1.0, 0.02, 1.0));
        assertThrows(IllegalArgumentException.class, () -> hw.caplet(1.0, 2.0, -1.5, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> hw.cap(new double[] {1.0}, 0.02, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> hw.jamshidianSwaption(0.0, new double[] {1.0}, 0.02, 1.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> hw.jamshidianSwaption(1.0, new double[] {}, 0.02, 1.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> hw.jamshidianSwaption(1.0, new double[] {2.0, 2.0}, 0.02, 1.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> hw.mcCaplet(0.0, 1.0, 0.02, 1.0, 8, 100, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> hw.mcCaplet(1.0, 2.0, 0.02, 1.0, 0, 100, 1L));
    }
}
