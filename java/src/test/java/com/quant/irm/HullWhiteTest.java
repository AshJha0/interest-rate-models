package com.quant.irm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
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

    // ---- validation of notional / strike / schedule (MAJOR-5) --------

    @Test
    public void nonFiniteNotionalAndMcStrikeRejected() {
        HullWhite hw = model();
        double nan = Double.NaN;
        double inf = Double.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class, () -> hw.caplet(1.0, 2.0, 0.025, nan));
        assertThrows(IllegalArgumentException.class, () -> hw.floorlet(1.0, 2.0, 0.025, inf));
        assertThrows(IllegalArgumentException.class,
                () -> hw.cap(new double[] {1.0, 2.0, 3.0}, 0.02, inf));
        assertThrows(IllegalArgumentException.class,
                () -> hw.jamshidianSwaption(1.0, new double[] {2.0, 3.0}, 0.02, nan, true));
        assertThrows(IllegalArgumentException.class,
                () -> hw.mcCaplet(1.0, 2.0, nan, 100.0, 8, 100, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> hw.mcCaplet(1.0, 2.0, -1.5, 100.0, 8, 100, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> hw.mcCaplet(1.0, 2.0, 0.02, inf, 8, 100, 1L));
        assertThrows(IllegalArgumentException.class, () -> hw.caplet(nan, 2.0, 0.02, 1.0));
        assertThrows(IllegalArgumentException.class, () -> hw.bFactor(nan, 5.0));
        assertThrows(IllegalArgumentException.class, () -> hw.bFactor(-1.0, 5.0));
        assertThrows(IllegalArgumentException.class,
                () -> hw.jamshidianSwaption(1.0, new double[] {2.0, inf}, 0.02, 1.0, true));
        assertThrows(IllegalArgumentException.class, () -> hw.theta(1.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> hw.zcbPrice(1.0, 5.0, -1e6));
        // A negative notional is a short position, not an error.
        assertEquals(-hw.caplet(1.0, 2.0, 0.025, 100.0), hw.caplet(1.0, 2.0, 0.025, -100.0),
                1e-15);
    }

    @Test
    public void capScheduleNonIncreasingRejected() {
        HullWhite hw = model();
        assertThrows(IllegalArgumentException.class,
                () -> hw.cap(new double[] {1.0, 2.0, 2.0, 3.0}, 0.02, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> hw.cap(new double[] {1.0, Double.NaN, 3.0}, 0.02, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> hw.cap(new double[] {-1.0, 1.0}, 0.02, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> hw.cap(new double[] {1.0, 2.0}, Double.NaN, 1.0));
    }

    // ---- Jamshidian contract (MAJOR-6, MINOR-4) -------------------------

    @Test
    public void jamshidianResidualContractEnforcedWithIllegalArgument() {
        HullWhite hw = model();
        double[] pays = {2.0, 3.0, 4.0, 5.0, 6.0};
        HullWhite.JamshidianResult res = hw.jamshidianSwaption(1.0, pays, 0.028, 100.0, true);
        HullWhite.JamshidianResult again =
                hw.jamshidianAtRStar(1.0, pays, 0.028, res.rStar(), 100.0, true);
        assertEquals(res.value(), again.value(), 1e-14);
        assertEquals(res.residual(), again.residual(), 0.0);
        assertTrue(Math.abs(again.residual()) < HullWhite.JAMSHIDIAN_RESIDUAL_TOL);
        // A wrong r* violates |g(r*)| < 1e-10: IllegalArgumentException (not IllegalState).
        assertThrows(IllegalArgumentException.class,
                () -> hw.jamshidianAtRStar(1.0, pays, 0.028, -1.0, 100.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> hw.jamshidianAtRStar(1.0, pays, 0.028, res.rStar() + 1e-6, 100.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> hw.jamshidianAtRStar(1.0, pays, 0.028, Double.NaN, 100.0, true));
        HullWhite.JamshidianResult rec =
                hw.jamshidianAtRStar(1.0, pays, 0.028, res.rStar(), 100.0, false);
        assertEquals(hw.jamshidianSwaption(1.0, pays, 0.028, 100.0, false).value(), rec.value(),
                1e-14);
    }

    /**
     * Payer swaption per unit notional by direct integration of the payoff
     * (1 - g(r_T))^+ under the T-forward Gaussian law of r_T (Simpson from
     * the payoff kink to mu + 12 sd) — independent of the decomposition.
     */
    private static double swaptionByIntegration(HullWhite hw, double expiry, double[] pays,
            double x) {
        int n = 2000;
        DiscountCurve c = hw.curve();
        double v = hw.sigma() * hw.sigma() * (-Math.expm1(-2.0 * hw.a() * expiry))
                / (2.0 * hw.a());
        double b1 = hw.bFactor(expiry, pays[0]);
        double a1 = hw.aFactor(expiry, pays[0]);
        double mu = (Math.log(a1 * c.df(expiry) / c.df(pays[0])) + b1 * b1 * v / 2.0) / b1;
        double[] coupons = new double[pays.length];
        double prev = expiry;
        for (int i = 0; i < pays.length; i++) {
            coupons[i] = x * (pays[i] - prev);
            prev = pays[i];
        }
        coupons[pays.length - 1] += 1.0;
        double sd = Math.sqrt(v);
        java.util.function.DoubleUnaryOperator payoff = r -> {
            double g = 0.0;
            for (int i = 0; i < pays.length; i++) {
                g += coupons[i] * hw.zcbPrice(expiry, pays[i], r);
            }
            return Math.max(1.0 - g, 0.0);
        };
        double lo = mu - 12.0 * sd;
        double hi = mu + 12.0 * sd;
        while (lo < mu - 400.0 * sd || payoff.applyAsDouble(lo) > 0.0) {
            lo -= 12.0 * sd;
        }
        for (int i = 0; i < 200; i++) {
            double mid = 0.5 * (lo + hi);
            if (payoff.applyAsDouble(mid) > 0.0) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        double kink = hi;
        double top = mu + 12.0 * sd;
        double h = (top - kink) / n;
        double total = 0.0;
        for (int i = 0; i <= n; i++) {
            double r = kink + i * h;
            double w = Math.exp(-0.5 * ((r - mu) / sd) * ((r - mu) / sd))
                    / (sd * Math.sqrt(2.0 * Math.PI));
            double coef = (i == 0 || i == n) ? 1.0 : ((i % 2 == 1) ? 4.0 : 2.0);
            total += coef * payoff.applyAsDouble(r) * w;
        }
        return c.df(expiry) * total * h / 3.0;
    }

    @Test
    public void jamshidianNegativeFixedRateStaysExact() {
        // For X < 0 the coupon signs are (-, ..., -, +): g is not monotone but
        // g - 1 still has a single crossing, so the decomposition is exact;
        // 1 + X tau_n <= 0 is the genuinely degenerate case.
        HullWhite hw = model();
        double[] pays = {2.0, 3.0, 4.0, 5.0, 6.0};
        for (double x : new double[] {-0.005, -0.2, -0.5}) {
            HullWhite.JamshidianResult res = hw.jamshidianSwaption(1.0, pays, x, 1.0, true);
            assertTrue(res.value() > 0.0);
            assertTrue(Math.abs(res.residual()) < 1e-10);
            assertEquals("X=" + x, 1.0, res.value() / swaptionByIntegration(hw, 1.0, pays, x),
                    1e-9);
        }
        assertTrue(hw.jamshidianSwaption(1.0, pays, -0.005, 1.0, true).rStar() < 0.0);
        assertThrows(IllegalArgumentException.class,
                () -> hw.jamshidianSwaption(1.0, pays, -1.0, 1.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> hw.jamshidianSwaption(1.0, pays, -1.5, 1.0, true));
        HullWhite.JamshidianResult ok = hw.jamshidianSwaption(1.0, pays, -0.8, 1.0, true);
        assertTrue(Math.abs(ok.residual()) < 1e-10);
        // Beyond about -0.95 the coupon-bond terms are O(1e8+) and |g(r*)| cannot
        // beat 1e-10 in double precision: the contract rejects it honestly.
        assertThrows(IllegalArgumentException.class,
                () -> hw.jamshidianSwaption(1.0, pays, -0.99, 1.0, true));
        HullWhite.JamshidianResult pos = hw.jamshidianSwaption(1.0, pays, 0.028, 1.0, true);
        assertEquals(1.0, pos.value() / swaptionByIntegration(hw, 1.0, pays, 0.028), 1e-9);
    }

    // ---- Monte Carlo determinism / small mean reversion ------------------

    @Test
    public void mcCapletSameSeedAndSinglePath() {
        HullWhite hw = model();
        McEstimate a = hw.mcCaplet(1.0, 2.0, 0.025, 100.0, 4, 500, 3L);
        McEstimate b = hw.mcCaplet(1.0, 2.0, 0.025, 100.0, 4, 500, 3L);
        McEstimate c = hw.mcCaplet(1.0, 2.0, 0.025, 100.0, 4, 500, 4L);
        assertEquals(a, b);
        assertTrue(a.price() != c.price());
        McEstimate single = hw.mcCaplet(1.0, 2.0, 0.02, 100.0, 1, 1, 1L);
        assertTrue(Double.isFinite(single.price()) && single.price() >= 0.0);
        assertEquals(0.0, single.standardError(), 0.0);
    }

    @Test
    public void mcCapletTinyMeanReversionUnbiased() {
        // a*dt = 5e-7 per step exercises the series branch of var_i and V(t).
        HullWhite slow = new HullWhite(1e-4, 0.01, usdLikeCurve());
        double analytic = slow.caplet(1.0, 2.0, 0.025, 100.0);
        McEstimate est = slow.mcCaplet(1.0, 2.0, 0.025, 100.0, 200, 20_000, 9L);
        assertTrue(Math.abs(est.price() - analytic) < 3.0 * est.standardError());
    }

    @Test
    public void varianceIntegralUsesCancellationFreeKernel() {
        HullWhite slow = new HullWhite(1e-5, 0.02, usdLikeCurve());
        assertEquals(0.5 * MathUtils.ouIntegralVariance(1e-5, 0.02, 1.0),
                slow.varianceIntegral(1.0), 1e-20);
        assertEquals(1.0, slow.varianceIntegral(1.0) / (0.02 * 0.02 / 6.0), 1e-5);
        assertEquals(0.0, slow.varianceIntegral(0.0), 0.0);
    }

    // ---- calibration of (a, sigma) (MAJOR-7) -------------------------------

    private static List<HullWhite.CapletQuote> capletQuotes(HullWhite truth) {
        List<HullWhite.CapletQuote> caps = new ArrayList<>();
        for (double r : new double[] {1.0, 2.0, 4.0}) {
            caps.add(new HullWhite.CapletQuote(r, r + 1.0, 0.025,
                    truth.caplet(r, r + 1.0, 0.025, 1.0)));
        }
        return caps;
    }

    private static List<HullWhite.SwaptionQuote> swaptionQuotes(HullWhite truth) {
        double[] pays = {2.0, 3.0, 4.0, 5.0, 6.0};
        return List.of(new HullWhite.SwaptionQuote(1.0, pays, 0.028,
                truth.jamshidianSwaption(1.0, pays, 0.028, 1.0, true).value()));
    }

    @Test
    public void calibrationRecoversAAndSigma() {
        HullWhite truth = new HullWhite(0.08, 0.012, usdLikeCurve());
        List<HullWhite.CapletQuote> caps = capletQuotes(truth);
        List<HullWhite.SwaptionQuote> swps = swaptionQuotes(truth);
        HullWhite.Calibration cal = HullWhite.calibrate(usdLikeCurve(), caps, swps, null, null, 4000);
        assertTrue(cal.converged() && cal.identified() && !cal.atBound() && !cal.aFixed());
        assertEquals(3, cal.nStarts());
        assertEquals(0.08, cal.a(), 1e-8);
        assertEquals(0.012, cal.sigma(), 1e-9);
        assertTrue(cal.rmse() < 1e-10);
        HullWhite model = cal.model(usdLikeCurve());
        for (HullWhite.CapletQuote q : caps) {
            assertEquals(q.price(), model.caplet(q.reset(), q.pay(), q.strike(), 1.0), 1e-10);
        }
    }

    @Test
    public void calibrationAFixedAndWarmStart() {
        HullWhite truth = new HullWhite(0.08, 0.012, usdLikeCurve());
        List<HullWhite.CapletQuote> caps = capletQuotes(truth);
        List<HullWhite.SwaptionQuote> swps = swaptionQuotes(truth);
        HullWhite.Calibration cal = HullWhite.calibrate(usdLikeCurve(), caps, swps, 0.08, null, 4000);
        assertTrue(cal.aFixed());
        assertEquals(0.08, cal.a(), 0.0);
        assertTrue(cal.converged() && cal.identified());
        assertEquals(0.012, cal.sigma(), 1e-9);
        HullWhite.Calibration warm = HullWhite.calibrate(
                usdLikeCurve(), caps, swps, null, new double[] {0.2, 0.02}, 4000);
        assertEquals(1, warm.nStarts());
        assertFalse(warm.identified());
        assertTrue(warm.converged());
        assertEquals(0.08, warm.a(), 1e-6);
        assertEquals(0.012, warm.sigma(), 1e-8);
        HullWhite.Calibration onlyCaps =
                HullWhite.calibrate(usdLikeCurve(), caps, List.of(), null, null, 4000);
        assertEquals(0.012, onlyCaps.sigma(), 1e-8);
        HullWhite.Calibration onlySwp =
                HullWhite.calibrate(usdLikeCurve(), null, swps, 0.08, null, 4000);
        assertEquals(0.012, onlySwp.sigma(), 1e-8);
    }

    @Test
    public void calibrationSingleQuoteNotIdentified() {
        HullWhite truth = new HullWhite(0.08, 0.012, usdLikeCurve());
        List<HullWhite.CapletQuote> caps = capletQuotes(truth).subList(0, 1);
        HullWhite.Calibration cal = HullWhite.calibrate(usdLikeCurve(), caps, null, null, null, 4000);
        assertTrue(cal.converged());
        assertFalse(cal.identified()); // one price, two parameters: a ridge
        assertTrue(cal.aSpread() > 1e-2);
        assertTrue(cal.rmse() < 1e-9);
    }

    @Test
    public void calibrationNonConvergenceAndValidation() {
        HullWhite truth = new HullWhite(0.08, 0.012, usdLikeCurve());
        List<HullWhite.CapletQuote> caps = capletQuotes(truth);
        List<HullWhite.SwaptionQuote> swps = swaptionQuotes(truth);
        DiscountCurve c = usdLikeCurve();
        HullWhite.Calibration res = HullWhite.calibrate(c, caps, swps, null, null, 2);
        assertFalse(res.converged());
        assertTrue(Double.isFinite(res.rmse()));
        assertThrows(IllegalArgumentException.class,
                () -> HullWhite.calibrate(c, List.of(), List.of(), null, null, 4000));
        assertThrows(IllegalArgumentException.class,
                () -> HullWhite.calibrate(c, caps, swps, 0.0, null, 4000));
        assertThrows(IllegalArgumentException.class,
                () -> HullWhite.calibrate(c, caps, swps, null, new double[] {0.1}, 4000));
        assertThrows(IllegalArgumentException.class,
                () -> HullWhite.calibrate(c, caps, swps, null, new double[] {Double.NaN, 0.01},
                        4000));
        assertThrows(IllegalArgumentException.class,
                () -> HullWhite.calibrate(null, caps, swps, null, null, 4000));
        assertThrows(IllegalArgumentException.class,
                () -> new HullWhite.CapletQuote(1.0, 2.0, 0.02, Double.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> new HullWhite.CapletQuote(2.0, 1.0, 0.02, 0.001));
        assertThrows(IllegalArgumentException.class,
                () -> new HullWhite.CapletQuote(1.0, 2.0, 0.02, -0.001));
        assertThrows(IllegalArgumentException.class,
                () -> new HullWhite.SwaptionQuote(1.0, new double[] {0.5, 2.0}, 0.02, 0.001));
        assertThrows(IllegalArgumentException.class,
                () -> new HullWhite.SwaptionQuote(1.0, new double[] {}, 0.02, 0.001));
        assertThrows(IllegalArgumentException.class,
                () -> new HullWhite.SwaptionQuote(Double.POSITIVE_INFINITY, new double[] {2.0},
                        0.02, 0.001));
    }
}
