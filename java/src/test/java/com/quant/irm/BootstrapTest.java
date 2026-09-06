package com.quant.irm;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import org.junit.Test;

/** Bootstrap round-trip repricing, gap handling and quote validation. */
public class BootstrapTest {

    private static final Path DATA_DIR = Paths.get("..", "data");

    @Test
    public void annualScheduleHandlesIntegerStubAndSubYearMaturities() {
        assertArrayEquals(new double[] {1.0, 2.0}, Bootstrap.annualSchedule(2.0), 0.0);
        assertArrayEquals(new double[] {0.5, 1.5, 2.5}, Bootstrap.annualSchedule(2.5), 1e-15);
        assertArrayEquals(new double[] {0.25}, Bootstrap.annualSchedule(0.25), 0.0);
        assertThrows(IllegalArgumentException.class, () -> Bootstrap.annualSchedule(0.0));
        assertThrows(IllegalArgumentException.class, () -> Bootstrap.annualSchedule(-1.0));
    }

    @Test
    public void eurCurveRepricesEveryInputInstrument() {
        List<Instrument> quotes =
                Quotes.loadCurveQuotes(DATA_DIR.resolve("curve_quotes.csv"), "EUR");
        DiscountCurve curve = Bootstrap.bootstrap(quotes);
        for (Instrument ins : quotes) {
            assertTrue("EUR residual at t=" + ins.pillar(),
                    Math.abs(ins.residual(curve)) < 1e-9);
        }
        // Negative short end: DF(1y) must exceed 1.
        assertTrue("EUR DF(1) should be > 1", curve.df(1.0) > 1.0);
    }

    @Test
    public void usdCurveRepricesEveryInputInstrument() {
        List<Instrument> quotes =
                Quotes.loadCurveQuotes(DATA_DIR.resolve("curve_quotes.csv"), "USD");
        DiscountCurve curve = Bootstrap.bootstrap(quotes);
        for (Instrument ins : quotes) {
            assertTrue("USD residual at t=" + ins.pillar(),
                    Math.abs(ins.residual(curve)) < 1e-9);
        }
        // Bootstrapped par rates must reproduce the quoted swap rates.
        for (Instrument ins : quotes) {
            if (ins instanceof Swap swap) {
                double[] sched = Bootstrap.annualSchedule(swap.maturity());
                double[] times = new double[sched.length + 1];
                System.arraycopy(sched, 0, times, 1, sched.length);
                assertEquals("par rate at " + swap.maturity(),
                        swap.rate(), DiscountCurve.parSwapRate(curve, times), 1e-10);
            }
        }
    }

    @Test
    public void oisCurveRepricesEveryInputInstrument() {
        List<OisSwap> quotes = Quotes.loadOisQuotes(DATA_DIR.resolve("ois_quotes.csv"));
        DiscountCurve curve = Bootstrap.bootstrap(quotes);
        for (OisSwap ins : quotes) {
            assertTrue("OIS residual at t=" + ins.pillar(),
                    Math.abs(ins.residual(curve)) < 1e-9);
        }
    }

    @Test
    public void depositOnlyShortEndAndSwapOnlyLongEndWithGaps() {
        // A 7y swap with no 6y pillar: intermediate cashflows interpolate.
        List<Instrument> mixed = List.of(
                new Deposit(0.5, 0.02),
                new Deposit(1.0, 0.021),
                new Swap(3.0, 0.022),
                new Swap(7.0, 0.024));
        DiscountCurve curve = Bootstrap.bootstrap(mixed);
        for (Instrument ins : mixed) {
            assertTrue(Math.abs(ins.residual(curve)) < 1e-9);
        }
        assertEquals(4, curve.size());
    }

    @Test
    public void fraBootstrapReproducesForwardRate() {
        List<Instrument> ins = List.of(
                new Deposit(0.5, 0.03),
                new Fra(0.5, 1.0, 0.032),
                new Swap(2.0, 0.031));
        DiscountCurve curve = Bootstrap.bootstrap(ins);
        assertEquals(0.032, curve.fwdRate(0.5, 1.0), 1e-10);
        for (Instrument i : ins) {
            assertTrue(Math.abs(i.residual(curve)) < 1e-9);
        }
    }

    @Test
    public void orderingViolationsAreRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> Bootstrap.bootstrap(List.of()));
        assertThrows(IllegalArgumentException.class, // duplicate pillar
                () -> Bootstrap.bootstrap(List.of(
                        new Deposit(1.0, 0.02), new Deposit(1.0, 0.021))));
        assertThrows(IllegalArgumentException.class, // out of order
                () -> Bootstrap.bootstrap(List.of(
                        new Deposit(2.0, 0.02), new Deposit(1.0, 0.021))));
    }

    @Test
    public void crossedQuotesImplyingNegativeDfAreRejected() {
        // FRA over [1, 2] at a rate implying DF(2) far beyond any admissible
        // value given DF(1): 1 + R*tau close to the excluded region.
        assertThrows(IllegalArgumentException.class,
                () -> Bootstrap.bootstrap(List.of(
                        new Deposit(1.0, 0.02),
                        new Fra(1.0, 2.0, -0.999999999))));
        // A 300% 2y par rate cannot be repriced by any positive DF(2).
        IllegalArgumentException exc = assertThrows(IllegalArgumentException.class,
                () -> Bootstrap.bootstrap(List.of(new Deposit(1.0, 0.05), new Swap(2.0, 3.0))));
        assertTrue(exc.getMessage().contains("t=2.0"));
        assertTrue(exc.getMessage().contains("crossed"));
    }

    @Test
    public void nonConvergenceMessageNotCrossedQuotes() {
        // A solver failure that is not a bracketing failure must not be
        // relabelled as crossed quotes (MINOR-10).
        IllegalArgumentException exc = assertThrows(IllegalArgumentException.class,
                () -> Bootstrap.solvePillarDf(
                        x -> (x > 1.0) ? Double.POSITIVE_INFINITY : -1.0, 2.0, "test"));
        assertTrue(exc.getMessage().contains("solver failed"));
        assertTrue(exc.getMessage().contains("not finite"));
        assertTrue(!exc.getMessage().contains("crossed"));
        // Same through a duck-typed instrument whose residual is NaN inside the bracket.
        Instrument nanIns = new Instrument() {
            @Override public double pillar() {
                return 1.0;
            }

            @Override public double rate() {
                return 0.02;
            }

            @Override public double residual(DiscountCurve curve) {
                double df = curve.df(1.0);
                return (df > 0.5 && df < 1.5) ? Double.NaN : df - 0.97;
            }
        };
        IllegalArgumentException exc2 = assertThrows(IllegalArgumentException.class,
                () -> Bootstrap.bootstrap(List.of(nanIns)));
        assertTrue(exc2.getMessage().contains("solver failed"));
        assertTrue(!exc2.getMessage().contains("crossed"));
        IllegalArgumentException exc3 = assertThrows(IllegalArgumentException.class,
                () -> Bootstrap.solvePillarDf(x -> 1.0 + x, 2.0, "test"));
        assertTrue(exc3.getMessage().contains("crossed"));
        assertEquals(0.9, Bootstrap.solvePillarDf(x -> x - 0.9, 2.0, "test"), 1e-13);
    }

    @Test
    public void maturityOutOfRangeRejected() {
        // 1e12 used to saturate (int) Math.ceil and allocate 16 GB.
        for (double bad : new double[] {1e12, 1e9, Bootstrap.MAX_MATURITY * (1.0 + 1e-9), 1e-13,
                Bootstrap.MIN_MATURITY / 2.0, Double.POSITIVE_INFINITY}) {
            assertThrows(IllegalArgumentException.class, () -> new Swap(bad, 0.03));
            assertThrows(IllegalArgumentException.class, () -> new OisSwap(bad, 0.03));
            assertThrows(IllegalArgumentException.class, () -> new Deposit(bad, 0.03));
            assertThrows(IllegalArgumentException.class, () -> new Fra(0.0, bad, 0.03));
            assertThrows(IllegalArgumentException.class, () -> Bootstrap.annualSchedule(bad));
        }
        assertEquals(200, Bootstrap.annualSchedule(Bootstrap.MAX_MATURITY).length);
        assertArrayEquals(new double[] {Bootstrap.MIN_MATURITY},
                Bootstrap.annualSchedule(Bootstrap.MIN_MATURITY), 0.0);
        assertEquals(200, Bootstrap.annualSchedule(199.5).length);
    }

    @Test
    public void fraStripReprices() {
        List<Instrument> ins = List.of(new Deposit(0.5, 0.030), new Fra(0.5, 1.0, 0.032),
                new Fra(1.0, 1.5, 0.034), new Swap(3.0, 0.033));
        DiscountCurve curve = Bootstrap.bootstrap(ins);
        for (Instrument i : ins) {
            assertTrue(Math.abs(i.residual(curve)) < 1e-9);
        }
        assertEquals(0.032, curve.fwdRate(0.5, 1.0), 1e-12);
    }

    @Test
    public void oisSubYearMaturity() {
        DiscountCurve c = Bootstrap.bootstrap(List.of(new OisSwap(1.0 / 12.0, 0.03)));
        assertEquals(1.0 / (1.0 + 0.03 / 12.0), c.df(1.0 / 12.0), 1e-12);
    }

    private static DiscountCurve bumpedUsd(List<Instrument> quotes, double pillar) {
        List<Instrument> out = new java.util.ArrayList<>();
        for (Instrument q : quotes) {
            double bump = (q.pillar() == pillar) ? 1e-4 : 0.0;
            if (q instanceof Deposit d) {
                out.add(new Deposit(d.maturity(), d.rate() + bump));
            } else {
                Swap s = (Swap) q;
                out.add(new Swap(s.maturity(), s.rate() + bump));
            }
        }
        return Bootstrap.bootstrap(out);
    }

    @Test
    public void dv01LocalisesInQuoteSpace() {
        List<Instrument> quotes =
                Quotes.loadCurveQuotes(DATA_DIR.resolve("curve_quotes.csv"), "USD");
        double[] times = new double[11];
        for (int i = 0; i <= 10; i++) {
            times[i] = i;
        }
        assertEquals(0.0425, DiscountCurve.parSwapRate(Bootstrap.bootstrap(quotes), times), 1e-12);
        assertEquals(0.0425, DiscountCurve.parSwapRate(bumpedUsd(quotes, 5.0), times), 1e-12);
        assertEquals(1e-4, DiscountCurve.parSwapRate(bumpedUsd(quotes, 10.0), times) - 0.0425,
                1e-12);
    }

    @Test
    public void invalidInstrumentQuotesAreRejectedAtConstruction() {
        assertThrows(IllegalArgumentException.class, () -> new Deposit(-1.0, 0.02));
        assertThrows(IllegalArgumentException.class, () -> new Deposit(1.0, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new Deposit(2.0, -0.5)); // 1+RT<=0
        assertThrows(IllegalArgumentException.class, () -> new Fra(-0.5, 1.0, 0.02));
        assertThrows(IllegalArgumentException.class, () -> new Fra(1.0, 1.0, 0.02));
        assertThrows(IllegalArgumentException.class, () -> new Fra(0.5, 1.5, -1.5)); // 1+Rtau<=0
        assertThrows(IllegalArgumentException.class, () -> new Swap(0.0, 0.02));
        assertThrows(IllegalArgumentException.class,
                () -> new Swap(2.0, Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> new OisSwap(-2.0, 0.02));
    }

    @Test
    public void quoteLoadersRejectUnknownCurveId() {
        assertThrows(IllegalArgumentException.class,
                () -> Quotes.loadCurveQuotes(DATA_DIR.resolve("curve_quotes.csv"), "GBP"));
    }
}
