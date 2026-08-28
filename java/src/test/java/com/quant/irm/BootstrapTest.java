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
