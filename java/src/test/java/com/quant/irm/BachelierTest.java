package com.quant.irm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Bachelier (normal) model: price, implied vol, parity, validation. */
public class BachelierTest {

    @Test
    public void atmClosedForm() {
        // ATM: price = A sigma sqrt(T) / sqrt(2 pi).
        double a = 0.98;
        double vol = 0.005;
        double t = 1.0;
        double p = Bachelier.price(-0.002, -0.002, t, vol, a, true);
        assertEquals(a * vol * Math.sqrt(t) / Math.sqrt(2.0 * Math.PI), p, 1e-17);
        assertEquals(p, Bachelier.price(-0.002, -0.002, t, vol, a, false), 1e-17);
        assertEquals(vol, Bachelier.impliedVol(p, -0.002, -0.002, t, a, true), 1e-14);
    }

    @Test
    public void impliedVolRoundTripGrid() {
        double t = 2.5;
        double annuity = 3.7;
        for (double forward : new double[] {-0.01, 0.0, 0.02, 0.05}) {
            for (double strike : new double[] {-0.02, 0.0, 0.03}) {
                for (double vol : new double[] {0.001, 0.0075, 0.02}) {
                    for (boolean payer : new boolean[] {true, false}) {
                        double p = Bachelier.price(forward, strike, t, vol, annuity, payer);
                        double iv = Bachelier.impliedVol(p, forward, strike, t, annuity, payer);
                        double d = Math.abs(forward - strike) / (vol * Math.sqrt(t));
                        if (d < 4.0) {
                            assertEquals(forward + " " + strike + " " + vol, vol, iv, 1e-12);
                        } else {
                            // Time value below rounding: only the price round-trips.
                            assertTrue(iv >= 0.0 && iv <= vol * 1.001);
                        }
                        assertEquals(p, Bachelier.price(forward, strike, t, iv, annuity, payer),
                                1e-15);
                    }
                }
            }
        }
    }

    @Test
    public void payerReceiverParityAndMonotonicity() {
        double f = 0.03;
        double k = 0.025;
        double t = 2.0;
        double a = 4.2;
        for (double vol : new double[] {0.0, 0.002, 0.0075, 0.03}) {
            double payer = Bachelier.price(f, k, t, vol, a, true);
            double receiver = Bachelier.price(f, k, t, vol, a, false);
            assertEquals(a * (f - k), payer - receiver, 1e-15);
        }
        double prev = -1.0;
        for (double vol : new double[] {0.001, 0.002, 0.005, 0.01, 0.02}) {
            double p = Bachelier.price(f, k, t, vol, a, true);
            assertTrue(p > prev);
            prev = p;
        }
    }

    @Test
    public void intrinsicAndBelowIntrinsic() {
        double f = 0.03;
        double k = 0.025;
        double t = 2.0;
        double a = 4.2;
        double intrinsic = a * (f - k);
        assertEquals(intrinsic, Bachelier.price(f, k, t, 0.0, a, true), 0.0);
        assertEquals(0.0, Bachelier.price(f, k, t, 0.0, a, false), 0.0);
        assertEquals(0.0, Bachelier.impliedVol(intrinsic, f, k, t, a, true), 0.0);
        assertEquals(0.0, Bachelier.impliedVol(0.0, f, k, t, a, false), 0.0);
        assertThrows(IllegalArgumentException.class,
                () -> Bachelier.impliedVol(intrinsic - 1e-6, f, k, t, a, true));
        assertThrows(IllegalArgumentException.class,
                () -> Bachelier.impliedVol(-1e-9, f, k, t, a, false));
    }

    @Test
    public void validation() {
        assertThrows(IllegalArgumentException.class,
                () -> Bachelier.price(Double.NaN, 0.02, 1.0, 0.01, 1.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> Bachelier.price(0.02, 0.02, 0.0, 0.01, 1.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> Bachelier.price(0.02, 0.02, 1.0, -0.01, 1.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> Bachelier.price(0.02, 0.02, 1.0, 0.01, 0.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> Bachelier.impliedVol(Double.NaN, 0.02, 0.02, 1.0, 1.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> Bachelier.impliedVol(0.01, 0.02, 0.02, Double.POSITIVE_INFINITY, 1.0, true));
        assertThrows(IllegalArgumentException.class,
                () -> Bachelier.impliedVol(1e300, 0.02, 0.02, 1.0, 1.0, true));
    }
}
