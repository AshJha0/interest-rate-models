package com.quant.irm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Brent and bisection solvers plus the Nelder-Mead optimizer. */
public class RootFindTest {

    @Test
    public void brentFindsSimpleRoots() {
        assertEquals(Math.sqrt(2.0),
                RootFind.brentq(x -> x * x - 2.0, 0.0, 2.0), 1e-13);
        assertEquals(0.7390851332151607, // fixed point of cos
                RootFind.brentq(x -> Math.cos(x) - x, 0.0, 1.0), 1e-13);
    }

    @Test
    public void brentReturnsExactEndpointRoots() {
        assertEquals(1.0, RootFind.brentq(x -> x - 1.0, 1.0, 5.0), 0.0);
        assertEquals(1.0, RootFind.brentq(x -> x - 1.0, -3.0, 1.0), 0.0);
    }

    @Test
    public void brentRejectsUnbracketedRootAndBadInput() {
        assertThrows(IllegalArgumentException.class,
                () -> RootFind.brentq(x -> x * x + 1.0, -1.0, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> RootFind.brentq(x -> x, Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> RootFind.brentq(x -> x, -1.0, 1.0, -1e-14, RootFind.DEFAULT_RTOL, 100));
    }

    @Test
    public void bisectionAgreesWithBrent() {
        double rootB = RootFind.brentq(x -> Math.exp(x) - 3.0, 0.0, 2.0);
        double rootBi = RootFind.bisect(x -> Math.exp(x) - 3.0, 0.0, 2.0);
        assertEquals(Math.log(3.0), rootB, 1e-13);
        assertEquals(rootB, rootBi, 1e-11);
    }

    @Test
    public void bisectionRejectsUnbracketedRoot() {
        assertThrows(IllegalArgumentException.class,
                () -> RootFind.bisect(x -> x * x + 1.0, -1.0, 1.0));
    }

    @Test
    public void nelderMeadMinimisesRosenbrock() {
        NelderMead.Result res = NelderMead.minimize(p -> {
            double a = 1.0 - p[0];
            double b = p[1] - p[0] * p[0];
            return a * a + 100.0 * b * b;
        }, new double[] {-1.2, 1.0});
        assertEquals(1.0, res.x()[0], 1e-6);
        assertEquals(1.0, res.x()[1], 1e-6);
        assertEquals(0.0, res.fx(), 1e-12);
        assertTrue(res.converged());
    }

    @Test
    public void nelderMeadReportsNonConvergenceWithoutThrowing() {
        // One iteration cannot converge on a sloped objective.
        NelderMead.Result res = NelderMead.minimize(
                p -> p[0] * p[0] + p[1] * p[1] + 10.0,
                new double[] {5.0, 5.0}, 0.1, 1e-10, 1e-14, 1);
        org.junit.Assert.assertFalse(res.converged());
    }

    @Test
    public void nelderMeadRejectsBadStart() {
        assertThrows(IllegalArgumentException.class,
                () -> NelderMead.minimize(p -> 0.0, new double[] {}));
        assertThrows(IllegalArgumentException.class,
                () -> NelderMead.minimize(p -> 0.0, new double[] {Double.NaN}));
    }

    @Test
    public void normCdfMatchesReferenceValues() {
        assertEquals(0.5, MathUtils.normCdf(0.0), 1e-15);
        assertEquals(0.8413447460685429, MathUtils.normCdf(1.0), 1e-12);
        assertEquals(0.02275013194817921, MathUtils.normCdf(-2.0), 1e-12);
        // Symmetry N(x) + N(-x) = 1 on a grid.
        for (double x = -6.0; x <= 6.0; x += 0.25) {
            assertEquals(1.0, MathUtils.normCdf(x) + MathUtils.normCdf(-x), 1e-14);
        }
    }

    @Test
    public void brentNonConvergenceIsError() {
        assertThrows(IllegalArgumentException.class,
                () -> RootFind.brentq(x -> Math.tanh(1e3 * (x - 0.123456)), -10.0, 10.0, 1e-14,
                        RootFind.DEFAULT_RTOL, 1));
    }

    @Test
    public void bisectNonConvergenceIsError() {
        assertThrows(IllegalArgumentException.class,
                () -> RootFind.bisect(x -> x - 0.3, 0.0, 1.0, 1e-300, 5));
        assertThrows(IllegalArgumentException.class,
                () -> RootFind.bisect(x -> x, 0.0, Double.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> RootFind.bisect(x -> x, -1.0, 1.0, 0.0, 200));
        assertEquals(0.25, RootFind.bisect(x -> x - 0.25, 0.0, 1.0), 1e-11);
    }

    @Test
    public void nonFiniteFunctionValuesRejected() {
        // A NaN residual must never be returned as a "root" or misreported as
        // a bracketing failure.
        IllegalArgumentException exc = assertThrows(IllegalArgumentException.class,
                () -> RootFind.brentq(
                        x -> (x > 0.2 && x < 0.8) ? Double.NaN : (x < 0.5 ? 1.0 : -1.0),
                        0.0, 1.0));
        assertTrue(exc.getMessage().contains("not finite"));
        assertThrows(IllegalArgumentException.class,
                () -> RootFind.brentq(x -> Double.NaN, 0.0, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> RootFind.bisect(x -> x > 0.4 ? Double.POSITIVE_INFINITY : -1.0, 0.0, 1.0));
    }
}
