package com.quant.irm;

import java.util.function.DoubleUnaryOperator;

/**
 * Native 1-D root finders: Brent's method and plain bisection.
 *
 * <p>Implemented from scratch (no external numerical library) per project
 * rules. Brent's method combines bisection, secant and inverse quadratic
 * interpolation (the classic Numerical-Recipes {@code zbrent} variant) and
 * is guaranteed to converge for a continuous function with a sign change on
 * the bracket, with superlinear speed on well-behaved functions.</p>
 */
public final class RootFind {

    /** Machine-epsilon-scale relative tolerance (2 * eps, as in the classic zbrent). */
    public static final double DEFAULT_RTOL = 4.44e-16;

    private RootFind() {
    }

    /**
     * Rejects NaN/inf function values: a silent NaN would otherwise be
     * treated as "same sign" and either misreported as a bracketing failure
     * or, worse, returned as a converged root.
     */
    private static double finiteValue(String who, double x, double fx) {
        if (!Double.isFinite(fx)) {
            throw new IllegalArgumentException(
                    who + ": function value not finite at x=" + x + ": " + fx);
        }
        return fx;
    }

    /** {@link #brentq(DoubleUnaryOperator, double, double, double, double, int)}
     *  with defaults {@code xtol = 1e-14, rtol = 4.44e-16, maxiter = 100}. */
    public static double brentq(DoubleUnaryOperator f, double a, double b) {
        return brentq(f, a, b, 1e-14, DEFAULT_RTOL, 100);
    }

    /** {@code brentq} with the given {@code xtol} and default rtol/maxiter. */
    public static double brentq(DoubleUnaryOperator f, double a, double b, double xtol) {
        return brentq(f, a, b, xtol, DEFAULT_RTOL, 100);
    }

    /**
     * Finds a root of {@code f} in {@code [a, b]} using Brent's method.
     *
     * <p>{@code f(a)} and {@code f(b)} must have opposite signs (or one
     * endpoint be an exact root). The per-iteration convergence test on the
     * abscissa is {@code |c - b|/2 <= 2*rtol*|b| + xtol/2} (Numerical-Recipes
     * zbrent convention).</p>
     *
     * @throws IllegalArgumentException on a non-finite bracket, non-positive
     *     xtol, an unbracketed root, a non-finite function value anywhere, or
     *     non-convergence after maxiter (an unconverged or NaN root is never
     *     returned)
     */
    public static double brentq(
            DoubleUnaryOperator f, double a, double b, double xtol, double rtol, int maxiter) {
        if (!(Double.isFinite(a) && Double.isFinite(b))) {
            throw new IllegalArgumentException("brentq: bracket endpoints must be finite");
        }
        if (xtol <= 0.0) {
            throw new IllegalArgumentException("brentq: xtol must be positive");
        }
        double fa = finiteValue("brentq", a, f.applyAsDouble(a));
        double fb = finiteValue("brentq", b, f.applyAsDouble(b));
        if (fa == 0.0) {
            return a;
        }
        if (fb == 0.0) {
            return b;
        }
        if ((fa > 0.0) == (fb > 0.0)) {
            throw new IllegalArgumentException(
                    "brentq: root not bracketed: f(" + a + ")=" + fa + ", f(" + b + ")=" + fb);
        }
        double c = a;
        double fc = fa;
        double d = b - a;
        double e = b - a;
        for (int it = 0; it < maxiter; it++) {
            if ((fb > 0.0) == (fc > 0.0)) {
                // Rename so that the root lies between b and c.
                c = a;
                fc = fa;
                d = b - a;
                e = d;
            }
            if (Math.abs(fc) < Math.abs(fb)) {
                a = b;
                b = c;
                c = a;
                fa = fb;
                fb = fc;
                fc = fa;
            }
            double tol1 = 2.0 * rtol * Math.abs(b) + 0.5 * xtol;
            double xm = 0.5 * (c - b);
            if (Math.abs(xm) <= tol1 || fb == 0.0) {
                return b;
            }
            if (Math.abs(e) >= tol1 && Math.abs(fa) > Math.abs(fb)) {
                // Attempt inverse quadratic interpolation (secant if a == c).
                double s = fb / fa;
                double p;
                double q;
                if (a == c) {
                    p = 2.0 * xm * s;
                    q = 1.0 - s;
                } else {
                    q = fa / fc;
                    double r = fb / fc;
                    p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0);
                }
                if (p > 0.0) {
                    q = -q;
                }
                p = Math.abs(p);
                if (2.0 * p < Math.min(3.0 * xm * q - Math.abs(tol1 * q), Math.abs(e * q))) {
                    e = d; // accept interpolation
                    d = p / q;
                } else {
                    d = xm; // interpolation failed; bisect
                    e = d;
                }
            } else {
                d = xm; // bounds decreasing too slowly; bisect
                e = d;
            }
            a = b;
            fa = fb;
            if (Math.abs(d) > tol1) {
                b += d;
            } else {
                b += (xm > 0.0) ? tol1 : -tol1;
            }
            fb = finiteValue("brentq", b, f.applyAsDouble(b));
        }
        throw new IllegalArgumentException("brentq: no convergence after " + maxiter + " iterations");
    }

    /** {@link #bisect(DoubleUnaryOperator, double, double, double, int)}
     *  with defaults {@code xtol = 1e-12, maxiter = 200}. */
    public static double bisect(DoubleUnaryOperator f, double a, double b) {
        return bisect(f, a, b, 1e-12, 200);
    }

    /**
     * Plain bisection on {@code [a, b]}; robust fallback, linear convergence.
     * Same bracket contract as {@link #brentq}.
     *
     * @throws IllegalArgumentException on a non-finite bracket, non-positive
     *     xtol, an unbracketed root, a non-finite function value, or
     *     non-convergence after maxiter
     */
    public static double bisect(
            DoubleUnaryOperator f, double a, double b, double xtol, int maxiter) {
        if (!(Double.isFinite(a) && Double.isFinite(b))) {
            throw new IllegalArgumentException("bisect: bracket endpoints must be finite");
        }
        if (xtol <= 0.0) {
            throw new IllegalArgumentException("bisect: xtol must be positive");
        }
        double fa = finiteValue("bisect", a, f.applyAsDouble(a));
        double fb = finiteValue("bisect", b, f.applyAsDouble(b));
        if (fa == 0.0) {
            return a;
        }
        if (fb == 0.0) {
            return b;
        }
        if ((fa > 0.0) == (fb > 0.0)) {
            throw new IllegalArgumentException(
                    "bisect: root not bracketed: f(" + a + ")=" + fa + ", f(" + b + ")=" + fb);
        }
        double lo = (a < b) ? a : b;
        double hi = (a < b) ? b : a;
        double flo = (a < b) ? fa : fb;
        for (int it = 0; it < maxiter; it++) {
            double mid = 0.5 * (lo + hi);
            double fm = finiteValue("bisect", mid, f.applyAsDouble(mid));
            if (fm == 0.0 || 0.5 * (hi - lo) < xtol) {
                return mid;
            }
            if ((fm > 0.0) == (flo > 0.0)) {
                lo = mid;
                flo = fm;
            } else {
                hi = mid;
            }
        }
        throw new IllegalArgumentException("bisect: no convergence after " + maxiter + " iterations");
    }
}
