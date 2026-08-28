package com.quant.irm;

import java.util.Arrays;
import java.util.function.ToDoubleFunction;

/**
 * Native Nelder-Mead simplex minimiser (P02-style API).
 *
 * <p>Used by the Vasicek least-squares calibration. Standard coefficients:
 * reflection 1, expansion 2, contraction 0.5, shrink 0.5. Non-convergence
 * is <em>reported</em> via {@code converged=false} in the result, never
 * thrown — callers decide how to react (per the shared conventions: report,
 * don't crash).</p>
 */
public final class NelderMead {

    private NelderMead() {
    }

    /**
     * Outcome of a Nelder-Mead minimisation.
     *
     * @param x          best point found
     * @param fx         objective value at {@code x}
     * @param iterations iterations performed
     * @param converged  whether both spread tolerances were met
     */
    public record Result(double[] x, double fx, int iterations, boolean converged) {

        /** Defensive copy of the best point. */
        @Override
        public double[] x() {
            return x.clone();
        }

        @Override
        public String toString() {
            return "Result(x=" + Arrays.toString(x) + ", fx=" + fx
                    + ", iterations=" + iterations + ", converged=" + converged + ")";
        }
    }

    /** {@link #minimize(ToDoubleFunction, double[], double, double, double, int)}
     *  with defaults {@code initialStep = 0.1, xtol = 1e-10, ftol = 1e-14, maxiter = 5000}. */
    public static Result minimize(ToDoubleFunction<double[]> f, double[] x0) {
        return minimize(f, x0, 0.1, 1e-10, 1e-14, 5000);
    }

    /**
     * Minimises {@code f} starting from {@code x0}.
     *
     * <p>The initial simplex is {@code x0} plus {@code initialStep} added to
     * each coordinate in turn. Converged when both the simplex diameter is
     * below {@code xtol} and the function-value spread below {@code ftol}.</p>
     *
     * @throws IllegalArgumentException on an empty or non-finite start point
     */
    public static Result minimize(
            ToDoubleFunction<double[]> f, double[] x0,
            double initialStep, double xtol, double ftol, int maxiter) {
        int n = (x0 == null) ? 0 : x0.length;
        if (n == 0) {
            throw new IllegalArgumentException("x0 must be non-empty");
        }
        for (double v : x0) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("x0 must be finite");
            }
        }

        double[][] pts = new double[n + 1][];
        double[] vals = new double[n + 1];
        pts[0] = x0.clone();
        for (int i = 0; i < n; i++) {
            double[] p = x0.clone();
            p[i] += initialStep;
            pts[i + 1] = p;
        }
        for (int j = 0; j <= n; j++) {
            vals[j] = f.applyAsDouble(pts[j]);
        }
        order(pts, vals);

        int it = 0;
        boolean converged = false;
        while (it < maxiter) {
            it++;
            double spreadF = Math.abs(vals[n] - vals[0]);
            double spreadX = 0.0;
            for (int j = 1; j <= n; j++) {
                for (int i = 0; i < n; i++) {
                    spreadX = Math.max(spreadX, Math.abs(pts[j][i] - pts[0][i]));
                }
            }
            if (spreadF <= ftol && spreadX <= xtol) {
                converged = true;
                break;
            }

            double[] centroid = new double[n];
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    centroid[i] += pts[j][i];
                }
            }
            for (int i = 0; i < n; i++) {
                centroid[i] /= n;
            }
            double[] worst = pts[n];
            double[] refl = new double[n];
            for (int i = 0; i < n; i++) {
                refl[i] = centroid[i] + (centroid[i] - worst[i]);
            }
            double fRefl = f.applyAsDouble(refl);

            if (fRefl < vals[0]) {
                // Try expanding further along the reflection direction.
                double[] exp = new double[n];
                for (int i = 0; i < n; i++) {
                    exp[i] = centroid[i] + 2.0 * (centroid[i] - worst[i]);
                }
                double fExp = f.applyAsDouble(exp);
                if (fExp < fRefl) {
                    pts[n] = exp;
                    vals[n] = fExp;
                } else {
                    pts[n] = refl;
                    vals[n] = fRefl;
                }
            } else if (fRefl < vals[n - 1]) {
                pts[n] = refl;
                vals[n] = fRefl;
            } else {
                // Contract toward the centroid (outside if reflection helped a bit).
                double[] contr = new double[n];
                if (fRefl < vals[n]) {
                    for (int i = 0; i < n; i++) {
                        contr[i] = centroid[i] + 0.5 * (refl[i] - centroid[i]);
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        contr[i] = centroid[i] + 0.5 * (worst[i] - centroid[i]);
                    }
                }
                double fContr = f.applyAsDouble(contr);
                if (fContr < Math.min(fRefl, vals[n])) {
                    pts[n] = contr;
                    vals[n] = fContr;
                } else {
                    // Shrink the whole simplex toward the best vertex.
                    for (int j = 1; j <= n; j++) {
                        for (int i = 0; i < n; i++) {
                            pts[j][i] = pts[0][i] + 0.5 * (pts[j][i] - pts[0][i]);
                        }
                        vals[j] = f.applyAsDouble(pts[j]);
                    }
                }
            }
            order(pts, vals);
        }

        return new Result(pts[0].clone(), vals[0], it, converged);
    }

    /** Sorts simplex vertices in place by ascending objective value (stable). */
    private static void order(double[][] pts, double[] vals) {
        int m = vals.length;
        Integer[] idx = new Integer[m];
        for (int k = 0; k < m; k++) {
            idx[k] = k;
        }
        Arrays.sort(idx, (p, q) -> Double.compare(vals[p], vals[q]));
        double[][] p2 = new double[m][];
        double[] v2 = new double[m];
        for (int k = 0; k < m; k++) {
            p2[k] = pts[idx[k]];
            v2[k] = vals[idx[k]];
        }
        System.arraycopy(p2, 0, pts, 0, m);
        System.arraycopy(v2, 0, vals, 0, m);
    }
}
