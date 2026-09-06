package com.quant.irm;

/**
 * Immutable discount curve with log-linear interpolation in ln(DF).
 *
 * <p>Conventions (see API_SPEC.md):</p>
 * <ul>
 *   <li>Day count is ACT/365F everywhere; all times are year fractions
 *       {@code days/365} (a documented simplification versus the ACT/360
 *       money-market convention used for real deposits/FRAs).</li>
 *   <li>Rates are decimals (0.05 = 5%).</li>
 *   <li>Interpolation is <em>linear in ln(DF)</em> between pillars, with an
 *       implicit node {@code (t=0, DF=1)}. Linear-in-ln(DF) is exactly the
 *       piecewise-constant instantaneous-forward interpolation: on a segment
 *       {@code [t_i, t_i+1]} the instantaneous forward is the constant
 *       {@code f_i = -(ln DF_i+1 - ln DF_i) / (t_i+1 - t_i)}.</li>
 *   <li>Extrapolation beyond the last pillar is flat-forward: the last
 *       segment's constant forward is carried on indefinitely.</li>
 *   <li>Negative rates are fully supported: DF &gt; 1 is legal (EUR/JPY
 *       style); the only requirement is DF &gt; 0.</li>
 * </ul>
 */
public final class DiscountCurve {

    private final double[] times;  // pillar times, strictly increasing, > 0
    private final double[] dfs;    // pillar discount factors, > 0
    private final double[] t;      // nodes with the implicit 0 prepended
    private final double[] lnp;    // ln DF at the nodes (lnp[0] = 0)

    /**
     * Builds a curve from pillar times and discount factors.
     *
     * @param times strictly increasing pillar times in years (ACT/365F), all &gt; 0
     * @param dfs   discount factors at the pillars, all finite and &gt; 0
     *              (values &gt; 1 allowed: negative rates)
     * @throws IllegalArgumentException on empty input, length mismatch,
     *     non-finite or non-increasing times, or non-positive/non-finite DFs
     */
    public DiscountCurve(double[] times, double[] dfs) {
        if (times == null || dfs == null || times.length == 0) {
            throw new IllegalArgumentException("curve needs at least one pillar");
        }
        if (times.length != dfs.length) {
            throw new IllegalArgumentException(
                    "times and dfs length mismatch: " + times.length + " vs " + dfs.length);
        }
        double prev = 0.0;
        for (double tt : times) {
            if (!Double.isFinite(tt)) {
                throw new IllegalArgumentException("non-finite pillar time " + tt);
            }
            if (tt <= prev) {
                throw new IllegalArgumentException(
                        "pillar times must be strictly increasing and positive; "
                        + "offending time " + tt + " after " + prev);
            }
            prev = tt;
        }
        for (int i = 0; i < times.length; i++) {
            if (!Double.isFinite(dfs[i]) || dfs[i] <= 0.0) {
                throw new IllegalArgumentException(
                        "discount factor at t=" + times[i] + " must be finite and > 0, got " + dfs[i]);
            }
        }
        this.times = times.clone();
        this.dfs = dfs.clone();
        // Internal nodes include the implicit (0, ln 1 = 0).
        this.t = new double[times.length + 1];
        this.lnp = new double[times.length + 1];
        for (int i = 0; i < times.length; i++) {
            this.t[i + 1] = this.times[i];
            this.lnp[i + 1] = Math.log(this.dfs[i]);
        }
    }

    /** Number of pillars (excluding the implicit zero node). */
    public int size() {
        return times.length;
    }

    /** Copy of the pillar times. */
    public double[] times() {
        return times.clone();
    }

    /** Copy of the pillar discount factors. */
    public double[] dfs() {
        return dfs.clone();
    }

    // ------------------------------------------------------------------

    /**
     * Index i of the segment [t_i, t_i+1] used for time {@code x}.
     *
     * <p>Times at or beyond the last pillar map to the last segment
     * (flat-forward extrapolation); a time exactly at an interior node maps
     * to the segment starting at that node (right-hand segment).</p>
     */
    private int segment(double x) {
        // upper bound: first node index with t[idx] > x
        int lo = 0;
        int hi = t.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (t[mid] <= x) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        int i = lo - 1;
        int nSeg = t.length - 1;
        if (i < 0) {
            i = 0;
        }
        if (i > nSeg - 1) {
            i = nSeg - 1;
        }
        return i;
    }

    /** d ln(DF)/dt on segment i (equals minus the constant forward). */
    private double slope(int i) {
        return (lnp[i + 1] - lnp[i]) / (t[i + 1] - t[i]);
    }

    private double lnDf(double x) {
        if (!Double.isFinite(x) || x < 0.0) {
            throw new IllegalArgumentException("time must be finite and >= 0, got " + x);
        }
        if (x == 0.0) {
            return 0.0;
        }
        int i = segment(x);
        return lnp[i] + slope(i) * (x - t[i]);
    }

    // ------------------------------------------------------------------

    /**
     * Discount factor DF(t); DF(0) = 1, flat-forward beyond the last pillar.
     * A query exactly at a pillar returns the pillar DF exactly.
     */
    public double df(double time) {
        return Math.exp(lnDf(time));
    }

    /**
     * Continuously compounded zero rate {@code z(t) = -ln DF(t) / t}.
     *
     * <p>At {@code t = 0} the limit is returned, i.e. the instantaneous
     * forward of the first segment.</p>
     */
    public double zeroRate(double time) {
        if (!Double.isFinite(time) || time < 0.0) {
            throw new IllegalArgumentException("time must be finite and >= 0, got " + time);
        }
        if (time == 0.0) {
            return -slope(0);
        }
        return -lnDf(time) / time;
    }

    /**
     * Instantaneous forward {@code f(t) = -d ln DF / dt}.
     *
     * <p>Piecewise constant by construction. At a pillar the <em>right</em>
     * segment's forward is returned; at/after the last pillar the last
     * segment's forward (the flat extrapolation level).</p>
     */
    public double instForward(double time) {
        if (!Double.isFinite(time) || time < 0.0) {
            throw new IllegalArgumentException("time must be finite and >= 0, got " + time);
        }
        return -slope(segment(time));
    }

    /**
     * Simply compounded forward rate over {@code [t1, t2]}:
     * {@code F(t1, t2) = (DF(t1)/DF(t2) - 1) / (t2 - t1)}.
     *
     * @throws IllegalArgumentException unless {@code 0 <= t1 < t2} and both
     *     finite, or if the result is not representable (DF(t2) underflowed
     *     to 0 or the ratio overflowed) — never returns infinity
     */
    public double fwdRate(double t1, double t2) {
        if (!(Double.isFinite(t1) && Double.isFinite(t2))) {
            throw new IllegalArgumentException("forward-rate times must be finite");
        }
        if (t1 < 0.0 || t2 <= t1) {
            throw new IllegalArgumentException(
                    "need 0 <= t1 < t2 for a forward rate, got t1=" + t1 + ", t2=" + t2);
        }
        double df1 = df(t1);
        double df2 = df(t2);
        if (df2 == 0.0) {
            throw new IllegalArgumentException("forward rate over [" + t1 + ", " + t2
                    + "] not representable: DF(t2) underflowed to 0");
        }
        double fwd = (df1 / df2 - 1.0) / (t2 - t1);
        if (!Double.isFinite(fwd)) {
            throw new IllegalArgumentException(
                    "forward rate over [" + t1 + ", " + t2 + "] not representable: " + fwd);
        }
        return fwd;
    }

    /**
     * Forward-starting par swap rate off a single curve.
     *
     * <p>{@code times = [t0, t1, ..., tn]} gives the swap start
     * {@code t0 >= 0} and the fixed payment times. With accruals
     * {@code tau_i = t_i - t_i-1} and the single-curve identity (floating
     * leg PV = DF(t0) - DF(tn)):
     * {@code par = (DF(t0) - DF(tn)) / sum_i tau_i DF(t_i)}.</p>
     *
     * @throws IllegalArgumentException on fewer than two times, a negative
     *     start, or a non-increasing schedule
     */
    public static double parSwapRate(DiscountCurve curve, double[] times) {
        if (times == null || times.length < 2) {
            throw new IllegalArgumentException("parSwapRate needs a start and at least one payment");
        }
        double prev = times[0];
        if (prev < 0.0) {
            throw new IllegalArgumentException("swap start must be >= 0, got " + prev);
        }
        double annuity = 0.0;
        for (int i = 1; i < times.length; i++) {
            double ti = times[i];
            if (ti <= prev) {
                throw new IllegalArgumentException("swap schedule times must be strictly increasing");
            }
            annuity += (ti - prev) * curve.df(ti);
            prev = ti;
        }
        return (curve.df(times[0]) - curve.df(times[times.length - 1])) / annuity;
    }

    @Override
    public String toString() {
        return "DiscountCurve(pillars=" + times.length + ", lastT=" + times[times.length - 1] + ")";
    }
}
