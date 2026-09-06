package com.quant.irm;

import java.util.List;
import java.util.function.DoubleUnaryOperator;

/**
 * Sequential single-curve bootstrap from deposits, FRAs and par swaps.
 *
 * <p>Instruments must be supplied sorted by pillar (maturity) time, strictly
 * increasing. For each instrument in order, the discount factor at its
 * pillar is the single unknown {@code x}; all earlier dates interpolate on
 * the already-solved pillars (plus the trial pillar) — this is how missing
 * pillar gaps are handled — so each pillar is found by a 1-D root find of
 * the instrument's residual (PV mispricing) with the native Brent solver on
 * the bracket {@code DF in [1e-10, 100]}, {@code xtol = 1e-14}. A
 * <em>bracketing</em> failure means no positive discount factor can reprice
 * the quote — i.e. crossed/arbitrageable quotes (the "negative implied DF"
 * case) — and raises {@link IllegalArgumentException} saying so; any other
 * solver failure (non-finite residual, iteration budget) is reported as a
 * solver failure at that pillar, never relabelled as crossed quotes.
 * Maturities are bounded to {@code [MIN_MATURITY, MAX_MATURITY] = [1e-6, 200]}
 * years at construction.</p>
 */
public final class Bootstrap {

    private static final double DF_LO = 1e-10;
    private static final double DF_HI = 100.0;

    /**
     * Smallest instrument maturity / FRA end accepted (years). Guarantees the
     * annual schedule has at least one payment.
     */
    public static final double MIN_MATURITY = 1e-6;
    /**
     * Largest instrument maturity accepted (years). Bounds the schedule length
     * (at most 200 payments) so a maturity typed in days cannot saturate the
     * int conversion and allocate gigabytes.
     */
    public static final double MAX_MATURITY = 200.0;

    /** Validates an instrument maturity against {@code [MIN_MATURITY, MAX_MATURITY]}. */
    static void checkMaturity(String what, double maturity) {
        if (!Double.isFinite(maturity) || maturity < MIN_MATURITY || maturity > MAX_MATURITY) {
            throw new IllegalArgumentException(what + " must be finite and within ["
                    + MIN_MATURITY + ", " + MAX_MATURITY + "] years, got " + maturity);
        }
    }

    private Bootstrap() {
    }

    /**
     * Annual fixed-leg payment times ending exactly at {@code maturity}.
     *
     * <p>{@code n = ceil(maturity - 1e-12)} payments at
     * {@code maturity - (n-1), ..., maturity}; e.g. 2.0 -&gt; (1.0, 2.0);
     * 2.5 -&gt; (0.5, 1.5, 2.5); 0.25 -&gt; (0.25). A short first stub
     * (&lt; 1y) is created for non-integer maturities. {@code maturity} must
     * lie in {@code [MIN_MATURITY, MAX_MATURITY]} ({@code [1e-6, 200]} years),
     * so {@code 1 <= n <= 200} always.</p>
     *
     * @throws IllegalArgumentException if maturity is out of range or non-finite
     */
    public static double[] annualSchedule(double maturity) {
        checkMaturity("maturity", maturity);
        int n = (int) Math.ceil(maturity - 1e-12); // <= 200: no saturation
        double[] out = new double[n];
        for (int k = 0; k < n; k++) {
            out[k] = maturity - (n - 1 - k);
        }
        return out;
    }

    /**
     * Solves {@code residual(DF) = 0} for one pillar on the bracket
     * {@code [1e-10, 100]} ({@code xtol = 1e-14}).
     *
     * <p>This is the per-pillar kernel of {@link #bootstrap}, exposed so the
     * error classification can be tested directly:</p>
     * <ul>
     *   <li>no sign change on the bracket: "no admissible positive discount
     *       factor — crossed/arbitrageable quotes";</li>
     *   <li>any other solver failure (non-finite residual, iteration budget):
     *       "solver failed at pillar" carrying the solver message.</li>
     * </ul>
     *
     * @throws IllegalArgumentException in both cases, naming the pillar
     */
    public static double solvePillarDf(DoubleUnaryOperator residual, double pillar, String label) {
        try {
            return RootFind.brentq(residual, DF_LO, DF_HI, 1e-14);
        } catch (IllegalArgumentException exc) {
            String msg = String.valueOf(exc.getMessage());
            if (msg.contains("not bracketed")) {
                throw new IllegalArgumentException("bootstrap failed at pillar t=" + pillar + " ("
                        + label + "): no admissible positive discount factor in [" + DF_LO + ", "
                        + DF_HI + "] — crossed/arbitrageable quotes? [" + msg + "]", exc);
            }
            throw new IllegalArgumentException(
                    "bootstrap: solver failed at pillar t=" + pillar + " (" + label + "): " + msg,
                    exc);
        }
    }

    /**
     * Sequentially bootstraps a discount curve from sorted instruments.
     *
     * <p>Instruments must be ordered with strictly increasing pillar times
     * (duplicates rejected). Each pillar DF is solved with Brent's method on
     * {@code [1e-10, 100]} (DF &gt; 1 allowed: negative rates) via
     * {@link #solvePillarDf}.</p>
     *
     * @throws IllegalArgumentException on an empty list, ordering violations,
     *     when no positive DF can reprice a quote (crossed/arbitrageable
     *     inputs, message says so) or when the solver fails for any other
     *     reason (message says "solver failed")
     */
    public static DiscountCurve bootstrap(List<? extends Instrument> instruments) {
        if (instruments == null || instruments.isEmpty()) {
            throw new IllegalArgumentException("bootstrap needs at least one instrument");
        }
        double prevPillar = 0.0;
        for (Instrument ins : instruments) {
            if (ins.pillar() <= prevPillar) {
                throw new IllegalArgumentException(
                        "instrument pillars must be strictly increasing "
                        + "(duplicate or out-of-order pillar at t=" + ins.pillar() + ")");
            }
            prevPillar = ins.pillar();
        }

        int n = instruments.size();
        double[] times = new double[n];
        double[] dfs = new double[n];
        for (int i = 0; i < n; i++) {
            Instrument ins = instruments.get(i);
            double pillar = ins.pillar();
            double[] trialTimes = new double[i + 1];
            double[] trialDfs = new double[i + 1];
            System.arraycopy(times, 0, trialTimes, 0, i);
            System.arraycopy(dfs, 0, trialDfs, 0, i);
            trialTimes[i] = pillar;
            final int idx = i;
            double df = solvePillarDf(x -> {
                trialDfs[idx] = x;
                return ins.residual(new DiscountCurve(trialTimes, trialDfs));
            }, pillar, ins.getClass().getSimpleName() + ", rate=" + ins.rate());
            times[i] = pillar;
            dfs[i] = df;
        }
        return new DiscountCurve(times, dfs);
    }
}
