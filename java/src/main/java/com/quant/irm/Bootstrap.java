package com.quant.irm;

import java.util.List;

/**
 * Sequential single-curve bootstrap from deposits, FRAs and par swaps.
 *
 * <p>Instruments must be supplied sorted by pillar (maturity) time, strictly
 * increasing. For each instrument in order, the discount factor at its
 * pillar is the single unknown {@code x}; all earlier dates interpolate on
 * the already-solved pillars (plus the trial pillar) — this is how missing
 * pillar gaps are handled — so each pillar is found by a 1-D root find of
 * the instrument's residual (PV mispricing) with the native Brent solver on
 * the bracket {@code DF in [1e-10, 100]}, {@code xtol = 1e-14}. A bracket
 * failure means no positive discount factor can reprice the quote — i.e.
 * crossed/arbitrageable quotes (the "negative implied DF" case) — and
 * raises {@link IllegalArgumentException}.</p>
 */
public final class Bootstrap {

    private static final double DF_LO = 1e-10;
    private static final double DF_HI = 100.0;

    private Bootstrap() {
    }

    /**
     * Annual fixed-leg payment times ending exactly at {@code maturity}.
     *
     * <p>{@code n = ceil(maturity)} payments at
     * {@code maturity - (n-1), ..., maturity}; e.g. 2.0 -&gt; (1.0, 2.0);
     * 2.5 -&gt; (0.5, 1.5, 2.5); 0.25 -&gt; (0.25). A short first stub
     * (&lt; 1y) is created for non-integer maturities.</p>
     *
     * @throws IllegalArgumentException if maturity is not finite and &gt; 0
     */
    public static double[] annualSchedule(double maturity) {
        if (!Double.isFinite(maturity) || maturity <= 0.0) {
            throw new IllegalArgumentException("maturity must be finite and > 0, got " + maturity);
        }
        int n = (int) Math.ceil(maturity - 1e-12);
        double[] out = new double[n];
        for (int k = 0; k < n; k++) {
            out[k] = maturity - (n - 1 - k);
        }
        return out;
    }

    /**
     * Sequentially bootstraps a discount curve from sorted instruments.
     *
     * <p>Instruments must be ordered with strictly increasing pillar times
     * (duplicates rejected). Each pillar DF is solved with Brent's method on
     * {@code [1e-10, 100]} (DF &gt; 1 allowed: negative rates).</p>
     *
     * @throws IllegalArgumentException on an empty list, ordering violations,
     *     or when no positive DF can reprice a quote (crossed/arbitrageable
     *     inputs)
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
            double df;
            try {
                df = RootFind.brentq(x -> {
                    trialDfs[idx] = x;
                    return ins.residual(new DiscountCurve(trialTimes, trialDfs));
                }, DF_LO, DF_HI, 1e-14);
            } catch (IllegalArgumentException exc) {
                throw new IllegalArgumentException(
                        "bootstrap failed at pillar t=" + pillar + " ("
                        + ins.getClass().getSimpleName() + ", rate=" + ins.rate()
                        + "): no admissible positive discount factor — "
                        + "crossed/arbitrageable quotes? [" + exc.getMessage() + "]", exc);
            }
            times[i] = pillar;
            dfs[i] = df;
        }
        return new DiscountCurve(times, dfs);
    }
}
