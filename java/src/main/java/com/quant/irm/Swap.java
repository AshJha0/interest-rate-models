package com.quant.irm;

/**
 * Par interest-rate swap: annual fixed leg vs single-curve floating leg.
 *
 * <p>The floating leg is projected off the same curve it is discounted on,
 * so its PV telescopes to {@code 1 - DF(T)} (single-curve framework; a
 * documented simplification of the multi-curve/OIS-discounting reality).</p>
 *
 * @param maturity swap maturity in years, &gt; 0
 * @param rate     par fixed rate (decimal)
 */
public record Swap(double maturity, double rate) implements Instrument {

    /** Validates the rate and maturity (via the annual schedule builder). */
    public Swap {
        if (!Double.isFinite(rate)) {
            throw new IllegalArgumentException("quote rate must be finite, got " + rate);
        }
        Bootstrap.annualSchedule(maturity); // validates maturity
    }

    @Override
    public double pillar() {
        return maturity;
    }

    /** R * annuity - (1 - DF(T)), the fixed-minus-floating PV. */
    @Override
    public double residual(DiscountCurve curve) {
        double annuity = 0.0;
        double prev = 0.0;
        for (double ti : Bootstrap.annualSchedule(maturity)) {
            annuity += (ti - prev) * curve.df(ti);
            prev = ti;
        }
        return rate * annuity - (1.0 - curve.df(maturity));
    }
}
