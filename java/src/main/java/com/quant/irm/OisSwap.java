package com.quant.irm;

/**
 * OIS par swap: annual fixed leg vs daily-compounded overnight leg.
 *
 * <p>Daily compounding of the overnight rate is approximated as continuous
 * compounding at the curve's short rate, under which the compounded floating
 * coupon over {@code [t_i-1, t_i]} pays {@code DF(t_i-1)/DF(t_i) - 1} and
 * the floating leg PV telescopes to {@code 1 - DF(T)} — identical in form to
 * {@link Swap}. The daily-vs-continuous compounding gap is
 * {@code O(rate^2 / 365)} and is a documented approximation.</p>
 *
 * @param maturity swap maturity in years, within {@code [1e-6, 200]}
 * @param rate     par fixed rate (decimal)
 */
public record OisSwap(double maturity, double rate) implements Instrument {

    /** Validates the rate and maturity (via the annual schedule builder). */
    public OisSwap {
        if (!Double.isFinite(rate)) {
            throw new IllegalArgumentException("quote rate must be finite, got " + rate);
        }
        Bootstrap.annualSchedule(maturity);
    }

    @Override
    public double pillar() {
        return maturity;
    }

    /** R * annuity - (1 - DF(T)); floating leg telescopes as for {@link Swap}. */
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
