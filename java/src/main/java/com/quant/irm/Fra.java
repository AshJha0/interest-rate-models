package com.quant.irm;

/**
 * Forward rate agreement fixing the simple forward over {@code [start, end]}.
 *
 * @param start accrual start in years, &gt;= 0
 * @param end   accrual end in years, &gt; start (the pillar), within {@code [1e-6, 200]}
 * @param rate  simple forward rate (decimal); must satisfy {@code 1 + R*tau > 0}
 */
public record Fra(double start, double end, double rate) implements Instrument {

    /** Validates times and that the quote implies a positive forward DF. */
    public Fra {
        if (!Double.isFinite(rate)) {
            throw new IllegalArgumentException("quote rate must be finite, got " + rate);
        }
        if (!(Double.isFinite(start) && Double.isFinite(end))) {
            throw new IllegalArgumentException("FRA times must be finite");
        }
        Bootstrap.checkMaturity("FRA end", end);
        if (start < 0.0 || end <= start) {
            throw new IllegalArgumentException(
                    "FRA needs 0 <= start < end, got start=" + start + ", end=" + end);
        }
        if (1.0 + rate * (end - start) <= 0.0) {
            throw new IllegalArgumentException("FRA quote implies non-positive discount factor");
        }
    }

    @Override
    public double pillar() {
        return end;
    }

    /** DF-linear mispricing: DF(T1) - (1 + R tau) DF(T2). */
    @Override
    public double residual(DiscountCurve curve) {
        double tau = end - start;
        return curve.df(start) - (1.0 + rate * tau) * curve.df(end);
    }
}
