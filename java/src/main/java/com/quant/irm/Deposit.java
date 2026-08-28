package com.quant.irm;

/**
 * Cash deposit paying simple interest {@code rate} at {@code maturity}.
 *
 * @param maturity deposit maturity in years, &gt; 0
 * @param rate     simple annual rate (decimal); must satisfy {@code 1 + R*T > 0}
 */
public record Deposit(double maturity, double rate) implements Instrument {

    /** Validates the quote; {@code 1 + R*T <= 0} would imply DF &lt;= 0. */
    public Deposit {
        if (!Double.isFinite(rate)) {
            throw new IllegalArgumentException("quote rate must be finite, got " + rate);
        }
        if (!Double.isFinite(maturity) || maturity <= 0.0) {
            throw new IllegalArgumentException("deposit maturity must be > 0, got " + maturity);
        }
        if (1.0 + rate * maturity <= 0.0) {
            throw new IllegalArgumentException(
                    "deposit quote implies non-positive discount factor (1 + R*T = "
                    + (1.0 + rate * maturity) + ")");
        }
    }

    @Override
    public double pillar() {
        return maturity;
    }

    /** Mispricing of investing 1 at the simple rate: DF(T)(1+RT) - 1. */
    @Override
    public double residual(DiscountCurve curve) {
        return curve.df(maturity) * (1.0 + rate * maturity) - 1.0;
    }
}
