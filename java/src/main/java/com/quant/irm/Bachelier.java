package com.quant.irm;

import java.util.function.DoubleUnaryOperator;

/**
 * Bachelier (normal) model for caplets and swaptions: price and implied vol.
 *
 * <p>Rates desks quote cap/floor and swaption prices as <em>normal</em>
 * ("bp") volatilities: the volatility {@code sigma_N} such that the
 * Bachelier formula reproduces the price. For a payer option on a forward
 * {@code F} with strike {@code K}, expiry {@code T} and annuity {@code A}
 * ({@code A = tau DF(T_pay)} for a caplet, {@code sum_i tau_i DF(u_i)} for
 * a swaption):</p>
 * <pre>
 * price = A [ (F - K) N(d) + sigma_N sqrt(T) phi(d) ],  d = (F - K) / (sigma_N sqrt(T))
 * </pre>
 * <p>and the receiver is {@code A [ (K - F) N(-d) + sigma_N sqrt(T) phi(d) ]}.
 * With {@code sigma_N = 0} both collapse to discounted intrinsic. The price
 * is strictly increasing in {@code sigma_N}, so the implied volatility is
 * unique and found with the native Brent solver on an expanding bracket.
 * Negative forwards and strikes are first-class.</p>
 */
public final class Bachelier {

    /** Absolute price tolerance for deciding a quote is at intrinsic (vol 0). */
    public static final double INTRINSIC_TOL = 1e-15;
    /** Root tolerance on the implied volatility. */
    public static final double IMPLIED_VOL_XTOL = 1e-14;
    private static final double VOL_BRACKET_START = 0.01;
    private static final int VOL_BRACKET_DOUBLINGS = 40;

    private Bachelier() {
    }

    private static void checkCommon(double forward, double strike, double expiry, double annuity) {
        if (!(Double.isFinite(forward) && Double.isFinite(strike))) {
            throw new IllegalArgumentException("bachelier: forward and strike must be finite");
        }
        if (!Double.isFinite(expiry) || expiry <= 0.0) {
            throw new IllegalArgumentException(
                    "bachelier: expiry must be finite and > 0, got " + expiry);
        }
        if (!Double.isFinite(annuity) || annuity <= 0.0) {
            throw new IllegalArgumentException(
                    "bachelier: annuity must be finite and > 0, got " + annuity);
        }
    }

    /**
     * Normal-model option price (per unit notional, annuity included).
     *
     * <p>{@code vol} is the absolute (normal) volatility in rate units per
     * sqrt(year): 0.0075 means 75 bp. {@code vol = 0} returns discounted
     * intrinsic.</p>
     *
     * @throws IllegalArgumentException unless forward/strike finite,
     *     {@code expiry > 0}, {@code annuity > 0} and {@code vol >= 0} (finite)
     */
    public static double price(double forward, double strike, double expiry, double vol,
            double annuity, boolean payer) {
        checkCommon(forward, strike, expiry, annuity);
        if (!Double.isFinite(vol) || vol < 0.0) {
            throw new IllegalArgumentException("bachelier: vol must be finite and >= 0, got " + vol);
        }
        double moneyness = payer ? forward - strike : strike - forward;
        if (vol == 0.0) {
            return annuity * Math.max(moneyness, 0.0);
        }
        double sd = vol * Math.sqrt(expiry);
        double d = moneyness / sd;
        return annuity * (moneyness * MathUtils.normCdf(d) + sd * MathUtils.normPdf(d));
    }

    /**
     * Normal implied volatility of a payer/receiver price.
     *
     * <p>{@code price} must be finite and at least the discounted intrinsic
     * {@code annuity * max(+-(F - K), 0)}; a price within
     * {@link #INTRINSIC_TOL} of intrinsic returns 0. Otherwise Brent's method
     * solves {@code price(vol) = price} on {@code [0, hi]} with {@code hi}
     * found by doubling from 0.01 (at most 40 doublings). Root tolerance
     * 1e-14 on the vol.</p>
     *
     * @throws IllegalArgumentException on bad inputs, a price below intrinsic,
     *     or a price too large to imply a volatility
     */
    public static double impliedVol(double price, double forward, double strike, double expiry,
            double annuity, boolean payer) {
        checkCommon(forward, strike, expiry, annuity);
        if (!Double.isFinite(price)) {
            throw new IllegalArgumentException("bachelier: price must be finite, got " + price);
        }
        double intrinsic = price(forward, strike, expiry, 0.0, annuity, payer);
        if (price < intrinsic - INTRINSIC_TOL) {
            throw new IllegalArgumentException("bachelier: price " + price + " below intrinsic "
                    + intrinsic + " — no implied vol");
        }
        if (price <= intrinsic + INTRINSIC_TOL) {
            return 0.0;
        }
        DoubleUnaryOperator diff = v -> price(forward, strike, expiry, v, annuity, payer) - price;
        double hi = VOL_BRACKET_START;
        boolean bracketed = false;
        for (int i = 0; i < VOL_BRACKET_DOUBLINGS; i++) {
            if (diff.applyAsDouble(hi) > 0.0) {
                bracketed = true;
                break;
            }
            hi *= 2.0;
        }
        if (!bracketed) {
            throw new IllegalArgumentException(
                    "bachelier: price " + price + " too large to imply a volatility");
        }
        return RootFind.brentq(diff, 0.0, hi, IMPLIED_VOL_XTOL);
    }
}
