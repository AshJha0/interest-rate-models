package com.quant.irm;

import java.util.List;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleFunction;

/**
 * Hull-White one-factor model {@code dr = (theta(t) - a r) dt + sigma dW}.
 *
 * <p>The model is fitted to an input {@link DiscountCurve} so that model ZCB
 * prices at {@code t = 0} reproduce the market discount factors
 * <em>exactly</em>. All formulas use market DFs {@code P^M(0, .)} directly:</p>
 *
 * <pre>
 * B(t,T) = (1 - e^(-a (T-t))) / a
 * A(t,T) = (P^M(0,T)/P^M(0,t))
 *          * exp[B(t,T) f^M(0,t) - (sigma^2/(4a)) (1 - e^(-2 a t)) B(t,T)^2]
 * P(t,T) = A(t,T) e^(-B(t,T) r(t))
 * </pre>
 *
 * <p>{@code f^M(0,t)} is the market instantaneous forward, computed by the
 * <em>exact</em> finite-difference scheme documented in API_SPEC.md (step
 * {@code h = 1e-5}, central for {@code t >= h}, one-sided at the short end)
 * so that every language port reproduces bit-comparable values.</p>
 *
 * <p>Options: ZBP/ZBC closed forms, caplets/floorlets as ZCB puts/calls,
 * caps as caplet strips, and Jamshidian's decomposition for European
 * swaptions. Monte Carlo simulates {@code r(t) = x(t) + alpha(t)} with exact
 * Gaussian OU moments (trinomial-tree-free),
 * {@code alpha(t) = f^M(0,t) + (sigma^2/(2a^2)) (1 - e^(-a t))^2}.</p>
 *
 * <p>Calibration: {@link #calibrate} fits {@code (a, sigma)} (or sigma with
 * {@code a} fixed) to caplet / payer-swaption prices by least squares from
 * several Nelder-Mead starts; {@link Bachelier} converts prices to normal
 * (bp) implied volatilities.</p>
 */
public final class HullWhite {

    /** Finite-difference step for the market instantaneous forward f(0,t). */
    public static final double FD_STEP = 1e-5;
    /** The coupon-bond residual {@code |g(r*)|} must be below this after the root find. */
    public static final double JAMSHIDIAN_RESIDUAL_TOL = 1e-10;
    /** Maximum bracket doublings when searching for r*. */
    private static final int JAMSHIDIAN_MAX_DOUBLINGS = 24;
    /** exp() argument above which a ZCB price is declared not representable. */
    private static final double MAX_EXP_ARG = 700.0;

    /** Calibration domain: a in (1e-6, 5], sigma in [0, 1]. */
    public static final double A_LO = 1e-6;
    /** Upper mean-reversion limit of the calibration domain. */
    public static final double A_HI = 5.0;
    /** Upper sigma limit of the calibration domain. */
    public static final double SIGMA_HI = 1.0;
    private static final double[] START_A = {0.03, 0.1, 0.5};
    private static final double[] START_SIGMA = {0.005, 0.01, 0.02};
    private static final double A_LO_BOUND = 1e-5;
    private static final double A_HI_BOUND = 4.995;
    private static final double SIGMA_LO_BOUND = 1e-5;
    private static final double SIGMA_HI_BOUND = 0.999;
    private static final double EQUIV_RMSE_FACTOR = 1.05;
    private static final double EQUIV_RMSE_FLOOR = 1e-14;
    private static final double IDENT_A = 1e-2;
    private static final double IDENT_SIGMA = 1e-4;
    /** sigma is optimised as sigma * SIGMA_SCALE so one Nelder-Mead step fits both. */
    private static final double SIGMA_SCALE = 10.0;

    private static void checkFinite(String name, double v) {
        if (!Double.isFinite(v)) {
            throw new IllegalArgumentException(name + " must be finite, got " + v);
        }
    }

    private final double a;
    private final double sigma;
    private final DiscountCurve curve;

    /**
     * Creates the model on a market curve.
     *
     * @param a     mean-reversion speed, &gt; 0
     * @param sigma volatility, &gt;= 0
     * @param curve market discount curve (non-null)
     * @throws IllegalArgumentException on invalid parameters
     */
    public HullWhite(double a, double sigma, DiscountCurve curve) {
        if (!Double.isFinite(a) || a <= 0.0) {
            throw new IllegalArgumentException("mean reversion a must be finite and > 0, got " + a);
        }
        if (!Double.isFinite(sigma) || sigma < 0.0) {
            throw new IllegalArgumentException("sigma must be finite and >= 0, got " + sigma);
        }
        if (curve == null) {
            throw new IllegalArgumentException("curve must be a DiscountCurve");
        }
        this.a = a;
        this.sigma = sigma;
        this.curve = curve;
    }

    /** Mean-reversion speed a. */
    public double a() {
        return a;
    }

    /** Volatility sigma. */
    public double sigma() {
        return sigma;
    }

    /** The fitted market curve. */
    public DiscountCurve curve() {
        return curve;
    }

    // ----------------------- market forward via FD --------------------

    /**
     * Market instantaneous forward {@code f^M(0,t)} by finite difference.
     *
     * <p>Exact scheme (h = {@link #FD_STEP} = 1e-5), reproduced identically
     * by all ports:</p>
     * <ul>
     *   <li>{@code t >= h}: central, {@code (ln DF(t-h) - ln DF(t+h)) / (2h)}</li>
     *   <li>{@code t < h}: one-sided, {@code (ln DF(t) - ln DF(t+h)) / h}</li>
     * </ul>
     *
     * <p>With piecewise-constant forwards the FD equals the segment forward
     * away from pillars and the average of the two adjacent forwards when
     * the stencil straddles a pillar.</p>
     *
     * @throws IllegalArgumentException on a negative or non-finite time
     */
    public double fwd0(double t) {
        if (!Double.isFinite(t) || t < 0.0) {
            throw new IllegalArgumentException("time must be finite and >= 0, got " + t);
        }
        double h = FD_STEP;
        if (t >= h) {
            return (Math.log(curve.df(t - h)) - Math.log(curve.df(t + h))) / (2.0 * h);
        }
        return (Math.log(curve.df(t)) - Math.log(curve.df(t + h))) / h;
    }

    /** {@link #theta(double, double)} with the default step {@code h = 1e-4}. */
    public double theta(double t) {
        return theta(t, 1e-4);
    }

    /**
     * Fitted drift {@code theta(t)} (documentation/testing helper).
     *
     * <p>{@code theta(t) = df/dt + a f + (sigma^2/(2a)) (1 - e^(-2 a t))}
     * with {@code df/dt} approximated by the forward difference
     * {@code (fwd0(t+h) - fwd0(t))/h}. Pricing never needs theta(t)
     * explicitly — it is absorbed into A via the market DFs.</p>
     */
    public double theta(double t, double h) {
        if (!Double.isFinite(h) || h <= 0.0) {
            throw new IllegalArgumentException("theta step h must be finite and > 0, got " + h);
        }
        double dfdt = (fwd0(t + h) - fwd0(t)) / h;
        return dfdt + a * fwd0(t)
                + (sigma * sigma / (2.0 * a)) * (-Math.expm1(-2.0 * a * t));
    }

    // ----------------------- affine ZCB -------------------------------

    /**
     * {@code B(t,T) = (1 - e^(-a (T-t))) / a} ({@code expm1} form).
     *
     * @throws IllegalArgumentException unless t and maturity are finite and
     *     {@code 0 <= t <= maturity}
     */
    public double bFactor(double t, double maturity) {
        checkFinite("t", t);
        checkFinite("maturity", maturity);
        if (t < 0.0 || maturity < t) {
            throw new IllegalArgumentException(
                    "need 0 <= t <= maturity, got t=" + t + ", maturity=" + maturity);
        }
        return -Math.expm1(-a * (maturity - t)) / a;
    }

    /** {@code A(t,T)} built from market DFs and {@code f^M(0,t)} (see class doc). */
    public double aFactor(double t, double maturity) {
        double b = bFactor(t, maturity);
        double pmT = curve.df(t);
        double pmMat = curve.df(maturity);
        double conv = sigma * sigma / (4.0 * a) * (-Math.expm1(-2.0 * a * t)) * b * b;
        return (pmMat / pmT) * Math.exp(b * fwd0(t) - conv);
    }

    /** {@code P(t,T)} at the default short rate {@code r(0) = f^M(0,0)},
     *  under which {@code P(0,T)} equals the market DF exactly (the
     *  {@code e^(B f)} and {@code e^(-B r)} terms cancel). */
    public double zcbPrice(double t, double maturity) {
        return zcbPrice(t, maturity, fwd0(0.0));
    }

    /**
     * {@code P(t,T) = A(t,T) e^(-B(t,T) r)}.
     *
     * @throws IllegalArgumentException on a non-finite short rate or when
     *     {@code exp(-B r)} would overflow (result not representable)
     */
    public double zcbPrice(double t, double maturity, double r) {
        if (!Double.isFinite(r)) {
            throw new IllegalArgumentException("short rate must be finite, got " + r);
        }
        double expo = -bFactor(t, maturity) * r;
        if (expo > MAX_EXP_ARG) {
            throw new IllegalArgumentException(
                    "ZCB price not representable: exp(" + expo + ") overflows");
        }
        return aFactor(t, maturity) * Math.exp(expo);
    }

    // ----------------------- ZCB options ------------------------------

    /** Std dev of ln P at expiry:
     *  {@code sigma_p = sigma sqrt((1 - e^(-2 a T))/(2a)) B(T,S)}. */
    private double sigmaP(double expiry, double bondMaturity) {
        return sigma
                * Math.sqrt((-Math.expm1(-2.0 * a * expiry)) / (2.0 * a))
                * bFactor(expiry, bondMaturity);
    }

    private double zbOption(double expiry, double bondMaturity, double strike, boolean call) {
        if (!(Double.isFinite(expiry) && Double.isFinite(bondMaturity) && Double.isFinite(strike))) {
            throw new IllegalArgumentException("option inputs must be finite");
        }
        if (expiry < 0.0 || bondMaturity < expiry) {
            throw new IllegalArgumentException(
                    "need 0 <= expiry <= bond maturity, got " + expiry + ", " + bondMaturity);
        }
        if (strike <= 0.0) {
            throw new IllegalArgumentException("strike must be > 0, got " + strike);
        }
        double ps = curve.df(bondMaturity);
        double pt = curve.df(expiry);
        double sp = sigmaP(expiry, bondMaturity);
        if (sp == 0.0) {
            double intrinsic = ps - strike * pt;
            if (!call) {
                intrinsic = -intrinsic;
            }
            return Math.max(intrinsic, 0.0);
        }
        double h = Math.log(ps / (pt * strike)) / sp + 0.5 * sp;
        if (call) {
            return ps * MathUtils.normCdf(h) - strike * pt * MathUtils.normCdf(h - sp);
        }
        return strike * pt * MathUtils.normCdf(sp - h) - ps * MathUtils.normCdf(-h);
    }

    /** European call on a unit ZCB (uses market DFs — model-consistent). */
    public double zbc(double expiry, double bondMaturity, double strike) {
        return zbOption(expiry, bondMaturity, strike, true);
    }

    /** European put on a unit ZCB. */
    public double zbp(double expiry, double bondMaturity, double strike) {
        return zbOption(expiry, bondMaturity, strike, false);
    }

    // ----------------------- caplets / caps ---------------------------

    /** Validates caplet-style inputs and returns tau. */
    private static double checkCapletInputs(
            String what, double reset, double pay, double strike, double notional) {
        checkFinite("reset", reset);
        checkFinite("pay", pay);
        checkFinite("strike", strike);
        checkFinite("notional", notional);
        double tau = pay - reset;
        if (tau <= 0.0 || reset < 0.0) {
            throw new IllegalArgumentException(
                    what + ": need 0 <= reset < pay, got " + reset + ", " + pay);
        }
        if (1.0 + strike * tau <= 0.0) {
            throw new IllegalArgumentException(
                    what + ": strike implies non-positive 1 + K*tau (" + strike + ")");
        }
        return tau;
    }

    /**
     * Caplet on the simple forward over {@code [reset, pay]}.
     *
     * <p>Pays {@code notional * tau * max(F(reset,pay) - K, 0)} at
     * {@code pay}. A caplet is {@code (1 + K tau)} ZCB <em>puts</em> struck
     * at {@code 1/(1 + K tau)}:
     * {@code caplet = notional (1 + K tau) ZBP(reset, pay, 1/(1 + K tau))}.
     * With {@code sigma = 0} this degenerates to discounted intrinsic on the
     * forward, {@code notional * DF(pay) * tau * max(F - K, 0)}.</p>
     *
     * @throws IllegalArgumentException unless reset, pay, strike and notional
     *     are finite, {@code 0 <= reset < pay} and {@code 1 + K tau > 0}
     *     (notional may be negative: a short position)
     */
    public double caplet(double reset, double pay, double strike, double notional) {
        double tau = checkCapletInputs("caplet", reset, pay, strike, notional);
        double kBond = 1.0 / (1.0 + strike * tau);
        return notional * (1.0 + strike * tau) * zbp(reset, pay, kBond);
    }

    /**
     * Floorlet: {@code notional (1 + K tau) ZBC(reset, pay, 1/(1 + K tau))}.
     *
     * @throws IllegalArgumentException as for {@link #caplet}
     */
    public double floorlet(double reset, double pay, double strike, double notional) {
        double tau = checkCapletInputs("floorlet", reset, pay, strike, notional);
        double kBond = 1.0 / (1.0 + strike * tau);
        return notional * (1.0 + strike * tau) * zbc(reset, pay, kBond);
    }

    /**
     * Cap = strip of caplets over consecutive schedule times.
     *
     * <p>{@code schedule = [t0, t1, ..., tn]} (&gt;= 2 finite, strictly
     * increasing times, {@code t0 >= 0}) prices caplets on
     * {@code [t0,t1], ..., [t_n-1, t_n]}.</p>
     *
     * @throws IllegalArgumentException on a bad schedule or non-finite
     *     strike/notional
     */
    public double cap(double[] schedule, double strike, double notional) {
        if (schedule == null || schedule.length < 2) {
            throw new IllegalArgumentException("cap schedule needs at least two times");
        }
        for (double t : schedule) {
            if (!Double.isFinite(t)) {
                throw new IllegalArgumentException("cap schedule times must be finite");
            }
        }
        if (schedule[0] < 0.0) {
            throw new IllegalArgumentException(
                    "cap schedule must be strictly increasing with t0 >= 0");
        }
        for (int i = 0; i + 1 < schedule.length; i++) {
            if (schedule[i + 1] <= schedule[i]) {
                throw new IllegalArgumentException(
                        "cap schedule must be strictly increasing with t0 >= 0");
            }
        }
        checkFinite("strike", strike);
        checkFinite("notional", notional);
        double total = 0.0;
        for (int i = 0; i + 1 < schedule.length; i++) {
            total += caplet(schedule[i], schedule[i + 1], strike, notional);
        }
        return total;
    }

    /** Par swap rate helper off the model's market curve. */
    public double parSwapRate(double[] times) {
        return DiscountCurve.parSwapRate(curve, times);
    }

    // ----------------------- Jamshidian swaption ----------------------

    /**
     * Jamshidian swaption decomposition output.
     *
     * @param value    swaption PV (notional applied)
     * @param rStar    critical short rate at expiry
     * @param residual coupon-bond repricing residual at rStar
     * @param strikes  decomposition strikes {@code K_i = P(T, u_i; r*)}
     */
    public record JamshidianResult(double value, double rStar, double residual, double[] strikes) {

        /** Defensive copy of the decomposition strikes. */
        @Override
        public double[] strikes() {
            return strikes.clone();
        }
    }

    /** Validates a swaption schedule and returns its coupons. */
    private static double[] swaptionCoupons(
            double expiry, double[] payTimes, double fixedRate, double notional) {
        if (!Double.isFinite(expiry) || expiry <= 0.0) {
            throw new IllegalArgumentException(
                    "swaption expiry must be finite and > 0, got " + expiry);
        }
        if (payTimes == null || payTimes.length < 1) {
            throw new IllegalArgumentException("swaption needs at least one payment time");
        }
        checkFinite("fixedRate", fixedRate);
        checkFinite("notional", notional);
        int m = payTimes.length;
        double[] coupons = new double[m];
        double prev = expiry;
        for (int i = 0; i < m; i++) {
            if (!Double.isFinite(payTimes[i]) || payTimes[i] <= prev) {
                throw new IllegalArgumentException(
                        "payment times must be finite, strictly increasing and after expiry");
            }
            coupons[i] = fixedRate * (payTimes[i] - prev);
            prev = payTimes[i];
        }
        coupons[m - 1] += 1.0;
        if (coupons[m - 1] <= 0.0) {
            throw new IllegalArgumentException(
                    "jamshidian: degenerate coupon bond — 1 + X*tau_n must be > 0 (got "
                    + coupons[m - 1] + "); no r* exists");
        }
        return coupons;
    }

    /**
     * Jamshidian decomposition for a <em>given</em> critical rate {@code r*}.
     *
     * <p>Enforces the contract of {@link #jamshidianSwaption} step 2: the
     * residual {@code g(r*) = sum_i c_i P(T,u_i;r*) - 1} must satisfy
     * {@code |g(r*)| < 1e-10} and the slope {@code g'(r*) = -sum_i c_i B_i K_i}
     * must be negative (the crossing is on the decreasing branch of g).
     * Exposed so the contract is testable and so payer/receiver can share
     * one root find.</p>
     *
     * @throws IllegalArgumentException on invalid inputs, a non-finite r*, or
     *     when the residual or slope contract fails
     */
    public JamshidianResult jamshidianAtRStar(
            double expiry, double[] payTimes, double fixedRate, double rStar, double notional,
            boolean payer) {
        double[] coupons = swaptionCoupons(expiry, payTimes, fixedRate, notional);
        if (!Double.isFinite(rStar)) {
            throw new IllegalArgumentException("jamshidian: r* must be finite, got " + rStar);
        }
        int m = payTimes.length;
        double[] ts = payTimes.clone();
        double[] strikes = new double[m];
        double residual = -1.0;
        for (int i = 0; i < m; i++) {
            strikes[i] = zcbPrice(expiry, ts[i], rStar);
            residual += coupons[i] * strikes[i];
        }
        if (!(Math.abs(residual) < JAMSHIDIAN_RESIDUAL_TOL)) {
            throw new IllegalArgumentException("jamshidian: r* residual " + residual
                    + " exceeds " + JAMSHIDIAN_RESIDUAL_TOL);
        }
        double slope = 0.0;
        for (int i = 0; i < m; i++) {
            slope -= coupons[i] * bFactor(expiry, ts[i]) * strikes[i];
        }
        if (!(slope < 0.0)) {
            throw new IllegalArgumentException("jamshidian: coupon bond not decreasing at r* (slope "
                    + slope + "); decomposition invalid");
        }
        double value = 0.0;
        for (int i = 0; i < m; i++) {
            value += coupons[i] * (payer
                    ? zbp(expiry, ts[i], strikes[i])
                    : zbc(expiry, ts[i], strikes[i]));
        }
        return new JamshidianResult(notional * value, rStar, residual, strikes);
    }

    /**
     * European swaption via Jamshidian's decomposition.
     *
     * <p>The underlying swap exchanges the fixed rate for the (single-curve)
     * floating leg over {@code [expiry, payTimes]}; accruals are
     * {@code tau_1 = t_1 - expiry}, {@code tau_i = t_i - t_i-1}. A payer
     * swaption's payoff at expiry is {@code max(1 - sum_i c_i P(T, t_i), 0)}
     * with coupons {@code c_i = X tau_i} and {@code c_n = 1 + X tau_n} — a
     * put with strike 1 on a coupon bond {@code g(r) = sum_i c_i P(T,t_i;r)}.</p>
     *
     * <p>Exactness needs {@code g(r) - 1} to change sign exactly once. For
     * {@code X >= 0} every coupon is positive and g is strictly decreasing
     * (each {@code B > 0}). For {@code X < 0} the coupon signs are
     * {@code (-, ..., -, +)} when {@code 1 + X tau_n > 0}: g' is an
     * exponential sum with one sign change, so g has a single minimum, tends
     * to +inf as r -&gt; -inf and to 0^- as r -&gt; +inf, and still crosses 1
     * exactly once — the decomposition remains exact (verified against direct
     * integration in the tests). If {@code 1 + X tau_n <= 0} every coupon is
     * non-positive and no r* exists ("degenerate").</p>
     *
     * <p>{@code r*} is found with the native Brent solver on an expanding
     * bracket (start {@code [-1, 1]}, double the failing side, at most 24
     * doublings, never past the exp() overflow point), {@code xtol = 1e-15};
     * then {@code |g(r*)| < 1e-10} and {@code g'(r*) < 0} are enforced by
     * {@link #jamshidianAtRStar}. Setting {@code K_i = P(T, t_i; r*)}, the
     * coupon-bond option splits <em>exactly</em> into ZCB options all
     * exercised on the same event {r_T &gt; r*}: payer =
     * {@code sum c_i ZBP(T, t_i, K_i)}, receiver = {@code sum c_i ZBC(T, t_i, K_i)}.</p>
     *
     * @throws IllegalArgumentException on invalid schedule/inputs, a
     *     degenerate coupon bond, a bracketing failure, or a residual/slope
     *     contract violation
     */
    public JamshidianResult jamshidianSwaption(
            double expiry, double[] payTimes, double fixedRate, double notional, boolean payer) {
        double[] coupons = swaptionCoupons(expiry, payTimes, fixedRate, notional);
        int m = payTimes.length;
        double[] ts = payTimes.clone();

        DoubleUnaryOperator bondMinusOne = r -> {
            double sum = 0.0;
            for (int i = 0; i < m; i++) {
                sum += coupons[i] * zcbPrice(expiry, ts[i], r);
            }
            return sum - 1.0;
        };

        // Expanding bracket: g(lo) > 0 > g(hi) is needed.  g(lo) grows like
        // exp(B_n |lo|); refuse to evaluate past the exp() overflow point.
        double bLast = bFactor(expiry, ts[m - 1]);
        double lo = -1.0;
        double hi = 1.0;
        boolean bracketed = false;
        for (int k = 0; k < JAMSHIDIAN_MAX_DOUBLINGS; k++) {
            if (bLast * (-lo) > MAX_EXP_ARG) {
                break;
            }
            double gLo = bondMinusOne.applyAsDouble(lo);
            double gHi = bondMinusOne.applyAsDouble(hi);
            if (gLo > 0.0 && gHi < 0.0) {
                bracketed = true;
                break;
            }
            if (gLo <= 0.0) {
                lo *= 2.0;
            }
            if (gHi >= 0.0) {
                hi *= 2.0;
            }
        }
        if (!bracketed) {
            throw new IllegalArgumentException(
                    "jamshidian: could not bracket r* (degenerate coupon bond — "
                    + "check fixed rate/schedule)");
        }
        double rStar = RootFind.brentq(bondMinusOne, lo, hi, 1e-15);
        return jamshidianAtRStar(expiry, ts, fixedRate, rStar, notional, payer);
    }

    // ----------------------- Monte Carlo ------------------------------

    /**
     * Deterministic shift {@code alpha(t) = f^M(0,t)
     * + (sigma^2/(2 a^2)) (1 - e^(-a t))^2} with {@code r(t) = x(t) + alpha(t)}.
     */
    public double alpha(double t) {
        double g = -Math.expm1(-a * t);
        return fwd0(t) + (sigma * sigma / (2.0 * a * a)) * g * g;
    }

    /**
     * {@code V(t) = int_0^t (sigma^2/(2a^2))(1 - e^(-a s))^2 ds} so that
     * {@code int_0^t alpha ds = -ln P^M(0,t) + V(t)}; closed form
     * {@code V(t) = (sigma^2/(2a^2)) [t - 2(1-e^(-a t))/a + (1-e^(-2 a t))/(2a)]},
     * which is half the integrated-OU variance kernel
     * ({@link MathUtils#ouIntegralVariance}), evaluated without
     * cancellation. {@code V(0) = 0}.
     */
    public double varianceIntegral(double t) {
        if (t == 0.0) {
            return 0.0;
        }
        return 0.5 * MathUtils.ouIntegralVariance(a, sigma, t);
    }

    /**
     * Unbiased MC caplet price and standard error.
     *
     * <p>Simulates {@code x} to the reset date with exact OU joint moments
     * of {@code (x, int x dt)}; the pathwise discount factor is exact:
     * {@code D(0,T) = P^M(0,T) exp(-V(T) - int_0^T x dt)}. At reset,
     * {@code P(T, pay)} is computed analytically from
     * {@code r(T) = x(T) + alpha(T)}; the payoff (paid at {@code pay}, known
     * at {@code T}) is valued as
     * {@code D(0,T) P(T,pay) * notional * tau * max(F-K,0)}. No
     * discretisation bias at any step count (sigma = 0 gives the
     * deterministic intrinsic and a standard error of exactly 0). Same
     * validation as {@link #caplet} plus {@code reset > 0}.</p>
     *
     * @throws IllegalArgumentException unless {@code 0 < reset < pay}, finite
     *     strike/notional with {@code 1 + K tau > 0}, and positive step/path counts
     */
    public McEstimate mcCaplet(
            double reset, double pay, double strike, double notional,
            int nSteps, int nPaths, long seed) {
        double tau = checkCapletInputs("mcCaplet", reset, pay, strike, notional);
        if (reset <= 0.0) {
            throw new IllegalArgumentException(
                    "mcCaplet: need 0 < reset < pay, got " + reset + ", " + pay);
        }
        if (nSteps < 1 || nPaths < 1) {
            throw new IllegalArgumentException("nSteps and nPaths must be >= 1");
        }
        double dt = reset / nSteps;
        MathUtils.OuStepMoments mo = MathUtils.ouStepMoments(a, sigma, dt);
        Random rng = new Random(seed);
        double dfReset = curve.df(reset);
        double vReset = varianceIntegral(reset);
        double alphaReset = alpha(reset);
        double b = bFactor(reset, pay);
        double aF = aFactor(reset, pay);

        double sdX = 0.0;
        double beta = 0.0;
        double sdResid = 0.0;
        boolean stochastic = mo.varX() > 0.0;
        if (stochastic) {
            sdX = Math.sqrt(mo.varX());
            beta = mo.cov() / mo.varX();
            sdResid = Math.sqrt(Math.max(mo.varI() - mo.cov() * mo.cov() / mo.varX(), 0.0));
        }

        double[] pv = new double[nPaths];
        double mean = 0.0;
        for (int p = 0; p < nPaths; p++) {
            double x = 0.0;
            double integralX = 0.0;
            if (stochastic) {
                for (int s = 0; s < nSteps; s++) {
                    double z1 = rng.nextGaussian();
                    double z2 = rng.nextGaussian();
                    double meanX = x * mo.decay();
                    double xNew = meanX + sdX * z1;
                    integralX += x * mo.b() + beta * (xNew - meanX) + sdResid * z2;
                    x = xNew;
                }
            }
            // sigma = 0: x stays 0, integral stays 0 (deterministic curve replay).
            double disc = dfReset * Math.exp(-vReset - integralX);
            double rReset = x + alphaReset;
            double pResetPay = aF * Math.exp(-b * rReset);
            double fwd = (1.0 / pResetPay - 1.0) / tau;
            double payoff = notional * tau * Math.max(fwd - strike, 0.0);
            pv[p] = disc * pResetPay * payoff;
            mean += pv[p];
        }
        mean /= nPaths;
        // sigma = 0 is deterministic: the standard error is exactly 0 (avoid
        // spurious 1-ulp noise from summing identical values).
        double se = 0.0;
        if (nPaths > 1 && stochastic) {
            double ss = 0.0;
            for (double v : pv) {
                ss += (v - mean) * (v - mean);
            }
            se = Math.sqrt(ss / (nPaths - 1)) / Math.sqrt(nPaths);
        }
        return new McEstimate(mean, se);
    }

    // ----------------------- calibration of (a, sigma) ---------------

    /**
     * Market caplet price per unit notional on {@code [reset, pay]} at {@code strike}.
     *
     * @param reset  reset (fixing) time, &gt;= 0
     * @param pay    payment time, &gt; reset
     * @param strike strike rate with {@code 1 + K tau > 0}
     * @param price  market price per unit notional, &gt;= 0
     */
    public record CapletQuote(double reset, double pay, double strike, double price) {

        /** Validates the quote. */
        public CapletQuote {
            checkFinite("reset", reset);
            checkFinite("pay", pay);
            checkFinite("strike", strike);
            checkFinite("price", price);
            if (reset < 0.0 || pay <= reset) {
                throw new IllegalArgumentException(
                        "caplet quote needs 0 <= reset < pay, got " + reset + ", " + pay);
            }
            if (1.0 + strike * (pay - reset) <= 0.0) {
                throw new IllegalArgumentException(
                        "caplet quote strike implies non-positive 1 + K*tau");
            }
            if (price < 0.0) {
                throw new IllegalArgumentException(
                        "caplet quote price must be >= 0, got " + price);
            }
        }
    }

    /**
     * Market <em>payer</em> swaption price per unit notional.
     *
     * @param expiry    option expiry (swap start), &gt; 0
     * @param payTimes  fixed-leg payment times, strictly increasing and after expiry
     * @param fixedRate fixed rate of the underlying swap
     * @param price     market price per unit notional, &gt;= 0
     */
    public record SwaptionQuote(double expiry, double[] payTimes, double fixedRate, double price) {

        /** Validates the quote and copies the schedule. */
        public SwaptionQuote {
            checkFinite("expiry", expiry);
            checkFinite("fixedRate", fixedRate);
            checkFinite("price", price);
            if (expiry <= 0.0) {
                throw new IllegalArgumentException(
                        "swaption quote expiry must be > 0, got " + expiry);
            }
            if (payTimes == null || payTimes.length < 1) {
                throw new IllegalArgumentException(
                        "swaption quote needs at least one payment time");
            }
            double prev = expiry;
            for (double t : payTimes) {
                if (!Double.isFinite(t) || t <= prev) {
                    throw new IllegalArgumentException("swaption quote payment times must be "
                            + "finite, strictly increasing and after expiry");
                }
                prev = t;
            }
            if (price < 0.0) {
                throw new IllegalArgumentException(
                        "swaption quote price must be >= 0, got " + price);
            }
            payTimes = payTimes.clone();
        }

        /** Defensive copy of the payment schedule. */
        @Override
        public double[] payTimes() {
            return payTimes.clone();
        }
    }

    /**
     * Outcome of {@link #calibrate} (same diagnostics as {@link Vasicek.Calibration}).
     *
     * <p>{@code rmse} is in price units per unit notional. {@code converged}
     * means the optimiser converged at the best start <em>and</em> the
     * solution is not at a domain bound ({@code a < 1e-5} or {@code > 4.995},
     * {@code sigma < 1e-5} or {@code > 0.999}). {@code identified} requires
     * {@code nStarts >= 2} and {@code aSpread <= 1e-2}, {@code sigmaSpread <= 1e-4}
     * over the starts whose rmse is within 5 % of the best. {@code aFixed} is
     * true when {@code a} was supplied, not fitted.</p>
     *
     * @param a           fitted (or fixed) mean reversion
     * @param sigma       fitted volatility
     * @param rmse        root-mean-square price error per unit notional
     * @param iterations  optimizer iterations of the best start
     * @param converged   optimizer converged and not at a bound
     * @param atBound     solution at the edge of the domain
     * @param nStarts     number of starts run
     * @param aSpread     spread of a over equivalent starts
     * @param sigmaSpread spread of sigma over equivalent starts
     * @param identified  equivalent starts agree (multi-start only)
     * @param aFixed      a held fixed at the supplied value
     */
    public record Calibration(
            double a, double sigma, double rmse, int iterations, boolean converged,
            boolean atBound, int nStarts, double aSpread, double sigmaSpread,
            boolean identified, boolean aFixed) {

        /** Builds the calibrated model on {@code curve}. */
        public HullWhite model(DiscountCurve curve) {
            return new HullWhite(a, sigma, curve);
        }
    }

    /** One Nelder-Mead start's outcome. */
    private record StartResult(double fx, double a, double sigma, int iterations,
            boolean converged) {
    }

    /**
     * Fits {@code (a, sigma)} to caplet and payer-swaption prices by least squares.
     *
     * <p>Minimises {@code sum_j (model_price_j - market_price_j)^2} over all
     * quotes (prices per unit notional) with Nelder-Mead on
     * {@code (a, 10 sigma)} ({@code initialStep = 0.05}, i.e. 0.005 in sigma);
     * out-of-domain points ({@code a <= 1e-6}, {@code a > 5}, {@code sigma < 0},
     * {@code sigma > 1}) get the penalty {@code 1e6 (1 + distance)}.</p>
     * <ul>
     *   <li>{@code x0 == null}: three starts {@code (a, sigma)} in
     *       {@code {(0.03, 0.005), (0.1, 0.01), (0.5, 0.02)}}, best objective
     *       returned with spreads over equivalent starts.</li>
     *   <li>{@code x0 = {a, sigma}}: single warm start.</li>
     *   <li>{@code aFixed != null}: fit sigma only (1-D), starting at the sigma
     *       values of the default starts; {@code x0[0]} is ignored.</li>
     * </ul>
     * <p>At least one quote is required; with both parameters free at least
     * two quotes of different expiry/tenor are needed for identifiability (a
     * single quote fits sigma for any {@code a} — {@code identified} will be
     * false). Non-convergence is reported, never thrown. Quotes are trusted
     * as given: convert broker normal vols with {@link Bachelier#price} first.</p>
     *
     * @param curve     market discount curve
     * @param caplets   caplet quotes (may be empty)
     * @param swaptions payer swaption quotes (may be empty)
     * @param aFixed    fixed mean reversion, or null to fit it
     * @param x0        optional warm start {@code {a, sigma}}
     * @param maxiter   Nelder-Mead iteration budget per start (&gt;= 1)
     * @throws IllegalArgumentException on no quotes, a wrong-length / non-finite
     *     x0, non-positive aFixed, or {@code maxiter < 1}
     */
    public static Calibration calibrate(
            DiscountCurve curve, List<CapletQuote> caplets, List<SwaptionQuote> swaptions,
            Double aFixed, double[] x0, int maxiter) {
        if (curve == null) {
            throw new IllegalArgumentException("curve must be a DiscountCurve");
        }
        final List<CapletQuote> caps = caplets == null ? List.of() : List.copyOf(caplets);
        final List<SwaptionQuote> swps = swaptions == null ? List.of() : List.copyOf(swaptions);
        if (caps.isEmpty() && swps.isEmpty()) {
            throw new IllegalArgumentException(
                    "calibrate needs at least one caplet or swaption quote");
        }
        if (maxiter < 1) {
            throw new IllegalArgumentException("maxiter must be >= 1");
        }
        if (x0 != null) {
            if (x0.length != 2) {
                throw new IllegalArgumentException("x0 must have 2 entries (a, sigma)");
            }
            for (double v : x0) {
                if (!Double.isFinite(v)) {
                    throw new IllegalArgumentException("x0 must be finite");
                }
            }
        }
        if (aFixed != null && (!Double.isFinite(aFixed) || aFixed <= 0.0)) {
            throw new IllegalArgumentException("aFixed must be finite and > 0, got " + aFixed);
        }
        final boolean fixed = aFixed != null;
        final double aFix = fixed ? aFixed : Double.NaN;
        final double nQuotes = caps.size() + swps.size();

        // Nelder-Mead uses one initialStep (0.05) for every coordinate, so the
        // optimiser works on sigma * SIGMA_SCALE (step 0.005 in sigma units).
        ToDoubleFunction<double[]> objective = p -> {
            double a = fixed ? aFix : p[0];
            double sigma = (fixed ? p[0] : p[1]) / SIGMA_SCALE;
            if (a <= A_LO || a > A_HI || sigma < 0.0 || sigma > SIGMA_HI) {
                return 1e6 * (1.0
                        + Math.max(0.0, A_LO - a)
                        + Math.max(0.0, a - A_HI)
                        + Math.max(0.0, -sigma)
                        + Math.max(0.0, sigma - SIGMA_HI));
            }
            HullWhite hw = new HullWhite(a, sigma, curve);
            double total = 0.0;
            for (CapletQuote q : caps) {
                double d = hw.caplet(q.reset(), q.pay(), q.strike(), 1.0) - q.price();
                total += d * d;
            }
            for (SwaptionQuote q : swps) {
                double d = hw.jamshidianSwaption(
                        q.expiry(), q.payTimes(), q.fixedRate(), 1.0, true).value() - q.price();
                total += d * d;
            }
            return total;
        };

        double[][] starts;
        if (fixed) {
            if (x0 != null) {
                starts = new double[][] {{x0[1] * SIGMA_SCALE}};
            } else {
                starts = new double[3][];
                for (int k = 0; k < 3; k++) {
                    starts[k] = new double[] {START_SIGMA[k] * SIGMA_SCALE};
                }
            }
        } else {
            if (x0 != null) {
                starts = new double[][] {{x0[0], x0[1] * SIGMA_SCALE}};
            } else {
                starts = new double[3][];
                for (int k = 0; k < 3; k++) {
                    starts[k] = new double[] {START_A[k], START_SIGMA[k] * SIGMA_SCALE};
                }
            }
        }
        StartResult[] results = new StartResult[starts.length];
        for (int k = 0; k < starts.length; k++) {
            NelderMead.Result res =
                    NelderMead.minimize(objective, starts[k], 0.05, 1e-10, 1e-14, maxiter);
            double[] best = res.x();
            double aVal = fixed ? aFix : best[0];
            double sVal = (fixed ? Math.abs(best[0]) : Math.abs(best[1])) / SIGMA_SCALE;
            results[k] = new StartResult(res.fx(), aVal, sVal, res.iterations(), res.converged());
        }
        int bestIdx = 0;
        for (int k = 1; k < results.length; k++) {
            if (results[k].fx() < results[bestIdx].fx()) {
                bestIdx = k;
            }
        }
        StartResult b = results[bestIdx];
        double rmse = Math.sqrt(Math.max(b.fx(), 0.0) / nQuotes);
        boolean atBound = (!fixed && (b.a() < A_LO_BOUND || b.a() > A_HI_BOUND))
                || b.sigma() < SIGMA_LO_BOUND || b.sigma() > SIGMA_HI_BOUND;
        double aLo = b.a();
        double aHi = b.a();
        double sLo = b.sigma();
        double sHi = b.sigma();
        for (StartResult r : results) {
            double rRmse = Math.sqrt(Math.max(r.fx(), 0.0) / nQuotes);
            if (rRmse <= EQUIV_RMSE_FACTOR * rmse + EQUIV_RMSE_FLOOR) {
                aLo = Math.min(aLo, r.a());
                aHi = Math.max(aHi, r.a());
                sLo = Math.min(sLo, r.sigma());
                sHi = Math.max(sHi, r.sigma());
            }
        }
        double aSpread = aHi - aLo;
        double sigmaSpread = sHi - sLo;
        boolean identified = results.length >= 2 && aSpread <= IDENT_A
                && sigmaSpread <= IDENT_SIGMA;
        return new Calibration(b.a(), b.sigma(), rmse, b.iterations(),
                b.converged() && !atBound, atBound, results.length, aSpread, sigmaSpread,
                identified, fixed);
    }
}
