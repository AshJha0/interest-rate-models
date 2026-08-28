package com.quant.irm;

import java.util.Random;

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
 */
public final class HullWhite {

    /** Finite-difference step for the market instantaneous forward f(0,t). */
    public static final double FD_STEP = 1e-5;

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
        double dfdt = (fwd0(t + h) - fwd0(t)) / h;
        return dfdt + a * fwd0(t)
                + (sigma * sigma / (2.0 * a)) * (1.0 - Math.exp(-2.0 * a * t));
    }

    // ----------------------- affine ZCB -------------------------------

    /**
     * {@code B(t,T) = (1 - e^(-a (T-t))) / a}.
     *
     * @throws IllegalArgumentException if {@code maturity < t}
     */
    public double bFactor(double t, double maturity) {
        if (maturity < t) {
            throw new IllegalArgumentException("maturity " + maturity + " before t " + t);
        }
        return (1.0 - Math.exp(-a * (maturity - t))) / a;
    }

    /** {@code A(t,T)} built from market DFs and {@code f^M(0,t)} (see class doc). */
    public double aFactor(double t, double maturity) {
        double b = bFactor(t, maturity);
        double pmT = curve.df(t);
        double pmMat = curve.df(maturity);
        double conv = sigma * sigma / (4.0 * a) * (1.0 - Math.exp(-2.0 * a * t)) * b * b;
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
     * @throws IllegalArgumentException on a non-finite short rate
     */
    public double zcbPrice(double t, double maturity, double r) {
        if (!Double.isFinite(r)) {
            throw new IllegalArgumentException("short rate must be finite, got " + r);
        }
        return aFactor(t, maturity) * Math.exp(-bFactor(t, maturity) * r);
    }

    // ----------------------- ZCB options ------------------------------

    /** Std dev of ln P at expiry:
     *  {@code sigma_p = sigma sqrt((1 - e^(-2 a T))/(2a)) B(T,S)}. */
    private double sigmaP(double expiry, double bondMaturity) {
        return sigma
                * Math.sqrt((1.0 - Math.exp(-2.0 * a * expiry)) / (2.0 * a))
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
     * @throws IllegalArgumentException unless {@code 0 <= reset < pay} and
     *     {@code 1 + K tau > 0}
     */
    public double caplet(double reset, double pay, double strike, double notional) {
        double tau = pay - reset;
        if (tau <= 0.0 || reset < 0.0) {
            throw new IllegalArgumentException("need 0 <= reset < pay, got " + reset + ", " + pay);
        }
        if (!Double.isFinite(strike) || 1.0 + strike * tau <= 0.0) {
            throw new IllegalArgumentException(
                    "caplet strike implies non-positive 1 + K*tau (" + strike + ")");
        }
        double kBond = 1.0 / (1.0 + strike * tau);
        return notional * (1.0 + strike * tau) * zbp(reset, pay, kBond);
    }

    /**
     * Floorlet: {@code notional (1 + K tau) ZBC(reset, pay, 1/(1 + K tau))}.
     *
     * @throws IllegalArgumentException unless {@code 0 <= reset < pay} and
     *     {@code 1 + K tau > 0}
     */
    public double floorlet(double reset, double pay, double strike, double notional) {
        double tau = pay - reset;
        if (tau <= 0.0 || reset < 0.0) {
            throw new IllegalArgumentException("need 0 <= reset < pay, got " + reset + ", " + pay);
        }
        if (!Double.isFinite(strike) || 1.0 + strike * tau <= 0.0) {
            throw new IllegalArgumentException(
                    "floorlet strike implies non-positive 1 + K*tau (" + strike + ")");
        }
        double kBond = 1.0 / (1.0 + strike * tau);
        return notional * (1.0 + strike * tau) * zbc(reset, pay, kBond);
    }

    /**
     * Cap = strip of caplets over consecutive schedule times.
     *
     * <p>{@code schedule = [t0, t1, ..., tn]} prices caplets on
     * {@code [t0,t1], ..., [t_n-1, t_n]} (first reset {@code t0 >= 0}).</p>
     *
     * @throws IllegalArgumentException on fewer than two schedule times
     */
    public double cap(double[] schedule, double strike, double notional) {
        if (schedule == null || schedule.length < 2) {
            throw new IllegalArgumentException("cap schedule needs at least two times");
        }
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

    /**
     * European swaption via Jamshidian's decomposition.
     *
     * <p>The underlying swap exchanges the fixed rate for the (single-curve)
     * floating leg over {@code [expiry, payTimes]}; accruals are
     * {@code tau_1 = t_1 - expiry}, {@code tau_i = t_i - t_i-1}. A payer
     * swaption's payoff at expiry is {@code max(1 - sum_i c_i P(T, t_i), 0)}
     * with coupons {@code c_i = X tau_i} and {@code c_n = 1 + X tau_n} — a
     * put with strike 1 on a coupon bond.</p>
     *
     * <p>Because {@code P(T, t; r)} is strictly decreasing in r (B &gt; 0),
     * the coupon bond price is monotone in r, so there is a unique
     * {@code r*} with {@code sum_i c_i P(T, t_i; r*) = 1} (found with the
     * native Brent solver on an expanding bracket, {@code xtol = 1e-15};
     * {@code |residual| < 1e-10} required). Setting
     * {@code K_i = P(T, t_i; r*)}, the coupon-bond option splits
     * <em>exactly</em> into ZCB options all exercised on the same event
     * {r_T &gt; r*}: payer = {@code sum c_i ZBP(T, t_i, K_i)}, receiver =
     * {@code sum c_i ZBC(T, t_i, K_i)}.</p>
     *
     * @throws IllegalArgumentException on invalid schedule/inputs or when
     *     {@code r*} cannot be bracketed (degenerate coupon bond)
     */
    public JamshidianResult jamshidianSwaption(
            double expiry, double[] payTimes, double fixedRate, double notional, boolean payer) {
        if (!Double.isFinite(expiry) || expiry <= 0.0) {
            throw new IllegalArgumentException("swaption expiry must be > 0, got " + expiry);
        }
        if (payTimes == null || payTimes.length < 1) {
            throw new IllegalArgumentException("swaption needs at least one payment time");
        }
        if (!Double.isFinite(fixedRate)) {
            throw new IllegalArgumentException("fixed rate must be finite");
        }
        int m = payTimes.length;
        double[] ts = payTimes.clone();
        double[] coupons = new double[m];
        double prev = expiry;
        for (int i = 0; i < m; i++) {
            if (ts[i] <= prev) {
                throw new IllegalArgumentException(
                        "payment times must be strictly increasing and after expiry");
            }
            coupons[i] = fixedRate * (ts[i] - prev);
            prev = ts[i];
        }
        coupons[m - 1] += 1.0;

        java.util.function.DoubleUnaryOperator bondMinusOne = r -> {
            double sum = 0.0;
            for (int i = 0; i < m; i++) {
                sum += coupons[i] * zcbPrice(expiry, ts[i], r);
            }
            return sum - 1.0;
        };

        // Expanding bracket: the function is strictly decreasing in r.
        double lo = -1.0;
        double hi = 1.0;
        boolean bracketed = false;
        for (int k = 0; k < 24; k++) {
            if (bondMinusOne.applyAsDouble(lo) > 0.0 && bondMinusOne.applyAsDouble(hi) < 0.0) {
                bracketed = true;
                break;
            }
            if (bondMinusOne.applyAsDouble(lo) <= 0.0) {
                lo *= 2.0;
            }
            if (bondMinusOne.applyAsDouble(hi) >= 0.0) {
                hi *= 2.0;
            }
        }
        if (!bracketed && !(bondMinusOne.applyAsDouble(lo) > 0.0
                && bondMinusOne.applyAsDouble(hi) < 0.0)) {
            throw new IllegalArgumentException(
                    "jamshidian: could not bracket r* (degenerate coupon bond — "
                    + "check fixed rate/schedule)");
        }
        double rStar = RootFind.brentq(bondMinusOne, lo, hi, 1e-15);
        double residual = bondMinusOne.applyAsDouble(rStar);
        if (Math.abs(residual) >= 1e-10) {
            throw new IllegalStateException(
                    "jamshidian: r* residual " + residual + " exceeds 1e-10");
        }
        double[] strikes = new double[m];
        for (int i = 0; i < m; i++) {
            strikes[i] = zcbPrice(expiry, ts[i], rStar);
        }
        double value = 0.0;
        for (int i = 0; i < m; i++) {
            value += coupons[i] * (payer
                    ? zbp(expiry, ts[i], strikes[i])
                    : zbc(expiry, ts[i], strikes[i]));
        }
        return new JamshidianResult(notional * value, rStar, residual, strikes);
    }

    // ----------------------- Monte Carlo ------------------------------

    /**
     * Deterministic shift {@code alpha(t) = f^M(0,t)
     * + (sigma^2/(2 a^2)) (1 - e^(-a t))^2} with {@code r(t) = x(t) + alpha(t)}.
     */
    public double alpha(double t) {
        double g = 1.0 - Math.exp(-a * t);
        return fwd0(t) + (sigma * sigma / (2.0 * a * a)) * g * g;
    }

    /**
     * {@code V(t) = int_0^t (sigma^2/(2a^2))(1 - e^(-a s))^2 ds} so that
     * {@code int_0^t alpha ds = -ln P^M(0,t) + V(t)}; closed form:
     * {@code V(t) = (sigma^2/(2a^2)) [t - 2(1-e^(-a t))/a + (1-e^(-2 a t))/(2a)]}.
     */
    private double varianceIntegral(double t) {
        return (sigma * sigma / (2.0 * a * a)) * (t
                - 2.0 * (1.0 - Math.exp(-a * t)) / a
                + (1.0 - Math.exp(-2.0 * a * t)) / (2.0 * a));
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
     * deterministic intrinsic).</p>
     *
     * @throws IllegalArgumentException unless {@code 0 < reset < pay} and
     *     positive step/path counts
     */
    public McEstimate mcCaplet(
            double reset, double pay, double strike, double notional,
            int nSteps, int nPaths, long seed) {
        double tau = pay - reset;
        if (tau <= 0.0 || reset <= 0.0) {
            throw new IllegalArgumentException("need 0 < reset < pay, got " + reset + ", " + pay);
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
}
