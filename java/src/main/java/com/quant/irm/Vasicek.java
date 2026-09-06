package com.quant.irm;

import java.util.Random;
import java.util.function.ToDoubleFunction;

/**
 * Vasicek short-rate model {@code dr = kappa (theta - r) dt + sigma dW}.
 *
 * <p>Gaussian model: rates can go negative, which is a <em>feature</em> here
 * (EUR/JPY style curves) and the reason no lognormal short-rate model is
 * included in this project.</p>
 *
 * <p>Closed forms implemented (affine term structure):</p>
 * <pre>
 * B(tau) = (1 - e^(-kappa tau)) / kappa
 * A(tau) = exp[(theta - sigma^2/(2 kappa^2)) (B - tau) - sigma^2 B^2/(4 kappa)]
 * P(t,T) = A e^(-B r(t))
 * r_T | r_t ~ N(theta + (r_t - theta) e^(-kappa tau),
 *               sigma^2 (1 - e^(-2 kappa tau)) / (2 kappa))
 * </pre>
 *
 * <p>plus the Jamshidian (1989) bond-option formula for European calls/puts
 * on ZCBs. Monte Carlo uses the <em>exact</em> OU transition of the pair
 * {@code (r, int r dt)} (see {@link MathUtils#ouStepMoments}), so the ZCB
 * estimator is unbiased at any step count. Calibration fits
 * {@code (kappa, theta, sigma)} — or {@code (kappa, theta)} with sigma fixed
 * from option prices — to market zero yields by least squares with the
 * native {@link NelderMead}, from several starting points, and reports
 * identifiability and boundary diagnostics.</p>
 */
public final class Vasicek {

    /** Calibration domain: kappa in (1e-6, 50], |theta| &lt;= 5, sigma in [0, 5]. */
    public static final double KAPPA_LO = 1e-6;
    /** Upper kappa limit of the calibration domain. */
    public static final double KAPPA_HI = 50.0;
    /** Absolute theta limit of the calibration domain. */
    public static final double THETA_MAX = 5.0;
    /** Upper sigma limit of the calibration domain. */
    public static final double SIGMA_HI = 5.0;
    /** Default multi-start kappa values; theta starts at the mean yield. */
    private static final double[] START_KAPPA = {0.2, 0.5, 1.5};
    /** Default multi-start sigma values. */
    private static final double[] START_SIGMA = {0.005, 0.01, 0.02};
    /** exp() argument above which a ZCB price is declared not representable. */
    private static final double MAX_EXP_ARG = 700.0;
    // at_bound thresholds (documented in API_SPEC section 7).
    private static final double KAPPA_LO_BOUND = 1e-5;
    private static final double KAPPA_HI_BOUND = 49.95;
    private static final double THETA_BOUND = 4.995;
    private static final double SIGMA_LO_BOUND = 1e-5;
    private static final double SIGMA_HI_BOUND = 4.995;
    // Starts whose rmse is within 5 % of the best are "equivalent fits".
    private static final double EQUIV_RMSE_FACTOR = 1.05;
    private static final double EQUIV_RMSE_FLOOR = 1e-12;
    private static final double IDENT_KAPPA = 1e-2;
    private static final double IDENT_THETA = 1e-3;
    private static final double IDENT_SIGMA = 1e-3;

    private final double kappa;
    private final double theta;
    private final double sigma;
    private final double r0;

    /**
     * Creates the model.
     *
     * @param kappa mean-reversion speed, &gt; 0
     * @param theta long-run mean level (any sign)
     * @param sigma volatility, &gt;= 0
     * @param r0    initial short rate (any sign)
     * @throws IllegalArgumentException on non-finite input, {@code kappa <= 0}
     *     or {@code sigma < 0}
     */
    public Vasicek(double kappa, double theta, double sigma, double r0) {
        checkFinite("kappa", kappa);
        checkFinite("theta", theta);
        checkFinite("sigma", sigma);
        checkFinite("r0", r0);
        if (kappa <= 0.0) {
            throw new IllegalArgumentException("kappa must be > 0, got " + kappa);
        }
        if (sigma < 0.0) {
            throw new IllegalArgumentException("sigma must be >= 0, got " + sigma);
        }
        this.kappa = kappa;
        this.theta = theta;
        this.sigma = sigma;
        this.r0 = r0;
    }

    private static void checkFinite(String name, double v) {
        if (!Double.isFinite(v)) {
            throw new IllegalArgumentException(name + " must be finite, got " + v);
        }
    }

    /** Mean-reversion speed kappa. */
    public double kappa() {
        return kappa;
    }

    /** Long-run mean theta. */
    public double theta() {
        return theta;
    }

    /** Volatility sigma. */
    public double sigma() {
        return sigma;
    }

    /** Initial short rate r0. */
    public double r0() {
        return r0;
    }

    // ----------------------------- affine functions -------------------

    /**
     * {@code B(t,T)} with {@code tau = T - t}: duration of the ZCB w.r.t. r.
     *
     * <p>{@code B = (1 - e^(-kappa tau))/kappa} (evaluated with
     * {@code expm1}); it is the sensitivity {@code -d ln P / d r} and appears
     * in every option/moment formula.</p>
     *
     * @throws IllegalArgumentException unless {@code tau} is finite and &gt;= 0
     */
    public double bFactor(double tau) {
        if (!Double.isFinite(tau) || tau < 0.0) {
            throw new IllegalArgumentException("tau must be finite and >= 0, got " + tau);
        }
        return -Math.expm1(-kappa * tau) / kappa;
    }

    /**
     * {@code A(t,T)} with {@code tau = T - t} (depends on tau only):
     * {@code A = exp[(theta - sigma^2/(2 kappa^2)) (B - tau) - sigma^2 B^2/(4 kappa)]}.
     *
     * <p>The first term is the risk-adjusted drift contribution of the mean
     * level; the second is the convexity from rate volatility.</p>
     */
    public double aFactor(double tau) {
        double b = bFactor(tau);
        double k = kappa;
        double s = sigma;
        return Math.exp((theta - s * s / (2.0 * k * k)) * (b - tau) - s * s * b * b / (4.0 * k));
    }

    // ----------------------------- ZCB / yields -----------------------

    /** {@code P(0, maturity)} using {@code r0}. */
    public double zcbPrice(double maturity) {
        return zcbPrice(maturity, 0.0, r0);
    }

    /**
     * Zero-coupon bond price {@code P(t, maturity) = A e^(-B r)}.
     *
     * @throws IllegalArgumentException on non-finite inputs, {@code maturity < t},
     *     or when {@code exp(-B r)} would overflow (result not representable)
     */
    public double zcbPrice(double maturity, double t, double r) {
        if (!(Double.isFinite(maturity) && Double.isFinite(t))) {
            throw new IllegalArgumentException("maturity and t must be finite");
        }
        if (maturity < t) {
            throw new IllegalArgumentException(
                    "maturity " + maturity + " before valuation time " + t);
        }
        if (!Double.isFinite(r)) {
            throw new IllegalArgumentException("short rate must be finite, got " + r);
        }
        double tau = maturity - t;
        double expo = -bFactor(tau) * r;
        if (expo > MAX_EXP_ARG) {
            throw new IllegalArgumentException(
                    "ZCB price not representable: exp(" + expo + ") overflows");
        }
        return aFactor(tau) * Math.exp(expo);
    }

    /** Continuously compounded zero yield {@code -ln P(0,T) / T} using {@code r0}. */
    public double zeroYield(double maturity) {
        return zeroYield(maturity, 0.0, r0);
    }

    /**
     * Continuously compounded zero yield {@code -ln P(t,T) / (T - t)}.
     *
     * @throws IllegalArgumentException unless {@code maturity > t}
     */
    public double zeroYield(double maturity, double t, double r) {
        double tau = maturity - t;
        if (!Double.isFinite(tau) || tau <= 0.0) {
            throw new IllegalArgumentException("need maturity > t, got tau=" + tau);
        }
        return -Math.log(zcbPrice(maturity, t, r)) / tau;
    }

    // ----------------------------- r_T distribution -------------------

    /**
     * {@code E[r_T | r_0] = theta + (r_0 - theta) e^(-kappa T)}.
     *
     * @throws IllegalArgumentException on a negative or non-finite horizon
     */
    public double rMean(double horizon) {
        return rMean(horizon, r0);
    }

    /**
     * {@code E[r_T | r_t = r] = theta + (r - theta) e^(-kappa T)} from an
     * explicit short rate {@code r}.
     *
     * @throws IllegalArgumentException on a negative or non-finite horizon or
     *     a non-finite {@code r}
     */
    public double rMean(double horizon, double r) {
        if (!Double.isFinite(horizon) || horizon < 0.0) {
            throw new IllegalArgumentException("horizon must be >= 0, got " + horizon);
        }
        if (!Double.isFinite(r)) {
            throw new IllegalArgumentException("short rate must be finite, got " + r);
        }
        return theta + (r - theta) * Math.exp(-kappa * horizon);
    }

    /**
     * {@code Var[r_T | r_0] = sigma^2 (1 - e^(-2 kappa T)) / (2 kappa)}.
     *
     * @throws IllegalArgumentException on a negative or non-finite horizon
     */
    public double rVar(double horizon) {
        if (!Double.isFinite(horizon) || horizon < 0.0) {
            throw new IllegalArgumentException("horizon must be >= 0, got " + horizon);
        }
        return sigma * sigma * (-Math.expm1(-2.0 * kappa * horizon)) / (2.0 * kappa);
    }

    // ----------------------------- bond options -----------------------

    /**
     * Std dev of ln P(T,S) at option expiry T for bond maturity S:
     * {@code sigma_p = sigma sqrt((1 - e^(-2 kappa T))/(2 kappa)) B(S - T)}.
     */
    private double sigmaP(double expiry, double bondMaturity) {
        return sigma
                * Math.sqrt((-Math.expm1(-2.0 * kappa * expiry)) / (2.0 * kappa))
                * bFactor(bondMaturity - expiry);
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
        double ps = zcbPrice(bondMaturity);
        double pt = zcbPrice(expiry);
        double sp = sigmaP(expiry, bondMaturity);
        if (sp == 0.0) {
            // sigma = 0, expiry = 0, or S = T: option collapses to intrinsic
            // on the forward bond price.
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

    /**
     * European call on a unit ZCB (Jamshidian 1989 closed form):
     * {@code ZBC = P(0,S) N(h) - K P(0,T) N(h - sigma_p)} with
     * {@code h = ln(P(0,S)/(K P(0,T)))/sigma_p + sigma_p/2}.
     */
    public double zbc(double expiry, double bondMaturity, double strike) {
        return zbOption(expiry, bondMaturity, strike, true);
    }

    /**
     * European put on a unit ZCB:
     * {@code ZBP = K P(0,T) N(sigma_p - h) - P(0,S) N(-h)}.
     */
    public double zbp(double expiry, double bondMaturity, double strike) {
        return zbOption(expiry, bondMaturity, strike, false);
    }

    // ----------------------------- Monte Carlo ------------------------

    /**
     * Terminal short rates and pathwise integrals from {@link #simulate}.
     *
     * @param rEnd      {@code r(horizon)} per path
     * @param integralR {@code int_0^horizon r dt} per path
     */
    public record SimPaths(double[] rEnd, double[] integralR) {
    }

    /**
     * Exact-transition simulation of {@code (r_T, int_0^T r dt)}.
     *
     * <p>Writes {@code r = theta + x} with {@code x} a mean-zero OU process
     * and samples {@code (x_next, int x dt)} per step from their exact joint
     * Gaussian (conditional decomposition: two standard-normal draws per
     * step per path, x first), so both outputs are free of discretisation
     * bias. {@code sigma = 0} short-circuits to the deterministic decay.</p>
     *
     * @throws IllegalArgumentException on a non-positive horizon or counts
     */
    public SimPaths simulate(double horizon, int nSteps, int nPaths, long seed) {
        if (!(Double.isFinite(horizon) && horizon > 0.0)) {
            throw new IllegalArgumentException("horizon must be finite and > 0, got " + horizon);
        }
        if (nSteps < 1 || nPaths < 1) {
            throw new IllegalArgumentException("nSteps and nPaths must be >= 1");
        }
        double dt = horizon / nSteps;
        MathUtils.OuStepMoments m = MathUtils.ouStepMoments(kappa, sigma, dt);
        Random rng = new Random(seed);
        double[] rEnd = new double[nPaths];
        double[] integralR = new double[nPaths];
        if (m.varX() > 0.0) {
            double sdX = Math.sqrt(m.varX());
            double beta = m.cov() / m.varX(); // regression of the integral on x_next
            double sdResid = Math.sqrt(Math.max(m.varI() - m.cov() * m.cov() / m.varX(), 0.0));
            for (int p = 0; p < nPaths; p++) {
                double x = r0 - theta;
                double integralX = 0.0;
                for (int s = 0; s < nSteps; s++) {
                    double z1 = rng.nextGaussian();
                    double z2 = rng.nextGaussian();
                    double meanX = x * m.decay();
                    double xNew = meanX + sdX * z1;
                    integralX += x * m.b() + beta * (xNew - meanX) + sdResid * z2;
                    x = xNew;
                }
                rEnd[p] = theta + x;
                integralR[p] = theta * horizon + integralX;
            }
        } else { // sigma = 0: fully deterministic decay
            double x = r0 - theta;
            double integralX = 0.0;
            for (int s = 0; s < nSteps; s++) {
                integralX += x * m.b();
                x *= m.decay();
            }
            java.util.Arrays.fill(rEnd, theta + x);
            java.util.Arrays.fill(integralR, theta * horizon + integralX);
        }
        return new SimPaths(rEnd, integralR);
    }

    /**
     * MC ZCB price {@code E[e^(-int r)]} and its standard error (sample
     * stdev with ddof 1 over sqrt(nPaths); exactly 0 for sigma = 0 or a
     * single path).
     */
    public McEstimate mcZcb(double maturity, int nSteps, int nPaths, long seed) {
        SimPaths sim = simulate(maturity, nSteps, nPaths, seed);
        double[] disc = new double[nPaths];
        double mean = 0.0;
        for (int p = 0; p < nPaths; p++) {
            disc[p] = Math.exp(-sim.integralR()[p]);
            mean += disc[p];
        }
        mean /= nPaths;
        // sigma = 0 is deterministic: the standard error is exactly 0 (avoid
        // spurious 1-ulp noise from summing identical values).
        double se = 0.0;
        if (nPaths > 1 && sigma > 0.0) {
            double ss = 0.0;
            for (double d : disc) {
                ss += (d - mean) * (d - mean);
            }
            se = Math.sqrt(ss / (nPaths - 1)) / Math.sqrt(nPaths);
        }
        return new McEstimate(mean, se);
    }

    // ----------------------------- calibration ------------------------

    /**
     * Least-squares calibration outcome (non-convergence reported, not thrown).
     *
     * <ul>
     *   <li>{@code converged}: the optimiser met its tolerances at the reported
     *       solution <em>and</em> the solution is not at a domain bound.</li>
     *   <li>{@code atBound}: the solution sits at the edge of the calibration
     *       domain ({@code kappa < 1e-5} or {@code > 49.95}, {@code |theta| > 4.995},
     *       {@code sigma < 1e-5} or {@code > 4.995}) — a penalty-boundary
     *       artefact, not a fit.</li>
     *   <li>{@code nStarts}: Nelder-Mead starts run (3 by default; 1 with x0).</li>
     *   <li>{@code kappaSpread} / {@code thetaSpread} / {@code sigmaSpread}: max
     *       minus min of each parameter over the starts whose rmse is within
     *       5 % of the best (0 for a single start).</li>
     *   <li>{@code identified}: {@code nStarts >= 2} and all equivalent starts
     *       agree ({@code kappaSpread <= 1e-2}, {@code thetaSpread <= 1e-3},
     *       {@code sigmaSpread <= 1e-3}). With a single start identifiability
     *       is not assessed and this is false.</li>
     *   <li>{@code sigmaFixed}: true when sigma was supplied, not fitted.</li>
     * </ul>
     * <p>Price on the result only if {@code converged && (identified || sigmaFixed)}.</p>
     *
     * @param kappa       fitted mean-reversion speed
     * @param theta       fitted long-run mean
     * @param sigma       fitted volatility (returned as {@code |sigma|}) or the fixed value
     * @param r0          initial short rate (held fixed)
     * @param rmse        root-mean-square yield error
     * @param iterations  optimizer iterations of the best start
     * @param converged   optimizer converged and not at a bound
     * @param atBound     solution at the edge of the domain
     * @param nStarts     number of starts run
     * @param kappaSpread spread of kappa over equivalent starts
     * @param thetaSpread spread of theta over equivalent starts
     * @param sigmaSpread spread of sigma over equivalent starts
     * @param identified  equivalent starts agree (multi-start only)
     * @param sigmaFixed  sigma held fixed at the supplied value
     */
    public record Calibration(
            double kappa, double theta, double sigma, double r0,
            double rmse, int iterations, boolean converged, boolean atBound, int nStarts,
            double kappaSpread, double thetaSpread, double sigmaSpread, boolean identified,
            boolean sigmaFixed) {

        /** Builds the calibrated model. */
        public Vasicek model() {
            return new Vasicek(kappa, theta, sigma, r0);
        }
    }

    /** One Nelder-Mead start's outcome. */
    private record StartResult(
            double fx, double kappa, double theta, double sigma, int iterations,
            boolean converged) {
    }

    /** {@link #calibrate(double[], double[], double, double[], int, Double)} with the
     *  three default starts, {@code maxiter = 4000} and sigma free. */
    public static Calibration calibrate(double[] maturities, double[] yields, double r0) {
        return calibrate(maturities, yields, r0, null, 4000, null);
    }

    /**
     * Fits {@code (kappa, theta, sigma)} to market zero yields by least squares.
     *
     * <p>Minimises {@code sum_i (y_model(T_i) - y_i)^2} with Nelder-Mead
     * ({@code initialStep = 0.05}); out-of-domain points get a smooth penalty
     * pointing back toward the feasible region. {@code r0} is held fixed.</p>
     * <ul>
     *   <li>{@code x0 == null}: three starts {@code (kappa, mean(y), sigma)} for
     *       {@code (kappa, sigma)} in {@code {(0.2, 0.005), (0.5, 0.01), (1.5, 0.02)}};
     *       the lowest objective is returned with the parameter spreads across
     *       equivalent starts.</li>
     *   <li>{@code x0 = {kappa, theta, sigma}}: a single warm start (exactly 3
     *       finite entries; identifiability is then not assessed).</li>
     *   <li>{@code sigmaFixed != null}: fit only {@code (kappa, theta)} with sigma
     *       held at the given value ({@code >= 0}) — the practitioner workflow;
     *       {@code x0[2]} is ignored.</li>
     * </ul>
     * <p>Yields identify the parameters weakly (the bundled data has two minima
     * with rmse within 2 %); always check {@code converged}, {@code atBound}
     * and {@code identified}. Non-convergence is reported, never thrown.</p>
     *
     * @throws IllegalArgumentException on fewer than 3 points, length
     *     mismatch, non-finite/non-positive inputs, a wrong-length or non-finite
     *     x0, a negative sigmaFixed, or {@code maxiter < 1}
     */
    public static Calibration calibrate(
            double[] maturities, double[] yields, double r0, double[] x0, int maxiter,
            Double sigmaFixed) {
        if (maturities == null || yields == null
                || maturities.length != yields.length || maturities.length < 3) {
            throw new IllegalArgumentException(
                    "need at least 3 (maturity, yield) pairs of equal length");
        }
        for (double t : maturities) {
            if (!Double.isFinite(t) || t <= 0.0) {
                throw new IllegalArgumentException("maturities must be finite and > 0");
            }
        }
        double ySum = 0.0;
        for (double y : yields) {
            if (!Double.isFinite(y)) {
                throw new IllegalArgumentException("yields must be finite");
            }
            ySum += y;
        }
        if (!Double.isFinite(r0)) {
            throw new IllegalArgumentException("r0 must be finite");
        }
        if (maxiter < 1) {
            throw new IllegalArgumentException("maxiter must be >= 1");
        }
        if (x0 != null) {
            if (x0.length != 3) {
                throw new IllegalArgumentException("x0 must have 3 entries (kappa, theta, sigma)");
            }
            for (double v : x0) {
                if (!Double.isFinite(v)) {
                    throw new IllegalArgumentException("x0 must be finite");
                }
            }
        }
        if (sigmaFixed != null && (!Double.isFinite(sigmaFixed) || sigmaFixed < 0.0)) {
            throw new IllegalArgumentException(
                    "sigmaFixed must be finite and >= 0, got " + sigmaFixed);
        }
        final boolean fixed = sigmaFixed != null;
        final double sigmaFix = fixed ? sigmaFixed : Double.NaN;
        final double[] ts = maturities.clone();
        final double[] ys = yields.clone();
        final double r0f = r0;

        ToDoubleFunction<double[]> objective = p -> {
            double kappa = p[0];
            double theta = p[1];
            double sigma = fixed ? sigmaFix : p[2];
            if (kappa <= KAPPA_LO || sigma < 0.0 || kappa > KAPPA_HI
                    || Math.abs(theta) > THETA_MAX || sigma > SIGMA_HI) {
                // Smooth penalty pointing back toward the feasible region.
                return 1e6 * (1.0
                        + Math.max(0.0, KAPPA_LO - kappa)
                        + Math.max(0.0, -sigma)
                        + Math.max(0.0, kappa - KAPPA_HI)
                        + Math.max(0.0, Math.abs(theta) - THETA_MAX)
                        + Math.max(0.0, sigma - SIGMA_HI));
            }
            Vasicek model = new Vasicek(kappa, theta, sigma, r0f);
            double sse = 0.0;
            for (int i = 0; i < ts.length; i++) {
                double d = model.zeroYield(ts[i]) - ys[i];
                sse += d * d;
            }
            return sse;
        };

        double meanY = ySum / ys.length;
        double[][] starts;
        if (x0 != null) {
            starts = new double[][] {x0.clone()};
        } else {
            starts = new double[3][];
            for (int k = 0; k < 3; k++) {
                starts[k] = new double[] {START_KAPPA[k], meanY, START_SIGMA[k]};
            }
        }
        StartResult[] results = new StartResult[starts.length];
        for (int k = 0; k < starts.length; k++) {
            double[] p0 = fixed ? new double[] {starts[k][0], starts[k][1]} : starts[k];
            NelderMead.Result res = NelderMead.minimize(objective, p0, 0.05, 1e-10, 1e-14, maxiter);
            double[] best = res.x();
            double sigma = fixed ? sigmaFix : Math.abs(best[2]);
            results[k] = new StartResult(
                    res.fx(), best[0], best[1], sigma, res.iterations(), res.converged());
        }
        int bestIdx = 0;
        for (int k = 1; k < results.length; k++) {
            if (results[k].fx() < results[bestIdx].fx()) {
                bestIdx = k;
            }
        }
        StartResult b = results[bestIdx];
        double n = ts.length;
        double rmse = Math.sqrt(Math.max(b.fx(), 0.0) / n);
        boolean atBound = b.kappa() < KAPPA_LO_BOUND || b.kappa() > KAPPA_HI_BOUND
                || Math.abs(b.theta()) > THETA_BOUND
                || (!fixed && (b.sigma() < SIGMA_LO_BOUND || b.sigma() > SIGMA_HI_BOUND));
        double kLo = b.kappa();
        double kHi = b.kappa();
        double tLo = b.theta();
        double tHi = b.theta();
        double sLo = b.sigma();
        double sHi = b.sigma();
        for (StartResult r : results) {
            double rRmse = Math.sqrt(Math.max(r.fx(), 0.0) / n);
            if (rRmse <= EQUIV_RMSE_FACTOR * rmse + EQUIV_RMSE_FLOOR) {
                kLo = Math.min(kLo, r.kappa());
                kHi = Math.max(kHi, r.kappa());
                tLo = Math.min(tLo, r.theta());
                tHi = Math.max(tHi, r.theta());
                sLo = Math.min(sLo, r.sigma());
                sHi = Math.max(sHi, r.sigma());
            }
        }
        double kappaSpread = kHi - kLo;
        double thetaSpread = tHi - tLo;
        double sigmaSpread = sHi - sLo;
        boolean identified = results.length >= 2 && kappaSpread <= IDENT_KAPPA
                && thetaSpread <= IDENT_THETA && sigmaSpread <= IDENT_SIGMA;
        return new Calibration(b.kappa(), b.theta(), b.sigma(), r0, rmse, b.iterations(),
                b.converged() && !atBound, atBound, results.length,
                kappaSpread, thetaSpread, sigmaSpread, identified, fixed);
    }
}
