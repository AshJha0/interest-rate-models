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
 * {@code (kappa, theta, sigma)} to market zero yields by least squares with
 * the native {@link NelderMead}.</p>
 */
public final class Vasicek {

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
     * <p>{@code B = (1 - e^(-kappa tau))/kappa}; it is the sensitivity
     * {@code -d ln P / d r} and appears in every option/moment formula.</p>
     *
     * @throws IllegalArgumentException unless {@code tau} is finite and &gt;= 0
     */
    public double bFactor(double tau) {
        if (!Double.isFinite(tau) || tau < 0.0) {
            throw new IllegalArgumentException("tau must be finite and >= 0, got " + tau);
        }
        return (1.0 - Math.exp(-kappa * tau)) / kappa;
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
     * @throws IllegalArgumentException if {@code maturity < t}
     */
    public double zcbPrice(double maturity, double t, double r) {
        if (maturity < t) {
            throw new IllegalArgumentException(
                    "maturity " + maturity + " before valuation time " + t);
        }
        double tau = maturity - t;
        return aFactor(tau) * Math.exp(-bFactor(tau) * r);
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
        if (tau <= 0.0) {
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
        if (!Double.isFinite(horizon) || horizon < 0.0) {
            throw new IllegalArgumentException("horizon must be >= 0, got " + horizon);
        }
        return theta + (r0 - theta) * Math.exp(-kappa * horizon);
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
        return sigma * sigma * (1.0 - Math.exp(-2.0 * kappa * horizon)) / (2.0 * kappa);
    }

    // ----------------------------- bond options -----------------------

    /**
     * Std dev of ln P(T,S) at option expiry T for bond maturity S:
     * {@code sigma_p = sigma sqrt((1 - e^(-2 kappa T))/(2 kappa)) B(S - T)}.
     */
    private double sigmaP(double expiry, double bondMaturity) {
        return sigma
                * Math.sqrt((1.0 - Math.exp(-2.0 * kappa * expiry)) / (2.0 * kappa))
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

    /** MC ZCB price {@code E[e^(-int r)]} and its standard error. */
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
     * @param kappa      fitted mean-reversion speed
     * @param theta      fitted long-run mean
     * @param sigma      fitted volatility (returned as {@code |sigma|})
     * @param r0         initial short rate (held fixed)
     * @param rmse       root-mean-square yield error
     * @param iterations optimizer iterations
     * @param converged  optimizer convergence flag
     */
    public record Calibration(
            double kappa, double theta, double sigma, double r0,
            double rmse, int iterations, boolean converged) {

        /** Builds the calibrated model. */
        public Vasicek model() {
            return new Vasicek(kappa, theta, sigma, r0);
        }
    }

    /** {@link #calibrate(double[], double[], double, double[], int)} with the
     *  default start {@code (0.5, mean(yields), 0.01)} and {@code maxiter = 4000}. */
    public static Calibration calibrate(double[] maturities, double[] yields, double r0) {
        return calibrate(maturities, yields, r0, null, 4000);
    }

    /**
     * Fits {@code (kappa, theta, sigma)} to market zero yields by least squares.
     *
     * <p>Minimises {@code sum_i (y_model(T_i) - y_i)^2} with Nelder-Mead
     * ({@code initialStep = 0.05}); out-of-domain points get a smooth penalty
     * pointing back toward the feasible region. {@code r0} is held fixed;
     * non-convergence is reported via the result flag, never thrown.</p>
     *
     * @param x0 optional start point {@code (kappa, theta, sigma)}; when null
     *     the default {@code (0.5, mean(yields), 0.01)} is used
     * @throws IllegalArgumentException on fewer than 3 points, length
     *     mismatch, or non-finite/non-positive inputs
     */
    public static Calibration calibrate(
            double[] maturities, double[] yields, double r0, double[] x0, int maxiter) {
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
        final double[] ts = maturities.clone();
        final double[] ys = yields.clone();
        final double r0f = r0;

        ToDoubleFunction<double[]> objective = p -> {
            double kappa = p[0];
            double theta = p[1];
            double sigma = p[2];
            if (kappa <= 1e-6 || sigma < 0.0 || kappa > 50.0
                    || Math.abs(theta) > 5.0 || sigma > 5.0) {
                // Smooth penalty pointing back toward the feasible region.
                return 1e6 * (1.0
                        + Math.max(0.0, 1e-6 - kappa)
                        + Math.max(0.0, -sigma)
                        + Math.max(0.0, kappa - 50.0)
                        + Math.max(0.0, Math.abs(theta) - 5.0)
                        + Math.max(0.0, sigma - 5.0));
            }
            Vasicek model = new Vasicek(kappa, theta, sigma, r0f);
            double sse = 0.0;
            for (int i = 0; i < ts.length; i++) {
                double d = model.zeroYield(ts[i]) - ys[i];
                sse += d * d;
            }
            return sse;
        };

        double[] start = (x0 != null) ? x0.clone()
                : new double[] {0.5, ySum / ys.length, 0.01};
        NelderMead.Result res =
                NelderMead.minimize(objective, start, 0.05, 1e-10, 1e-14, maxiter);
        double[] best = res.x();
        double rmse = Math.sqrt(Math.max(res.fx(), 0.0) / ts.length);
        return new Calibration(
                best[0], best[1], Math.abs(best[2]), r0, rmse, res.iterations(), res.converged());
    }
}
