package com.quant.irm;

/**
 * Numerical kernels shared by the models.
 *
 * <p>Contains the standard normal CDF built on Cody's rational-minimax
 * {@code erf}/{@code erfc} (netlib CALERF, relative error ~1e-16 — far
 * better than the 1e-12 absolute accuracy the API spec requires), and the
 * exact Ornstein-Uhlenbeck one-step transition moments used by both the
 * Vasicek and Hull-White Monte Carlo engines.</p>
 */
public final class MathUtils {

    private static final double SQRT_2 = Math.sqrt(2.0);

    private MathUtils() {
    }

    // ------------------------------------------------------------------
    // Cody's erf / erfc (netlib CALERF coefficients).
    // ------------------------------------------------------------------

    private static final double[] ERF_A = {
        3.16112374387056560e0, 1.13864154151050156e2, 3.77485237685302021e2,
        3.20937758913846947e3, 1.85777706184603153e-1};
    private static final double[] ERF_B = {
        2.36012909523441209e1, 2.44024637934444173e2, 1.28261652607737228e3,
        2.84423683343917062e3};
    private static final double[] ERF_C = {
        5.64188496988670089e-1, 8.88314979438837594e0, 6.61191906371416295e1,
        2.98635138197400131e2, 8.81952221241769090e2, 1.71204761263407058e3,
        2.05107837782607147e3, 1.23033935479799725e3, 2.15311535474403846e-8};
    private static final double[] ERF_D = {
        1.57449261107098347e1, 1.17693950891312499e2, 5.37181101862009858e2,
        1.62138957456669019e3, 3.29079923573345963e3, 4.36261909014324716e3,
        3.43936767414372164e3, 1.23033935480374942e3};
    private static final double[] ERF_P = {
        3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1,
        1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2};
    private static final double[] ERF_Q = {
        2.56852019228982242e0, 1.87295284992346047e0, 5.27905102951428412e-1,
        6.05183413124413191e-2, 2.33520497626869185e-3};
    private static final double ONE_OVER_SQRT_PI = 5.6418958354775628695e-1;
    private static final double ERF_THRESH = 0.46875;

    /** erfc(y) for y &gt; ERF_THRESH (rational approximation branches of CALERF). */
    private static double erfcTail(double y) {
        double result;
        if (y <= 4.0) {
            double xnum = ERF_C[8] * y;
            double xden = y;
            for (int i = 0; i < 7; i++) {
                xnum = (xnum + ERF_C[i]) * y;
                xden = (xden + ERF_D[i]) * y;
            }
            result = (xnum + ERF_C[7]) / (xden + ERF_D[7]);
        } else {
            double ysq = 1.0 / (y * y);
            double xnum = ERF_P[5] * ysq;
            double xden = ysq;
            for (int i = 0; i < 4; i++) {
                xnum = (xnum + ERF_P[i]) * ysq;
                xden = (xden + ERF_Q[i]) * ysq;
            }
            result = ysq * (xnum + ERF_P[4]) / (xden + ERF_Q[4]);
            result = (ONE_OVER_SQRT_PI - result) / y;
        }
        // exp(-y^2) split as exp(-ysq^2) exp(-del) for accuracy (Cody's trick).
        double ysq = Math.floor(y * 16.0) / 16.0;
        double del = (y - ysq) * (y + ysq);
        return Math.exp(-ysq * ysq) * Math.exp(-del) * result;
    }

    /** Error function, relative error ~1e-16 (Cody 1969). */
    public static double erf(double x) {
        double y = Math.abs(x);
        if (y <= ERF_THRESH) {
            double ysq = y * y;
            double xnum = ERF_A[4] * ysq;
            double xden = ysq;
            for (int i = 0; i < 3; i++) {
                xnum = (xnum + ERF_A[i]) * ysq;
                xden = (xden + ERF_B[i]) * ysq;
            }
            return x * (xnum + ERF_A[3]) / (xden + ERF_B[3]);
        }
        double v = 1.0 - erfcTail(y);
        return x >= 0.0 ? v : -v;
    }

    /** Complementary error function {@code erfc(x) = 1 - erf(x)}, tail-accurate. */
    public static double erfc(double x) {
        double y = Math.abs(x);
        if (y <= ERF_THRESH) {
            return 1.0 - erf(x);
        }
        double v = erfcTail(y);
        return x >= 0.0 ? v : 2.0 - v;
    }

    /**
     * Standard normal CDF {@code N(x) = 0.5 * erfc(-x / sqrt(2))}.
     *
     * <p>{@code erfc} is used instead of {@code erf} for better relative
     * accuracy in the far tails (large negative {@code x}).</p>
     */
    public static double normCdf(double x) {
        return 0.5 * erfc(-x / SQRT_2);
    }

    // ------------------------------------------------------------------
    // Exact Ornstein-Uhlenbeck one-step transition moments.
    // ------------------------------------------------------------------

    /**
     * Exact one-step transition moments of a mean-zero OU process
     * {@code dx = -a x dt + sigma dW} over a step of length {@code dt},
     * conditional on the current state {@code x} (writing
     * {@code decay = exp(-a dt)}, {@code b = (1 - decay)/a}):
     *
     * <pre>
     * x_next ~ N(x * decay, varX)   varX = sigma^2 (1 - decay^2) / (2a)
     * I = int x ds ~ N(x * b, varI) varI = (sigma^2/a^2)(dt - 2b + (1 - decay^2)/(2a))
     * Cov(x_next, I) = cov          cov  = (sigma^2/(2a^2)) (1 - decay)^2
     * </pre>
     *
     * <p>Sampling the pair from this bivariate Gaussian gives a bias-free
     * discretisation of both the short rate and its time integral, which is
     * what makes the MC zero-coupon-bond estimators unbiased at any step
     * size.</p>
     *
     * @param decay {@code exp(-a dt)}
     * @param b     {@code (1 - decay)/a}
     * @param varX  variance of {@code x_next}
     * @param varI  variance of the step integral
     * @param cov   covariance of {@code (x_next, I)}
     */
    public record OuStepMoments(double decay, double b, double varX, double varI, double cov) {
    }

    /**
     * Computes {@link OuStepMoments} for mean reversion {@code a > 0},
     * volatility {@code sigma >= 0} and step {@code dt > 0}.
     *
     * @throws IllegalArgumentException if {@code a <= 0} or {@code dt <= 0}
     */
    public static OuStepMoments ouStepMoments(double a, double sigma, double dt) {
        if (a <= 0.0) {
            throw new IllegalArgumentException("mean reversion must be positive, got " + a);
        }
        if (dt <= 0.0) {
            throw new IllegalArgumentException("time step must be positive, got " + dt);
        }
        double decay = Math.exp(-a * dt);
        double b = (1.0 - decay) / a;
        double s2 = sigma * sigma;
        double varX = s2 * (1.0 - decay * decay) / (2.0 * a);
        double varI = (s2 / (a * a)) * (dt - 2.0 * b + (1.0 - decay * decay) / (2.0 * a));
        double cov = (s2 / (2.0 * a * a)) * (1.0 - decay) * (1.0 - decay);
        // Guard tiny negative values from floating-point cancellation.
        return new OuStepMoments(decay, b, Math.max(varX, 0.0), Math.max(varI, 0.0), cov);
    }
}
