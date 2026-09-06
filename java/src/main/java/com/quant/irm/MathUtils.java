package com.quant.irm;

/**
 * Numerical kernels shared by the models.
 *
 * <p>Contains the standard normal CDF built on Cody's rational-minimax
 * {@code erf}/{@code erfc} (netlib CALERF, relative error ~1e-16 — far
 * better than the 1e-12 absolute accuracy the API spec requires), the exact
 * Ornstein-Uhlenbeck one-step transition moments used by both the Vasicek
 * and Hull-White Monte Carlo engines, and the cancellation-free
 * integrated-OU variance kernel shared with the Hull-White pathwise
 * discount factor.</p>
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

    /** Standard normal density {@code phi(x) = exp(-x^2/2) / sqrt(2 pi)}. */
    public static double normPdf(double x) {
        return Math.exp(-0.5 * x * x) / Math.sqrt(2.0 * Math.PI);
    }

    /**
     * Below this value of {@code x = a * dt} the integrated-OU variance uses
     * its power series instead of the closed form.
     */
    public static final double OU_SERIES_THRESHOLD = 1e-2;

    // Coefficients of  x - 2(1 - e^{-x}) + (1 - e^{-2x})/2  =  x^3 * sum_k c_k x^k,
    // c_k = (-1)^(k+3) (2 - 2^(k+2)) / (k+3)!  for k = 0, 1, ...
    private static final double[] OU_SERIES = {
        1.0 / 3.0, -1.0 / 4.0, 7.0 / 60.0, -1.0 / 24.0, 31.0 / 2520.0, -1.0 / 320.0};

    private static void checkOuInputs(double a, double sigma, double dt) {
        if (!(a > 0.0) || !Double.isFinite(a)) {
            throw new IllegalArgumentException(
                    "mean reversion must be positive and finite, got " + a);
        }
        if (!(dt > 0.0) || !Double.isFinite(dt)) {
            throw new IllegalArgumentException("time step must be positive and finite, got " + dt);
        }
        if (!Double.isFinite(sigma)) {
            throw new IllegalArgumentException("sigma must be finite, got " + sigma);
        }
    }

    /**
     * {@code Var[int_0^dt x(s) ds | x(0)]} for {@code dx = -a x dt + sigma dW}.
     *
     * <p>Closed form {@code (sigma^2/a^2)(dt - 2b + (1 - e^(-2 a dt))/(2a))}
     * with {@code b = (1 - e^(-a dt))/a}. The three O(dt) terms cancel to an
     * O(a^2 dt^3) result, which loses all precision for small {@code a*dt}
     * (80x too large at {@code a*dt = 1e-4} in plain double arithmetic).
     * With {@code x = a*dt}:</p>
     * <ul>
     *   <li>{@code x < OU_SERIES_THRESHOLD}: the series
     *       {@code sigma^2 dt^3 (1/3 - x/4 + 7x^2/60 - x^3/24 + 31x^4/2520 - x^5/320)}
     *       (relative truncation error below 3e-15 at the threshold);</li>
     *   <li>otherwise the closed form written with {@code expm1} (relative
     *       rounding error below ~1e-11 for {@code x >= 1e-2}).</li>
     * </ul>
     * <p>This is also {@code 2 V(t)} of the Hull-White model with {@code dt = t}.</p>
     *
     * @throws IllegalArgumentException unless {@code a > 0}, {@code dt > 0}
     *     (finite) and {@code sigma} finite
     */
    public static double ouIntegralVariance(double a, double sigma, double dt) {
        checkOuInputs(a, sigma, dt);
        double x = a * dt;
        double s2 = sigma * sigma;
        if (x < OU_SERIES_THRESHOLD) {
            double poly = 0.0;
            for (int k = OU_SERIES.length - 1; k >= 0; k--) {
                poly = poly * x + OU_SERIES[k];
            }
            return s2 * dt * dt * dt * poly;
        }
        double bracket = dt + 2.0 * Math.expm1(-x) / a + (-Math.expm1(-2.0 * x)) / (2.0 * a);
        return Math.max((s2 / (a * a)) * bracket, 0.0);
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
     * size. {@code 1 - decay} terms use {@code expm1} and {@code varI} uses
     * {@link #ouIntegralVariance}, so the moments stay accurate down to
     * {@code a*dt ~ 1e-12}.</p>
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
     * @throws IllegalArgumentException if {@code a <= 0}, {@code dt <= 0},
     *     {@code sigma < 0} or any input is non-finite
     */
    public static OuStepMoments ouStepMoments(double a, double sigma, double dt) {
        checkOuInputs(a, sigma, dt);
        if (sigma < 0.0) {
            throw new IllegalArgumentException("sigma must be >= 0, got " + sigma);
        }
        double x = a * dt;
        double decay = Math.exp(-x);
        double oneMinusDecay = -Math.expm1(-x);
        double b = oneMinusDecay / a;
        double s2 = sigma * sigma;
        double varX = s2 * (-Math.expm1(-2.0 * x)) / (2.0 * a);
        double varI = ouIntegralVariance(a, sigma, dt);
        double cov = (s2 / (2.0 * a * a)) * oneMinusDecay * oneMinusDecay;
        return new OuStepMoments(decay, b, varX, varI, cov);
    }
}
