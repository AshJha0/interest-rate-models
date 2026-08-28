#include "irm/rootfind.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace irm {

double brentq(const std::function<double(double)>& f, double a, double b,
              double xtol, double rtol, int maxiter) {
    if (!(std::isfinite(a) && std::isfinite(b))) {
        throw std::invalid_argument("brentq: bracket endpoints must be finite");
    }
    if (!(xtol > 0.0)) {
        throw std::invalid_argument("brentq: xtol must be positive");
    }
    double fa = f(a);
    double fb = f(b);
    if (fa == 0.0) return a;
    if (fb == 0.0) return b;
    if ((fa > 0.0) == (fb > 0.0)) {
        throw std::invalid_argument("brentq: root not bracketed: f(" + std::to_string(a) +
                                    ")=" + std::to_string(fa) + ", f(" + std::to_string(b) +
                                    ")=" + std::to_string(fb));
    }
    double c = a, fc = fa;
    double d = b - a, e = b - a;
    for (int it = 0; it < maxiter; ++it) {
        if ((fb > 0.0) == (fc > 0.0)) {
            // Rename so that the root lies between b and c.
            c = a;
            fc = fa;
            d = e = b - a;
        }
        if (std::fabs(fc) < std::fabs(fb)) {
            const double old_b = b, old_fb = fb;
            a = b;
            fa = fb;
            b = c;
            fb = fc;
            c = old_b;
            fc = old_fb;
        }
        const double tol1 = 2.0 * rtol * std::fabs(b) + 0.5 * xtol;
        const double xm = 0.5 * (c - b);
        if (std::fabs(xm) <= tol1 || fb == 0.0) {
            return b;
        }
        if (std::fabs(e) >= tol1 && std::fabs(fa) > std::fabs(fb)) {
            // Attempt inverse quadratic interpolation (secant if a == c).
            const double s = fb / fa;
            double p, q;
            if (a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                q = fa / fc;
                const double r = fb / fc;
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q;
            p = std::fabs(p);
            if (2.0 * p < std::min(3.0 * xm * q - std::fabs(tol1 * q), std::fabs(e * q))) {
                e = d;  // accept interpolation
                d = p / q;
            } else {
                d = xm;  // interpolation failed; bisect
                e = d;
            }
        } else {
            d = xm;  // bounds decreasing too slowly; bisect
            e = d;
        }
        a = b;
        fa = fb;
        if (std::fabs(d) > tol1) {
            b += d;
        } else {
            b += (xm > 0.0) ? tol1 : -tol1;
        }
        fb = f(b);
    }
    throw std::domain_error("brentq: no convergence after " + std::to_string(maxiter) +
                            " iterations");
}

double bisect(const std::function<double(double)>& f, double a, double b,
              double xtol, int maxiter) {
    if (!(std::isfinite(a) && std::isfinite(b))) {
        throw std::invalid_argument("bisect: bracket endpoints must be finite");
    }
    const double fa = f(a);
    const double fb = f(b);
    if (fa == 0.0) return a;
    if (fb == 0.0) return b;
    if ((fa > 0.0) == (fb > 0.0)) {
        throw std::invalid_argument("bisect: root not bracketed: f(" + std::to_string(a) +
                                    ")=" + std::to_string(fa) + ", f(" + std::to_string(b) +
                                    ")=" + std::to_string(fb));
    }
    double lo = (a < b) ? a : b;
    double hi = (a < b) ? b : a;
    double flo = (a < b) ? fa : fb;
    for (int it = 0; it < maxiter; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double fm = f(mid);
        if (fm == 0.0 || 0.5 * (hi - lo) < xtol) {
            return mid;
        }
        if ((fm > 0.0) == (flo > 0.0)) {
            lo = mid;
            flo = fm;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

}  // namespace irm
