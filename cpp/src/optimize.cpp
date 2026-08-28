#include "irm/optimize.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace irm {

NelderMeadResult nelder_mead(
    const std::function<double(const std::vector<double>&)>& f,
    const std::vector<double>& x0, double initial_step, double xtol,
    double ftol, int maxiter) {
    const std::size_t n = x0.size();
    if (n == 0) {
        throw std::invalid_argument("nelder_mead: x0 must be non-empty");
    }
    for (double v : x0) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument("nelder_mead: x0 must be finite");
        }
    }

    // Initial simplex: x0 plus initial_step added to each coordinate in turn.
    std::vector<std::vector<double>> pts;
    pts.reserve(n + 1);
    pts.push_back(x0);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> p = x0;
        p[i] += initial_step;
        pts.push_back(std::move(p));
    }
    std::vector<double> vals(n + 1);
    for (std::size_t j = 0; j <= n; ++j) vals[j] = f(pts[j]);

    const auto order = [&]() {
        std::vector<std::size_t> idx(n + 1);
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        std::stable_sort(idx.begin(), idx.end(),
                         [&](std::size_t p, std::size_t q) { return vals[p] < vals[q]; });
        std::vector<std::vector<double>> new_pts(n + 1);
        std::vector<double> new_vals(n + 1);
        for (std::size_t j = 0; j <= n; ++j) {
            new_pts[j] = std::move(pts[idx[j]]);
            new_vals[j] = vals[idx[j]];
        }
        pts = std::move(new_pts);
        vals = std::move(new_vals);
    };

    order();
    int it = 0;
    bool converged = false;
    while (it < maxiter) {
        ++it;
        const double spread_f = std::fabs(vals[n] - vals[0]);
        double spread_x = 0.0;
        for (std::size_t j = 1; j <= n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                spread_x = std::max(spread_x, std::fabs(pts[j][i] - pts[0][i]));
            }
        }
        if (spread_f <= ftol && spread_x <= xtol) {
            converged = true;
            break;
        }

        // Centroid of all vertices except the worst.
        std::vector<double> centroid(n, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) centroid[i] += pts[j][i];
        }
        for (std::size_t i = 0; i < n; ++i) centroid[i] /= static_cast<double>(n);

        const std::vector<double>& worst = pts[n];
        std::vector<double> refl(n);
        for (std::size_t i = 0; i < n; ++i) refl[i] = centroid[i] + (centroid[i] - worst[i]);
        const double f_refl = f(refl);

        if (f_refl < vals[0]) {
            // Try expanding further along the reflection direction.
            std::vector<double> exp_pt(n);
            for (std::size_t i = 0; i < n; ++i) {
                exp_pt[i] = centroid[i] + 2.0 * (centroid[i] - worst[i]);
            }
            const double f_exp = f(exp_pt);
            if (f_exp < f_refl) {
                pts[n] = std::move(exp_pt);
                vals[n] = f_exp;
            } else {
                pts[n] = std::move(refl);
                vals[n] = f_refl;
            }
        } else if (f_refl < vals[n - 1]) {
            pts[n] = std::move(refl);
            vals[n] = f_refl;
        } else {
            // Contract toward the centroid (outside if reflection helped a bit).
            std::vector<double> contr(n);
            if (f_refl < vals[n]) {
                for (std::size_t i = 0; i < n; ++i) {
                    contr[i] = centroid[i] + 0.5 * (refl[i] - centroid[i]);
                }
            } else {
                for (std::size_t i = 0; i < n; ++i) {
                    contr[i] = centroid[i] + 0.5 * (worst[i] - centroid[i]);
                }
            }
            const double f_contr = f(contr);
            if (f_contr < std::min(f_refl, vals[n])) {
                pts[n] = std::move(contr);
                vals[n] = f_contr;
            } else {
                // Shrink the whole simplex toward the best vertex.
                for (std::size_t j = 1; j <= n; ++j) {
                    for (std::size_t i = 0; i < n; ++i) {
                        pts[j][i] = pts[0][i] + 0.5 * (pts[j][i] - pts[0][i]);
                    }
                    vals[j] = f(pts[j]);
                }
            }
        }
        order();
    }

    return NelderMeadResult{pts[0], vals[0], it, converged};
}

}  // namespace irm
