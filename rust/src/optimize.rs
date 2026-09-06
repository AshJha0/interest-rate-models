//! Native Nelder-Mead simplex minimiser.
//!
//! Used by the Vasicek least-squares calibration.  Standard coefficients:
//! reflection 1, expansion 2, contraction 0.5, shrink 0.5.
//! Non-convergence is *reported* via `converged = false` in the result,
//! never returned as an error — callers decide how to react (per the
//! shared conventions: report, don't crash).

use crate::error::{IrmError, Result};

/// Outcome of a Nelder-Mead minimisation.
#[derive(Debug, Clone, PartialEq)]
pub struct NelderMeadResult {
    /// Best vertex found.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub fx: f64,
    /// Iterations performed.
    pub iterations: usize,
    /// Whether the simplex met both tolerance criteria.
    pub converged: bool,
}

/// Minimise `f` starting from `x0`.
///
/// The initial simplex is `x0` plus `initial_step` added to each coordinate
/// in turn.  Converged when both the simplex diameter is below `xtol`
/// (1e-10) and the function-value spread below `ftol` (1e-14).
pub fn nelder_mead<F>(
    mut f: F,
    x0: &[f64],
    initial_step: f64,
    xtol: f64,
    ftol: f64,
    maxiter: usize,
) -> Result<NelderMeadResult>
where
    F: FnMut(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(IrmError::InvalidInput("x0 must be non-empty".into()));
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(IrmError::InvalidInput("x0 must be finite".into()));
    }

    let mut pts: Vec<Vec<f64>> = vec![x0.to_vec()];
    for i in 0..n {
        let mut p = x0.to_vec();
        p[i] += initial_step;
        pts.push(p);
    }
    let mut vals: Vec<f64> = pts.iter().map(|p| f(p)).collect();

    let order = |pts: &mut Vec<Vec<f64>>, vals: &mut Vec<f64>| {
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&i, &j| vals[i].total_cmp(&vals[j]));
        *pts = idx.iter().map(|&k| pts[k].clone()).collect();
        *vals = idx.iter().map(|&k| vals[k]).collect();
    };

    order(&mut pts, &mut vals);
    let mut it = 0;
    let mut converged = false;
    while it < maxiter {
        it += 1;
        let spread_f = (vals[n] - vals[0]).abs();
        let spread_x = (1..=n)
            .map(|j| {
                (0..n)
                    .map(|i| (pts[j][i] - pts[0][i]).abs())
                    .fold(0.0_f64, f64::max)
            })
            .fold(0.0_f64, f64::max);
        if spread_f <= ftol && spread_x <= xtol {
            converged = true;
            break;
        }

        let centroid: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| pts[j][i]).sum::<f64>() / n as f64)
            .collect();
        let worst = pts[n].clone();
        let refl: Vec<f64> = (0..n).map(|i| centroid[i] + (centroid[i] - worst[i])).collect();
        let f_refl = f(&refl);

        if f_refl < vals[0] {
            // Try expanding further along the reflection direction.
            let exp_pt: Vec<f64> = (0..n)
                .map(|i| centroid[i] + 2.0 * (centroid[i] - worst[i]))
                .collect();
            let f_exp = f(&exp_pt);
            if f_exp < f_refl {
                pts[n] = exp_pt;
                vals[n] = f_exp;
            } else {
                pts[n] = refl;
                vals[n] = f_refl;
            }
        } else if f_refl < vals[n - 1] {
            pts[n] = refl;
            vals[n] = f_refl;
        } else {
            // Contract toward the centroid (outside if reflection helped a bit).
            let contr: Vec<f64> = if f_refl < vals[n] {
                (0..n)
                    .map(|i| centroid[i] + 0.5 * (refl[i] - centroid[i]))
                    .collect()
            } else {
                (0..n)
                    .map(|i| centroid[i] + 0.5 * (worst[i] - centroid[i]))
                    .collect()
            };
            let f_contr = f(&contr);
            if f_contr < f_refl.min(vals[n]) {
                pts[n] = contr;
                vals[n] = f_contr;
            } else {
                // Shrink the whole simplex toward the best vertex.
                for j in 1..=n {
                    let best = pts[0].clone();
                    for i in 0..n {
                        pts[j][i] = best[i] + 0.5 * (pts[j][i] - best[i]);
                    }
                    vals[j] = f(&pts[j]);
                }
            }
        }
        order(&mut pts, &mut vals);
    }

    Ok(NelderMeadResult {
        x: pts[0].clone(),
        fx: vals[0],
        iterations: it,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimises_shifted_quadratic() {
        let res = nelder_mead(
            |p| (p[0] - 1.5).powi(2) + 3.0 * (p[1] + 0.5).powi(2),
            &[0.0, 0.0],
            0.1,
            1e-10,
            1e-14,
            5000,
        )
        .unwrap();
        assert!(res.converged);
        assert!((res.x[0] - 1.5).abs() < 1e-6);
        assert!((res.x[1] + 0.5).abs() < 1e-6);
        assert!(res.fx < 1e-12);
    }

    #[test]
    fn minimises_rosenbrock() {
        let res = nelder_mead(
            |p| (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0] * p[0]).powi(2),
            &[-1.2, 1.0],
            0.1,
            1e-10,
            1e-14,
            5000,
        )
        .unwrap();
        assert!(res.converged);
        assert!((res.x[0] - 1.0).abs() < 1e-5);
        assert!((res.x[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn non_convergence_reported_not_error() {
        let res = nelder_mead(|p| p[0] * p[0], &[5.0], 0.1, 1e-10, 1e-14, 2).unwrap();
        assert!(!res.converged);
        assert_eq!(res.iterations, 2);
    }

    #[test]
    fn rejects_bad_start() {
        assert!(nelder_mead(|_| 0.0, &[], 0.1, 1e-10, 1e-14, 10).is_err());
        assert!(nelder_mead(|p| p[0], &[f64::NAN], 0.1, 1e-10, 1e-14, 10).is_err());
    }
}
