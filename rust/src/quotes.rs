//! CSV loaders for the bundled sample market data (see `../data`).
//!
//! File formats (headers included, plain ASCII, comma-separated):
//!
//! * `curve_quotes.csv`: `curve_id,instrument,start,maturity,rate` where
//!   `instrument` is `deposit`, `fra` or `swap`; `start` is used only for
//!   FRAs (0 otherwise); times in ACT/365F years, rates decimal.
//! * `ois_quotes.csv`: `maturity,rate` (OIS par swap quotes).
//! * `zero_yields.csv`: `maturity,yield` (Vasicek calibration targets).

use std::fs;
use std::path::Path;

use crate::bootstrap::Instrument;
use crate::error::{IrmError, Result};

fn read_lines(path: &Path) -> Result<Vec<String>> {
    let text = fs::read_to_string(path)
        .map_err(|e| IrmError::Data(format!("cannot read {}: {e}", path.display())))?;
    Ok(text
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect())
}

fn parse_f64(field: &str, path: &Path) -> Result<f64> {
    field.trim().parse::<f64>().map_err(|_| {
        IrmError::Data(format!(
            "bad numeric field {field:?} in {}",
            path.display()
        ))
    })
}

/// Load the instruments of one curve from `curve_quotes.csv`, sorted by
/// pillar time.
pub fn load_curve_quotes(path: &Path, curve_id: &str) -> Result<Vec<Instrument>> {
    let mut out: Vec<Instrument> = Vec::new();
    for line in read_lines(path)?.iter().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 5 {
            return Err(IrmError::Data(format!(
                "expected 5 fields in {}, got line {line:?}",
                path.display()
            )));
        }
        if fields[0] != curve_id {
            continue;
        }
        let start = parse_f64(fields[2], path)?;
        let maturity = parse_f64(fields[3], path)?;
        let rate = parse_f64(fields[4], path)?;
        let ins = match fields[1] {
            "deposit" => Instrument::deposit(maturity, rate)?,
            "fra" => Instrument::fra(start, maturity, rate)?,
            "swap" => Instrument::swap(maturity, rate)?,
            other => {
                return Err(IrmError::Data(format!(
                    "unknown instrument type {other:?} in {}",
                    path.display()
                )))
            }
        };
        out.push(ins);
    }
    if out.is_empty() {
        return Err(IrmError::Data(format!(
            "no quotes found for curve_id={curve_id:?} in {}",
            path.display()
        )));
    }
    out.sort_by(|a, b| a.pillar().total_cmp(&b.pillar()));
    Ok(out)
}

/// Load OIS par swap quotes from `ois_quotes.csv`, sorted by maturity.
pub fn load_ois_quotes(path: &Path) -> Result<Vec<Instrument>> {
    let mut out: Vec<Instrument> = Vec::new();
    for line in read_lines(path)?.iter().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 2 {
            return Err(IrmError::Data(format!(
                "expected 2 fields in {}, got line {line:?}",
                path.display()
            )));
        }
        let maturity = parse_f64(fields[0], path)?;
        let rate = parse_f64(fields[1], path)?;
        out.push(Instrument::ois_swap(maturity, rate)?);
    }
    if out.is_empty() {
        return Err(IrmError::Data(format!("no OIS quotes in {}", path.display())));
    }
    out.sort_by(|a, b| a.pillar().total_cmp(&b.pillar()));
    Ok(out)
}

/// Load `(maturities, yields)` Vasicek calibration targets from
/// `zero_yields.csv`.
pub fn load_zero_yields(path: &Path) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut ts = Vec::new();
    let mut ys = Vec::new();
    for line in read_lines(path)?.iter().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 2 {
            return Err(IrmError::Data(format!(
                "expected 2 fields in {}, got line {line:?}",
                path.display()
            )));
        }
        ts.push(parse_f64(fields[0], path)?);
        ys.push(parse_f64(fields[1], path)?);
    }
    if ts.is_empty() {
        return Err(IrmError::Data(format!("no zero yields in {}", path.display())));
    }
    Ok((ts, ys))
}
