//! Crate-wide error type.
//!
//! Per the shared conventions, invalid input never panics: every fallible
//! public function returns `Result<_, IrmError>` with a descriptive message.
//! Calibration *non-convergence* is not an error — it is reported through a
//! `converged` flag on the calibration result.

use std::fmt;

/// Errors produced by the `irm` crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrmError {
    /// An argument violated its domain (non-finite value, bad ordering,
    /// non-positive discount factor, ...).
    InvalidInput(String),
    /// A root finder failed: the root was not bracketed or the iteration
    /// budget was exhausted.
    RootFind(String),
    /// The bootstrap could not find an admissible positive discount factor
    /// for a pillar (crossed / arbitrageable quotes).
    Bootstrap(String),
    /// A bundled data file could not be read or parsed.
    Data(String),
}

impl fmt::Display for IrmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrmError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            IrmError::RootFind(msg) => write!(f, "root finding failed: {msg}"),
            IrmError::Bootstrap(msg) => write!(f, "bootstrap failed: {msg}"),
            IrmError::Data(msg) => write!(f, "data error: {msg}"),
        }
    }
}

impl std::error::Error for IrmError {}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, IrmError>;
