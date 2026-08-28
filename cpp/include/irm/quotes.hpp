#ifndef IRM_QUOTES_HPP
#define IRM_QUOTES_HPP

/// \file quotes.hpp
/// \brief CSV loaders for the bundled sample market data (see ../data).
///
/// File formats (headers included, plain ASCII, comma-separated):
///
///  * `curve_quotes.csv`: `curve_id,instrument,start,maturity,rate` where
///    `instrument` is `deposit`, `fra` or `swap`; `start` is used only for
///    FRAs (0 otherwise); times in ACT/365F years, rates decimal.
///  * `ois_quotes.csv`: `maturity,rate` (OIS par swap quotes).
///  * `zero_yields.csv`: `maturity,yield` (Vasicek calibration targets).

#include <string>
#include <utility>
#include <vector>

#include "irm/bootstrap.hpp"

namespace irm {

/// Load the instruments of one curve from `curve_quotes.csv`, sorted by
/// pillar time.
/// \throws std::invalid_argument if the file cannot be opened, contains an
///   unknown instrument type, or holds no quotes for \p curve_id.
std::vector<Instrument> load_curve_quotes(const std::string& path,
                                          const std::string& curve_id);

/// Load OIS par swap quotes from `ois_quotes.csv`, sorted by maturity.
std::vector<Instrument> load_ois_quotes(const std::string& path);

/// Load (maturities, yields) calibration targets from `zero_yields.csv`.
std::pair<std::vector<double>, std::vector<double>> load_zero_yields(
    const std::string& path);

}  // namespace irm

#endif  // IRM_QUOTES_HPP
