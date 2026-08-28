#include "irm/quotes.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace irm {

namespace {

/// Split one CSV line on commas (the bundled files contain no quoting).
std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        // Strip a trailing CR from Windows-style line endings.
        if (!field.empty() && field.back() == '\r') field.pop_back();
        out.push_back(field);
    }
    return out;
}

/// Read all non-empty data rows (header skipped) of a CSV file.
std::vector<std::vector<std::string>> read_rows(const std::string& path) {
    std::ifstream fh(path);
    if (!fh) {
        throw std::invalid_argument("cannot open CSV file: " + path);
    }
    std::vector<std::vector<std::string>> rows;
    std::string line;
    bool header = true;
    while (std::getline(fh, line)) {
        if (header) {
            header = false;
            continue;
        }
        if (line.empty() || line == "\r") continue;
        rows.push_back(split_csv(line));
    }
    return rows;
}

void sort_by_pillar(std::vector<Instrument>& out) {
    std::stable_sort(out.begin(), out.end(), [](const Instrument& a, const Instrument& b) {
        return instrument_pillar(a) < instrument_pillar(b);
    });
}

}  // namespace

std::vector<Instrument> load_curve_quotes(const std::string& path,
                                          const std::string& curve_id) {
    std::vector<Instrument> out;
    for (const auto& row : read_rows(path)) {
        if (row.size() != 5) {
            throw std::invalid_argument("malformed row in " + path);
        }
        if (row[0] != curve_id) continue;
        const std::string& kind = row[1];
        const double start = std::stod(row[2]);
        const double maturity = std::stod(row[3]);
        const double rate = std::stod(row[4]);
        if (kind == "deposit") {
            out.emplace_back(Deposit(maturity, rate));
        } else if (kind == "fra") {
            out.emplace_back(FRA(start, maturity, rate));
        } else if (kind == "swap") {
            out.emplace_back(Swap(maturity, rate));
        } else {
            throw std::invalid_argument("unknown instrument type '" + kind + "' in " + path);
        }
    }
    if (out.empty()) {
        throw std::invalid_argument("no quotes found for curve_id='" + curve_id + "' in " + path);
    }
    sort_by_pillar(out);
    return out;
}

std::vector<Instrument> load_ois_quotes(const std::string& path) {
    std::vector<Instrument> out;
    for (const auto& row : read_rows(path)) {
        if (row.size() != 2) {
            throw std::invalid_argument("malformed row in " + path);
        }
        out.emplace_back(OISSwap(std::stod(row[0]), std::stod(row[1])));
    }
    if (out.empty()) {
        throw std::invalid_argument("no OIS quotes in " + path);
    }
    sort_by_pillar(out);
    return out;
}

std::pair<std::vector<double>, std::vector<double>> load_zero_yields(const std::string& path) {
    std::vector<double> ts;
    std::vector<double> ys;
    for (const auto& row : read_rows(path)) {
        if (row.size() != 2) {
            throw std::invalid_argument("malformed row in " + path);
        }
        ts.push_back(std::stod(row[0]));
        ys.push_back(std::stod(row[1]));
    }
    if (ts.empty()) {
        throw std::invalid_argument("no zero yields in " + path);
    }
    return {ts, ys};
}

}  // namespace irm
