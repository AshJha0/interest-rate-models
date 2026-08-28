// Golden-value suite: every case in ../data/golden/golden.json is
// recomputed through the public API and must agree within its stored
// tolerance.  The dispatcher mirrors python/tests/test_golden.py — the
// executable definition of each case name (see API_SPEC.md section 9).

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "irm/irm.hpp"
#include "mini_json.hpp"

namespace {

const std::string kDataDir = IRM_DATA_DIR;

const mini_json::Value& golden() {
    static mini_json::ValuePtr root = [] {
        std::ifstream fh(kDataDir + "/golden/golden.json");
        if (!fh) throw std::runtime_error("cannot open golden.json");
        std::stringstream ss;
        ss << fh.rdbuf();
        return mini_json::parse(ss.str());
    }();
    return *root;
}

/// Rebuild a curve from flat keys t1..tn / df1..dfn.
irm::DiscountCurve curve_from_inputs(const mini_json::Value& in) {
    std::vector<double> ts, dfs;
    for (int i = 1; in.has("df" + std::to_string(i)); ++i) {
        ts.push_back(in.at("t" + std::to_string(i)).as_number());
        dfs.push_back(in.at("df" + std::to_string(i)).as_number());
    }
    return irm::DiscountCurve(ts, dfs);
}

irm::Vasicek vasicek_from_inputs(const mini_json::Value& in) {
    return irm::Vasicek(in.at("kappa").as_number(), in.at("theta").as_number(),
                        in.at("sigma").as_number(), in.at("r0").as_number());
}

irm::HullWhite hw_from_inputs(const mini_json::Value& in) {
    return irm::HullWhite(in.at("a").as_number(), in.at("sigma").as_number(),
                          curve_from_inputs(in));
}

const irm::DiscountCurve& boot(const std::string& curve_id) {
    static std::map<std::string, irm::DiscountCurve> cache;
    auto it = cache.find(curve_id);
    if (it == cache.end()) {
        irm::DiscountCurve c =
            (curve_id == "OIS")
                ? irm::bootstrap(irm::load_ois_quotes(kDataDir + "/ois_quotes.csv"))
                : irm::bootstrap(
                      irm::load_curve_quotes(kDataDir + "/curve_quotes.csv", curve_id));
        it = cache.emplace(curve_id, std::move(c)).first;
    }
    return it->second;
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// Compute the expected quantities for one golden case.
std::map<std::string, double> evaluate(const std::string& name, const mini_json::Value& in) {
    if (starts_with(name, "curve_")) {
        const irm::DiscountCurve c = curve_from_inputs(in);
        if (name == "curve_df_pillar" || name == "curve_df_mid" || name == "curve_df_extrap") {
            return {{"df", c.df(in.at("t").as_number())}};
        }
        if (name == "curve_zero_mid") {
            return {{"zero", c.zero_rate(in.at("t").as_number())}};
        }
        if (name == "curve_fwd_simple") {
            return {{"fwd", c.fwd_rate(in.at("ta").as_number(), in.at("tb").as_number())}};
        }
    }
    if (starts_with(name, "boot_")) {
        const irm::DiscountCurve& c = boot(in.at("curve_id").as_string());
        if (ends_with(name, "_par_10y")) {
            std::vector<double> times = {0.0};
            for (int i = 1; i <= 10; ++i) times.push_back(static_cast<double>(i));
            return {{"par", irm::par_swap_rate(c, times)}};
        }
        if (name.find("zero") != std::string::npos) {
            return {{"zero", c.zero_rate(in.at("t").as_number())}};
        }
        return {{"df", c.df(in.at("t").as_number())}};
    }
    if (starts_with(name, "vas_")) {
        const irm::Vasicek m = vasicek_from_inputs(in);
        if (name == "vas_rT_mean") return {{"mean", m.r_mean(in.at("T").as_number())}};
        if (name == "vas_rT_var") return {{"var", m.r_var(in.at("T").as_number())}};
        if (name == "vas_zbc") {
            return {{"price", m.zbc(in.at("expiry").as_number(),
                                    in.at("bond_maturity").as_number(),
                                    in.at("strike").as_number())}};
        }
        if (name == "vas_zbp") {
            return {{"price", m.zbp(in.at("expiry").as_number(),
                                    in.at("bond_maturity").as_number(),
                                    in.at("strike").as_number())}};
        }
        if (name == "vas_mc_zcb") {
            const int n_paths = static_cast<int>(in.at("min_paths").as_number());
            const auto pr = m.mc_zcb(in.at("T").as_number(), /*n_steps=*/8, n_paths,
                                     /*seed=*/901);
            return {{"price", pr.first}};
        }
        return {{"price", m.zcb_price(in.at("T").as_number())}};
    }
    if (starts_with(name, "hw_")) {
        const irm::HullWhite hw = hw_from_inputs(in);
        if (name == "hw_zcb_t0") {
            return {{"price", hw.zcb_price(0.0, in.at("T").as_number())}};
        }
        if (name == "hw_caplet" || name == "hw_caplet_sigma0") {
            return {{"price", hw.caplet(in.at("reset").as_number(), in.at("pay").as_number(),
                                        in.at("strike").as_number(),
                                        in.at("notional").as_number())}};
        }
        if (name == "hw_floorlet") {
            return {{"price",
                     hw.floorlet(in.at("reset").as_number(), in.at("pay").as_number(),
                                 in.at("strike").as_number(), in.at("notional").as_number())}};
        }
        if (name == "hw_cap_3y") {
            const std::vector<double> sched = {in.at("t0").as_number(), in.at("t1").as_number(),
                                               in.at("t2").as_number(), in.at("t3").as_number()};
            return {{"price",
                     hw.cap(sched, in.at("strike").as_number(), in.at("notional").as_number())}};
        }
        if (name == "hw_swaption_payer" || name == "hw_swaption_neg_curve") {
            const double expiry = in.at("expiry").as_number();
            const int n = static_cast<int>(std::lround(in.at("tenor_years").as_number()));
            std::vector<double> pay_times;
            for (int i = 1; i <= n; ++i) pay_times.push_back(expiry + i);
            const irm::JamshidianResult res = hw.jamshidian_swaption(
                expiry, pay_times, in.at("fixed_rate").as_number(),
                in.at("notional").as_number());
            return {{"price", res.value}};
        }
        if (name == "hw_mc_caplet") {
            const int n_paths = static_cast<int>(in.at("min_paths").as_number());
            const auto pr = hw.mc_caplet(in.at("reset").as_number(), in.at("pay").as_number(),
                                         in.at("strike").as_number(),
                                         in.at("notional").as_number(),
                                         /*n_steps=*/8, n_paths, /*seed=*/902);
            return {{"price", pr.first}};
        }
    }
    ADD_FAILURE() << "golden dispatcher has no rule for case '" << name << "'";
    return {};
}

TEST(Golden, AllCasesWithinTolerance) {
    const auto& cases = golden().at("cases").as_array();
    ASSERT_GE(cases.size(), 20u);
    std::vector<std::string> names;
    for (const auto& cp : cases) {
        const mini_json::Value& c = *cp;
        const std::string name = c.at("name").as_string();
        names.push_back(name);
        const double tol = c.at("tol").as_number();
        const std::map<std::string, double> got = evaluate(name, c.at("inputs"));
        for (const auto& [key, expected_ptr] : c.at("expect").obj) {
            const double expected = expected_ptr->as_number();
            ASSERT_TRUE(got.count(key)) << name << ": missing output " << key;
            EXPECT_NEAR(got.at(key), expected, tol)
                << name << "." << key << ": got " << got.at(key) << ", want " << expected
                << " +- " << tol;
        }
    }
    // No duplicate case names.
    std::vector<std::string> sorted = names;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_TRUE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end())
        << "duplicate golden case names";
}

}  // namespace
