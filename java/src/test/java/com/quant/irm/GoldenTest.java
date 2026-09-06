package com.quant.irm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/**
 * Golden-value suite: every case in ../data/golden/golden.json is recomputed
 * through the public API and must agree within its stored tolerance. The
 * dispatcher mirrors the Python reference test one-to-one (see API_SPEC.md
 * sections 8-9); bootstrap cases load quotes from ../data/*.csv. Each case
 * is its own parameterised test, so a single bad case cannot mask the others.
 */
@RunWith(Parameterized.class)
public class GoldenTest {

    private static final Path DATA_DIR = Paths.get("..", "data");
    private static final int EXPECTED_CASES = 33;

    /** Parameterised over every golden case name. */
    @Parameters(name = "{0}")
    public static List<Object[]> cases() throws IOException {
        List<Object[]> out = new ArrayList<>();
        for (Map<String, Object> c : loadCases()) {
            out.add(new Object[] {(String) c.get("name"), c});
        }
        return out;
    }

    private final String caseName;
    private final Map<String, Object> theCase;

    /**
     * One golden case.
     *
     * @param caseName case name (test label)
     * @param theCase  parsed case object
     */
    public GoldenTest(String caseName, Map<String, Object> theCase) {
        this.caseName = caseName;
        this.theCase = theCase;
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> loadCases() throws IOException {
        String text = Files.readString(DATA_DIR.resolve("golden").resolve("golden.json"));
        Map<String, Object> root = (Map<String, Object>) Json.parse(text);
        List<Map<String, Object>> cases = new ArrayList<>();
        for (Object c : (List<Object>) root.get("cases")) {
            cases.add((Map<String, Object>) c);
        }
        return cases;
    }

    private static double num(Map<String, Object> m, String key) {
        Object v = m.get(key);
        assertTrue("missing input " + key, v instanceof Double);
        return (Double) v;
    }

    /** Rebuild a curve from flat keys t1..tn / df1..dfn. */
    private static DiscountCurve curveFromInputs(Map<String, Object> inp) {
        List<Double> ts = new ArrayList<>();
        List<Double> dfs = new ArrayList<>();
        for (int i = 1; inp.containsKey("df" + i); i++) {
            ts.add(num(inp, "t" + i));
            dfs.add(num(inp, "df" + i));
        }
        double[] t = ts.stream().mapToDouble(Double::doubleValue).toArray();
        double[] p = dfs.stream().mapToDouble(Double::doubleValue).toArray();
        return new DiscountCurve(t, p);
    }

    private static final Map<String, DiscountCurve> BOOT_CACHE = new HashMap<>();

    private static synchronized DiscountCurve boot(String curveId) {
        return BOOT_CACHE.computeIfAbsent(curveId, id -> {
            if (id.equals("OIS")) {
                return Bootstrap.bootstrap(Quotes.loadOisQuotes(DATA_DIR.resolve("ois_quotes.csv")));
            }
            return Bootstrap.bootstrap(
                    Quotes.loadCurveQuotes(DATA_DIR.resolve("curve_quotes.csv"), id));
        });
    }

    /** Compute the expected quantities for one golden case. */
    private static Map<String, Double> evaluate(String name, Map<String, Object> inp) {
        Map<String, Double> out = new HashMap<>();
        if (name.startsWith("curve_")) {
            DiscountCurve c = curveFromInputs(inp);
            switch (name) {
                case "curve_df_pillar", "curve_df_mid", "curve_df_extrap" ->
                        out.put("df", c.df(num(inp, "t")));
                case "curve_zero_mid" -> out.put("zero", c.zeroRate(num(inp, "t")));
                case "curve_fwd_simple" ->
                        out.put("fwd", c.fwdRate(num(inp, "ta"), num(inp, "tb")));
                default -> fail("no rule for " + name);
            }
            return out;
        }
        if (name.startsWith("boot_")) {
            DiscountCurve c = boot((String) inp.get("curve_id"));
            if (name.endsWith("_par_10y")) {
                double[] times = new double[11];
                for (int i = 0; i <= 10; i++) {
                    times[i] = i;
                }
                out.put("par", DiscountCurve.parSwapRate(c, times));
            } else if (name.contains("zero")) {
                out.put("zero", c.zeroRate(num(inp, "t")));
            } else {
                out.put("df", c.df(num(inp, "t")));
            }
            return out;
        }
        if (name.equals("vas_calib_sigma_fixed")) {
            double[][] zy = Quotes.loadZeroYields(DATA_DIR.resolve("zero_yields.csv"));
            Vasicek.Calibration cal = Vasicek.calibrate(
                    zy[0], zy[1], num(inp, "r0"), null, 4000, num(inp, "sigma_fixed"));
            assertTrue(cal.converged() && cal.identified() && !cal.atBound());
            out.put("kappa", cal.kappa());
            out.put("theta", cal.theta());
            return out;
        }
        if (name.startsWith("bach_")) {
            if (name.equals("bach_implied_vol_payer")) {
                out.put("vol", Bachelier.impliedVol(num(inp, "price"), num(inp, "forward"),
                        num(inp, "strike"), num(inp, "expiry"), num(inp, "annuity"), true));
            } else {
                out.put("price", Bachelier.price(num(inp, "forward"), num(inp, "strike"),
                        num(inp, "expiry"), num(inp, "vol"), num(inp, "annuity"), true));
            }
            return out;
        }
        if (name.equals("hw_calib_recover")) {
            DiscountCurve curve = curveFromInputs(inp);
            List<HullWhite.CapletQuote> caps = new ArrayList<>();
            for (int i = 1; inp.containsKey("cap_price_" + i); i++) {
                caps.add(new HullWhite.CapletQuote(num(inp, "cap_reset_" + i),
                        num(inp, "cap_pay_" + i), num(inp, "strike"), num(inp, "cap_price_" + i)));
            }
            double e = num(inp, "swpt_expiry");
            int n = (int) Math.round(num(inp, "swpt_tenor_years"));
            double[] pays = new double[n];
            for (int k = 1; k <= n; k++) {
                pays[k - 1] = e + k;
            }
            List<HullWhite.SwaptionQuote> swps = List.of(new HullWhite.SwaptionQuote(
                    e, pays, num(inp, "swpt_fixed_rate"), num(inp, "swpt_price")));
            HullWhite.Calibration cal = HullWhite.calibrate(curve, caps, swps, null, null, 4000);
            assertTrue(cal.converged() && cal.identified() && !cal.atBound());
            out.put("a", cal.a());
            out.put("sigma", cal.sigma());
            return out;
        }
        if (name.startsWith("vas_")) {
            Vasicek m = new Vasicek(
                    num(inp, "kappa"), num(inp, "theta"), num(inp, "sigma"), num(inp, "r0"));
            switch (name) {
                case "vas_rT_mean" -> out.put("mean", m.rMean(num(inp, "T")));
                case "vas_rT_var" -> out.put("var", m.rVar(num(inp, "T")));
                case "vas_zbc" -> out.put("price",
                        m.zbc(num(inp, "expiry"), num(inp, "bond_maturity"), num(inp, "strike")));
                case "vas_zbp" -> out.put("price",
                        m.zbp(num(inp, "expiry"), num(inp, "bond_maturity"), num(inp, "strike")));
                case "vas_mc_zcb" -> out.put("price",
                        m.mcZcb(num(inp, "T"), 8, (int) num(inp, "min_paths"), 901L).price());
                default -> out.put("price", m.zcbPrice(num(inp, "T")));
            }
            return out;
        }
        if (name.startsWith("hw_")) {
            HullWhite hw = new HullWhite(num(inp, "a"), num(inp, "sigma"), curveFromInputs(inp));
            switch (name) {
                case "hw_zcb_t0" -> out.put("price", hw.zcbPrice(0.0, num(inp, "T")));
                case "hw_zcb_t1_explicit_r" -> out.put("price",
                        hw.zcbPrice(num(inp, "t"), num(inp, "T"), num(inp, "r")));
                case "hw_caplet", "hw_caplet_sigma0" -> out.put("price", hw.caplet(
                        num(inp, "reset"), num(inp, "pay"), num(inp, "strike"),
                        num(inp, "notional")));
                case "hw_floorlet" -> out.put("price", hw.floorlet(
                        num(inp, "reset"), num(inp, "pay"), num(inp, "strike"),
                        num(inp, "notional")));
                case "hw_cap_3y" -> {
                    // Schedule keys are s0..s3 (t1..t5 are the curve pillars).
                    double[] sched = {num(inp, "s0"), num(inp, "s1"), num(inp, "s2"),
                            num(inp, "s3")};
                    out.put("price", hw.cap(sched, num(inp, "strike"), num(inp, "notional")));
                }
                case "hw_swaption_payer", "hw_swaption_neg_curve", "hw_swaption_receiver" -> {
                    double expiry = num(inp, "expiry");
                    int n = (int) Math.round(num(inp, "tenor_years"));
                    double[] payTimes = new double[n];
                    for (int i = 1; i <= n; i++) {
                        payTimes[i - 1] = expiry + i;
                    }
                    HullWhite.JamshidianResult res = hw.jamshidianSwaption(
                            expiry, payTimes, num(inp, "fixed_rate"), num(inp, "notional"),
                            !name.equals("hw_swaption_receiver"));
                    out.put("price", res.value());
                    out.put("r_star", res.rStar());
                    double[] strikes = res.strikes();
                    for (int i = 0; i < strikes.length; i++) {
                        out.put("strike_" + (i + 1), strikes[i]);
                    }
                }
                case "hw_mc_caplet" -> out.put("price", hw.mcCaplet(
                        num(inp, "reset"), num(inp, "pay"), num(inp, "strike"),
                        num(inp, "notional"), 8, (int) num(inp, "min_paths"), 902L).price());
                default -> fail("no rule for " + name);
            }
            return out;
        }
        fail("golden dispatcher has no rule for case '" + name + "'");
        return out; // unreachable
    }

    @Test
    @SuppressWarnings("unchecked")
    public void caseMatchesWithinTolerance() {
        Map<String, Object> inputs = (Map<String, Object>) theCase.get("inputs");
        Map<String, Object> expect = (Map<String, Object>) theCase.get("expect");
        double tol = (Double) theCase.get("tol");
        Map<String, Double> got = evaluate(caseName, inputs);
        for (Map.Entry<String, Object> e : expect.entrySet()) {
            assertTrue(caseName + ": missing output " + e.getKey(),
                    got.containsKey(e.getKey()));
            double expected = (Double) e.getValue();
            assertEquals(caseName + "." + e.getKey(), expected, got.get(e.getKey()), tol);
        }
    }

    @Test
    public void goldenFileHasExpectedCaseCount() throws IOException {
        List<Map<String, Object>> cases = loadCases();
        assertEquals("expected exactly " + EXPECTED_CASES + " golden cases",
                EXPECTED_CASES, cases.size());
        Set<String> names = new HashSet<>();
        for (Map<String, Object> c : cases) {
            assertTrue("duplicate golden case name", names.add((String) c.get("name")));
        }
    }
}
