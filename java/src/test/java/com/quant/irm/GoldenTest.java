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

/**
 * Golden-value suite: every case in ../data/golden/golden.json is recomputed
 * through the public API and must agree within its stored tolerance. The
 * dispatcher mirrors the Python reference test one-to-one (see API_SPEC.md
 * sections 8-9); bootstrap cases load quotes from ../data/*.csv.
 */
public class GoldenTest {

    private static final Path DATA_DIR = Paths.get("..", "data");

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

    private final Map<String, DiscountCurve> bootCache = new HashMap<>();

    private DiscountCurve boot(String curveId) {
        return bootCache.computeIfAbsent(curveId, id -> {
            if (id.equals("OIS")) {
                return Bootstrap.bootstrap(Quotes.loadOisQuotes(DATA_DIR.resolve("ois_quotes.csv")));
            }
            return Bootstrap.bootstrap(
                    Quotes.loadCurveQuotes(DATA_DIR.resolve("curve_quotes.csv"), id));
        });
    }

    /** Compute the expected quantities for one golden case. */
    private Map<String, Double> evaluate(String name, Map<String, Object> inp) {
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
                case "hw_caplet", "hw_caplet_sigma0" -> out.put("price", hw.caplet(
                        num(inp, "reset"), num(inp, "pay"), num(inp, "strike"),
                        num(inp, "notional")));
                case "hw_floorlet" -> out.put("price", hw.floorlet(
                        num(inp, "reset"), num(inp, "pay"), num(inp, "strike"),
                        num(inp, "notional")));
                case "hw_cap_3y" -> {
                    double[] sched = {num(inp, "t0"), num(inp, "t1"), num(inp, "t2"),
                            num(inp, "t3")};
                    out.put("price", hw.cap(sched, num(inp, "strike"), num(inp, "notional")));
                }
                case "hw_swaption_payer", "hw_swaption_neg_curve" -> {
                    double expiry = num(inp, "expiry");
                    int n = (int) Math.round(num(inp, "tenor_years"));
                    double[] payTimes = new double[n];
                    for (int i = 1; i <= n; i++) {
                        payTimes[i - 1] = expiry + i;
                    }
                    HullWhite.JamshidianResult res = hw.jamshidianSwaption(
                            expiry, payTimes, num(inp, "fixed_rate"), num(inp, "notional"), true);
                    out.put("price", res.value());
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
    public void allGoldenCasesMatchWithinTolerance() throws IOException {
        List<Map<String, Object>> cases = loadCases();
        for (Map<String, Object> c : cases) {
            String name = (String) c.get("name");
            Map<String, Object> inputs = (Map<String, Object>) c.get("inputs");
            Map<String, Object> expect = (Map<String, Object>) c.get("expect");
            double tol = (Double) c.get("tol");
            Map<String, Double> got = evaluate(name, inputs);
            for (Map.Entry<String, Object> e : expect.entrySet()) {
                assertTrue(name + ": missing output " + e.getKey(),
                        got.containsKey(e.getKey()));
                double expected = (Double) e.getValue();
                assertEquals(name + "." + e.getKey(), expected, got.get(e.getKey()), tol);
            }
        }
    }

    @Test
    public void goldenFileHasExpectedCaseCount() throws IOException {
        List<Map<String, Object>> cases = loadCases();
        assertTrue("expected >= 20 golden cases, got " + cases.size(), cases.size() >= 20);
        Set<String> names = new HashSet<>();
        for (Map<String, Object> c : cases) {
            assertTrue("duplicate golden case name", names.add((String) c.get("name")));
        }
    }
}
