package com.quant.irm;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * CSV loaders for the bundled sample market data (see {@code ../data}).
 *
 * <p>File formats (headers included, plain ASCII, comma-separated):</p>
 * <ul>
 *   <li>{@code curve_quotes.csv}: {@code curve_id,instrument,start,maturity,rate}
 *       where {@code instrument} is {@code deposit}, {@code fra} or
 *       {@code swap}; {@code start} is used only for FRAs (0 otherwise);
 *       times in ACT/365F years, rates decimal.</li>
 *   <li>{@code ois_quotes.csv}: {@code maturity,rate} (OIS par swap quotes).</li>
 *   <li>{@code zero_yields.csv}: {@code maturity,yield} (Vasicek calibration
 *       targets).</li>
 * </ul>
 */
public final class Quotes {

    private Quotes() {
    }

    private static List<String[]> readCsv(Path path) {
        List<String> lines;
        try {
            lines = Files.readAllLines(path);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + path, e);
        }
        List<String[]> rows = new ArrayList<>();
        for (int i = 1; i < lines.size(); i++) { // skip header
            String line = lines.get(i).trim();
            if (!line.isEmpty()) {
                rows.add(line.split(","));
            }
        }
        return rows;
    }

    /**
     * Loads the instruments of one curve, sorted by pillar time.
     *
     * @throws IllegalArgumentException on an unknown instrument type or when
     *     no quotes match the curve id
     */
    public static List<Instrument> loadCurveQuotes(Path path, String curveId) {
        List<Instrument> out = new ArrayList<>();
        for (String[] row : readCsv(path)) {
            if (!row[0].equals(curveId)) {
                continue;
            }
            String kind = row[1];
            double start = Double.parseDouble(row[2]);
            double maturity = Double.parseDouble(row[3]);
            double rate = Double.parseDouble(row[4]);
            switch (kind) {
                case "deposit" -> out.add(new Deposit(maturity, rate));
                case "fra" -> out.add(new Fra(start, maturity, rate));
                case "swap" -> out.add(new Swap(maturity, rate));
                default -> throw new IllegalArgumentException(
                        "unknown instrument type '" + kind + "' in " + path);
            }
        }
        if (out.isEmpty()) {
            throw new IllegalArgumentException(
                    "no quotes found for curve_id='" + curveId + "' in " + path);
        }
        out.sort(Comparator.comparingDouble(Instrument::pillar));
        return out;
    }

    /**
     * Loads OIS par swap quotes, sorted by maturity.
     *
     * @throws IllegalArgumentException when the file holds no quotes
     */
    public static List<OisSwap> loadOisQuotes(Path path) {
        List<OisSwap> out = new ArrayList<>();
        for (String[] row : readCsv(path)) {
            out.add(new OisSwap(Double.parseDouble(row[0]), Double.parseDouble(row[1])));
        }
        if (out.isEmpty()) {
            throw new IllegalArgumentException("no OIS quotes in " + path);
        }
        out.sort(Comparator.comparingDouble(OisSwap::pillar));
        return out;
    }

    /**
     * Loads (maturities, yields) calibration targets.
     *
     * @return a 2-row array: {@code [maturities, yields]}
     * @throws IllegalArgumentException when the file holds no rows
     */
    public static double[][] loadZeroYields(Path path) {
        List<String[]> rows = readCsv(path);
        if (rows.isEmpty()) {
            throw new IllegalArgumentException("no zero yields in " + path);
        }
        double[] ts = new double[rows.size()];
        double[] ys = new double[rows.size()];
        for (int i = 0; i < rows.size(); i++) {
            ts[i] = Double.parseDouble(rows.get(i)[0]);
            ys[i] = Double.parseDouble(rows.get(i)[1]);
        }
        return new double[][] {ts, ys};
    }
}
