package com.quant.irm;

/**
 * Monte Carlo estimate with its standard error.
 *
 * @param price         sample-mean price estimate
 * @param standardError sample standard deviation (ddof 1) / sqrt(nPaths);
 *                      exactly 0 for deterministic (sigma = 0) runs
 */
public record McEstimate(double price, double standardError) {
}
