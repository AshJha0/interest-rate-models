package com.quant.irm;

/**
 * A bootstrappable market instrument quoted at par.
 *
 * <p>Each instrument knows its pillar time (the maturity whose discount
 * factor it pins down) and its PV mispricing ("residual") on a trial curve;
 * the bootstrap drives the residual to zero pillar by pillar.</p>
 */
public interface Instrument {

    /** Pillar (maturity) time in years whose DF this instrument determines. */
    double pillar();

    /** Quoted par rate (decimal). */
    double rate();

    /** PV mispricing of the instrument on the given curve (zero at par). */
    double residual(DiscountCurve curve);
}
