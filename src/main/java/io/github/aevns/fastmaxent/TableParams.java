package io.github.aevns.fastmaxent;

/**
 * Tuning parameters for the sampling tables; {@link #DEFAULT} holds the defaults. The priors are given to the sampler
 * separately (see {@link io.github.aevns.fastmaxent.priors}).
 *
 * @param saturationTolerance           how close the supported target range comes, at each end, to a single
 *                                      category holding all the probability mass (smaller values push the end
 *                                      closer, at the cost of a finer row grid near it); must be in the range (0, 1)
 * @param targetRowsPerConstraintPreset number of precomputed rows placed along the primary constraint, for each
 *                                      secondary constraint preset (more rows trade memory and construction time
 *                                      for resolution); minimum 2
 * @param categoryBlockSize             number of categories grouped per block in the shared precomputed category
 *                                      table (larger blocks trade more per-sample work for less memory, roughly
 *                                      1 / categoryBlockSize)
 * @param categoryTableBudgetBytes      memory budget for the shared precomputed category table (acts as an indirect
 *                                      cap on row count)
 * @param betaCandidatesPerSign         number of candidate values per sign considered when placing rows by
 *                                      Fisher-Rao arc length; more candidates trade construction time for finer
 *                                      row placement, not final sampling accuracy; minimum 2
 */
public record TableParams(double saturationTolerance, int targetRowsPerConstraintPreset, int categoryBlockSize,
                          long categoryTableBudgetBytes, int betaCandidatesPerSign) {
    public static final TableParams DEFAULT = new TableParams(1e-5, 256, 8, 20_000_000L, 500);

    /** Validates the parameters; throws {@link IllegalArgumentException} if any is out of range. */
    public TableParams {
        if (!(saturationTolerance > 0 && saturationTolerance < 1))
            throw new IllegalArgumentException("saturation tolerance must be in (0, 1)");
        if (targetRowsPerConstraintPreset < 2)
            throw new IllegalArgumentException("need at least 2 target rows per constraint preset");
        if (categoryBlockSize < 1) throw new IllegalArgumentException("category block size must be at least 1");
        if (categoryTableBudgetBytes < 1) throw new IllegalArgumentException("category table budget must be positive");
        if (betaCandidatesPerSign < 2) throw new IllegalArgumentException("need at least 2 beta candidates per sign");
    }

}
