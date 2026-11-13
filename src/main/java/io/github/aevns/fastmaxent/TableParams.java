package io.github.aevns.fastmaxent;

/**
 * Tuning parameters for the sampling tables; {@link #DEFAULT} holds the defaults. The priors are given to the sampler
 * separately (see {@link io.github.aevns.fastmaxent.priors}).
 *
 * @param saturationTolerance           the extreme value group must hold at least 1 - tolerance of the mass at the
 *                                      clamp; in (0, 1)
 * @param targetRowsPerExpectedQuantity stored target-s rows per expected quantity n; at least 2
 * @param categoryBlockSize             block size of the coarse category CDF (memory ~ 1 / block size); at least 1
 * @param categoryTableBudgetBytes      budget for the shared category CDF table, which sets its row count; positive
 * @param betaCandidatesPerSign         candidate betas per sign used to place target rows by Fisher arc length;
 *                                      at least 2
 */
public record TableParams(double saturationTolerance, int targetRowsPerExpectedQuantity, int categoryBlockSize,
                          long categoryTableBudgetBytes, int betaCandidatesPerSign) {
    public static final TableParams DEFAULT = new TableParams(1e-5, 256, 8, 20_000_000L, 3000);

    public TableParams {
        if (!(saturationTolerance > 0 && saturationTolerance < 1))
            throw new IllegalArgumentException("saturation tolerance must be in (0, 1)");
        if (targetRowsPerExpectedQuantity < 2)
            throw new IllegalArgumentException("need at least 2 target rows per expected quantity");
        if (categoryBlockSize < 1) throw new IllegalArgumentException("category block size must be at least 1");
        if (categoryTableBudgetBytes < 1) throw new IllegalArgumentException("category table budget must be positive");
        if (betaCandidatesPerSign < 2) throw new IllegalArgumentException("need at least 2 beta candidates per sign");
    }

}
