package io.github.aevns.fastmaxent;

import java.util.random.RandomGenerator;

/**
 * Draws a category and a quantity from the maximum-entropy distribution relative to a prior, with E[N] = n and
 * E[N V] = s for quantity N and category value V. n is one of a set fixed at construction; s can be any value in the
 * supported range. An optional null outcome draws neither a quantity nor a category, with its own prior weight,
 * independent of whether 0 is itself one of the allowed quantities.
 * <p>
 * With category prior w, quantity prior m and null-outcome weight m_null: for category c with value v_c, and
 * quantity index i with value q_i,
 *   p(null) ~ m_null,   p(c, i) ~ w_c m_i exp(-lambda q_i - beta q_i v_c)
 * <p>
 * Immutable once initialized; sampling is thread-safe if each caller supplies its own {@link RandomGenerator}.
 * <p>
 * For a single, deterministic quantity (no count distribution, no null outcome), where only E[V] = target needs to
 * be met, see {@link ValueSampler} instead: it skips the quantity side entirely and builds correspondingly
 * faster.
 */
public final class JointSampler {
    /**
     * Result of a draw.
     *  Indices correspond to the values and quantities arrays being sampled from.
     * */
    public static final class Sample {
        /** Index into the values array, or -1 for the null outcome. */
        public final int categoryIndex;
        /** Index into the quantities array, or -1 for the null outcome. */
        public final int quantityIndex;

        Sample(int categoryIndex, int quantityIndex) {
            this.categoryIndex = categoryIndex;
            this.quantityIndex = quantityIndex;
        }
    }

    private final UniqueValueGroups groups;
    private final double valueScale, inverseValueScale;
    private final CategoryTable categoryTable;
    private final int nullStateIndex;
    private final QuantityTable[] quantityTables;

    /**
     * Builds the sampling tables.
     *
     * @param categoryValues                the value of each category; finite, sorted ascending, at least two
     *                                      distinct values present
     * @param categoryPriorWeights          prior weight of each category; strictly positive
     * @param quantities                    the allowed quantities (if 0 is included, it is an ordinary quantity,
     *                                      not the null outcome); finite
     * @param quantityPriorWeights          prior weight of each quantity; strictly positive
     * @param nullStateWeight               prior weight of the null outcome (no quantity, no category), on the same
     *                                      scale as quantityPriorWeights; 0 to disallow it
     * @param supportedExpectedQuantities   the supported secondary-constraint presets for expected quantity; each
     *                                      strictly between the smallest and largest allowed quantity (treating the
     *                                      null outcome, if allowed, as a quantity of 0)
     * @param params                        table tuning parameters
     */
    public JointSampler(double[] categoryValues, double[] categoryPriorWeights, double[] quantities,
                         double[] quantityPriorWeights, double nullStateWeight, double[] supportedExpectedQuantities,
                         TableParams params) {
        if (categoryValues.length == 0) throw new IllegalArgumentException("no categoryValues");
        if (supportedExpectedQuantities.length == 0) throw new IllegalArgumentException("no expected quantities");
        if (!(nullStateWeight >= 0) || Double.isInfinite(nullStateWeight))
            throw new IllegalArgumentException("null-state weight must be non-negative and finite");
        boolean hasNullState = nullStateWeight > 0;
        QuantitySet quantitySet = new QuantitySet(quantities, hasNullState);
        for (double expectedQuantity : supportedExpectedQuantities) quantitySet.requireReachable(expectedQuantity);
        if (quantityPriorWeights.length != quantities.length)
            throw new IllegalArgumentException("one quantity weight per quantity required");
        nullStateIndex = quantitySet.nullStateIndex;

        groups = new UniqueValueGroups(categoryValues, categoryPriorWeights);
        valueScale = groups.entryAbsoluteMean;
        inverseValueScale = 1.0 / valueScale;
        double[] scaledValues = new double[groups.uniqueCount];
        for (int i = 0; i < groups.uniqueCount; i++) scaledValues[i] = groups.uniqueValues[i] * inverseValueScale;
        CategoryFamily categoryFamily = new CategoryFamily(scaledValues, groups.uniqueWeights, params);
        categoryTable = categoryFamily.table();

        double[] logQuantityPrior = new double[quantitySet.stateCount];
        for (int state = 0; state < quantityPriorWeights.length; state++) {
            double weight = quantityPriorWeights[state];
            if (!(weight > 0) || Double.isInfinite(weight))
                throw new IllegalArgumentException("quantity weights must be positive and finite");
            logQuantityPrior[state] = Math.log(weight);
        }
        if (hasNullState) logQuantityPrior[nullStateIndex] = Math.log(nullStateWeight);
        quantityTables = new QuantityTable[supportedExpectedQuantities.length];
        for (int i = 0; i < quantityTables.length; i++)
            quantityTables[i] = QuantityTableBuilder.build(supportedExpectedQuantities[i], quantitySet, categoryFamily,
                    logQuantityPrior, params);
    }

    /**
     * Draws a sample at target s (clamped to the supported range for the given n).
     *
     * @param expectedQuantityIndex index into the supportedExpectedQuantities array given at construction
     */
    public Sample sample(int expectedQuantityIndex, double s, RandomGenerator rng) {
        long stateAndCategoryRow =
                quantityTables[expectedQuantityIndex].drawStateAndCategoryRow(s * inverseValueScale, rng);
        int state = (int) (stateAndCategoryRow >>> 32);
        if (state == nullStateIndex) return new Sample(-1, -1);
        int uniqueIndex = categoryTable.drawUniqueIndex((int) stateAndCategoryRow, rng.nextLong());
        return new Sample(groups.drawEntryIndex(uniqueIndex, rng), state);
    }

    /** Lower end of the clamp interval for s, for the given expected-quantity index. */
    public double minTargetS(int expectedQuantityIndex) {
        return quantityTables[expectedQuantityIndex].minScaledTarget() * valueScale;
    }

    /** Upper end of the clamp interval for s, for the given expected-quantity index. */
    public double maxTargetS(int expectedQuantityIndex) {
        return quantityTables[expectedQuantityIndex].maxScaledTarget() * valueScale;
    }

    public long approximateMemoryBytes() {
        long bytes = categoryTable.approximateMemoryBytes() + groups.approximateMemoryBytes();
        for (QuantityTable table : quantityTables) bytes += table.approximateMemoryBytes();
        return bytes;
    }
}
