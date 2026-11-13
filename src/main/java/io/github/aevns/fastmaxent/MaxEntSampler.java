package io.github.aevns.fastmaxent;

import java.util.random.RandomGenerator;

/**
 * Draws a category and a quantity from the maximum-entropy distribution relative to a prior, with E[N] = n and
 * E[N V] = s for quantity N and category value V. n is one of a set fixed at construction; s can be any value in the
 * supported range. Quantity 0, if allowed, is a null state with no category.
 *
 * With category prior w and quantity prior m:
 *   p(null) ~ m(0),   p(c, q) ~ w_c m(q) exp(-lambda q - beta q v_c)   (q != 0)
 */
public final class MaxEntSampler {
    /** Result of a draw. */
    public static final class Sample {
        /** Index into the values array, or -1 for the null state. */
        public final int categoryIndex;
        /** Index into the quantities array. */
        public final int quantityIndex;
        /** The drawn quantity, quantities[quantityIndex]; 0 for the null state. */
        public final double quantity;

        Sample(int categoryIndex, int quantityIndex, double quantity) {
            this.categoryIndex = categoryIndex;
            this.quantityIndex = quantityIndex;
            this.quantity = quantity;
        }
    }

    private final UniqueValueGroups groups;
    private final double valueScale, inverseValueScale;
    private final CategoryTable categoryTable;
    private final double[] quantityValues;
    private final int nullStateIndex;
    private final QuantityTable[] quantityTables;

    /**
     * Builds the sampling tables. Indices in a {@link Sample} refer to positions in the arrays as given.
     *
     * @param values               the value of each category; positive, sorted ascending, at least two distinct
     * @param categoryPriorWeights prior weight of each category; positive, any scale
     * @param quantities           allowed quantities; distinct, any order, may be negative or non-integer; include 0
     *                             to allow the null state
     * @param quantityPriorWeights prior weight of each quantity; positive, any scale
     * @param expectedQuantities   the supported n, each strictly between the smallest and largest quantity
     * @param params               table tuning parameters
     */
    public MaxEntSampler(double[] values, double[] categoryPriorWeights, double[] quantities,
                         double[] quantityPriorWeights, double[] expectedQuantities, TableParams params) {
        if (values.length == 0) throw new IllegalArgumentException("no values");
        if (expectedQuantities.length == 0) throw new IllegalArgumentException("no expected quantities");
        QuantitySet quantitySet = new QuantitySet(quantities);
        for (double expectedQuantity : expectedQuantities) quantitySet.requireReachable(expectedQuantity);
        if (quantityPriorWeights.length != quantitySet.stateCount)
            throw new IllegalArgumentException("one quantity weight per quantity required");
        quantityValues = quantitySet.values;
        nullStateIndex = quantitySet.nullStateIndex;

        groups = new UniqueValueGroups(values, categoryPriorWeights);
        valueScale = groups.entryMean;
        inverseValueScale = 1.0 / valueScale;
        double[] scaledValues = new double[groups.uniqueCount];
        for (int i = 0; i < groups.uniqueCount; i++) scaledValues[i] = groups.uniqueValues[i] * inverseValueScale;
        CategoryFamily categoryFamily = new CategoryFamily(scaledValues, groups.uniqueWeights, params);
        categoryTable = categoryFamily.table();

        double[] logQuantityPrior = new double[quantitySet.stateCount];
        for (int state = 0; state < logQuantityPrior.length; state++) {
            double weight = quantityPriorWeights[state];
            if (!(weight > 0) || Double.isInfinite(weight))
                throw new IllegalArgumentException("quantity weights must be positive and finite");
            logQuantityPrior[state] = Math.log(weight);
        }
        quantityTables = new QuantityTable[expectedQuantities.length];
        for (int i = 0; i < quantityTables.length; i++)
            quantityTables[i] = QuantityTableBuilder.build(expectedQuantities[i], quantitySet, categoryFamily,
                    logQuantityPrior, params);
    }

    /**
     * Draws a sample at target s (raw units; clamped to the supported range for the given n).
     *
     * @param expectedQuantityIndex index into the expectedQuantities array given at construction
     */
    public Sample sample(int expectedQuantityIndex, double s, RandomGenerator rng) {
        long stateAndCategoryRow =
                quantityTables[expectedQuantityIndex].drawStateAndCategoryRow(s * inverseValueScale, rng);
        int state = (int) (stateAndCategoryRow >>> 32);
        if (state == nullStateIndex) return new Sample(-1, state, 0);
        int uniqueIndex = categoryTable.drawUniqueIndex((int) stateAndCategoryRow, rng.nextLong());
        return new Sample(groups.drawEntryIndex(uniqueIndex, rng), state, quantityValues[state]);
    }

    /** Lower end of the clamp interval for s (raw units) for the given expected-quantity index. */
    public double minTargetS(int expectedQuantityIndex) {
        return quantityTables[expectedQuantityIndex].minScaledTarget() * valueScale;
    }

    /** Upper end of the clamp interval for s (raw units) for the given expected-quantity index. */
    public double maxTargetS(int expectedQuantityIndex) {
        return quantityTables[expectedQuantityIndex].maxScaledTarget() * valueScale;
    }

    public long approximateMemoryBytes() {
        long bytes = categoryTable.approximateMemoryBytes() + groups.approximateMemoryBytes();
        for (QuantityTable table : quantityTables) bytes += table.approximateMemoryBytes();
        return bytes;
    }
}
