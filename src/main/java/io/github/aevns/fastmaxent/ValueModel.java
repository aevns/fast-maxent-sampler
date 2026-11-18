package io.github.aevns.fastmaxent;

import java.util.Arrays;
import java.util.Objects;
import java.util.random.RandomGenerator;

/**
 * The maximum-entropy distributions over categories relative to a prior, with E[V] = target for category value V,
 * precomputed for every target in the supported range.
 * <p>
 * With category prior w:
 *   p(c) ~ w_c exp(-beta v_c)
 * <p>
 * Immutable and thread-safe once built. Draw from it through a {@link Sampler}, one per thread.
 */
public final class ValueModel {
    /**
     * Draws from its {@link ValueModel} using its own random number generator and scratch space. Cheap to create
     * and allocation-free per draw, but not thread-safe: use one per thread, as with the generator itself.
     */
    public static final class Sampler {
        private final RandomGenerator rng;
        private final double[] blockWeights;
        // Copied from the model so a cold draw reaches the tables without first loading the model.
        private final CategoryTable categoryTable;
        private final UniqueValueGroups groups;
        private final double inverseValueScale;
        private final double[] sortedMeanScaledValue;
        private final int[] rowOfSortedIndex;

        private Sampler(ValueModel model, RandomGenerator rng) {
            this.rng = rng;
            categoryTable = model.categoryTable;
            groups = model.groups;
            inverseValueScale = model.inverseValueScale;
            sortedMeanScaledValue = model.sortedMeanScaledValue;
            rowOfSortedIndex = model.rowOfSortedIndex;
            blockWeights = new double[categoryTable.blockSize()];
        }

        /**
         * Draws a category index at the given target value (clamped to the supported range).
         *
         * @param target the target mean value E[V]; clamped to [{@link ValueModel#minTargetValue()},
         *               {@link ValueModel#maxTargetValue()}]
         * @return an index into the values array given at construction
         */
        public int sample(double target) {
            double scaledMean = target * inverseValueScale;
            int lowerRow = lowerSortedRow(scaledMean);
            int upperRowMixThreshold = FixedPoint.fromProbability(upperRowWeight(scaledMean, lowerRow));
            int sortedRow = lowerRow;
            if (FixedPoint.isBelow(FixedPoint.highBits(rng.nextLong()), upperRowMixThreshold)) sortedRow++;
            int uniqueIndex =
                    categoryTable.drawUniqueIndex(rowOfSortedIndex[sortedRow], rng.nextLong(), blockWeights);
            return groups.drawEntryIndex(uniqueIndex, rng);
        }

        /** Lower of the two sorted rows bracketing scaledMean; clamped to the first/last row outside the stored range. */
        private int lowerSortedRow(double scaledMean) {
            int rowCount = sortedMeanScaledValue.length;
            if (scaledMean <= sortedMeanScaledValue[0]) return 0;
            if (scaledMean >= sortedMeanScaledValue[rowCount - 1]) return rowCount - 1;
            return SortedSearch.bracket(sortedMeanScaledValue, scaledMean);
        }

        /** Probability of using the sorted row above lowerRow instead of lowerRow; zero when clamped. */
        private double upperRowWeight(double scaledMean, int lowerRow) {
            if (lowerRow >= sortedMeanScaledValue.length - 1 || scaledMean <= sortedMeanScaledValue[0]) return 0;
            double lower = sortedMeanScaledValue[lowerRow], upper = sortedMeanScaledValue[lowerRow + 1];
            return (scaledMean - lower) / (upper - lower);
        }
    }

    private final UniqueValueGroups groups;
    private final double valueScale, inverseValueScale;
    private final CategoryTable categoryTable;
    private final double[] sortedMeanScaledValue;
    private final int[] rowOfSortedIndex;

    /**
     * Builds the sampling table.
     *
     * @param values               the value of each category; finite, in any order (already-sorted input skips the
     *                             internal sort), at least two distinct values present
     * @param priorWeights         prior weight of each category; strictly positive
     * @param params               table tuning parameters
     */
    public ValueModel(double[] values, double[] priorWeights, TableParams params) {
        if (values.length == 0) throw new IllegalArgumentException("no values");
        groups = new UniqueValueGroups(values, priorWeights);
        valueScale = groups.entryAbsoluteMean;
        inverseValueScale = 1.0 / valueScale;
        double[] scaledValues = new double[groups.uniqueCount];
        for (int i = 0; i < groups.uniqueCount; i++) scaledValues[i] = groups.uniqueValues[i] * inverseValueScale;
        categoryTable = new CategoryFamily(scaledValues, groups.uniqueWeights, params).table();

        int rowCount = categoryTable.rowCount();
        Integer[] order = new Integer[rowCount];
        for (int row = 0; row < rowCount; row++) order[row] = row;
        Arrays.sort(order, (a, b) ->
                Double.compare(categoryTable.rowMeanScaledValue(a), categoryTable.rowMeanScaledValue(b)));
        sortedMeanScaledValue = new double[rowCount];
        rowOfSortedIndex = new int[rowCount];
        for (int i = 0; i < rowCount; i++) {
            rowOfSortedIndex[i] = order[i];
            sortedMeanScaledValue[i] = categoryTable.rowMeanScaledValue(order[i]);
        }
    }

    /**
     * Creates a sampler that draws from this model. Samplers are not thread-safe; create one per thread.
     *
     * @param rng the random number generator the sampler draws with
     * @return a new sampler
     */
    public Sampler newSampler(RandomGenerator rng) {
        return new Sampler(this, Objects.requireNonNull(rng, "rng"));
    }

    /**
     * Lower end of the clamp interval for target.
     *
     * @return the smallest supported target value
     */
    public double minTargetValue() { return sortedMeanScaledValue[0] * valueScale; }

    /**
     * Upper end of the clamp interval for target.
     *
     * @return the largest supported target value
     */
    public double maxTargetValue() {
        return sortedMeanScaledValue[sortedMeanScaledValue.length - 1] * valueScale;
    }

    /**
     * Approximate memory held by this model (each {@link Sampler} adds only its small scratch buffer).
     *
     * @return approximate size in bytes of the precomputed table and grouped category data
     */
    public long approximateMemoryBytes() {
        return categoryTable.approximateMemoryBytes() + groups.approximateMemoryBytes()
                + 8L * sortedMeanScaledValue.length + 4L * rowOfSortedIndex.length;
    }
}
