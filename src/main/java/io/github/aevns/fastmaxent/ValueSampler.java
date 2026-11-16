package io.github.aevns.fastmaxent;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * Draws a category from the maximum-entropy distribution relative to a prior, with E[V] = target for category
 * value V. target can be any value in the supported range.
 * <p>
 * With category prior w:
 *   p(c) ~ w_c exp(-beta v_c)
 * <p>
 * Immutable once initialized; sampling is thread-safe if each caller supplies its own {@link RandomGenerator}.
 */
public final class ValueSampler {
    private final UniqueValueGroups groups;
    private final double valueScale, inverseValueScale;
    private final CategoryTable categoryTable;
    private final double[] sortedMeanScaledValue;
    private final int[] rowOfSortedIndex;

    /**
     * Builds the sampling table.
     *
     * @param values               the value of each category; finite, sorted ascending, at least two distinct
     * @param priorWeights         prior weight of each category; positive, any scale
     * @param params               table tuning parameters
     */
    public ValueSampler(double[] values, double[] priorWeights, TableParams params) {
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
     * Draws a category index at the given target value (raw units; clamped to the supported range).
     *
     * @return an index into the values array given at construction
     */
    public int sample(double target, RandomGenerator rng) {
        double scaledMean = target * inverseValueScale;
        int lowerRow = lowerSortedRow(scaledMean);
        int upperRowMixThreshold = FixedPoint.fromProbability(upperRowWeight(scaledMean, lowerRow));
        int sortedRow = lowerRow;
        if (FixedPoint.isBelow(FixedPoint.highBits(rng.nextLong()), upperRowMixThreshold)) sortedRow++;
        int uniqueIndex = categoryTable.drawUniqueIndex(rowOfSortedIndex[sortedRow], rng.nextLong());
        return groups.drawEntryIndex(uniqueIndex, rng);
    }

    /** Lower end of the clamp interval for target (raw units). */
    public double minTargetValue() { return sortedMeanScaledValue[0] * valueScale; }

    /** Upper end of the clamp interval for target (raw units). */
    public double maxTargetValue() {
        return sortedMeanScaledValue[sortedMeanScaledValue.length - 1] * valueScale;
    }

    public long approximateMemoryBytes() {
        return categoryTable.approximateMemoryBytes() + groups.approximateMemoryBytes()
                + 8L * sortedMeanScaledValue.length + 4L * rowOfSortedIndex.length;
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
