package io.github.aevns.fastmaxent;

import java.util.random.RandomGenerator;
import java.util.Arrays;

/**
 * Sorted values collapsed into unique values with their multiplicities and prior weights. Each group is a contiguous
 * entry range, so an entry is drawn from its group directly, in proportion to the entries' prior weights.
 */
final class UniqueValueGroups {
    final int uniqueCount;
    final double[] uniqueValues;
    final int[] groupStartIndex, groupSize;
    /** Prior mass of each unique value: the sum of its entries' weights, normalized to sum to 1. */
    final double[] uniqueWeights;
    /** Mean over all entries (not over unique values), summed in entry order. */
    final double entryMean;
    /** Per-group {@link FixedPointCdf} over the entries of groups with unequal weights; null if there are none. */
    private final int[] entryCdf;
    private final boolean[] weightedGroup;

    UniqueValueGroups(double[] sortedValues, double[] entryWeights) {
        int entryCount = sortedValues.length;
        if (entryWeights.length != entryCount)
            throw new IllegalArgumentException("one category weight per value required");
        for (int i = 0; i < entryCount; i++) {
            if (!(sortedValues[i] > 0) || Double.isInfinite(sortedValues[i]))
                throw new IllegalArgumentException("values must be positive and finite");
            if (i > 0 && sortedValues[i] < sortedValues[i - 1])
                throw new IllegalArgumentException("values must be sorted");
            if (!(entryWeights[i] > 0) || Double.isInfinite(entryWeights[i]))
                throw new IllegalArgumentException("category weights must be positive and finite");
        }
        int[] groupStartScratch = new int[entryCount + 1];
        int groups = 0;
        for (int i = 0; i < entryCount; i++)
            if (i == 0 || sortedValues[i] != sortedValues[i - 1]) groupStartScratch[groups++] = i;
        uniqueCount = groups;
        if (uniqueCount < 2) throw new IllegalArgumentException("need at least 2 distinct values");
        groupStartIndex = Arrays.copyOf(groupStartScratch, uniqueCount);
        groupSize = new int[uniqueCount];
        uniqueValues = new double[uniqueCount];
        uniqueWeights = new double[uniqueCount];
        boolean[] weighted = new boolean[uniqueCount];
        boolean anyWeighted = false;
        double totalWeight = 0;
        for (int k = 0; k < uniqueCount; k++) {
            int start = groupStartIndex[k];
            int groupEnd = (k + 1 < uniqueCount) ? groupStartIndex[k + 1] : entryCount;
            groupSize[k] = groupEnd - start;
            uniqueValues[k] = sortedValues[start];
            double groupWeight = 0;
            for (int i = start; i < groupEnd; i++) {
                groupWeight += entryWeights[i];
                if (entryWeights[i] != entryWeights[start]) weighted[k] = true;
            }
            uniqueWeights[k] = groupWeight;
            totalWeight += groupWeight;
            anyWeighted |= weighted[k];
        }
        for (int k = 0; k < uniqueCount; k++) uniqueWeights[k] /= totalWeight;

        if (anyWeighted) {
            entryCdf = new int[entryCount];
            double[] cumulativeWeight = new double[entryCount];
            for (int k = 0; k < uniqueCount; k++) {
                if (!weighted[k]) continue;
                int start = groupStartIndex[k], size = groupSize[k];
                double cumulative = 0;
                for (int i = start; i < start + size; i++) cumulativeWeight[i] = cumulative += entryWeights[i];
                FixedPointCdf.encode(cumulativeWeight, start, size, cumulative, entryCdf, start);
            }
            weightedGroup = weighted;
        } else {
            entryCdf = null;
            weightedGroup = null;
        }

        double valueSum = 0;
        for (double value : sortedValues) valueSum += value;
        entryMean = valueSum / entryCount;
    }

    /** Picks an entry index within the group (consumes a random long only if the group has duplicates). */
    int drawEntryIndex(int uniqueIndex, RandomGenerator rng) {
        int size = groupSize[uniqueIndex];
        int entryIndex = groupStartIndex[uniqueIndex];
        if (size > 1) {
            long key = FixedPoint.highBits(rng.nextLong());
            if (weightedGroup != null && weightedGroup[uniqueIndex])
                return entryIndex + FixedPointCdf.search(entryCdf, entryIndex, size, key);
            entryIndex += (int) ((key * size) >>> 32);
        }
        return entryIndex;
    }

    long approximateMemoryBytes() {
        long bytes = 4L * (groupStartIndex.length + groupSize.length);
        if (entryCdf != null) bytes += 4L * entryCdf.length + weightedGroup.length;
        return bytes;
    }
}
