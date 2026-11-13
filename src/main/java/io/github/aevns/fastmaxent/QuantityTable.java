package io.github.aevns.fastmaxent;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * For one expected quantity n: stored rows at ascending targets s. Each row holds, per quantity state, an alias entry
 * and the pair of category rows to mix. Between two rows one is picked with probability linear in s, which keeps
 * E[N] = n and E[N V] = s exact. Entries are interleaved so one cache line serves both lookups.
 */
final class QuantityTable {
    private static final int INTS_PER_STATE_ENTRY = 4;
    private static final int ALIAS_THRESHOLD = 0;             // fixed-point chance of keeping the drawn column
    private static final int ALIAS_STATE = 1;                 // state to return otherwise
    private static final int CATEGORY_ROW = 2;                // lower category row for this state
    private static final int CATEGORY_ROW_MIX_THRESHOLD = 3;  // fixed-point chance of using CATEGORY_ROW + 1

    private final int stateCount, nullStateIndex;
    private final double[] scaledTargets;
    private final int[] packedRows;

    /** Packs the entries of one target row (arrays are indexed by state). */
    static int[] packRow(int[] aliasThresholds, int[] aliasStates, int[] categoryRows, int[] categoryRowMixThresholds) {
        int stateCount = aliasThresholds.length;
        int[] entries = new int[stateCount * INTS_PER_STATE_ENTRY];
        for (int state = 0; state < stateCount; state++) {
            int entryOffset = INTS_PER_STATE_ENTRY * state;
            entries[entryOffset + ALIAS_THRESHOLD] = aliasThresholds[state];
            entries[entryOffset + ALIAS_STATE] = aliasStates[state];
            entries[entryOffset + CATEGORY_ROW] = categoryRows[state];
            entries[entryOffset + CATEGORY_ROW_MIX_THRESHOLD] = categoryRowMixThresholds[state];
        }
        return entries;
    }

    /**
     * rows[r] must have been produced by packRow for the row whose target is unsortedScaledTargets[r]. Rows may be
     * given in any order; they are stored sorted by target, which lowerTargetRow's bracketing search relies on.
     * nullStateIndex is the state that has no category (quantity 0), or -1.
     */
    QuantityTable(int stateCount, int nullStateIndex, double[] unsortedScaledTargets, int[][] rows) {
        this.stateCount = stateCount;
        this.nullStateIndex = nullStateIndex;
        int rowCount = unsortedScaledTargets.length;
        int rowInts = stateCount * INTS_PER_STATE_ENTRY;
        if (rowCount == 0 || rows.length != rowCount) throw new IllegalArgumentException("row/target count mismatch");
        Integer[] order = new Integer[rowCount];
        for (int row = 0; row < rowCount; row++) order[row] = row;
        Arrays.sort(order, (a, b) -> Double.compare(unsortedScaledTargets[a], unsortedScaledTargets[b]));
        scaledTargets = new double[rowCount];
        packedRows = new int[rowCount * rowInts];
        for (int row = 0; row < rowCount; row++) {
            int[] source = rows[order[row]];
            if (source.length != rowInts) throw new IllegalArgumentException("row not produced by packRow");
            scaledTargets[row] = unsortedScaledTargets[order[row]];
            System.arraycopy(source, 0, packedRows, row * rowInts, rowInts);
        }
    }

    double minScaledTarget() { return scaledTargets[0]; }

    double maxScaledTarget() { return scaledTargets[scaledTargets.length - 1]; }

    long approximateMemoryBytes() {
        return 4L * packedRows.length + 8L * scaledTargets.length;
    }

    /** Lower of the two stored rows bracketing scaledS; clamped to the first/last row outside the stored range. */
    private int lowerTargetRow(double scaledS) {
        int rowCount = scaledTargets.length;
        if (scaledS <= scaledTargets[0]) return 0;
        if (scaledS >= scaledTargets[rowCount - 1]) return rowCount - 1;
        return SortedSearch.bracket(scaledTargets, scaledS);
    }

    /** Probability of using row lowerRow + 1 instead of lowerRow; zero when clamped. */
    private double upperRowWeight(double scaledS, int lowerRow) {
        if (lowerRow >= scaledTargets.length - 1 || scaledS <= scaledTargets[0]) return 0;
        return (scaledS - scaledTargets[lowerRow]) / (scaledTargets[lowerRow + 1] - scaledTargets[lowerRow]);
    }

    /**
     * Draws a quantity state and the category row to use for it, packed as (state << 32) | categoryRow. The null state
     * has no category row; its low half is 0. Consumes two random longs.
     */
    long drawStateAndCategoryRow(double scaledS, RandomGenerator rng) {
        int lowerRow = lowerTargetRow(scaledS);
        double upperRowMixProbability = upperRowWeight(scaledS, lowerRow);
        // First long: high half chooses between the two target rows; low half decides the alias keep/redirect.
        long rowMixAndAliasRandom = rng.nextLong();
        int targetRow = lowerRow;
        if (FixedPoint.highBits(rowMixAndAliasRandom) < (long) (upperRowMixProbability * FixedPoint.SCALE)) targetRow++;
        // Second long: high half picks the alias column; low half decides the category-row mix.
        long columnAndCategoryRowRandom = rng.nextLong();
        int column = (int) ((FixedPoint.highBits(columnAndCategoryRowRandom) * stateCount) >>> 32);
        int columnEntry = (targetRow * stateCount + column) * INTS_PER_STATE_ENTRY;
        boolean keepColumn =
                FixedPoint.isBelow(FixedPoint.lowBits(rowMixAndAliasRandom), packedRows[columnEntry + ALIAS_THRESHOLD]);
        int state = keepColumn ? column : packedRows[columnEntry + ALIAS_STATE];
        if (state == nullStateIndex) return (long) state << 32;
        int stateEntry = (targetRow * stateCount + state) * INTS_PER_STATE_ENTRY;
        int categoryRow = packedRows[stateEntry + CATEGORY_ROW]
                + (FixedPoint.isBelow(FixedPoint.lowBits(columnAndCategoryRowRandom),
                        packedRows[stateEntry + CATEGORY_ROW_MIX_THRESHOLD]) ? 1 : 0);
        return ((long) state << 32) | categoryRow;
    }
}
