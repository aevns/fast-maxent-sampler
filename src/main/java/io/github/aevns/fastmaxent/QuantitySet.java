package io.github.aevns.fastmaxent;

import java.util.Arrays;

/**
 * The allowed quantities, in the caller's order (a state index is a position in values), with the facts about them
 * that the tables and solvers need. Quantity 0, if present, is the null state: no category is drawn for it.
 */
final class QuantitySet {
    final double[] values;
    final int stateCount;
    /** State index of quantity 0, or -1 if the null state is not allowed. */
    final int nullStateIndex;
    final double min, max;
    /** Largest |quantity|. */
    final double maxMagnitude;
    /** Smallest positive quantity and smallest |negative quantity|, or NaN if there are none of that sign. */
    final double smallestPositive, smallestNegativeMagnitude;

    QuantitySet(double[] quantities) {
        values = quantities.clone();
        stateCount = values.length;
        if (stateCount < 2) throw new IllegalArgumentException("need at least 2 quantities");
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        for (int i = 0; i < stateCount; i++) {
            if (!Double.isFinite(sorted[i])) throw new IllegalArgumentException("quantities must be finite");
            if (i > 0 && sorted[i] == sorted[i - 1]) throw new IllegalArgumentException("quantities must be distinct");
        }
        int zero = -1;
        for (int i = 0; i < stateCount; i++) if (values[i] == 0) zero = i;
        nullStateIndex = zero;
        min = sorted[0];
        max = sorted[stateCount - 1];
        maxMagnitude = Math.max(Math.abs(min), Math.abs(max));
        double positive = Double.NaN, negative = Double.NaN;
        for (double quantity : sorted) {
            if (quantity < 0) negative = -quantity;
            else if (quantity > 0 && Double.isNaN(positive)) positive = quantity;
        }
        smallestPositive = positive;
        smallestNegativeMagnitude = negative;
    }

    /** An expected quantity is reachable only strictly between the smallest and largest allowed quantity. */
    void requireReachable(double expectedQuantity) {
        if (!(expectedQuantity > min && expectedQuantity < max))
            throw new IllegalArgumentException("expected quantity " + expectedQuantity
                    + " must lie strictly between the smallest and largest allowed quantity");
    }
}
