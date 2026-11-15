package io.github.aevns.fastmaxent;

import java.util.Arrays;

/**
 * The allowed quantities, in the caller's order (a state index is a position in values), with the facts about them
 * that the tables and solvers need. When a null outcome is allowed, one extra state is appended after the caller's
 * own, at quantity 0, standing for "no quantity, no category" regardless of whether 0 is itself an allowed quantity.
 */
final class QuantitySet {
    final double[] values;
    final int stateCount;
    /** State index of the null outcome (always the last state), or -1 if it is not allowed. */
    final int nullStateIndex;
    final double min, max;
    /** Largest |quantity|, over the allowed quantities and the null outcome's 0 alike. */
    final double maxMagnitude;
    /** Smallest positive quantity and smallest |negative quantity|, or NaN if there are none of that sign. */
    final double smallestPositive, smallestNegativeMagnitude;

    QuantitySet(double[] quantities, boolean includeNullState) {
        int providedStateCount = quantities.length;
        if (providedStateCount + (includeNullState ? 1 : 0) < 2)
            throw new IllegalArgumentException("need at least 2 states");
        double[] sortedProvided = quantities.clone();
        Arrays.sort(sortedProvided);
        for (int i = 0; i < providedStateCount; i++) {
            if (!Double.isFinite(sortedProvided[i])) throw new IllegalArgumentException("quantities must be finite");
            if (i > 0 && sortedProvided[i] == sortedProvided[i - 1])
                throw new IllegalArgumentException("quantities must be distinct");
        }
        nullStateIndex = includeNullState ? providedStateCount : -1;
        // Arrays.copyOf pads with 0.0, exactly the null outcome's quantity.
        values = includeNullState ? Arrays.copyOf(quantities, providedStateCount + 1) : quantities.clone();
        stateCount = values.length;

        double[] sorted = values.clone();
        Arrays.sort(sorted);
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
