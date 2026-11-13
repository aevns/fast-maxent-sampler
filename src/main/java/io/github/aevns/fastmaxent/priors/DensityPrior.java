package io.github.aevns.fastmaxent.priors;

import java.util.Arrays;

/**
 * Prior weights that correct for crowding: values close together along an axis share weight, after Leinster and
 * Cobbold's similarity-sensitive diversity.
 *
 * Each unique value k, occurring g_k times, gets W_k ~ g_k / (1 - sigma + sigma * rho_k), split evenly among its
 * occurrences. rho_k = sum_j g_j K(u_k - u_j) is the density around its axis position u_k, with K an Epanechnikov
 * kernel of half-width h. Sigma = 0 weights every value equally; sigma = 1 weights regions by how much of the axis they
 * cover, not by how many values crowd into them.
 *
 * @param axis               the axis distances are measured along
 * @param halfWidth          kernel half-width, as a difference between values that the axis scales to positions:
 *                           0.15 on a symlog axis means neighbours within 15%
 * @param similarity         sigma, in [0, 1]
 * @param continueBeyondEnds for sets that truncate a longer sequence (such as quantities 1..64): extends the values
 *                           past each outer end at their outermost spacing, so the end values are not treated as
 *                           uncrowded
 * @param zeroIsNullState    whether values equal to 0 are a separate null state, off the axis; otherwise 0 is an
 *                           ordinary value
 * @param nullStateWeight    with a null state, the weight its values share, with the other values scaled to mean 1;
 *                           positive
 */
public record DensityPrior(Axis axis, double halfWidth, double similarity, boolean continueBeyondEnds,
                           boolean zeroIsNullState, double nullStateWeight) {

    /** Guards against outermost values spaced too finely to continue. */
    private static final int MAX_CONTINUED_VALUES = 1 << 20;

    public DensityPrior {
        if (axis == null) throw new IllegalArgumentException("no axis");
        double positionHalfWidth = axis.scale(halfWidth);
        if (!(positionHalfWidth > 0) || Double.isInfinite(positionHalfWidth))
            throw new IllegalArgumentException("half-width must scale to a positive, finite width");
        if (!(similarity >= 0 && similarity <= 1)) throw new IllegalArgumentException("similarity must be in [0, 1]");
        if (!(nullStateWeight > 0) || Double.isInfinite(nullStateWeight))
            throw new IllegalArgumentException("null-state weight must be positive and finite");
    }

    /**
     * Weight of each value, in the order given: summing to 1, or with a null state, the other values averaging 1.
     *
     * @param values finite, in any order; duplicates allowed
     */
    public double[] weights(double[] values) {
        if (!zeroIsNullState) return densityWeights(values);
        int zeroCount = 0;
        for (double value : values) if (value == 0) zeroCount++;
        int nonzeroCount = values.length - zeroCount;
        double[] result = new double[values.length];
        double[] nonzero = new double[nonzeroCount];
        for (int i = 0, next = 0; i < values.length; i++) {
            if (values[i] == 0) result[i] = nullStateWeight / zeroCount;
            else nonzero[next++] = values[i];
        }
        if (nonzeroCount == 0) return result;
        double[] nonzeroWeights = densityWeights(nonzero);
        for (int i = 0, next = 0; i < values.length; i++)
            if (values[i] != 0) result[i] = nonzeroWeights[next++] * nonzeroCount;
        return result;
    }

    /** Weight of each value, in the order given, summing to 1. */
    private double[] densityWeights(double[] values) {
        int valueCount = values.length;
        if (valueCount == 0) throw new IllegalArgumentException("no values");
        double positionHalfWidth = axis.scale(halfWidth);
        for (double value : values)
            if (!Double.isFinite(value)) throw new IllegalArgumentException("values must be finite");

        Integer[] order = new Integer[valueCount];
        for (int i = 0; i < valueCount; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(values[a], values[b]));
        double[] unique = new double[valueCount];
        int[] multiplicity = new int[valueCount], uniqueIndexOfValue = new int[valueCount];
        int uniqueCount = 0;
        for (int index : order) {
            if (uniqueCount == 0 || values[index] != unique[uniqueCount - 1]) unique[uniqueCount++] = values[index];
            multiplicity[uniqueCount - 1]++;
            uniqueIndexOfValue[index] = uniqueCount - 1;
        }

        int last = uniqueCount - 1;
        boolean lowSideContinues = continueBeyondEnds && uniqueCount >= 2 && unique[1] < 0;
        boolean highSideContinues = continueBeyondEnds && uniqueCount >= 2 && unique[last - 1] > 0;
        double lowGap = lowSideContinues ? unique[1] - unique[0] : 0;
        double highGap = highSideContinues ? unique[last] - unique[last - 1] : 0;
        int lowContinued = lowSideContinues ? continuedValueCount(unique[0], -lowGap, axis, positionHalfWidth) : 0;
        int highContinued = highSideContinues ? continuedValueCount(unique[last], highGap, axis, positionHalfWidth) : 0;

        int positionCount = lowContinued + uniqueCount + highContinued;
        double[] positions = new double[positionCount];
        int[] positionMultiplicity = new int[positionCount];
        Arrays.fill(positionMultiplicity, 1);
        for (int i = 0; i < lowContinued; i++)
            positions[i] = axis.position(unique[0] - (lowContinued - i) * lowGap);
        for (int k = 0; k < uniqueCount; k++) {
            positions[lowContinued + k] = axis.position(unique[k]);
            positionMultiplicity[lowContinued + k] = multiplicity[k];
        }
        for (int j = 1; j <= highContinued; j++)
            positions[lowContinued + last + j] = axis.position(unique[last] + j * highGap);
        for (int i = 0; i < positionCount; i++) {
            if (!Double.isFinite(positions[i])) throw new IllegalArgumentException("axis positions must be finite");
            if (i > 0 && !(positions[i] > positions[i - 1]))
                throw new IllegalArgumentException("axis must be strictly increasing over the values");
        }
        double[] positionWeights = weightsAtPositions(positions, positionMultiplicity, similarity, positionHalfWidth);

        double weightSum = 0;
        for (int k = 0; k < uniqueCount; k++) weightSum += positionWeights[lowContinued + k];
        double[] valueWeights = new double[valueCount];
        for (int i = 0; i < valueCount; i++) {
            int k = uniqueIndexOfValue[i];
            valueWeights[i] = positionWeights[lowContinued + k] / weightSum / multiplicity[k];
        }
        return valueWeights;
    }

    /** Number of steps of signedGap outward from outer that stay within one half-width of it along the axis. */
    private static int continuedValueCount(double outer, double signedGap, Axis axis, double halfWidth) {
        double outerPosition = axis.position(outer);
        int steps = 0;
        while (Math.abs(axis.position(outer + (steps + 1) * signedGap) - outerPosition) <= halfWidth) {
            if (++steps > MAX_CONTINUED_VALUES)
                throw new IllegalArgumentException("outermost values are too finely spaced for the kernel width");
        }
        return steps;
    }

    /** Unnormalized W_k for sorted positions, in O(K) via a sliding window of running sums of g, g*d and g*d^2. */
    private static double[] weightsAtPositions(double[] sortedPositions, int[] multiplicities, double similarity,
                                               double halfWidth) {
        int positionCount = sortedPositions.length;
        double origin = sortedPositions[0];
        double[] position = new double[positionCount];
        for (int i = 0; i < positionCount; i++) position[i] = sortedPositions[i] - origin;

        // Sliding window [windowStart, windowEnd) around k holding sums of g, g*d and g*d^2 where d is the offset
        // from the current position. Moving the anchor from one position to the next shifts the sums algebraically, so
        // every quantity stays on the scale of one kernel width (global prefix sums would cancel catastrophically
        // for narrow kernels).
        double[] weights = new double[positionCount];
        int windowStart = 0, windowEnd = 0;
        double sumCount = 0, sumOffset = 0, sumSquaredOffset = 0;
        double anchor = position[0];
        for (int k = 0; k < positionCount; k++) {
            double shift = position[k] - anchor;      // d_j' = d_j - shift
            anchor = position[k];
            if (shift > 2 * halfWidth) {
                // The old and new windows cannot overlap: restart from exact zeros instead of carrying rounding
                // residue across the gap (it would be amplified by the large shift).
                sumCount = sumOffset = sumSquaredOffset = 0;
                int restart = k;
                while (restart > 0 && anchor - position[restart - 1] <= halfWidth) restart--;
                windowStart = windowEnd = restart;
            } else {
                sumSquaredOffset += -2 * shift * sumOffset + shift * shift * sumCount;
                sumOffset -= shift * sumCount;
                while (windowStart < windowEnd && anchor - position[windowStart] > halfWidth) {
                    double g = multiplicities[windowStart], d = position[windowStart] - anchor;
                    sumCount -= g; sumOffset -= g * d; sumSquaredOffset -= g * d * d;
                    windowStart++;
                }
            }
            while (windowEnd < positionCount && position[windowEnd] - anchor <= halfWidth) {
                double g = multiplicities[windowEnd], d = position[windowEnd] - anchor;
                sumCount += g; sumOffset += g * d; sumSquaredOffset += g * d * d;
                windowEnd++;
            }
            double density = sumCount - sumSquaredOffset / (halfWidth * halfWidth);
            density = Math.max(density, multiplicities[k]);   // guards rounding; exact value is >= g_k
            weights[k] = multiplicities[k] / (1 - similarity + similarity * density);
        }
        return weights;
    }
}
