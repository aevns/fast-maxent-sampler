package io.github.aevns.fastmaxent;

/**
 * For a given beta, solves for the lambda at which p(q) ~ m(q) Z(beta q) exp(-lambda q) has the requested
 * mean. Arrays are indexed by state. Used only while building tables.
 */
final class QuantityDistributionSolver {
    private final CategoryFamily categoryFamily;
    private final QuantitySet quantities;
    private final double[] logPrior;
    /** State with the largest |quantity|; its category branch sets the reference value shared by all states. */
    private final int anchorState;
    private final CategoryFamily.Point categoryPoint = new CategoryFamily.Point();
    private double warmStartLambda = 0;

    final double[] logWeightBeforeLambda;
    final double[] stateProbabilities;
    /** E[x | q] under the stored category rows, mixed to match the exact family mean at t = beta * q. */
    final double[] meanScaledValueGivenState;
    final double[] varianceScaledValueGivenState;
    final int[] lowerCategoryRow;
    final int[] categoryRowMixThreshold;

    QuantityDistributionSolver(CategoryFamily categoryFamily, QuantitySet quantities, double[] logPrior) {
        this.categoryFamily = categoryFamily;
        this.quantities = quantities;
        this.logPrior = logPrior;
        int stateCount = quantities.stateCount;
        int anchor = 0;
        for (int state = 1; state < stateCount; state++)
            if (Math.abs(quantities.values[state]) > Math.abs(quantities.values[anchor])) anchor = state;
        anchorState = anchor;
        logWeightBeforeLambda = new double[stateCount];
        stateProbabilities = new double[stateCount];
        meanScaledValueGivenState = new double[stateCount];
        varianceScaledValueGivenState = new double[stateCount];
        lowerCategoryRow = new int[stateCount];
        categoryRowMixThreshold = new int[stateCount];
    }

    void configureForBeta(double beta) {
        // The category family reports ln Z(t) relative to a reference value that depends on the sign of t. The dropped
        // term -t * reference is linear in q only while every state shares a reference, so states whose t has the
        // other sign (possible when quantities of both signs are allowed) are shifted onto the anchor's reference.
        categoryFamily.evaluateAtT(beta * quantities.values[anchorState], categoryPoint);
        double sharedReference = categoryPoint.referenceValue;
        for (int state = 0; state < quantities.stateCount; state++) {
            if (state == quantities.nullStateIndex) {
                logWeightBeforeLambda[state] = logPrior[state];
                meanScaledValueGivenState[state] = 0;
                varianceScaledValueGivenState[state] = 0;
                lowerCategoryRow[state] = 0;
                categoryRowMixThreshold[state] = 0;
                continue;
            }
            double quantity = quantities.values[state];
            double t = beta * quantity;
            categoryFamily.evaluateAtT(t, categoryPoint);
            varianceScaledValueGivenState[state] = categoryPoint.varianceScaledValue;
            lowerCategoryRow[state] = categoryPoint.lowerCategoryRow;
            categoryRowMixThreshold[state] = categoryPoint.mixThreshold;
            meanScaledValueGivenState[state] = categoryPoint.meanScaledValue;
            double logPartition = categoryPoint.logPartition;
            if (categoryPoint.referenceValue != sharedReference)
                logPartition -= t * (categoryPoint.referenceValue - sharedReference);
            logWeightBeforeLambda[state] = logPrior[state] + logPartition;
        }
    }

    /** Fills stateProbabilities for the given lambda and returns E[quantity]. */
    double expectedQuantityForLambda(double lambda) {
        double maxLogWeight = Double.NEGATIVE_INFINITY;
        for (int state = 0; state < quantities.stateCount; state++)
            maxLogWeight = Math.max(maxLogWeight, logWeightBeforeLambda[state] - lambda * quantities.values[state]);
        double normalizer = 0, quantitySum = 0;
        for (int state = 0; state < quantities.stateCount; state++) {
            double quantity = quantities.values[state];
            double weight = Math.exp(logWeightBeforeLambda[state] - lambda * quantity - maxLogWeight);
            stateProbabilities[state] = weight;
            normalizer += weight;
            quantitySum += weight * quantity;
        }
        for (int state = 0; state < quantities.stateCount; state++) stateProbabilities[state] /= normalizer;
        return quantitySum / normalizer;
    }

    /** Var[quantity] of the distribution currently in stateProbabilities, with the given mean. */
    private double quantityVariance(double mean) {
        double variance = 0;
        for (int state = 0; state < quantities.stateCount; state++) {
            double offset = quantities.values[state] - mean;
            variance += stateProbabilities[state] * offset * offset;
        }
        return variance;
    }

    /**
     * Newton's method on lambda (dE[q]/dlambda = -Var[q]), warm-started from the previous solve. Steps that leave the
     * bracket found so far bisect it instead. Leaves the solution in stateProbabilities.
     */
    void solveLambdaForExpectedQuantity(double expectedQuantity) {
        double lambda = warmStartLambda;
        double low = Double.NEGATIVE_INFINITY, high = Double.POSITIVE_INFINITY;
        for (int iteration = 0; iteration < 200; iteration++) {
            double mean = expectedQuantityForLambda(lambda);
            if (mean > expectedQuantity) low = lambda; else high = lambda;
            double next = lambda + (mean - expectedQuantity) / quantityVariance(mean);
            if (Math.abs(next - lambda) <= 1e-15 * Math.max(1, Math.abs(lambda))) break;
            if (!(next > low && next < high)) {
                boolean bracketed = !Double.isInfinite(low) && !Double.isInfinite(high);
                double widened = lambda + Math.copySign(1 + Math.abs(lambda), mean - expectedQuantity);
                next = bracketed ? 0.5 * (low + high) : widened;
                if (next == low || next == high) break;
            }
            lambda = next;
        }
        warmStartLambda = lambda;
        expectedQuantityForLambda(lambda);
    }

    /** Fisher-Rao speed d(arc length)/d(beta) along the fixed-E[quantity] curve, for the current solution. */
    double fisherSpeedWithRespectToBeta() {
        return FisherArcLength.speedAlongFixedExpectedQuantityCurve(stateProbabilities, quantities.values,
                meanScaledValueGivenState, varianceScaledValueGivenState);
    }
}
