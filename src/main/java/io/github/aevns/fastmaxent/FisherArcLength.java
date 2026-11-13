package io.github.aevns.fastmaxent;

/**
 * Fisher-Rao arc length helpers. Rows spaced equally in arc length are roughly equally distinguishable in KL
 * divergence, however the curve is parameterized.
 */
final class FisherArcLength {
    private FisherArcLength() {}

    /**
     * Speed along the curve of fixed E[q] as beta varies: speed^2 = Var(q x) - Cov(q, q x)^2 / Var(q). Computed on
     * q (x - a) with a = E[|q| x] / E[|q|], which leaves the result unchanged but avoids cancellation.
     */
    static double speedAlongFixedExpectedQuantityCurve(double[] stateProbabilities, double[] quantities,
                                                       double[] meanScaledValueGivenState,
                                                       double[] varianceScaledValueGivenState) {
        int stateCount = stateProbabilities.length;
        double expectedQuantity = 0, expectedMagnitude = 0, expectedMagnitudeTimesValue = 0;
        for (int state = 0; state < stateCount; state++) {
            double magnitude = Math.abs(quantities[state]);
            expectedQuantity += stateProbabilities[state] * quantities[state];
            expectedMagnitude += stateProbabilities[state] * magnitude;
            expectedMagnitudeTimesValue += stateProbabilities[state] * magnitude * meanScaledValueGivenState[state];
        }
        double centeringValue = expectedMagnitudeTimesValue / expectedMagnitude;
        double quantitySecondMoment = 0, quantityValueCovarianceTerm = 0, valueVarianceTerm = 0;
        double centeredQuantityValueMean = 0;
        for (int state = 0; state < stateCount; state++) {
            double quantity = quantities[state], probability = stateProbabilities[state];
            double centeredMean = meanScaledValueGivenState[state] - centeringValue;
            quantitySecondMoment += probability * quantity * quantity;
            quantityValueCovarianceTerm += probability * quantity * quantity * centeredMean;
            valueVarianceTerm += probability * quantity * quantity
                    * (varianceScaledValueGivenState[state] + centeredMean * centeredMean);
            centeredQuantityValueMean += probability * quantity * centeredMean;
        }
        double quantityVariance = quantitySecondMoment - expectedQuantity * expectedQuantity;
        double quantityValueCovariance = quantityValueCovarianceTerm - expectedQuantity * centeredQuantityValueMean;
        double valueVariance = valueVarianceTerm - centeredQuantityValueMean * centeredQuantityValueMean;
        double residualVariance = valueVariance
                - (quantityVariance > 0 ? quantityValueCovariance * quantityValueCovariance / quantityVariance : 0);
        return Math.sqrt(Math.max(residualVariance, 0));
    }

    /** Trapezoid-rule cumulative arc length at each position; element 0 is 0. positions must be ascending. */
    static double[] cumulativeArcLength(double[] positions, double[] speeds) {
        double[] cumulative = new double[positions.length];
        for (int i = 1; i < positions.length; i++)
            cumulative[i] = cumulative[i - 1] + 0.5 * (speeds[i - 1] + speeds[i]) * (positions[i] - positions[i - 1]);
        return cumulative;
    }

    /** Position at which the cumulative arc length reaches target, by linear interpolation between positions. */
    static double positionAtArcLength(double[] positions, double[] cumulative, double targetArcLength) {
        int low = SortedSearch.bracket(cumulative, targetArcLength), high = low + 1;
        double span = cumulative[high] - cumulative[low];
        double fraction = span > 0 ? (targetArcLength - cumulative[low]) / span : 0;
        return positions[low] + fraction * (positions[high] - positions[low]);
    }

    /** Index of the position whose cumulative arc length is nearest to the target. */
    static int nearestPositionIndex(double[] cumulative, double targetArcLength) {
        int low = 0, high = cumulative.length - 1;
        while (low < high) {
            int middle = (low + high) >>> 1;
            if (cumulative[middle] >= targetArcLength) high = middle; else low = middle + 1;
        }
        int pick = low;
        if (low > 0 && targetArcLength - cumulative[low - 1] < cumulative[low] - targetArcLength) pick = low - 1;
        return pick;
    }
}
