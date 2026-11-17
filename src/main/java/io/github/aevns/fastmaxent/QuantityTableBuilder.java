package io.github.aevns.fastmaxent;

/** Builds the {@link QuantityTable} for one expected quantity n. All work here is offline. */
final class QuantityTableBuilder {
    private QuantityTableBuilder() {}

    static QuantityTable build(double expectedQuantity, QuantitySet quantities, CategoryFamily categoryFamily,
                               double[] logQuantityPrior, TableParams params) {
        QuantityDistributionSolver solver =
                new QuantityDistributionSolver(categoryFamily, quantities, logQuantityPrior);
        double[] rowBetas = placeRowsByFisherArcLength(expectedQuantity, quantities, categoryFamily, solver, params);
        int rowCount = rowBetas.length;
        int stateCount = quantities.stateCount;

        double[] rowScaledS = new double[rowCount];
        int[][] rowEntries = new int[rowCount][];
        int[] aliasThresholds = new int[stateCount], aliasStates = new int[stateCount];
        for (int row = 0; row < rowCount; row++) {
            solver.configureForBeta(rowBetas[row]);
            solver.solveLambdaForExpectedQuantity(expectedQuantity);
            AliasMethod.buildTables(solver.stateProbabilities, aliasThresholds, aliasStates);
            // Use the probabilities the quantized alias tables actually realize, so each row's s is exact.
            double[] realizedProbabilities = new double[stateCount];
            for (int column = 0; column < stateCount; column++)
                AliasMethod.addColumnMass(realizedProbabilities, column, aliasThresholds[column], aliasStates[column]);
            double scaledS = 0;
            for (int state = 0; state < stateCount; state++) {
                double quantity = quantities.values[state];
                scaledS += realizedProbabilities[state] * quantity * solver.meanScaledValueGivenState[state];
            }
            rowScaledS[row] = scaledS;
            rowEntries[row] = QuantityTable.packRow(aliasThresholds, aliasStates,
                    solver.lowerCategoryRow, solver.categoryRowMixThreshold);
        }
        return new QuantityTable(stateCount, quantities.nullStateIndex, rowScaledS, rowEntries);
    }

    /**
     * Places the stored rows' betas equally in Fisher-Rao arc length, integrated over candidate betas log-spaced in
     * |beta| on each side. Log spacing matters because the two sides can differ in scale by many orders of magnitude.
     */
    private static double[] placeRowsByFisherArcLength(double expectedQuantity, QuantitySet quantities,
                                                       CategoryFamily categoryFamily, QuantityDistributionSolver solver,
                                                       TableParams params) {
        int candidatesPerSign = params.betaCandidatesPerSign();
        int candidateCount = 2 * candidatesPerSign + 1;
        double[] candidateBetas = new double[candidateCount];
        double[] candidateSpeeds = new double[candidateCount];
        double smallestRelevant = smallestRelevantBetaMagnitude(quantities, categoryFamily);
        double[] positiveMagnitudes = logSpacedMagnitudes(smallestRelevant,
                saturationBeta(quantities, categoryFamily, +1), candidatesPerSign);
        double[] negativeMagnitudes = logSpacedMagnitudes(smallestRelevant,
                saturationBeta(quantities, categoryFamily, -1), candidatesPerSign);
        candidateBetas[candidatesPerSign] = 0;
        for (int i = 0; i < candidatesPerSign; i++) {
            candidateBetas[candidatesPerSign + 1 + i] = positiveMagnitudes[i];
            candidateBetas[candidatesPerSign - 1 - i] = -negativeMagnitudes[i];
        }
        for (int i = 0; i < candidateCount; i++) {
            solver.configureForBeta(candidateBetas[i]);
            solver.solveLambdaForExpectedQuantity(expectedQuantity);
            candidateSpeeds[i] = solver.fisherSpeedWithRespectToBeta();
        }
        double[] cumulativeArcLength = FisherArcLength.cumulativeArcLength(candidateBetas, candidateSpeeds);
        double totalArcLength = cumulativeArcLength[candidateCount - 1];
        int rowCount = params.targetRowsPerConstraintPreset();
        double[] rowBetas = new double[rowCount];
        for (int row = 0; row < rowCount; row++) {
            double targetArcLength = totalArcLength * row / (rowCount - 1);
            rowBetas[row] = FisherArcLength.positionAtArcLength(candidateBetas, cumulativeArcLength, targetArcLength);
        }
        rowBetas[0] = candidateBetas[0];
        rowBetas[rowCount - 1] = candidateBetas[candidateCount - 1];
        return rowBetas;
    }

    /**
     * Smallest |beta| at which any category weight changes by more than about 0.1% at any quantity, i.e.
     * 1e-3 / (max |quantity| * value span). Below this the category distributions are indistinguishable from beta = 0,
     * so beta candidates need not go lower. The beta range [this, saturation] can span many decades.
     */
    private static double smallestRelevantBetaMagnitude(QuantitySet quantities, CategoryFamily categoryFamily) {
        return 1e-3 / (quantities.maxMagnitude * categoryFamily.scaledValueSpan());
    }

    /**
     * |beta| on the given side at which every non-null state is saturated: t = beta * q reaches the saturation point
     * of its branch for each nonzero q. On each side, the quantities of smallest magnitude are the last to saturate.
     */
    private static double saturationBeta(QuantitySet quantities, CategoryFamily categoryFamily, int signOfBeta) {
        // t = beta * q has the sign of beta for positive quantities and the opposite sign for negative ones.
        double forPositiveQuantities = categoryFamily.saturationTau(signOfBeta) / quantities.smallestPositive;
        double forNegativeQuantities = categoryFamily.saturationTau(-signOfBeta) / quantities.smallestNegativeMagnitude;
        if (Double.isNaN(forPositiveQuantities)) return forNegativeQuantities;
        if (Double.isNaN(forNegativeQuantities)) return forPositiveQuantities;
        return Math.max(forPositiveQuantities, forNegativeQuantities);
    }

    /** Magnitudes from lowest to highest, geometrically spaced (lowest is clamped below highest). */
    private static double[] logSpacedMagnitudes(double lowest, double highest, int count) {
        double effectiveLowest = Math.min(lowest, highest * 1e-6);
        double[] magnitudes = new double[count];
        for (int i = 0; i < count; i++)
            magnitudes[i] = effectiveLowest * Math.pow(highest / effectiveLowest, i / (double) (count - 1));
        return magnitudes;
    }
}
