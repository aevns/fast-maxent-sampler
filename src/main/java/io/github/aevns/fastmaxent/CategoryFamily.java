package io.github.aevns.fastmaxent;

/**
 * The category family p(k | t) ~ W_k exp(-t x_k) over both signs of t, and the {@link CategoryTable} rows that
 * approximate it. Gives the log-partition, mean and variance at any t, and the two rows to mix to match the mean.
 */
final class CategoryFamily {
    /** Everything the quantity side needs to know about the category family at one t. */
    static final class Point {
        /** ln sum_k W_k exp(-t (x_k - referenceValue)) in the sign-appropriate gap form (see CategoryTiltBranch). */
        double logPartition;
        /** x_ref of logPartition: the smallest scaled value for t >= 0, the largest for t < 0. */
        double referenceValue;
        /** Var[x | t]. */
        double varianceScaledValue;
        /** E[x | t] as realized by mixing the two bracketing rows with mixThreshold. */
        double meanScaledValue;
        /** Lower bracketing stored row (global row index). The upper row is lowerCategoryRow + 1. */
        int lowerCategoryRow;
        /** Fixed-point probability of using the upper row instead of the lower one. */
        int mixThreshold;
        final double[] interpolationScratch = new double[3];
    }

    private final CategoryTiltBranch positiveBranch, negativeBranch;
    /** Rows [0, negativeRowOffset) belong to the positive-t branch, the rest to the negative-t branch. */
    private final int negativeRowOffset;
    private final CategoryTable table;
    private final double scaledValueSpan;

    CategoryFamily(double[] scaledValues, double[] uniqueWeights, TableParams params) {
        scaledValueSpan = scaledValues[scaledValues.length - 1] - scaledValues[0];
        positiveBranch = new CategoryTiltBranch(+1, scaledValues, uniqueWeights);
        negativeBranch = new CategoryTiltBranch(-1, scaledValues, uniqueWeights);
        for (CategoryTiltBranch branch : new CategoryTiltBranch[] {positiveBranch, negativeBranch}) {
            branch.findSaturationTau(params.saturationTolerance());
            branch.buildAdaptiveGrid();
        }
        long bytesPerCategoryRow = 4L * CategoryTable.blocksPerRow(scaledValues.length, params.categoryBlockSize());
        int totalCategoryRows =
                (int) Math.max(32, Math.min(2000, params.categoryTableBudgetBytes() / bytesPerCategoryRow));
        int categoryRowsPerBranch = Math.max(8, totalCategoryRows / 2);
        positiveBranch.selectCategoryRowNodes(categoryRowsPerBranch);
        negativeBranch.selectCategoryRowNodes(categoryRowsPerBranch);
        negativeRowOffset = positiveBranch.categoryRowTau.length;
        double[] rowSignedT = new double[negativeRowOffset + negativeBranch.categoryRowTau.length];
        for (int row = 0; row < rowSignedT.length; row++) {
            CategoryTiltBranch branch = row < negativeRowOffset ? positiveBranch : negativeBranch;
            int branchRow = row < negativeRowOffset ? row : row - negativeRowOffset;
            rowSignedT[row] = branch.signOfT * branch.categoryRowTau[branchRow];
        }
        table = new CategoryTable(scaledValues, uniqueWeights, rowSignedT, params.categoryBlockSize());
    }

    CategoryTable table() { return table; }

    /** Smallest tau = |t| on the branch with the given sign of t at which the distribution is saturated. */
    double saturationTau(int signOfT) { return (signOfT > 0 ? positiveBranch : negativeBranch).saturationTau; }

    /** Largest minus smallest scaled value. */
    double scaledValueSpan() { return scaledValueSpan; }

    /** Evaluates the family at signed t (|t| beyond the saturation point is treated as saturated). */
    void evaluateAtT(double signedT, Point out) {
        boolean positive = signedT >= 0;
        CategoryTiltBranch branch = positive ? positiveBranch : negativeBranch;
        int rowOffset = positive ? 0 : negativeRowOffset;
        double[] categoryRowTau = branch.categoryRowTau;
        int lastRowPosition = categoryRowTau.length - 1;
        double tau = Math.min(Math.abs(signedT), branch.saturationTau);
        branch.interpolate(tau, out.interpolationScratch);
        out.logPartition = out.interpolationScratch[0];
        out.referenceValue = branch.referenceValue;
        out.varianceScaledValue = out.interpolationScratch[2];

        int lowerCategoryRow;
        double effectiveMeanGap;
        int mixThreshold = 0;
        if (tau >= categoryRowTau[lastRowPosition]) {
            lowerCategoryRow = lastRowPosition;
            effectiveMeanGap = branch.signOfT
                    * (table.rowMeanScaledValue(rowOffset + lastRowPosition) - branch.referenceValue);
        } else {
            lowerCategoryRow = SortedSearch.bracket(categoryRowTau, tau);
            // Mix the two stored rows so the realized mean gap equals the exact interpolated mean gap at tau.
            double meanGapAtLowerRow = branch.signOfT
                    * (table.rowMeanScaledValue(rowOffset + lowerCategoryRow) - branch.referenceValue);
            double meanGapAtUpperRow = branch.signOfT
                    * (table.rowMeanScaledValue(rowOffset + lowerCategoryRow + 1) - branch.referenceValue);
            double mixWeight = meanGapAtLowerRow > meanGapAtUpperRow
                    ? (meanGapAtLowerRow - out.interpolationScratch[1]) / (meanGapAtLowerRow - meanGapAtUpperRow)
                    : 0;
            mixThreshold = FixedPoint.fromProbability(Math.min(1, Math.max(0, mixWeight)));
            effectiveMeanGap = meanGapAtLowerRow
                    - FixedPoint.toProbability(mixThreshold) * (meanGapAtLowerRow - meanGapAtUpperRow);
        }
        out.lowerCategoryRow = rowOffset + lowerCategoryRow;
        out.mixThreshold = mixThreshold;
        out.meanScaledValue = branch.referenceValue + branch.signOfT * effectiveMeanGap;
    }
}
