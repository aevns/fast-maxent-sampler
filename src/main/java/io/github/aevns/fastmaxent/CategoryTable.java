package io.github.aevns.fastmaxent;

/**
 * Stored category distributions p(k | t) ~ W_k exp(-t x_k), one row per t. Each row is a CDF over blocks of unique
 * values; a draw finds the block, then picks within it by recomputing its few weights, so memory shrinks by the block
 * size with no approximation beyond 32-bit quantization.
 */
final class CategoryTable {
    private final int uniqueCount, blockSize, blocksPerRow;
    /** Interleaved [W_0, x_0, W_1, x_1, ...] for cache-friendly block scans. */
    private final double[] interleavedWeightAndScaledValue;
    /** rowCount x blocksPerRow; each row is a {@link FixedPointCdf} over its blocks. */
    private final int[] blockCdf;
    private final double[] rowT;
    /** Realized E[x] of each row under the quantized CDF. */
    private final double[] rowMeanScaledValue;

    static int blocksPerRow(int uniqueCount, int blockSize) {
        return (uniqueCount + blockSize - 1) / blockSize;
    }

    CategoryTable(double[] scaledValues, double[] uniqueWeights, double[] rowT, int blockSize) {
        this.uniqueCount = scaledValues.length;
        this.blockSize = blockSize;
        this.blocksPerRow = blocksPerRow(uniqueCount, blockSize);
        this.rowT = rowT;
        interleavedWeightAndScaledValue = new double[2 * uniqueCount];
        for (int k = 0; k < uniqueCount; k++) {
            interleavedWeightAndScaledValue[2 * k] = uniqueWeights[k];
            interleavedWeightAndScaledValue[2 * k + 1] = scaledValues[k];
        }
        blockCdf = new int[rowT.length * blocksPerRow];
        buildBlockCdf();
        rowMeanScaledValue = new double[rowT.length];
        for (int row = 0; row < rowT.length; row++) rowMeanScaledValue[row] = computeRowMeanScaledValue(row);
    }

    int rowCount() { return rowT.length; }

    int blockSize() { return blockSize; }

    double rowMeanScaledValue(int row) { return rowMeanScaledValue[row]; }

    long approximateMemoryBytes() {
        return 4L * blockCdf.length + 8L * interleavedWeightAndScaledValue.length;
    }

    private double referenceValue(double t, int blockStart, int blockEnd) {
        // Anchoring at the value that dominates for this sign of t keeps every exponent <= 0.
        return t >= 0
                ? interleavedWeightAndScaledValue[2 * blockStart + 1]
                : interleavedWeightAndScaledValue[2 * (blockEnd - 1) + 1];
    }

    private void buildBlockCdf() {
        double[] weights = new double[uniqueCount];
        double[] blockCumulativeWeight = new double[blocksPerRow];
        for (int row = 0; row < rowT.length; row++) {
            double t = rowT[row];
            double reference = referenceValue(t, 0, uniqueCount);
            double totalWeight = 0;
            for (int k = 0; k < uniqueCount; k++) {
                weights[k] = interleavedWeightAndScaledValue[2 * k]
                        * Math.exp(-t * (interleavedWeightAndScaledValue[2 * k + 1] - reference));
                totalWeight += weights[k];
            }
            double cumulativeWeight = 0;
            for (int block = 0; block < blocksPerRow; block++) {
                int blockStart = block * blockSize, blockEnd = Math.min(blockStart + blockSize, uniqueCount);
                for (int k = blockStart; k < blockEnd; k++) cumulativeWeight += weights[k];
                blockCumulativeWeight[block] = cumulativeWeight;
            }
            FixedPointCdf.encode(blockCumulativeWeight, 0, blocksPerRow, totalWeight, blockCdf, row * blocksPerRow);
        }
    }

    /**
     * Draws a unique-value index from the given row. The high 32 bits of the random long locate the block in
     * the coarse CDF; the low 32 bits pick within the block.
     *
     * @param blockWeights scratch space of at least {@link #blockSize()} entries; overwritten
     */
    int drawUniqueIndex(int row, long random, double[] blockWeights) {
        int block = FixedPointCdf.search(blockCdf, row * blocksPerRow, blocksPerRow, FixedPoint.highBits(random));
        int blockStart = block * blockSize, blockEnd = Math.min(blockStart + blockSize, uniqueCount);
        double t = rowT[row];
        double reference = referenceValue(t, blockStart, blockEnd);
        double blockTotal = 0;
        for (int k = blockStart; k < blockEnd; k++) {
            double weight = interleavedWeightAndScaledValue[2 * k]
                    * Math.exp(-t * (interleavedWeightAndScaledValue[2 * k + 1] - reference));
            blockWeights[k - blockStart] = weight;
            blockTotal += weight;
        }
        double target = FixedPoint.lowBits(random) * (1.0 / FixedPoint.SCALE) * blockTotal;
        int chosen = blockEnd - 1;
        double accumulated = 0;
        for (int k = blockStart; k < blockEnd; k++) {
            accumulated += blockWeights[k - blockStart];
            if (accumulated > target) { chosen = k; break; }
        }
        return chosen;
    }

    /**
     * The exact distribution over unique values that drawUniqueIndex realizes for this row: each block's
     * mass comes from the quantized CDF, and within a block mass is proportional to the prior-times-tilt weights.
     */
    double[] rowDistribution(int row) {
        double[] distribution = new double[uniqueCount];
        double t = rowT[row];
        double previousCdf = 0;
        for (int block = 0; block < blocksPerRow; block++) {
            double currentCdf = FixedPointCdf.probabilityAt(blockCdf, row * blocksPerRow + block);
            double blockMass = currentCdf - previousCdf;
            previousCdf = currentCdf;
            if (blockMass <= 0) continue;
            int blockStart = block * blockSize, blockEnd = Math.min(blockStart + blockSize, uniqueCount);
            double reference = referenceValue(t, blockStart, blockEnd);
            double totalWeight = 0;
            for (int k = blockStart; k < blockEnd; k++) {
                distribution[k] = interleavedWeightAndScaledValue[2 * k]
                        * Math.exp(-t * (interleavedWeightAndScaledValue[2 * k + 1] - reference));
                totalWeight += distribution[k];
            }
            for (int k = blockStart; k < blockEnd; k++) distribution[k] *= blockMass / totalWeight;
        }
        return distribution;
    }

    /** Exact E[x | row] under the stored (quantized) block CDF. */
    private double computeRowMeanScaledValue(int row) {
        double[] distribution = rowDistribution(row);
        double mean = 0;
        for (int k = 0; k < uniqueCount; k++) mean += distribution[k] * interleavedWeightAndScaledValue[2 * k + 1];
        return mean;
    }
}
