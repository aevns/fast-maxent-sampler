package io.github.aevns.fastmaxent;

/**
 * Cumulative distributions over a run of consecutive slots, stored as unsigned 32-bit fixed point XOR
 * Integer.MIN_VALUE so the search can use signed comparison. A stored MAX_UNSIGNED stands for a CDF of 1 (2^32, which
 * 32 bits cannot hold); slots after the first such slot are empty, and the last slot always stores it.
 */
final class FixedPointCdf {
    private FixedPointCdf() {}

    /**
     * Encodes cumulative[from + i] / total for i in [0, count) into out[outOffset + i]. Rounding is kept monotone, and
     * the last slot is forced to 1 so the slots cover every key.
     */
    static void encode(double[] cumulative, int from, int count, double total, int[] out, int outOffset) {
        long previous = 0;
        for (int i = 0; i < count; i++) {
            long fixed = Math.round(cumulative[from + i] / total * FixedPoint.SCALE);
            if (fixed > FixedPoint.MAX_UNSIGNED) fixed = FixedPoint.MAX_UNSIGNED;
            if (fixed < previous) fixed = previous;
            if (i == count - 1) fixed = FixedPoint.MAX_UNSIGNED;
            previous = fixed;
            out[outOffset + i] = ((int) fixed) ^ Integer.MIN_VALUE;
        }
    }

    /**
     * Slot (relative to from) of the CDF stored in cdf[from, from + count) that the uniform 32-bit key falls in: the
     * first whose stored CDF exceeds it.
     */
    static int search(int[] cdf, int from, int count, long unsignedRandom32) {
        // Capped one below MAX_UNSIGNED: the slot that reaches the ceiling stores MAX_UNSIGNED but stands for 2^32,
        // so it must also take the top key. Uncapped, that key would fall through to the last slot, giving it 2^-32
        // of mass even when it should have none.
        int key = Math.min(((int) unsignedRandom32) ^ Integer.MIN_VALUE, Integer.MAX_VALUE - 1);
        int low = 0, high = count - 1;
        while (low < high) {
            int middle = (low + high) >>> 1;
            if (cdf[from + middle] > key) high = middle; else low = middle + 1;
        }
        return low;
    }

    /** The stored CDF at cdf[index] as a probability in [0, 1]; the ceiling decodes to exactly 1. */
    static double probabilityAt(int[] cdf, int index) {
        long stored = Integer.toUnsignedLong(cdf[index] ^ Integer.MIN_VALUE);
        return stored == FixedPoint.MAX_UNSIGNED ? 1.0 : stored / FixedPoint.SCALE;
    }
}
