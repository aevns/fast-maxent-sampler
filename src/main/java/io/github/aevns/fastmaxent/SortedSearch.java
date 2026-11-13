package io.github.aevns.fastmaxent;

/** Binary search over ascending arrays. */
final class SortedSearch {
    private SortedSearch() {}

    /**
     * The interval of an ascending array (length >= 2) that contains x: the i in [0, length - 2] with
     * ascending[i] <= x < ascending[i + 1], clamped to that range when x lies outside the array.
     */
    static int bracket(double[] ascending, double x) {
        int low = 0, high = ascending.length - 1;
        while (high - low > 1) {
            int middle = (low + high) >>> 1;
            if (ascending[middle] <= x) low = middle; else high = middle;
        }
        return low;
    }
}
