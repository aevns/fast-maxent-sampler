package io.github.aevns.fastmaxent;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import io.github.aevns.fastmaxent.priors.Axis;
import io.github.aevns.fastmaxent.priors.DensityPrior;

import static java.lang.Math.abs;
import static java.lang.Math.sqrt;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;

@DisplayName("Tests for the public ValueModel class")
class ValueModelTest {
    static final long SEED = 0;
    static final int DRAWS = 200_000;
    static final int TARGET_STEPS = 7;
    static final double TOLERANCE_SIGMA = 5;
    static final double TOLERANCE_EPSILON = 1e-9;

    /** 40 scaled values over three decades. */
    static final double[] VALUES = IntStream.range(0, 40).mapToDouble(i -> Math.pow(10, 3.0 * i / 39)).toArray();
    /** Mixed-sign values, including 0, such as energies measured relative to an arbitrary zero point. */
    static final double[] MIXED_SIGN_VALUES = {-100, -50, -10, -1, -0.5, 0, 0.5, 1, 10, 50, 100};
    static final DensityPrior CATEGORY_PRIOR = new DensityPrior(new Axis.Symlog(10), 0.15, 0.1, false, false);

    record Config(String name, double[] values) {
        @Override public String toString() { return name; }
    }

    static Stream<Config> configs() {
        return Stream.of(
                new Config("log-spaced positive values", VALUES),
                new Config("mixed-sign values", MIXED_SIGN_VALUES),
                new Config("unsorted log-spaced values", permuted(VALUES, permutation(VALUES.length, SEED))));
    }

    /** A fixed pseudo-random permutation of 0..size-1. */
    static int[] permutation(int size, long seed) {
        int[] order = IntStream.range(0, size).toArray();
        SplittableRandom random = new SplittableRandom(seed);
        for (int i = size - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int swap = order[i];
            order[i] = order[j];
            order[j] = swap;
        }
        return order;
    }

    /** result[i] = array[order[i]] */
    static double[] permuted(double[] array, int[] order) {
        double[] result = new double[order.length];
        for (int i = 0; i < order.length; i++) result[i] = array[order[i]];
        return result;
    }

    static ValueModel build(double[] values) {
        return new ValueModel(values, CATEGORY_PRIOR.weights(values), TableParams.DEFAULT);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("configs")
    @DisplayName("Testing that E[V] matches its target across the supported range")
    void meanTests(Config config) {
        double[] values = config.values();
        ValueModel model = build(values);
        ValueModel.Sampler sampler = model.newSampler(new SplittableRandom(SEED));
        double minTarget = model.minTargetValue(), maxTarget = model.maxTargetValue();
        List<Executable> checks = new ArrayList<>();
        for (int step = 0; step < TARGET_STEPS; step++) {
            double target = minTarget + (maxTarget - minTarget) * (step + 1) / (TARGET_STEPS + 1);
            // Welford's running mean and variance
            double mean = 0, m2 = 0;
            for (int j = 1; j <= DRAWS; j++) {
                double value = values[sampler.sample(target)];
                double delta = value - mean;
                mean += delta / j;
                m2 += delta * (value - mean);
            }
            // 5 sigma tolerance isn't foolproof; it should be fine, but outliers can happen.
            double tolerance = TOLERANCE_SIGMA * sqrt(m2 / (DRAWS - 1) / DRAWS) + TOLERANCE_EPSILON * abs(target);
            double observed = mean;
            String cell = String.format("target = %.6g", target);
            checks.add(() -> assertEquals(target, observed, tolerance, "E[V] at " + cell));
        }
        assertAll(checks);
    }

    @Test
    @DisplayName("Testing that unsorted values give the same draws as sorted ones, at the caller's indices")
    void unsortedValuesMatchSorted() {
        double[] sortedWeights = CATEGORY_PRIOR.weights(VALUES);
        int[] order = permutation(VALUES.length, SEED);
        ValueModel sorted = new ValueModel(VALUES, sortedWeights, TableParams.DEFAULT);
        ValueModel unsorted =
                new ValueModel(permuted(VALUES, order), permuted(sortedWeights, order), TableParams.DEFAULT);
        assertEquals(sorted.minTargetValue(), unsorted.minTargetValue());
        assertEquals(sorted.maxTargetValue(), unsorted.maxTargetValue());
        ValueModel.Sampler sortedSampler = sorted.newSampler(new SplittableRandom(SEED));
        ValueModel.Sampler unsortedSampler = unsorted.newSampler(new SplittableRandom(SEED));
        int mismatches = 0;
        for (int step = 0; step < TARGET_STEPS; step++) {
            double target = sorted.minTargetValue()
                    + (sorted.maxTargetValue() - sorted.minTargetValue()) * (step + 1) / (TARGET_STEPS + 1);
            for (int i = 0; i < 10_000; i++)
                if (sortedSampler.sample(target) != order[unsortedSampler.sample(target)]) mismatches++;
        }
        assertEquals(0, mismatches, "draws whose unsorted index does not map to the sorted draw");
    }

    record GroupCase(String name, double[] values, double[] categoryWeights, int[][] groups) {
        @Override public String toString() { return name; }
    }

    /** Groups mix equal weights (drawn uniformly) and unequal weights (drawn by CDF); both paths are covered. */
    static Stream<GroupCase> groupCases() {
        return Stream.of(
                new GroupCase("sorted", new double[] {1, 1, 1, 2, 2}, new double[] {1, 2, 3, 4, 4},
                        new int[][] {{0, 1, 2}, {3, 4}}),
                new GroupCase("unsorted", new double[] {2, 1, 1, 2, 1}, new double[] {4, 1, 2, 4, 3},
                        new int[][] {{1, 2, 4}, {0, 3}}));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("groupCases")
    @DisplayName("Testing that entries sharing a value are drawn in proportion to their category weights")
    void withinGroupWeights(GroupCase groupCase) {
        double[] values = groupCase.values(), categoryWeights = groupCase.categoryWeights();
        ValueModel model = new ValueModel(values, categoryWeights, TableParams.DEFAULT);
        ValueModel.Sampler sampler = model.newSampler(new SplittableRandom(SEED));
        double target = 0.5 * (model.minTargetValue() + model.maxTargetValue());
        long[] hits = new long[values.length];
        for (int i = 0; i < DRAWS; i++) hits[sampler.sample(target)]++;
        int[][] groups = groupCase.groups();
        List<Executable> checks = new ArrayList<>();
        for (int[] group : groups) {
            long groupHits = 0;
            double groupWeight = 0;
            for (int entry : group) { groupHits += hits[entry]; groupWeight += categoryWeights[entry]; }
            for (int entry : group) {
                double expected = categoryWeights[entry] / groupWeight;
                double observed = hits[entry] / (double) groupHits;
                double tolerance = TOLERANCE_SIGMA * sqrt(expected * (1 - expected) / groupHits);
                checks.add(() -> assertEquals(expected, observed, tolerance,
                        "share of entry " + entry + " in its group"));
            }
        }
        assertAll(checks);
    }
}
