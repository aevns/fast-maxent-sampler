package io.github.aevns.fastmaxent;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
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
import static org.junit.jupiter.api.Assertions.assertNotNull;

@DisplayName("Tests for the public ValueSampler class")
class ValueSamplerTest {
    static final long SEED = 0;
    static final int DRAWS = 200_000;
    static final int TARGET_STEPS = 7;
    static final double TOLERANCE_SIGMA = 5;
    static final double TOLERANCE_EPSILON = 1e-9;
    static final String EMC_DATA_PATH = "/mc_pe_emc.txt";
    static final int COLD_DRAWS = 1000;
    static volatile long sink = 0;

    /** 40 scaled values over three decades. */
    static final double[] VALUES = IntStream.range(0, 40).mapToDouble(i -> Math.pow(10, 3.0 * i / 39)).toArray();
    /** Mixed-sign values, including 0, such as energies measured relative to an arbitrary zero point. */
    static final double[] MIXED_SIGN_VALUES = {-100, -50, -10, -1, -0.5, 0, 0.5, 1, 10, 50, 100};
    static final DensityPrior CATEGORY_PRIOR = new DensityPrior(new Axis.Symlog(10), 0.15, 0.1, false, false);

    static final double[] EMC_VALUES = loadValues();

    static double[] loadValues() {
        try (var stream = ValueSamplerTest.class.getResourceAsStream(EMC_DATA_PATH)) {
            assertNotNull(stream);
            return new BufferedReader(new InputStreamReader(stream))
                    .lines().mapToDouble(Double::parseDouble).toArray();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    record Config(String name, double[] values) {
        @Override public String toString() { return name; }
    }

    static Stream<Config> configs() {
        return Stream.of(
                new Config("log-spaced positive values", VALUES),
                new Config("mixed-sign values", MIXED_SIGN_VALUES));
    }

    static ValueSampler build(double[] values) {
        return new ValueSampler(values, CATEGORY_PRIOR.weights(values), TableParams.DEFAULT);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("configs")
    @DisplayName("Testing that E[V] matches its target across the supported range")
    void meanTests(Config config) {
        double[] values = config.values();
        SplittableRandom random = new SplittableRandom(SEED);
        ValueSampler sampler = build(values);
        double minTarget = sampler.minTargetValue(), maxTarget = sampler.maxTargetValue();
        List<Executable> checks = new ArrayList<>();
        for (int step = 0; step < TARGET_STEPS; step++) {
            double target = minTarget + (maxTarget - minTarget) * (step + 1) / (TARGET_STEPS + 1);
            // Welford's running mean and variance
            double mean = 0, m2 = 0;
            for (int j = 1; j <= DRAWS; j++) {
                double value = values[sampler.sample(target, random)];
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
    @DisplayName("Testing that entries sharing a value are drawn in proportion to their category weights")
    void withinGroupWeights() {
        double[] values = {1, 1, 1, 2, 2};
        double[] categoryWeights = {1, 2, 3, 4, 4};
        ValueSampler sampler = new ValueSampler(values, categoryWeights, TableParams.DEFAULT);
        SplittableRandom random = new SplittableRandom(SEED);
        double target = 0.5 * (sampler.minTargetValue() + sampler.maxTargetValue());
        long[] hits = new long[values.length];
        for (int i = 0; i < DRAWS; i++) hits[sampler.sample(target, random)]++;
        int[][] groups = {{0, 1, 2}, {3, 4}};
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

    @Tag("profile")
    @Test
    @DisplayName("Profiling precomputation and sampling time")
    void timingProfile() {
        SplittableRandom random = new SplittableRandom(SEED);

        long start = System.nanoTime();
        ValueSampler sampler = build(EMC_VALUES);
        long loadTimer = System.nanoTime() - start;

        double minTarget = sampler.minTargetValue(), maxTarget = sampler.maxTargetValue();
        double[] targetSet = new double[DRAWS];
        for (int i = 0; i < DRAWS; i++) targetSet[i] = random.nextDouble(minTarget, maxTarget);

        //warmup
        long dummy = 0;
        for (int i = 0; i < DRAWS; i++) dummy += sampler.sample(targetSet[i], random);

        //cold
        long coldTimer = 0;
        long[] distraction = new long[8 << 20]; // 64 MB, how much will flush the cache is machine dependent!
        for (int i = 0; i < COLD_DRAWS; i++) {
            // 8 longs per 64 B cache line
            for (int j = 0; j < distraction.length; j += 8) dummy += distraction[j];
            double target = targetSet[i];
            start = System.nanoTime();
            int index = sampler.sample(target, random);
            coldTimer += System.nanoTime() - start;
            dummy += index;
        }

        //warm
        start = System.nanoTime();
        for (int i = 0; i < DRAWS; i++) dummy += sampler.sample(targetSet[i], random);
        long warmTimer = System.nanoTime() - start;

        sink = dummy;
        System.out.print("sampler load time: " + loadTimer / 1_000_000 + " ms\n");
        System.out.print("time per sample (cold): " + (double) coldTimer / COLD_DRAWS + " ns\n");
        System.out.print("time per sample (warm): " + (double) warmTimer / DRAWS + " ns\n");
    }
}
