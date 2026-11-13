package io.github.aevns.fastmaxent;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import static java.lang.Math.*;

import io.github.aevns.fastmaxent.priors.DensityPrior;
import io.github.aevns.fastmaxent.priors.Axis;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Consumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

@DisplayName("Tests for the public MaxEntSampler class")
public class MaxEntSamplerTest {
    static final long SEED = 0;
    static final String EMC_DATA_PATH = "/mc_pe_emc.txt";
    static final int DRAWS = 500_000;
    static final int COLD_DRAWS = 1000;
    static final int S_STEPS = 9;
    static final double TOLERANCE_SIGMA = 5;
    static final double TOLERANCE_EPSILON = 1e-9;
    static volatile long sink = 0;
    /** 0..64, with 0 the null state. */
    static final double[] DEFAULT_QUANTITIES = IntStream.rangeClosed(0, 64).asDoubleStream().toArray();
    static final double[] DEFAULT_EXPECTED_QUANTITIES = {0.25, 1, 4, 16};
    /** A threshold of 10 suits the EMC data. */
    static final Axis VALUE_AXIS = new Axis.Symlog(10), QUANTITY_AXIS = new Axis.Symlog(1);
    static final DensityPrior CATEGORY_PRIOR = new DensityPrior(VALUE_AXIS, 0.15, 0.1, false, false, 1);
    static final DensityPrior QUANTITY_PRIOR = new DensityPrior(QUANTITY_AXIS, 0.15, 1, true, true, 1);

    static final double[] EMC_VALUES = loadValues();

    static double[] loadValues() {
        try (var stream = MaxEntSamplerTest.class.getResourceAsStream(EMC_DATA_PATH)) {
            assertNotNull(stream);
            return new BufferedReader(new InputStreamReader(stream))
                    .lines().mapToDouble(Double::parseDouble).toArray();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static final class Case {
        final String name;
        double[] values = EMC_VALUES;
        double[] quantities = DEFAULT_QUANTITIES, expectedQuantities = DEFAULT_EXPECTED_QUANTITIES;
        DensityPrior categoryPrior = CATEGORY_PRIOR, quantityPrior = QUANTITY_PRIOR;
        TableParams params = TableParams.DEFAULT;

        Case(String name, Consumer<Case> changes) {
            this.name = name;
            changes.accept(this);
        }

        MaxEntSampler build() {
            return new MaxEntSampler(values, categoryPrior.weights(values), quantities,
                    quantityPrior.weights(quantities), expectedQuantities, params);
        }

        @Override public String toString() { return name; }
    }

    static Stream<Case> cases() {
        double[] twelveDecades = new double[241];
        for (int i = 0; i < twelveDecades.length; i++) twelveDecades[i] = pow(10, 12.0 * i / (twelveDecades.length - 1));
        return Stream.of(
                new Case("default", c -> {}),
                new Case("categoryBlockSize = 1",
                        c -> c.params = new TableParams(1e-5, 256, 1, 20_000_000L, 3000)),
                new Case("categoryBlockSize = 5 (partial last block)",
                        c -> c.params = new TableParams(1e-5, 256, 5, 20_000_000L, 3000)),
                new Case("minimal categoryTableBudgetBytes",
                        c -> c.params = new TableParams(1e-5, 256, 8, 1, 3000)),
                new Case("category similarity = 0",
                        c -> c.categoryPrior = new DensityPrior(
                                VALUE_AXIS, 0.15, 0, false, false, 1)),
                new Case("category similarity = 1",
                        c -> c.categoryPrior = new DensityPrior(
                                VALUE_AXIS, 0.15, 1, false, false, 1)),
                new Case("null-state weight = 0.000_01",
                        c -> c.quantityPrior = new DensityPrior(
                                QUANTITY_AXIS, 0.15, 1, true, true, 0.000_01)),
                new Case("null-state weight = 100_000",
                        c -> c.quantityPrior = new DensityPrior(
                                QUANTITY_AXIS, 0.15, 1, true, true, 100_000)),
                new Case("two unique values",
                        c -> c.values = new double[] {1, 1, 1, 2, 2}),
                new Case("values spanning 12 decades",
                        c -> c.values = twelveDecades),
                new Case("non-integer quantities",
                        c -> {
                    c.quantities = new double[] {0, 0.5, 1, 1.5, 2.5, 4, 6.5};
                    c.expectedQuantities = new double[] {0.25, 1, 3};
                }),
                new Case("negative quantities",
                        c -> {
                    c.quantities = IntStream.rangeClosed(-16, 0).asDoubleStream().toArray();
                    c.expectedQuantities = new double[] {-8, -1, -0.25};
                }),
                new Case("mixed-sign quantities",
                        c -> {
                    c.quantities = new double[] {-4, -2, -1, 0, 1, 2, 4};
                    c.expectedQuantities = new double[] {-1, 0, 1.5};
                }),
                new Case("no null state",
                        c -> {
                    c.quantities = IntStream.rangeClosed(1, 64).asDoubleStream().toArray();
                    c.expectedQuantities = new double[] {1.5, 4, 16};
                }),
                new Case("unsorted quantities",
                        c -> {
                    c.quantities = new double[] {4, 0, 2, 1, 3};
                    c.expectedQuantities = new double[] {1, 2.5};
                })
        );
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    @DisplayName("Testing that <N> and <NV> match their targets across a range of n and s for each test case")
    void meanTests(Case configuration) {
        SplittableRandom random = new SplittableRandom(SEED);
        double[] values = configuration.values, quantities = configuration.quantities;
        double[] expectedQuantities = configuration.expectedQuantities;
        MaxEntSampler sampler = configuration.build();
        List<Executable> checks = new ArrayList<>();
        for (int n = 0; n < expectedQuantities.length; n++) {
            double setN = expectedQuantities[n];
            double minS = sampler.minTargetS(n), maxS = sampler.maxTargetS(n);
            for (int step = 0; step < S_STEPS; step++) {
                double setS = minS + (maxS - minS) * step / (S_STEPS - 1);
                // Welford's running mean and variance
                double meanN = 0, m2N = 0, meanNV = 0, m2NV = 0;
                int inconsistentSamples = 0;
                for (int j = 1; j <= DRAWS; j++) {
                    var sample = sampler.sample(n, setS, random);
                    // Read the quantity through the returned index, so a wrong index fails the moment checks.
                    double quantity = quantities[sample.quantityIndex];
                    boolean nullMismatch = (sample.categoryIndex < 0) != (quantity == 0);
                    if (sample.quantity != quantity || nullMismatch) inconsistentSamples++;
                    double nv = sample.categoryIndex >= 0 ? quantity * values[sample.categoryIndex] : 0;
                    double dN = quantity - meanN;
                    meanN += dN / j;
                    m2N += dN * (quantity - meanN);
                    double dNV = nv - meanNV;
                    meanNV += dNV / j;
                    m2NV += dNV * (nv - meanNV);
                }
                // 5 sigma tolerance isn't foolproof; it should be fine, but outliers can happen.
                double toleranceN = TOLERANCE_SIGMA * sqrt(m2N / (DRAWS - 1) / DRAWS) + TOLERANCE_EPSILON * abs(setN);
                double toleranceNV = TOLERANCE_SIGMA * sqrt(m2NV / (DRAWS - 1) / DRAWS) + TOLERANCE_EPSILON * abs(setS);
                double observedN = meanN, observedNV = meanNV;
                int inconsistent = inconsistentSamples;
                String cell = String.format("n = %s, s = %.6g", setN, setS);
                checks.add(() -> assertEquals(setN, observedN, toleranceN, "<N> at " + cell));
                checks.add(() -> assertEquals(setS, observedNV, toleranceNV, "<NV> at " + cell));
                checks.add(() -> assertEquals(0, inconsistent,
                        "samples whose quantity, quantityIndex and null-state category disagree at " + cell));
            }
        }
        assertAll(configuration.name, checks);
    }

    @Test
    @DisplayName("Testing that entries sharing a value are drawn in proportion to their category weights")
    void withinGroupWeights() {
        double[] values = {1, 1, 1, 2, 2};
        double[] categoryWeights = {1, 2, 3, 4, 4};
        MaxEntSampler sampler = new MaxEntSampler(values, categoryWeights, new double[] {0, 1}, new double[] {1, 1},
                new double[] {0.5}, TableParams.DEFAULT);
        SplittableRandom random = new SplittableRandom(SEED);
        double s = 0.5 * (sampler.minTargetS(0) + sampler.maxTargetS(0));
        long[] hits = new long[values.length];
        for (int i = 0; i < DRAWS; i++) {
            var sample = sampler.sample(0, s, random);
            if (sample.categoryIndex >= 0) hits[sample.categoryIndex]++;
        }
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
        MaxEntSampler sampler = new Case("default", c -> {}).build();
        long loadTimer = System.nanoTime() - start;

        int[] nSet = new int[DRAWS];
        double[] sSet = new double[DRAWS];

        for (int i = 0; i < DRAWS; i++) {
            nSet[i] = random.nextInt(DEFAULT_EXPECTED_QUANTITIES.length);
            sSet[i] = random.nextDouble(sampler.minTargetS(nSet[i]), sampler.maxTargetS(nSet[i]));
        }

        //warmup
        long dummy = 0;
        for (int i = 0; i < DRAWS; i++) {
            MaxEntSampler.Sample sample = sampler.sample(nSet[i], sSet[i], random);
            dummy += sample.quantityIndex + sample.categoryIndex;
        }

        //cold
        long coldTimer = 0;
        long[] distraction = new long[8 << 20]; // 64 MB, how much will flush the cache is machine dependent!
        for (int i = 0; i < COLD_DRAWS; i++) {
            // 8 longs per 64 B cache line
            for (int j = 0; j < distraction.length; j += 8) dummy += distraction[j];
            int n = nSet[i];
            double s = sSet[i];
            start = System.nanoTime();
            MaxEntSampler.Sample sample = sampler.sample(n, s, random);
            coldTimer += System.nanoTime() - start;
            dummy += sample.quantityIndex + sample.categoryIndex;
        }

        //warm
        start = System.nanoTime();
        for (int i = 0; i < DRAWS; i++) {
            MaxEntSampler.Sample sample = sampler.sample(nSet[i], sSet[i], random);
            dummy += sample.quantityIndex + sample.categoryIndex;
        }
        long warmTimer = System.nanoTime() - start;

        sink = dummy;
        System.out.print("sampler load time: " + loadTimer / 1_000_000 + " ms\n");
        System.out.print("time per sample (cold): " + (double) coldTimer / COLD_DRAWS + " ns\n");
        System.out.print("time per sample (warm): " + (double) warmTimer / DRAWS + " ns\n");
    }
}