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
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Consumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

@DisplayName("Tests for the public JointModel class")
public class JointModelTest {
    static final long SEED = 0;
    static final String EMC_DATA_PATH = "/mc_pe_emc.txt";
    static final int DRAWS = 500_000;
    static final int S_STEPS = 9;
    static final double TOLERANCE_SIGMA = 5;
    static final double TOLERANCE_EPSILON = 1e-9;
    static final double[] DEFAULT_QUANTITIES = IntStream.rangeClosed(1, 64).asDoubleStream().toArray();
    static final double[] DEFAULT_EXPECTED_QUANTITIES = {0.25, 1, 4, 16};
    /** A threshold of 10 suits the EMC data. */
    static final Axis VALUE_AXIS = new Axis.Symlog(10), QUANTITY_AXIS = new Axis.Symlog(1);
    static final DensityPrior CATEGORY_PRIOR = new DensityPrior(VALUE_AXIS, 0.15, 0.1, false, false);
    static final DensityPrior QUANTITY_PRIOR = new DensityPrior(QUANTITY_AXIS, 0.15, 1, false, true);

    static final double[] EMC_VALUES = loadValues();

    static double[] loadValues() {
        try (var stream = JointModelTest.class.getResourceAsStream(EMC_DATA_PATH)) {
            assertNotNull(stream);
            return new BufferedReader(new InputStreamReader(stream))
                    .lines().mapToDouble(Double::parseDouble).toArray();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static final class Config {
        final String name;
        double[] values = EMC_VALUES;
        double[] quantities = DEFAULT_QUANTITIES, expectedQuantities = DEFAULT_EXPECTED_QUANTITIES;
        DensityPrior categoryPrior = CATEGORY_PRIOR, quantityPrior = QUANTITY_PRIOR;
        /** null means derive the null-state weight from quantityPrior itself, by treating it as an implied 0. */
        Double nullStateWeight = null;
        TableParams params = TableParams.DEFAULT;

        Config(String name, Consumer<Config> changes) {
            this.name = name;
            changes.accept(this);
        }

        JointModel build() {
            double[] quantitiesWithImpliedZero = Arrays.copyOf(quantities, quantities.length + 1);
            quantitiesWithImpliedZero[quantities.length] = 0;
            double[] weightsWithImpliedZero = quantityPrior.weights(quantitiesWithImpliedZero);
            double[] quantityPriorWeights = Arrays.copyOf(weightsWithImpliedZero, quantities.length);
            double resolvedNullStateWeight =
                    nullStateWeight != null ? nullStateWeight : weightsWithImpliedZero[quantities.length];
            return new JointModel(values, categoryPrior.weights(values), quantities,
                    quantityPriorWeights, resolvedNullStateWeight, expectedQuantities, params);
        }

        @Override public String toString() { return name; }
    }

    static Stream<Config> configs() {
        double[] twelveDecades = new double[241];
        for (int i = 0; i < twelveDecades.length; i++) twelveDecades[i] = pow(10, 12.0 * i / (twelveDecades.length - 1));
        return Stream.of(
                new Config("default", c -> {}),
                new Config("categoryBlockSize = 1",
                        c -> c.params = new TableParams(1e-5, 256, 1, 20_000_000L, 3000)),
                new Config("categoryBlockSize = 5 (partial last block)",
                        c -> c.params = new TableParams(1e-5, 256, 5, 20_000_000L, 3000)),
                new Config("minimal categoryTableBudgetBytes",
                        c -> c.params = new TableParams(1e-5, 256, 8, 1, 3000)),
                new Config("category similarity = 0",
                        c -> c.categoryPrior = new DensityPrior(VALUE_AXIS, 0.15, 0, false, false)),
                new Config("category similarity = 1",
                        c -> c.categoryPrior = new DensityPrior(VALUE_AXIS, 0.15, 1, false, false)),
                new Config("null-state weight = 0.000_01", c -> c.nullStateWeight = 0.000_01),
                new Config("null-state weight = 100_000", c -> c.nullStateWeight = 100_000.0),
                new Config("two unique values",
                        c -> c.values = new double[] {1, 1, 1, 2, 2}),
                new Config("values spanning 12 decades",
                        c -> c.values = twelveDecades),
                new Config("mixed-sign category values",
                        c -> c.values = new double[] {-100, -50, -10, -1, -0.5, 0, 0.5, 1, 10, 50, 100}),
                new Config("unsorted category values",
                        c -> c.values = ValueModelTest.permuted(EMC_VALUES,
                                ValueModelTest.permutation(EMC_VALUES.length, SEED))),
                new Config("non-integer quantities",
                        c -> {
                    c.quantities = new double[] {0, 0.5, 1, 1.5, 2.5, 4, 6.5};
                    c.expectedQuantities = new double[] {0.25, 1, 3};
                }),
                new Config("negative quantities",
                        c -> {
                    c.quantities = IntStream.rangeClosed(-16, 0).asDoubleStream().toArray();
                    c.expectedQuantities = new double[] {-8, -1, -0.25};
                }),
                new Config("mixed-sign quantities",
                        c -> {
                    c.quantities = new double[] {-4, -2, -1, 0, 1, 2, 4};
                    c.expectedQuantities = new double[] {-1, 0, 1.5};
                }),
                new Config("no null state",
                        c -> {
                    c.quantities = IntStream.rangeClosed(1, 64).asDoubleStream().toArray();
                    c.expectedQuantities = new double[] {1.5, 4, 16};
                    c.nullStateWeight = 0.0;
                }),
                new Config("repeated quantities",
                        c -> {
                    c.quantities = new double[] {0, 1, 1, 2, 3, 3, 3, 4};
                    c.expectedQuantities = new double[] {1, 2.5};
                }),
                new Config("unsorted quantities",
                        c -> {
                    c.quantities = new double[] {4, 0, 2, 1, 3};
                    c.expectedQuantities = new double[] {1, 2.5};
                })
        );
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("configs")
    @DisplayName("Testing that E[Q] and E[Q V] match their targets across a range of n and s for each test case")
    void meanTests(Config config) {
        double[] values = config.values, quantities = config.quantities;
        double[] expectedQuantities = config.expectedQuantities;
        JointModel model = config.build();
        JointModel.Sampler sampler = model.newSampler(new SplittableRandom(SEED));
        List<Executable> checks = new ArrayList<>();
        for (int n = 0; n < expectedQuantities.length; n++) {
            double setN = expectedQuantities[n];
            double minS = model.minTargetS(n), maxS = model.maxTargetS(n);
            for (int step = 0; step < S_STEPS; step++) {
                double setS = minS + (maxS - minS) * step / (S_STEPS - 1);
                // Welford's running mean and variance
                double meanQ = 0, m2Q = 0, meanQV = 0, m2QV = 0;
                int inconsistentSamples = 0;
                for (int j = 1; j <= DRAWS; j++) {
                    var sample = sampler.sample(n, setS);
                    boolean isNull = sample.quantityIndex() < 0;
                    boolean nullMismatch = isNull != (sample.categoryIndex() < 0);
                    if (nullMismatch) inconsistentSamples++;
                    double quantity = isNull ? 0 : quantities[sample.quantityIndex()];
                    double qv = sample.categoryIndex() >= 0 ? quantity * values[sample.categoryIndex()] : 0;
                    double dQ = quantity - meanQ;
                    meanQ += dQ / j;
                    m2Q += dQ * (quantity - meanQ);
                    double dQV = qv - meanQV;
                    meanQV += dQV / j;
                    m2QV += dQV * (qv - meanQV);
                }
                // 5 sigma tolerance isn't foolproof; it should be fine, but outliers can happen.
                double toleranceQ = TOLERANCE_SIGMA * sqrt(m2Q / (DRAWS - 1) / DRAWS) + TOLERANCE_EPSILON * abs(setN);
                double toleranceQV = TOLERANCE_SIGMA * sqrt(m2QV / (DRAWS - 1) / DRAWS) + TOLERANCE_EPSILON * abs(setS);
                double observedQ = meanQ, observedQV = meanQV;
                int inconsistent = inconsistentSamples;
                String cell = String.format("n = %s, s = %.6g", setN, setS);
                checks.add(() -> assertEquals(setN, observedQ, toleranceQ, "E[Q] at " + cell));
                checks.add(() -> assertEquals(setS, observedQV, toleranceQV, "E[Q V] at " + cell));
                checks.add(() -> assertEquals(0, inconsistent,
                        "samples whose quantityIndex and categoryIndex disagree about the null outcome at " + cell));
            }
        }
        assertAll(config.name, checks);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("io.github.aevns.fastmaxent.ValueModelTest#groupCases")
    @DisplayName("Testing that entries sharing a value are drawn in proportion to their category weights")
    void withinGroupWeights(ValueModelTest.GroupCase groupCase) {
        double[] values = groupCase.values(), categoryWeights = groupCase.categoryWeights();
        JointModel model = new JointModel(values, categoryWeights, new double[] {0, 1}, new double[] {1, 1},
                0, new double[] {0.5}, TableParams.DEFAULT);
        JointModel.Sampler sampler = model.newSampler(new SplittableRandom(SEED));
        double s = 0.5 * (model.minTargetS(0) + model.maxTargetS(0));
        long[] hits = new long[values.length];
        for (int i = 0; i < DRAWS; i++) {
            var sample = sampler.sample(0, s);
            if (sample.categoryIndex() >= 0) hits[sample.categoryIndex()]++;
        }
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