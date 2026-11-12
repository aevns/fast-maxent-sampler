package io.github.aevns.fastmaxent;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import static java.lang.Math.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Consumer;
import java.util.stream.Stream;

import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

public class MaxEntSamplerTest {
    static final long SEED = 0;
    static final String EMC_DATA_PATH = "/mc_pe_emc.txt";
    static final int DRAWS = 500_000;
    static final int S_STEPS = 9;
    static final double TOLERANCE_SIGMA = 5;
    static final double TOLERANCE_EPSILON = 1e-9;

    static double[] loadValues() throws IOException {
        try (var stream = MaxEntSamplerTest.class.getResourceAsStream(MaxEntSamplerTest.EMC_DATA_PATH)) {
            assertNotNull(stream);
            return new BufferedReader(new InputStreamReader(stream))
                    .lines().mapToDouble(Double::parseDouble).toArray();
        }
    }

    record TestConfig(String name, double[] values, Consumer<Params> setParams) {
        @Override public String toString() { return name; }
    }

    static Stream<TestConfig> ConfigStream() throws IOException {
        double[] dataValues = loadValues();
        double[] twoUnique = {1, 1, 1, 2, 2};
        double[] twelveDecades = new double[241];
        for (int i = 0; i < twelveDecades.length; i++) twelveDecades[i] = pow(10, 12.0 * i / (twelveDecades.length - 1));
        return Stream.of(
                new TestConfig("default", dataValues, p -> {}),
                new TestConfig("labelBlockSize = 1", dataValues, p -> p.labelBlockSize = 1),
                new TestConfig("labelBlockSize = 5 (partial last block)", dataValues, p -> p.labelBlockSize = 5),
                new TestConfig("minimal labelTableBudgetBytes", dataValues, p -> p.labelTableBudgetBytes = 1),
                new TestConfig("valueSimilarity = 0", dataValues, p -> p.valueSimilarity = 0),
                new TestConfig("valueSimilarity = 1", dataValues, p -> p.valueSimilarity = 1),
                new TestConfig("nullStatePrior = 0.000_01", dataValues, p -> p.nullStatePrior = 0.000_01),
                new TestConfig("nullStatePrior = 100_000", dataValues, p -> p.nullStatePrior = 100_000),
                new TestConfig("two unique values", twoUnique, p -> {}),
                new TestConfig("values spanning 12 decades", twelveDecades, p -> {}));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("ConfigStream")
    @DisplayName("Testing that <N> and <NV> match their targets across a range of n and s for each test case.")
    void momentTests(TestConfig configuration) {
        SplittableRandom random = new SplittableRandom(SEED);
        Params configParams = new Params();
        configuration.setParams().accept(configParams);
        double[] configValues = configuration.values();
        MaxEntSampler configSampler = new MaxEntSampler(configValues, configParams);
        List<Executable> checks = new ArrayList<>();
        for (int n = 0; n < MaxEntSampler.SUPPORTED_EXPECTED_COUNTS.length; n++) {
            double setN = MaxEntSampler.SUPPORTED_EXPECTED_COUNTS[n];
            double minS = configSampler.minTargetS(n), maxS = configSampler.maxTargetS(n);
            for (int step = 0; step < S_STEPS; step++) {
                double setS = minS + (maxS - minS) * step / (S_STEPS - 1);
                // Welford's running mean and variance
                double meanN = 0, m2N = 0, meanNV = 0, m2NV = 0;
                for (int j = 1; j <= DRAWS; j++) {
                    var sample = configSampler.sample(n, setS, random);
                    double nv = sample.count > 0 ? sample.count * configValues[sample.labelIndex] : 0;
                    double dN = sample.count - meanN;
                    meanN += dN / j;
                    m2N += dN * (sample.count - meanN);
                    double dNV = nv - meanNV;
                    meanNV += dNV / j;
                    m2NV += dNV * (nv - meanNV);
                }
                // 5 sigma tolerance isn't foolproof; it should be fine, but outliers can happen.
                double toleranceN = TOLERANCE_SIGMA * sqrt(m2N / (DRAWS - 1) / DRAWS) + TOLERANCE_EPSILON * setN;
                double toleranceNV = TOLERANCE_SIGMA * sqrt(m2NV / (DRAWS - 1) / DRAWS) + TOLERANCE_EPSILON * abs(setS);
                double observedN = meanN, observedNV = meanNV;
                String cell = String.format("n = %s, s = %.6g", setN, setS);
                checks.add(() -> assertEquals(setN, observedN, toleranceN, "<N> at " + cell));
                checks.add(() -> assertEquals(setS, observedNV, toleranceNV, "<NV> at " + cell));
            }
        }
        assertAll(configuration.name(), checks);
    }
}
