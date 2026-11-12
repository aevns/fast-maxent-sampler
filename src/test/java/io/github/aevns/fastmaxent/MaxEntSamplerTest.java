package io.github.aevns.fastmaxent;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import static java.lang.Math.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Consumer;
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
    @DisplayName("Testing that <N> and <NV> match their targets across a range of n and s for each test case")
    void meanTests(TestConfig configuration) {
        SplittableRandom random = new SplittableRandom(SEED);
        Params configParams = new Params();
        configuration.setParams().accept(configParams);
        double[] values = configuration.values();
        MaxEntSampler sampler = new MaxEntSampler(values, configParams);
        List<Executable> checks = new ArrayList<>();
        for (int n = 0; n < MaxEntSampler.SUPPORTED_EXPECTED_COUNTS.length; n++) {
            double setN = MaxEntSampler.SUPPORTED_EXPECTED_COUNTS[n];
            double minS = sampler.minTargetS(n), maxS = sampler.maxTargetS(n);
            for (int step = 0; step < S_STEPS; step++) {
                double setS = minS + (maxS - minS) * step / (S_STEPS - 1);
                // Welford's running mean and variance
                double meanN = 0, m2N = 0, meanNV = 0, m2NV = 0;
                for (int j = 1; j <= DRAWS; j++) {
                    var sample = sampler.sample(n, setS, random);
                    double nv = sample.count > 0 ? sample.count * values[sample.labelIndex] : 0;
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

    @Tag("profile")
    @Test
    @DisplayName("Profiling precomputation and sampling time")
    void timingProfile() throws IOException {
        SplittableRandom random = new SplittableRandom(SEED);
        double[] values = loadValues();

        long start = System.nanoTime();
        MaxEntSampler sampler = new MaxEntSampler(values, new Params());
        long loadTimer = System.nanoTime() - start;

        int[] nSet = new int[DRAWS];
        double[] sSet = new double[DRAWS];

        for (int i = 0; i < DRAWS; i++) {
            nSet[i] = random.nextInt(4);
            sSet[i] = random.nextDouble(sampler.minTargetS(nSet[i]), sampler.maxTargetS(nSet[i]));
        }

        //warmup
        long dummy = 0;
        for (int i = 0; i < DRAWS; i++) {
            MaxEntSampler.Sample sample = sampler.sample(nSet[i], sSet[i], random);
            dummy += sample.count + sample.labelIndex;
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
            dummy += sample.count + sample.labelIndex;
        }

        //warm
        start = System.nanoTime();
        for (int i = 0; i < DRAWS; i++) {
            MaxEntSampler.Sample sample = sampler.sample(nSet[i], sSet[i], random);
            dummy += sample.count + sample.labelIndex;
        }
        long warmTimer = System.nanoTime() - start;

        sink = dummy;
        System.out.print("sampler load time: " + loadTimer / 1_000_000 + " ms\n");
        System.out.print("time per sample (cold): " + (double) coldTimer / COLD_DRAWS + " ns\n");
        System.out.print("time per sample (warm): " + (double) warmTimer / DRAWS + " ns\n");
    }
}