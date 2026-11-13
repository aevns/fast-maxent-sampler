package io.github.aevns.fastmaxent;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks that the tables realize the max-entropy form p(q, k) ~ m(q) W_k exp(-lambda q - beta q x_k) at a given beta:
 * the quantity distribution has the requested mean and the right shape, and each category distribution matches the
 * exact one.
 */
class MaxEntFormTest {
    /** 300 scaled values over four decades, with uneven prior weights summing to 1. */
    static final double[] VALUES = IntStream.range(0, 300).mapToDouble(i -> Math.pow(10, -2 + 4.0 * i / 299)).toArray();
    static final double[] WEIGHTS = normalized(IntStream.range(0, 300).mapToDouble(i -> 1 + i % 7).toArray());
    static final CategoryFamily FAMILY = new CategoryFamily(VALUES, WEIGHTS, TableParams.DEFAULT);

    record Case(String name, double[] quantities, double expectedQuantity, double beta) {
        @Override public String toString() { return name; }
    }

    /** Betas keep |beta q| below the saturation point, past which the category family is clamped by design. */
    static Stream<Case> cases() {
        double[] counts = {0, 1, 2, 3, 4, 5, 6, 7, 8}, mixedSigns = {-4, -2, -1, 0, 1, 2, 4};
        return Stream.of(
                new Case("counts, beta > 0", counts, 2, 0.3),
                new Case("counts, beta < 0", counts, 2, -0.3),
                new Case("mixed signs, beta > 0", mixedSigns, 0.5, 0.4),
                new Case("mixed signs, beta < 0", mixedSigns, 0.5, -0.4),
                new Case("non-integer, no null state", new double[] {0.5, 1, 1.5, 2.5}, 1.2, 0.2));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void quantityAndCategoryDistributionsHaveMaxEntForm(Case test) {
        double[] logPrior = Arrays.stream(test.quantities).map(q -> Math.log(1 + Math.abs(q))).toArray();
        QuantityDistributionSolver solver =
                new QuantityDistributionSolver(FAMILY, new QuantitySet(test.quantities), logPrior);
        solver.configureForBeta(test.beta);
        solver.solveLambdaForExpectedQuantity(test.expectedQuantity);
        double[] p = solver.stateProbabilities;

        // Quantities: the requested mean, and ln p(q) - ln m(q) - ln Z(beta q) affine in q (the -lambda q term).
        double mean = 0;
        double[] residual = new double[p.length];
        for (int state = 0; state < p.length; state++) {
            double q = test.quantities[state];
            mean += p[state] * q;
            residual[state] = Math.log(p[state]) - logPrior[state] - Math.log(partition(test.beta * q));
        }
        assertEquals(test.expectedQuantity, mean, 1e-9, "mean quantity");
        double slope = (residual[1] - residual[0]) / (test.quantities[1] - test.quantities[0]);
        for (int state = 0; state < p.length; state++) {
            double affine = residual[0] + slope * (test.quantities[state] - test.quantities[0]);
            assertEquals(affine, residual[state], 1e-6, "quantity " + test.quantities[state]);
        }

        // Categories: for each nonzero quantity, the table's distribution at t = beta q matches the exact one, up to
        // mixing between stored rows and 32-bit rounding. Total variation, since rounding can leave a negligible
        // block with no mass at all.
        CategoryFamily.Point point = new CategoryFamily.Point();
        for (double q : test.quantities) {
            if (q == 0) continue;
            double t = test.beta * q;
            FAMILY.evaluateAtT(t, point);
            double mix = FixedPoint.toProbability(point.mixThreshold);
            double[] lower = FAMILY.table().rowDistribution(point.lowerCategoryRow);
            double[] upper = mix > 0 ? FAMILY.table().rowDistribution(point.lowerCategoryRow + 1) : lower;
            double totalVariation = 0;
            for (int k = 0; k < VALUES.length; k++) {
                double exact = WEIGHTS[k] * Math.exp(-t * VALUES[k]) / partition(t);
                totalVariation += 0.5 * Math.abs(exact - ((1 - mix) * lower[k] + mix * upper[k]));
            }
            assertTrue(totalVariation < 1e-4, "total variation " + totalVariation + " at quantity " + q);
        }
    }

    /** Z(t) = sum_k W_k exp(-t x_k); Z(0) = 1. */
    static double partition(double t) {
        double sum = 0;
        for (int k = 0; k < VALUES.length; k++) sum += WEIGHTS[k] * Math.exp(-t * VALUES[k]);
        return sum;
    }

    static double[] normalized(double[] weights) {
        double sum = Arrays.stream(weights).sum();
        return Arrays.stream(weights).map(w -> w / sum).toArray();
    }
}
