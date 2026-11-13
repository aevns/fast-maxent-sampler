package io.github.aevns.fastmaxent.priors;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DensityPriorTest {
    static final double[] ONE_TO_TEN = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    /** Linear axis, half-width 2.5, similarity 1, continued beyond the ends, no null state. */
    static final DensityPrior LINEAR =
            new DensityPrior(new Axis.Linear(), 2.5, 1, true, false, 1);

    @Test
    void continuationMakesTheTopEndAsCrowdedAsTheInterior() {
        // With half-width 2.5, each interior value has two neighbours on each side within the kernel.
        // If continuation works, weights 3 through 10 should be equal.
        // 1 and 2 should still have higher weights, as continuation doesn't apply there with all positive values.
        double[] weights = LINEAR.weights(ONE_TO_TEN);
        for (int i = 3; i < ONE_TO_TEN.length; i++)
            assertEquals(weights[2], weights[i], 1e-12, "value " + ONE_TO_TEN[i]);
        assertTrue(weights[0] > weights[2]);
    }

    @Test
    void withoutContinuationTheTopEndIsLessCrowded() {
        DensityPrior notContinued = new DensityPrior(new Axis.Linear(), 2.5, 1, false, false, 1);
        double[] weights = notContinued.weights(ONE_TO_TEN);
        assertTrue(weights[9] > weights[5]);
    }

    @Test
    void nullStateGetsItsWeightAndTheRestAverageOne() {
        double[] values = {2, 0, 1, 3};
        double[] weights = new DensityPrior(new Axis.Linear(), 2.5, 1, true, true, 0.5).weights(values);
        assertEquals(0.5, weights[1]);
        assertEquals(3, weights[0] + weights[2] + weights[3], 1e-12);
    }

    @Test
    void rejectsAnAxisThatIsNotIncreasing() {
        assertThrows(IllegalArgumentException.class, 
                () -> new DensityPrior(Axis.of(x -> -x, d -> d), 2.5, 1, true, false, 1).weights(ONE_TO_TEN));
    }
}
