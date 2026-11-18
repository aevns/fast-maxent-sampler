package io.github.aevns.fastmaxent;

import io.github.aevns.fastmaxent.priors.Axis;
import io.github.aevns.fastmaxent.priors.DensityPrior;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.stream.IntStream;

/** The data set (1470 category values) and priors behind the README's performance figures, as in the unit tests. */
final class BenchmarkData {
    private BenchmarkData() {}

    static final DensityPrior CATEGORY_PRIOR = new DensityPrior(new Axis.Symlog(10), 0.15, 0.1, false, false);
    static final DensityPrior QUANTITY_PRIOR = new DensityPrior(new Axis.Symlog(1), 0.15, 1, false, true);
    static final double[] QUANTITIES = IntStream.rangeClosed(1, 64).asDoubleStream().toArray();
    static final double[] SUPPORTED_EXPECTED_QUANTITIES = {0.25, 1, 4, 16};

    static double[] emcValues() {
        try (InputStream stream = BenchmarkData.class.getResourceAsStream("/mc_pe_emc.txt")) {
            if (stream == null) throw new IllegalStateException("mc_pe_emc.txt not on the classpath");
            return new BufferedReader(new InputStreamReader(stream)).lines().mapToDouble(Double::parseDouble).toArray();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    record ValueInputs(double[] values, double[] priorWeights) {
        static ValueInputs emc() {
            double[] values = emcValues();
            return new ValueInputs(values, CATEGORY_PRIOR.weights(values));
        }

        ValueModel build() {
            return new ValueModel(values, priorWeights, TableParams.DEFAULT);
        }
    }

    record JointInputs(double[] categoryValues, double[] categoryPriorWeights, double[] quantityPriorWeights,
                       double nullStateWeight) {
        /** The null outcome's weight is the quantity prior's weight for an implied quantity of 0. */
        static JointInputs emc() {
            double[] values = emcValues();
            double[] quantitiesWithImpliedZero = Arrays.copyOf(QUANTITIES, QUANTITIES.length + 1);
            double[] weightsWithImpliedZero = QUANTITY_PRIOR.weights(quantitiesWithImpliedZero);
            return new JointInputs(values, CATEGORY_PRIOR.weights(values),
                    Arrays.copyOf(weightsWithImpliedZero, QUANTITIES.length), weightsWithImpliedZero[QUANTITIES.length]);
        }

        JointModel build() {
            return new JointModel(categoryValues, categoryPriorWeights, QUANTITIES, quantityPriorWeights,
                    nullStateWeight, SUPPORTED_EXPECTED_QUANTITIES, TableParams.DEFAULT);
        }
    }
}
