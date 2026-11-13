package io.github.aevns.fastmaxent.priors;

import java.util.function.DoubleUnaryOperator;

/** A transformation of values onto an axis: {@link Linear}, {@link Symlog}, or a custom one via {@link #of}. */
public interface Axis {
    /** Position of the value along the axis; strictly increasing. */
    double position(double value);

    /** Transforms a difference between values into a difference between positions. */
    double scale(double difference);

    /** An axis from its position and scale functions. */
    static Axis of(DoubleUnaryOperator position, DoubleUnaryOperator scale) {
        return new Axis() {
            @Override public double position(double value) { return position.applyAsDouble(value); }
            @Override public double scale(double difference) { return scale.applyAsDouble(difference); }
        };
    }

    /** Positions are the values themselves; differences are absolute and unchanged. */
    record Linear() implements Axis {
        @Override public double position(double value) { return value; }
        @Override public double scale(double difference) { return difference; }
    }

    /**
     * sign(x) ln(1 + |x| / threshold): linear below the threshold, logarithmic above it (the log-modulus transform of
     * x / threshold, also called the bi-symmetric log). Differences are relative: a relative difference f becomes
     * ln(1 + f), exact well above the threshold.
     *
     * @param threshold positive and finite
     */
    record Symlog(double threshold) implements Axis {
        public Symlog {
            if (!(threshold > 0) || Double.isInfinite(threshold))
                throw new IllegalArgumentException("threshold must be positive and finite");
        }

        @Override public double position(double value) {
            return Math.copySign(Math.log1p(Math.abs(value) / threshold), value);
        }
        @Override public double scale(double difference) { return Math.log1p(difference); }
    }
}
