package io.github.aevns.fastmaxent;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;

/** {@link ValueModel} draw latency and build time on the benchmark data set with default table parameters. */
@State(Scope.Thread)
@Fork(3)
public class ValueModelBenchmark {
    /** Targets cycle through a fixed random set, spread over the supported range. */
    private static final int TARGET_MASK = (1 << 16) - 1;

    private BenchmarkData.ValueInputs inputs;
    private ValueModel.Sampler sampler;
    private double[] targets;
    private int next;

    @Setup(Level.Trial)
    public void setUp() {
        inputs = BenchmarkData.ValueInputs.emc();
        ValueModel model = inputs.build();
        SplittableRandom random = new SplittableRandom(0);
        sampler = model.newSampler(random);
        targets = new double[TARGET_MASK + 1];
        for (int i = 0; i < targets.length; i++)
            targets[i] = random.nextDouble(model.minTargetValue(), model.maxTargetValue());
    }

    /** Average time per draw with the tables in cache. */
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    @Warmup(iterations = 5, time = 1)
    @Measurement(iterations = 5, time = 1)
    public int sample() {
        return sampler.sample(targets[next++ & TARGET_MASK]);
    }

    /** Time to build the model's tables from its values and prior weights. */
    @Benchmark
    @BenchmarkMode(Mode.SingleShotTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @Warmup(iterations = 10)
    @Measurement(iterations = 20)
    public ValueModel build() {
        return inputs.build();
    }
}
