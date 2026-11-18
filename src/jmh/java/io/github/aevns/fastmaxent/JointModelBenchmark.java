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

/**
 * {@link JointModel} draw latency and build time on the benchmark data set: 64 quantities, 4 expected-quantity
 * presets, a null outcome, and default table parameters.
 */
@State(Scope.Thread)
@Fork(3)
public class JointModelBenchmark {
    /** Presets and targets cycle through a fixed random set, spread over each preset's supported range. */
    private static final int TARGET_MASK = (1 << 16) - 1;

    private BenchmarkData.JointInputs inputs;
    private JointModel.Sampler sampler;
    private int[] presets;
    private double[] targets;
    private int next;

    @Setup(Level.Trial)
    public void setUp() {
        inputs = BenchmarkData.JointInputs.emc();
        JointModel model = inputs.build();
        SplittableRandom random = new SplittableRandom(0);
        sampler = model.newSampler(random);
        presets = new int[TARGET_MASK + 1];
        targets = new double[TARGET_MASK + 1];
        for (int i = 0; i < targets.length; i++) {
            presets[i] = random.nextInt(BenchmarkData.SUPPORTED_EXPECTED_QUANTITIES.length);
            targets[i] = random.nextDouble(model.minTargetS(presets[i]), model.maxTargetS(presets[i]));
        }
    }

    /** Average time per draw with the tables in cache. */
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    @Warmup(iterations = 5, time = 1)
    @Measurement(iterations = 5, time = 1)
    public JointModel.Sample sample() {
        int i = next++ & TARGET_MASK;
        return sampler.sample(presets[i], targets[i]);
    }

    /** Time to build the model's tables from its values and prior weights. */
    @Benchmark
    @BenchmarkMode(Mode.SingleShotTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @Warmup(iterations = 10)
    @Measurement(iterations = 20)
    public JointModel build() {
        return inputs.build();
    }
}
