
[![Maven Central](https://shields.io/maven-central/v/io.github.aevns/fast-maxent-sampler)](https://central.sonatype.com/artifact/io.github.aevns/fast-maxent-sampler)

*The development of this library was assisted by agentic AI under precise direction and manual review.*

# fast-maxent-sampler

A library for high-performance maximum-entropy sampling under certain moment constraints. A primary constraint can be set freely on each draw, and any secondary constraints are selected on each draw from among a set of presets chosen at construction. Constraints are met exactly (up to 32-bit fixed point precision), and the maximum entropy condition is approximated with fisher-spaced tables.

The library currently supports two models: a **Value Model** over categories and a **Joint Model** over categories and quantities.

## Model Overview

A model draws one output index per dimension it samples (such as over a collection of categories and a collection of quantities), under one primary constraint and zero or more secondary constraints.

- The **primary constraint** is a continuous moment constraint on an output, set freely at sample time. It's realized by picking among rows of a precomputed CDF table, spaced by Fisher-Rao arc length. Each row corresponds to a maxent distribution for some constraint value.

- The **secondary constraints** are discrete: they are restricted to presets fixed at construction, and samples draw from an alias table built for that preset. Multiple secondary constraints could be flattened into one such alias-indexed state space, though no models have been implemented yet which take advantage of this.

## Supported Models

### Value Model

This model samples from categories with associated values, such as items of set prices or microstates of given energies. Samples are distributed according to

$$p_C(c \mid \mathbb{E}[V]) \propto w_c \cdot e^{-\beta(\mathbb{E}[V]) \cdot v_c}$$

for category $c$ with value $v_c$ and prior $w_c$. Sampling gives a random index $C$ with value $V = v_C$. Category values and weights are provided at construction. The desired mean value $\mathbb{E}[V]$ is the primary constraint, provided at sampling time.

### Joint Model

This model samples from categories with associated values and over a set of possible quantities, such as bundles of identical items with total undiscounted prices. Samples are distributed according to

$$p_{I, C}(i, c \mid \mathbb{E}[Q], \mathbb{E}[Q \cdot V]) \propto m_i \cdot w_c \cdot e^{-\lambda(\mathbb{E}[Q]) \cdot q_i -\beta(\mathbb{E}[Q \cdot V]) \cdot q_i \cdot v_c}$$

$$p_{I, C}(\textit{NULL} \mid \mathbb{E}[Q], \mathbb{E}[Q \cdot V]) \propto m_\textit{NULL}$$

for quantity $q_i$ with associated prior $m_i$, and for null state weight $m_\textit{NULL}$. Sampling gives two random indices, $I$ and $C$, with $Q = q_I$ and $V = v_C$. Values and priors for categories and quantities are provided at construction, as well as the optional null state weight. The expected product of quantity and value $\mathbb{E}[Q \cdot V]$ is the primary constraint, and the desired mean quantity $\mathbb{E}[Q]$ is a secondary constraint.

## Installation

Requires Java 21+.

The project is available on Maven Central with coordinates `io.github.aevns:fast-maxent-sampler:0.2.0`. It can be included as a dependency with:

```kotlin
// Gradle (Kotlin DSL)
dependencies {
    implementation("io.github.aevns:fast-maxent-sampler:0.2.0")
}
```

```groovy
// Gradle (Groovy DSL)
dependencies {
    implementation 'io.github.aevns:fast-maxent-sampler:0.2.0'
}
```

```xml
<!-- Maven -->
<dependency>
    <groupId>io.github.aevns</groupId>
    <artifactId>fast-maxent-sampler</artifactId>
    <version>0.2.0</version>
</dependency>
```

## Usage

Each model contains its precomputed tables, is immutable, and is thread safe. Sampling is done through a `Sampler` created with `model.newSampler(rng)`, which holds a random number generator implementing `nextLong()` and a small scratch buffer. Samplers are not thread-safe: one should be created per thread, just like the generators they use.

### ValueModel

`ValueModel(values, priorWeights, params)`

- `values` — the value of each category; finite, in any order (already-sorted input skips the internal sort), at least two distinct values present.
- `priorWeights` — prior weight of each category; strictly positive.
- `params` — a `TableParams` (see below); `TableParams.DEFAULT` is a reasonable starting point.

`sampler.sample(target)` draws and returns a category index given a target `E[V]` (clamped to `[model.minTargetValue(), model.maxTargetValue()]`).

```java
double[] values = {1.0, 2.5, 5.0, 10.0};
double[] priorWeights = {1, 1, 1, 1};
ValueModel model = new ValueModel(values, priorWeights, TableParams.DEFAULT);
ValueModel.Sampler sampler = model.newSampler(new SplittableRandom());

int index = sampler.sample(4.0);
double drawnValue = values[index];
```

### JointModel

`JointModel(categoryValues, categoryPriorWeights, quantities, quantityPriorWeights, nullStateWeight, supportedExpectedQuantities, params)`

- `categoryValues` — the value of each category; finite, in any order (already-sorted input skips the internal sort), at least two distinct values present.
- `categoryPriorWeights` — prior weight of each category; strictly positive.
- `quantities` — the allowed quantities (if 0 is included, it is an ordinary quantity, not the null outcome); finite.
- `quantityPriorWeights` — prior weight of each quantity; strictly positive.
- `nullStateWeight` — prior weight of the null outcome (no quantity, no category), on the same scale as `quantityPriorWeights`; 0 to disallow it.
- `supportedExpectedQuantities` — the supported secondary-constraint presets for expected quantity; each strictly between the smallest and largest allowed quantity (treating the null outcome, if allowed, as a quantity of 0).
- `params` — a `TableParams` (see below); `TableParams.DEFAULT` is a reasonable starting point.

`sampler.sample(expectedQuantityIndex, s)` obtains the secondary-constraint preset by index into `supportedExpectedQuantities`, drawing with it and the given primary-constraint target `s = E[Q · V]` (clamped to `[model.minTargetS(expectedQuantityIndex), model.maxTargetS(expectedQuantityIndex)]`). It draws and returns a `JointModel.Sample` with a `categoryIndex` and a `quantityIndex`. Both indices are -1 for a null state outcome.

```java
double[] categoryValues = {1.0, 2.5, 5.0, 10.0};
double[] categoryPriorWeights = {1, 1, 1, 1};
double[] quantities = {0, 1, 2, 3, 4};
double[] quantityPriorWeights = {1, 1, 1, 1, 1};
double[] supportedExpectedQuantities = {1, 2};
JointModel model = new JointModel(categoryValues, categoryPriorWeights, quantities,
        quantityPriorWeights, 0.0, supportedExpectedQuantities, TableParams.DEFAULT);
JointModel.Sampler sampler = model.newSampler(new SplittableRandom());

JointModel.Sample sample = sampler.sample(0, 2.5);
boolean isNull = sample.quantityIndex() < 0;
if (!isNull) {
    double drawnValue = categoryValues[sample.categoryIndex()];
    double drawnQuantity = quantities[sample.quantityIndex()];
}
```

### TableParams

`TableParams.DEFAULT` is `(1e-5, 256, 8, 20_000_000L, 500)`. Fields, in order:

- `saturationTolerance` — how close the supported target range comes, at each end, to a single category holding all the probability mass (smaller values push the end closer, at the cost of a finer row grid near it); must be in the range (0, 1).
- `targetRowsPerConstraintPreset` — number of precomputed rows placed along the primary constraint, for each secondary constraint preset (more rows trade memory and construction time for resolution); minimum 2.
- `categoryBlockSize` — number of categories grouped per block in the shared precomputed category table (larger blocks trade more per-sample work for less memory, roughly 1 / categoryBlockSize).
- `categoryTableBudgetBytes` — memory budget for the shared precomputed category table (acts as an indirect cap on row count).
- `betaCandidatesPerSign` — number of candidate values per sign considered when placing rows by Fisher-Rao arc length; more candidates trade construction time for finer row placement, not final sampling accuracy; minimum 2.

### Priors

Utilities for generating prior weights under different assumptions.

`DensityPrior` produces prior weights that correct for crowding along an axis, after Leinster and Cobbold's similarity-sensitive diversity. Values close together share weight rather than each counting as fully independent. Ideal for priors in which similarly valued categories or quantities are qualitatively similar in the domain under consideration.

- `DensityPrior(axis, halfWidth, similarity, continueLow, continueHigh).weights(values)` returns one weight per value, summing to 1.
- `axis` is an `Axis` — `Axis.Linear` (absolute difference), `Axis.Symlog(threshold)` (roughly linear difference below `threshold`, smoothly transitioning to relative difference above it), or a custom `Axis`.
- `halfWidth` — sets how close two values must be (in axis-scaled distance) to compete for weight
- `similarity` —  `sigma`, in `[0, 1]`, trades weighting every value equally (0) against weighting by how much of the axis a region covers (1).
- `continueLow`/`continueHigh` treat a truncated set's low/high end as if it continued at its outermost spacing, so the end isn't treated as artificially uncrowded.

## Performance

`approximateMemoryBytes()`, on both models, reports the approximate size of its precomputed tables plus the grouped category/quantity data.

Rough results tested on an Intel i7-12700K with 4800 MT/s DDR5 memory, measured against a set of 1470 category values and `TableParams.DEFAULT`:

### Value Model
- ~101 ns per `sample()` call
- no allocation per `sample()` call
- ~7.3 ms for table construction
- ~190 KB table size

### Joint Model
Tested with 64 quantities and 4 `supportedExpectedQuantities` presets
- ~81 ns per `sample()` call
- 24 B allocated per `sample()` call for the returned `Sample` record
- ~30 ms for table construction
- ~1.2 MB table size

These figures represent performance over repeated samples, and so table data will remain in cache between samples. Sampling with tables flushed from the cache may take on the order of a microsecond on typical desktop hardware.

Benchmarks use [JMH](https://github.com/openjdk/jmh) and can be found in `src/jmh`. Table build time and sample time are testable for each model. [JMH options](https://github.com/openjdk/jmh) can be passed with `--args`:

```bash
./gradlew jmh                                    # run all benchmarks
./gradlew jmh --args="ValueModelBenchmark"       # run one class (regex match on benchmark names)
./gradlew jmh --args="sample -prof gc"           # include bytes allocated per sample
```

## Building and testing

```bash
./gradlew build   # compile and run all tests
./gradlew test    # run all tests
./gradlew jmh     # run benchmarks (not part of build or test)
```

## License

[MIT](LICENSE)
