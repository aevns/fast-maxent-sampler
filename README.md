
**Disclaimer: The development of this library was assisted by agentic AI under precise direction and manual review.**

# fast-maxent-sampler

A library for high-performance maximum-entropy sampling under certain moment constraints. A primary constraint can be set freely on each draw, and any secondary constraints are selected on each draw from among a set of presets chosen at construction.

The library currently supports two models: a **Value Sampler** over categories and a **Joint Sampler** over categories and quantities.

## Model Overview

A model draws one output index per dimension it samples (such as over a collection of categories and a collection of quantities), under one primary constraint and zero or more secondary constraints.

- The **primary constraint** is a continuous moment constraint on an output, set freely at sample time. It's realized by picking among rows of a precomputed CDF table, spaced by Fisher-Rao arc length. Each row corresponds to a maxent distribution for some constraint value.

- The **secondary constraints** are discrete: they are restricted to presets fixed at construction, and samples draw from an alias table built for that preset. Multiple secondary constraints could be flattened into one such alias-indexed state space, though no models have been implemented yet which take advantage of this.

## Supported Models

### Value Sampler

This model samples from categories with associated values, such as items of set prices or microstates of given energies. Samples are distributed according to

$$p_C(c \mid \mathbb{E}[V]) \propto w_c \cdot e^{-\beta(\mathbb{E}[V]) \cdot v_c}$$

for category $c$ with value $v_c$ and prior $w_c$. Sampling gives a random index $C$ with value $V = v_C$. Category values and weights are provided at construction. The desired mean value $\mathbb{E}[V]$ is the primary constraint, provided at sampling time.

### Joint Sampler

This model samples from categories with associated values and over a set of possible quantities, such as bundles of identical items with total undiscounted prices. Samples are distributed according to

$$p_{I, C}(i, c \mid \mathbb{E}[Q], \mathbb{E}[Q \cdot V]) \propto m_i \cdot w_c \cdot e^{-\lambda(\mathbb{E}[Q]) \cdot q_i -\beta(\mathbb{E}[Q \cdot V]) \cdot q_i \cdot v_c}$$

$$p_{I, C}(\textit{NULL} \mid \mathbb{E}[Q], \mathbb{E}[Q \cdot V]) \propto m_\textit{NULL}$$

for quantity $q_i$ with associated prior $m_i$, and for null state weight $m_\textit{NULL}$. Sampling gives two random indices, $I$ and $C$, with $Q = q_I$ and $V = v_C$. Values and priors for categories and quantities are provided at construction, as well as the optional null state weight. The expected product of quantity and value $\mathbb{E}[Q \cdot V]$ is the primary constraint, and the desired mean quantity $\mathbb{E}[Q]$ is a secondary constraint.

## Installation

Requires Java 21+.

The project is available on Maven Central with coordinates `io.github.aevns:fast-maxent-sampler:0.1.0`. It can be included as a dependency with:

```kotlin
// Gradle (Kotlin DSL)
dependencies {
    implementation("io.github.aevns:fast-maxent-sampler:0.1.0")
}
```

```groovy
// Gradle (Groovy DSL)
dependencies {
    implementation 'io.github.aevns:fast-maxent-sampler:0.1.0'
}
```

```xml
<!-- Maven -->
<dependency>
    <groupId>io.github.aevns</groupId>
    <artifactId>fast-maxent-sampler</artifactId>
    <version>0.1.0</version>
</dependency>
```

## Usage

### ValueSampler

`ValueSampler(values, priorWeights, params)`

- `values` — the value of each category; finite, sorted ascending, at least two distinct values present.
- `priorWeights` — prior weight of each category; strictly positive.
- `params` — a `TableParams` (see below); `TableParams.DEFAULT` is a reasonable starting point.

`sample(target, rng)` draws and returns a category index given a target `E[V]` (clamped to `[minTargetValue(), maxTargetValue()]`).

```java
double[] values = {1.0, 2.5, 5.0, 10.0};
double[] priorWeights = {1, 1, 1, 1};
ValueSampler sampler = new ValueSampler(values, priorWeights, TableParams.DEFAULT);

int index = sampler.sample(4.0, new SplittableRandom());
double drawnValue = values[index];
```

### JointSampler

`JointSampler(categoryValues, categoryPriorWeights, quantities, quantityPriorWeights, nullStateWeight, supportedExpectedQuantities, params)`

- `categoryValues` — the value of each category; finite, sorted ascending, at least two distinct values present.
- `categoryPriorWeights` — prior weight of each category; strictly positive.
- `quantities` — the allowed quantities (if 0 is included, it is an ordinary quantity, not the null outcome); finite.
- `quantityPriorWeights` — prior weight of each quantity; strictly positive.
- `nullStateWeight` — prior weight of the null outcome (no quantity, no category), on the same scale as `quantityPriorWeights`; 0 to disallow it.
- `supportedExpectedQuantities` — the supported secondary-constraint presets for expected quantity; each strictly between the smallest and largest allowed quantity (treating the null outcome, if allowed, as a quantity of 0).
- `params` — a `TableParams` (see below); `TableParams.DEFAULT` is a reasonable starting point.

`sample(expectedQuantityIndex, s, rng)` obtains the secondary-constraint preset by index into `supportedExpectedQuantities`, then draws at the given primary-constraint target `s = E[Q · V]` (clamped to `[minTargetS(expectedQuantityIndex), maxTargetS(expectedQuantityIndex)]`). It draws and returns a `JointSampler.Sample` with a `categoryIndex` and a `quantityIndex`. Both indices are -1 for a null state outcome.

```java
double[] categoryValues = {1.0, 2.5, 5.0, 10.0};
double[] categoryPriorWeights = {1, 1, 1, 1};
double[] quantities = {0, 1, 2, 3, 4};
double[] quantityPriorWeights = {1, 1, 1, 1, 1};
double[] supportedExpectedQuantities = {1, 2};
JointSampler sampler = new JointSampler(categoryValues, categoryPriorWeights, quantities,
        quantityPriorWeights, 0.0, supportedExpectedQuantities, TableParams.DEFAULT);

JointSampler.Sample sample = sampler.sample(0, 2.5, new SplittableRandom());
boolean isNull = sample.quantityIndex < 0;
if (!isNull) {
    double drawnValue = categoryValues[sample.categoryIndex];
    double drawnQuantity = quantities[sample.quantityIndex];
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

`approximateMemoryBytes()`, on all samplers, reports the approximate size of its precomputed tables plus the grouped category/quantity data.

Rough results tested on an Intel i7-12700K with 4800 MT/s DDR5 memory, measured against a set of 1470 category values and `TableParams.DEFAULT`:

### Value Sampler
- ~106 ns per `sample()` call
- ~1.6 µs per `sample()` call with a flushed cache
- ~8 ms for table construction
- ~190 KB table size

### Joint Sampler
Tested with 64 quantities and 4 `supportedExpectedQuantities` presets
- ~99 ns per `sample()` call
- ~1.7 µs per `sample()` call with a flushed cache
- ~45 ms for table construction
- ~1.2 MB table size

Profiling tests are included in test classes under the `profile` tag. They're included in the default `test` task (`./gradlew test`), and can be run individually via

```bash
./gradlew test --tests "io.github.aevns.fastmaxent.ValueSamplerTest.timingProfile" --tests "io.github.aevns.fastmaxent.JointSamplerTest.timingProfile"
```

## Building and testing

```bash
./gradlew build   # compile and run all tests
./gradlew test    # run all tests
```

The default `test` task includes profiling tests (tagged `profile`).

## License

[MIT](LICENSE)
