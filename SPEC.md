# mobius_processing

A bridge between [Mobius](https://github.com/mobius-home/mobius) metric storage
and the Nx ecosystem (Nx, Scholar, NxSignal, ...). Pulls Mobius data out as
Arrow columns, hands it to Nx as tensors, and **documents** what processing the
Nx ecosystem can do on the typical shapes of data Nerves devices collect.

This library does not pick a backend. Bring your own — `Nx.BinaryBackend`,
`EXLA`, `Torchx`, `nx_eigen`. The math doesn't care. **All examples,
benchmarks, and the LiveBook companion use `nx_eigen` as the demonstration
backend** because it's the right pick for the embedded Nerves context this
library is aimed at: CPU SIMD acceleration via Eigen, small footprint, no
XLA / LLVM toolchain required.

This library does not decide when computation happens. On-demand, periodic,
streaming — that's the caller's call. We expose pure functions over tensors.

## Principles

- **Documentation first.** Most of the value here is showing what's possible
  with Nx on Mobius-shaped data. Working examples beat clever abstractions.
- **Pure functions.** No GenServers, no schedulers. The caller decides cadence.
- **BYO backend.** `nx` is a hard dep. `scholar` and `nx_signal` are optional
  (loaded only when the corresponding functions are called). `nx_eigen` is
  the demonstration backend used throughout the docs and LiveBook companion,
  but is not pulled in as a dep — users opt in.
- **No opinions on storage.** Read from Mobius, return a result. Whether you
  cache, stream, or recompute is the caller's problem.
- **Default calculations are starting points, not prescriptions.** They cover
  the obvious "what does this device look like" view; the interesting math
  lives in the docs as recipes.

## Scope

### In scope

- Reading Mobius metric history and emitting it as Arrow record batches
  (depends on the `arrow` library at `~/sprawl/arrow`).
- Converting Arrow primitive columns to Nx tensors (one memcpy via
  `Nx.from_binary/2`).
- Convenience helpers for the recurring shapes: per-metric tensor, time index
  tensor, validity mask if any.
- A `MobiusProcessing.Default` module with a small set of sensible default
  computations for system health metrics (described below).
- A `MobiusProcessing.Recipes` module — really a docs hub — that walks through
  what Nx core, Scholar, and NxSignal can each do for the data Nerves devices
  produce.

### Out of scope

- Visualization. Tucan / VegaLite / Kino are the right tools downstream.
- Persistence of computed results. If you want to cache, you cache.
- Push/streaming infrastructure. Telemetry, broadcasts, GenStage — caller's
  problem.
- Cross-device aggregation. This is per-device; fleet-wide rollup belongs in
  NervesHub or wherever your fleet data lands.
- Opinions about windowing. We expose primitives; you decide whether your
  window is the last hour, the last 1000 samples, or all of history.

## Data shapes we expect to see

These are the typical sources of numeric data on a Nerves device. The library's
job is to make all of them addressable as Nx tensors with the same shape:
`{n_samples}` for a single metric, or `{n_samples, n_metrics}` for a batch.

### Nerves system telemetry (`nerves_motd`)

`NervesMOTD.Runtime` exposes:

| Metric                | Type     | Notes                                  |
|-----------------------|----------|----------------------------------------|
| `cpu_temperature`     | f64      | Celsius, smooth gauge                  |
| `load_average` (1/5/15min) | f64 × 3 | Smoothed by source — derivatives less interesting |
| `memory_stats.size_mb` / `used_mb` / `used_percent` | u32/u32/u8 | size_mb is ~constant; the others move |
| `filesystem_stats.*`  | same    | per mount point                        |
| `time_synchronized?`  | bool     | binary signal                         |

### NervesHub health extension (`nerves_hub_link.extensions.health.metric_set.*`)

`NervesHubLink.Extensions.Health.MetricSet.*` exposes:

| Source         | Keys                                                              |
|----------------|-------------------------------------------------------------------|
| `CPU`          | `cpu_temp`, `cpu_usage_percent`, `load_1min`, `load_5min`, `load_15min` |
| `Memory`       | `mem_size_mb`, `mem_used_mb`, `mem_used_percent`                  |
| `Disk`         | `disk_total_kb`, `disk_available_kb`, `disk_used_percentage`      |
| `NetworkTraffic` | `<iface>_bytes_received_total`, `<iface>_bytes_sent_total` (monotonic counters) |

### BEAM telemetry (no extra deps)

Available via `:erlang.memory/0`, `:erlang.statistics/1`, `:scheduler.utilization/1`:

| Metric              | Source                                      | Shape          |
|---------------------|---------------------------------------------|----------------|
| `vm.memory.total`   | `:erlang.memory(:total)`                    | gauge (bytes)  |
| `vm.memory.processes` / `:binary` / `:atom` / `:code` / `:ets` | `:erlang.memory/1` | gauges |
| `vm.process_count`  | `length(Process.list())`                    | gauge          |
| `vm.port_count`     | `length(Port.list())`                       | gauge          |
| `vm.run_queue`      | `:erlang.statistics(:total_run_queue_lengths_all)` | gauge   |
| `vm.reductions`     | `:erlang.statistics(:reductions)`           | monotonic counter (resets on wrap) |
| `vm.io.bytes_in` / `bytes_out` | `:erlang.statistics(:io)`        | monotonic counter |
| `vm.gc_count` / `gc_words_reclaimed` | `:erlang.statistics(:garbage_collection)` | monotonic counter |
| `vm.scheduler_utilization` | `:scheduler.utilization(1)` mean | gauge ratio |

### Domain sensor data

Whatever the device measures. Common shapes:

| Shape         | Examples                                | Suitable processing                  |
|---------------|------------------------------------------|--------------------------------------|
| Smooth gauge  | temperature, humidity, pressure, voltage | rolling stats, FFT, smoothing, anomaly detection |
| Noisy gauge   | accelerometer, microphone level          | filtering, FFT, RMS, peak detection  |
| Binary state  | door open, relay on/off, presence        | duty cycle, transition counts, time-in-state |
| Event count   | button presses, alarms, packet errors    | histogram by time, rate, burst detection |
| Monotonic counter | energy use (kWh), packet count       | first difference → rate, then gauge-style |

These are taxonomies, not exhaustive lists. Devices in the wild emit any
combination of these.

## API surface

Roughly three layers. Each is a thin module of pure functions.

### `MobiusProcessing.Source` — Mobius → Arrow

Pulls data out of Mobius and into Arrow record batches. The interesting
question is *which* shape: long format (one row per sample, one column
identifies the metric) or wide format (one column per metric).

```elixir
@spec long(opts) :: Arrow.RecordBatch.t()
@spec wide(opts) :: Arrow.RecordBatch.t()
```

`opts` includes `:from`, `:to`, `:metrics` (list of names), `:tags` filter,
`:resolution` (which RRD bucket to pull from), `:instance`.

Long format is the natural Mobius shape: `timestamp :: ts64, name :: utf8,
type :: utf8, value :: f64, tags :: struct`. Easy to filter, slow to math
over directly.

Wide format is the natural Nx shape: `timestamp :: ts64, metric_a :: f64,
metric_b :: f64, ...`. Each metric becomes its own column, aligned on
timestamp. Requires resampling/alignment if metrics weren't sampled
synchronously — see `MobiusProcessing.Align`.

### `MobiusProcessing.Align` — alignment & resampling

Real device metrics aren't sampled at exactly the same instants. To put them
into a `{n_samples, n_metrics}` tensor you need a common time axis.

```elixir
@spec resample(Arrow.RecordBatch.t(), opts) :: Arrow.RecordBatch.t()
```

Methods:

- `:nearest` — pick the nearest sample to each grid point. Fast, lossy.
- `:previous` — last-observation-carried-forward. Right for gauges that hold
  state between samples.
- `:linear` — linear interpolation. Right for smooth signals between samples.
- `:bucket_mean` / `:bucket_max` / `:bucket_min` — bucket aggregation. Right
  when the source rate is higher than the analysis rate.

Resampling is the price of admission for any cross-metric analysis. It's a
deliberate, explicit step here so the lossy choice is visible.

### `MobiusProcessing.Tensor` — Arrow → Nx

```elixir
@spec from_column(Arrow.Array.t()) :: {Nx.Tensor.t(), validity :: Nx.Tensor.t() | nil}
@spec to_column(Nx.Tensor.t(), validity :: Nx.Tensor.t() | nil, type :: atom()) :: Arrow.Array.t()
@spec batch_to_tensor(Arrow.RecordBatch.t(), keep: [name]) :: Nx.Tensor.t()
```

Dtype mapping is direct: Arrow Int8/16/32/64, UInt8/16/32/64, Float32/64 map
to Nx s8/.../u8/.../f32/f64 with the same bytes. Single memcpy via
`Nx.from_binary/2`. Strings, lists, structs raise — those don't ride the Nx
path.

Validity bitmaps come back as a separate unpacked u8 tensor (1 = valid).
Callers using mask-aware ops thread the mask through; callers who don't care
discard it. Most embedded telemetry has `null_count == 0`, so the mask is
often `nil`.

### `MobiusProcessing.Default` — opinionated starting points

A small set of named computations that answer "what does this device look
like right now?" Useful as both reference and runnable code.

For every numeric gauge:

- **Descriptive stats** over a window: `min`, `max`, `mean`, `stddev`,
  `p50`, `p95`, `p99` → `Nx.{reduce_min, reduce_max, mean,
  standard_deviation, quantile}`
- **Rolling 1-minute / 5-minute / 15-minute mean** → `Nx.window_mean/2`
- **Z-score of the latest sample** against window stats — simple anomaly
  signal
- **Trend slope** via least squares against the time axis → `Scholar.Linear.LinearRegression`
  (or hand-rolled `Nx.LinAlg.lstsq` for no extra dep)

For monotonic counters (bytes, reductions, GC counts):

- **First difference → rate** → `Nx.diff`
- **Rate over windows** → rate, then `Nx.window_mean`
- **Reset detection** → indices where `Nx.diff < 0`

For binary state (booleans, on/off):

- **Duty cycle** over a window → `Nx.mean` of the u8 tensor
- **Transition count** → `Nx.diff |> Nx.abs |> Nx.sum`
- **Run lengths** → cumulative-segment trick: `Nx.cumulative_sum` on the
  transition indicator, then bucket

For event counts (counter-of-events, not monotonic byte counters):

- **Histogram over time** → bucket index by `Nx.quotient(timestamp - t0, bucket)`
  then `Nx.bincount`
- **Burst detection** — flag windows where rate exceeds mean + k·stddev

### `MobiusProcessing.Recipes` — the documentation hub

This is where most of the writing goes. Each recipe is a moduledoc-driven
walkthrough with runnable code. **Every code example assumes `nx_eigen` is
configured as the default Nx backend** — that's the embedded-CPU story this
library is positioned around, and the examples carry timing notes
("~0.2ms on a Raspberry Pi 4 with nx_eigen") so readers can see what's
realistic. Examples are copy-pasteable into IEx or LiveBook against the
sample dataset shipped in `priv/`.

#### From core Nx (no extra deps)

- **Rolling statistics**: `Nx.window_*` for mean/min/max/sum/product over a
  fixed window
- **Cumulative statistics**: `Nx.cumulative_sum/cumulative_min/...` for
  running aggregates
- **Differencing**: `Nx.diff` for rate-of-change on counters
- **Bucketing**: `Nx.quotient` + `Nx.bincount` for time-based histograms
- **Sorting and quantiles**: `Nx.sort`, `Nx.argsort`, `Nx.quantile`
- **Gather/scatter**: `Nx.take`, `Nx.gather`, `Nx.indexed_add` for sparse
  aggregations (group-by-key style)
- **Masked reductions**: `Nx.select(mask, x, 0) |> Nx.sum` for null-aware
  aggregation
- **Z-scoring**: `(x - mean) / stddev` for normalization and anomaly scoring
- **Linear regression manually**: `Nx.LinAlg.lstsq` for trend estimation
  without a Scholar dep

#### From Scholar (optional dep)

- **`Scholar.Stats`** — histogram, correlation, covariance, moments. The
  obvious tools for "how are these metrics related?"
- **`Scholar.Preprocessing`** — StandardScaler, MinMaxScaler. Useful when
  comparing metrics on different scales (CPU% vs. memory MB vs. temperature)
- **`Scholar.Linear.LinearRegression`** — trend detection, simple forecasting
- **`Scholar.Cluster.KMeans` / `Scholar.Cluster.DBSCAN`** — clustering of
  multi-metric vectors, e.g. "what does normal device behavior look like
  vs. when it's misbehaving"
- **`Scholar.Decomposition.PCA`** — dimensionality reduction when you have
  many correlated metrics
- **`Scholar.Neighbors.KNN`** — anomaly detection by distance to k-nearest
  historical states
- **`Scholar.Metrics`** — when you build a classifier or threshold rule,
  evaluate it properly

#### From NxSignal (optional dep)

NxSignal is where time-series and DSP live. Suggest where applicable:

- **`NxSignal.detrend`** — remove linear/constant trend before spectral
  analysis. Important before FFT.
- **`NxSignal.fft / stft`** — periodogram of a metric, find dominant
  frequencies. Useful for sensor data with cyclic behavior (a compressor
  cycling, a fan modulating, a daily temperature pattern).
- **`NxSignal.Windows`** — Hann, Hamming, Blackman; pair with FFT to reduce
  spectral leakage
- **`NxSignal.Filters`** (IIR/FIR) — low-pass smoothing as an alternative to
  rolling mean, with controllable frequency response
- **`NxSignal.peaks`** — peak detection for event extraction from continuous
  signals
- **`NxSignal.spectrogram`** — when you want to see how spectral content
  changes over time

#### Hand-written defn kernels

Not everything has a library form. Some shapes that come up often and want
maybe 10-20 lines of defn:

- **Time-bucketed group-by**: sort by bucket index, segment-reduce. Pattern:
  `bucket = Nx.quotient(ts - t0, width)` then scatter-add into `Nx.broadcast(0, n_buckets)`
- **State transition detection**: `Nx.diff(state) |> Nx.not_equal(0)`
- **Time-in-state**: cumulative duration where state matches
- **Cross-correlation lag**: max of `NxSignal.correlate` between two metrics

## Default calculations — what runs out of the box

The `MobiusProcessing.Default` module exposes one entry point per data
category. Calling it with a Mobius instance returns a result tensor or map of
tensors. No GenServer, no schedule — call it when you want a snapshot.

```elixir
MobiusProcessing.Default.system_health(instance, window: :last_hour)
# => %{
#   cpu_temp: %{mean: 48.2, max: 51.0, p95: 50.4, slope: +0.012},
#   mem_used_percent: %{mean: 38.1, max: 41.0, p95: 40.7, slope: -0.001},
#   load_1min: %{...},
#   ...
# }

MobiusProcessing.Default.beam(instance, window: :last_hour)
MobiusProcessing.Default.network(instance, window: :last_hour)
MobiusProcessing.Default.sensors(instance, metrics: [:temperature, :humidity], window: :last_hour)
```

These are deliberately boring: descriptive stats, rolling means, simple
slopes. On `nx_eigen` (the demonstration backend) they run in well under a
millisecond for typical Mobius window sizes (1-hour at 1Hz = 3600 samples)
on commodity ARM Cortex-A targets. `Nx.BinaryBackend` works too, slower but
zero-dep. They are the "is the device alive and behaving" view. Anything
more interesting belongs in user code building on the recipes.

## When does computation happen?

Not our call. Three patterns the library supports equally:

- **On demand.** Call `Default.system_health/2` from an IEx session, a Phoenix
  controller, a CLI tool. Cheap enough at typical window sizes that you don't
  need to cache.
- **Periodic.** Schedule a recurring computation from a GenServer, Oban,
  Quantum — store the result somewhere of your choosing (back into Mobius
  as a derived metric, in ETS, in a file, push to NervesHub).
- **Continuous / streaming.** Subscribe to Mobius's report stream, accumulate
  windows yourself, call into our pure functions on each tick. The window
  primitives in `MobiusProcessing.Default` work just as well on a manually
  maintained ring buffer of values as on a Mobius window query.

The library is structured so the same function works for all three. That's
why every operation is a pure function on tensors.

## Dependencies

```elixir
defp deps do
  [
    {:nx, "~> 0.7"},
    {:arrow, path: "../arrow"},        # local for now; from hex eventually
    {:mobius, "~> 0.6"},               # to talk to the storage layer
    # optional:
    {:scholar, "~> 0.3", optional: true},
    {:nx_signal, "~> 0.2", optional: true},
    # backends are not deps — caller picks one
  ]
end
```

`nx_eigen` is **not** a dependency but **is the recommended backend** for
embedded Nerves deployments and is the backend used in every example, every
benchmark, and the LiveBook companion notebook. Users opt in by adding it
to their own deps:

```elixir
# In the consuming application
defp deps do
  [
    {:mobius_processing, "~> 0.1"},
    {:nx_eigen, "~> 0.1"}
  ]
end

# config/config.exs
config :nx, default_backend: NxEigen.Backend
```

`Nx.BinaryBackend` (built into Nx itself) works for everything in this
library without any extra dep — slower, but a useful zero-friction starting
point and the right fallback when running tests in CI without native
toolchains.

## Milestones

### v0.1 — pull and convert

- [ ] `MobiusProcessing.Source.long/1` returns Mobius history as Arrow record batch
- [ ] `MobiusProcessing.Tensor.from_column/1` and `from_batch/1`
- [ ] Round-trip tests on synthetic data
- [ ] Documentation skeleton for `MobiusProcessing.Recipes`

### v0.2 — defaults

- [ ] `MobiusProcessing.Default.system_health/2`, `beam/2`, `network/2`
- [ ] `MobiusProcessing.Source.wide/1` with default alignment
- [ ] `MobiusProcessing.Align.resample/2` with `:previous`, `:linear`, `:bucket_mean`

### v0.3 — recipes

- [ ] Full `MobiusProcessing.Recipes` module with examples for every section
      above
- [ ] LiveBook companion notebook with worked examples on a sample Mobius
      dataset, configured to use `nx_eigen` as the default backend

### Later

- Helpers for emitting computed results back into Mobius as derived metrics
  (if that turns out to be a common pattern — could equally live elsewhere)
- A `MobiusProcessing.Anomaly` module if a handful of anomaly recipes
  consolidate into something with a stable shape (rolling z-score,
  multivariate KNN distance — both viable, both small)

## Non-goals — explicit

- Not a metrics framework. Mobius is. We read from it.
- Not a backend. `nx_eigen`, `EXLA`, `Torchx` are. We use them.
- Not a DataFrame. Explorer/Polars is. We work in tensors.
- Not a scheduler. OTP / Oban / Quantum are. We expose pure functions.
- Not a visualization library. Tucan / VegaLite / Kino are. We output tensors;
  charts are downstream.
