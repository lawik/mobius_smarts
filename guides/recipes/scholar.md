# Recipes with Scholar

Scholar is where the "compare metrics" and "classify behavior" recipes
live. Optional dep:

```elixir
{:scholar, "~> 0.3"}
```

Each recipe here is framed around a question you might ask of one or
more device metrics. The data names are illustrative.

## Are CPU temperature and CPU load actually related?

A reasonable hypothesis: the device is hot when it's busy. Pearson
correlation tells you whether that's true *for this device* over the
last hour, with a number between -1 and 1.

```elixir
cpu_temp_c    = pull_metric_tensor("cpu_temperature")
cpu_usage_pct = pull_metric_tensor("cpu_usage_percent")

# Stack into {n_samples, 2} and ask for the correlation matrix
xs = Nx.stack([cpu_temp_c, cpu_usage_pct], axis: 1)
correlation = Scholar.Stats.correlation_matrix(xs)
# correlation[[0, 1]] is r(temp, usage)
```

If `r` is near +1, throttling logic could fire on load instead of
waiting for the thermometer. If `r` is near 0, the device is heating
from something else — ambient, a peripheral, the sun.

## Putting unlike metrics on a common scale

`mem_used_pct` is in [0, 100], `cpu_temp_c` is around 30–80, and
`bytes_rx_per_sec` is in the millions. You can't compare them
visually without scaling.

```elixir
# {n_samples, n_metrics}
metrics_raw =
  Nx.stack(
    [mem_used_pct, cpu_temp_c, Nx.as_type(bytes_rx_per_sec, :f32)],
    axis: 1
  )

scaler        = Scholar.Preprocessing.StandardScaler.fit(metrics_raw)
metrics_zscore = Scholar.Preprocessing.StandardScaler.transform(scaler, metrics_raw)
# Every column is now mean=0, stddev=1 — directly comparable.
```

`StandardScaler` is the right pick when you care about *deviation from
typical*. Use `MinMaxScaler` when you care about *position in the
observed range*, e.g. "what fraction of its way to full is the disk?"

## Estimating "days until disk is full"

Same job as the manual `lstsq` recipe in [Core Nx](core_nx.md), but
Scholar's `LinearRegression` saves the design-matrix bookkeeping when
you have several predictors.

```elixir
disk_used_pct = pull_metric_tensor("disk_used_percentage")  # daily
n             = Nx.size(disk_used_pct)
days          = Nx.iota({n, 1}, type: :f32)

model = Scholar.Linear.LinearRegression.fit(days, disk_used_pct)
[slope_pct_per_day] = Nx.to_flat_list(model.coefficients)
intercept_pct       = Nx.to_number(model.intercept)

current_pct = Nx.to_number(disk_used_pct[-1])
days_to_95  = (95.0 - current_pct) / slope_pct_per_day
```

The model object holds `coefficients` and `intercept` so you can keep
predicting from new data without re-fitting.

## Clustering device state: idle / busy / hot

Group every minute-long snapshot of the device into "modes" — a quick
proxy for "what kind of work is this device usually doing?"

```elixir
# Per-minute means of three metrics, all 1Hz, windowed to 60 samples.
cpu_per_min  = Nx.window_mean(cpu_usage_pct, 60)
mem_per_min  = Nx.window_mean(mem_used_pct, 60)
temp_per_min = Nx.window_mean(cpu_temp_c, 60)

device_state =
  Nx.stack([cpu_per_min, mem_per_min, temp_per_min], axis: 1)
  |> Scholar.Preprocessing.StandardScaler.fit_transform()

key = Nx.Random.key(0)
{model, _} = Scholar.Cluster.KMeans.fit(device_state, num_clusters: 3, key: key)
labels = Scholar.Cluster.KMeans.predict(model, device_state)
# labels[i] ∈ {0, 1, 2} — inspect cluster centers to label them
# idle / busy / hot by hand.
```

The cluster centers (`model.clusters`) give the "average" reading of
each mode in the standardised space. Reverse the scaler to read them
in their original units.

## A "device health score" from many BEAM metrics

`vm.memory.processes`, `vm.memory.binary`, `vm.process_count`,
`vm.port_count`, `vm.run_queue` are all correlated — most of them say
the same thing when the device is busy. PCA collapses that into a
single "load" axis.

```elixir
beam_metrics =
  Nx.stack(
    [vm_mem_processes_mb, vm_mem_binary_mb, vm_proc_count,
     vm_port_count, vm_run_queue],
    axis: 1
  )
  |> Scholar.Preprocessing.StandardScaler.fit_transform()

{model, _} = Scholar.Decomposition.PCA.fit(beam_metrics, num_components: 2)
health_2d  = Scholar.Decomposition.PCA.transform(model, beam_metrics)

# The first component is usually a "general activity" axis.
# Track it over time; spikes are interesting.
activity_axis = health_2d[[.., 0]]
```

The variance ratio (`model.explained_variance_ratio`) tells you how
much of the original signal that one axis kept. If it's > 0.7, the
five BEAM metrics were mostly carrying one piece of information.

## Anomaly detection by nearest-neighbor distance

Given a corpus of "the last week of normal operation", flag the
current minute when it sits far from anything we've seen before.

```elixir
# Corpus: every minute of last week as a 3-feature row
historical_state = Nx.stack([cpu_per_min, mem_per_min, temp_per_min], axis: 1)
scaler  = Scholar.Preprocessing.StandardScaler.fit(historical_state)
corpus  = Scholar.Preprocessing.StandardScaler.transform(scaler, historical_state)

# Index it with KDTree for fast lookup
tree = Scholar.Neighbors.KDTree.fit(corpus)

# Score the *current* minute against the 5 nearest historical states
current = Nx.tensor([[cpu_now_pct, mem_now_pct, temp_now_c]])
current = Scholar.Preprocessing.StandardScaler.transform(scaler, current)

{distances, _indices} = Scholar.Neighbors.KDTree.predict(tree, current, k: 5)
anomaly_score = Nx.mean(distances) |> Nx.to_number()
```

A growing `anomaly_score` over successive ticks is "this device is
drifting away from how it usually behaves" — the kind of signal
you'd raise to NervesHub for a closer look.

## Was my threshold rule any good?

If you wrote a "temperature > 70°C ⇒ throttle" rule, evaluate it
against logged ground truth before shipping it.

```elixir
# Boolean ground truth (1 = throttling was needed) and predictions
needed_throttle = Nx.tensor([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
predicted       = Nx.greater(cpu_temp_minute_max, 70.0)

precision = Scholar.Metrics.Classification.precision(needed_throttle, predicted)
recall    = Scholar.Metrics.Classification.recall(needed_throttle, predicted)
```

The same two numbers tell the story for any "alert when X" rule:
how often did you cry wolf (1 - precision), and how often did you miss
real trouble (1 - recall).
