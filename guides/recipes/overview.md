# Recipes — overview

You have a Nerves device, it's emitting telemetry into Mobius, and you
want to know something about it. These recipes are organised around
questions, not functions.

The runnable code lives on four modules — each function has a doctest
verifying the math against synthetic data:

- `MobiusProcessing.Recipes.CoreNx` — is the CPU running hot? is memory
  leaking? how fast is the disk filling? what does the distribution
  look like? Everything you can answer with just Nx, no extra deps.

- `MobiusProcessing.Recipes.Scholar` — are CPU and load actually
  related? cluster every minute into idle/busy/hot. PCA five BEAM
  metrics down to a single "health" axis. Score the current minute
  against historical normal with KNN. Optional dep, but worth it as
  soon as more than one metric is on the table.

- `MobiusProcessing.Recipes.NxSignal` — is the compressor cycling, and
  how often? detect button presses from a vibration trace. low-pass a
  noisy accelerometer trace. Optional dep; reach for it when the data
  is *periodic* or *event-bearing*.

- `MobiusProcessing.Recipes.DefnKernels` — events per hour of day,
  time the heater was on today, lag between a temperature spike and the
  fan ramping up. Short `defn` pieces that fill the gaps between the
  library-level recipes.

## Pulling data in

Every recipe starts from a `{n_samples}` Nx tensor. A one-liner
stand-in for "pull this metric":

```elixir
defp pull_metric_tensor(metric_name) do
  batch = MobiusProcessing.Source.long(metrics: [metric_name])

  value_column = Enum.at(batch.columns, 3)
  {tensor, _validity} = MobiusProcessing.Tensor.from_column(value_column)
  tensor
end
```

Multi-metric work wants a `{n_samples, n_metrics}` tensor, which is
what `MobiusProcessing.Tensor.batch_to_tensor/2` produces from a wide-
format batch (planned for v0.2).

## Pick by data shape

| Shape on the wire | Examples                                | Start here                                                          |
|-------------------|------------------------------------------|---------------------------------------------------------------------|
| Smooth gauge      | `cpu_temp_c`, `mem_used_pct`, `voltage` | `Recipes.CoreNx.ran_hot?/2`, `mem_slope_per_day/1`                  |
| Noisy gauge       | `accelerometer_z`, `microphone_level`   | `Recipes.NxSignal.low_pass/4`                                       |
| Binary state      | `door_opened`, `heater_on`              | `Recipes.DefnKernels.seconds_on/2`, `transition_count/1`            |
| Event count       | `gc_count`, `button_presses_total`      | `Recipes.DefnKernels.events_per_hour/1`                             |
| Monotonic counter | `bytes_rx_total`, `vm.reductions`       | `Recipes.CoreNx.mean_rate/1` first                                  |
| Periodic signal   | `cabinet_temp_c`, vibration             | `Recipes.NxSignal.dominant_period_seconds/2`, `count_peaks_above/3` |
| Many metrics      | full BEAM + system snapshot             | `Recipes.Scholar.pca_first_component/1`                             |

## Backend, briefly

All recipes assume `nx_eigen` is the default Nx backend. That's the
embedded-CPU SIMD story this library is positioned around. They run
unchanged on `Nx.BinaryBackend` — slower, no native toolchain
required, fine for CI and what the doctests exercise.
