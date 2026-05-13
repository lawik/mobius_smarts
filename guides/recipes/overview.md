# Recipes — overview

You have a Nerves device, it's emitting telemetry into Mobius, and you
want to know something about it. These guides are organised around
questions, not functions:

- [Core Nx](core_nx.md) — is the CPU running hot? is memory leaking?
  how fast is the disk filling? what does the temperature distribution
  look like? Everything you can answer with just Nx, no extra deps.

- [Scholar](scholar.md) — are CPU and load actually related? cluster
  every minute into idle/busy/hot. PCA five BEAM metrics down to a
  single "health" axis. Score the current minute against historical
  normal with KNN. Optional dep, but worth it as soon as more than one
  metric is on the table.

- [NxSignal](nx_signal.md) — is the compressor cycling, and how often?
  detect button presses from a vibration trace. spectrogram of CPU
  usage to see when the workload shifted. Optional dep; reach for it
  when the data is *periodic* or *event-bearing*.

- [Hand-written defn kernels](defn_kernels.md) — events per hour of
  day, time the heater was on today, lag between a temperature spike
  and the fan ramping up. Short pieces of `defn` that fill gaps
  between the library-level recipes.

## Pulling data in

Every recipe starts from a `{n_samples}` Nx tensor. `pull_metric_tensor/1`
in the examples is a one-liner stand-in for:

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

| Shape on the wire | Examples                                | Start here                                |
|-------------------|------------------------------------------|-------------------------------------------|
| Smooth gauge      | `cpu_temp_c`, `mem_used_pct`, `voltage` | Core Nx → "Is the CPU running hot?"       |
| Noisy gauge       | `accelerometer_z`, `microphone_level`   | NxSignal → "Smoothing a noisy trace"      |
| Binary state      | `door_opened`, `heater_on`              | defn → "Time-in-state"                    |
| Event count       | `gc_count`, `button_presses_total`      | defn → "Events per hour"                  |
| Monotonic counter | `bytes_rx_total`, `vm.reductions`       | Core Nx → "Counter into a rate" first     |
| Periodic signal   | `cabinet_temp_c`, vibration             | NxSignal → "Compressor cycling"           |
| Many metrics      | full BEAM + system snapshot             | Scholar → "Device health score from PCA"  |

## Backend, briefly

All snippets assume `nx_eigen` is the default Nx backend. That's the
embedded-CPU SIMD story this library is positioned around. They run
unchanged on `Nx.BinaryBackend` — slower, no native toolchain
required, fine for CI.
