# Recipes — overview

You have a Nerves device, it's emitting telemetry into Mobius, and you want
to know something about it. The *detectors* answer "is something wrong?" —
see `MobiusSmarts.Detect` for that stack. The recipes here are the
general-purpose building blocks around them, organised by question:

- `MobiusSmarts.Recipes.CoreNx` — is the CPU running hot? is memory
  leaking? how fast is the disk filling? what does the distribution look
  like? Everything answerable with just Nx.

- `MobiusSmarts.Recipes.DefnKernels` — events per hour of day, time the
  heater was on today, run-length structure of a binary state. Short `defn`
  pieces that fill the gaps.

Every function has a doctest verifying the math against synthetic data.

## Pulling data in

Detector-shaped data comes from `MobiusSmarts.Source`:

```elixir
# Summary windows (average/std_dev per window) — the detector input
%{average: avgs, std_dev: stds} =
  MobiusSmarts.Source.summary_series("vm.memory.used_percent", %{}, last: {1, :day})

# Plain numeric series for the recipes
%{values: values} =
  MobiusSmarts.Source.series("cpu_temp", :last_value, %{}, last: {1, :hour})
```

Both return `:empty` when the window has no data — handle it; a silent
device is a finding, not an edge case.

## Pick by data shape

| Shape on the wire | Examples                                | Start here                                               |
|-------------------|------------------------------------------|-----------------------------------------------------------|
| Smooth gauge      | `cpu_temp_c`, `mem_used_pct`, `voltage` | `Detect.Drift`, `Detect.Trend`, `Recipes.CoreNx.ran_hot?/2` |
| Erratic-when-failing | sensor noise, link quality           | `Detect.Jump` (the wobble side)                         |
| Latency-like      | request duration, queue wait            | `Detect.Shape` over DDSketch pairs                        |
| Binary state      | `door_opened`, `heater_on`              | `Recipes.DefnKernels.seconds_on/2`, `transition_count/1`  |
| Event count       | `gc_count`, `button_presses_total`      | `Recipes.DefnKernels.events_per_hour/1`                   |
| Monotonic counter | `bytes_rx_total`, `vm.reductions`       | `Recipes.CoreNx.mean_rate/1` first                        |
| Many metrics      | full BEAM + system snapshot             | `Detect.Novelty`                                           |

## Backend, briefly

`nx_eigen` is the recommended default backend on Nerves targets — the
embedded-CPU SIMD story this library is positioned around. Everything runs
unchanged on `Nx.BinaryBackend` — slower, no native toolchain required, fine
for CI and what the doctests exercise.
