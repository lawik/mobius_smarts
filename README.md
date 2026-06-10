# MobiusSmarts

On-device health detection over [Mobius](https://github.com/mobius-home/mobius)
metric storage, modeled in Nx.

Mobius stores per-window summary statistics (average, std_dev, report count)
and DDSketch histograms. Those happen to be the native input formats of a
century of statistical process control and distribution-drift theory — this
library implements that detector stack as pure functions over Nx tensors:

| Detector | Detects |
|---|---|
| `Detect.Jump` | sudden jumps (X̄ chart); erratic "wobble" (S chart) |
| `Detect.Drift` | small persistent drifts, with onset dating |
| `Detect.Shift` | moderate sustained shifts, noise-smoothed |
| `Detect.Trend` | slow monotonic trends; time-to-threshold ETAs |
| `Detect.Changepoint` | retrospective "when did behavior change" |
| `Detect.Shape` | tail/shape drift the moments can't see (PSI, JS, earth-mover) |
| `Detect.Novelty` | cross-metric correlation breaks |
| `Detect.Outlier` | fleet-trained Isolation Forest scoring; weird windows no rule covers (inference) |

```elixir
alias MobiusSmarts.{Detect, Source}

# Mobius stores several RRD resolutions at once; you state the tier
# you mean to read at — nothing is inferred from the data.
%{average: avgs, std_dev: stds, reports: counts} =
  Source.summary_series("vm.memory.used_percent", %{},
    last: {2, :day},
    resolution: {1, :hour}
  )

baseline = Detect.Jump.baseline(avgs, stds, counts)

# Each detector picks its own noise scale from the baseline map.
Detect.Drift.scan(avgs, baseline: baseline)
#=> %{upper_alarm: 161, upper_onset: 142, ...}

Detect.Jump.scan(avgs, stds, counts, baseline: baseline)
```

The baseline map carries two distinct noise scales — `sigma_avg` (sd of
the window averages, what Drift/Shift calibrate against) and
`sigma_reports` (per-report sd, what Jump's `sqrt(n)`-scaled limits
want). They differ by `sqrt(reports_per_window)`, and the wrong one
changes detector sensitivity by that factor — which is why the
recommended wiring is to hand every detector the map and let it pick.
Explicit `:target`/`:sigma` options remain for overrides.

Detectors take tensors or plain lists; batch scans are vectorized in Nx and
the streaming forms (`Drift.step/2`, `Shift.step/2`) carry O(1) state per
metric. The detectors are deliberately pure and schedule-free; scheduling
and findings are the optional runtime's job (below), while alert delivery
and persistence stay the host app's, via telemetry.

## The runtime

You don't have to drive the detectors yourself. The optional `MobiusSmarts`
supervision tree polls Mobius on an interval, runs the stack over every
watched metric, and maintains *findings* and an aggregate health level:

```elixir
children = [
  {Mobius, metrics: metrics()},
  {MobiusSmarts,
   config: [
     watch: [
       "vm.memory.used_percent",
       [metric: "disk.used_percent", ceiling: 95.0],
       [metric: "cpu.temp_c", ceiling: 85.0]
     ],
     # The Mobius RRD tier the detectors operate on, and the unit
     # behind the false-alarm math — stated, never inferred.
     resolution: {1, :minute},
     # Tolerate about one false alarm per week, instance-wide.
     false_alarm_every: {1, :week},
     # Ceilings opt into ETA projections, fitted at a coarser tier
     # over the (default 24-hour) trend window.
     trend_resolution: {1, :hour}
   ]}
]

MobiusSmarts.status()
#=> %{level: :watch, concern: 1.2, findings: [...], ...}
```

There are no per-detector thresholds to tune: the configuration states one
false-alarm budget (`false_alarm_every: {1, :week}` — "this device may cry
wolf about once a week") and the window cadence it is counted in
(`resolution:`), and every detector's threshold is derived from those two
via that detector's average-run-length math; the derived numbers are logged
once at startup. Baselines are fitted from the device's own stored history.
Findings surface through `MobiusSmarts.status/0` and telemetry events —
what to do with them (alarms, notifications, persistence) is the host
app's call. The `MobiusSmarts` moduledoc has the full story: learning,
finding lifecycle, health levels, telemetry.

## Installation

> **Unreleased.** Not on Hex yet — this tracks Mobius `main` on GitHub
> (`{:mobius, github: "mobius-home/mobius", branch: "main"}`) for the
> summary-window and histogram APIs not yet in a Hex release. The snippet
> below is the intended shape once both are published.

```elixir
def deps do
  [
    {:mobius_smarts, "~> 0.1.0"},
    {:nx_eigen, "~> 0.1"}
  ]
end

# config/config.exs
config :nx, default_backend: NxEigen.Backend
```

`nx_eigen` is the recommended backend on Nerves targets — CPU SIMD via Eigen,
small footprint, no XLA/LLVM toolchain. It is optional: everything runs on the
built-in `Nx.BinaryBackend` (the right pick for CI). The O(n) detectors run in
microseconds-to-milliseconds at RRD scale on either backend; `Detect.Trend`'s
`theil_sen/2` is the O(n²) exception (~100 ms at a thousand windows, plain
Elixir — its moduledoc has the numbers and the scheduling guidance). Quoted
false-alarm rates assume independent windows; see the calibration caveat in
`MobiusSmarts.Detect` before trusting them on seasonal or autocorrelated
metrics (`Detect.lag1_autocorrelation/1` is the quick check).

## What lives where

- `MobiusSmarts` — the optional runtime: supervision tree, config,
  budget-derived calibration, findings and health level.
- `MobiusSmarts.Source` — Mobius → tensors: summary window series,
  numeric series, DDSketch reconstruction.
- `MobiusSmarts.Detect.*` — the detectors. Each module documents its
  theory, tuning knobs, and blind spots.

Tests are split the same way: per-detector conformance tests pin the
implementations to the textbook math (hand-computed statistics, closed forms,
published SPC constants), and `test/mobius_smarts/scenarios_test.exs`
runs end-to-end detection narratives — a slow memory leak, a sensor going
erratic, a latency distribution going bimodal, a CPU/network correlation
break, a deploy regression — asserting the right detector fires and the wrong
one stays quiet.

See [SPEC.md](SPEC.md) for design rationale.
