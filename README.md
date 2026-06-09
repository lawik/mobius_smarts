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

%{average: avgs, std_dev: stds, reports: counts} =
  Source.summary_series("vm.memory.used_percent", %{}, last: {7, :day})

baseline = Detect.Jump.baseline(avgs, stds, counts)

# Drift/Shift watch the average series — they take sigma_avg.
Detect.Drift.scan(avgs, target: baseline.target, sigma: baseline.sigma_avg)
#=> %{upper_alarm: 161, upper_onset: 142, ...}

# Jump scales its own limits by sqrt(n) — it takes the baseline map.
Detect.Jump.scan(avgs, stds, counts, baseline: baseline)
```

Mind the two sigma scales: `baseline.sigma_avg` (sd of the window averages)
calibrates Drift/Shift; `baseline.sigma_reports` (per-report sd) calibrates
Jump. They differ by `sqrt(reports_per_window)`, and the wrong one changes
detector sensitivity by that factor.

Detectors take tensors or plain lists; batch scans are vectorized in Nx and
the streaming forms (`Drift.step/2`, `Shift.step/2`) carry O(1) state per
metric. Scheduling, alerting, and persistence are deliberately the caller's
problem.

## Installation

> **Unreleased.** Not on Hex yet — this tracks the unreleased Mobius
> `histograms` branch via a path dependency (`{:mobius, path: "../mobius"}`).
> The snippet below is the intended shape once both are published.

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
metrics.

## What lives where

- `MobiusSmarts.Source` — Mobius → tensors: summary window series,
  numeric series, DDSketch reconstruction.
- `MobiusSmarts.Detect.*` — the detectors. Each module documents its
  theory, tuning knobs, and blind spots.
- `MobiusSmarts.Recipes.*` — general-purpose tensor recipes for
  Mobius-shaped data (rates from counters, duty cycles, bucketed histograms).

Tests are split the same way: per-detector conformance tests pin the
implementations to the textbook math (hand-computed statistics, closed forms,
published SPC constants), and `test/mobius_smarts/scenarios_test.exs`
runs end-to-end detection narratives — a slow memory leak, a sensor going
erratic, a latency distribution going bimodal, a CPU/network correlation
break, a deploy regression — asserting the right detector fires and the wrong
one stays quiet.

See [SPEC.md](SPEC.md) for design rationale.
