# mobius_smarts — design

On-device health detection over [Mobius](https://github.com/mobius-home/mobius)
metric storage, modeled in Nx. The library turns what Mobius already stores
into detections: level shifts, slow drifts, rising erraticness, distribution
drift, correlation breaks, change points, and time-to-threshold projections.

The on-device side sees full-resolution data, works offline, and can
trigger richer capture (extra metrics, event-log snapshots) while a
problem is live. Findings surface through `MobiusSmarts.status/0` &
friends and telemetry events; what to do with them — alerting,
persistence, forwarding anywhere — is the host app's call.

## Why these inputs

Mobius summary windows store sufficient statistics per window: average,
std_dev, report count. That is *exactly* the subgroup format classical SPC
(X̄/S charts) was designed for — no approximation involved. The std_dev
solves the hardest practical problem in drift detection for free: estimating
the noise scale that calibrates CUSUM/EWMA thresholds. The DDSketch
histograms carry distribution shape — the failure modes (p99 drift,
bimodality) that are mathematically invisible to the first two moments.

## Principles

- **Pure functions.** No GenServers, no schedulers. On demand, periodic, or
  streaming is the caller's call; the same functions serve all three.
- **Batch and streaming forms.** Batch scans are vectorized in Nx (CUSUM via
  the reflection identity, change-point scans via prefix sums, Theil-Sen via
  pairwise matrices). Sequential recursions that Nx can't vectorize (EWMA)
  run as plain folds — at RRD scale that is microseconds. Streaming forms
  (`Drift.step/2`, `Shift.step/2`) carry O(1) float state per metric.
- **BYO backend.** `nx` is the only hard runtime dep besides `mobius`.
  `nx_eigen` is the recommended backend on Nerves targets (CPU SIMD, no
  XLA/LLVM toolchain); `Nx.BinaryBackend` is fine for CI and is what the
  test suite exercises.
- **Theory you can audit.** Every detector module documents its source
  theory and tuning knobs; the test suite pins implementations to
  hand-computed statistics, closed forms, and published SPC constants, plus
  end-to-end scenario tests that assert the right detector fires and the
  wrong one stays quiet.

## API surface

### `MobiusSmarts.Source` — Mobius → tensors

- `summary_series/3` — per-window `average`/`std_dev`/`reports` tensors
  from `Mobius.Data.summary_windows/3`. The main detector input.
- `series/4` — numeric series of any non-summary metric type
  (`Mobius.Data.metrics/4`).
- `sketch/3` — DDSketch reconstruction over a window
  (`Mobius.Data.histogram/3`).
- `from_metrics/1`, `from_summary_windows/1` — pure converters for tests and
  replays. Empty input returns `:empty` (Nx has no zero-sized tensors, and
  an empty window is a missingness signal the caller should handle anyway).

### `MobiusSmarts.Detect` — the detectors

Each is one module; the umbrella moduledoc carries the
which-detector-for-which-failure-mode table.

| Module | Theory | Mobius input |
|---|---|---|
| `Jump` | Shewhart X̄/S with c4 correction, pooled sigma | averages + std_devs + counts |
| `Drift` | Page's CUSUM, two-sided, onset estimation | averages (+ sigma from baseline) |
| `Shift` | Lucas & Saccucci EWMA chart, exact time-varying limits | averages (+ sigma) |
| `Trend` | Theil-Sen, Mann-Kendall, threshold ETA | trailing averages (+ timestamps) |
| `Changepoint` | binary segmentation, SSE cost, BIC-style penalty, robust sigma | trailing averages |
| `Shape` | PSI, Jensen-Shannon, Wasserstein over aligned bins | DDSketch pairs |
| `Novelty` | covariance-aware distance, ridge-stabilized | per-window means across metrics |

Detector outputs favor *diagnosis* over bare booleans: CUSUM dates the onset
of a shift, not just its detection; Trend returns ETAs in seconds; the
change-point sweep returns the regime boundaries worth correlating with the
event log.

### `MobiusSmarts.Recipes` — general-purpose tensor recipes

`CoreNx` (rates from counters, rolling stats, z-scores, bucketed histograms)
and `DefnKernels` (events-per-hour, duty cycles, run lengths). Kept from the
library's first life as documented, doctested building blocks.

## Data shapes expected on a Nerves device

| Shape | Examples | Detector path |
|---|---|---|
| Smooth gauge | cpu_temp, mem_used_pct, voltage | summary windows → Jump/Shift/Drift/Trend |
| Noisy gauge | accelerometer, mic level | summary windows; S chart carries most signal |
| Latency-like | request duration, queue wait | DDSketch → Shape (p99/shape drift) |
| Binary state | relay on, link up | Recipes (duty cycle, transitions) |
| Monotonic counter | bytes_rx, gc_count | Recipes.mean_rate → then gauge-style detection |
| Many metrics | full health snapshot | Novelty over window-mean vectors |

## Nx strategy — how accelerated, and why not more

Nx earns its keep here on the O(n) pipelines: the cumulative-op CUSUM
identity (Drift), the all-split-points prefix-sum scan (Changepoint),
the batched quadratic form (Novelty), the bin arithmetic (Shape), and
the elementwise chart limits (Jump, Shift). Those are `defn` kernels:
under `Nx.BinaryBackend` or `nx_eigen` — backends, not compilers — they
run through the evaluator at eager speed, and the kernels exist so the
same code JITs and fuses unchanged under a compiling backend (EXLA),
the fleet-side batch-scoring scenario.

Deliberately *not* Nx, each for a measured or structural reason:

- **Trend is plain Elixir.** The original pairwise tensor kernels were
  benchmarked against plain BEAM loops computing identical statistics
  (n = 1000, BinaryBackend): tensor Mann–Kendall 1.44 s vs 4 ms as a
  loop vs **0.2 ms** as an O(n log n) inversion-counting merge sort;
  tensor Theil–Sen 3.9 s vs **109 ms** as a loop. A 35–350×
  interpretation penalty on the backends devices actually run buys
  nothing for a module whose matrices also cost O(n²) memory. The
  earlier claim here — that pairwise formulations "would be slow as
  Enum code" — was exactly backwards and is preserved as a warning:
  measure before vectorizing.
- The EWMA recursion is a fold; `Nx.while` would be tensor ceremony
  for sequential math.
- The Changepoint *recursion* produces a different segment length at
  every level — retrace-per-shape is the pathological JIT case (its
  inner cost scan is the defn part).
- All nil-returning alarm/onset logic and the streaming `step/2` forms
  are plain floats: O(1) state per metric is the on-device deployment
  story, and no tensor runtime belongs in that loop.

The "one compute graph, shared inputs, many outputs" idea stays
deferred. The real batch win, if it ever matters, is not fusing one
metric's detectors but *vectorizing across metrics* —
`{n_metrics, n_windows}` tensors pushing every metric through
Jump/Drift/Shift in one broadcasted pass under EXLA fleet-side. That
restructuring is mechanical on the current elementwise kernels;
on-device, per-metric calls at microseconds each are already past the
point of diminishing returns.

## Non-goals — explicit

- Not a metrics framework. Mobius is. We read from it.
- Not a backend. `nx_eigen`, `EXLA`, `Torchx` are. We use them.
- Not an alerter. The `MobiusSmarts` runtime owns cadence and findings;
  delivery (alarms, notifications, persistence) is the host app's, via
  telemetry handlers.
- Not the fleet side. Cross-device aggregation is a separate concern;
  this library is on-device only.
- Not a DataFrame. Tensors in, tensors/numbers out.

## Open threads

- Seasonal handling (hour-of-day baselines) for metrics with daily cycles —
  currently the caller's slicing problem. Per the calibration caveat in
  `MobiusSmarts.Detect`, for strongly cyclic metrics this is a
  prerequisite for the quoted false-alarm rates, not an enhancement; the
  Hamed–Rao autocorrelation correction for Mann–Kendall belongs in the
  same bundle.
- Done: Trend rewritten in plain Elixir after benchmarking — Mann–Kendall
  is now O(n log n) inversion counting (sub-ms at any RRD scale),
  Theil–Sen a plain O(n²) loop (~100 ms at n = 1000, ~35× faster than
  the tensor kernels it replaced). Numbers in the Nx strategy section.
- Done: Isolation Forest *inference* on-device (trees shipped from
  fleet-side training) landed as `MobiusSmarts.Detect.Outlier` —
  plain Elixir tree-walking, no new deps; a fitted
  `sklearn.ensemble.IsolationForest` exports to the module's JSON shape,
  loads via `load!/1`, and scores window vectors with `score/2`.
