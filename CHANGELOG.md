# Changelog

## v0.1.0 (unreleased)

Initial release: on-device health detection over
[Mobius](https://github.com/mobius-home/mobius) metric storage, modeled
in Nx.

### Detectors — `MobiusSmarts.Detect.*`

Pure functions over tensors or lists; each module documents its theory,
tuning knobs, and blind spots.

- `Jump` — Shewhart X̄/S charts with c4 correction: sudden jumps,
  rising within-window "wobble", and collapsed spread ("flatlined",
  the stuck-signal signature). The S-chart's lower limit arms only
  when the baseline pool carried dispersion in every window —
  zero-inflated metrics (an idle run queue is exactly 0 for whole
  windows at a time) would otherwise alarm on every healthy idle
  window (`baseline/3` records the pool's `:sd_floor`).
- `Drift` — two-sided CUSUM with onset dating; batch scan (reflection
  identity) and O(1)-state streaming form.
- `Shift` — EWMA chart with exact time-varying limits; batch and
  streaming forms.
- `Trend` — Theil–Sen slope (O(n²) exact), Mann–Kendall test (O(n log n)
  inversion counting), and time-to-threshold ETAs anchored on the
  fitted line.
- `Changepoint` — binary segmentation with prefix-sum SSE cost,
  BIC-style penalty, and a robust (MAD-based) noise estimate;
  shift-invariant at any metric magnitude.
- `Shape` — PSI, Jensen–Shannon, and first Wasserstein distances over
  aligned DDSketch bins, plus the signed `mean_shift/3`.
- `Novelty` — Mahalanobis distance against the device's own
  mean/covariance, Cholesky-solved, ridge-stabilized.
- `Outlier` — Isolation Forest inference for fleet-trained models;
  loads scikit-learn exports (including sklearn's `-2` leaf markers).
- `Detect.lag1_autocorrelation/1` — the executable first step of the
  calibration checklist for autocorrelated/seasonal metrics.

Drift and Shift accept the baseline map from `Jump.baseline/3` directly
(`baseline:` option) and pick the correct sigma scale themselves.

### Runtime — `MobiusSmarts`

An optional supervision tree over the pure layers: polls Mobius on an
interval, runs the detector stack, and maintains findings and an
aggregate health level. Configuration states facts and tolerances
explicitly — nothing is inferred from the data: `:watch` (the metrics),
`:resolution` (the RRD window cadence every detector operates on, and
the unit of the false-alarm math), `:false_alarm_every` (the budget;
`{1, :week}` is one false alarm a week), and `:trend_resolution` when
ceilings/floors opt into ETA projections. Every detector threshold
derives from the budget via its own average-run-length math
(`MobiusSmarts.Calibrate`), the derived numbers are logged once at
startup, baselines are fitted from the device's own stored history, and
configuration is validated with pointed errors at startup (including
that `:analysis_window` can hold `:min_baseline_windows` windows of
`:resolution` width). ETA projections are additionally gated on the
fitted series spanning at least half of `:trend_window`, so a 7-day
forecast is never extrapolated from an hour of data. Findings carry
onset, a cross-detector `concern` ratio, evidence, and a human message;
lifecycle and health events emit via telemetry.

### Source — `MobiusSmarts.Source`

Mobius data as Nx tensors: per-window summary series (average/std_dev/
reports), plain numeric series, and DDSketch reconstruction, plus pure
converters for tests and replays.

`summary_series/3` resamples Mobius's mixed-cadence RRD windows to a
required, explicitly stated `resolution:` (a duration, or `:native` for
the raw windows; `resample_windows/2` is the pure form — merging is
exact, recombining the sum / sum-of-squares / count deltas). Mobius
deltas consecutive snapshots across all four RRD archives at once, so
without this any query spanning more than the seconds archive fed the
detectors second-cadence windows for the freshest minutes and
minute-cadence behind them — the runtime read every archive-tier
boundary as a reporting gap, re-anchored to the ~2-minute
second-resolution tail (single-report windows, no dispersion), and
every watched metric sat in `:learning` forever. The cadence is never
inferred from the data: stating the tier you monitor at is part of the
configuration contract.

### Pre-release review history

The detector math and runtime went through three review/fix passes
before release; the records are kept as local working notes outside
the published repo.
