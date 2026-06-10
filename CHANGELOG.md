# Changelog

## v0.1.0 (unreleased)

Initial release: on-device health detection over
[Mobius](https://github.com/mobius-home/mobius) metric storage, modeled
in Nx.

### Detectors — `MobiusSmarts.Detect.*`

Pure functions over tensors or lists; each module documents its theory,
tuning knobs, and blind spots.

- `Jump` — Shewhart X̄/S charts with c4 correction: sudden jumps and
  rising within-window "wobble".
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
aggregate health level. One `:watch` list and one `:false_alarm_budget`;
every detector threshold derives from the budget via its own
average-run-length math (`MobiusSmarts.Calibrate`), baselines are fitted
from the device's own stored history, and configuration is validated
with pointed errors at startup. Findings carry onset, a cross-detector
`concern` ratio, evidence, and a human message; lifecycle and health
events emit via telemetry.

### Source — `MobiusSmarts.Source`

Mobius data as Nx tensors: per-window summary series (average/std_dev/
reports), plain numeric series, and DDSketch reconstruction, plus pure
converters for tests and replays.

`summary_series/3` resamples Mobius's mixed-cadence RRD windows to a
uniform cadence (`resolution: :auto` by default, explicit duration or
`:native` available; `resample_windows/2` is the pure form). Mobius
deltas consecutive snapshots across all four RRD archives at once, so
without this any query spanning more than the seconds archive fed the
detectors second-cadence windows for the freshest minutes and
minute-cadence behind them — the runtime read every archive-tier
boundary as a reporting gap, re-anchored to the ~2-minute
second-resolution tail (single-report windows, no dispersion), and
every watched metric sat in `:learning` forever.

### Pre-release review history

The detector math and runtime went through three review/fix passes
before release; the records are in `CRITIQUE.md`, `QUALITY.md`, and
`REVIEW.md`.
