defmodule MobiusSmarts.Detect do
  @moduledoc """
  On-device health detection over Mobius data.

  A small stack of classical detectors, each named after the question
  it answers, each covering the others' blind spot, all driven by what
  Mobius already stores. The per-metric summary window — `average`,
  `std_dev`, report count — is the native input format of subgrouped
  statistical process control, and the DDSketch histograms carry the
  distribution shape the moments can't.

  Route by symptom:

  | You suspect... | Reach for | (implements) |
  |---|---|---|
  | "it suddenly jumped way off" | `MobiusSmarts.Detect.Jump` | Shewhart X̄ chart |
  | "it's gotten shaky / erratic" | `MobiusSmarts.Detect.Jump` (wobbles) | Shewhart S chart |
  | "it moved and stayed moved" | `MobiusSmarts.Detect.Shift` | EWMA chart |
  | "it's slowly creeping — since when?" | `MobiusSmarts.Detect.Drift` | CUSUM |
  | "it's heading for a wall — when does it hit?" | `MobiusSmarts.Detect.Trend` | Theil–Sen / Mann–Kendall |
  | "something changed in the past — when, exactly?" | `MobiusSmarts.Detect.Changepoint` | binary segmentation |
  | "the average is fine but the shape feels wrong" | `MobiusSmarts.Detect.Shape` | PSI / JS / Wasserstein |
  | "each metric is fine but together they're weird" | `MobiusSmarts.Detect.Novelty` | Mahalanobis distance |
  | "is this weird in a way no rule covers?" | `MobiusSmarts.Detect.Outlier` | Isolation Forest (fleet-trained) |

  Jump, Shift, and Drift form a size/speed gradient over level changes
  — big-and-sudden, moderate-and-sustained, small-and-slow. The
  boundaries are fuzzy by nature; running all three in parallel is the
  intended deployment, and each module's docs say which sibling covers
  its blind spot.

  ## The shape of a deployment

  Pull window series with `MobiusSmarts.Source`, establish a
  baseline over a healthy stretch, then run the cheap detectors every
  window and the sweep detectors periodically:

      alias MobiusSmarts.{Detect, Source}

      %{average: avgs, std_dev: stds, reports: counts} =
        Source.summary_series("vm.memory.used_percent", %{}, last: {7, :day})

      baseline = Detect.Jump.baseline(avgs, stds, counts)

      # Drift/Shift watch the *average* series: they take sigma_avg.
      Detect.Drift.scan(avgs, target: baseline.target, sigma: baseline.sigma_avg)
      #=> %{upper_alarm: 161, upper_onset: 142, ...}

      # Jump scales its own limits by sqrt(n): it takes the baseline map
      # (or its {target, sigma_reports} pair) directly.
      Detect.Jump.scan(avgs, stds, counts, baseline: baseline)

  The two sigma scales differ by `sqrt(reports_per_window)` — wiring
  `sigma_reports` into Drift/Shift makes them nearly deaf, which is why
  `Detect.Jump.baseline/3` returns both, explicitly named.

  ## Calibration assumes independent windows

  Every quoted false-alarm number — Jump's "once in ~370 windows",
  Drift's ARL (~940 per side at defaults), Shift's exact bands,
  Mann–Kendall p-values, Novelty's chi-square threshold recipe —
  presumes in-control windows that are independent and identically
  distributed. Real device telemetry often is not: diurnal temperature
  cycles, load patterns, and autocorrelated noise can inflate
  false-alarm rates by an order of magnitude (a CUSUM baselined on a
  cool morning will alarm every warm afternoon, forever). Before
  trusting the numbers: baseline over a window covering the full cycle
  (whole days, not hours), lengthen aggregation windows until lag-1
  autocorrelation is small, consider differencing strongly seasonal
  metrics or keeping hour-of-day baselines, and sanity-check the
  false-alarm rate empirically on a known-healthy fortnight. Seasonal
  baselines are a known gap, tracked in SPEC.md — for strongly cyclic
  metrics they are a prerequisite, not an enhancement.

  Everything is a pure function over tensors; scheduling, alerting, and
  persistence stay the caller's call — or `MobiusSmarts` runs the whole
  stack for you on a schedule.

  All detectors accept plain lists as well as tensors and run on any Nx
  backend; `nx_eigen` is the recommended backend on Nerves targets
  (CPU SIMD, no XLA toolchain), and `Nx.BinaryBackend` works everywhere
  at RRD scale.
  """
end
