defmodule MobiusSmarts.Analysis do
  @moduledoc false
  # The pure middle layer of the runtime: window series in, finding
  # candidates out. Everything here is a plain function over lists and
  # maps so it tests without Mobius or processes; Watcher and Sweeper
  # are thin shells that fetch data and hand candidates to the Board.
  #
  # A "candidate" is a map with :kind, :detector, :class, :severity,
  # :concern, :onset, :evidence, :message — the Board merges candidates
  # into Finding lifecycles.

  alias MobiusSmarts.{Calibrate, Config}
  alias MobiusSmarts.Detect.{Changepoint, Drift, Jump, Novelty, Shape, Shift, Trend}

  # A condition this far past its alarm threshold escalates to critical.
  @critical_concern 1.5

  @type lists() :: %{ts: [integer()], avg: [float()], std: [float()], reports: [integer()]}
  @type candidate() :: map()

  ## Series plumbing

  @doc "Summary-series tensors (from `MobiusSmarts.Source`) as plain lists."
  @spec to_lists(map()) :: lists()
  def to_lists(%{timestamps: ts, average: avg, std_dev: std, reports: reports}) do
    %{
      ts: Nx.to_flat_list(ts),
      avg: Nx.to_flat_list(avg),
      std: Nx.to_flat_list(std),
      reports: Nx.to_flat_list(reports)
    }
  end

  @doc """
  Find reporting gaps: window-to-window steps longer than `gap_factor`
  times `cadence_s`, the configured window width in seconds (the
  cadence is stated, never inferred — see `Config` `:resolution`).
  Returns gaps as `{last_seen_ts, resumed_ts}` pairs.
  """
  @spec gaps([integer()], number(), pos_integer()) :: [{integer(), integer()}]
  def gaps(ts, gap_factor, cadence_s) do
    ts
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.filter(fn [a, b] -> b - a > gap_factor * cadence_s end)
    |> Enum.map(fn [a, b] -> {a, b} end)
  end

  @doc """
  The series' measured median window-to-window step in seconds, or
  `nil` under 3 windows. Diagnostic only — the runtime detects gaps
  against the configured `:resolution` and uses this to warn when the
  stored data disagrees with it.
  """
  @spec median_cadence([integer()]) :: number() | nil
  def median_cadence(ts) when length(ts) < 3, do: nil

  def median_cadence(ts) do
    ts |> Enum.chunk_every(2, 1, :discard) |> Enum.map(fn [a, b] -> b - a end) |> median()
  end

  @doc """
  The series after its last long gap. Detectors only run on this
  segment: a long gap can hide a legitimate level move (reboot, day
  offline), and letting CUSUM/EWMA bridge it would hallucinate
  continuity. Re-anchoring is the gap policy; the gap itself is
  reported as a `:reporting_gap` observation.
  """
  @spec active_segment(lists(), [{integer(), integer()}]) :: lists()
  def active_segment(lists, []), do: lists

  def active_segment(lists, gaps) do
    {_last_seen, resumed} = List.last(gaps)
    from = Enum.find_index(lists.ts, &(&1 >= resumed))

    %{
      ts: Enum.drop(lists.ts, from),
      avg: Enum.drop(lists.avg, from),
      std: Enum.drop(lists.std, from),
      reports: Enum.drop(lists.reports, from)
    }
  end

  ## Baseline lifecycle

  @doc """
  Fit a baseline from a candidate stretch, using the stack on itself
  so "was this stretch actually healthy?" is not the caller's opinion:

  1. Run `Changepoint` over the stretch and keep only the most recent
     homogeneous segment — never average across a regime change.
  2. Phase-I trim: scan that segment with its own estimated baseline
     and drop out-of-control windows from the pool, then refit.

  Returns `{:ok, baseline}` with `target`/`sigma_reports`/`sigma_avg`
  plus fit metadata, or `{:error, :insufficient | :unsettled |
  :zero_variance | :no_dispersion}` while the metric must keep
  learning.
  """
  @spec fit_baseline(lists(), keyword()) :: {:ok, map()} | {:error, atom()}
  def fit_baseline(lists, opts) do
    min_windows = Keyword.fetch!(opts, :min_windows)
    now = Keyword.fetch!(opts, :now)

    if length(lists.avg) < min_windows do
      {:error, :insufficient}
    else
      segment = settled_segment(lists)

      if length(segment.avg) < min_windows do
        {:error, :unsettled}
      else
        do_fit(segment, now)
      end
    end
  end

  defp settled_segment(lists) do
    case Changepoint.detect(lists.avg) do
      [] -> lists
      changepoints -> slice(lists, List.last(changepoints))
    end
  end

  defp do_fit(segment, now) do
    estimate = Jump.baseline(segment.avg, segment.std, segment.reports)
    scan = Jump.scan(segment.avg, segment.std, segment.reports, baseline: estimate)

    flagged =
      Enum.zip_with(
        Nx.to_flat_list(scan.jumps),
        Nx.to_flat_list(scan.wobbles),
        &(&1 == 1 or &2 == 1)
      )

    kept = reject_flagged(segment, flagged)
    # Keep the trim only when it leaves a usable pool.
    kept = if length(kept.avg) >= max(div(length(segment.avg), 2), 2), do: kept, else: segment

    baseline = Jump.baseline(kept.avg, kept.std, kept.reports)

    if baseline.sigma_avg > 0.0 do
      {:ok,
       Map.merge(baseline, %{
         fitted_at: now,
         windows: length(kept.avg),
         from: List.first(segment.ts),
         to: List.last(segment.ts)
       })}
    else
      {:error, :zero_variance}
    end
  rescue
    # Jump.baseline raises when no window carries dispersion information
    # — a learning state, not a bug. Anything else propagates.
    Jump.NoDispersionError -> {:error, :no_dispersion}
  end

  defp reject_flagged(lists, flagged) do
    keep = fn values ->
      values |> Enum.zip(flagged) |> Enum.reject(&elem(&1, 1)) |> Enum.map(&elem(&1, 0))
    end

    %{
      ts: keep.(lists.ts),
      avg: keep.(lists.avg),
      std: keep.(lists.std),
      reports: keep.(lists.reports)
    }
  end

  ## Tick candidates: Jump / wobble / Shift / Drift

  @doc """
  Every-tick candidates for one metric's active segment against its
  baseline: `:jumped`, `:spiked`, `:wobbling`, `:shifted_*`,
  `:drifting_*`.
  """
  @spec tick_candidates(lists(), map(), Calibrate.t(), Config.t()) :: [candidate()]
  def tick_candidates(%{avg: []}, _baseline, _calib, _config), do: []

  def tick_candidates(lists, baseline, calib, config) do
    jump_candidates(lists, baseline, calib) ++
      shift_candidates(lists, baseline, calib, config) ++
      drift_candidates(lists, baseline, calib, config)
  end

  defp jump_candidates(lists, baseline, calib) do
    result =
      Jump.scan(lists.avg, lists.std, lists.reports, baseline: baseline, limit: calib.jump_limit)

    jumps = Nx.to_flat_list(result.jumps)
    wobbles = Nx.to_flat_list(result.wobbles)
    jump_ucl = Nx.to_flat_list(result.jump_ucl)
    jump_lcl = Nx.to_flat_list(result.jump_lcl)
    wobble_ucl = Nx.to_flat_list(result.wobble_ucl)
    wobble_lcl = Nx.to_flat_list(result.wobble_lcl)
    last = length(lists.avg) - 1

    spikes =
      for {1, i} <- Enum.with_index(jumps), i < last do
        ts = Enum.at(lists.ts, i)
        value = Enum.at(lists.avg, i)

        %{
          kind: :spiked,
          detector: :jump,
          class: :observation,
          severity: :info,
          concern: 0.0,
          onset: ts,
          evidence: %{value: value, ucl: Enum.at(jump_ucl, i), lcl: Enum.at(jump_lcl, i)},
          message: "spiked to #{round2(value)} at #{fmt_ts(ts)}, then returned"
        }
      end

    jumped =
      if Enum.at(jumps, last) == 1 do
        value = Enum.at(lists.avg, last)
        ucl = Enum.at(jump_ucl, last)
        lcl = Enum.at(jump_lcl, last)
        half = (ucl - lcl) / 2.0
        concern = abs(value - baseline.target) / max(half, 1.0e-12)

        [
          %{
            kind: :jumped,
            detector: :jump,
            class: :condition,
            severity: :critical,
            concern: concern,
            onset: Enum.at(lists.ts, last),
            evidence: %{value: value, ucl: ucl, lcl: lcl, target: baseline.target},
            message:
              "at #{round2(value)}, outside its band (#{round2(lcl)}–#{round2(ucl)}) right now"
          }
        ]
      else
        []
      end

    wobbling =
      if Enum.at(wobbles, last) == 1 do
        std = Enum.at(lists.std, last)
        ucl = Enum.at(wobble_ucl, last)
        lcl = Enum.at(wobble_lcl, last)

        # Above the band: how many UCLs of spread (unbounded is fine —
        # std really can be arbitrarily large). Below the band: measure
        # the shortfall in band-half-widths, mirroring how :jumped and
        # :shifted_* scale by half the band. Exactly 1.0 at the lower
        # limit, growing linearly as the spread collapses, and bounded
        # (1 + lcl/band_half) even at std = 0 — a ratio against std
        # would explode toward 1e12 for a stuck sensor and poison the
        # Board's max-concern aggregation across detectors.
        if std > ucl do
          concern = std / max(ucl, 1.0e-12)

          [
            %{
              kind: :wobbling,
              detector: :jump,
              class: :condition,
              severity: severity_from(concern),
              concern: concern,
              onset: Enum.at(lists.ts, last),
              evidence: %{std_dev: std, ucl: ucl, lcl: lcl},
              message:
                "within-window spread at #{round2(std)}, above its band " <>
                  "(#{round2(lcl)}–#{round2(ucl)}) — erratic, a pre-failure signature"
            }
          ]
        else
          band_half = (ucl - lcl) / 2.0
          concern = 1.0 + (lcl - std) / max(band_half, 1.0e-12)

          # Only reachable when the baseline pool had spread in every
          # window (Jump disarms the lower limit otherwise), so a
          # collapse really is anomalous for this metric.
          [
            %{
              kind: :flatlined,
              detector: :jump,
              class: :condition,
              severity: severity_from(concern),
              concern: concern,
              onset: Enum.at(lists.ts, last),
              evidence: %{std_dev: std, ucl: ucl, lcl: lcl},
              message:
                "within-window spread collapsed to #{round2(std)}, below its healthy " <>
                  "floor #{round2(lcl)} — flat, stuck-signal signature"
            }
          ]
        end
      else
        []
      end

    spikes ++ jumped ++ wobbling
  end

  defp shift_candidates(lists, baseline, calib, config) do
    result =
      Shift.chart(lists.avg,
        baseline: baseline,
        lambda: config.ewma_lambda,
        l: calib.ewma_l
      )

    violations = Nx.to_flat_list(result.violations)
    last = length(lists.avg) - 1

    if Enum.at(violations, last) == 1 do
      smoothed = result.smoothed |> Nx.to_flat_list() |> Enum.at(last)
      ucl = result.ucl |> Nx.to_flat_list() |> Enum.at(last)
      lcl = result.lcl |> Nx.to_flat_list() |> Enum.at(last)
      half = (ucl - lcl) / 2.0
      concern = abs(smoothed - baseline.target) / max(half, 1.0e-12)
      up? = smoothed > baseline.target
      onset_index = run_start(violations, last)
      onset = Enum.at(lists.ts, onset_index)

      [
        %{
          kind: if(up?, do: :shifted_up, else: :shifted_down),
          detector: :shift,
          class: :condition,
          severity: severity_from(concern),
          concern: concern,
          onset: onset,
          evidence: %{smoothed: smoothed, target: baseline.target, ucl: ucl, lcl: lcl},
          message:
            "level has settled at ~#{round2(smoothed)} " <>
              "(target #{round2(baseline.target)}) since #{fmt_ts(onset)}"
        }
      ]
    else
      []
    end
  end

  defp drift_candidates(lists, baseline, calib, config) do
    result =
      Drift.scan(lists.avg,
        baseline: baseline,
        k: config.cusum_k,
        h: calib.cusum_h
      )

    upper = Nx.to_flat_list(result.upper)
    lower = Nx.to_flat_list(result.lower)

    drift_side(:drifting_up, List.last(upper), result.upper_onset, lists, calib) ++
      drift_side(:drifting_down, List.last(lower), result.lower_onset, lists, calib)
  end

  defp drift_side(kind, bucket, onset_index, lists, calib) do
    if bucket != nil and bucket >= calib.cusum_h do
      concern = bucket / calib.cusum_h
      onset = onset_index && Enum.at(lists.ts, onset_index)
      direction = if kind == :drifting_up, do: "up", else: "down"

      [
        %{
          kind: kind,
          detector: :drift,
          class: :condition,
          severity: severity_from(concern),
          concern: concern,
          onset: onset,
          evidence: %{bucket: bucket, h: calib.cusum_h},
          message:
            "drifting #{direction}#{if onset, do: " since ~#{fmt_ts(onset)}", else: ""} " <>
              "(evidence #{round2(bucket)} of #{round2(calib.cusum_h)} σ·windows)"
        }
      ]
    else
      []
    end
  end

  ## Missingness candidates

  @doc "Condition for a metric that has stopped reporting."
  @spec silent_candidate(integer() | nil, integer()) :: candidate()
  def silent_candidate(last_seen, now) do
    %{
      kind: :silent,
      detector: nil,
      class: :condition,
      severity: :warning,
      concern: 1.0,
      onset: last_seen,
      evidence: %{last_seen: last_seen},
      message:
        case last_seen do
          nil -> "no windows in the whole analysis window"
          ts -> "no windows since #{fmt_ts(ts)} (#{fmt_dur(now - ts)} ago)"
        end
    }
  end

  @doc "Observations for past reporting gaps."
  @spec gap_candidates([{integer(), integer()}]) :: [candidate()]
  def gap_candidates(gaps) do
    for {last_seen, resumed} <- gaps do
      %{
        kind: :reporting_gap,
        detector: nil,
        class: :observation,
        severity: :info,
        concern: 0.0,
        onset: last_seen,
        evidence: %{last_seen: last_seen, resumed: resumed},
        message: "went quiet for #{fmt_dur(resumed - last_seen)} at #{fmt_ts(last_seen)}"
      }
    end
  end

  ## Sweep candidates: Trend / Changepoint / Shape / Novelty

  @doc """
  `:approaching_limit` candidates for a metric with a configured
  ceiling or floor, gated twice so a slope is only projected when
  there is real evidence behind it: Mann–Kendall significance (the
  trend is statistically there) and span coverage (the fitted series
  spans at least half of `:trend_window` — an ETA extrapolated from a
  sliver of the window it claims to summarize is a guess, not a fit).
  """
  @spec trend_candidates(lists(), Config.Metric.t(), Config.t()) :: [candidate()]
  def trend_candidates(lists, metric, config) when length(lists.avg) >= 5 do
    span = List.last(lists.ts) - List.first(lists.ts)
    mk = if 2 * span >= Config.seconds(config.trend_window), do: Trend.mann_kendall(lists.avg)

    if mk == nil or mk.trend == :none do
      []
    else
      warn_s = Config.seconds(config.warn_horizon)
      critical_s = Config.seconds(config.critical_horizon)

      # The O(n²) Theil-Sen fit is paid once; each threshold only
      # projects the precomputed line.
      fit = Trend.theil_sen(lists.avg, lists.ts)
      slope = fit.slope
      last_ts = List.last(lists.ts)

      for threshold <- [metric.ceiling, metric.floor],
          threshold != nil,
          {:eta, eta_s} <- [Trend.eta_from_fit(fit, last_ts, threshold)],
          eta_s <= warn_s do
        %{
          kind: :approaching_limit,
          detector: :trend,
          class: :condition,
          severity: if(eta_s <= critical_s, do: :critical, else: :warning),
          concern: critical_s / max(eta_s, 1.0),
          onset: last_ts,
          evidence: %{
            eta_s: eta_s,
            threshold: threshold,
            slope_per_hour: slope * 3600.0,
            p: mk.p,
            current: List.last(lists.avg)
          },
          message:
            "headed for #{round2(threshold)} in ~#{fmt_dur(eta_s)} " <>
              "(now #{round2(List.last(lists.avg))}, #{round2(slope * 3600.0)}/hour, p=#{round_p(mk.p)})"
        }
      end
    end
  end

  def trend_candidates(_lists, _metric, _config), do: []

  @doc "`:regime_change` observations from a retrospective changepoint sweep."
  @spec changepoint_candidates(lists()) :: [candidate()]
  def changepoint_candidates(lists) do
    changepoints = Changepoint.detect(lists.avg)

    for tau <- changepoints do
      ts = Enum.at(lists.ts, tau)
      before_mean = lists.avg |> Enum.take(tau) |> mean()
      after_mean = lists.avg |> Enum.drop(tau) |> mean()

      %{
        kind: :regime_change,
        detector: :changepoint,
        class: :observation,
        severity: :info,
        concern: 0.0,
        onset: ts,
        evidence: %{before_mean: before_mean, after_mean: after_mean},
        message:
          "changed character at #{fmt_ts(ts)} " <>
            "(~#{round2(before_mean)} → ~#{round2(after_mean)})"
      }
    end
  end

  @doc """
  `:shape_drift` candidate from a baseline/current DDSketch pair.
  Severity uses PSI's citable conventions (0.1 watch, 0.25 act);
  the message speaks the metric's own units via Wasserstein.
  """
  @spec shape_candidates(Mobius.DDSketch.t(), Mobius.DDSketch.t()) :: [candidate()]
  def shape_candidates(baseline_sketch, current_sketch) do
    %{baseline: p, current: q, values: v} = Shape.from_sketches(baseline_sketch, current_sketch)

    psi = Shape.psi(p, q)

    if psi >= 0.1 do
      moved = Shape.moved_by(p, q, v)
      jsd = Shape.js_divergence(p, q)

      [
        %{
          kind: :shape_drift,
          detector: :shape,
          class: :condition,
          severity: if(psi >= 0.25, do: :critical, else: :warning),
          concern: psi / 0.25,
          onset: nil,
          evidence: %{psi: psi, js_divergence: jsd, moved_by: moved},
          message:
            "distribution shape moved by ~#{round2(moved)} (in the metric's own units); " <>
              "PSI #{round2(psi)}"
        }
      ]
    else
      []
    end
  end

  @doc """
  Fit the cross-metric novelty model on the rows where all watched
  metrics reported in the same window. Returns `{:ok, model_map}` or
  `{:error, :insufficient}`.
  """
  @spec fit_novelty([{term(), lists()}], number(), keyword()) ::
          {:ok, map()} | {:error, :insufficient}
  def fit_novelty(series_by_key, arl, opts \\ []) do
    keys = Enum.map(series_by_key, &elem(&1, 0))
    df = length(keys)
    min_rows = Keyword.get(opts, :min_rows, max(5 * df, df + 2))

    by_ts =
      Enum.map(series_by_key, fn {_key, lists} -> Map.new(Enum.zip(lists.ts, lists.avg)) end)

    common =
      by_ts
      |> Enum.map(&MapSet.new(Map.keys(&1)))
      |> Enum.reduce(&MapSet.intersection/2)
      |> Enum.sort()

    if length(common) < min_rows do
      {:error, :insufficient}
    else
      matrix = for ts <- common, do: Enum.map(by_ts, & &1[ts])

      {:ok,
       %{
         model: Novelty.fit(matrix),
         keys: keys,
         threshold: Calibrate.chi2_threshold(arl, df),
         rows: length(common)
       }}
    end
  end

  @doc """
  Score the latest cross-metric window vector against the fitted
  novelty model; a `:novel_behavior` candidate when it exceeds the
  calibrated threshold.
  """
  @spec novelty_candidates(map(), [float()]) :: [candidate()]
  def novelty_candidates(model_map, vector) do
    score = Novelty.score(model_map.model, vector)

    if score > model_map.threshold do
      concern = score / model_map.threshold

      [
        %{
          kind: :novel_behavior,
          detector: :novelty,
          class: :condition,
          severity: severity_from(concern),
          concern: concern,
          onset: nil,
          evidence: %{
            score: score,
            threshold: model_map.threshold,
            metrics: Enum.map(model_map.keys, &elem(&1, 0)),
            vector: vector
          },
          message:
            "metric combination is #{round2(score)} typical-variations from this " <>
              "device's habits (threshold #{round2(model_map.threshold)})"
        }
      ]
    else
      []
    end
  end

  ## Shared helpers

  defp severity_from(concern) when concern >= @critical_concern, do: :critical
  defp severity_from(_concern), do: :warning

  defp run_start(flags, last) do
    last..0//-1
    |> Enum.take_while(&(Enum.at(flags, &1) == 1))
    |> List.last()
  end

  defp slice(lists, from) do
    %{
      ts: Enum.drop(lists.ts, from),
      avg: Enum.drop(lists.avg, from),
      std: Enum.drop(lists.std, from),
      reports: Enum.drop(lists.reports, from)
    }
  end

  defp median(values) do
    sorted = Enum.sort(values)
    n = length(sorted)
    mid = div(n, 2)

    if rem(n, 2) == 1 do
      Enum.at(sorted, mid)
    else
      (Enum.at(sorted, mid - 1) + Enum.at(sorted, mid)) / 2
    end
  end

  defp mean([]), do: 0.0
  defp mean(values), do: Enum.sum(values) / length(values)

  defp round2(value) when is_float(value), do: Float.round(value, 2)
  defp round2(value), do: value

  defp round_p(p), do: Float.round(p, 4)

  defp fmt_ts(ts) do
    ts |> DateTime.from_unix!() |> Calendar.strftime("%Y-%m-%d %H:%MZ")
  end

  defp fmt_dur(s) when s < 0, do: fmt_dur(0)
  defp fmt_dur(s) when s < 90, do: "#{round(s)}s"
  defp fmt_dur(s) when s < 90 * 60, do: "#{round(s / 60)}m"
  defp fmt_dur(s) when s < 36 * 3600, do: "#{Float.round(s / 3600, 1)}h"
  defp fmt_dur(s), do: "#{Float.round(s / 86_400, 1)} days"
end
