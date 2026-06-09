defmodule MobiusSmarts.ScenariosTest do
  @moduledoc """
  Big-picture detection scenarios: synthetic but realistic device
  histories run through the detector stack the way an on-device
  deployment would, asserting that the right detector fires, the wrong
  one stays quiet, and the diagnostic outputs (onsets, ETAs, change
  points) land near the truth planted in the data.

  All randomness is seeded — the scenarios are deterministic.
  """
  use ExUnit.Case, async: true

  alias Mobius.DDSketch
  alias MobiusSmarts.Detect.{Changepoint, Drift, Jump, Shape, Shift}
  alias MobiusSmarts.Detect.{Novelty, Trend}

  defp seeded(seed), do: :rand.seed(:exsss, {seed, seed * 7, seed * 13})
  defp noise(sigma), do: sigma * :rand.normal()

  describe "scenario: slow memory leak" do
    # 14 days of hourly windows. Healthy memory sits around 40% ± 1.
    # From day 7 a leak adds 0.05 percentage points per hour — one
    # twentieth of a sigma per window, invisible to any single-window
    # test. CUSUM runs at h = 6: in-control ARL grows exponentially in
    # h, and ~340 quiet windows at the default h = 5 (ARL ~940/side,
    # Siegmund) is a sizable false-alarm gamble — tuning h to the
    # monitoring horizon is part of deploying CUSUM, so the scenario
    # does it too.
    setup do
      seeded(1001)
      leak_start = 7 * 24

      averages =
        Enum.map(0..(14 * 24 - 1), fn hour ->
          leak = if hour >= leak_start, do: 0.05 * (hour - leak_start), else: 0.0
          40.0 + leak + noise(1.0)
        end)

      %{averages: averages, leak_start: leak_start}
    end

    test "CUSUM catches the leak and dates its onset", ctx do
      result = Drift.scan(ctx.averages, target: 40.0, sigma: 1.0, h: 6.0)

      assert result.upper_alarm != nil
      # Detected within the first two days of leaking...
      assert result.upper_alarm < ctx.leak_start + 2 * 24
      # ...and the onset estimate points at where the leak began,
      # not where it was detected.
      assert_in_delta result.upper_onset, ctx.leak_start, 24
    end

    test "the per-window X-bar chart sleeps through the early leak", ctx do
      # Half a day into the leak the level has moved 0.6% — deep inside
      # the ±3 sigma band. An individuals chart (n = 1, known baseline)
      # sees nothing where CUSUM has already started accumulating.
      early = Enum.take(ctx.averages, ctx.leak_start + 12)

      result =
        Jump.scan(
          early,
          List.duplicate(0.0, length(early)),
          1,
          baseline: {40.0, 1.0}
        )

      violations = result.jumps |> Nx.sum() |> Nx.to_number()
      assert violations == 0
    end

    test "trend confirms, quantifies, and projects to exhaustion", ctx do
      last_week = Enum.take(ctx.averages, -168)
      timestamps = Enum.map(0..167, &(&1 * 3600))

      %{slope: slope} = Trend.theil_sen(last_week, timestamps)
      per_hour = slope * 3600

      assert_in_delta per_hour, 0.05, 0.02
      assert %{trend: :increasing} = Trend.mann_kendall(last_week)

      {:eta, seconds} = Trend.eta_to_threshold(last_week, timestamps, 95.0)
      hours_left = seconds / 3600

      # ~47% headroom at ~0.05%/hour: around 950 hours, give noise.
      assert hours_left > 500
      assert hours_left < 1500
    end
  end

  describe "scenario: sensor going erratic before failure" do
    # Mean rock steady; within-window dispersion triples partway
    # through. The mean-watching detectors must stay quiet — only the
    # S chart sees it.
    setup do
      seeded(2002)
      onset = 120
      n = 200
      reports_per_window = 60

      std_devs =
        Enum.map(0..(n - 1), fn w ->
          base = if w >= onset, do: 3.0, else: 1.0
          base + abs(noise(0.05))
        end)

      averages =
        Enum.map(std_devs, fn s -> 50.0 + noise(s / :math.sqrt(reports_per_window)) end)

      %{averages: averages, std_devs: std_devs, n: reports_per_window, onset: onset}
    end

    test "S chart flags the dispersion change, X-bar does not", ctx do
      healthy_avg = Enum.take(ctx.averages, 100)
      healthy_std = Enum.take(ctx.std_devs, 100)
      baseline = Jump.baseline(healthy_avg, healthy_std, ctx.n)

      result = Jump.scan(ctx.averages, ctx.std_devs, ctx.n, baseline: baseline)

      s_flags = Nx.to_flat_list(result.wobbles)
      mean_flags = Nx.to_flat_list(result.jumps)

      {s_before, s_after} = Enum.split(s_flags, ctx.onset)

      assert Enum.sum(s_before) == 0
      # The tripled sigma blows through the S limits in essentially
      # every post-onset window.
      assert Enum.sum(s_after) > 70
      # The window *means* inherit the inflated variance (sigma/sqrt(n)
      # tripled too), so the X-bar chart blips occasionally — but it has
      # no sustained, unambiguous signal. The S chart is the detector
      # here; the mean chart alone would read as flaky noise.
      assert Enum.sum(mean_flags) < 20
    end

    test "the level genuinely never moves — dispersion is the whole story", ctx do
      # The EWMA statistic tracks the level; even with the inflated
      # post-onset noise it never strays a practically meaningful
      # distance from 50. (Its variance-tuned bands do blip — variance
      # inflation widens every mean-based statistic — which is exactly
      # why the S chart, not a level chart, owns this failure mode.)
      result = Shift.chart(ctx.averages, target: 50.0, sigma: 0.13)

      z = Nx.to_flat_list(result.smoothed)
      assert Enum.all?(z, &(abs(&1 - 50.0) < 1.0))
    end
  end

  describe "scenario: thermal throttling turns a latency unimodal -> bimodal" do
    # Baseline: render times around 16 ms. Current: the device spends
    # 40% of its time throttled at ~45 ms. The histogram pair sees what
    # a mean would soft-pedal.
    setup do
      seeded(3003)

      baseline =
        Enum.reduce(1..2000, DDSketch.new(), fn _, sketch ->
          DDSketch.insert(sketch, 16.0 + noise(1.5))
        end)

      current =
        Enum.reduce(1..2000, DDSketch.new(), fn _, sketch ->
          value = if :rand.uniform() < 0.4, do: 45.0 + noise(3.0), else: 16.0 + noise(1.5)
          DDSketch.insert(sketch, value)
        end)

      %{baseline: baseline, current: current}
    end

    test "every distribution distance crosses its action threshold", ctx do
      %{baseline: p, current: q, values: v} =
        Shape.from_sketches(ctx.baseline, ctx.current)

      psi = Shape.psi(p, q)
      jsd = Shape.js_divergence(p, q)
      w = Shape.moved_by(p, q, v)

      # PSI: > 0.25 is the conventional "significant shift" line.
      assert psi > 0.25
      # JS: for disjoint modes with a 40% split the analytic value is
      # 0.5·[ln(1/0.8) + 0.6·ln(0.75) + 0.4·ln(2)] ≈ 0.164 nats.
      analytic_jsd = 0.5 * (:math.log(1 / 0.8) + 0.6 * :math.log(0.75) + 0.4 * :math.log(2.0))
      assert_in_delta jsd, analytic_jsd, 0.04
      # Wasserstein speaks milliseconds: 40% of mass moved ~29 ms,
      # so roughly 11-12 ms of expected displacement.
      assert_in_delta w, 0.4 * 29.0, 3.0
    end

    test "the quantiles tell the story the mean obscures", ctx do
      # The shift is plainly visible in p95 — and DDSketch already
      # carries it; no extra detector needed once the drift is flagged.
      p95_before = DDSketch.quantile(ctx.baseline, 0.95)
      p95_after = DDSketch.quantile(ctx.current, 0.95)

      assert p95_before < 20.0
      assert p95_after > 40.0
    end
  end

  describe "scenario: correlation break between CPU and network" do
    # History: network traffic tracks CPU tightly (the device's work is
    # network-driven). Failure mode: CPU pegged while the network sits
    # idle — each metric individually unremarkable, the *pair*
    # impossible.
    setup do
      seeded(4004)

      history =
        Enum.map(1..300, fn _ ->
          cpu = 25.0 + 10.0 * :rand.uniform()
          net = 2.0 * cpu + noise(2.0)
          [cpu, net]
        end)

      %{model: Novelty.fit(history)}
    end

    test "on-the-line stays close, off-the-line is flagged", ctx do
      # Busy-but-consistent window: high CPU, proportional traffic.
      consistent = Novelty.score(ctx.model, [34.0, 68.0])

      # Broken window: same CPU, traffic gone. Both values are inside
      # their marginal historical ranges.
      broken = Novelty.score(ctx.model, [34.0, 20.0])

      assert consistent < 3.0
      assert broken > 5.0
      assert broken > 4 * consistent
    end
  end

  describe "scenario: the documented pipeline, end to end" do
    # CRITIQUE.md §14: every other scenario feeds detectors the true
    # generating parameters. This one uses ONLY what the library's own
    # API hands back: synthetic Mobius summary windows → Source →
    # Jump.baseline → detectors, exactly as the README wires it.
    setup do
      seeded(6006)
      shift_at = 200
      sigma_within = 2.0

      windows =
        Enum.map(0..349, fn w ->
          reports = 40 + :rand.uniform(40)
          level = if w >= shift_at, do: 51.0, else: 50.0
          avg = level + noise(sigma_within / :math.sqrt(reports))
          std = sigma_within + noise(0.1)

          %{
            timestamp: 1_700_000_000 + w * 600,
            average: avg,
            std_dev: std,
            reports: reports
          }
        end)

      %{windows: windows, shift_at: shift_at}
    end

    test "Source → baseline → Drift detects with no hand-fed parameters", ctx do
      alias MobiusSmarts.Source

      %{average: avgs, std_dev: stds, reports: counts} =
        Source.from_summary_windows(ctx.windows)

      healthy = Enum.take(ctx.windows, ctx.shift_at)

      %{average: h_avgs, std_dev: h_stds, reports: h_counts} =
        Source.from_summary_windows(healthy)

      baseline = Jump.baseline(h_avgs, h_stds, h_counts)

      # The shift is ~1.0 in absolute terms — about 3.5 sigma_avg, but
      # only ~0.5 sigma_reports. Wired with sigma_avg (the documented
      # pipeline), Drift catches it promptly.
      result = Drift.scan(avgs, target: baseline.target, sigma: baseline.sigma_avg)

      assert result.upper_alarm != nil
      assert result.upper_alarm in ctx.shift_at..(ctx.shift_at + 15)
      assert_in_delta result.upper_onset, ctx.shift_at, 10

      # And Jump, fed the same baseline map, flags post-shift windows.
      jump = Jump.scan(avgs, stds, counts, baseline: baseline)
      {pre, post} = jump.jumps |> Nx.to_flat_list() |> Enum.split(ctx.shift_at)

      assert Enum.sum(pre) <= 3
      assert Enum.sum(post) > 50
    end
  end

  describe "scenario: retrospective deploy regression" do
    # Latency means hold at 20 ms for 100 windows; a deploy at window
    # 100 regresses them to 28 ms. The sweep dates the regression and
    # CUSUM agrees in retrospect.
    setup do
      seeded(5005)

      series =
        Enum.map(1..100, fn _ -> 20.0 + noise(1.2) end) ++
          Enum.map(1..80, fn _ -> 28.0 + noise(1.4) end)

      %{series: series}
    end

    test "changepoint sweep dates the regression", ctx do
      assert [tau] = Changepoint.detect(ctx.series)
      assert tau in 98..102
    end

    test "CUSUM onset agrees with the changepoint", ctx do
      result = Drift.scan(ctx.series, target: 20.0, sigma: 1.2)

      assert result.upper_alarm != nil
      assert_in_delta result.upper_onset, 100, 3
    end
  end
end
