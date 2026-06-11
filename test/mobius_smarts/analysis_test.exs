defmodule MobiusSmarts.AnalysisTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.{Analysis, Config}

  defp seeded(seed), do: :rand.seed(:exsss, {seed, seed * 7, seed * 13})
  defp noise(sigma), do: sigma * :rand.normal()

  # Hand-tuned thresholds in the detector-default range, so scenarios
  # don't depend on Calibrate internals.
  defp calib, do: %{arl: 1000.0, jump_limit: 3.5, ewma_l: 3.0, cusum_h: 6.0}

  defp config do
    Config.new!(watch: [], resolution: {1, :minute}, false_alarm_every: {1, :week})
  end

  defp windows(values, opts \\ []) do
    cadence = Keyword.get(opts, :cadence, 60)
    start = Keyword.get(opts, :start, 1_700_000_000)
    std = Keyword.get(opts, :std, 1.0)
    reports = Keyword.get(opts, :reports, 30)

    %{
      ts: Enum.map(0..(length(values) - 1), &(start + &1 * cadence)),
      avg: values,
      std: List.duplicate(std * 1.0, length(values)),
      reports: List.duplicate(reports, length(values))
    }
  end

  defp healthy_baseline(target, sigma_avg) do
    %{
      target: target,
      sigma_reports: sigma_avg * :math.sqrt(30),
      sigma_avg: sigma_avg,
      fitted_at: 1_700_000_000,
      windows: 100,
      from: 1_699_000_000,
      to: 1_700_000_000
    }
  end

  describe "gaps/3 and active_segment/2" do
    test "finds gaps against the stated cadence and re-anchors after them" do
      ts = Enum.map(0..9, &(&1 * 60)) ++ Enum.map(0..9, &(3600 + &1 * 60))
      lists = %{ts: ts, avg: List.duplicate(1.0, 20), std: [], reports: []}
      lists = %{lists | std: List.duplicate(0.1, 20), reports: List.duplicate(5, 20)}

      gaps = Analysis.gaps(lists.ts, 3.0, 60)

      assert gaps == [{540, 3600}]

      segment = Analysis.active_segment(lists, gaps)
      assert length(segment.ts) == 10
      assert hd(segment.ts) == 3600
    end

    test "a steady series has no gaps" do
      assert Analysis.gaps(Enum.map(0..99, &(&1 * 60)), 3.0, 60) == []
    end
  end

  describe "median_cadence/1" do
    test "measures the dominant step, undistorted by an isolated gap" do
      ts = Enum.map(0..9, &(&1 * 60)) ++ Enum.map(0..9, &(3600 + &1 * 60))
      assert Analysis.median_cadence(ts) == 60
    end

    test "refuses to guess from fewer than 3 timestamps" do
      assert Analysis.median_cadence([0, 60]) == nil
      assert Analysis.median_cadence([]) == nil
    end
  end

  describe "fit_baseline/2" do
    test "fits on healthy data" do
      seeded(11)
      values = Enum.map(1..100, fn _ -> 50.0 + noise(1.0) end)
      lists = windows(values)

      assert {:ok, baseline} =
               Analysis.fit_baseline(lists, min_windows: 60, now: 1_700_010_000)

      assert_in_delta baseline.target, 50.0, 0.5
      assert baseline.sigma_avg > 0.0
      assert baseline.windows > 50
    end

    test "refuses a stretch that ends in a fresh regime" do
      seeded(12)

      values =
        Enum.map(1..80, fn _ -> 50.0 + noise(1.0) end) ++
          Enum.map(1..30, fn _ -> 70.0 + noise(1.0) end)

      lists = windows(values)

      # The settled segment after the changepoint is only 30 windows.
      assert {:error, %{reason: :unsettled, windows: 30, needed: 60}} =
               Analysis.fit_baseline(lists, min_windows: 60, now: 1_700_010_000)
    end

    test "fits on the segment after an old regime change" do
      seeded(13)

      values =
        Enum.map(1..30, fn _ -> 20.0 + noise(1.0) end) ++
          Enum.map(1..80, fn _ -> 50.0 + noise(1.0) end)

      lists = windows(values)

      assert {:ok, baseline} =
               Analysis.fit_baseline(lists, min_windows: 60, now: 1_700_010_000)

      # Baselined on the *recent* regime, not the average of both.
      assert_in_delta baseline.target, 50.0, 1.0
    end

    test "too little data" do
      lists = windows(List.duplicate(1.0, 10))

      assert {:error, %{reason: :insufficient, windows: 10, needed: 60}} =
               Analysis.fit_baseline(lists, min_windows: 60, now: 0)
    end

    test "a perfectly constant series fits a degenerate baseline (#6)" do
      lists = windows(List.duplicate(42.0, 100), std: 0.0)

      assert {:ok, baseline} = Analysis.fit_baseline(lists, min_windows: 60, now: 0)
      assert baseline.degenerate
      assert baseline.target == 42.0
    end

    test "a smooth ramp is refused with :trending — never a mid-ramp target (#3)" do
      seeded(18)
      # Gentle enough that the changepoint check cannot slice it into
      # steps, monotonic enough that Mann-Kendall is unambiguous.
      values = Enum.map(0..179, fn w -> 50.0 + 0.001 * w + noise(0.25) end)
      lists = windows(values)

      assert {:error, %{reason: :trending, needed: 60}} =
               Analysis.fit_baseline(lists, min_windows: 60, now: 0)
    end

    test "windows of single reports carry no dispersion: a learning state, not a crash" do
      seeded(14)
      values = Enum.map(1..100, fn _ -> 50.0 + noise(1.0) end)
      lists = windows(values, reports: 1, std: 0.0)

      assert {:error, %{reason: :no_dispersion, windows: 100, needed: 60}} =
               Analysis.fit_baseline(lists, min_windows: 60, now: 0)
    end

    test "the no-dispersion rescue does not swallow unrelated ArgumentErrors" do
      seeded(15)
      values = Enum.map(1..100, fn _ -> 50.0 + noise(1.0) end)
      lists = windows(values)
      # A malformed window count is a programming error, not a learning
      # state — it must raise, not read as {:error, :no_dispersion}.
      lists = %{lists | reports: List.replace_at(lists.reports, 50, 0)}

      assert_raise ArgumentError, ~r/count/, fn ->
        Analysis.fit_baseline(lists, min_windows: 60, now: 0)
      end
    end
  end

  describe "tick_candidates/4" do
    test "healthy data raises nothing" do
      seeded(21)
      values = Enum.map(1..120, fn _ -> 50.0 + noise(0.2) end)
      lists = windows(values, std: 1.0)

      assert Analysis.tick_candidates(lists, healthy_baseline(50.0, 0.2), calib(), config()) ==
               []
    end

    test "a fresh sustained shift raises :shifted_up with a dated onset" do
      seeded(22)

      values =
        Enum.map(1..80, fn _ -> 50.0 + noise(0.2) end) ++
          Enum.map(1..40, fn _ -> 51.5 + noise(0.2) end)

      lists = windows(values, std: 1.0)

      candidates =
        Analysis.tick_candidates(lists, healthy_baseline(50.0, 0.2), calib(), config())

      assert shifted = Enum.find(candidates, &(&1.kind == :shifted_up))
      assert shifted.class == :condition
      # Onset points near where the shift began (window 80), not the end.
      onset_index = Enum.find_index(lists.ts, &(&1 == shifted.onset))
      assert onset_index in 75..95
    end

    test "a slow drift raises :drifting_up and dates its onset" do
      seeded(23)
      drift_start = 60

      values =
        Enum.map(0..119, fn w ->
          drift = if w >= drift_start, do: 0.05 * (w - drift_start), else: 0.0
          50.0 + drift + noise(0.2)
        end)

      lists = windows(values, std: 1.0)

      candidates =
        Analysis.tick_candidates(lists, healthy_baseline(50.0, 0.2), calib(), config())

      assert drifting = Enum.find(candidates, &(&1.kind == :drifting_up))
      assert drifting.concern >= 1.0
      onset_index = Enum.find_index(lists.ts, &(&1 == drifting.onset))
      assert_in_delta onset_index, drift_start, 15
    end

    test "a drift landing exactly on the alarm line carries a dated onset" do
      # calib's cusum_h is 6.0 and config's cusum_k is 0.5: windows at
      # exactly target + 1.5 sigma add exactly 1.0 sigma-window to the
      # CUSUM bucket each, so six of them land the bucket on h exactly.
      # The candidate gate is `bucket >= h`; the detector must alarm at
      # the same boundary or the candidate's onset comes back nil.
      values = List.duplicate(50.0, 30) ++ List.duplicate(51.5, 6)
      lists = windows(values, std: 1.0)

      candidates =
        Analysis.tick_candidates(lists, healthy_baseline(50.0, 1.0), calib(), config())

      assert drifting = Enum.find(candidates, &(&1.kind == :drifting_up))
      assert drifting.evidence.bucket == calib().cusum_h
      # Onset dates from the last window before the drift began.
      assert drifting.onset == Enum.at(lists.ts, 29)
    end

    test "a sustained excursion is :jumped; an old blip is a :spiked observation (k-of-n, #7)" do
      seeded(24)
      values = Enum.map(1..100, fn _ -> 50.0 + noise(0.18) end)
      values = List.replace_at(values, 49, 58.0)
      # Three of the trailing five windows beyond the band: enough for
      # the k-of-n persistence gate. A single excursion stays an
      # observation (see the dedicated test below).
      values =
        values
        |> List.replace_at(97, 57.0)
        |> List.replace_at(98, 57.0)
        |> List.replace_at(99, 57.0)

      lists = windows(values, std: 1.0)

      candidates =
        Analysis.tick_candidates(lists, healthy_baseline(50.0, 0.2), calib(), config())

      assert jumped = Enum.find(candidates, &(&1.kind == :jumped))
      assert jumped.severity == :critical

      assert spiked = Enum.find(candidates, &(&1.kind == :spiked))
      assert spiked.class == :observation
      assert spiked.onset == Enum.at(lists.ts, 49)
    end

    test "a single trailing excursion does not raise :jumped (k-of-n, #7)" do
      seeded(27)
      values = Enum.map(1..100, fn _ -> 50.0 + noise(0.18) end)
      values = List.replace_at(values, 99, 57.0)
      lists = windows(values, std: 1.0)

      candidates =
        Analysis.tick_candidates(lists, healthy_baseline(50.0, 0.2), calib(), config())

      refute Enum.find(candidates, &(&1.kind == :jumped))
    end

    test "rising within-window spread raises :wobbling while the mean stays quiet" do
      seeded(25)
      stds = List.duplicate(1.0, 100) ++ List.duplicate(4.0, 20)
      values = Enum.map(stds, fn s -> 50.0 + noise(s / :math.sqrt(30)) end)

      lists = %{
        ts: Enum.map(0..119, &(1_700_000_000 + &1 * 60)),
        avg: values,
        std: stds,
        reports: List.duplicate(30, 120)
      }

      candidates =
        Analysis.tick_candidates(lists, healthy_baseline(50.0, 0.2), calib(), config())

      assert Enum.find(candidates, &(&1.kind == :wobbling))
      refute Enum.find(candidates, &(&1.kind == :jumped))
    end

    test "a collapsed spread (stuck sensor) raises :flatlined with a bounded concern" do
      seeded(26)
      # Healthy spread throughout, then the last three windows go
      # perfectly flat: below the wobble band's lower limit with
      # std = 0.0, persistently enough for the k-of-n gate. The
      # baseline pool carried spread in every window, so the lower
      # limit is armed and the collapse really is anomalous.
      stds = List.duplicate(1.0, 117) ++ [0.0, 0.0, 0.0]
      values = Enum.map(stds, fn s -> 50.0 + noise(s / :math.sqrt(30)) end)

      lists = %{
        ts: Enum.map(0..119, &(1_700_000_000 + &1 * 60)),
        avg: values,
        std: stds,
        reports: List.duplicate(30, 120)
      }

      candidates =
        Analysis.tick_candidates(lists, healthy_baseline(50.0, 0.2), calib(), config())

      assert flatlined = Enum.find(candidates, &(&1.kind == :flatlined))
      assert flatlined.severity in [:warning, :critical]
      assert flatlined.message =~ "stuck-signal"
      # Concern must stay comparable to other detectors' concerns — a
      # ratio against std would explode toward 1e12 here and poison the
      # Board's max-concern aggregation.
      assert flatlined.concern >= 1.0
      assert flatlined.concern < 100.0
    end

    test "zero spread stays quiet for a metric whose baseline includes flat windows" do
      # The zero-inflated case (an idle run queue): healthy history is
      # mostly perfectly-flat windows with occasional bursts. The
      # textbook lower S-limit would sit above zero and alarm on every
      # idle window; the recorded sd_floor of 0.0 disarms it.
      stds = Enum.map(1..119, fn i -> if rem(i, 10) == 0, do: 0.8, else: 0.0 end) ++ [0.0]
      avgs = Enum.map(stds, fn s -> if s > 0.0, do: 0.1, else: 0.0 end)

      lists = %{
        ts: Enum.map(0..119, &(1_700_000_000 + &1 * 60)),
        avg: avgs,
        std: stds,
        reports: List.duplicate(30, 120)
      }

      baseline =
        MobiusSmarts.Detect.Jump.baseline(lists.avg, lists.std, lists.reports)

      assert baseline.sd_floor == 0.0

      candidates = Analysis.tick_candidates(lists, baseline, calib(), config())

      refute Enum.find(candidates, &(&1.kind == :flatlined))
    end

    test "both drift directions in one scan collapse into :baseline_stale (#5)" do
      # A long stretch below target then a fresh stretch above: both
      # CUSUM buckets exceed h at scan end. That is not two drifts —
      # the target no longer describes the series.
      values = List.duplicate(49.0, 100) ++ List.duplicate(51.0, 5)
      lists = windows(values, std: 1.0)

      candidates =
        Analysis.tick_candidates(lists, healthy_baseline(50.0, 0.2), calib(), config())

      assert [stale] = Enum.filter(candidates, &(&1.kind == :baseline_stale))
      assert stale.class == :observation
      assert stale.message =~ "both sides"
      refute Enum.any?(candidates, &(&1.kind in [:drifting_up, :drifting_down]))
    end
  end

  describe "severity_from/1 (#9)" do
    test "wide bands: warning to 3x the threshold, critical beyond" do
      # A finding at its threshold is worth watching, not paging. The
      # old 1.5x critical edge turned every steady incident into a
      # wall of criticals (21 on-device on 2026-06-10).
      assert Analysis.severity_from(1.0) == :warning
      assert Analysis.severity_from(2.9) == :warning
      assert Analysis.severity_from(3.0) == :critical
      assert Analysis.severity_from(57.9) == :critical
    end

    test "a confirmed band excursion is banded too, not hardcoded critical" do
      seeded(28)
      # Sustained but mild: just past the X-bar band. half-band is
      # 3.5 * 1.095 / sqrt(30) ~ 0.7, so 51.0 is ~1.43x — warning.
      values = List.duplicate(50.0, 100) ++ List.duplicate(51.0, 4)
      lists = windows(values, std: 1.0)

      candidates =
        Analysis.tick_candidates(lists, healthy_baseline(50.0, 0.2), calib(), config())

      assert jumped = Enum.find(candidates, &(&1.kind == :jumped))
      assert jumped.severity == :warning
    end
  end

  describe "departure_candidates/2 (#6)" do
    test "leaving the constant raises :departed after k-of-n; a single blip stays quiet" do
      baseline = %{target: 100.0, degenerate: true}

      blip = windows(List.duplicate(100.0, 39) ++ [113.0], std: 0.0)
      assert Analysis.departure_candidates(blip, baseline) == []

      stepped = windows(List.duplicate(100.0, 37) ++ List.duplicate(113.0, 3), std: 0.0)
      assert [departed] = Analysis.departure_candidates(stepped, baseline)
      assert departed.kind == :departed
      assert departed.class == :condition
      assert departed.onset == Enum.at(stepped.ts, 37)
      assert departed.message =~ "left its constant"
    end
  end

  describe "trend_candidates/3" do
    test "projects an ETA when trending at a configured ceiling" do
      seeded(31)
      # 1.0/hour toward 95, currently ~70: ~25h out, inside a 7-day horizon.
      values = Enum.map(0..47, &(46.0 + &1 * 0.5 + noise(0.1)))
      lists = windows(values, cadence: 1800)
      metric = %Config.Metric{name: "disk", ceiling: 95.0}

      assert [candidate] = Analysis.trend_candidates(lists, metric, config())
      assert candidate.kind == :approaching_limit
      assert candidate.severity == :warning
      assert_in_delta candidate.evidence.eta_s / 3600, 25.0, 5.0
    end

    test "imminent crossing is critical" do
      seeded(32)
      values = Enum.map(0..47, &(80.0 + &1 * 0.3 + noise(0.05)))
      lists = windows(values, cadence: 1800)
      metric = %Config.Metric{name: "disk", ceiling: 95.0}

      assert [candidate] = Analysis.trend_candidates(lists, metric, config())
      assert candidate.severity == :critical
    end

    test "a flat series projects nothing" do
      seeded(33)
      values = Enum.map(0..47, fn _ -> 50.0 + noise(1.0) end)
      lists = windows(values, cadence: 1800)
      metric = %Config.Metric{name: "disk", ceiling: 95.0}

      assert Analysis.trend_candidates(lists, metric, config()) == []
    end

    test "approaching a floor works the same way" do
      seeded(34)
      values = Enum.map(0..47, &(30.0 - &1 * 0.5 + noise(0.1)))
      lists = windows(values, cadence: 1800)
      metric = %Config.Metric{name: "battery", floor: 5.0}

      assert [candidate] = Analysis.trend_candidates(lists, metric, config())
      assert candidate.kind == :approaching_limit
    end

    test "a series spanning a sliver of the trend window projects nothing, however steep" do
      seeded(36)
      # The same significant climb as the ETA scenario, but the series
      # spans only ~47 minutes of the 24-hour trend window — an ETA
      # extrapolated from that sliver would be a guess, not a fit.
      values = Enum.map(0..47, &(46.0 + &1 * 0.5 + noise(0.1)))
      lists = windows(values, cadence: 60)
      metric = %Config.Metric{name: "disk", ceiling: 95.0}

      assert Analysis.trend_candidates(lists, metric, config()) == []
    end

    test "evidence matches an independent per-threshold Trend computation exactly" do
      # The candidate path computes the Theil-Sen fit once and reuses it
      # for every threshold; this pins that restructure to the values the
      # public per-call API produces.
      seeded(35)
      values = Enum.map(0..47, &(46.0 + &1 * 0.5 + noise(0.1)))
      lists = windows(values, cadence: 1800)
      metric = %Config.Metric{name: "disk", ceiling: 95.0}

      alias MobiusSmarts.Detect.Trend
      assert {:eta, eta_s} = Trend.eta_to_threshold(lists.avg, lists.ts, 95.0)
      %{slope: slope} = Trend.theil_sen(lists.avg, lists.ts)

      assert [candidate] = Analysis.trend_candidates(lists, metric, config())
      assert candidate.evidence.eta_s == eta_s
      assert candidate.evidence.slope_per_hour == slope * 3600.0
      assert candidate.evidence.threshold == 95.0
    end
  end

  describe "changepoint_candidates/1" do
    test "dates a regime change as an observation" do
      seeded(41)

      values =
        Enum.map(1..60, fn _ -> 20.0 + noise(1.0) end) ++
          Enum.map(1..60, fn _ -> 28.0 + noise(1.0) end)

      lists = windows(values)

      assert [candidate] = Analysis.changepoint_candidates(lists)
      assert candidate.kind == :regime_change
      assert candidate.class == :observation
      onset_index = Enum.find_index(lists.ts, &(&1 == candidate.onset))
      assert_in_delta onset_index, 60, 3
    end
  end

  describe "novelty" do
    test "fits on common timestamps and flags a correlation break" do
      seeded(51)
      ts = Enum.map(0..299, &(&1 * 60))
      cpu = Enum.map(1..300, fn _ -> 25.0 + 10.0 * :rand.uniform() end)
      net = Enum.map(cpu, fn c -> 2.0 * c + noise(2.0) end)

      series = [
        {{"cpu", %{}}, %{ts: ts, avg: cpu, std: [], reports: []}},
        {{"net", %{}}, %{ts: ts, avg: net, std: [], reports: []}}
      ]

      assert {:ok, model} = Analysis.fit_novelty(series, 1000.0)
      assert model.rows == 300

      # Consistent pair: quiet. Broken pair: flagged.
      assert Analysis.novelty_candidates(model, [34.0, 68.0]) == []
      assert [candidate] = Analysis.novelty_candidates(model, [34.0, 20.0])
      assert candidate.kind == :novel_behavior
      assert candidate.concern > 1.0
    end

    test "novelty concern is the square root of the score ratio (#10)" do
      # Mahalanobis distances explode along the model's tight
      # directions in a way band-ratio concerns do not (96x observed
      # on-device while every other detector topped out far lower);
      # the square root keeps crossings at 1.0 and ordering intact
      # while damping the blowout into the shared concern scale.
      series =
        for {key, base} <- [{"a", 1.0}, {"b", 2.0}] do
          {{key, %{}},
           %{
             ts: Enum.map(0..19, &(&1 * 60)),
             avg: Enum.map(0..19, &(base * (&1 + 1.0) + rem(&1, 3) * 0.1)),
             std: [],
             reports: []
           }}
        end

      assert {:ok, model} = Analysis.fit_novelty(series, 1000.0)
      assert [candidate] = Analysis.novelty_candidates(model, [34.0, 20.0])

      score = candidate.evidence.score
      assert_in_delta candidate.concern, :math.sqrt(score / model.threshold), 1.0e-9
    end

    test "refuses to fit on too few aligned rows" do
      series = [
        {{"a", %{}}, %{ts: [0, 60], avg: [1.0, 2.0], std: [], reports: []}},
        {{"b", %{}}, %{ts: [0, 60], avg: [2.0, 3.0], std: [], reports: []}}
      ]

      assert {:error, :insufficient} = Analysis.fit_novelty(series, 1000.0)
    end
  end

  describe "missingness" do
    test "silent and gap candidates carry their dates" do
      silent = Analysis.silent_candidate(1_700_000_000, 1_700_007_200)
      assert silent.kind == :silent
      assert silent.class == :condition
      assert silent.onset == 1_700_000_000
      assert silent.message =~ "2.0h ago"

      assert [gap] = Analysis.gap_candidates([{1_700_000_000, 1_700_003_600}])
      assert gap.kind == :reporting_gap
      assert gap.class == :observation
      assert gap.message =~ "went quiet for 60m"
    end
  end
end
