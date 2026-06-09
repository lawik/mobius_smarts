defmodule MobiusSmarts.AnalysisTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.{Analysis, Config}

  defp seeded(seed), do: :rand.seed(:exsss, {seed, seed * 7, seed * 13})
  defp noise(sigma), do: sigma * :rand.normal()

  # Hand-tuned thresholds in the detector-default range, so scenarios
  # don't depend on Calibrate internals.
  defp calib, do: %{arl: 1000.0, jump_limit: 3.5, ewma_l: 3.0, cusum_h: 6.0}
  defp config, do: Config.new!(watch: [])

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

  describe "gaps/2 and active_segment/2" do
    test "finds gaps against the series' own cadence and re-anchors after them" do
      ts = Enum.map(0..9, &(&1 * 60)) ++ Enum.map(0..9, &(3600 + &1 * 60))
      lists = %{ts: ts, avg: List.duplicate(1.0, 20), std: [], reports: []}
      lists = %{lists | std: List.duplicate(0.1, 20), reports: List.duplicate(5, 20)}

      {cadence, gaps} = Analysis.gaps(lists.ts, 3.0)

      assert cadence == 60
      assert gaps == [{540, 3600}]

      segment = Analysis.active_segment(lists, gaps)
      assert length(segment.ts) == 10
      assert hd(segment.ts) == 3600
    end

    test "a steady series has no gaps" do
      {cadence, gaps} = Analysis.gaps(Enum.map(0..99, &(&1 * 60)), 3.0)
      assert cadence == 60
      assert gaps == []
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
      assert {:error, :unsettled} =
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
      assert {:error, :insufficient} = Analysis.fit_baseline(lists, min_windows: 60, now: 0)
    end

    test "a perfectly constant series has no variance to calibrate against" do
      lists = windows(List.duplicate(42.0, 100), std: 0.0)
      assert {:error, :zero_variance} = Analysis.fit_baseline(lists, min_windows: 60, now: 0)
    end

    test "windows of single reports carry no dispersion: a learning state, not a crash" do
      seeded(14)
      values = Enum.map(1..100, fn _ -> 50.0 + noise(1.0) end)
      lists = windows(values, reports: 1, std: 0.0)

      assert {:error, :no_dispersion} = Analysis.fit_baseline(lists, min_windows: 60, now: 0)
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

    test "a jump in the last window is a :jumped condition; an old one is a :spiked observation" do
      seeded(24)
      values = Enum.map(1..100, fn _ -> 50.0 + noise(0.18) end)
      values = List.replace_at(values, 49, 58.0)
      values = List.replace_at(values, 99, 57.0)
      lists = windows(values, std: 1.0)

      candidates =
        Analysis.tick_candidates(lists, healthy_baseline(50.0, 0.2), calib(), config())

      assert jumped = Enum.find(candidates, &(&1.kind == :jumped))
      assert jumped.severity == :critical

      assert spiked = Enum.find(candidates, &(&1.kind == :spiked))
      assert spiked.class == :observation
      assert spiked.onset == Enum.at(lists.ts, 49)
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

    test "a collapsed spread (stuck sensor) wobbles with a bounded concern" do
      seeded(26)
      # Healthy spread throughout, then the last window goes perfectly
      # flat: below the wobble band's lower limit with std = 0.0.
      stds = List.duplicate(1.0, 119) ++ [0.0]
      values = Enum.map(stds, fn s -> 50.0 + noise(s / :math.sqrt(30)) end)

      lists = %{
        ts: Enum.map(0..119, &(1_700_000_000 + &1 * 60)),
        avg: values,
        std: stds,
        reports: List.duplicate(30, 120)
      }

      candidates =
        Analysis.tick_candidates(lists, healthy_baseline(50.0, 0.2), calib(), config())

      assert wobbling = Enum.find(candidates, &(&1.kind == :wobbling))
      assert wobbling.severity in [:warning, :critical]
      # Concern must stay comparable to other detectors' concerns — a
      # ratio against std would explode toward 1e12 here and poison the
      # Board's max-concern aggregation.
      assert wobbling.concern >= 1.0
      assert wobbling.concern < 100.0
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
