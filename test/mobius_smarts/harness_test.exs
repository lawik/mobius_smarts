defmodule MobiusSmarts.HarnessTest do
  use ExUnit.Case, async: true

  # The synthetic-data replay harness (issue #1): datasets with known
  # ground truth, replayed through the real detection pipeline and a
  # real Board, asserting on the findings that actually surface.
  #
  # Several tests deliberately assert CURRENT, KNOWN-BAD behavior,
  # each labeled with its GitHub issue. When a fix lands, the test
  # fails and gets consciously flipped to assert the desired behavior
  # — that's the harness doing its job.

  alias MobiusSmarts.{Replay, Synthetic}

  defp conditions(findings), do: Enum.filter(findings, &(&1.class == :condition))
  defp kinds(findings), do: findings |> Enum.map(& &1.kind) |> Enum.uniq()

  describe "calibration (issue #12)" do
    test "ideal i.i.d. data: realized false alarms stay in the budget's regime" do
      # Three days of healthy, independent, stationary Gaussian data
      # at a {1, :day} budget: the ARL chain end to end should yield
      # on the order of 3 false alarms. Within 3x validates the math;
      # drift beyond that means a calibration bug, not bad luck —
      # the dataset is seeded and deterministic.
      windows =
        Synthetic.series(
          seed: 1,
          segments: [%{minutes: 3 * 24 * 60, level: 100.0, sigma: 6.0}]
        )

      result =
        Replay.run(windows,
          config: [false_alarm_every: {1, :day}],
          tick_every: {5, :minute}
        )

      false_alarms = conditions(result.raised)

      assert length(false_alarms) <= 9,
             "expected <= 9 condition raises (3x a {1, :day} budget over 3 days), " <>
               "got #{length(false_alarms)}: #{inspect(kinds(false_alarms))}"
    end

    test "autocorrelated data blows through the budget — the divergence issue #12 documents" do
      # The same budget on AR(1)-wandering data: the windows are no
      # longer independent, the ARL assumptions are void, and the
      # realized alarm rate exceeds the budget many times over. This
      # is the honesty measurement: the budget is exact on ideal
      # data, directional on real data.
      windows =
        Synthetic.series(
          seed: 2,
          wander: %{phi: 0.995, sigma: 1.0},
          segments: [%{minutes: 3 * 24 * 60, level: 100.0, sigma: 6.0}]
        )

      result =
        Replay.run(windows,
          config: [false_alarm_every: {1, :day}],
          tick_every: {5, :minute}
        )

      false_alarms = conditions(result.raised)

      assert length(false_alarms) > 9,
             "expected the wandering series to exceed 3x the budget, " <>
               "got #{length(false_alarms)}"
    end
  end

  describe "boot ramp (fixed: #2, #3, #5)" do
    # The 2026-06-11 on-device incident: memory ramps after boot, then
    # creeps, then settles. Fixed behavior: the trend gate (#3) defers
    # the fit until the plateau, the fit-horizon rule (#2) never
    # re-scores pre-fit history, so no phantom drift in either
    # direction (#5) — the ramp is learned around, not alarmed about.
    defp boot_ramp_result do
      windows =
        Synthetic.series(
          seed: 3,
          segments: [
            # Boot ramp: 79MB -> 90MB over 20 minutes...
            %{minutes: 20, from: 79_000.0, to: 90_000.0, sigma: 2_500.0},
            # ...creeping toward its working level...
            %{minutes: 60, from: 90_000.0, to: 93_000.0, sigma: 2_500.0},
            # ...then settled there.
            %{minutes: 120, level: 93_000.0, sigma: 2_500.0}
          ]
        )

      Replay.run(windows, config: [false_alarm_every: {1, :week}])
    end

    test "the baseline waits out the ramp and lands on the settled level (#3)" do
      result = boot_ramp_result()

      assert %{target: target, degenerate: degenerate} =
               Map.put_new(result.baseline || %{}, :degenerate, false)

      refute degenerate
      # Fitted on the plateau, not frozen mid-ramp.
      assert_in_delta target, 93_000.0, 700.0
    end

    test "no phantom drift in either direction (#2, #5)" do
      result = boot_ramp_result()

      phantom =
        Enum.filter(
          result.raised,
          &(&1.kind in [:drifting_down, :drifting_up, :shifted_up, :shifted_down])
        )

      assert phantom == [],
             "expected no drift/shift findings on a healthy boot, " <>
               "got #{inspect(kinds(phantom))}"
    end
  end

  describe "constant metrics (fixed: #6)" do
    test "a constant metric fits a degenerate baseline and a level step raises :departed" do
      # A perfectly constant metric (process_count-shaped): the sigma
      # charts have nothing to price, so a degenerate baseline arms
      # departure-only detection — and the one real event, the step to
      # a new constant, is caught.
      windows =
        Synthetic.series(
          seed: 4,
          segments: [
            %{minutes: 90, level: 100.0, distribution: :constant},
            %{minutes: 70, level: 113.0, distribution: :constant}
          ]
        )

      result = Replay.run(windows, config: [false_alarm_every: {1, :week}])

      assert result.baseline.degenerate
      assert result.baseline.target == 100.0
      assert [%{detection: :active, detectors: detectors}] = result.status.metrics
      assert :departure in detectors

      assert [departed | _rest] = Enum.filter(result.raised, &(&1.kind == :departed))
      # Onset is the first window of the new constant (segment two
      # begins at global window index 90; windows are end-stamped).
      assert departed.onset == 1_750_000_000 + 90 * 60
      assert departed.message =~ "left its constant"
    end
  end

  describe "baseline starvation on choppy metrics (#13)" do
    test "CURRENT BEHAVIOR #13: character changes faster than min_baseline_windows never fit" do
      # A binary-memory-shaped metric: the level steps every 20 minutes
      # as sessions come and go. The honest gates refuse every stretch
      # (changepoint :unsettled before 60 settled windows accrue), so
      # six hours pass with the chart stack never armed and only a
      # perpetually resetting countdown to show for it.
      windows =
        Synthetic.series(
          seed: 6,
          segments:
            for i <- 0..17 do
              %{minutes: 20, level: 100.0 + rem(i, 2) * 20.0, sigma: 1.0}
            end
        )

      result = Replay.run(windows, config: [false_alarm_every: {1, :week}])

      assert result.baseline == nil
      assert [%{detection: :learning, learning: %{reason: reason}}] = result.status.metrics
      assert reason in [:unsettled, :trending]
    end
  end

  describe "durable constant steps (#15)" do
    test "CURRENT BEHAVIOR #15: a permanent new constant keeps :departed active forever" do
      # The step is real and worth one alarm — but four hours later the
      # metric has been perfectly stable at its new value and the
      # finding is still being re-confirmed against the old constant.
      # Nothing ever accepts the new normal.
      windows =
        Synthetic.series(
          seed: 7,
          segments: [
            %{minutes: 90, level: 100.0, distribution: :constant},
            %{minutes: 240, level: 113.0, distribution: :constant}
          ]
        )

      result = Replay.run(windows, config: [false_alarm_every: {1, :week}])

      # Raised exactly once (re-confirmations update in place)...
      assert [_one] = Enum.filter(result.raised, &(&1.kind == :departed))
      # ...but still active after 4h of stability at the new constant,
      # against a baseline that still says 100.
      assert [%{kind: :departed, status: :active}] = result.findings
      assert result.baseline.target == 100.0
    end
  end

  describe "single-window excursions (fixed: #7)" do
    test "one bad window stays an observation; a sustained excursion raises :jumped" do
      # k-of-n persistence: a single window beyond the X-bar band (but
      # inside CUSUM/EWMA territory) surfaces as a :spiked observation
      # once history moves past it — never as a condition. Three
      # out-of-band windows in the trailing five do raise.
      steady =
        Synthetic.series(
          seed: 5,
          segments: [%{minutes: 120, level: 100.0, sigma: 1.0}]
        )

      last = List.last(steady)

      excursion = fn count ->
        for i <- 1..count do
          %{timestamp: last.timestamp + i * 60, average: 101.5, std_dev: 1.0, reports: 18}
        end
      end

      calm = fn count, offset ->
        for i <- 1..count do
          %{
            timestamp: last.timestamp + (offset + i) * 60,
            average: 100.0,
            std_dev: 1.0,
            reports: 18
          }
        end
      end

      blip = steady ++ excursion.(1) ++ calm.(5, 1)
      result = Replay.run(blip, config: [false_alarm_every: {1, :week}])

      refute Enum.find(result.raised, &(&1.kind == :jumped)),
             "a single excursion must not raise a condition " <>
               "(raised: #{inspect(kinds(result.raised))})"

      assert Enum.find(result.raised, &(&1.kind == :spiked))

      sustained = steady ++ excursion.(4)
      sustained_result = Replay.run(sustained, config: [false_alarm_every: {1, :week}])

      assert Enum.find(sustained_result.raised, &(&1.kind == :jumped)),
             "a sustained excursion must still raise " <>
               "(raised: #{inspect(kinds(sustained_result.raised))})"
    end
  end
end
