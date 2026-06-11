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

  describe "boot ramp (issues #2, #3, #5)" do
    # The 2026-06-11 on-device incident, reproduced: memory ramps
    # after boot, the baseline fits mid-creep, and the tick rescans
    # pre-fit history against the new target.
    defp boot_ramp_result do
      windows =
        Synthetic.series(
          seed: 3,
          segments: [
            # Boot ramp: 79MB -> 90MB over 20 minutes...
            %{minutes: 20, from: 79_000.0, to: 90_000.0, sigma: 2_500.0},
            # ...then a plateau that keeps creeping upward slowly.
            %{minutes: 160, from: 90_000.0, to: 93_000.0, sigma: 2_500.0}
          ]
        )

      Replay.run(windows, config: [false_alarm_every: {1, :week}])
    end

    test "CURRENT BEHAVIOR #2: pre-fit history raises a phantom :drifting_down about the past" do
      result = boot_ramp_result()

      assert %{fitted_at: fitted_at} = result.baseline

      assert down = Enum.find(result.raised, &(&1.kind == :drifting_down)),
             "expected the phantom drifting_down (raised kinds: #{inspect(kinds(result.raised))})"

      # The smoking gun from the device: the "drift" began before the
      # baseline existed — the detector is re-reading already-adjudicated
      # ramp data against the later-fitted target.
      assert down.onset < fitted_at
    end

    test "CURRENT BEHAVIOR #3: the mid-creep target makes real slow growth alarm immediately" do
      result = boot_ramp_result()

      assert Enum.find(result.raised, &(&1.kind == :drifting_up)),
             "expected drifting_up shortly after the mid-ramp fit " <>
               "(raised kinds: #{inspect(kinds(result.raised))})"
    end

    test "CURRENT BEHAVIOR #5: both drift directions report against the same target" do
      result = boot_ramp_result()
      raised_kinds = kinds(result.raised)

      assert :drifting_down in raised_kinds and :drifting_up in raised_kinds,
             "expected the contradiction (raised kinds: #{inspect(raised_kinds)})"
    end
  end

  describe "constant metrics (issue #6)" do
    test "CURRENT BEHAVIOR #6: zero variance blocks detection and a level step goes unseen" do
      # A perfectly constant metric (process_count-shaped): the fit
      # correctly refuses (zero variance), but then the one real event
      # — a step to a new constant — passes completely undetected.
      windows =
        Synthetic.series(
          seed: 4,
          segments: [
            %{minutes: 90, level: 100.0, distribution: :constant},
            %{minutes: 70, level: 113.0, distribution: :constant}
          ]
        )

      result = Replay.run(windows, config: [false_alarm_every: {1, :week}])

      assert result.baseline == nil
      assert [%{detection: :blocked}] = result.status.metrics

      assert conditions(result.raised) == [],
             "the 100 -> 113 step was expected to pass unseen (current blind spot); " <>
               "raised: #{inspect(kinds(result.raised))}"
    end
  end

  describe "single-window excursions (issue #7)" do
    test "CURRENT BEHAVIOR #7: one bad window raises a :jumped condition with no persistence" do
      # Steady data, then exactly one window beyond the X-bar band but
      # inside CUSUM/EWMA territory: 25 years of shipped systems
      # require k-of-n confirmation before alarming; we raise on the
      # single excursion.
      steady =
        Synthetic.series(
          seed: 5,
          segments: [%{minutes: 120, level: 100.0, sigma: 1.0}]
        )

      last = List.last(steady)

      spike = %{
        timestamp: last.timestamp + 60,
        average: last.average + 1.5,
        std_dev: 1.0,
        reports: 18
      }

      result = Replay.run(steady ++ [spike], config: [false_alarm_every: {1, :week}])

      assert Enum.find(result.raised, &(&1.kind == :jumped)),
             "expected the single-window :jumped (raised: #{inspect(kinds(result.raised))})"

      refute Enum.find(result.raised, &(&1.kind in [:drifting_up, :shifted_up]))
    end
  end
end
