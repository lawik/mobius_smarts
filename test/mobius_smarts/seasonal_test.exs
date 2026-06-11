defmodule MobiusSmarts.SeasonalTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.Seasonal

  # A 4-slot cycle at minute resolution, values shaped like the slot
  # index so expectations are exact and assertable.
  defp cycle_lists(cycles, base \\ 0) do
    count = cycles * 4

    %{
      ts: Enum.map(0..(count - 1), &((base + &1) * 60)),
      avg: Enum.map(0..(count - 1), &(100.0 + 10.0 * rem(base + &1, 4)))
    }
  end

  test "learns per-slot expectations and produces centered residuals" do
    model =
      Seasonal.new({4, :minute}, {1, :minute})
      |> Seasonal.update(cycle_lists(3))

    assert Seasonal.ready?(model)
    assert {4, 4} = Seasonal.progress(model)

    # Residuals of the data the model learned are ~zero everywhere —
    # the cycle is fully explained.
    residuals = Seasonal.residuals(model, cycle_lists(1, 12))
    assert Enum.all?(residuals.avg, &(abs(&1) < 1.0e-9))

    # An in-envelope anomaly survives the transform untouched.
    anomalous = %{ts: [48 * 60], avg: [100.0 + 10.0 * rem(48, 4) - 7.0]}
    assert [residual] = Seasonal.residuals(model, anomalous).avg
    assert_in_delta residual, -7.0, 1.0e-9
  end

  test "re-presented windows are no-ops; the high-water mark advances" do
    lists = cycle_lists(3)

    model =
      Seasonal.new({4, :minute}, {1, :minute})
      |> Seasonal.update(lists)

    # The runtime re-scans trailing history every tick: feeding the
    # same windows again must not double-count visits.
    again = Seasonal.update(model, lists)
    assert again == model
  end

  test "not ready before ~three cycles; under-visited slots fall back gracefully" do
    model =
      Seasonal.new({4, :minute}, {1, :minute})
      |> Seasonal.update(cycle_lists(2))

    refute Seasonal.ready?(model)
    assert {0, 4} = Seasonal.progress(model)

    # Even unready, residuals are defined for every window (fallback
    # to the mean of ready slots — here none, so 0.0).
    assert length(Seasonal.residuals(model, cycle_lists(1)).avg) == 4
  end

  test "slot means fade rather than accumulate, so a changed season relearns" do
    # Ten cycles at one shape, then ten at a shifted shape: the
    # post-fade expectation tracks the new shape.
    model =
      Seasonal.new({4, :minute}, {1, :minute})
      |> Seasonal.update(cycle_lists(10))

    shifted = %{
      ts: Enum.map(40..119, &(&1 * 60)),
      avg: Enum.map(40..119, &(150.0 + 10.0 * rem(&1, 4)))
    }

    model = Seasonal.update(model, shifted)
    [residual] = Seasonal.residuals(model, %{ts: [120 * 60], avg: [150.0]}).avg

    # Slot 0's expectation has moved most of the way from 100 to 150.
    assert abs(residual) < 15.0
  end
end
