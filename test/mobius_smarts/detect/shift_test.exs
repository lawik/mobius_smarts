defmodule MobiusSmarts.Detect.ShiftTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.Detect.Shift

  doctest Shift

  describe "conformance to theory" do
    test "z follows the closed form for constant input" do
      # z_t - x = (1 - lambda)^t * (z_0 - x), with z_0 = target.
      lambda = 0.2
      target = 10.0
      x = 14.0

      result = Shift.chart(List.duplicate(x, 20), target: target, sigma: 1.0, lambda: lambda)

      for {z, t} <- Enum.with_index(Nx.to_flat_list(result.smoothed), 1) do
        expected = x + :math.pow(1.0 - lambda, t) * (target - x)
        assert_in_delta z, expected, 1.0e-9
      end
    end

    test "first-window control limit is exactly target ± L·sigma·lambda" do
      # Var(z_1) = sigma^2 * (lambda / (2-lambda)) * (1 - (1-lambda)^2)
      #          = sigma^2 * lambda^2
      result =
        Shift.chart([50.0, 50.0], target: 50.0, sigma: 2.0, lambda: 0.25, l: 3.0)

      assert_in_delta Nx.to_number(result.ucl[0]), 50.0 + 3.0 * 2.0 * 0.25, 1.0e-9
      assert_in_delta Nx.to_number(result.lcl[0]), 50.0 - 3.0 * 2.0 * 0.25, 1.0e-9
    end

    test "limits widen monotonically to the asymptote" do
      result = Shift.chart(List.duplicate(0.0, 200), target: 0.0, sigma: 1.0, lambda: 0.2, l: 3.0)
      ucl = Nx.to_flat_list(result.ucl)

      assert ucl == Enum.sort(ucl)

      asymptote = 3.0 * :math.sqrt(0.2 / 1.8)
      assert_in_delta List.last(ucl), asymptote, 1.0e-6
    end

    test "streaming step matches batch chart" do
      values = [50.0, 51.0, 49.5, 52.0, 55.0, 56.0, 57.0, 58.0]

      result = Shift.chart(values, target: 50.0, sigma: 1.0)

      {statuses, final} =
        Enum.map_reduce(values, Shift.new(target: 50.0, sigma: 1.0), fn x, state ->
          Shift.step(state, x)
        end)

      assert_in_delta final.smoothed, Nx.to_number(result.smoothed[-1]), 1.0e-12

      streaming_first =
        Enum.find_index(statuses, &(&1 in [:upper_violation, :lower_violation]))

      assert streaming_first == result.first_violation
    end
  end

  describe "detection behavior" do
    test "stays quiet on in-control noise" do
      # In-control ARL for EWMA(0.2, L=3) is ~500 windows; a 200-window
      # stretch alarms by chance for some draws, so the seed is part of
      # the test contract.
      :rand.seed(:exsss, {6, 7, 8})
      values = Enum.map(1..200, fn _ -> 30.0 + 1.5 * :rand.normal() end)

      result = Shift.chart(values, target: 30.0, sigma: 1.5)
      assert result.first_violation == nil
    end

    test "empty series raises a domain error, mirroring Source's :empty" do
      assert_raise ArgumentError, ~r/:empty/, fn ->
        Shift.chart([], target: 0.0, sigma: 1.0)
      end
    end

    test "catches a 1.5-sigma shift faster than a Shewhart band would" do
      # A 1.5-sigma shift never crosses a 3-sigma per-window band on
      # noiseless data, but the EWMA accumulates straight through it.
      values = List.duplicate(10.0, 20) ++ List.duplicate(11.5, 20)

      result = Shift.chart(values, target: 10.0, sigma: 1.0)

      assert result.first_violation != nil
      assert result.first_violation < 30
    end
  end
end
