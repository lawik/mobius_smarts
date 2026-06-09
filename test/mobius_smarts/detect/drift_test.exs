defmodule MobiusSmarts.Detect.DriftTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.Detect.Drift

  doctest Drift

  # Direct transcription of Page's recursion, the definition the
  # vectorized identity must reproduce exactly.
  defp reference_cusum(values, target, sigma, k) do
    values
    |> Enum.map(&((&1 - target) / sigma))
    |> Enum.scan({0.0, 0.0}, fn y, {sp, sm} ->
      {max(0.0, sp + y - k), max(0.0, sm - y - k)}
    end)
  end

  defp seeded_noise(count, seed) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})
    Enum.map(1..count, fn _ -> :rand.normal() end)
  end

  describe "conformance to the recursive definition" do
    test "scan reproduces Page's recursion exactly on noisy data" do
      values = Enum.map(seeded_noise(200, 42), &(10.0 + 2.0 * &1))

      result = Drift.scan(values, target: 10.0, sigma: 2.0, k: 0.5, h: 5.0)
      reference = reference_cusum(values, 10.0, 2.0, 0.5)

      scan_pairs =
        Enum.zip(Nx.to_flat_list(result.upper), Nx.to_flat_list(result.lower))

      for {{got_p, got_m}, {ref_p, ref_m}} <- Enum.zip(scan_pairs, reference) do
        assert_in_delta got_p, ref_p, 1.0e-9
        assert_in_delta got_m, ref_m, 1.0e-9
      end
    end

    test "streaming step matches batch scan" do
      values = Enum.map(seeded_noise(100, 7), &(5.0 + &1)) ++ List.duplicate(7.5, 30)

      result = Drift.scan(values, target: 5.0, sigma: 1.0)

      {statuses, final} =
        Enum.map_reduce(values, Drift.new(target: 5.0, sigma: 1.0), fn x, state ->
          Drift.step(state, x)
        end)

      assert_in_delta final.upper, Nx.to_number(result.upper[-1]), 1.0e-9

      first_streaming_alarm = Enum.find_index(statuses, &(&1 == :upper_alarm))
      assert first_streaming_alarm == result.upper_alarm
      assert final.upper_onset == result.upper_onset
    end
  end

  describe "detection behavior" do
    test "in-control series does not alarm" do
      values = Enum.map(seeded_noise(300, 99), &(50.0 + 3.0 * &1))
      result = Drift.scan(values, target: 50.0, sigma: 3.0, h: 8.0)

      assert result.upper_alarm == nil
      assert result.lower_alarm == nil
    end

    test "a one-sigma upward shift is caught with textbook delay" do
      # With k = 0.5, h = 5 the expected detection delay for a 1-sigma
      # shift is ~10.4 windows (Siegmund's approximation).
      values =
        Enum.map(seeded_noise(100, 13), &(20.0 + &1)) ++
          Enum.map(seeded_noise(40, 14), &(21.0 + &1))

      result = Drift.scan(values, target: 20.0, sigma: 1.0)

      assert result.upper_alarm != nil
      delay = result.upper_alarm - 100
      assert delay in 3..25
      assert result.lower_alarm == nil
    end

    test "downward shifts hit the lower side" do
      values = List.duplicate(10.0, 20) ++ List.duplicate(8.0, 10)
      result = Drift.scan(values, target: 10.0, sigma: 1.0)

      assert result.lower_alarm != nil
      assert result.upper_alarm == nil
    end

    test "onset estimates the start of the shift, not the detection" do
      values = List.duplicate(10.0, 50) ++ List.duplicate(10.7, 50)
      result = Drift.scan(values, target: 10.0, sigma: 1.0)

      # 0.7-sigma shift with k = 0.5: slow accumulation, late alarm —
      # but the onset should still point near window 50.
      assert result.upper_alarm > 55
      assert result.upper_onset in 48..52
    end

    test "empty series raises a domain error, mirroring Source's :empty" do
      assert_raise ArgumentError, ~r/:empty/, fn ->
        Drift.scan([], target: 0.0, sigma: 1.0)
      end
    end

    test "alarms accept lists and tensors alike" do
      values = List.duplicate(0.0, 10) ++ List.duplicate(3.0, 10)

      from_list = Drift.scan(values, target: 0.0, sigma: 1.0)
      from_tensor = Drift.scan(Nx.tensor(values), target: 0.0, sigma: 1.0)

      assert from_list.upper_alarm == from_tensor.upper_alarm
    end
  end

  describe "alarm at exactly the threshold" do
    # With target 0, sigma 1, k 0.5, values of 1.5 add exactly 1.0 to
    # the bucket per window — after five windows the level is exactly
    # h = 5.0. Page's CUSUM alarms when the statistic *reaches* h, and
    # the Analysis layer raises a candidate at `bucket >= h`; the
    # detector must agree or those candidates surface with a nil onset.
    @exact_series List.duplicate(1.5, 5)

    test "scan alarms when the bucket exactly equals h" do
      result = Drift.scan(@exact_series, target: 0.0, sigma: 1.0, k: 0.5, h: 5.0)

      assert Nx.to_number(result.upper[-1]) == 5.0
      assert result.upper_alarm == 4
      assert result.upper_onset == 0
    end

    test "streaming step agrees with scan at exact equality" do
      {statuses, final} =
        Enum.map_reduce(@exact_series, Drift.new(target: 0.0, sigma: 1.0), fn x, state ->
          Drift.step(state, x)
        end)

      assert final.upper == 5.0
      assert Enum.find_index(statuses, &(&1 == :upper_alarm)) == 4
      assert final.upper_onset == 0
    end

    test "scan parity with the Analysis layer's >= candidate gate" do
      # Analysis.drift_side/5 raises :drifting_* at `bucket >= h`;
      # whenever it would, the detector must have registered the alarm,
      # so the candidate's onset is dated rather than nil.
      result = Drift.scan(@exact_series, target: 0.0, sigma: 1.0, k: 0.5, h: 5.0)
      bucket = Nx.to_number(result.upper[-1])

      assert bucket >= 5.0
      assert result.upper_alarm != nil
      assert result.upper_onset != nil
    end
  end
end
