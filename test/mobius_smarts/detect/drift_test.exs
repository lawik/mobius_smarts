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

  describe "the :baseline option" do
    # The map shape returned by Jump.baseline/3, carrying both noise
    # scales — the detector must pick sigma_avg, never sigma_reports.
    @baseline %{target: 10.0, sigma_reports: 5.0, sigma_avg: 1.0}

    test "scan reads target and sigma_avg from the baseline map" do
      values = List.duplicate(10.0, 10) ++ List.duplicate(12.0, 10)

      explicit = Drift.scan(values, target: 10.0, sigma: 1.0)
      from_baseline = Drift.scan(values, baseline: @baseline)

      assert from_baseline.upper_alarm == explicit.upper_alarm
      assert from_baseline.upper_onset == explicit.upper_onset
      assert Nx.to_flat_list(from_baseline.upper) == Nx.to_flat_list(explicit.upper)
      assert Nx.to_flat_list(from_baseline.lower) == Nx.to_flat_list(explicit.lower)
    end

    test "new reads target and sigma_avg from the baseline map" do
      assert Drift.new(baseline: @baseline) == Drift.new(target: 10.0, sigma: 1.0)
    end

    test "explicit :target and :sigma win over the baseline" do
      values = List.duplicate(10.0, 10) ++ List.duplicate(12.0, 10)
      deaf = %{target: 0.0, sigma_reports: 1.0, sigma_avg: 100.0}

      overridden = Drift.scan(values, baseline: deaf, target: 10.0, sigma: 1.0)
      explicit = Drift.scan(values, target: 10.0, sigma: 1.0)

      assert overridden.upper_alarm == explicit.upper_alarm
      assert Nx.to_flat_list(overridden.upper) == Nx.to_flat_list(explicit.upper)

      assert Drift.new(baseline: deaf, target: 10.0, sigma: 1.0) ==
               Drift.new(target: 10.0, sigma: 1.0)
    end

    test "neither a baseline nor both explicit values raises pointedly" do
      assert_raise ArgumentError, ~r/sigma_avg/, fn -> Drift.scan([1.0, 2.0], []) end
      assert_raise ArgumentError, ~r/sigma_avg/, fn -> Drift.scan([1.0, 2.0], target: 10.0) end
      assert_raise ArgumentError, ~r/sigma_avg/, fn -> Drift.new(sigma: 1.0) end
    end

    test "a baseline without :sigma_avg raises instead of guessing a scale" do
      assert_raise ArgumentError, ~r/sigma_avg/, fn ->
        Drift.scan([1.0, 2.0], baseline: %{target: 10.0, sigma_reports: 5.0})
      end

      assert_raise ArgumentError, ~r/sigma_avg/, fn ->
        Drift.new(baseline: %{target: 10.0, sigma_reports: 5.0})
      end
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
end
