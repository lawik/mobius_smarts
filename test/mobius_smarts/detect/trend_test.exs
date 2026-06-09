defmodule MobiusSmarts.Detect.TrendTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.Detect.Trend

  doctest Trend

  describe "Theil-Sen conformance" do
    test "median of pairwise slopes, hand-computed" do
      # values [0, 1, 5] at t = [0, 1, 2]:
      # slopes (1-0)/1 = 1, (5-0)/2 = 2.5, (5-1)/1 = 4 — median 2.5.
      %{slope: slope} = Trend.theil_sen([0.0, 1.0, 5.0])
      assert_in_delta slope, 2.5, 1.0e-9
    end

    test "exact on clean linear data, with intercept" do
      ts = Enum.map(0..20, &(&1 * 60))
      values = Enum.map(ts, &(3.0 + 0.5 * &1))

      %{slope: slope, intercept: intercept} = Trend.theil_sen(values, ts)

      assert_in_delta slope, 0.5, 1.0e-9
      assert_in_delta intercept, 3.0, 1.0e-9
    end

    test "duplicate timestamps raise instead of silently poisoning the median" do
      assert_raise ArgumentError, ~r/strictly increasing/, fn ->
        Trend.theil_sen([1.0, 2.0, 3.0], [0, 0, 1])
      end
    end

    test "survives 20% arbitrary outliers where least squares does not" do
      ts = Enum.to_list(0..19)
      clean = Enum.map(ts, &(10.0 + 2.0 * &1))
      corrupted = List.replace_at(clean, 4, 10_000.0) |> List.replace_at(15, -10_000.0)

      %{slope: robust} = Trend.theil_sen(corrupted, ts)
      assert_in_delta robust, 2.0, 0.1

      # Reference: ordinary least squares on the same data is destroyed.
      n = length(ts)
      mean_t = Enum.sum(ts) / n
      mean_v = Enum.sum(corrupted) / n

      ls_slope =
        Enum.zip(ts, corrupted)
        |> Enum.map(fn {t, v} -> (t - mean_t) * (v - mean_v) end)
        |> Enum.sum()
        |> Kernel./(Enum.map(ts, &((&1 - mean_t) ** 2)) |> Enum.sum())

      assert abs(ls_slope - 2.0) > 10.0
    end
  end

  describe "Mann-Kendall conformance" do
    test "S statistic hand-computed on a small case" do
      # [3, 1, 2]: sign(1-3) + sign(2-3) + sign(2-1) = -1 - 1 + 1 = -1.
      result = Trend.mann_kendall([3.0, 1.0, 2.0])
      assert result.s == -1
      assert result.trend == :none
    end

    test "perfect monotonic series maximizes S at n(n-1)/2" do
      n = 12
      result = Trend.mann_kendall(Enum.map(1..n, &(&1 * 1.0)))
      assert result.s == div(n * (n - 1), 2)
      assert result.trend == :increasing
      assert result.p < 0.001
    end

    test "null variance follows n(n-1)(2n+5)/18" do
      result = Trend.mann_kendall(Enum.map(1..10, &(&1 * 1.0)))
      assert_in_delta result.var_s, 10 * 9 * 25 / 18, 1.0e-9
    end

    test "no trend claimed on seeded noise" do
      :rand.seed(:exsss, {21, 22, 23})
      values = Enum.map(1..60, fn _ -> :rand.normal() end)

      assert %{trend: :none} = Trend.mann_kendall(values)
    end

    test "ties contribute zero to S" do
      result = Trend.mann_kendall([1.0, 1.0, 1.0, 1.0])
      assert result.s == 0
      assert result.trend == :none
    end
  end

  describe "eta_to_threshold/3" do
    test "projects a floor crossing for a draining metric" do
      ts = Enum.map(0..10, &(&1 * 3600))
      battery = Enum.map(0..10, &(100.0 - 2.0 * &1))

      {:eta, seconds} = Trend.eta_to_threshold(battery, ts, 20.0)
      assert_in_delta seconds / 3600, 30.0, 0.01
    end

    test "not approaching when trending away" do
      ts = Enum.map(0..10, &(&1 * 3600))
      values = Enum.map(0..10, &(50.0 - 1.0 * &1))

      assert Trend.eta_to_threshold(values, ts, 95.0) == :not_approaching
    end

    test "not approaching when flat" do
      ts = Enum.map(0..10, &(&1 * 3600))
      assert Trend.eta_to_threshold(List.duplicate(50.0, 11), ts, 95.0) == :not_approaching
    end
  end
end
