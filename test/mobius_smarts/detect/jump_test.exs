defmodule MobiusSmarts.Detect.JumpTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.Detect.Jump

  doctest Jump

  describe "conformance to SPC theory" do
    test "X-bar limits are grand_mean ± 3·sigma/sqrt(n)" do
      result =
        Jump.scan([10.0, 10.0], [1.0, 1.0], 25, baseline: {10.0, 2.0})

      assert_in_delta Nx.to_number(result.jump_ucl[0]), 10.0 + 3.0 * 2.0 / 5.0, 1.0e-9
      assert_in_delta Nx.to_number(result.jump_lcl[0]), 10.0 - 3.0 * 2.0 / 5.0, 1.0e-9
    end

    test "S-chart limits reproduce the published B5/B6 constants" do
      # Montgomery, Introduction to Statistical Quality Control:
      # for n = 5, B5 = 0 and B6 = 1.964 (sigma = 1).
      result = Jump.scan([0.0], [1.0], 5, baseline: {0.0, 1.0})

      assert_in_delta Nx.to_number(result.wobble_ucl[0]), 1.964, 0.02
      assert Nx.to_number(result.wobble_lcl[0]) == 0.0

      # For n = 10, B5 = 0.276 and B6 = 1.669.
      result10 = Jump.scan([0.0], [1.0], 10, baseline: {0.0, 1.0})

      assert_in_delta Nx.to_number(result10.wobble_ucl[0]), 1.669, 0.02
      assert_in_delta Nx.to_number(result10.wobble_lcl[0]), 0.276, 0.02
    end

    test "pooled sigma weights by degrees of freedom" do
      # s = 2 with 3 reports carries 2 dof; s = 7 with 1 report carries
      # none and must not contaminate the pool.
      %{sigma_reports: pooled} = Jump.baseline([0.0, 0.0], [2.0, 7.0], [3, 1])
      assert_in_delta pooled, 2.0, 1.0e-9
    end

    test "grand mean weights by report count" do
      %{target: grand} = Jump.baseline([10.0, 20.0], [1.0, 1.0], [30, 10])
      assert_in_delta grand, 12.5, 1.0e-9
    end

    test "sigma_avg is the sd of the window averages, a sqrt(n) apart from sigma_reports" do
      # 60 reports per window with per-report sd 1.0: the averages
      # wander with sd 1/sqrt(60). The two scales must come back
      # separately — conflating them was CRITIQUE.md finding 1.
      :rand.seed(:exsss, {61, 62, 63})
      sd_avg = 1.0 / :math.sqrt(60)
      avgs = Enum.map(1..400, fn _ -> 50.0 + sd_avg * :rand.normal() end)
      stds = Enum.map(1..400, fn _ -> 1.0 + 0.05 * :rand.normal() end)

      b = Jump.baseline(avgs, stds, 60)

      assert_in_delta b.sigma_reports, 1.0, 0.05
      assert_in_delta b.sigma_avg, sd_avg, 0.02
      assert_in_delta b.sigma_reports / b.sigma_avg, :math.sqrt(60), 1.0
    end

    test "scan accepts the baseline map directly" do
      b = %{target: 10.0, sigma_reports: 1.0, sigma_avg: 0.2}
      result = Jump.scan([10.0, 14.0], [1.0, 1.0], 25, baseline: b)

      assert Nx.to_flat_list(result.jumps) == [0, 1]
    end
  end

  describe "input validation" do
    test "an empty series raises an explanatory error, not a cryptic Nx one" do
      assert_raise ArgumentError, ~r/:empty/, fn -> Jump.scan([], [], 25) end
      assert_raise ArgumentError, ~r/:empty/, fn -> Jump.baseline([], [], 25) end
    end

    test "a window count below 1 raises instead of producing NaN/inf limits" do
      assert_raise ArgumentError, ~r/count/, fn ->
        Jump.scan([10.0, 10.0], [1.0, 1.0], [30, 0], baseline: {10.0, 1.0})
      end

      assert_raise ArgumentError, ~r/count/, fn ->
        Jump.scan([10.0, 10.0], [1.0, 1.0], [30, -2], baseline: {10.0, 1.0})
      end
    end

    test "a count of exactly 1 stays legal — excluded from the wobble chart, not rejected" do
      result = Jump.scan([10.0, 10.0], [1.0, 0.0], [30, 1], baseline: {10.0, 1.0})
      assert Nx.to_flat_list(result.wobbles) == [0, 0]
    end

    test "all-singleton windows raise the dedicated NoDispersionError" do
      assert_raise Jump.NoDispersionError, ~r/fewer than 2 reports/, fn ->
        Jump.baseline([1.0, 2.0], [0.0, 0.0], 1)
      end
    end
  end

  describe "detection behavior" do
    test "flags a dispersion blow-up that the mean chart misses" do
      # Mean dead stable; within-window std triples at window 3.
      averages = [10.0, 10.0, 10.0, 10.0, 10.0]
      std_devs = [1.0, 1.0, 1.0, 3.5, 3.5]

      result = Jump.scan(averages, std_devs, 30, baseline: {10.0, 1.0})

      assert Nx.to_flat_list(result.jumps) == [0, 0, 0, 0, 0]
      assert Nx.to_flat_list(result.wobbles) == [0, 0, 0, 1, 1]
    end

    test "singleton windows never flag on the S chart" do
      result = Jump.scan([10.0], [0.0], 1, baseline: {10.0, 1.0})
      assert Nx.to_flat_list(result.wobbles) == [0]
    end

    test "phase I self-estimation finds a moderate outlier window" do
      # The outlier must stay moderate: phase I estimates the grand mean
      # from all windows including the outlier, so a massive excursion
      # shifts the centerline enough to flag the healthy windows too —
      # which is textbook behavior, handled by iterating phase I.
      averages = [10.0, 10.1, 9.9, 10.0, 10.1, 9.9, 11.0]
      std_devs = List.duplicate(1.0, 7)

      result = Jump.scan(averages, std_devs, 50)

      assert Nx.to_flat_list(result.jumps) == [0, 0, 0, 0, 0, 0, 1]
    end
  end
end
