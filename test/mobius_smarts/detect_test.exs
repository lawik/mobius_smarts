defmodule MobiusSmarts.DetectTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.Detect

  doctest Detect

  defp seeded_noise(count, seed) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})
    Enum.map(1..count, fn _ -> :rand.normal() end)
  end

  # x_t = phi * x_{t-1} + e_t, started at 0 with a burn-in so the
  # series is stationary by the time we keep it.
  defp seeded_ar1(count, phi, seed) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})

    1..(count + 100)
    |> Enum.map_reduce(0.0, fn _, prev ->
      x = phi * prev + :rand.normal()
      {x, x}
    end)
    |> elem(0)
    |> Enum.drop(100)
  end

  describe "lag1_autocorrelation/1" do
    test "matches a hand-computed value on a short fixed series" do
      # x = [2, 4, 6, 4, 2], x̄ = 18/5 = 3.6
      # deviations d = [-1.6, 0.4, 2.4, 0.4, -1.6]
      # denominator Σ d_t²
      #   = 2.56 + 0.16 + 5.76 + 0.16 + 2.56 = 11.2
      # numerator Σ_{t=1..4} d_t d_{t+1}
      #   = (-1.6)(0.4) + (0.4)(2.4) + (2.4)(0.4) + (0.4)(-1.6)
      #   = -0.64 + 0.96 + 0.96 - 0.64 = 0.64
      # r1 = 0.64 / 11.2 = 2/35 ≈ 0.0571428...
      assert_in_delta Detect.lag1_autocorrelation([2.0, 4.0, 6.0, 4.0, 2.0]),
                      2 / 35,
                      1.0e-12
    end

    test "is near zero for seeded white noise" do
      # For iid noise r1 has standard error ~ 1/sqrt(n) ≈ 0.045 at
      # n = 500; 0.15 is over three standard errors.
      r1 = Detect.lag1_autocorrelation(seeded_noise(500, 42))
      assert abs(r1) < 0.15
    end

    test "is high for a seeded AR(1) with phi = 0.9" do
      r1 = Detect.lag1_autocorrelation(seeded_ar1(500, 0.9, 7))
      assert r1 > 0.7
    end

    test "strict alternation gives the closed-form -(n-1)/n" do
      # For x = [1, -1, 1, -1, ...] with even n: x̄ = 0, every d_t² = 1
      # so the denominator is n, and every adjacent product is -1 so
      # the numerator is -(n-1). r1 = -(n-1)/n.
      alternation = fn n -> Stream.cycle([1.0, -1.0]) |> Enum.take(n) end

      assert_in_delta Detect.lag1_autocorrelation(alternation.(10)), -9 / 10, 1.0e-12
      assert_in_delta Detect.lag1_autocorrelation(alternation.(4)), -3 / 4, 1.0e-12
    end

    test "accepts lists and tensors alike" do
      values = [2.0, 4.0, 6.0, 4.0, 2.0]

      assert Detect.lag1_autocorrelation(values) ==
               Detect.lag1_autocorrelation(Nx.tensor(values))
    end

    test "fewer than 3 windows raises a domain error" do
      assert_raise ArgumentError, ~r/too short to estimate autocorrelation/, fn ->
        Detect.lag1_autocorrelation([1.0, 2.0])
      end
    end

    test "a constant series raises a domain error" do
      assert_raise ArgumentError, ~r/a constant series has no autocorrelation/, fn ->
        Detect.lag1_autocorrelation(List.duplicate(5.0, 10))
      end
    end
  end
end
