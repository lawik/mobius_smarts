defmodule MobiusSmarts.Detect.ChangepointTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.Detect.Changepoint

  doctest Changepoint

  describe "conformance to the segmentation cost model" do
    test "recovers a noiseless step exactly" do
      series = List.duplicate(5.0, 40) ++ List.duplicate(9.0, 25)
      assert Changepoint.detect(series) == [40]
    end

    test "recovers two change points through recursion" do
      series =
        List.duplicate(10.0, 30) ++ List.duplicate(20.0, 30) ++ List.duplicate(5.0, 30)

      assert Changepoint.detect(series) == [30, 60]
    end

    test "split index maximizes SSE reduction against brute force" do
      :rand.seed(:exsss, {31, 32, 33})

      series =
        Enum.map(1..25, fn _ -> 10.0 + 0.5 * :rand.normal() end) ++
          Enum.map(1..25, fn _ -> 13.0 + 0.5 * :rand.normal() end)

      [detected] = Changepoint.detect(series, min_size: 5)

      sse = fn segment ->
        n = length(segment)
        mean = Enum.sum(segment) / n
        Enum.reduce(segment, 0.0, fn x, acc -> acc + (x - mean) * (x - mean) end)
      end

      brute_force_best =
        Enum.min_by(5..45, fn tau ->
          {left, right} = Enum.split(series, tau)
          sse.(left) + sse.(right)
        end)

      assert detected == brute_force_best
    end
  end

  describe "detection behavior" do
    test "noisy step recovered within tolerance" do
      :rand.seed(:exsss, {41, 42, 43})

      series =
        Enum.map(1..60, fn _ -> 50.0 + 1.0 * :rand.normal() end) ++
          Enum.map(1..60, fn _ -> 54.0 + 1.0 * :rand.normal() end)

      assert [tau] = Changepoint.detect(series)
      assert tau in 57..63
    end

    test "pure noise yields no change points" do
      :rand.seed(:exsss, {51, 52, 53})
      series = Enum.map(1..100, fn _ -> 50.0 + 2.0 * :rand.normal() end)

      assert Changepoint.detect(series) == []
    end

    test "min_size keeps splits away from the edges" do
      # The true step at index 2 is inside the margin; the algorithm may
      # still report the best *allowed* split, but never one within
      # min_size of either edge.
      series = List.duplicate(0.0, 2) ++ List.duplicate(10.0, 38)
      result = Changepoint.detect(series, min_size: 10)

      refute 2 in result
      assert Enum.all?(result, &(&1 >= 10 and &1 <= 30))
    end

    test "penalty override raises the bar" do
      series = List.duplicate(10.0, 30) ++ List.duplicate(10.5, 30)

      assert Changepoint.detect(series) == [30]
      assert Changepoint.detect(series, penalty: 1.0e6) == []
    end

    test "max_changepoints keeps the strongest changes, not the earliest" do
      # A small 0.4-step at 40 followed by a huge 8-step at 80 — capping
      # to one must keep the 8-step change (CRITIQUE.md §7).
      :rand.seed(:exsss, {71, 72, 73})

      series =
        Enum.map(1..40, fn _ -> 10.0 + 0.1 * :rand.normal() end) ++
          Enum.map(1..40, fn _ -> 10.4 + 0.1 * :rand.normal() end) ++
          Enum.map(1..40, fn _ -> 18.4 + 0.1 * :rand.normal() end)

      all = Changepoint.detect(series)
      assert Enum.any?(all, &(&1 in 78..82))

      [kept] = Changepoint.detect(series, max_changepoints: 1)
      assert kept in 78..82
    end

    test "series shorter than two minimum segments is never split" do
      assert Changepoint.detect([1.0, 9.0, 1.0, 9.0], min_size: 5) == []
    end

    test "quantized but stable series does not shatter into false segments" do
      # 0.5-step quantization with most consecutive windows equal
      # collapses the MAD of diffs to zero; the sd-of-diffs fallback
      # must keep the penalty real (CRITIQUE.md §3).
      :rand.seed(:exsss, {81, 82, 83})

      series =
        Enum.map(1..200, fn _ ->
          Float.round((20.0 + 0.2 * :rand.normal()) * 2) / 2
        end)

      diffs = series |> Enum.chunk_every(2, 1, :discard) |> Enum.map(fn [a, b] -> b - a end)
      zero_share = Enum.count(diffs, &(&1 == 0.0)) / length(diffs)
      assert zero_share > 0.5

      assert Changepoint.detect(series) == []
    end

    test "tiny series return [] instead of raising" do
      assert Changepoint.detect([]) == []
      assert Changepoint.detect([5.0]) == []
    end
  end
end
