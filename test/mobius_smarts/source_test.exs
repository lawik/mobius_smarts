defmodule MobiusSmarts.SourceTest do
  use ExUnit.Case, async: true

  alias MobiusSmarts.Source

  doctest Source

  describe "from_metrics/1" do
    test "converts numeric metrics to aligned tensors" do
      metrics = [
        %{timestamp: 100, name: "t", type: :last_value, value: 1.5, tags: %{}},
        %{timestamp: 160, name: "t", type: :last_value, value: 2, tags: %{}},
        %{timestamp: 220, name: "t", type: :last_value, value: 3.5, tags: %{}}
      ]

      %{timestamps: ts, values: v} = Source.from_metrics(metrics)

      assert Nx.type(ts) == {:s, 64}
      assert Nx.type(v) == {:f, 64}
      assert Nx.to_flat_list(ts) == [100, 160, 220]
      assert Nx.to_flat_list(v) == [1.5, 2.0, 3.5]
    end

    test "skips non-numeric values" do
      metrics = [
        %{timestamp: 1, name: "a", type: :last_value, value: 1.0, tags: %{}},
        %{timestamp: 2, name: "b", type: :summary, value: %{average: 1.0}, tags: %{}},
        %{timestamp: 3, name: "c", type: :last_value, value: 3.0, tags: %{}}
      ]

      %{values: v} = Source.from_metrics(metrics)
      assert Nx.to_flat_list(v) == [1.0, 3.0]
    end

    test "empty input is an explicit :empty, not zero-sized tensors" do
      assert Source.from_metrics([]) == :empty
      assert Source.from_summary_windows([]) == :empty

      only_summaries = [
        %{timestamp: 2, name: "b", type: :summary, value: %{average: 1.0}, tags: %{}}
      ]

      assert Source.from_metrics(only_summaries) == :empty
    end
  end

  describe "from_summary_windows/1" do
    test "splits summary windows into average, std_dev, and reports tensors" do
      windows = [
        %{timestamp: 100, average: 10.0, std_dev: 1.0, reports: 60},
        %{timestamp: 160, average: 12.0, std_dev: 0.5, reports: 31},
        %{timestamp: 220, average: 11.0, std_dev: 2.0, reports: 60}
      ]

      %{timestamps: ts, average: avg, std_dev: std, reports: reports} =
        Source.from_summary_windows(windows)

      assert Nx.to_flat_list(ts) == [100, 160, 220]
      assert Nx.to_flat_list(avg) == [10.0, 12.0, 11.0]
      assert Nx.to_flat_list(std) == [1.0, 0.5, 2.0]
      assert Nx.to_flat_list(reports) == [60, 31, 60]
      assert Nx.type(reports) == {:s, 64}
    end

    test "integer summary values become floats" do
      windows = [%{timestamp: 1, average: 10, std_dev: 0, reports: 1}]

      %{average: avg, std_dev: std} = Source.from_summary_windows(windows)
      assert Nx.type(avg) == {:f, 64}
      assert Nx.type(std) == {:f, 64}
    end

    test "windows without reports raise — silently guessing n was CRITIQUE §2" do
      windows = [%{timestamp: 1, average: 10.0, std_dev: 1.0}]

      assert_raise KeyError, fn -> Source.from_summary_windows(windows) end
    end
  end

  describe "summary_series/3" do
    test "raises without a stated :resolution — guessing the RRD tier is gone" do
      # The check fires before any Mobius call, so no instance is needed.
      assert_raise ArgumentError, ~r/needs a :resolution/, fn ->
        Source.summary_series("cpu.temp", %{}, last: {2, :hour})
      end
    end
  end

  describe "resample_windows/2" do
    # Helper: the summary window Mobius would compute from these raw reports.
    defp window_of(timestamp, values) do
      n = length(values)
      sum = Enum.sum(values)
      sum_sqrd = values |> Enum.map(&(&1 * &1)) |> Enum.sum()
      avg = sum / n

      std =
        if n == 1,
          do: 0.0,
          else: :math.sqrt(max(0.0, (sum_sqrd - sum * sum / n) / (n - 1)))

      %{timestamp: timestamp, average: avg, std_dev: std, reports: n}
    end

    test "merging windows reproduces the statistics of the pooled raw reports" do
      a = [1.0, 2.0, 3.0]
      b = [5.0, 7.0]

      [merged] =
        Source.resample_windows(
          [window_of(110, a), window_of(120, b)],
          {1, :minute}
        )

      expected = window_of(120, a ++ b)

      assert merged.timestamp == 120
      assert merged.reports == 5
      assert_in_delta merged.average, expected.average, 1.0e-9
      assert_in_delta merged.std_dev, expected.std_dev, 1.0e-9
    end

    test "a stated resolution merges finer-cadence windows into its buckets" do
      # The Mobius.Data.summary_windows/3 shape for a query spanning more
      # than the RRD seconds archive: minute-cadence windows trailing into
      # second-cadence ones for the freshest stretch.
      minute_windows =
        for i <- 1..5, do: window_of(600 + i * 60, [10.0 + i, 12.0 + i])

      second_windows =
        for i <- 1..10, do: window_of(900 + i, [20.0])

      resampled = Source.resample_windows(minute_windows ++ second_windows, {1, :minute})

      assert Enum.map(resampled, & &1.timestamp) == [660, 720, 780, 840, 900, 910]
      # The minute windows pass through untouched...
      assert Enum.take(resampled, 5) == minute_windows
      # ...and the second-cadence tail merges into one trailing bucket.
      assert List.last(resampled) == window_of(910, List.duplicate(20.0, 10))
    end

    test "windows on a bucket boundary close that bucket, not the next" do
      windows = [window_of(120, [1.0]), window_of(121, [2.0])]

      assert [%{timestamp: 120}, %{timestamp: 121}] =
               Source.resample_windows(windows, {1, :minute})
    end

    test ":native and second-resolution are identity; sub-second raises" do
      windows = [window_of(1, [1.0]), window_of(2, [2.0]), window_of(3, [3.0])]

      assert Source.resample_windows(windows, :native) == windows
      assert Source.resample_windows(windows, {1, :second}) == windows

      assert_raise ArgumentError, ~r/at least \{1, :second\}/, fn ->
        Source.resample_windows(windows, 500)
      end
    end
  end
end
