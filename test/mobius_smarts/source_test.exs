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
end
