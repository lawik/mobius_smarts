defmodule MobiusProcessing.SourceTest do
  use ExUnit.Case, async: true

  alias Arrow.{Buffer, RecordBatch, Schema}
  alias MobiusProcessing.Source

  defp metric(name, type, value, ts, tags) do
    %{name: name, type: type, value: value, timestamp: ts, tags: tags}
  end

  describe "from_metrics/1" do
    test "builds a record batch with the long schema" do
      batch =
        Source.from_metrics([
          metric("cpu_temp", :last_value, 48.2, 1_700_000_000, %{}),
          metric("cpu_temp", :last_value, 48.7, 1_700_000_001, %{}),
          metric("mem_used", :last_value, 38.1, 1_700_000_000, %{host: "rpi"})
        ])

      assert %RecordBatch{schema: %Schema{} = schema, length: 3, columns: cols} = batch

      assert Enum.map(schema.fields, & &1.name) == [
               "timestamp",
               "name",
               "type",
               "value",
               "tags"
             ]

      assert length(cols) == 5
    end

    test "skips non-numeric values (summary maps)" do
      batch =
        Source.from_metrics([
          metric("foo", :last_value, 1.0, 1, %{}),
          metric("bar", :summary, %{average: 1.0, min: 0.0, max: 2.0}, 2, %{}),
          metric("baz", :last_value, 3.0, 3, %{})
        ])

      assert batch.length == 2
    end

    test "produces a Float64 value column whose bytes are exactly the inputs" do
      batch =
        Source.from_metrics([
          metric("x", :last_value, 1.0, 10, %{}),
          metric("x", :last_value, 2.0, 11, %{}),
          metric("x", :last_value, 3.5, 12, %{})
        ])

      value_col = Enum.at(batch.columns, 3)
      assert Buffer.unpack_primitive(value_col.values, :float64, 3) == [1.0, 2.0, 3.5]
    end

    test "encodes tags as a JSON string per row" do
      batch =
        Source.from_metrics([
          metric("x", :last_value, 1.0, 10, %{}),
          metric("x", :last_value, 2.0, 11, %{host: "a", iface: "eth0"})
        ])

      tags_col = Enum.at(batch.columns, 4)
      strings = Buffer.slice_variable(tags_col.offsets, tags_col.values, tags_col.length)
      [t0, t1] = strings
      assert t0 == "{}"
      assert {:ok, %{"host" => "a", "iface" => "eth0"}} = Jason.decode(t1)
    end

    test "accepts integer values and casts them to f64" do
      batch =
        Source.from_metrics([
          metric("vm.run_queue", :last_value, 4, 1, %{}),
          metric("vm.run_queue", :last_value, 7, 2, %{})
        ])

      value_col = Enum.at(batch.columns, 3)
      assert Buffer.unpack_primitive(value_col.values, :float64, 2) == [4.0, 7.0]
    end

    test "is empty for an empty input" do
      batch = Source.from_metrics([])
      assert batch.length == 0
      assert Enum.all?(batch.columns, &(&1.length == 0))
    end
  end
end
