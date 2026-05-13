defmodule MobiusProcessing.TensorTest do
  use ExUnit.Case, async: true

  alias Arrow.{Array, Buffer}
  alias MobiusProcessing.{Source, Tensor}

  defp primitive_array(mod, values, kind, extras \\ %{}) do
    struct(
      mod,
      Map.merge(
        %{
          length: length(values),
          null_count: 0,
          values: Buffer.pack_primitive(values, kind)
        },
        extras
      )
    )
  end

  describe "from_column/1 on primitive arrays" do
    test "Int64 round-trips" do
      arr = primitive_array(Array.Int64, [-3, 0, 7], :int64)
      {tensor, validity} = Tensor.from_column(arr)

      assert validity == nil
      assert Nx.shape(tensor) == {3}
      assert Nx.type(tensor) == {:s, 64}
      assert Nx.to_flat_list(tensor) == [-3, 0, 7]
    end

    test "Int32 / Int16 / Int8 round-trip" do
      for {mod, kind, type} <- [
            {Array.Int32, :int32, {:s, 32}},
            {Array.Int16, :int16, {:s, 16}},
            {Array.Int8, :int8, {:s, 8}}
          ] do
        arr = primitive_array(mod, [-1, 0, 1, 2], kind)
        {tensor, _} = Tensor.from_column(arr)
        assert Nx.type(tensor) == type
        assert Nx.to_flat_list(tensor) == [-1, 0, 1, 2]
      end
    end

    test "UInt32 / UInt16 / UInt8 round-trip" do
      for {mod, kind, type} <- [
            {Array.UInt32, :uint32, {:u, 32}},
            {Array.UInt16, :uint16, {:u, 16}},
            {Array.UInt8, :uint8, {:u, 8}}
          ] do
        arr = primitive_array(mod, [0, 1, 250], kind)
        {tensor, _} = Tensor.from_column(arr)
        assert Nx.type(tensor) == type
        assert Nx.to_flat_list(tensor) == [0, 1, 250]
      end
    end

    test "Float32 / Float64 round-trip" do
      f32 = primitive_array(Array.Float32, [1.5, -2.25, 0.0], :float32)
      {t32, _} = Tensor.from_column(f32)
      assert Nx.type(t32) == {:f, 32}
      assert Nx.to_flat_list(t32) == [1.5, -2.25, 0.0]

      f64 = primitive_array(Array.Float64, [1.5, -2.25, 1.0e100], :float64)
      {t64, _} = Tensor.from_column(f64)
      assert Nx.type(t64) == {:f, 64}
      assert Nx.to_flat_list(t64) == [1.5, -2.25, 1.0e100]
    end

    test "Timestamp comes back as s64 integers" do
      arr = %Array.Timestamp{
        unit: :second,
        timezone: nil,
        length: 3,
        null_count: 0,
        values: Buffer.pack_primitive([1_700_000_000, 1_700_000_001, 1_700_000_002], :int64)
      }

      {tensor, validity} = Tensor.from_column(arr)
      assert validity == nil
      assert Nx.type(tensor) == {:s, 64}
      assert Nx.to_flat_list(tensor) == [1_700_000_000, 1_700_000_001, 1_700_000_002]
    end

    test "Bool unpacks the value bitmap into a u8 tensor" do
      arr = %Array.Bool{
        length: 4,
        null_count: 0,
        values: Buffer.pack_bool_values([true, false, true, true])
      }

      {tensor, _} = Tensor.from_column(arr)
      assert Nx.type(tensor) == {:u, 8}
      assert Nx.to_flat_list(tensor) == [1, 0, 1, 1]
    end

    test "validity bitmap surfaces as a {n} u8 tensor when null_count > 0" do
      {bitmap, null_count} = Buffer.pack_validity([1, 0, 1, 1, 0])
      assert null_count == 2

      arr = %Array.Int32{
        length: 5,
        null_count: null_count,
        validity: bitmap,
        values: Buffer.pack_primitive([10, 0, 30, 40, 0], :int32)
      }

      {tensor, validity} = Tensor.from_column(arr)
      assert Nx.to_flat_list(tensor) == [10, 0, 30, 40, 0]
      assert validity != nil
      assert Nx.type(validity) == {:u, 8}
      assert Nx.to_flat_list(validity) == [1, 0, 1, 1, 0]
    end

    test "raises on Utf8" do
      arr = %Array.Utf8{length: 1, null_count: 0, offsets: <<0::32, 1::32>>, values: "x"}

      assert_raise ArgumentError, ~r/cannot convert.*Utf8/, fn ->
        Tensor.from_column(arr)
      end
    end

    test "raises on Null" do
      arr = %Array.Null{length: 5}

      assert_raise ArgumentError, ~r/cannot convert.*Null/, fn ->
        Tensor.from_column(arr)
      end
    end

    test "raises on empty columns — Nx has no zero-length tensor" do
      arr = %Array.Float64{length: 0, null_count: 0, values: <<>>}

      assert_raise ArgumentError, ~r/empty tensor/, fn ->
        Tensor.from_column(arr)
      end
    end
  end

  describe "batch_to_tensor/2" do
    test "stacks the named columns into a {n_samples, n_metrics} f64 tensor" do
      metrics =
        Enum.flat_map(0..4, fn i ->
          [
            %{name: "cpu", type: :last_value, value: 40.0 + i, timestamp: i, tags: %{}},
            %{name: "mem", type: :last_value, value: 10.0 * i, timestamp: i, tags: %{}}
          ]
        end)

      batch = Source.from_metrics(metrics)
      tensor = Tensor.batch_to_tensor(batch, keep: ["timestamp", "value"])

      assert Nx.shape(tensor) == {10, 2}
      assert Nx.type(tensor) == {:f, 64}

      timestamps = tensor |> Nx.slice_along_axis(0, 1, axis: 1) |> Nx.to_flat_list()
      assert timestamps == Enum.flat_map(0..4, fn i -> [i + 0.0, i + 0.0] end)
    end

    test "raises on an unknown column" do
      batch =
        Source.from_metrics([
          %{name: "x", type: :last_value, value: 1.0, timestamp: 1, tags: %{}}
        ])

      assert_raise ArgumentError, ~r/no column named/, fn ->
        Tensor.batch_to_tensor(batch, keep: ["nope"])
      end
    end
  end
end
