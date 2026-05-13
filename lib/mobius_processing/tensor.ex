defmodule MobiusProcessing.Tensor do
  @moduledoc """
  Converts `Arrow.Array` columns to `Nx.Tensor` and back.

  Arrow primitive columns store their values as a little-endian packed byte
  buffer that matches Nx's in-memory layout one-to-one for every primitive
  type Nx knows about. That makes the conversion a single `Nx.from_binary/2`
  call — one memcpy at most, no element-wise pass.

  Variable-length columns (`Utf8`, `Binary`), nested columns (`List`,
  `Struct`, `Map`), and the all-null column don't ride the tensor path; the
  conversion functions raise on them with a clear message.

  Validity bitmaps are unpacked into a separate `{n}` u8 tensor with `1` for
  valid and `0` for null. When `null_count == 0` (the common case for
  embedded telemetry), the validity tensor is `nil`.
  """

  alias Arrow.Array
  alias Arrow.{Buffer, RecordBatch}

  @doc """
  Converts a primitive `Arrow.Array` to `{tensor, validity}`.

  `validity` is `nil` when the column has no nulls, or a `{n}` u8 tensor
  with `1` for valid / `0` for null. Mask-aware callers thread the mask
  through; everyone else discards it.

  Raises on non-primitive arrays (Utf8, Binary, List, Struct, Map,
  FixedSizeBinary, FixedSizeList, Decimal, Null).
  """
  @spec from_column(Arrow.Array.t()) :: {Nx.Tensor.t(), Nx.Tensor.t() | nil}
  def from_column(%Array.Int8{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:s, 8}, n), validity_to_tensor(v, n)}

  def from_column(%Array.Int16{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:s, 16}, n), validity_to_tensor(v, n)}

  def from_column(%Array.Int32{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:s, 32}, n), validity_to_tensor(v, n)}

  def from_column(%Array.Int64{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:s, 64}, n), validity_to_tensor(v, n)}

  def from_column(%Array.UInt8{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:u, 8}, n), validity_to_tensor(v, n)}

  def from_column(%Array.UInt16{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:u, 16}, n), validity_to_tensor(v, n)}

  def from_column(%Array.UInt32{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:u, 32}, n), validity_to_tensor(v, n)}

  def from_column(%Array.UInt64{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:u, 64}, n), validity_to_tensor(v, n)}

  def from_column(%Array.Float32{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:f, 32}, n), validity_to_tensor(v, n)}

  def from_column(%Array.Float64{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:f, 64}, n), validity_to_tensor(v, n)}

  # Date32 is days-since-epoch as s32; Date64 is ms-since-epoch as s64.
  # Timestamp is a fixed-unit s64. Surface them as integer tensors — the
  # caller knows the unit from the schema.
  def from_column(%Array.Date32{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:s, 32}, n), validity_to_tensor(v, n)}

  def from_column(%Array.Date64{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:s, 64}, n), validity_to_tensor(v, n)}

  def from_column(%Array.Timestamp{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:s, 64}, n), validity_to_tensor(v, n)}

  def from_column(%Array.Time32{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:s, 32}, n), validity_to_tensor(v, n)}

  def from_column(%Array.Time64{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:s, 64}, n), validity_to_tensor(v, n)}

  def from_column(%Array.Duration{values: bin, length: n, validity: v}),
    do: {primitive(bin, {:s, 64}, n), validity_to_tensor(v, n)}

  def from_column(%Array.Bool{} = arr) do
    # Bool values are a packed bitmap, not a u8 per slot. Unpack into a u8
    # tensor (1 = true, 0 = false). Not a memcpy — this one bit-unpacks.
    bits = Buffer.unpack_bool_values(arr.values, arr.length)
    tensor = Nx.tensor(bits, type: :u8)
    {tensor, validity_to_tensor(arr.validity, arr.length)}
  end

  def from_column(other) do
    raise ArgumentError,
          "cannot convert #{inspect(other.__struct__)} to an Nx tensor — " <>
            "only primitive (int/uint/float/bool/date/time/timestamp/duration) " <>
            "Arrow columns map to tensors. Variable-length, nested, and decimal " <>
            "columns don't ride the tensor path."
  end

  @doc """
  Stacks the named columns of a record batch into a single `{n_samples, n_metrics}`
  tensor.

  Each column is converted via `from_column/1`, cast to `f64`, and stacked
  along axis 1 in the order given by `:keep`. Validity is dropped — if you
  need null-aware processing, use `from_column/1` per column and thread the
  masks through yourself.
  """
  @spec batch_to_tensor(Arrow.RecordBatch.t(), keep: [String.t()]) :: Nx.Tensor.t()
  def batch_to_tensor(%RecordBatch{} = batch, opts) do
    keep = Keyword.fetch!(opts, :keep)

    tensors =
      Enum.map(keep, fn name ->
        col = column!(batch, name)
        {tensor, _validity} = from_column(col)
        Nx.as_type(tensor, {:f, 64})
      end)

    Nx.stack(tensors, axis: 1)
  end

  ## ---------------------------------------------------------------------
  ## Internals
  ## ---------------------------------------------------------------------

  defp primitive(bin, type, _n), do: Nx.from_binary(bin, type)

  defp validity_to_tensor(nil, _n), do: nil

  defp validity_to_tensor(bitmap, n) do
    bits = Buffer.unpack_validity(bitmap, n)
    Nx.tensor(bits, type: :u8)
  end

  defp column!(%RecordBatch{schema: schema, columns: columns}, name) do
    idx =
      Enum.find_index(schema.fields, fn f -> f.name == name end) ||
        raise ArgumentError,
              "no column named #{inspect(name)} in batch; have " <>
                inspect(Enum.map(schema.fields, & &1.name))

    Enum.at(columns, idx)
  end
end
