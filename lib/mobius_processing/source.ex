defmodule MobiusProcessing.Source do
  @moduledoc """
  Pulls Mobius history into `Arrow.RecordBatch` form.

  Two shapes:

  - **Long format** (`long/1`) — one row per Mobius sample. Columns:
    `timestamp :: Timestamp(:second)`, `name :: Utf8`, `type :: Utf8`,
    `value :: Float64`, `tags :: Utf8` (JSON-encoded map of the sample's tags).
    Easy to filter, slow to math over directly.

  - **Wide format** — one column per metric, aligned on a common time axis.
    Planned for v0.2.

  For tests and synthetic data, use `from_metrics/1` to convert an arbitrary
  list of `Mobius.metric()` maps into a record batch without needing a running
  Mobius instance.
  """

  alias Arrow.Array.{Float64, Timestamp, Utf8}
  alias Arrow.{Buffer, Field, RecordBatch, Schema, Type}

  @typedoc """
  Options for `long/1`. All are optional.

  - `:instance` — Mobius instance name (default `:mobius`).
  - `:from`, `:to` — UNIX timestamp range, in seconds. Forwarded to
    `Mobius.Scraper.all/2`.
  - `:metrics` — keep only rows whose `name` is in this list.
  - `:types` — keep only rows whose `type` is in this list.
  - `:tags` — keep only rows whose `tags` map matches the given map exactly.
  """
  @type opt ::
          {:instance, atom()}
          | {:from, integer()}
          | {:to, integer()}
          | {:metrics, [String.t()]}
          | {:types, [Mobius.metric_type()]}
          | {:tags, map()}

  @doc """
  Pulls Mobius history and emits a long-format `Arrow.RecordBatch`.

  Requires the Mobius instance to be running. Internally calls into
  the Mobius scraper. For converting an in-memory list of metrics
  (tests, derived data, replays) use `from_metrics/1`.
  """
  @spec long([opt()]) :: Arrow.RecordBatch.t()
  def long(opts \\ []) do
    instance = Keyword.get(opts, :instance, :mobius)
    scraper_opts = Keyword.take(opts, [:from, :to])

    instance
    |> Mobius.Scraper.all(scraper_opts)
    |> filter(opts)
    |> from_metrics()
  end

  @doc """
  Builds a long-format `Arrow.RecordBatch` from an already-fetched list of
  `Mobius.metric()` maps.

  Non-numeric values (for instance, `:summary` maps) are skipped — their
  shape doesn't ride the numeric tensor path.
  """
  @spec from_metrics([Mobius.metric()]) :: Arrow.RecordBatch.t()
  def from_metrics(metrics) when is_list(metrics) do
    rows =
      metrics
      |> Enum.flat_map(&numeric_rows/1)

    n = length(rows)

    {timestamps, names, types, values, tags_json} = unzip5(rows)

    columns = [
      timestamp_column(timestamps),
      utf8_column(names),
      utf8_column(types),
      float64_column(values),
      utf8_column(tags_json)
    ]

    %RecordBatch{schema: long_schema(), length: n, columns: columns}
  end

  @doc """
  The schema produced by `long/1` and `from_metrics/1`.
  """
  @spec long_schema() :: Arrow.Schema.t()
  def long_schema() do
    %Schema{
      fields: [
        %Field{
          name: "timestamp",
          type: %Type.Timestamp{unit: :second, timezone: nil},
          nullable: false
        },
        %Field{name: "name", type: %Type.Utf8{}, nullable: false},
        %Field{name: "type", type: %Type.Utf8{}, nullable: false},
        %Field{name: "value", type: %Type.FloatingPoint{precision: :double}, nullable: false},
        %Field{name: "tags", type: %Type.Utf8{}, nullable: false}
      ]
    }
  end

  ## ---------------------------------------------------------------------
  ## Filtering
  ## ---------------------------------------------------------------------

  defp filter(metrics, opts) do
    metrics
    |> maybe_filter(opts[:metrics], fn m, names -> m.name in names end)
    |> maybe_filter(opts[:types], fn m, types -> m.type in types end)
    |> maybe_filter(opts[:tags], fn m, tags -> m.tags == tags end)
  end

  defp maybe_filter(metrics, nil, _fun), do: metrics
  defp maybe_filter(metrics, criterion, fun), do: Enum.filter(metrics, &fun.(&1, criterion))

  ## ---------------------------------------------------------------------
  ## Row extraction
  ## ---------------------------------------------------------------------

  defp numeric_rows(%{value: v} = m) when is_number(v) do
    [
      {m.timestamp, m.name, Atom.to_string(m.type), v * 1.0, encode_tags(m.tags || %{})}
    ]
  end

  defp numeric_rows(_other), do: []

  defp encode_tags(tags) when is_map(tags), do: Jason.encode!(tags)

  ## ---------------------------------------------------------------------
  ## Column builders
  ## ---------------------------------------------------------------------

  defp timestamp_column(timestamps) do
    %Timestamp{
      unit: :second,
      timezone: nil,
      length: length(timestamps),
      null_count: 0,
      values: Buffer.pack_primitive(timestamps, :int64)
    }
  end

  defp float64_column(values) do
    %Float64{
      length: length(values),
      null_count: 0,
      values: Buffer.pack_primitive(values, :float64)
    }
  end

  defp utf8_column(strings) do
    lengths = Enum.map(strings, &byte_size/1)

    %Utf8{
      length: length(strings),
      null_count: 0,
      offsets: Buffer.pack_int32_offsets(lengths),
      values: IO.iodata_to_binary(strings)
    }
  end

  defp unzip5(rows) do
    {a, b, c, d, e} =
      Enum.reduce(rows, {[], [], [], [], []}, fn {ts, n, t, v, tg}, {as, bs, cs, ds, es} ->
        {[ts | as], [n | bs], [t | cs], [v | ds], [tg | es]}
      end)

    {Enum.reverse(a), Enum.reverse(b), Enum.reverse(c), Enum.reverse(d), Enum.reverse(e)}
  end
end
