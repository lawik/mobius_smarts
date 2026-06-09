defmodule MobiusSmarts.Source do
  @moduledoc """
  Pulls Mobius data into Nx tensors for the detectors.

  Three shapes, matching what `MobiusSmarts.Detect` consumes:

  - `summary_series/3` — per-window `average`/`std_dev` series of a
    summary metric. The main detector input.
  - `series/4` — plain numeric series of any non-summary metric type.
  - `sketch/3` — a `Mobius.DDSketch` over a window, for
    `MobiusSmarts.Detect.Shape`.

  All Mobius reads go through `Mobius.Data`, the programmatic
  data-access API. The pure converters (`from_metrics/1`,
  `from_summary_windows/1`) are exposed for tests, replays, and
  already-fetched data.
  """

  @type series() :: %{timestamps: Nx.Tensor.t(), values: Nx.Tensor.t()}
  @type summary_series() :: %{
          timestamps: Nx.Tensor.t(),
          average: Nx.Tensor.t(),
          std_dev: Nx.Tensor.t(),
          reports: Nx.Tensor.t()
        }

  @doc """
  Pull a numeric metric series from Mobius.

  `type` is any non-summary `Mobius.metric_type()` (`:last_value`,
  `:counter`, ...). Window options (`:last`, `:from`/`:to`,
  `:mobius_instance`) are forwarded to `Mobius.Data.metrics/4`.

  Returns `%{timestamps: s64 tensor, values: f64 tensor}`, or `:empty`
  when the window holds no points or the Mobius instance is unavailable.
  """
  @spec series(Mobius.metric_name(), Mobius.metric_type(), map(), keyword()) ::
          series() | :empty
  def series(metric_name, type, tags \\ %{}, opts \\ []) do
    case Mobius.Data.metrics(metric_name, type, tags, opts) do
      {:ok, metrics} -> from_metrics(metrics)
      {:error, :unavailable} -> :empty
    end
  end

  @doc """
  Pull the per-window summary series of a summary metric from Mobius.

  Each point is one RRD window's summary delta: the average and
  standard deviation of the reports that landed in that window.
  Window options are forwarded to `Mobius.Data.summary_windows/3`.

  Returns `%{timestamps: s64, average: f64, std_dev: f64, reports: s64}`
  tensors, ascending in time, or `:empty` when the window holds no
  points or the Mobius instance is unavailable. `reports` is the
  per-window report count — the subgroup size
  `MobiusSmarts.Detect.Jump` needs for its limits and baseline
  pooling. Windows with no reports contribute no point — gaps show as
  missing timestamps, which is itself a health signal worth checking
  before running detectors.
  """
  @spec summary_series(Mobius.metric_name(), map(), keyword()) :: summary_series() | :empty
  def summary_series(metric_name, tags \\ %{}, opts \\ []) do
    case Mobius.Data.summary_windows(metric_name, tags, opts) do
      {:ok, windows} -> from_summary_windows(windows)
      {:error, :unavailable} -> :empty
    end
  end

  @doc """
  Reconstruct the DDSketch histogram of a metric over a window.

  Thin delegation to `Mobius.Data.histogram/3`; the metric must have
  been registered with `reporter_options: [histogram: ...]`. Feed a
  baseline-window and current-window pair to
  `MobiusSmarts.Detect.Shape.from_sketches/2`.
  """
  @spec sketch(Mobius.metric_name(), map(), keyword()) ::
          {:ok, Mobius.DDSketch.t()} | {:error, term()}
  def sketch(metric_name, tags \\ %{}, opts \\ []) do
    Mobius.Data.histogram(metric_name, tags, opts)
  end

  @doc """
  Convert a list of `Mobius.metric()` maps (as returned by
  `Mobius.Data.metrics/4`) into tensors.

  Non-numeric values (e.g. summary maps) are skipped — use
  `from_summary_windows/1` for those. Returns `:empty` when no numeric
  points remain (Nx has no zero-sized tensors); an empty window is a
  missingness signal worth handling explicitly anyway.

  ## Examples

      iex> metrics = [
      ...>   %{timestamp: 100, name: "cpu_temp", type: :last_value, value: 48.2, tags: %{}},
      ...>   %{timestamp: 101, name: "cpu_temp", type: :last_value, value: 48.7, tags: %{}}
      ...> ]
      iex> %{timestamps: ts, values: v} = MobiusSmarts.Source.from_metrics(metrics)
      iex> {Nx.to_flat_list(ts), Nx.to_flat_list(v)}
      {[100, 101], [48.2, 48.7]}

      iex> MobiusSmarts.Source.from_metrics([])
      :empty
  """
  @spec from_metrics([Mobius.metric()]) :: series() | :empty
  def from_metrics(metrics) when is_list(metrics) do
    rows =
      for %{timestamp: ts, value: v} <- metrics, is_number(v) do
        {ts, v * 1.0}
      end

    case rows do
      [] ->
        :empty

      rows ->
        %{
          timestamps: Nx.tensor(Enum.map(rows, &elem(&1, 0)), type: :s64),
          values: Nx.tensor(Enum.map(rows, &elem(&1, 1)), type: :f64)
        }
    end
  end

  @doc """
  Convert summary windows (`%{timestamp: ..., average: ..., std_dev:
  ..., reports: ...}`, as returned by `Mobius.Data.summary_windows/3`)
  into tensors.

  ## Examples

      iex> windows = [
      ...>   %{timestamp: 100, average: 10.0, std_dev: 1.0, reports: 60},
      ...>   %{timestamp: 160, average: 11.0, std_dev: 1.5, reports: 58}
      ...> ]
      iex> %{average: avg, reports: reports} = MobiusSmarts.Source.from_summary_windows(windows)
      iex> {Nx.to_flat_list(avg), Nx.to_flat_list(reports)}
      {[10.0, 11.0], [60, 58]}
  """
  @spec from_summary_windows([
          %{
            timestamp: integer(),
            average: number(),
            std_dev: number(),
            reports: non_neg_integer()
          }
        ]) :: summary_series() | :empty
  def from_summary_windows([]), do: :empty

  def from_summary_windows(windows) when is_list(windows) do
    %{
      timestamps: Nx.tensor(Enum.map(windows, & &1.timestamp), type: :s64),
      average: Nx.tensor(Enum.map(windows, &(&1.average * 1.0)), type: :f64),
      std_dev: Nx.tensor(Enum.map(windows, &(&1.std_dev * 1.0)), type: :f64),
      reports: Nx.tensor(Enum.map(windows, &Map.fetch!(&1, :reports)), type: :s64)
    }
  end
end
