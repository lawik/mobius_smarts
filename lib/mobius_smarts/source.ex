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

  alias MobiusSmarts.Config

  @type series() :: %{timestamps: Nx.Tensor.t(), values: Nx.Tensor.t()}
  @type summary_series() :: %{
          timestamps: Nx.Tensor.t(),
          average: Nx.Tensor.t(),
          std_dev: Nx.Tensor.t(),
          reports: Nx.Tensor.t()
        }

  @typedoc "One summary window, the `Mobius.Data.summary_windows/3` shape."
  @type window() :: %{
          timestamp: integer(),
          average: number(),
          std_dev: number(),
          reports: non_neg_integer()
        }

  @typedoc """
  Target cadence for `summary_series/3` and `resample_windows/2`: an
  explicit `t:MobiusSmarts.Config.duration/0` naming the Mobius RRD
  tier you mean to read at, or `:native` for Mobius's mixed-cadence
  windows untouched.
  """
  @type resolution() :: :native | Config.duration()

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

  ## Window resolution (required)

  Mobius's RRD stores snapshots at four cadences at once (seconds,
  minutes, hours, days — see `Mobius.RRD`), and
  `Mobius.Data.summary_windows/3` deltas consecutive snapshots across
  *all* of them: any query spanning more than the seconds archive
  comes back with second-cadence windows for the freshest couple of
  minutes, minute-cadence behind that, and so on. The detectors assume
  comparable windows — mixed cadences read as reporting gaps and
  incomparable subgroups — so this function requires you to state the
  cadence; it will not guess one from the data:

  - `resolution: duration` — a `{1, :minute}`-style bucket width (or
    raw milliseconds) naming the RRD tier you mean to read at. Pick
    one whose archive spans the window you query — narrower than the
    coarsest tier in range leaves gaps between the coarse windows.
  - `resolution: :native` — Mobius's mixed-cadence windows, untouched.

  Resampling merges sum/sum-of-squares/count deltas, so a merged
  bucket carries exactly the statistics Mobius would have computed
  between the bucket's endpoint snapshots (see `resample_windows/2`).

  A caveat on `std_dev`: Mobius currently computes it with a naive
  sum-of-squares accumulator, which cancels catastrophically when the
  values are large floats relative to their spread — memory in bytes is
  the classic case. The result is a std_dev that is noise-floor garbage
  without being obviously wrong (it never goes negative), and std_dev is
  the calibration keystone for the whole detector stack. For
  large-magnitude metrics, prefer scaling at the reporter — report
  percent rather than bytes — until Mobius computes summaries with a
  stable algorithm.
  """
  @spec summary_series(Mobius.metric_name(), map(), keyword()) :: summary_series() | :empty
  def summary_series(metric_name, tags \\ %{}, opts \\ []) do
    {resolution, opts} = Keyword.pop(opts, :resolution)

    if is_nil(resolution) do
      raise ArgumentError,
            "summary_series/3 needs a :resolution — Mobius stores several RRD tiers at " <>
              "once and this function will not guess which one you mean. Pass the tier " <>
              "to read at (e.g. resolution: {1, :minute}), or resolution: :native for " <>
              "Mobius's mixed-cadence windows untouched."
    end

    case Mobius.Data.summary_windows(metric_name, tags, opts) do
      {:ok, windows} -> windows |> resample_windows(resolution) |> from_summary_windows()
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

  Windows that came out of Mobius inherit the `std_dev` precision
  caveat on `summary_series/3`.

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

  @doc """
  Resample summary windows to a uniform cadence (see the resolution
  section on `summary_series/3`).

  Windows are grouped into end-aligned buckets of the target width and
  merged exactly: each window's `average`/`std_dev`/`reports` is
  unwound back to the sum, sum-of-squares, and count deltas it came
  from, those are added, and the bucket's statistics recomputed — the
  same numbers Mobius would report between the bucket's endpoint
  snapshots. A merged bucket keeps the timestamp of its latest member
  (its end, for complete buckets), so a trailing partial bucket never
  claims a timestamp from the future.

  ## Examples

      iex> windows = [
      ...>   %{timestamp: 100, average: 10.0, std_dev: 0.0, reports: 1},
      ...>   %{timestamp: 120, average: 14.0, std_dev: 0.0, reports: 1}
      ...> ]
      iex> [merged] = MobiusSmarts.Source.resample_windows(windows, {1, :minute})
      iex> {merged.timestamp, merged.average, merged.reports}
      {120, 12.0, 2}
  """
  @spec resample_windows([window()], resolution()) :: [window()]
  def resample_windows(windows, :native), do: windows

  def resample_windows(windows, resolution) do
    case Config.seconds(resolution) do
      0 ->
        raise ArgumentError,
              "resolution must be at least {1, :second}, got: #{inspect(resolution)} " <>
                "(raw integers are milliseconds)"

      1 ->
        windows

      width ->
        windows
        |> Enum.group_by(fn window -> div(window.timestamp + width - 1, width) * width end)
        |> Enum.sort_by(&elem(&1, 0))
        |> Enum.map(fn {_bucket_end, members} -> merge_windows(members) end)
    end
  end

  defp merge_windows([window]), do: window

  defp merge_windows(members) do
    reports = members |> Enum.map(& &1.reports) |> Enum.sum()
    sum = members |> Enum.map(&(&1.average * &1.reports)) |> Enum.sum()

    sum_sqrd =
      members
      |> Enum.map(fn w ->
        partial = w.average * w.reports
        w.std_dev * w.std_dev * (w.reports - 1) + partial * partial / w.reports
      end)
      |> Enum.sum()

    %{
      timestamp: members |> Enum.map(& &1.timestamp) |> Enum.max(),
      average: sum / reports,
      std_dev: merged_std_dev(sum, sum_sqrd, reports),
      reports: reports
    }
  end

  defp merged_std_dev(_sum, _sum_sqrd, 1), do: 0.0

  defp merged_std_dev(sum, sum_sqrd, n) do
    :math.sqrt(max(0.0, (sum_sqrd - sum * sum / n) / (n - 1)))
  end
end
