defmodule MobiusSmarts.Config do
  @moduledoc """
  Runtime configuration: what to watch, and the few opinions the
  runtime cannot derive from the data.

  Built from the `:mobius_smarts` application environment by default,
  or from an explicit keyword list (so several instances can run
  side by side). See `MobiusSmarts` for the full story; the headline
  keys are:

  - `:watch` — the metrics to monitor (see `MobiusSmarts.Config.Metric`).
  - `:false_alarm_budget` — how often the whole instance may cry wolf
    on a healthy device, e.g. `{1, :week}`. Every detector threshold
    is derived from this one number (`MobiusSmarts.Calibrate`).
  - `:mobius_instance` — which Mobius instance to read.

  Everything else has defaults stated in sigma units, windows, or
  durations, with the tradeoff documented here:

  - `:interval` (default `{1, :minute}`) — tick cadence; should match
    the Mobius RRD resolution you want to monitor at.
  - `:sweep_interval` (default `{1, :hour}`) — cadence of the slow
    detectors (Trend/ETA, Shape, Changepoint) and of baseline upkeep.
  - `:analysis_window` (default `{2, :hour}`) — how much trailing
    history each tick re-scans. This bounds the slowest level change
    the tick detectors can confirm; slower-than-window creep is the
    Trend sweep's job. The default stays inside Mobius's default
    minute-resolution retention.
  - `:trend_window` (default `{24, :hour}`) — history for the Trend
    sweep's slope and ETA projections.
  - `:warn_horizon` / `:critical_horizon` (defaults `{7, :day}` /
    `{1, :day}`) — `:approaching_limit` severity: how close a
    projected ceiling/floor crossing must be to warn / go critical.
  - `:min_baseline_windows` (default `60`) — healthy windows required
    before a metric leaves `:learning` and detection starts.
  - `:refit_interval` (default `{1, :day}`) — how often a quiet
    metric's baseline is refreshed. Refits are skipped while findings
    are active, so trouble is never learned as the new normal.
  - `:clear_after` (default `3`) — consecutive quiet ticks before an
    active condition clears. Escalate fast, de-escalate slow.
  - `:gap_factor` (default `3.0`) — a window-to-window gap longer than
    this multiple of the metric's own cadence is a reporting gap: the
    series re-anchors after it and a `:reporting_gap` observation is
    recorded.
  - `:cusum_k` (default `0.5`) — Drift's drain rate: half the drift
    size (in sigma) you care about.
  - `:ewma_lambda` (default `0.2`) — Shift's nudge weight.
  - `:novelty` (default `:auto`) — cross-metric novelty detection:
    `:auto` enables it when at least 3 metrics are watched.
  - `:source` (default `MobiusSmarts.Source`) — the data-access
    module; replaceable for tests and replays.
  """

  defmodule Metric do
    @moduledoc """
    One watched metric. In config, an entry of `:watch` is a name, a
    keyword list, or a map:

        watch: [
          "vm.memory.used_percent",
          [metric: "disk.used_percent", ceiling: 95.0],
          [metric: "battery.percent", floor: 5.0],
          [metric: "http.request.duration", histogram: true, tags: %{route: "/api"}]
        ]

    - `:metric` — the Mobius summary metric name (required).
    - `:tags` — exact tag set to match, default `%{}`.
    - `:ceiling` / `:floor` — the value at which this metric is a
      problem (disk full, battery dead). Opting in enables the Trend
      sweep and its `:approaching_limit` ETA findings — the one truly
      domain-specific number in the whole configuration.
    - `:histogram` — set `true` for metrics registered with
      `reporter_options: [histogram: ...]` to enable the Shape sweep.
    """

    @type t() :: %__MODULE__{
            name: String.t(),
            tags: map(),
            ceiling: number() | nil,
            floor: number() | nil,
            histogram: boolean()
          }

    @enforce_keys [:name]
    defstruct [:name, :ceiling, :floor, tags: %{}, histogram: false]

    @doc "The `{name, tags}` identity used throughout the runtime."
    @spec key(t()) :: {String.t(), map()}
    def key(%__MODULE__{name: name, tags: tags}), do: {name, tags}
  end

  @type duration() :: {pos_integer(), :second | :minute | :hour | :day | :week}

  @type t() :: %__MODULE__{
          mobius_instance: atom(),
          source: module(),
          interval: duration(),
          sweep_interval: duration(),
          analysis_window: duration(),
          trend_window: duration(),
          false_alarm_budget: duration(),
          warn_horizon: duration(),
          critical_horizon: duration(),
          min_baseline_windows: pos_integer(),
          refit_interval: duration(),
          clear_after: pos_integer(),
          gap_factor: float(),
          cusum_k: float(),
          ewma_lambda: float(),
          novelty: :auto | boolean(),
          watch: [Metric.t()]
        }

  defstruct mobius_instance: :mobius,
            source: MobiusSmarts.Source,
            interval: {1, :minute},
            sweep_interval: {1, :hour},
            analysis_window: {2, :hour},
            trend_window: {24, :hour},
            false_alarm_budget: {1, :week},
            warn_horizon: {7, :day},
            critical_horizon: {1, :day},
            min_baseline_windows: 60,
            refit_interval: {1, :day},
            clear_after: 3,
            gap_factor: 3.0,
            cusum_k: 0.5,
            ewma_lambda: 0.2,
            novelty: :auto,
            watch: []

  @doc """
  Build a config from a keyword list, validating keys and normalizing
  `:watch` entries into `MobiusSmarts.Config.Metric` structs.

  ## Examples

      iex> config = MobiusSmarts.Config.new!(watch: ["cpu.temp", [metric: "disk.pct", ceiling: 95]])
      iex> [%MobiusSmarts.Config.Metric{name: "cpu.temp"}, disk] = config.watch
      iex> {disk.name, disk.ceiling}
      {"disk.pct", 95}

      iex> MobiusSmarts.Config.new!(watch: [], wat: 1)
      ** (ArgumentError) unknown MobiusSmarts config keys: [:wat]
  """
  @spec new!(keyword()) :: t()
  def new!(opts) when is_list(opts) do
    known = Map.keys(%__MODULE__{}) -- [:__struct__]

    case Keyword.keys(opts) -- known do
      [] -> :ok
      unknown -> raise ArgumentError, "unknown MobiusSmarts config keys: #{inspect(unknown)}"
    end

    config = struct!(__MODULE__, opts)
    %{config | watch: Enum.map(config.watch, &normalize_metric/1)}
  end

  @doc """
  A duration (or raw milliseconds) as milliseconds.

  ## Examples

      iex> MobiusSmarts.Config.ms({2, :minute})
      120000
  """
  @spec ms(duration() | non_neg_integer()) :: non_neg_integer()
  def ms(int) when is_integer(int), do: int
  def ms({n, :second}), do: n * 1_000
  def ms({n, :minute}), do: n * 60_000
  def ms({n, :hour}), do: n * 3_600_000
  def ms({n, :day}), do: n * 86_400_000
  def ms({n, :week}), do: n * 7 * 86_400_000

  @doc "A duration as seconds."
  @spec seconds(duration() | non_neg_integer()) :: non_neg_integer()
  def seconds(duration), do: div(ms(duration), 1_000)

  @doc "Whether cross-metric novelty detection is on for this config."
  @spec novelty?(t()) :: boolean()
  def novelty?(%__MODULE__{novelty: :auto, watch: watch}), do: length(watch) >= 3
  def novelty?(%__MODULE__{novelty: flag}), do: flag

  defp normalize_metric(%Metric{} = metric), do: metric
  defp normalize_metric(name) when is_binary(name), do: %Metric{name: name}

  defp normalize_metric(entry) when is_list(entry) or is_map(entry) do
    entry = Map.new(entry)
    {name, rest} = Map.pop(entry, :metric)

    unless is_binary(name) do
      raise ArgumentError, "watch entry needs a :metric name, got: #{inspect(entry)}"
    end

    struct!(Metric, Map.put(rest, :name, name))
  end
end
