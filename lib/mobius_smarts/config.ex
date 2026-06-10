defmodule MobiusSmarts.Config do
  @moduledoc """
  Runtime configuration: what to watch, and the few opinions the
  runtime cannot derive from the data.

  Built from the `:mobius_smarts` application environment by default,
  or from an explicit keyword list (so several instances can run
  side by side). See `MobiusSmarts` for the full story.

  Three keys are required — they are statements of fact and tolerance
  that the runtime refuses to guess:

  - `:watch` — the metrics to monitor (see `MobiusSmarts.Config.Metric`).
  - `:resolution` — the width of the summary windows every detector
    operates on, e.g. `{1, :minute}`. Mobius's RRD stores several
    resolutions simultaneously (by default 120 seconds of
    second-resolution data, 120 minutes of minute-resolution, 48
    hours of hour-resolution, 60 days of day-resolution); this names
    the tier you mean to monitor, and the runtime reads at exactly
    this cadence (`MobiusSmarts.Source.resample_windows/2` merges
    finer-grained stretches into these buckets). It is also the unit
    behind every window-counted option below and behind the
    false-alarm math. Pick a tier whose retention covers
    `:analysis_window` — `{1, :minute}` pairs with the default
    two-hour window and Mobius's default retention.
  - `:false_alarm_every` — the false-alarm budget: on a healthy
    device the whole instance may cry wolf about once per this
    duration. `{1, :week}` is one false alarm a week; `{2, :week}` is
    one every two weeks. Every detector threshold derives from this
    one tolerance and `:resolution` (`MobiusSmarts.Calibrate`).

  And one more becomes required when you opt into ETA projections:

  - `:trend_resolution` — window width for the Trend sweep's slope
    and time-to-threshold fits over `:trend_window`. Required when
    any watch entry declares a `:ceiling` or `:floor`; `{1, :hour}`
    pairs with the default 24-hour `:trend_window` (and Mobius's
    48-hour hour-resolution retention).

  Everything else has defaults stated in sigma units, windows (of
  `:resolution` width), or durations. A duration is a `{count, unit}`
  tuple with a positive integer count and a unit of `:second`,
  `:minute`, `:hour`, `:day`, or `:week` — or a raw positive integer
  of milliseconds (handy for fast test clocks). The tradeoffs:

  - `:mobius_instance` (default `:mobius`) — which Mobius instance to
    read.
  - `:interval` (defaults to `:resolution`) — how often the fast
    detectors re-scan. Scheduling only: no detector math depends on
    it, and ticking faster than `:resolution` cannot surface anything
    new. The exception that wants it is a fast test clock.
  - `:sweep_interval` (default `{1, :hour}`) — cadence of the slow
    detectors (Trend/ETA, Shape, Changepoint) and of baseline upkeep.
  - `:analysis_window` (default `{2, :hour}`) — how much trailing
    history each tick re-scans. This bounds the slowest level change
    the tick detectors can confirm; slower-than-window creep is the
    Trend sweep's job. Must hold at least `:min_baseline_windows`
    windows of `:resolution` width, or learning could never complete.
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
    size (in sigma) you care about. Must be in `(0, 3]`: above 3 the
    threshold calibration's Siegmund inversion overflows `:math.exp`
    (and a CUSUM tuned for 6-sigma drifts is a jump detector, not a
    drift detector).
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

  @typedoc """
  A time span: `{count, unit}` with a positive integer count, or raw
  positive-integer milliseconds.
  """
  @type duration() ::
          {pos_integer(), :second | :minute | :hour | :day | :week} | pos_integer()

  @type t() :: %__MODULE__{
          mobius_instance: atom(),
          source: module(),
          resolution: duration(),
          trend_resolution: duration() | nil,
          interval: duration(),
          sweep_interval: duration(),
          analysis_window: duration(),
          trend_window: duration(),
          false_alarm_every: duration(),
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
            resolution: nil,
            trend_resolution: nil,
            interval: nil,
            sweep_interval: {1, :hour},
            analysis_window: {2, :hour},
            trend_window: {24, :hour},
            false_alarm_every: nil,
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

  @duration_keys [
    :resolution,
    :interval,
    :sweep_interval,
    :analysis_window,
    :trend_window,
    :false_alarm_every,
    :warn_horizon,
    :critical_horizon,
    :refit_interval
  ]

  @duration_units [:second, :minute, :hour, :day, :week]

  @doc """
  Build a config from a keyword list, validating keys and value
  ranges, and normalizing `:watch` entries into
  `MobiusSmarts.Config.Metric` structs.

  A bad value raises at build time, naming the key, the value
  received, and the valid range — rather than detonating later inside
  a detector at boot. `:resolution` and `:false_alarm_every` are
  required; `:trend_resolution` is required when any watch entry
  declares a `:ceiling` or `:floor`.

  ## Examples

      iex> config =
      ...>   MobiusSmarts.Config.new!(
      ...>     watch: ["cpu.temp"],
      ...>     resolution: {1, :minute},
      ...>     false_alarm_every: {1, :week}
      ...>   )
      iex> [%MobiusSmarts.Config.Metric{name: "cpu.temp"}] = config.watch
      iex> {config.resolution, config.interval}
      {{1, :minute}, {1, :minute}}

      iex> MobiusSmarts.Config.new!(watch: [], wat: 1)
      ** (ArgumentError) unknown MobiusSmarts config keys: [:wat]

      iex> MobiusSmarts.Config.new!(watch: [], false_alarm_every: {1, :week})
      ** (ArgumentError) MobiusSmarts config is missing :resolution — the width of the summary windows every detector operates on. Name the Mobius RRD tier you mean to monitor; {1, :minute} pairs with the default two-hour :analysis_window.
  """
  @spec new!(keyword()) :: t()
  def new!(opts) when is_list(opts) do
    known = Map.keys(%__MODULE__{}) -- [:__struct__]

    case Keyword.keys(opts) -- known do
      [] -> :ok
      unknown -> raise ArgumentError, "unknown MobiusSmarts config keys: #{inspect(unknown)}"
    end

    config = struct!(__MODULE__, opts)

    unless is_list(config.watch) do
      invalid!(:watch, config.watch, "a list of watch entries")
    end

    config = %{config | watch: Enum.map(config.watch, &normalize_metric/1)}
    require_stated!(config)
    config = %{config | interval: config.interval || config.resolution}
    validate!(config)
    config
  end

  # The keys the runtime refuses to guess: stating them is the whole
  # contract, so their absence gets a teaching error, not a default.
  defp require_stated!(config) do
    if is_nil(config.resolution) do
      raise ArgumentError,
            "MobiusSmarts config is missing :resolution — the width of the summary " <>
              "windows every detector operates on. Name the Mobius RRD tier you mean " <>
              "to monitor; {1, :minute} pairs with the default two-hour :analysis_window."
    end

    if is_nil(config.false_alarm_every) do
      raise ArgumentError,
            "MobiusSmarts config is missing :false_alarm_every — the false-alarm " <>
              "budget. {1, :week} tolerates about one false alarm per week across " <>
              "the whole instance; every detector threshold is derived from it."
    end

    if is_nil(config.trend_resolution) and
         Enum.any?(config.watch, &(&1.ceiling != nil or &1.floor != nil)) do
      raise ArgumentError,
            "MobiusSmarts config is missing :trend_resolution — watch entries with a " <>
              ":ceiling or :floor enable the Trend sweep's ETA projections, which need " <>
              "a stated window width over :trend_window. {1, :hour} pairs with the " <>
              "default 24-hour :trend_window."
    end

    :ok
  end

  @doc """
  A duration (or raw milliseconds) as milliseconds.

  ## Examples

      iex> MobiusSmarts.Config.ms({2, :minute})
      120000
  """
  @spec ms(duration()) :: pos_integer()
  def ms(int) when is_integer(int), do: int
  def ms({n, :second}), do: n * 1_000
  def ms({n, :minute}), do: n * 60_000
  def ms({n, :hour}), do: n * 3_600_000
  def ms({n, :day}), do: n * 86_400_000
  def ms({n, :week}), do: n * 7 * 86_400_000

  @doc "A duration as seconds."
  @spec seconds(duration()) :: non_neg_integer()
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

  defp validate!(%__MODULE__{} = c) do
    Enum.each(@duration_keys, &validate_duration!(&1, Map.fetch!(c, &1)))
    validate_trend_resolution!(c)
    validate_budget!(c)
    validate_learnable!(c)

    check!(:mobius_instance, c.mobius_instance, is_atom(c.mobius_instance), "an atom")
    check!(:source, c.source, is_atom(c.source), "a module")

    check!(
      :min_baseline_windows,
      c.min_baseline_windows,
      is_integer(c.min_baseline_windows) and c.min_baseline_windows >= 2,
      "an integer >= 2"
    )

    check!(
      :clear_after,
      c.clear_after,
      is_integer(c.clear_after) and c.clear_after >= 1,
      "an integer >= 1"
    )

    check!(
      :gap_factor,
      c.gap_factor,
      is_number(c.gap_factor) and c.gap_factor > 1,
      "a number > 1"
    )

    check!(
      :cusum_k,
      c.cusum_k,
      is_number(c.cusum_k) and c.cusum_k > 0 and c.cusum_k <= 3,
      "a number in (0, 3] (above 3 the Siegmund ARL inversion overflows :math.exp)"
    )

    check!(
      :ewma_lambda,
      c.ewma_lambda,
      is_number(c.ewma_lambda) and c.ewma_lambda > 0 and c.ewma_lambda <= 1,
      "a number in (0, 1]"
    )

    check!(
      :novelty,
      c.novelty,
      c.novelty == :auto or is_boolean(c.novelty),
      ":auto, true, or false"
    )

    Enum.each(c.watch, &validate_metric!/1)
  end

  defp validate_duration!(_key, {n, unit})
       when is_integer(n) and n > 0 and unit in @duration_units do
    :ok
  end

  defp validate_duration!(_key, milliseconds)
       when is_integer(milliseconds) and milliseconds > 0 do
    :ok
  end

  defp validate_duration!(key, value) do
    invalid!(
      key,
      value,
      "a duration ({count, :second | :minute | :hour | :day | :week} " <>
        "with a positive integer count, or positive integer milliseconds)"
    )
  end

  defp validate_trend_resolution!(%__MODULE__{trend_resolution: nil}), do: :ok

  defp validate_trend_resolution!(%__MODULE__{} = c) do
    validate_duration!(:trend_resolution, c.trend_resolution)

    if ms(c.trend_resolution) > ms(c.trend_window) do
      raise ArgumentError,
            "invalid MobiusSmarts config: :trend_resolution (#{inspect(c.trend_resolution)}) " <>
              "must fit inside :trend_window (#{inspect(c.trend_window)})"
    end

    :ok
  end

  # A budget under one window clamps every detector threshold to its
  # floor (Calibrate's ARL bottoms out at 2), so the instance would
  # alarm near-constantly while looking healthy on paper.
  defp validate_budget!(%__MODULE__{false_alarm_every: budget, resolution: resolution}) do
    if ms(budget) < ms(resolution) do
      raise ArgumentError,
            "invalid MobiusSmarts config: :false_alarm_every (#{inspect(budget)}) must be " <>
              "at least one :resolution window (#{inspect(resolution)}); a sub-window budget " <>
              "clamps every detector threshold to its floor"
    end
  end

  # Learning needs :min_baseline_windows healthy windows inside one
  # analysis window — if the window can't even hold that many, every
  # metric would sit in :learning forever.
  defp validate_learnable!(%__MODULE__{} = c) do
    capacity = div(ms(c.analysis_window), ms(c.resolution))

    if capacity < c.min_baseline_windows do
      raise ArgumentError,
            "invalid MobiusSmarts config: :analysis_window (#{inspect(c.analysis_window)}) " <>
              "holds only #{capacity} windows of :resolution #{inspect(c.resolution)} — " <>
              "fewer than :min_baseline_windows (#{c.min_baseline_windows}), so no baseline " <>
              "could ever be fitted"
    end
  end

  defp validate_metric!(%Metric{} = metric) do
    unless is_binary(metric.name) do
      raise ArgumentError,
            "watch entry :metric name must be a string, got: #{inspect(metric.name)}"
    end

    check_metric!(metric, :tags, metric.tags, is_map(metric.tags), "a map")

    check_metric!(
      metric,
      :histogram,
      metric.histogram,
      is_boolean(metric.histogram),
      "a boolean"
    )

    for {key, value} <- [ceiling: metric.ceiling, floor: metric.floor] do
      check_metric!(metric, key, value, is_nil(value) or is_number(value), "a number or nil")
    end

    if is_number(metric.floor) and is_number(metric.ceiling) and metric.floor >= metric.ceiling do
      raise ArgumentError,
            "watch entry #{inspect(metric.name)}: :floor (#{inspect(metric.floor)}) must be " <>
              "below :ceiling (#{inspect(metric.ceiling)})"
    end

    :ok
  end

  defp check!(_key, _value, true, _expected), do: :ok
  defp check!(key, value, false, expected), do: invalid!(key, value, expected)

  defp check_metric!(_metric, _key, _value, true, _expected), do: :ok

  defp check_metric!(metric, key, value, false, expected) do
    raise ArgumentError,
          "watch entry #{inspect(metric.name)}: #{inspect(key)} must be #{expected}, " <>
            "got: #{inspect(value)}"
  end

  defp invalid!(key, value, expected) do
    raise ArgumentError,
          "invalid MobiusSmarts config: #{inspect(key)} must be #{expected}, got: #{inspect(value)}"
  end
end
