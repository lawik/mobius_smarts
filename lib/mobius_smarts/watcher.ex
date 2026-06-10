defmodule MobiusSmarts.Watcher do
  @moduledoc false
  # The fast loop: every `config.interval`, pull each watched metric's
  # recent summary windows and run the cheap detectors (Jump/wobble,
  # Shift, Drift) plus the cross-metric novelty score. Holds no
  # detection state of its own — every tick is a pure function of
  # (stored RRD history, baseline), so crashes and reboots cost
  # nothing; the Mobius-persisted RRD *is* the state.

  use GenServer

  alias MobiusSmarts.{Analysis, Board, Calibrate, Config}

  require Logger

  @tick_kinds [
    :jumped,
    :spiked,
    :wobbling,
    :flatlined,
    :shifted_up,
    :shifted_down,
    :drifting_up,
    :drifting_down,
    :silent,
    :reporting_gap
  ]

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    board = Keyword.fetch!(opts, :name)
    GenServer.start_link(__MODULE__, opts, name: :"#{board}.Watcher")
  end

  @impl GenServer
  def init(opts) do
    config = Keyword.fetch!(opts, :config)

    state = %{
      board: Keyword.fetch!(opts, :name),
      config: config,
      calib: Calibrate.for_config(config),
      cadence_warned: MapSet.new()
    }

    {:ok, state, {:continue, :tick}}
  end

  @impl GenServer
  def handle_continue(:tick, state) do
    state = tick(state)

    # Scheduled after the tick's work, so the effective cadence is
    # interval + work time. Known and acceptable drift: every tick
    # re-scans trailing history, so nothing is missed — confirmation
    # just lands a moment later.
    Process.send_after(self(), :tick, Config.ms(state.config.interval))
    {:noreply, state}
  end

  @impl GenServer
  def handle_info(:tick, state), do: {:noreply, state, {:continue, :tick}}

  defp tick(state) do
    {latest, state} =
      Enum.map_reduce(state.config.watch, state, fn metric, state ->
        key = Config.Metric.key(metric)

        {latest, state} =
          try do
            watch_metric(state, metric, key)
          rescue
            error ->
              Logger.warning(
                "MobiusSmarts tick failed for #{metric.name}: #{Exception.message(error)}"
              )

              {nil, state}
          end

        {{key, latest}, state}
      end)

    score_novelty(state, Map.new(latest))
    state
  end

  # Returns {latest, state}: the metric's latest {ts, avg} for the
  # novelty vector — nil when the metric is empty or stale — plus
  # state with any cadence warning recorded.
  defp watch_metric(state, metric, key) do
    config = state.config
    now = System.system_time(:second)

    series =
      config.source.summary_series(metric.name, metric.tags,
        last: config.analysis_window,
        resolution: config.resolution,
        mobius_instance: config.mobius_instance
      )

    case series do
      :empty ->
        Board.report(state.board, key, @tick_kinds, [Analysis.silent_candidate(nil, now)])
        {nil, state}

      series ->
        lists = Analysis.to_lists(series)
        resolution_s = Config.seconds(config.resolution)
        gaps = Analysis.gaps(lists.ts, config.gap_factor, resolution_s)
        segment = Analysis.active_segment(lists, gaps)
        last_ts = List.last(segment.ts)

        state = warn_cadence_mismatch(state, metric, key, Analysis.median_cadence(lists.ts))

        stale_after = config.gap_factor * resolution_s
        stale? = now - last_ts > stale_after

        candidates =
          Analysis.gap_candidates(gaps) ++
            if stale? do
              [Analysis.silent_candidate(last_ts, now)]
            else
              detector_candidates(state, key, segment, now)
            end

        Board.report(state.board, key, @tick_kinds, candidates)

        # A stale metric contributes nothing to the novelty vector: its
        # hour-old "latest" paired with other metrics' live values would
        # describe a cross-metric window that never happened.
        if stale?, do: {nil, state}, else: {{last_ts, List.last(segment.avg)}, state}
    end
  end

  # Gap detection, staleness, and the false-alarm calibration all key
  # off the configured `:resolution`; resampling guarantees the series
  # matches it whenever Mobius stores data at least that fine. So a
  # measured median cadence disagreeing by more than 50% means Mobius
  # is not recording at the cadence the config claims (e.g. the RRD
  # tier is coarser than `:resolution`) — every derived number is off
  # by roughly the ratio. Worth one warning per metric, not per tick.
  defp warn_cadence_mismatch(state, metric, key, measured) do
    resolution_s = Config.ms(state.config.resolution) / 1000

    mismatched? =
      measured != nil and measured > 0 and
        abs(measured - resolution_s) > 0.5 * resolution_s

    if mismatched? and not MapSet.member?(state.cadence_warned, key) do
      ratio = Float.round(max(measured / resolution_s, resolution_s / measured), 1)

      Logger.warning(
        "MobiusSmarts: #{metric.name} windows arrive every ~#{measured}s but :resolution " <>
          "is #{resolution_s}s — Mobius does not seem to store data at the configured " <>
          "resolution, so gap detection and the false-alarm calibration are off by " <>
          "roughly #{ratio}x"
      )

      %{state | cadence_warned: MapSet.put(state.cadence_warned, key)}
    else
      state
    end
  end

  defp detector_candidates(state, key, segment, now) do
    case Board.baseline(state.board, key) || fit_baseline(state, key, segment, now) do
      nil -> []
      baseline -> Analysis.tick_candidates(segment, baseline, state.calib, state.config)
    end
  end

  defp fit_baseline(state, key, segment, now) do
    case Analysis.fit_baseline(segment,
           min_windows: state.config.min_baseline_windows,
           now: now
         ) do
      {:ok, baseline} ->
        Board.put_baseline(state.board, key, baseline)
        baseline

      {:error, _learning} ->
        nil
    end
  end

  defp score_novelty(state, latest) do
    with model when model != nil <- Board.novelty(state.board),
         vector when vector != nil <- vector_for(model, latest, state.config) do
      candidates = Analysis.novelty_candidates(model, vector)
      Board.report(state.board, {"*", %{}}, [:novel_behavior], candidates)
    else
      _missing -> :ok
    end
  end

  # The model's metrics in its fitted order — all must have reported
  # this tick, or the vector (and a correlation judgement on it) would
  # be fabricated. And because the model was fitted on timestamp-ALIGNED
  # rows only (`Analysis.fit_novelty/3` intersects timestamps), the
  # scored vector must be aligned too: the per-metric latest timestamps
  # may spread by at most one configured `:resolution` window. Values
  # further apart than that straddle windows the model never saw side
  # by side.
  defp vector_for(model, latest, config) do
    values = Enum.map(model.keys, &latest[&1])

    if Enum.any?(values, &is_nil/1) or not aligned?(values, config) do
      nil
    else
      Enum.map(values, &elem(&1, 1))
    end
  end

  defp aligned?(values, config) do
    {min_ts, max_ts} = values |> Enum.map(&elem(&1, 0)) |> Enum.min_max()
    max_ts - min_ts <= Config.seconds(config.resolution)
  end
end
