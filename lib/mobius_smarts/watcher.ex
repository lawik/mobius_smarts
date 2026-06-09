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
      calib: Calibrate.for_config(config)
    }

    {:ok, state, {:continue, :tick}}
  end

  @impl GenServer
  def handle_continue(:tick, state) do
    tick(state)
    Process.send_after(self(), :tick, Config.ms(state.config.interval))
    {:noreply, state}
  end

  @impl GenServer
  def handle_info(:tick, state), do: {:noreply, state, {:continue, :tick}}

  defp tick(state) do
    latest =
      for metric <- state.config.watch, into: %{} do
        key = Config.Metric.key(metric)

        latest =
          try do
            watch_metric(state, metric, key)
          rescue
            error ->
              Logger.warning(
                "MobiusSmarts tick failed for #{metric.name}: #{Exception.message(error)}"
              )

              nil
          end

        {key, latest}
      end

    score_novelty(state, latest)
  end

  # Returns the metric's latest {ts, avg} for the novelty vector.
  defp watch_metric(state, metric, key) do
    config = state.config
    now = System.system_time(:second)

    series =
      config.source.summary_series(metric.name, metric.tags,
        last: config.analysis_window,
        mobius_instance: config.mobius_instance
      )

    case series do
      :empty ->
        Board.report(state.board, key, @tick_kinds, [Analysis.silent_candidate(nil, now)])
        nil

      series ->
        lists = Analysis.to_lists(series)
        {cadence, gaps} = Analysis.gaps(lists.ts, config.gap_factor)
        segment = Analysis.active_segment(lists, gaps)
        last_ts = List.last(segment.ts)

        stale_after = config.gap_factor * max(cadence || 0, Config.seconds(config.interval))

        candidates =
          Analysis.gap_candidates(gaps) ++
            if now - last_ts > stale_after do
              [Analysis.silent_candidate(last_ts, now)]
            else
              detector_candidates(state, key, segment, now)
            end

        Board.report(state.board, key, @tick_kinds, candidates)
        {last_ts, List.last(segment.avg)}
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
         vector when vector != nil <- vector_for(model, latest) do
      candidates = Analysis.novelty_candidates(model, vector)
      Board.report(state.board, {"*", %{}}, [:novel_behavior], candidates)
    else
      _missing -> :ok
    end
  end

  # The model's metrics in its fitted order — all must have reported
  # this tick, or the vector (and a correlation judgement on it) would
  # be fabricated.
  defp vector_for(model, latest) do
    values = Enum.map(model.keys, &latest[&1])

    case Enum.find(values, &is_nil/1) do
      nil -> Enum.map(values, &elem(&1, 1))
      _missing -> nil
    end
  end
end
