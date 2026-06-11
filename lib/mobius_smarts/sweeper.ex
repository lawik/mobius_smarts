defmodule MobiusSmarts.Sweeper do
  @moduledoc false
  # The slow loop: every `config.sweep_interval`, run the detectors
  # that want hindsight or longer horizons — Trend/ETA projections,
  # retrospective changepoints, distribution-shape comparison — and do
  # baseline upkeep: scheduled refits (guarded: never while the metric
  # has active findings, so trouble is not learned as normal) and the
  # novelty model fit.

  use GenServer

  alias MobiusSmarts.{Analysis, Board, Calibrate, Config, Seasonal}

  require Logger

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    board = Keyword.fetch!(opts, :name)
    GenServer.start_link(__MODULE__, opts, name: :"#{board}.Sweeper")
  end

  @impl GenServer
  def init(opts) do
    config = Keyword.fetch!(opts, :config)

    state = %{
      board: Keyword.fetch!(opts, :name),
      config: config,
      calib: Calibrate.for_config(config)
    }

    # First sweep after the first full interval: the Watcher needs to
    # have established baselines first, and sweep findings look back
    # far enough that startup latency costs nothing.
    Process.send_after(self(), :sweep, Config.ms(config.sweep_interval))
    {:ok, state}
  end

  @impl GenServer
  def handle_info(:sweep, state) do
    sweep(state)

    # Scheduled after the sweep's work, so the effective cadence is
    # sweep_interval + work time. Known and acceptable drift: every
    # sweep re-scans trailing history, so nothing is missed —
    # confirmation just lands a moment later.
    Process.send_after(self(), :sweep, Config.ms(state.config.sweep_interval))
    {:noreply, state}
  end

  defp sweep(state) do
    for metric <- state.config.watch do
      try do
        sweep_metric(state, metric)
      rescue
        error ->
          Logger.warning(
            "MobiusSmarts sweep failed for #{metric.name}: #{Exception.message(error)}"
          )
      end
    end

    if Config.novelty?(state.config), do: fit_novelty(state)
  end

  defp sweep_metric(state, metric) do
    key = Config.Metric.key(metric)
    now = System.system_time(:second)

    covered =
      [:regime_change] ++
        if(metric.ceiling || metric.floor, do: [:approaching_limit], else: []) ++
        if(metric.histogram, do: [:shape_drift], else: [])

    candidates =
      trend_candidates(state, metric) ++
        changepoint_candidates(state, metric) ++
        shape_candidates(state, metric, key)

    Board.report(state.board, key, covered, candidates)
    maybe_refit(state, metric, key, now)
  end

  defp trend_candidates(_state, %{ceiling: nil, floor: nil}), do: []

  # Trend fits run at :trend_resolution (config requires it alongside
  # any ceiling/floor) — the coarser cadence whose RRD tier actually
  # spans :trend_window.
  defp trend_candidates(state, metric) do
    case pull(state, metric, state.config.trend_window, state.config.trend_resolution) do
      :empty -> []
      lists -> Analysis.trend_candidates(lists, metric, state.config)
    end
  end

  defp changepoint_candidates(state, metric) do
    case pull(state, metric, state.config.analysis_window, state.config.resolution) do
      :empty -> []
      lists -> Analysis.changepoint_candidates(lists)
    end
  end

  defp shape_candidates(%{config: config} = state, %{histogram: true} = metric, key) do
    with baseline when baseline != nil <- Board.baseline(state.board, key),
         {:ok, base_sketch} <-
           config.source.sketch(metric.name, metric.tags,
             from: baseline.from,
             to: baseline.to,
             mobius_instance: config.mobius_instance
           ),
         # The current window covers only what the baseline has not
         # seen: from its fit horizon forward. A trailing-window shape
         # would contain the baseline period itself whenever the fit
         # is recent — comparing the reference against a diluted
         # superset of itself (issue #4).
         {:ok, current_sketch} <-
           config.source.sketch(metric.name, metric.tags,
             from: baseline.to,
             mobius_instance: config.mobius_instance
           ) do
      Analysis.shape_candidates(base_sketch, current_sketch)
    else
      _unavailable -> []
    end
  end

  defp shape_candidates(_state, _metric, _key), do: []

  # Refresh a stale baseline — only while the metric is quiet, so the
  # runtime never learns an active problem as the new normal.
  defp maybe_refit(state, metric, key, now) do
    baseline = Board.baseline(state.board, key)
    refit_after = Config.seconds(state.config.refit_interval)

    with %{fitted_at: fitted_at} <- baseline,
         true <- now - fitted_at >= refit_after,
         [] <- active_for(state.board, metric) do
      refit(state, metric, key, now)
    else
      _not_due -> :ok
    end
  end

  defp refit(state, metric, key, now) do
    with lists when lists != :empty <-
           pull(state, metric, state.config.analysis_window, state.config.resolution),
         {series, seasonal?} = seasonal_series(state, key, lists),
         {:ok, fresh} <-
           Analysis.fit_baseline(series,
             min_windows: state.config.min_baseline_windows,
             now: now
           ) do
      Board.put_baseline(state.board, key, Map.put(fresh, :seasonal, seasonal?))
    else
      _keep_old -> :ok
    end
  end

  # Refits must fit the same series detection runs on: residuals when
  # the seasonal model is warm, raw otherwise (issue #8). The Watcher
  # owns model updates; the sweep only reads.
  defp seasonal_series(%{config: %{seasonality: nil}}, _key, lists), do: {lists, false}

  defp seasonal_series(state, key, lists) do
    model = Board.seasonal(state.board, key)

    if model && Seasonal.ready?(model) do
      {Seasonal.residuals(model, lists), true}
    else
      {lists, false}
    end
  end

  defp fit_novelty(state) do
    series_by_key =
      for metric <- state.config.watch,
          lists = pull(state, metric, state.config.analysis_window, state.config.resolution),
          lists != :empty do
        {Config.Metric.key(metric), lists}
      end

    if length(series_by_key) >= 2 do
      case Analysis.fit_novelty(series_by_key, state.calib.arl) do
        {:ok, model} -> Board.put_novelty(state.board, model)
        {:error, :insufficient} -> :ok
      end
    end
  end

  defp active_for(board, metric) do
    key = Config.Metric.key(metric)
    Enum.filter(Board.findings(board), &({&1.metric, &1.tags} == key))
  end

  defp pull(%{config: config}, metric, window, resolution) do
    case config.source.summary_series(metric.name, metric.tags,
           last: window,
           resolution: resolution,
           mobius_instance: config.mobius_instance
         ) do
      :empty -> :empty
      series -> Analysis.to_lists(series)
    end
  end
end
