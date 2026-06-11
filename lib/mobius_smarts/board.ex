defmodule MobiusSmarts.Board do
  @moduledoc false
  # Owns the instance's ETS table and the finding lifecycle: candidates
  # come in from Watcher/Sweeper, findings and the aggregate health
  # level come out. All mutations are serialized through this process;
  # reads (status/findings) go straight to ETS.
  #
  # Lifecycle rules:
  # - A condition candidate with a known id updates the finding in
  #   place (concern, evidence, message, last_seen). Severity only
  #   escalates while active — it never quietly de-escalates; it
  #   clears instead. Telemetry: :raised on insert, :escalated on
  #   severity increase, :cleared on clear.
  # - A reporter declares which kinds it covered. Active conditions of
  #   covered kinds that were NOT re-confirmed accrue misses; at
  #   config.clear_after consecutive misses the finding clears.
  # - Observations insert once (id includes onset) and emit :raised;
  #   re-reports of the same observation are no-ops.

  use GenServer

  alias MobiusSmarts.{Config, Finding, Seasonal}

  @type scope() :: {String.t(), map()}

  ## Client

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    name = Keyword.fetch!(opts, :name)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Reconcile one reporter pass for `{metric, tags}`: `candidates` are
  the issues confirmed this pass, `covered_kinds` the kinds this
  reporter is able to confirm (so only those accrue misses).
  """
  @spec report(atom(), scope(), [Finding.kind()], [map()]) :: :ok
  def report(board, scope, covered_kinds, candidates) do
    GenServer.call(board, {:report, scope, covered_kinds, candidates})
  end

  @spec put_baseline(atom(), scope(), map()) :: :ok
  def put_baseline(board, key, baseline) do
    GenServer.call(board, {:put_baseline, key, baseline})
  end

  @doc """
  Record where baselining stands for a still-learning metric
  (`%{reason: ..., windows: ..., needed: ...}`, the
  `MobiusSmarts.Analysis.fit_baseline/2` error shape).
  """
  @spec put_learning(atom(), scope(), map()) :: :ok
  def put_learning(board, key, progress) do
    GenServer.call(board, {:put, {:learning, key}, progress})
  end

  @doc """
  Forget a metric's baseline — the stale-baseline response (issue #5):
  the metric returns to learning and refits from current history.
  """
  @spec drop_baseline(atom(), scope()) :: :ok
  def drop_baseline(board, key) do
    GenServer.call(board, {:drop_baseline, key})
  end

  @spec put_novelty(atom(), map()) :: :ok
  def put_novelty(board, model_map) do
    GenServer.call(board, {:put, :novelty, model_map})
  end

  @doc "Store a metric's incremental seasonal model (issue #8)."
  @spec put_seasonal(atom(), scope(), Seasonal.t()) :: :ok
  def put_seasonal(board, key, model) do
    GenServer.call(board, {:put, {:seasonal, key}, model})
  end

  @spec seasonal(atom(), scope()) :: Seasonal.t() | nil
  def seasonal(board, key), do: lookup(board, {:seasonal, key})

  @spec baseline(atom(), scope()) :: map() | nil
  def baseline(board, key), do: lookup(board, {:baseline, key})

  @spec novelty(atom()) :: map() | nil
  def novelty(board), do: lookup(board, :novelty)

  @spec status(atom()) :: map()
  def status(board), do: lookup(board, :status)

  @spec findings(atom()) :: [Finding.t()]
  def findings(board) do
    board
    |> fold_findings(fn
      {{:finding, _id}, %Finding{class: :condition, status: :active} = f, _miss}, acc ->
        [f | acc]

      _row, acc ->
        acc
    end)
    |> Enum.sort_by(&{severity_rank(&1.severity), -&1.concern})
  end

  @spec observations(atom(), pos_integer()) :: [Finding.t()]
  def observations(board, limit \\ 50) do
    board
    |> fold_findings(fn
      {{:finding, _id}, %Finding{class: :observation} = f, _miss}, acc -> [f | acc]
      _row, acc -> acc
    end)
    |> Enum.sort_by(&(-&1.raised_at))
    |> Enum.take(limit)
  end

  ## Server

  @impl GenServer
  def init(opts) do
    name = Keyword.fetch!(opts, :name)
    config = Keyword.fetch!(opts, :config)

    table = :ets.new(name, [:named_table, :set, :protected, read_concurrency: true])
    :ets.insert(table, {:config, config})

    state = %{table: table, name: name, config: config}
    refresh_status(state)
    {:ok, state}
  end

  @impl GenServer
  def handle_call({:report, scope, covered_kinds, candidates}, _from, state) do
    now = System.system_time(:second)
    {metric, tags} = scope

    confirmed_ids =
      for candidate <- candidates do
        finding = to_finding(candidate, metric, tags, now)
        upsert(state, finding, now)
        Finding.id(finding)
      end

    displace_opposites(state, scope, candidates, now)
    miss_uncovered(state, scope, covered_kinds, MapSet.new(confirmed_ids), now)
    refresh_status(state)
    {:reply, :ok, state}
  end

  def handle_call({:put, key, value}, _from, state) do
    :ets.insert(state.table, {key, value})
    refresh_status(state)
    {:reply, :ok, state}
  end

  # A fitted baseline supersedes the metric's learning progress.
  def handle_call({:put_baseline, key, baseline}, _from, state) do
    :ets.insert(state.table, {{:baseline, key}, baseline})
    :ets.delete(state.table, {:learning, key})
    refresh_status(state)
    {:reply, :ok, state}
  end

  def handle_call({:drop_baseline, key}, _from, state) do
    :ets.delete(state.table, {:baseline, key})
    refresh_status(state)
    {:reply, :ok, state}
  end

  ## Lifecycle

  defp to_finding(candidate, metric, tags, now) do
    struct!(
      Finding,
      candidate
      |> Map.put(:metric, metric)
      |> Map.put(:tags, tags)
      |> Map.put(:raised_at, now)
      |> Map.put(:last_seen_at, now)
      |> Map.put(:status, if(candidate.class == :observation, do: :noted, else: :active))
    )
  end

  defp upsert(state, %Finding{} = fresh, now) do
    id = Finding.id(fresh)

    case :ets.lookup(state.table, {:finding, id}) do
      [] ->
        :ets.insert(state.table, {{:finding, id}, fresh, 0})
        emit(state, :raised, fresh)

      [{_key, %Finding{class: :observation}, _miss}] ->
        # Same dated observation re-derived from overlapping history.
        :ok

      [{_key, %Finding{status: :cleared}, _miss}] ->
        # The trouble is back: a fresh raise, not an update.
        :ets.insert(state.table, {{:finding, id}, fresh, 0})
        emit(state, :raised, fresh)

      [{_key, %Finding{} = existing, _miss}] ->
        updated = %{
          existing
          | concern: fresh.concern,
            evidence: fresh.evidence,
            message: fresh.message,
            onset: existing.onset || fresh.onset,
            severity: max_severity(existing.severity, fresh.severity),
            last_seen_at: now
        }

        :ets.insert(state.table, {{:finding, id}, updated, 0})

        if severity_rank(updated.severity) < severity_rank(existing.severity) do
          emit(state, :escalated, updated)
        end
    end
  end

  defp miss_uncovered(state, {metric, tags}, covered_kinds, confirmed_ids, now) do
    covered = MapSet.new(covered_kinds)

    :ets.foldl(
      fn
        {{:finding, id}, %Finding{class: :condition, status: :active} = f, miss}, acc ->
          if f.metric == metric and f.tags == tags and
               MapSet.member?(covered, f.kind) and
               not MapSet.member?(confirmed_ids, id) do
            miss(state, id, f, miss + 1, now)
          end

          acc

        _row, acc ->
          acc
      end,
      :ok,
      state.table
    )
  end

  defp miss(state, id, finding, misses, now) do
    if misses >= state.config.clear_after do
      clear(state, id, finding, now)
    else
      :ets.insert(state.table, {{:finding, id}, finding, misses})
    end
  end

  defp clear(state, id, finding, now) do
    cleared = %{finding | status: :cleared, cleared_at: now}
    :ets.insert(state.table, {{:finding, id}, cleared, 0})
    emit(state, :cleared, cleared)
  end

  # Directional kind pairs displace each other immediately (issue #5):
  # a level cannot be shifted up and shifted down at once, so raising
  # one direction clears the other without waiting out clear_after —
  # unless both arrived in the same pass, which is its own
  # contradiction and handled upstream as :baseline_stale.
  @opposites %{
    shifted_up: :shifted_down,
    shifted_down: :shifted_up,
    drifting_up: :drifting_down,
    drifting_down: :drifting_up
  }

  defp displace_opposites(state, {metric, tags}, candidates, now) do
    confirmed =
      for %{class: :condition, kind: kind} <- candidates, into: MapSet.new(), do: kind

    displaced =
      for kind <- confirmed,
          opposite = @opposites[kind],
          opposite != nil,
          not MapSet.member?(confirmed, opposite),
          into: MapSet.new(),
          do: opposite

    if MapSet.size(displaced) > 0 do
      :ets.foldl(
        fn row, acc ->
          displace_row(state, row, metric, tags, displaced, now)
          acc
        end,
        :ok,
        state.table
      )
    end
  end

  defp displace_row(
         state,
         {{:finding, id}, %Finding{class: :condition, status: :active} = finding, _miss},
         metric,
         tags,
         displaced,
         now
       ) do
    if finding.metric == metric and finding.tags == tags and
         MapSet.member?(displaced, finding.kind) do
      clear(state, id, finding, now)
    end
  end

  defp displace_row(_state, _row, _metric, _tags, _displaced, _now), do: :ok

  ## Aggregate health

  defp refresh_status(state) do
    now = System.system_time(:second)
    active = findings(state.table)
    level = level(active)

    previous = lookup(state.table, :status)

    since =
      case previous do
        %{level: ^level, since: since} when since != nil -> since
        _other -> now
      end

    status = %{
      level: level,
      since: since,
      concern: active |> Enum.map(& &1.concern) |> Enum.max(fn -> 0.0 end),
      findings: active,
      metrics: metric_states(state),
      novelty: novelty_state(state),
      updated_at: now
    }

    :ets.insert(state.table, {:status, status})

    if previous != nil and previous.level != level do
      :telemetry.execute(
        [:mobius_smarts, :health, :level_changed],
        %{concern: status.concern},
        %{level: level, previous: previous.level, instance: state.name}
      )
    end

    status
  end

  # The ladder, from the design:
  # - :critical — a critical :approaching_limit (resource exhaustion
  #   has a date) is the device's own deadline.
  # - :degraded — any other critical condition, or correlated trouble
  #   (3+ warning-level METRICS — one physical event fans out across
  #   detectors on the same metric, and that is one cause, not three;
  #   issue #11).
  # - :watch — anything warning-level.
  defp level(active) do
    criticals = Enum.filter(active, &(&1.severity == :critical))

    warning_metrics =
      active
      |> Enum.filter(&(&1.severity == :warning))
      |> Enum.map(&{&1.metric, &1.tags})
      |> Enum.uniq()
      |> length()

    cond do
      Enum.any?(criticals, &(&1.kind == :approaching_limit)) -> :critical
      criticals != [] -> :degraded
      warning_metrics >= 3 -> :degraded
      warning_metrics > 0 -> :watch
      true -> :ok
    end
  end

  # Per-metric detection posture: which detector streams are armed
  # right now, and — while the baseline-gated ones aren't — where
  # baselining stands. Missingness (:silent / :reporting_gap) is
  # always on and carries no detector tag, so it isn't listed.
  defp metric_states(state) do
    for metric <- state.config.watch do
      key = Config.Metric.key(metric)
      baseline = lookup(state.table, {:baseline, key})

      seasonal = seasonal_state(state, key)

      if baseline do
        %{
          metric: metric.name,
          tags: metric.tags,
          detection: :active,
          detectors: armed_detectors(metric, baseline),
          seasonal: seasonal,
          learning: nil
        }
      else
        progress =
          lookup(state.table, {:learning, key}) ||
            %{reason: :no_data, windows: 0, needed: state.config.min_baseline_windows, seen: 0}

        detection = detection_state(progress)

        %{
          metric: metric.name,
          tags: metric.tags,
          detection: detection,
          detectors: armed_detectors(metric, nil),
          seasonal: seasonal,
          learning: learning_entry(detection, progress, state.config)
        }
      end
    end
  end

  # The cross-cutting seasonal posture: :off (not configured),
  # {:warming, ready_slots, slot_count} while the model learns the
  # cycle, :active once detection runs on residuals.
  defp seasonal_state(%{config: %{seasonality: nil}}, _key), do: :off

  defp seasonal_state(state, key) do
    case lookup(state.table, {:seasonal, key}) do
      nil ->
        {:warming, 0, expected_slots(state.config)}

      model ->
        if Seasonal.ready?(model) do
          :active
        else
          {ready, total} = Seasonal.progress(model)
          {:warming, ready, total}
        end
    end
  end

  defp expected_slots(config) do
    div(Config.seconds(config.seasonality), Config.seconds(config.resolution))
  end

  # An :unstable metric gets no ETA — that clock keeps resetting, so
  # an estimate would be a lie.
  defp learning_entry(:unstable, progress, _config), do: Map.put(progress, :eta_s, nil)
  defp learning_entry(_detection, progress, config), do: with_eta(progress, config)

  # No amount of waiting fits a baseline on data with nothing to model.
  defp detection_state(%{reason: reason}) when reason in [:no_dispersion, :zero_variance],
    do: :blocked

  # Abundant data that still will not settle: the metric's character
  # changes faster than min_baseline_windows of stability allows
  # (issue #13) — chronically unmonitorable by the chart stack, and
  # named as such instead of showing a forever-resetting countdown.
  defp detection_state(%{reason: reason, seen: seen, needed: needed})
       when reason in [:unsettled, :trending] and seen >= 2 * needed,
       do: :unstable

  defp detection_state(_progress), do: :learning

  defp with_eta(%{reason: reason} = progress, config)
       when reason in [:insufficient, :unsettled] do
    remaining = max(progress.needed - progress.windows, 0)
    Map.put(progress, :eta_s, remaining * Config.seconds(config.resolution))
  end

  defp with_eta(progress, _config), do: Map.put(progress, :eta_s, nil)

  defp armed_detectors(metric, baseline) do
    baseline_gated =
      cond do
        baseline == nil ->
          []

        # A constant metric: the chart stack is dark; departure
        # detection is what's armed (issue #6).
        baseline[:degenerate] ->
          [:departure]

        true ->
          [:jump, :shift, :drift] ++ if metric.histogram, do: [:shape], else: []
      end

    trend = if metric.ceiling || metric.floor, do: [:trend], else: []
    baseline_gated ++ trend ++ [:changepoint]
  end

  defp novelty_state(state) do
    cond do
      not Config.novelty?(state.config) -> :off
      lookup(state.table, :novelty) != nil -> :active
      true -> :learning
    end
  end

  ## Helpers

  defp lookup(board, key) do
    case :ets.lookup(board, key) do
      [{^key, value}] -> value
      [] -> nil
    end
  rescue
    ArgumentError -> raise_not_running(board)
  end

  defp fold_findings(board, fun) do
    :ets.foldl(fun, [], board)
  rescue
    ArgumentError -> raise_not_running(board)
  end

  @spec raise_not_running(atom()) :: no_return()
  defp raise_not_running(board) do
    raise ArgumentError,
          "no MobiusSmarts instance named #{inspect(board)} — is it started? " <>
            "(status/findings read the instance's ETS table)"
  end

  defp emit(state, event, %Finding{} = finding) do
    :telemetry.execute(
      [:mobius_smarts, :finding, event],
      %{concern: finding.concern},
      %{finding: finding, instance: state.name}
    )
  end

  defp severity_rank(:critical), do: 0
  defp severity_rank(:warning), do: 1
  defp severity_rank(:info), do: 2

  defp max_severity(a, b), do: Enum.min_by([a, b], &severity_rank/1)
end
