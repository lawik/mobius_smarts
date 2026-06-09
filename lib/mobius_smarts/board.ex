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

  alias MobiusSmarts.{Config, Finding}

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
    GenServer.call(board, {:put, {:baseline, key}, baseline})
  end

  @spec put_novelty(atom(), map()) :: :ok
  def put_novelty(board, model_map) do
    GenServer.call(board, {:put, :novelty, model_map})
  end

  @spec baseline(atom(), scope()) :: map() | nil
  def baseline(board, key), do: lookup(board, {:baseline, key})

  @spec novelty(atom()) :: map() | nil
  def novelty(board), do: lookup(board, :novelty)

  @spec status(atom()) :: map()
  def status(board) do
    lookup(board, :status) ||
      %{level: :ok, since: nil, concern: 0.0, findings: [], learning: []}
  end

  @spec findings(atom()) :: [Finding.t()]
  def findings(board) do
    :ets.foldl(
      fn
        {{:finding, _id}, %Finding{class: :condition, status: :active} = f, _miss}, acc ->
          [f | acc]

        _row, acc ->
          acc
      end,
      [],
      board
    )
    |> Enum.sort_by(&{severity_rank(&1.severity), -&1.concern})
  end

  @spec observations(atom(), pos_integer()) :: [Finding.t()]
  def observations(board, limit \\ 50) do
    :ets.foldl(
      fn
        {{:finding, _id}, %Finding{class: :observation} = f, _miss}, acc -> [f | acc]
        _row, acc -> acc
      end,
      [],
      board
    )
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

    miss_uncovered(state, scope, covered_kinds, MapSet.new(confirmed_ids), now)
    refresh_status(state)
    {:reply, :ok, state}
  end

  def handle_call({:put, key, value}, _from, state) do
    :ets.insert(state.table, {key, value})
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
      cleared = %{finding | status: :cleared, cleared_at: now}
      :ets.insert(state.table, {{:finding, id}, cleared, 0})
      emit(state, :cleared, cleared)
    else
      :ets.insert(state.table, {{:finding, id}, finding, misses})
    end
  end

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
      learning: learning(state),
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
  #   (3+ concurrent warnings).
  # - :watch — anything warning-level.
  defp level(active) do
    criticals = Enum.filter(active, &(&1.severity == :critical))
    warnings = Enum.count(active, &(&1.severity == :warning))

    cond do
      Enum.any?(criticals, &(&1.kind == :approaching_limit)) -> :critical
      criticals != [] -> :degraded
      warnings >= 3 -> :degraded
      warnings > 0 -> :watch
      true -> :ok
    end
  end

  defp learning(state) do
    for metric <- state.config.watch,
        key = Config.Metric.key(metric),
        :ets.lookup(state.table, {:baseline, key}) == [] do
      metric.name
    end
  end

  ## Helpers

  defp lookup(board, key) do
    case :ets.lookup(board, key) do
      [{^key, value}] -> value
      [] -> nil
    end
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
