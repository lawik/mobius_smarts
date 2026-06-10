defmodule MobiusSmartsTest do
  @moduledoc """
  End-to-end runtime test: a MobiusSmarts instance against a stub data
  source — staged summary windows in, findings and health levels out,
  driven by real ticks.
  """
  use ExUnit.Case

  alias MobiusSmarts.StubSource

  # Keep incidental runtime logging (e.g. a tick racing a re-stage)
  # out of passing-test output.
  @moduletag :capture_log

  doctest MobiusSmarts

  defp seeded(seed), do: :rand.seed(:exsss, {seed, seed * 7, seed * 13})
  defp noise(sigma), do: sigma * :rand.normal()

  defp windows(values, now) do
    n = length(values)

    values
    |> Enum.with_index()
    |> Enum.map(fn {avg, i} ->
      %{timestamp: now - (n - i) * 60, average: avg, std_dev: 1.0, reports: 30}
    end)
  end

  defp start_instance(ctx, watch) do
    name = :"smarts_#{ctx.test}"
    instance = :"stub_#{ctx.test}"

    on_exit(fn -> StubSource.clear(instance) end)

    config = [
      watch: watch,
      source: StubSource,
      mobius_instance: instance,
      # Staged windows are 60s apart; gap detection and staleness key
      # off :resolution, so it must match. :interval stays a fast test
      # clock — it is scheduling-only.
      resolution: {1, :minute},
      false_alarm_every: {1, :week},
      interval: 25,
      sweep_interval: 60_000,
      min_baseline_windows: 60,
      clear_after: 2,
      novelty: false
    ]

    {name, instance, config}
  end

  defp eventually(fun, attempts \\ 80) do
    case fun.() do
      {:ok, value} ->
        value

      :retry when attempts > 0 ->
        Process.sleep(25)
        eventually(fun, attempts - 1)

      :retry ->
        flunk("condition not reached")
    end
  end

  test "learning -> ok -> drift finding -> recovery, end to end", ctx do
    seeded(101)
    {name, instance, config} = start_instance(ctx, ["mem.pct"])
    now = System.system_time(:second)

    healthy = Enum.map(1..120, fn _ -> 40.0 + noise(0.5) end)
    StubSource.stage(instance, %{{"mem.pct", %{}} => windows(healthy, now)})

    start_supervised!({MobiusSmarts, name: name, config: config})

    # Learning resolves into a baseline fitted from the staged history.
    baseline =
      eventually(fn ->
        case MobiusSmarts.baseline(name, "mem.pct") do
          nil -> :retry
          baseline -> {:ok, baseline}
        end
      end)

    assert_in_delta baseline.target, 40.0, 0.5
    assert %{level: :ok, learning: []} = MobiusSmarts.status(name)

    # The metric shifts; ticks re-scan and confirm a drift.
    shifted =
      Enum.map(1..90, fn _ -> 40.0 + noise(0.5) end) ++
        Enum.map(1..30, fn _ -> 43.0 + noise(0.5) end)

    StubSource.stage(instance, %{{"mem.pct", %{}} => windows(shifted, now)})

    finding =
      eventually(fn ->
        case MobiusSmarts.findings(name) do
          [] -> :retry
          [finding | _rest] -> {:ok, finding}
        end
      end)

    assert finding.kind in [:drifting_up, :shifted_up, :jumped]
    assert finding.metric == "mem.pct"
    assert MobiusSmarts.status(name).level != :ok

    # Health recovers once the data does (hysteresis: clear_after misses).
    recovered = Enum.map(1..120, fn _ -> 40.0 + noise(0.5) end)
    StubSource.stage(instance, %{{"mem.pct", %{}} => windows(recovered, now)})

    eventually(fn ->
      if MobiusSmarts.findings(name) == [], do: {:ok, :recovered}, else: :retry
    end)

    assert %{level: :ok} = MobiusSmarts.status(name)
  end

  test "a metric going silent raises :silent; gaps are observations", ctx do
    seeded(102)
    {name, instance, config} = start_instance(ctx, ["sensor.hz"])
    now = System.system_time(:second)

    # Last window half an hour ago: stale against a 60s cadence.
    healthy = Enum.map(1..80, fn _ -> 10.0 + noise(0.3) end)
    stale = windows(healthy, now - 1800)
    StubSource.stage(instance, %{{"sensor.hz", %{}} => stale})

    start_supervised!({MobiusSmarts, name: name, config: config})

    finding =
      eventually(fn ->
        case MobiusSmarts.findings(name) do
          [] -> :retry
          [finding | _rest] -> {:ok, finding}
        end
      end)

    assert finding.kind == :silent
    assert finding.message =~ "no windows since"
  end

  test "explicit config and app-env config produce independent instances", ctx do
    seeded(103)
    {name, instance, config} = start_instance(ctx, ["a.b"])
    now = System.system_time(:second)

    StubSource.stage(instance, %{
      {"a.b", %{}} => windows(Enum.map(1..80, fn _ -> 5.0 + noise(0.1) end), now)
    })

    start_supervised!({MobiusSmarts, name: name, config: config})

    eventually(fn ->
      if MobiusSmarts.baseline(name, "a.b"), do: {:ok, :fitted}, else: :retry
    end)

    # The named instance answers; the default instance name is untouched.
    assert %{level: :ok} = MobiusSmarts.status(name)
    refute Process.whereis(MobiusSmarts.Supervisor)
  end
end
