defmodule MobiusSmarts.WatcherTest do
  @moduledoc """
  Watcher-specific runtime behavior against a stub source: what feeds
  the cross-metric novelty vector, and the cadence/`:interval`
  calibration warning.

  Sync, like `MobiusSmartsTest`: real instances emit global telemetry
  that async board tests listen for.
  """
  use ExUnit.Case

  import ExUnit.CaptureLog

  alias MobiusSmarts.{Analysis, Board, Calibrate, Config, StubSource}

  # The deliberate stub cadence (60s windows) never matches the
  # millisecond tick interval these tests run at, so every test here
  # legitimately triggers the cadence-mismatch warning.
  @moduletag :capture_log

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

  defp instance_opts(ctx, watch) do
    name = :"watcher_test_#{ctx.test}"
    instance = :"stub_#{ctx.test}"

    on_exit(fn -> StubSource.clear(instance) end)

    config = [
      watch: watch,
      source: StubSource,
      mobius_instance: instance,
      interval: 25,
      sweep_interval: 60_000,
      min_baseline_windows: 60,
      clear_after: 2
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

  # Three metrics (novelty auto-enables at 3): "net" tracks "cpu"
  # tightly, "tmp" is independent. Returns the healthy histories and a
  # novelty model fitted on their aligned rows, exactly as the Sweeper
  # would fit it.
  defp correlated_trio(config_kw) do
    cpu = Enum.map(1..120, fn _ -> 25.0 + 10.0 * :rand.uniform() end)
    net = Enum.map(cpu, fn c -> 2.0 * c + noise(1.0) end)
    tmp = Enum.map(1..120, fn _ -> 50.0 + noise(1.0) end)

    ts = Enum.map(1..120, &(&1 * 60))
    lists = fn avg -> %{ts: ts, avg: avg, std: [], reports: []} end

    series = [
      {{"cpu", %{}}, lists.(cpu)},
      {{"net", %{}}, lists.(net)},
      {{"tmp", %{}}, lists.(tmp)}
    ]

    arl = config_kw |> Config.new!() |> Calibrate.for_config() |> Map.fetch!(:arl)
    {:ok, model} = Analysis.fit_novelty(series, arl)

    %{cpu: cpu, net: net, tmp: tmp, model: model}
  end

  describe "novelty vector hygiene" do
    # The broken-but-individually-unremarkable state: cpu's last value
    # sits high in its marginal range (34), net's last value low in
    # its (52, consistent with cpu ~26). Scored together they violate
    # the fitted correlation — which is only a real alarm if both
    # values describe the same moment.
    @stale_cpu_last 34.0
    @fresh_net_last 52.0
    @fresh_tmp_last 50.0

    test "a stale metric's hour-old value never feeds the novelty vector", ctx do
      seeded(201)
      {name, instance, config} = instance_opts(ctx, ["cpu", "net", "tmp"])
      %{cpu: cpu, net: net, tmp: tmp, model: model} = correlated_trio(config)
      now = System.system_time(:second)

      # Sanity: the part-stale vector WOULD alarm if it were scored.
      assert [%{kind: :novel_behavior}] =
               Analysis.novelty_candidates(model, [
                 @stale_cpu_last,
                 @fresh_net_last,
                 @fresh_tmp_last
               ])

      # cpu went quiet half an hour ago; net and tmp are live.
      StubSource.stage(instance, %{
        {"cpu", %{}} => windows(List.replace_at(cpu, -1, @stale_cpu_last), now - 1800),
        {"net", %{}} => windows(List.replace_at(net, -1, @fresh_net_last), now),
        {"tmp", %{}} => windows(List.replace_at(tmp, -1, @fresh_tmp_last), now)
      })

      start_supervised!({MobiusSmarts, name: name, config: config})
      Board.put_novelty(name, model)

      # The stale metric raises :silent — proof the ticks are flowing
      # and the stale branch ran.
      eventually(fn ->
        silent? =
          Enum.any?(
            MobiusSmarts.findings(name),
            &(&1.kind == :silent and &1.metric == "cpu")
          )

        if silent?, do: {:ok, :silent}, else: :retry
      end)

      # Several more ticks: cpu's hour-old "latest" must not be paired
      # with net/tmp's live values into a part-now, part-history vector.
      Process.sleep(250)
      refute Enum.any?(MobiusSmarts.findings(name), &(&1.kind == :novel_behavior))
    end

    test "live metrics are only scored when their latest windows align", ctx do
      seeded(202)
      {name, instance, config} = instance_opts(ctx, ["cpu", "net", "tmp"])
      %{cpu: cpu, net: net, tmp: tmp, model: model} = correlated_trio(config)
      now = System.system_time(:second)

      assert [%{kind: :novel_behavior}] =
               Analysis.novelty_candidates(model, [
                 @stale_cpu_last,
                 @fresh_net_last,
                 @fresh_tmp_last
               ])

      # cpu lags one 60s window behind net/tmp — still live (well
      # inside stale_after), but its latest window is not the same
      # moment the model's aligned rows describe.
      StubSource.stage(instance, %{
        {"cpu", %{}} => windows(List.replace_at(cpu, -1, @stale_cpu_last), now - 60),
        {"net", %{}} => windows(List.replace_at(net, -1, @fresh_net_last), now),
        {"tmp", %{}} => windows(List.replace_at(tmp, -1, @fresh_tmp_last), now)
      })

      start_supervised!({MobiusSmarts, name: name, config: config})
      Board.put_novelty(name, model)

      # Baselines fitting is proof the ticks are flowing.
      eventually(fn ->
        if MobiusSmarts.baseline(name, "net"), do: {:ok, :fitted}, else: :retry
      end)

      Process.sleep(250)
      refute Enum.any?(MobiusSmarts.findings(name), &(&1.kind == :novel_behavior))
    end
  end

  describe "cadence calibration warning" do
    test "warns once per metric when the measured cadence disagrees with :interval", ctx do
      seeded(203)
      {name, instance, config} = instance_opts(ctx, ["n7.cadence"])
      now = System.system_time(:second)

      # 60s-cadence windows against a 25ms tick interval: the
      # false-alarm budget was converted to windows via :interval, so
      # every threshold is miscalibrated by roughly the ratio.
      healthy = Enum.map(1..120, fn _ -> 40.0 + noise(0.5) end)
      StubSource.stage(instance, %{{"n7.cadence", %{}} => windows(healthy, now)})

      log =
        capture_log(fn ->
          start_supervised!({MobiusSmarts, name: name, config: config})

          eventually(fn ->
            if MobiusSmarts.baseline(name, "n7.cadence"), do: {:ok, :fitted}, else: :retry
          end)

          # Plenty more ticks — the warning must not repeat per tick.
          Process.sleep(300)
        end)

      assert log =~ "n7.cadence"
      assert log =~ "false-alarm budget is calibrated against :interval"

      warnings = log |> String.split("\n") |> Enum.count(&(&1 =~ "n7.cadence"))
      assert warnings == 1
    end
  end
end
