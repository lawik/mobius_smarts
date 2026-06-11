defmodule MobiusSmarts.Replay do
  @moduledoc """
  Drives the tick-detection pipeline over a synthetic dataset with a
  virtual clock (issue #1): the Watcher's per-tick walk — slice,
  gaps, active segment, fit-or-fetch baseline, tick candidates —
  re-run deterministically against a real `MobiusSmarts.Board`, so
  scenarios assert on actual finding lifecycle (raised / escalated /
  cleared via telemetry), not detector internals.

  This mirrors `MobiusSmarts.Watcher.watch_metric/3` rather than
  running it: the real Watcher reads the wall clock, so it cannot
  sweep weeks of virtual time. Keep the two in sync when the tick
  walk changes.

      result =
        MobiusSmarts.Replay.run(windows,
          config: [false_alarm_every: {1, :day}],
          tick_every: {5, :minute}
        )

      result.raised      # findings raised, in order
      result.findings    # active conditions at the end
      result.status      # Board status at the end
      result.baseline    # fitted baseline (or nil)
  """

  alias MobiusSmarts.{Analysis, Board, Calibrate, Config, Source}

  @metric "synthetic"

  # Mirror of the Watcher's covered tick kinds.
  @tick_kinds [
    :jumped,
    :spiked,
    :wobbling,
    :flatlined,
    :departed,
    :shifted_up,
    :shifted_down,
    :drifting_up,
    :drifting_down,
    :silent,
    :reporting_gap
  ]

  def run(windows, opts \\ []) do
    config =
      [watch: [@metric], resolution: {1, :minute}, false_alarm_every: {1, :week}]
      |> Keyword.merge(Keyword.get(opts, :config, []))
      |> Config.new!()

    calib = Calibrate.for_config(config)
    key = {@metric, %{}}

    name = :"replay_#{System.unique_integer([:positive])}"
    {:ok, _board} = Board.start_link(name: name, config: config)

    handler = "replay-#{name}"

    :telemetry.attach_many(
      handler,
      [
        [:mobius_smarts, :finding, :raised],
        [:mobius_smarts, :finding, :escalated],
        [:mobius_smarts, :finding, :cleared]
      ],
      &__MODULE__.forward_telemetry/4,
      {self(), name}
    )

    timestamps = Enum.map(windows, & &1.timestamp)
    from = Keyword.get(opts, :from, List.first(timestamps))
    until = Keyword.get(opts, :until, List.last(timestamps))
    tick_s = Config.seconds(Keyword.get(opts, :tick_every, config.resolution))

    for now <- :lists.seq(from + tick_s, until, tick_s) do
      tick(name, key, windows, now, config, calib)
    end

    :telemetry.detach(handler)
    events = drain_events([])

    %{
      events: events,
      raised: for({:raised, finding} <- events, do: finding),
      cleared: for({:cleared, finding} <- events, do: finding),
      status: Board.status(name),
      findings: Board.findings(name),
      baseline: Board.baseline(name, key),
      board: name
    }
  end

  # Telemetry handlers are global: only forward events from THIS
  # replay's board, or concurrently running tests pollute each other's
  # event streams.
  @doc false
  def forward_telemetry([:mobius_smarts, :finding, event], _measurements, meta, {parent, board}) do
    if meta.instance == board do
      send(parent, {:replay_event, event, meta.finding})
    end
  end

  defp tick(name, key, windows, now, config, calib) do
    window_s = Config.seconds(config.analysis_window)
    resolution_s = Config.seconds(config.resolution)
    slice = Enum.filter(windows, &(&1.timestamp > now - window_s and &1.timestamp <= now))

    case Source.from_summary_windows(slice) do
      :empty ->
        Board.put_learning(name, key, %{
          reason: :no_data,
          windows: 0,
          needed: config.min_baseline_windows,
          seen: 0
        })

        Board.report(name, key, @tick_kinds, [Analysis.silent_candidate(nil, now)])

      series ->
        lists = Analysis.to_lists(series)
        gaps = Analysis.gaps(lists.ts, config.gap_factor, resolution_s)
        segment = Analysis.active_segment(lists, gaps)
        last_ts = List.last(segment.ts)
        stale? = now - last_ts > config.gap_factor * resolution_s

        candidates =
          Analysis.gap_candidates(gaps) ++
            if stale? do
              [Analysis.silent_candidate(last_ts, now)]
            else
              detector_candidates(name, key, segment, now, config, calib)
            end

        if Enum.any?(candidates, &(&1.kind == :baseline_stale)) do
          Board.drop_baseline(name, key)
        end

        Board.report(name, key, @tick_kinds, candidates)
    end
  end

  defp detector_candidates(name, key, segment, now, config, calib) do
    case Board.baseline(name, key) || fit_baseline(name, key, segment, now, config) do
      nil ->
        []

      baseline ->
        # Mirrors the Watcher: only windows after the fit horizon (#2),
        # departure detection for degenerate constants (#6).
        fresh = Analysis.since(segment, baseline.to)

        if baseline[:degenerate] do
          Analysis.departure_candidates(fresh, baseline, config.min_baseline_windows)
        else
          Analysis.tick_candidates(fresh, baseline, calib, config)
        end
    end
  end

  defp fit_baseline(name, key, segment, now, config) do
    case Analysis.fit_baseline(segment, min_windows: config.min_baseline_windows, now: now) do
      {:ok, baseline} ->
        Board.put_baseline(name, key, baseline)
        baseline

      {:error, progress} ->
        Board.put_learning(name, key, progress)
        nil
    end
  end

  defp drain_events(acc) do
    receive do
      {:replay_event, event, finding} -> drain_events([{event, finding} | acc])
    after
      0 -> Enum.reverse(acc)
    end
  end
end
