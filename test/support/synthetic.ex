defmodule MobiusSmarts.Synthetic do
  @moduledoc """
  Deterministic synthetic summary-window datasets for the replay
  harness (issue #1).

  `series/1` builds the `Mobius.Data.summary_windows/3` window shape
  from a scenario: a seed, a cadence, and a list of segments. Each
  window's `average`/`std_dev` are computed from actually-drawn
  per-report values with the same math Mobius uses, so the windows
  are statistically faithful subgroups, not idealized ones.

  Segments are maps with `:minutes` plus a level shape:

      %{minutes: 120, level: 100.0}                  # steady
      %{minutes: 20, from: 79_000.0, to: 90_000.0}   # linear ramp
      %{minutes: 30, level: 0.0, gap: true}          # no windows at all

  Per-segment options:

  - `:sigma` — per-report noise standard deviation (default `1.0`)
  - `:distribution` — `:normal` (default), `:constant` (no noise at
    all — exact zero variance), or `{:zero_inflated, p_active}` (each
    report is `0.0` except with probability `p_active`, the
    run-queue shape)

  Series-level options:

  - `:seed` (required) — same seed, same dataset
  - `:start` (default `1_750_000_000`) — unix second the first window
    *ends* at (windows are end-stamped, like Mobius's)
  - `:resolution` (default `{1, :minute}`) — window cadence
  - `:reports` (default `18`) — reports per window
  - `:daily_cycle` — `%{amplitude: a}` adds `a·sin(2πt/period)` to
    every level; `:period` (seconds) defaults to a day
  - `:wander` — `%{phi: p, sigma: s}` adds an AR(1) random walk to the
    level across windows: autocorrelated data that violates the
    independence assumption behind the false-alarm budget (issue #12)
  """

  alias MobiusSmarts.Config

  @day_s 86_400

  @spec series(keyword()) :: [
          %{timestamp: integer(), average: float(), std_dev: float(), reports: pos_integer()}
        ]
  def series(opts) do
    seed = Keyword.fetch!(opts, :seed)
    :rand.seed(:exsss, {seed, seed * 7919, seed * 104_729})

    start = Keyword.get(opts, :start, 1_750_000_000)
    resolution_s = Config.seconds(Keyword.get(opts, :resolution, {1, :minute}))
    reports = Keyword.get(opts, :reports, 18)
    cycle = Keyword.get(opts, :daily_cycle)
    wander = Keyword.get(opts, :wander)
    segments = Keyword.fetch!(opts, :segments)

    {windows, _index, _wander_state} =
      Enum.reduce(segments, {[], 0, 0.0}, fn segment, {acc, index, wstate} ->
        count = window_count(segment, resolution_s)

        if segment[:gap] do
          {acc, index + count, wstate}
        else
          {rows, wstate} =
            Enum.map_reduce(0..(count - 1), wstate, fn i, wstate ->
              ts = start + (index + i) * resolution_s
              wstate = advance_wander(wstate, wander)
              level = segment_level(segment, i, count) + cycle_at(cycle, ts) + wstate
              {window(ts, level, segment, reports), wstate}
            end)

          {acc ++ rows, index + count, wstate}
        end
      end)

    windows
  end

  defp window_count(segment, resolution_s) do
    max(div(Map.fetch!(segment, :minutes) * 60, resolution_s), 1)
  end

  defp segment_level(%{from: from, to: to}, i, count) when count > 1 do
    from + (to - from) * i / (count - 1)
  end

  defp segment_level(%{from: from}, _i, _count), do: from
  defp segment_level(%{level: level}, _i, _count), do: level

  defp cycle_at(nil, _ts), do: 0.0

  defp cycle_at(%{amplitude: amplitude} = cycle, ts) do
    period = Map.get(cycle, :period, @day_s)
    amplitude * :math.sin(2.0 * :math.pi() * ts / period)
  end

  defp advance_wander(_state, nil), do: 0.0

  defp advance_wander(state, %{phi: phi, sigma: sigma}) do
    phi * state + sigma * :rand.normal()
  end

  defp window(ts, level, segment, reports) do
    values = Enum.map(1..reports, fn _i -> draw(level, segment) end)
    n = length(values)
    sum = Enum.sum(values)
    avg = sum / n

    std =
      if n == 1 do
        0.0
      else
        sum_sqrd = values |> Enum.map(&(&1 * &1)) |> Enum.sum()
        :math.sqrt(max(0.0, (sum_sqrd - sum * sum / n) / (n - 1)))
      end

    %{timestamp: ts, average: avg, std_dev: std, reports: n}
  end

  defp draw(level, segment) do
    sigma = Map.get(segment, :sigma, 1.0)

    case Map.get(segment, :distribution, :normal) do
      :normal ->
        level + sigma * :rand.normal()

      :constant ->
        level

      {:zero_inflated, p} ->
        if :rand.uniform() < p, do: abs(level + sigma * :rand.normal()), else: 0.0
    end
  end
end
