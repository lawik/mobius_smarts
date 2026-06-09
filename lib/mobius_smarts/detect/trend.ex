defmodule MobiusSmarts.Detect.Trend do
  @moduledoc """
  Which way is this metric heading, how fast, is it real, and when does
  it hit the wall? A time-to-impact — "tmpfs full in ~9 days" — is
  categorically more actionable than an anomaly score.

  Implements: Theil–Sen estimator, Mann–Kendall test.

  - `theil_sen/2` — to find the slope, draw a line through *every pair*
    of points and take the **median** of all those slopes. A median
    doesn't care about wild values, so up to ~29% of the points can be
    complete garbage — reboot spikes, sensor glitches — without moving
    the answer at all. Ordinary least squares would chase every spike.
  - `mann_kendall/2` — to decide whether the trend is *real* and not
    luck, just count: of all pairs of points, how many go up over time
    vs down? In random noise it's roughly a coin flip, so ups and downs
    nearly cancel. If nearly every pair goes up, that imbalance is
    astronomically unlikely by chance — the p-value quantifies exactly
    how unlikely. It only ever looks at *order*, which is also why
    spikes can't fool it.
  - `eta_to_threshold/3` — extend the robust slope until it crosses the
    ceiling (or floor): "disk full in ~9 days".

  ## Cost

  This is the one module in the stack implemented in plain Elixir
  rather than Nx, by measurement: on the non-compiling backends devices
  actually run, the pairwise tensor formulation paid a 35–350×
  interpretation penalty over a plain BEAM loop computing the same
  statistic (n = 1000, BinaryBackend: tensor Mann–Kendall 1.44 s vs
  4 ms as a loop vs 0.2 ms as a merge sort). As implemented:

  - `mann_kendall/2` counts inversions with a merge sort —
    `O(n log n)`, sub-millisecond even at a week of 5-minute windows
    (n = 2016).
  - `theil_sen/2` is inherently `O(n²)` pairs (exact Theil–Sen):
    ~100 ms at n = 1000 on a desktop BEAM and a few seconds at
    n ≈ 2000 — slower again on a device core, so schedule sweeps
    accordingly and aggregate to coarser windows for long horizons. A
    day of 5-minute windows (n = 288, ~10 ms) is plenty for a trend.
    Memory tracks the same `n(n − 1)/2`: all pair slopes are
    materialized as a BEAM list before the median — at n = 1440 (a day
    of minute windows) that is ~1M boxed floats, tens of MB of
    transient process heap. If long horizons on-device ever matter,
    sampled Theil–Sen or the repeated-median estimator bounds time and
    memory alike.

  Mann–Kendall's p-value assumes independent observations — under
  positive autocorrelation it is severely anticonservative (the
  Hamed–Rao correction is not implemented). See the calibration caveat
  in `MobiusSmarts.Detect`.
  """

  @type mann_kendall_result() :: %{
          s: integer(),
          var_s: float(),
          z: float(),
          p: float(),
          trend: :increasing | :decreasing | :none
        }

  @doc """
  Theil–Sen estimator: `%{slope: ..., intercept: ...}`.

  `timestamps` defaults to the sample index (`0, 1, 2, ...`), making the
  slope per-window; pass real UNIX timestamps to get per-second slope.
  Timestamps must be strictly increasing (validated). The intercept is
  the median of `value - slope * timestamp` (Conover's form), so
  `slope * t + intercept` predicts the series. Needs at least 2 points.

  ## Examples

      iex> alias MobiusSmarts.Detect.Trend
      iex> %{slope: slope} = Trend.theil_sen([1.0, 3.0, 5.0, 7.0, 9.0])
      iex> slope
      2.0

      iex> alias MobiusSmarts.Detect.Trend
      iex> # One wild outlier does not move the estimate
      iex> %{slope: slope} = Trend.theil_sen([1.0, 3.0, 500.0, 7.0, 9.0])
      iex> slope
      2.0
  """
  @spec theil_sen(Nx.Tensor.t() | [number()], Nx.Tensor.t() | [number()] | nil) :: %{
          slope: float(),
          intercept: float()
        }
  def theil_sen(values, timestamps \\ nil) do
    values = to_list(values)
    n = length(values)

    if n < 2 do
      raise ArgumentError, "Theil-Sen needs at least 2 points, got #{n}"
    end

    timestamps = resolve_timestamps(timestamps, n)
    validate_increasing!(timestamps)

    varr = List.to_tuple(values)
    tarr = List.to_tuple(timestamps)

    slopes =
      for i <- 0..(n - 2), j <- (i + 1)..(n - 1) do
        (elem(varr, j) - elem(varr, i)) / (elem(tarr, j) - elem(tarr, i))
      end

    slope = median(slopes)

    intercept =
      values
      |> Enum.zip(timestamps)
      |> Enum.map(fn {v, t} -> v - slope * t end)
      |> median()

    %{slope: slope, intercept: intercept}
  end

  @doc """
  Mann–Kendall trend test.

  Returns the S statistic (sum of signs of all pairwise differences,
  taken forward in time), its variance under the no-trend null
  hypothesis, the standardized z score (with the standard continuity
  correction), the two-sided p-value, and a verdict at significance
  level `:alpha` (default `0.05`).

  S is computed exactly in `O(n log n)` by inversion counting; ties
  contribute zero, handled via value multiplicities. The variance
  formula does not apply the tie correction; with float-valued
  metrics, exact ties are rare and the test is slightly conservative
  when they occur.

  ## Examples

      iex> alias MobiusSmarts.Detect.Trend
      iex> result = Trend.mann_kendall([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
      iex> {result.s, result.trend}
      {45, :increasing}
  """
  @spec mann_kendall(Nx.Tensor.t() | [number()], keyword()) :: mann_kendall_result()
  def mann_kendall(values, opts \\ []) do
    alpha = Keyword.get(opts, :alpha, 0.05)
    values = to_list(values)
    n = length(values)

    # S = concordant - discordant = pairs - ties - 2 * inversions,
    # where inversions are strictly-decreasing pairs in time order.
    pairs = div(n * (n - 1), 2)

    ties =
      values
      |> Enum.frequencies()
      |> Enum.reduce(0, fn {_v, t}, acc -> acc + div(t * (t - 1), 2) end)

    {_sorted, inversions} = sort_count(values)
    s = pairs - ties - 2 * inversions

    var_s = n * (n - 1) * (2 * n + 5) / 18

    z =
      cond do
        s > 0 -> (s - 1) / :math.sqrt(var_s)
        s < 0 -> (s + 1) / :math.sqrt(var_s)
        true -> 0.0
      end

    p = 2.0 * (1.0 - phi(abs(z)))

    trend =
      cond do
        p >= alpha -> :none
        s > 0 -> :increasing
        true -> :decreasing
      end

    %{s: s, var_s: var_s, z: z, p: p, trend: trend}
  end

  @doc """
  Project when a trending series crosses `threshold`, using the
  Theil–Sen slope over (`values`, `timestamps`).

  Returns `{:eta, seconds_from_last_timestamp}` when the series is
  moving toward the threshold, `:not_approaching` otherwise (flat,
  moving away, or already past in the wrong direction). Works for
  ceilings above the data (disk filling) and floors below it (battery
  draining) alike.

  The projection is anchored on the *fitted* value at the last
  timestamp, not the raw final sample, so one noisy final window
  cannot swing the ETA — the same robustness the slope already has.

  ## Examples

      iex> alias MobiusSmarts.Detect.Trend
      iex> # 1%/hour toward a 95% ceiling, currently at 70%
      iex> ts = Enum.map(0..24, &(&1 * 3600))
      iex> disk = Enum.map(0..24, &(46.0 + &1 * 1.0))
      iex> {:eta, seconds} = Trend.eta_to_threshold(disk, ts, 95.0)
      iex> round(seconds / 3600)
      25
  """
  @spec eta_to_threshold(
          Nx.Tensor.t() | [number()],
          Nx.Tensor.t() | [number()],
          number()
        ) :: {:eta, float()} | :not_approaching
  def eta_to_threshold(values, timestamps, threshold) do
    timestamps = to_list(timestamps)
    fit = theil_sen(values, timestamps)
    eta_from_fit(fit, List.last(timestamps), threshold)
  end

  @doc """
  `eta_to_threshold/3` for a precomputed Theil–Sen `fit` — when one
  fit serves several thresholds (ceiling and floor), compute it once
  with `theil_sen/2` and project each crossing from here, instead of
  paying the `O(n²)` fit per threshold.

  Anchors on the fitted value `slope * last_timestamp + intercept` and
  returns `{:eta, seconds_from_last_timestamp}` or `:not_approaching`,
  exactly as `eta_to_threshold/3` does.

  ## Examples

      iex> alias MobiusSmarts.Detect.Trend
      iex> fit = %{slope: 1.0 / 3600, intercept: 46.0}
      iex> {:eta, seconds} = Trend.eta_from_fit(fit, 24 * 3600, 95.0)
      iex> round(seconds / 3600)
      25
  """
  @spec eta_from_fit(%{slope: float(), intercept: float()}, number(), number()) ::
          {:eta, float()} | :not_approaching
  def eta_from_fit(%{slope: slope, intercept: intercept}, last_timestamp, threshold) do
    fitted = slope * last_timestamp + intercept
    gap = threshold - fitted

    if slope != 0.0 and gap * slope > 0 do
      {:eta, gap / slope}
    else
      :not_approaching
    end
  end

  # Merge sort counting strict inversions (left > right). Equal values
  # take from the left without counting, so ties contribute zero.
  defp sort_count([]), do: {[], 0}
  defp sort_count([x]), do: {[x], 0}

  defp sort_count(list) do
    {l, r} = Enum.split(list, div(length(list), 2))
    {ls, li} = sort_count(l)
    {rs, ri} = sort_count(r)
    {merged, mi} = merge_count(ls, rs, length(ls), [], 0)
    {merged, li + ri + mi}
  end

  defp merge_count([], r, _ln, acc, inv), do: {Enum.reverse(acc, r), inv}
  defp merge_count(l, [], _ln, acc, inv), do: {Enum.reverse(acc, l), inv}

  defp merge_count([lh | lt], [rh | _] = r, ln, acc, inv) when lh <= rh,
    do: merge_count(lt, r, ln - 1, [lh | acc], inv)

  defp merge_count(l, [rh | rt], ln, acc, inv),
    do: merge_count(l, rt, ln, [rh | acc], inv + ln)

  # Branch-free median: lower and upper middle coincide for odd length.
  defp median(list) do
    arr = list |> Enum.sort() |> List.to_tuple()
    k = tuple_size(arr)
    (elem(arr, div(k - 1, 2)) + elem(arr, div(k, 2))) / 2.0
  end

  # Standard normal CDF via erf.
  defp phi(x), do: 0.5 * (1.0 + :math.erf(x / :math.sqrt(2.0)))

  defp resolve_timestamps(nil, n), do: Enum.map(0..(n - 1), &(&1 * 1.0))
  defp resolve_timestamps(timestamps, _n), do: to_list(timestamps)

  # Duplicate timestamps would put 0-denominator slopes into the median
  # silently; fail loudly instead.
  defp validate_increasing!([a, b | _]) when b <= a do
    raise ArgumentError,
          "timestamps must be strictly increasing (duplicate or out-of-order " <>
            "timestamps make pairwise slopes undefined)"
  end

  defp validate_increasing!([_ | rest]), do: validate_increasing!(rest)
  defp validate_increasing!([]), do: :ok

  defp to_list(values) when is_list(values), do: Enum.map(values, &(&1 * 1.0))
  defp to_list(values), do: values |> Nx.as_type(:f64) |> Nx.to_flat_list()
end
