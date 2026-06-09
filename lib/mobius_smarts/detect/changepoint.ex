defmodule MobiusSmarts.Detect.Changepoint do
  @moduledoc """
  Retrospective change-point detection: looking back over the stored
  history, when did this metric change character, and how many times?
  The answer is a list of timestamps to lay alongside deploys, config
  pushes, and the event log.

  Implements: binary segmentation with squared-error cost and a
  BIC-style penalty.

  Ask "if I had to cut this series into two segments, where would the
  cut make each half most internally consistent?" Try every possible
  cut — the prefix-sum trick makes trying all of them as cheap as one
  pass — and measure how much the best cut reduces the total wiggle
  (variance within each half). The catch the penalty solves: *any* cut
  reduces wiggle a little, even on pure noise, the same way a
  conspiracy theory always explains a bit more than the boring truth.
  So a cut is only accepted if its improvement beats a skepticism
  threshold scaled to the series' own noise level — and if accepted, we
  recurse: cut each half again, until no cut anywhere earns its keep.

  One self-defeating trap the implementation dodges: the noise estimate
  powering the skepticism threshold is computed from *median*
  window-to-window differences, so the very changes being hunted (a
  handful of big differences) can't inflate the noise estimate and hide
  themselves. The mirror-image trap — quantized gauges (ADC steps,
  0.5-degree reporting, idle stretches) where *most* consecutive windows
  are exactly equal, collapsing the median to zero — falls back to the
  plain standard deviation of the differences, so step-quantized but
  stable series don't shatter into false segments. The BIC-style
  penalty still presumes roughly continuous-valued noise; for heavily
  quantized series prefer an explicit `:penalty` scaled to the
  quantization step.

  **Relationship to `MobiusSmarts.Detect.Drift` and
  `MobiusSmarts.Detect.Shift`:** those run *live* and answer "is
  something happening right now?"; this runs as a periodic sweep and
  answers "what happened, precisely, in hindsight" — with better
  timestamps, because it gets to see both sides of each change.
  """

  @doc """
  Detect change points in a series of window means.

  Returns ascending indices, each the first window of a new segment.
  Empty list when the series is statistically homogeneous.

  Options:

  - `:min_size` — minimum segment length in windows, default `5`.
    Guards against fitting segments to single spikes.
  - `:penalty` — cost reduction a split must exceed, default
    `:bic` (`2·sigma²·ln(n)` with robust sigma). Raise it for fewer,
    stronger change points.
  - `:max_changepoints` — keep only the N **strongest** change points
    (largest cost reduction), default `:infinity`. Series shorter than
    2 windows return `[]`.

  ## Examples

      iex> alias MobiusSmarts.Detect.Changepoint
      iex> series = List.duplicate(10.0, 30) ++ List.duplicate(20.0, 30)
      iex> Changepoint.detect(series)
      [30]

      iex> alias MobiusSmarts.Detect.Changepoint
      iex> Changepoint.detect(List.duplicate(10.0, 60))
      []
  """
  @spec detect(Nx.Tensor.t() | [number()], keyword()) :: [non_neg_integer()]
  def detect(values, opts \\ [])

  def detect(values, _opts) when is_list(values) and length(values) < 2, do: []

  def detect(values, opts) do
    values = to_f64(values)
    n = Nx.size(values)
    min_size = Keyword.get(opts, :min_size, 5)
    max_changepoints = Keyword.get(opts, :max_changepoints, :infinity)

    if n < 2 do
      []
    else
      penalty =
        case Keyword.get(opts, :penalty, :bic) do
          :bic -> 2.0 * robust_variance(values) * :math.log(n)
          number when is_number(number) -> number * 1.0
        end

      segment(values, 0, penalty, min_size)
      |> cap(max_changepoints)
      |> Enum.map(&elem(&1, 0))
      |> Enum.sort()
    end
  end

  # Recursive binary segmentation. `offset` maps local split indices
  # back to positions in the original series. Returns {index, gain}
  # pairs so the cap can keep the strongest changes, not the earliest.
  defp segment(values, offset, penalty, min_size) do
    n = Nx.size(values)

    if n < 2 * min_size do
      []
    else
      case best_split(values, min_size) do
        {gain, tau} when gain > penalty ->
          left = Nx.slice(values, [0], [tau])
          right = Nx.slice(values, [tau], [n - tau])

          segment(left, offset, penalty, min_size) ++
            [{offset + tau, gain}] ++
            segment(right, offset + tau, penalty, min_size)

        _ ->
          []
      end
    end
  end

  # Scan all split points at once via prefix sums:
  # SSE(segment) = sum(x²) - sum(x)²/len, so the cost of every
  # (left, right) pair is elementwise arithmetic over the cumulative
  # sums. Returns {gain, tau}: the SSE reduction of the best split and
  # the first index of the right segment.
  defp best_split(values, min_size) do
    n = Nx.size(values)
    s1 = Nx.cumulative_sum(values)
    s2 = Nx.cumulative_sum(Nx.pow(values, 2))

    total1 = Nx.to_number(s1[-1])
    total2 = Nx.to_number(s2[-1])
    sse_total = total2 - total1 * total1 / n

    # Split tau in 1..n-1: left = values[0, tau), sums are s1/s2 at tau - 1.
    left_sum = Nx.slice(s1, [0], [n - 1])
    left_sq = Nx.slice(s2, [0], [n - 1])
    left_len = Nx.add(Nx.iota({n - 1}, type: :f64), 1.0)
    right_len = Nx.subtract(n * 1.0, left_len)

    sse_left = Nx.subtract(left_sq, Nx.divide(Nx.pow(left_sum, 2), left_len))

    sse_right =
      total2
      |> Nx.subtract(left_sq)
      |> Nx.subtract(Nx.divide(Nx.pow(Nx.subtract(total1, left_sum), 2), right_len))

    valid =
      Nx.logical_and(
        Nx.greater_equal(left_len, min_size),
        Nx.greater_equal(right_len, min_size)
      )

    cost =
      valid
      |> Nx.select(Nx.add(sse_left, sse_right), Nx.Constants.infinity({:f, 64}))

    best = Nx.to_number(Nx.argmin(cost))
    best_cost = Nx.to_number(cost[best])

    {sse_total - best_cost, best + 1}
  end

  # Robust noise variance from first differences: changes contribute a
  # handful of outlier diffs, which the median ignores. For Gaussian
  # noise, sigma = 1.4826 · MAD(diff) / sqrt(2).
  defp robust_variance(values) do
    diffs = Nx.diff(values)
    med = median(diffs)
    mad = diffs |> Nx.subtract(med) |> Nx.abs() |> median()

    sigma =
      if mad > 0.0 do
        1.4826 * mad / :math.sqrt(2.0)
      else
        # Quantized series: more than half of consecutive windows equal
        # collapses the MAD to zero, and a 1e-12 penalty floor would
        # accept every microscopic split. Fall back to the plain (less
        # robust, but nonzero) sd of the diffs.
        sd = diffs |> Nx.standard_deviation() |> Nx.to_number()
        sd / :math.sqrt(2.0)
      end

    # A perfectly noiseless series still has sigma 0; keep the penalty
    # positive so detect/2 stays well-defined.
    max(sigma * sigma, 1.0e-12)
  end

  defp median(tensor) do
    n = Nx.size(tensor)
    sorted = Nx.sort(tensor)
    mid = div(n, 2)

    if rem(n, 2) == 1 do
      Nx.to_number(sorted[mid])
    else
      (Nx.to_number(sorted[mid - 1]) + Nx.to_number(sorted[mid])) / 2.0
    end
  end

  # Keep the strongest changes by gain; the caller re-sorts by index.
  defp cap(changepoints, :infinity), do: changepoints

  defp cap(changepoints, max) do
    changepoints
    |> Enum.sort_by(&elem(&1, 1), :desc)
    |> Enum.take(max)
  end

  defp to_f64(values) when is_list(values), do: Nx.tensor(values, type: :f64)
  defp to_f64(values), do: Nx.as_type(values, :f64)
end
