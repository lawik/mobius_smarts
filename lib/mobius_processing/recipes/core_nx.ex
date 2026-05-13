defmodule MobiusProcessing.Recipes.CoreNx do
  @moduledoc """
  Recipes you can run on Mobius data using only Nx — no Scholar, no
  NxSignal. Each function is framed around a question you might ask of
  the kinds of metrics a Nerves device tends to emit; the data names
  are illustrative.

  All functions take an already-pulled tensor. See `MobiusProcessing.Source`
  for the long-format batch and `MobiusProcessing.Tensor.from_column/1` for
  `{tensor, validity}`.

  ## Is the CPU running hot?

  Short spikes hide in a plain mean. `ran_hot?/2` takes a window-max so
  a brief peak still counts.

  ## Is memory leaking?

  Memory pressure rarely jumps — it creeps. `mem_slope_per_day/1` fits a
  line over the window and reports how many percentage points of RAM the
  device gains per day.

  ## How fast is the disk filling?

  Same `least_squares` pattern as the memory recipe — `disk_days_to_threshold/2`
  projects when the disk crosses a given percentage.

  ## Turning a monotonic counter into a rate

  Counters only go up (until the interface restarts). `mean_rate/1`
  first-differences, masks counter resets, and returns the average rate
  per sample.

  ## What does the distribution look like?

  `value_histogram/4` is the general-purpose "pick a low bound, pick a
  width, get a count tensor" — usable for any gauge.

  ## Is the latest sample anomalous?

  `z_score/2` reports how many stddevs from the window mean a new sample
  sits. `anomalous?/3` lifts that into a boolean.

  ## How loaded is the device, smoothed at different timescales?

  `rolling_means/2` returns a map from window size to the smoothed tensor.

  ## Null-aware reductions

  When the validity tensor returned by `MobiusProcessing.Tensor.from_column/1`
  is non-nil, use `mean_with_validity/2` to keep gaps from skewing the answer.
  """

  @doc """
  Returns `true` if any `window`-sample-wide window of `temps` had a
  maximum above `threshold`.

  ## Options

  - `:window` — window size in samples. Defaults to `300`.
  - `:threshold` — value to compare against. Defaults to `75.0`.

  ## Examples

      iex> temps = Nx.tensor([45.0, 50.0, 76.0, 60.0, 55.0])
      iex> MobiusProcessing.Recipes.CoreNx.ran_hot?(temps, window: 2, threshold: 70.0)
      true

      iex> temps = Nx.tensor([45.0, 50.0, 60.0, 65.0])
      iex> MobiusProcessing.Recipes.CoreNx.ran_hot?(temps, window: 2, threshold: 70.0)
      false
  """
  @spec ran_hot?(Nx.Tensor.t(), keyword()) :: boolean()
  def ran_hot?(temps, opts \\ []) do
    window = Keyword.get(opts, :window, 300)
    threshold = Keyword.get(opts, :threshold, 75.0)

    temps
    |> Nx.window_max({window})
    |> Nx.greater(threshold)
    |> Nx.any()
    |> Nx.to_number()
    |> Kernel.==(1)
  end

  @doc """
  Fits `values ≈ a + b·t` and returns the slope scaled to percentage
  points per day, assuming `values` is sampled at 1Hz.

  ## Examples

      iex> # 100 samples, climbing 1.0%/s — implies 86_400%/day
      iex> mem = Nx.add(50.0, Nx.iota({100}, type: :f32))
      iex> slope = MobiusProcessing.Recipes.CoreNx.mem_slope_per_day(mem)
      iex> Float.round(slope, 0)
      86400.0
  """
  @spec mem_slope_per_day(Nx.Tensor.t()) :: float()
  def mem_slope_per_day(values) do
    slope_per_second(values) * 86_400.0
  end

  @doc """
  Linear projection of how many sample-intervals until `values` crosses
  `threshold`, given the trend over the window. Returns `:infinity` if
  the slope is non-positive (the metric isn't climbing toward the
  threshold).

  The natural unit is whatever `values` is sampled in. Pass daily samples
  and the result is in days; pass per-second samples and it's in seconds.

  ## Examples

      iex> disk = Nx.tensor([50.0, 55.0, 60.0, 65.0, 70.0])
      iex> MobiusProcessing.Recipes.CoreNx.disk_days_to_threshold(disk, 95.0)
      ...> |> Float.round(1)
      5.0

      iex> declining = Nx.tensor([60.0, 55.0, 50.0, 45.0])
      iex> MobiusProcessing.Recipes.CoreNx.disk_days_to_threshold(declining, 95.0)
      :infinity
  """
  @spec disk_days_to_threshold(Nx.Tensor.t(), number()) :: float() | :infinity
  def disk_days_to_threshold(values, threshold) do
    slope = slope_per_second(values)
    current = Nx.to_number(values[-1])

    if slope > 0 do
      (threshold - current) / slope
    else
      :infinity
    end
  end

  @doc """
  First-differences a monotonic counter and returns the mean per-sample
  rate, masking out negative diffs (counter resets).

  ## Examples

      iex> counter = Nx.tensor([10, 20, 30], type: :s64)
      iex> MobiusProcessing.Recipes.CoreNx.mean_rate(counter) |> Float.round(2)
      10.0

      iex> # A reset between sample 2 and 3 — negative diff is masked
      iex> counter_with_reset = Nx.tensor([10, 20, 5, 25], type: :s64)
      iex> MobiusProcessing.Recipes.CoreNx.mean_rate(counter_with_reset)
      ...> |> Float.round(2)
      15.0
  """
  @spec mean_rate(Nx.Tensor.t()) :: float()
  def mean_rate(counter) do
    rate = counter |> Nx.diff() |> Nx.as_type(:f32)
    valid = Nx.greater_equal(rate, 0)
    clean = Nx.select(valid, rate, 0.0)
    valid_f = Nx.as_type(valid, :f32)

    Nx.divide(Nx.sum(clean), Nx.sum(valid_f)) |> Nx.to_number()
  end

  @doc """
  Bins `values` into `n_bins` equal-width buckets starting at `lo`, each
  `width` wide. Values below `lo` go into bin 0; values above the last
  bucket go into bin `n_bins - 1`.

  Standing in for `Nx.bincount`, which Nx doesn't currently expose.

  ## Examples

      iex> temps = Nx.tensor([20.5, 21.3, 25.0, 30.1, 25.7])
      iex> hist = MobiusProcessing.Recipes.CoreNx.value_histogram(temps, 20.0, 5.0, 4)
      iex> Nx.to_flat_list(hist)
      [2, 2, 1, 0]
  """
  @spec value_histogram(Nx.Tensor.t(), number(), number(), pos_integer()) :: Nx.Tensor.t()
  def value_histogram(values, lo, width, n_bins) do
    bucket =
      values
      |> Nx.subtract(lo)
      |> Nx.divide(width)
      |> Nx.floor()
      |> Nx.clip(0, n_bins - 1)
      |> Nx.as_type(:s32)

    bincount(bucket, n_bins)
  end

  @doc """
  Z-score of a single observation against the mean and population stddev
  of a window tensor.

  ## Examples

      iex> window = Nx.tensor([50.0, 51.0, 49.0, 50.5, 49.5])
      iex> z = MobiusProcessing.Recipes.CoreNx.z_score(window, 55.0)
      iex> Float.round(z, 2)
      7.07
  """
  @spec z_score(Nx.Tensor.t(), number()) :: float()
  def z_score(window, latest) do
    mu = window |> Nx.mean() |> Nx.to_number()
    sigma = window |> Nx.standard_deviation() |> Nx.to_number()
    (latest - mu) / sigma
  end

  @doc """
  `true` when the absolute z-score of `latest` against `window` exceeds
  `k` (default `3`).

  ## Examples

      iex> window = Nx.tensor([50.0, 51.0, 49.0, 50.5, 49.5])
      iex> MobiusProcessing.Recipes.CoreNx.anomalous?(window, 55.0)
      true
      iex> MobiusProcessing.Recipes.CoreNx.anomalous?(window, 50.2)
      false
  """
  @spec anomalous?(Nx.Tensor.t(), number(), number()) :: boolean()
  def anomalous?(window, latest, k \\ 3.0) do
    abs(z_score(window, latest)) > k
  end

  @doc """
  Returns `%{size => window_mean_tensor}` for each window size given.
  The classic "1-/5-/15-minute load" rollup applied to any gauge.

  ## Examples

      iex> values = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> rolled = MobiusProcessing.Recipes.CoreNx.rolling_means(values, [2, 3])
      iex> Nx.to_flat_list(rolled[2])
      [1.5, 2.5, 3.5, 4.5]
      iex> Nx.to_flat_list(rolled[3])
      [2.0, 3.0, 4.0]
  """
  @spec rolling_means(Nx.Tensor.t(), [pos_integer()]) :: %{pos_integer() => Nx.Tensor.t()}
  def rolling_means(values, sizes) do
    Map.new(sizes, fn s -> {s, Nx.window_mean(values, {s})} end)
  end

  @doc """
  Mean of `values` ignoring positions where `validity` is `0`.
  `validity` is a `{n}` u8 tensor of `1` (present) / `0` (missing) — the
  same shape `MobiusProcessing.Tensor.from_column/1` returns.

  ## Examples

      iex> values = Nx.tensor([10.0, 20.0, 30.0, 40.0])
      iex> validity = Nx.tensor([1, 0, 1, 1], type: :u8)
      iex> MobiusProcessing.Recipes.CoreNx.mean_with_validity(values, validity)
      ...> |> Float.round(2)
      26.67
  """
  @spec mean_with_validity(Nx.Tensor.t(), Nx.Tensor.t()) :: float()
  def mean_with_validity(values, validity) do
    valid_f = Nx.as_type(validity, :f32)
    sum = values |> Nx.multiply(valid_f) |> Nx.sum()
    n = Nx.sum(valid_f)

    Nx.divide(sum, n) |> Nx.to_number()
  end

  @doc """
  q-quantile of a 1D tensor, by sort + lookup. Standing in for
  `Nx.quantile`, which Nx doesn't currently expose.

  ## Examples

      iex> t = Nx.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
      iex> MobiusProcessing.Recipes.CoreNx.quantile(t, 0.0)
      10.0
      iex> MobiusProcessing.Recipes.CoreNx.quantile(t, 1.0)
      50.0
      iex> MobiusProcessing.Recipes.CoreNx.quantile(t, 0.5)
      30.0
  """
  @spec quantile(Nx.Tensor.t(), float()) :: float()
  def quantile(values, q) when q >= 0.0 and q <= 1.0 do
    sorted = Nx.sort(values)
    n = Nx.size(values)
    idx = round(q * (n - 1))
    sorted[idx] |> Nx.to_number()
  end

  ## ---------------------------------------------------------------------
  ## Internals
  ## ---------------------------------------------------------------------

  defp slope_per_second(values) do
    n = Nx.size(values)
    t = Nx.iota({n}, type: :f32)
    design = Nx.stack([Nx.broadcast(1.0, {n}), t], axis: 1)

    coeffs = Nx.LinAlg.least_squares(design, Nx.as_type(values, :f32))
    [_intercept, slope] = Nx.to_flat_list(coeffs)
    slope
  end

  defp bincount(indices, n_bins) do
    zeros = Nx.broadcast(0, {n_bins})
    indices_2d = Nx.new_axis(indices, 1)
    updates = Nx.broadcast(1, Nx.shape(indices))
    Nx.indexed_add(zeros, indices_2d, updates)
  end
end
