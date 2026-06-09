defmodule MobiusSmarts.Detect.Shape do
  @moduledoc """
  Detects when the *shape* of a metric's distribution has gone bad even
  though the average and spread look fine — the p99 doubling under a
  calm median, one steady hum splitting into two alternating regimes
  (thermal-throttle cycling, retry storms).

  Implements: Population Stability Index, Jensen–Shannon divergence,
  and first Wasserstein distance over DDSketch bins.

  Why this needs its own detector: a mean and a std_dev are just two
  numbers, and infinitely many differently-shaped distributions share
  them — every other detector in this stack is *mathematically* blind
  to shape, not just bad at it. Mobius's DDSketch histograms are where
  the shape survives, and this module compares a healthy histogram
  against a current one.

  Three distances, three different judgement calls:

  - `psi/3` — "how much have the bin proportions reshuffled?" Its
    superpower is purely social: decades of production use in credit
    scoring mean its thresholds are *conventions you can cite* (0.1 =
    watch, 0.25 = act) rather than numbers you must invent.
  - `js_divergence/2` — "how distinguishable are these distributions,
    on a universal 0-to-ln 2 scale?" Symmetric and bounded, which makes
    scores comparable *across different metrics* — its real job is
    ranking which of fifty metrics drifted most.
  - `moved_by/3` — the most human of the three: imagine each histogram
    as piles of earth; how much earth must move, *and how far*, to
    reshape one into the other? The answer comes out in the metric's
    own units — "the latency distribution moved by about 40 ms" —
    which makes it the one to put in an alert message. It is
    *unsigned*: pair it with `mean_shift/3` when the alert should say
    which direction.

  `from_sketches/2` is the adapter that lines two `Mobius.DDSketch`es
  up on a shared bin axis so the three distances apply. Sketch bins are
  mergeable, so a trailing baseline sketch is just bin addition
  upstream.

  ## Resolution floor

  All three distances see at most what the sketch resolved. Mobius's
  default `relative_accuracy` is `0.1` — bins roughly 20% wide — so a
  tight distribution occupies only a handful of bins, drift below ~20%
  relative is invisible, and `moved_by`'s bin representative values
  carry up to ±10% error that flows straight into the headline number.
  Register metrics whose *shape* you intend to monitor with a finer
  accuracy (`relative_accuracy: 0.01` gives ~2%-wide bins) and treat
  the sketch accuracy as the floor of detectable drift.
  """

  import Nx.Defn

  alias Mobius.DDSketch

  @default_eps 1.0e-4

  @doc """
  Population Stability Index between an expected (baseline) and
  observed (current) binned distribution.

  `expected_counts` and `observed_counts` are aligned count tensors (or
  lists). Proportions are floored at `:eps` (default `#{@default_eps}`)
  in the standard way, so empty bins don't produce infinities.

      PSI = sum((p_obs - p_exp) * ln(p_obs / p_exp))

  Rule-of-thumb thresholds from its long production history: `< 0.1`
  stable, `0.1..0.25` drifting, `> 0.25` shifted.

  Raises `ArgumentError` when either count vector sums to zero — an
  all-zero histogram is missingness, not a shape to compare.

  ## Examples

      iex> alias MobiusSmarts.Detect.Shape
      iex> Shape.psi([100, 100, 100], [100, 100, 100])
      0.0

      iex> alias MobiusSmarts.Detect.Shape
      iex> Shape.psi([100, 100, 0], [0, 100, 100]) > 0.25
      true
  """
  @spec psi(Nx.Tensor.t() | [number()], Nx.Tensor.t() | [number()], keyword()) :: float()
  def psi(expected_counts, observed_counts, opts \\ []) do
    eps = Keyword.get(opts, :eps, @default_eps)

    nonzero_f64!(expected_counts, "expected")
    |> psi_kernel(nonzero_f64!(observed_counts, "observed"), f64(eps))
    |> Nx.to_number()
  end

  @doc """
  Jensen–Shannon divergence between two binned distributions, in nats.

  Symmetric, bounded in `[0, ln 2 ≈ 0.693]`. `0` for identical
  distributions; `ln 2` for fully disjoint supports. Uses the
  `0 · ln 0 = 0` convention, so empty bins need no flooring.

  Raises `ArgumentError` when either count vector sums to zero — an
  all-zero histogram is missingness, not a shape to compare.

  ## Examples

      iex> alias MobiusSmarts.Detect.Shape
      iex> Shape.js_divergence([10, 20, 10], [10, 20, 10])
      0.0

      iex> alias MobiusSmarts.Detect.Shape
      iex> jsd = Shape.js_divergence([100, 0], [0, 100])
      iex> Float.round(jsd, 4) == Float.round(:math.log(2), 4)
      true
  """
  @spec js_divergence(Nx.Tensor.t() | [number()], Nx.Tensor.t() | [number()]) :: float()
  def js_divergence(p_counts, q_counts) do
    nonzero_f64!(p_counts, "p")
    |> jsd_kernel(nonzero_f64!(q_counts, "q"))
    |> Nx.to_number()
  end

  @doc """
  How far the distribution moved, in the metric's own units — the first
  Wasserstein distance (earth-mover's distance) between two binned
  distributions on a common axis. **Unsigned**: it measures total
  displacement, not direction — a symmetric split moves mass without
  moving the mean. Pair with `mean_shift/3` for direction.

  `bin_values` are the representative values of the bins, ascending —
  the `values` element of `from_sketches/2`:

      W1 = sum(|CDF_p(i) - CDF_q(i)| * (v[i+1] - v[i]))

  Raises `ArgumentError` when either count vector sums to zero — an
  all-zero histogram is missingness, not a shape to compare.

  ## Examples

      iex> alias MobiusSmarts.Detect.Shape
      iex> # All mass moves from value 10 to value 13 — it moved by 3.
      iex> Shape.moved_by([100, 0, 0], [0, 0, 100], [10.0, 12.0, 13.0])
      3.0
  """
  @spec moved_by(
          Nx.Tensor.t() | [number()],
          Nx.Tensor.t() | [number()],
          Nx.Tensor.t() | [number()]
        ) :: float()
  def moved_by(p_counts, q_counts, bin_values) do
    nonzero_f64!(p_counts, "p")
    |> emd_kernel(nonzero_f64!(q_counts, "q"), to_f64(bin_values))
    |> Nx.to_number()
  end

  @doc """
  Signed companion to `moved_by/3`: the difference of distribution
  means, `mean(current) - mean(baseline)`, in the metric's own units.
  Positive means the distribution moved up/right.

  Raises `ArgumentError` when either count vector sums to zero — an
  all-zero histogram is missingness, not a shape to compare.

  ## Examples

      iex> alias MobiusSmarts.Detect.Shape
      iex> Shape.mean_shift([100, 0, 0], [0, 0, 100], [10.0, 12.0, 13.0])
      3.0
  """
  @spec mean_shift(
          Nx.Tensor.t() | [number()],
          Nx.Tensor.t() | [number()],
          Nx.Tensor.t() | [number()]
        ) :: float()
  def mean_shift(p_counts, q_counts, bin_values) do
    nonzero_f64!(p_counts, "p")
    |> mean_shift_kernel(nonzero_f64!(q_counts, "q"), to_f64(bin_values))
    |> Nx.to_number()
  end

  @doc """
  Align two `Mobius.DDSketch`es onto a common bin axis.

  Returns `%{baseline: counts, current: counts, values: bin_values}` —
  aligned f64 tensors ready for `psi/3`, `js_divergence/2`, or
  `moved_by/3`. Bin representative values use the DDSketch estimator
  `2·gamma^k / (gamma + 1)` from the sketch's own `gamma`; the zero
  bucket maps to `0.0` and negative bins mirror to negative values.

  Both sketches must share the same `relative_accuracy` (and therefore
  bin layout); raises `ArgumentError` otherwise. Also raises
  `ArgumentError` when either sketch has no bins — an empty sketch
  means the metric recorded nothing in that window, which is
  missingness, not a shape to compare.
  """
  @spec from_sketches(DDSketch.t(), DDSketch.t()) :: %{
          baseline: Nx.Tensor.t(),
          current: Nx.Tensor.t(),
          values: Nx.Tensor.t()
        }
  def from_sketches(%DDSketch{} = baseline, %DDSketch{} = current) do
    unless baseline.relative_accuracy == current.relative_accuracy do
      raise ArgumentError,
            "sketches use different relative_accuracy " <>
              "(#{baseline.relative_accuracy} vs #{current.relative_accuracy}); " <>
              "their bins are not alignable"
    end

    base_map = baseline |> DDSketch.bin_estimates() |> Map.new() |> ensure_bins!("baseline")
    curr_map = current |> DDSketch.bin_estimates() |> Map.new() |> ensure_bins!("current")

    values =
      base_map
      |> Map.keys()
      |> Enum.concat(Map.keys(curr_map))
      |> Enum.uniq()
      |> Enum.sort()

    %{
      baseline: Nx.tensor(Enum.map(values, &Map.get(base_map, &1, 0)), type: :f64),
      current: Nx.tensor(Enum.map(values, &Map.get(curr_map, &1, 0)), type: :f64),
      values: Nx.tensor(values, type: :f64)
    }
  end

  defnp psi_kernel(expected_counts, observed_counts, eps) do
    p = Nx.max(proportions(expected_counts), eps)
    q = Nx.max(proportions(observed_counts), eps)

    Nx.sum((q - p) * Nx.log(q / p))
  end

  defnp jsd_kernel(p_counts, q_counts) do
    p = proportions(p_counts)
    q = proportions(q_counts)
    m = (p + q) / 2.0

    0.5 * kl(p, m) + 0.5 * kl(q, m)
  end

  # W1 = sum(|CDF_p(i) - CDF_q(i)| * (v[i+1] - v[i]))
  defnp emd_kernel(p_counts, q_counts, values) do
    n = Nx.size(values)

    cdf_gap =
      (Nx.cumulative_sum(proportions(p_counts)) - Nx.cumulative_sum(proportions(q_counts)))
      |> Nx.abs()
      |> Nx.slice([0], [n - 1])

    Nx.sum(Nx.diff(values) * cdf_gap)
  end

  defnp mean_shift_kernel(p_counts, q_counts, values) do
    Nx.sum(proportions(q_counts) * values) - Nx.sum(proportions(p_counts) * values)
  end

  # 0 · ln 0 = 0 convention: zero-mass bins contribute nothing.
  defnp kl(p, m) do
    Nx.sum(Nx.select(p > 0.0, p * Nx.log(p / m), 0.0))
  end

  defnp proportions(counts) do
    counts / Nx.sum(counts)
  end

  defp to_f64(values) when is_list(values), do: Nx.tensor(values, type: :f64)
  defp to_f64(values), do: Nx.as_type(values, :f64)

  # A count vector with no mass has no distribution: dividing by its
  # zero sum yields NaN, and NaN fails every threshold comparison
  # *silently* — a real drift would go unreported. Raise instead so
  # missingness is loud.
  defp nonzero_f64!(counts, name) do
    tensor = to_f64(counts)

    if Nx.to_number(Nx.sum(tensor)) == 0.0 do
      raise ArgumentError,
            "#{name} counts sum to zero — an all-zero histogram means the " <>
              "metric recorded nothing in that window; treat it as " <>
              "missingness, not as a shape to compare"
    end

    tensor
  end

  defp ensure_bins!(bin_map, which) when map_size(bin_map) == 0 do
    raise ArgumentError,
          "#{which} sketch has no bins — an empty sketch means the metric " <>
            "recorded nothing in that window; treat it as missingness, " <>
            "not as a shape to compare"
  end

  defp ensure_bins!(bin_map, _which), do: bin_map

  defp f64(scalar), do: Nx.tensor(scalar * 1.0, type: :f64)
end
