defmodule MobiusSmarts.Detect.Jump do
  @moduledoc """
  Detects windows that suddenly land way outside normal — and,
  separately, windows where the device got *erratic* even though its
  average stayed fine (the `wobble` results).

  Implements: Shewhart X̄/S control charts (1920s; the foundation of
  statistical process control).

  First, learn what normal is from a healthy stretch: the typical level
  (grand mean) and the typical noise (pooled sigma). Then draw a band
  around the typical level — three noise-widths wide on each side.
  Honest random noise lands outside a 3-sigma band about once in 370
  windows; anything outside it is worth flagging *immediately*, no
  history required. That's the whole jump check: a tripwire calibrated
  to the device's own noise, so "way off" means way off *for this
  device*, not by some global constant.

  One subtlety the math handles automatically: a window averaging 100
  reports should sit much closer to the true level than a window
  averaging 5 reports, so the band tightens as the report count grows
  (`sigma/sqrt(n)`). Busy windows get held to a stricter standard.

  The second check — the wobble side — watches the noise itself instead
  of the level. Every Mobius window already stores its internal
  `std_dev`, a direct reading of "how shaky was the device during this
  window". A device whose average is rock-steady while its
  within-window spread climbs is a classic *pre-failure* signature
  (flapping network links, sagging power rails, degrading flash), and
  it is mathematically invisible to every detector that only looks at
  averages. The wobble band uses the standard `c4` small-sample
  correction (S underestimates sigma for small n; `c4` is computed with
  the `(4n-4)/(4n-3)` approximation, accurate to ~0.1% for n ≥ 4).

  The wobble band's *lower* limit (spread collapsing — a flat, stuck
  signal) assumes the process normally carries noise in every window.
  Zero-inflated metrics break that assumption: an idle device's run
  queue is exactly 0 for whole windows at a time, with occasional
  bursts inflating the pooled sigma, so the textbook lower limit sits
  above zero and every healthy idle window would alarm. The charts
  therefore arm the lower limit only when the baseline pool itself
  contained no zero-spread window (`baseline/3` records this as
  `:sd_floor`; in phase I the scanned windows speak for themselves) —
  if flat windows are part of normal life, a flat window is not
  evidence.

  The Mobius summary window — average, std_dev, count — *is* the
  subgroup format these charts were designed for. No adaptation, no
  approximation.

  **Blind spot:** anything gradual. A leak adding 0.1% per window never
  trips a per-window tripwire — that's what
  `MobiusSmarts.Detect.Shift` and `MobiusSmarts.Detect.Drift`
  are for.

  All computation is vectorized Nx over the window series; the chart
  math is a single `defn` kernel.
  """

  import Nx.Defn

  defmodule NoDispersionError do
    @moduledoc """
    Raised by `MobiusSmarts.Detect.Jump.baseline/3` (and `scan/4` in
    phase I, when it estimates its own baseline) when every window has
    fewer than 2 reports, so there is no within-window dispersion to
    pool into a sigma estimate.
    """
    defexception [:message]
  end

  @type result() :: %{
          grand_mean: float(),
          pooled_sigma: float(),
          jump_ucl: Nx.Tensor.t(),
          jump_lcl: Nx.Tensor.t(),
          jumps: Nx.Tensor.t(),
          wobble_ucl: Nx.Tensor.t(),
          wobble_lcl: Nx.Tensor.t(),
          wobbles: Nx.Tensor.t()
        }

  @doc """
  Check a series of summary windows for jumps and wobbles.

  - `averages` — per-window means (1D tensor or list).
  - `std_devs` — per-window standard deviations, same length.
  - `counts` — per-window report counts: a tensor/list, or a single
    integer when all windows have the same size.

  Options:

  - `:baseline` — the map returned by `baseline/3`, or a
    `{grand_mean, pooled_sigma}` tuple (pooled = per-report sigma),
    from a known-healthy period (phase II monitoring). When omitted,
    both are estimated from the given windows themselves (phase I —
    finding the out-of-control windows in a historical stretch; note a
    large outlier shifts the estimated centerline, textbook phase I is
    iterative).
  - `:limit` — band width in sigma units, default `3.0`.

  Windows with fewer than 2 reports carry no dispersion information;
  they are excluded from pooling and never flag as wobbles.

  ## Examples

      iex> alias MobiusSmarts.Detect.Jump
      iex> averages = [10.0, 10.1, 9.9, 10.0, 14.0]
      iex> std_devs = [1.0, 1.1, 0.9, 1.0, 1.0]
      iex> result = Jump.scan(averages, std_devs, 25, baseline: {10.0, 1.0})
      iex> Nx.to_flat_list(result.jumps)
      [0, 0, 0, 0, 1]
  """
  @spec scan(
          Nx.Tensor.t() | [number()],
          Nx.Tensor.t() | [number()],
          Nx.Tensor.t() | [number()] | pos_integer(),
          keyword()
        ) :: result()
  def scan(averages, std_devs, counts, opts \\ [])

  def scan([], _std_devs, _counts, _opts) do
    raise ArgumentError,
          "cannot scan an empty series — MobiusSmarts.Source returns :empty " <>
            "for windows with no data; handle that before detection"
  end

  def scan(averages, std_devs, counts, opts) do
    averages = to_f64(averages)
    std_devs = to_f64(std_devs)
    n = Nx.size(averages)
    counts = counts |> broadcast_counts(n) |> validate_counts!()
    limit = Keyword.get(opts, :limit, 3.0) * 1.0

    {grand_mean, pooled_sigma, sd_floor} =
      case Keyword.get(opts, :baseline) do
        # Explicit tuples are the textbook S-chart contract: both
        # limits armed, the caller owns the zero-inflation question.
        {gm, ps} -> {gm * 1.0, ps * 1.0, 1.0}
        %{target: gm, sigma_reports: ps} = baseline -> {gm * 1.0, ps * 1.0, sd_floor(baseline)}
        nil -> phase1_baseline(averages, std_devs, counts)
      end

    low_armed = if sd_floor > 0.0, do: 1.0, else: 0.0

    # Scalars enter the defn kernel as f64 tensors — bare floats would
    # be wrapped as f32 by Nx.
    {jump_ucl, jump_lcl, jumps, wobble_ucl, wobble_lcl, wobbles} =
      charts(
        averages,
        std_devs,
        counts,
        f64(grand_mean),
        f64(pooled_sigma),
        f64(limit),
        f64(low_armed)
      )

    %{
      grand_mean: grand_mean,
      pooled_sigma: pooled_sigma,
      jump_ucl: jump_ucl,
      jump_lcl: jump_lcl,
      jumps: jumps,
      wobble_ucl: wobble_ucl,
      wobble_lcl: wobble_lcl,
      wobbles: wobbles
    }
  end

  @doc """
  Estimate a baseline from a stretch of healthy windows. Returns a map
  with **two distinct noise scales** — using the wrong one silently
  changes detector sensitivity by `sqrt(reports_per_window)`:

  - `:target` — the grand mean, weighting window averages by report
    count.
  - `:sigma_reports` — the pooled *within-window* (per-report) sigma,
    weighting window variances by degrees of freedom (`count - 1`) so
    singleton windows don't contaminate the pool. This is the scale
    `scan/4`'s `:baseline` wants — its X̄ limits divide by `sqrt(n)`
    themselves.
  - `:sigma_avg` — the standard deviation **of the window averages**,
    measured directly from the baseline stretch. This is the scale
    `MobiusSmarts.Detect.Drift` and `MobiusSmarts.Detect.Shift`
    want for `:sigma` — they operate on the average series and do no
    `sqrt(n)` scaling of their own. Measuring it directly (rather than
    deriving `sigma_reports/sqrt(n)`) also keeps it honest when windows
    aren't i.i.d.

  `sigma_avg` needs at least 2 windows (it is `0.0` below that); use a
  healthy stretch of hundreds.

  The map also carries `:sd_floor` — the smallest within-window spread
  among the pool's dispersion-carrying windows. `scan/4` arms the
  wobble chart's lower (stuck-signal) limit only when it is positive:
  a pool containing flat windows says flat is normal here (see the
  zero-inflation note in the moduledoc).
  """
  @spec baseline(
          Nx.Tensor.t() | [number()],
          Nx.Tensor.t() | [number()],
          Nx.Tensor.t() | [number()] | pos_integer()
        ) :: %{target: float(), sigma_reports: float(), sigma_avg: float(), sd_floor: float()}
  def baseline([], _std_devs, _counts) do
    raise ArgumentError,
          "cannot estimate a baseline from an empty series — MobiusSmarts.Source " <>
            "returns :empty for windows with no data; handle that before detection"
  end

  def baseline(averages, std_devs, counts) do
    averages = to_f64(averages)
    std_devs = to_f64(std_devs)
    counts = broadcast_counts(counts, Nx.size(averages))
    {grand_mean, pooled_sigma} = estimate_baseline(averages, std_devs, counts)

    sigma_avg =
      if Nx.size(averages) >= 2 do
        averages |> Nx.standard_deviation(ddof: 1) |> Nx.to_number()
      else
        0.0
      end

    %{
      target: grand_mean,
      sigma_reports: pooled_sigma,
      sigma_avg: sigma_avg,
      sd_floor: chartable_sd_floor(std_devs, counts)
    }
  end

  # The smallest within-window spread among windows that carry
  # dispersion information. Zero means flat windows are part of this
  # metric's normal life, so the wobble chart's lower limit (the
  # stuck-signal alarm) must stay disarmed.
  defp chartable_sd_floor(std_devs, counts) do
    chartable = Nx.greater_equal(counts, 2)

    if Nx.to_number(Nx.any(chartable)) == 1 do
      chartable
      |> Nx.select(std_devs, Nx.Constants.infinity({:f, 64}))
      |> Nx.reduce_min()
      |> Nx.to_number()
    else
      0.0
    end
  end

  defp phase1_baseline(averages, std_devs, counts) do
    {grand_mean, pooled_sigma} = estimate_baseline(averages, std_devs, counts)
    {grand_mean, pooled_sigma, chartable_sd_floor(std_devs, counts)}
  end

  # Baseline maps predating :sd_floor (or hand-built ones) keep the
  # textbook armed-low-limit behavior.
  defp sd_floor(%{sd_floor: floor}), do: floor * 1.0
  defp sd_floor(_baseline), do: 1.0

  defp estimate_baseline(averages, std_devs, counts) do
    total = Nx.sum(counts)

    grand_mean =
      averages |> Nx.multiply(counts) |> Nx.sum() |> Nx.divide(total) |> Nx.to_number()

    dof = Nx.max(Nx.subtract(counts, 1.0), 0.0)

    if Nx.to_number(Nx.sum(dof)) == 0 do
      raise NoDispersionError,
            "cannot estimate a baseline sigma: every window has fewer than 2 reports, " <>
              "so there is no within-window dispersion to pool. Pass an explicit " <>
              ":baseline or use windows with more reports."
    end

    pooled_sigma =
      std_devs
      |> Nx.pow(2)
      |> Nx.multiply(dof)
      |> Nx.sum()
      |> Nx.divide(Nx.sum(dof))
      |> Nx.sqrt()
      |> Nx.to_number()

    {grand_mean, pooled_sigma}
  end

  # Both charts as one traced graph over the window series.
  defnp charts(averages, std_devs, counts, grand_mean, pooled_sigma, limit, low_armed) do
    # Jump side: limits scale with subgroup size.
    jump_half = limit * pooled_sigma * Nx.rsqrt(counts)
    jump_ucl = grand_mean + jump_half
    jump_lcl = grand_mean - jump_half
    jumps = averages > jump_ucl or averages < jump_lcl

    # Wobble side: E[S] = c4·sigma, SD[S] = sigma·sqrt(1 - c4²), with
    # c4(n) ≈ (4n - 4) / (4n - 3) (exact form uses gamma functions).
    # The lower limit is zeroed (disarmed) when flat windows are normal
    # for this metric — see the moduledoc on zero-inflated metrics.
    c4 = (counts * 4.0 - 4.0) / (counts * 4.0 - 3.0)
    spread = Nx.sqrt(1.0 - c4 * c4)
    wobble_ucl = pooled_sigma * (c4 + limit * spread)
    wobble_lcl = Nx.max(pooled_sigma * (c4 - limit * spread), 0.0) * low_armed

    chartable = counts >= 2
    wobbles = (std_devs > wobble_ucl or std_devs < wobble_lcl) and chartable

    {jump_ucl, jump_lcl, jumps, wobble_ucl, wobble_lcl, wobbles}
  end

  defp broadcast_counts(counts, n) when is_integer(counts),
    do: Nx.broadcast(Nx.tensor(counts, type: :f64), {n})

  defp broadcast_counts(counts, _n), do: to_f64(counts)

  # A count below 1 is impossible data — a Mobius summary window always
  # contains at least one report — and it silently corrupts the charts:
  # c4 hits 4/3 at n = 0 (sqrt of a negative → NaN wobble limits) and
  # rsqrt(0) blows the jump band out to ±infinity.
  defp validate_counts!(counts) do
    min = counts |> Nx.reduce_min() |> Nx.to_number()

    if min < 1 do
      raise ArgumentError,
            "every window count must be >= 1, got #{min} — a Mobius summary window " <>
              "always contains at least one report; counts below 1 make the c4 wobble " <>
              "correction undefined (NaN limits) and the jump band infinite"
    end

    counts
  end

  defp to_f64(values) when is_list(values), do: Nx.tensor(values, type: :f64)
  defp to_f64(values), do: Nx.as_type(values, :f64)

  defp f64(scalar), do: Nx.tensor(scalar * 1.0, type: :f64)
end
