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
  def scan(averages, std_devs, counts, opts \\ []) do
    averages = to_f64(averages)
    std_devs = to_f64(std_devs)
    n = Nx.size(averages)
    counts = broadcast_counts(counts, n)
    limit = Keyword.get(opts, :limit, 3.0) * 1.0

    {grand_mean, pooled_sigma} =
      case Keyword.get(opts, :baseline) do
        {gm, ps} -> {gm * 1.0, ps * 1.0}
        %{target: gm, sigma_reports: ps} -> {gm * 1.0, ps * 1.0}
        nil -> estimate_baseline(averages, std_devs, counts)
      end

    # Scalars enter the defn kernel as f64 tensors — bare floats would
    # be wrapped as f32 by Nx.
    {jump_ucl, jump_lcl, jumps, wobble_ucl, wobble_lcl, wobbles} =
      charts(averages, std_devs, counts, f64(grand_mean), f64(pooled_sigma), f64(limit))

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
  """
  @spec baseline(
          Nx.Tensor.t() | [number()],
          Nx.Tensor.t() | [number()],
          Nx.Tensor.t() | [number()] | pos_integer()
        ) :: %{target: float(), sigma_reports: float(), sigma_avg: float()}
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

    %{target: grand_mean, sigma_reports: pooled_sigma, sigma_avg: sigma_avg}
  end

  defp estimate_baseline(averages, std_devs, counts) do
    total = Nx.sum(counts)

    grand_mean =
      averages |> Nx.multiply(counts) |> Nx.sum() |> Nx.divide(total) |> Nx.to_number()

    dof = Nx.max(Nx.subtract(counts, 1.0), 0.0)

    if Nx.to_number(Nx.sum(dof)) == 0 do
      raise ArgumentError,
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
  defnp charts(averages, std_devs, counts, grand_mean, pooled_sigma, limit) do
    # Jump side: limits scale with subgroup size.
    jump_half = limit * pooled_sigma * Nx.rsqrt(counts)
    jump_ucl = grand_mean + jump_half
    jump_lcl = grand_mean - jump_half
    jumps = averages > jump_ucl or averages < jump_lcl

    # Wobble side: E[S] = c4·sigma, SD[S] = sigma·sqrt(1 - c4²), with
    # c4(n) ≈ (4n - 4) / (4n - 3) (exact form uses gamma functions).
    c4 = (counts * 4.0 - 4.0) / (counts * 4.0 - 3.0)
    spread = Nx.sqrt(1.0 - c4 * c4)
    wobble_ucl = pooled_sigma * (c4 + limit * spread)
    wobble_lcl = Nx.max(pooled_sigma * (c4 - limit * spread), 0.0)

    chartable = counts >= 2
    wobbles = (std_devs > wobble_ucl or std_devs < wobble_lcl) and chartable

    {jump_ucl, jump_lcl, jumps, wobble_ucl, wobble_lcl, wobbles}
  end

  defp broadcast_counts(counts, n) when is_integer(counts),
    do: Nx.broadcast(Nx.tensor(counts, type: :f64), {n})

  defp broadcast_counts(counts, _n), do: to_f64(counts)

  defp to_f64(values) when is_list(values), do: Nx.tensor(values, type: :f64)
  defp to_f64(values), do: Nx.as_type(values, :f64)

  defp f64(scalar), do: Nx.tensor(scalar * 1.0, type: :f64)
end
