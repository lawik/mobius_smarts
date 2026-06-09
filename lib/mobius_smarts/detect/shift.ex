defmodule MobiusSmarts.Detect.Shift do
  @moduledoc """
  Detects when a metric has moved off its target and *stayed* moved —
  even if no single window looks alarming on its own.

  Implements: EWMA control chart with exact time-varying limits
  (Lucas & Saccucci, 1990).

  Keep a running "impression" of the metric's level: each new window
  nudges the impression a fraction `lambda` of the way toward the new
  value, so the impression mostly remembers the recent past and
  gradually forgets the old. Random noise nudges it up and down in
  equal measure and cancels out — the impression barely moves. But a
  real sustained shift nudges it the same direction every window, and
  the impression gets dragged steadily off target until it exits its
  allowed band.

  The clever part is the band: because we know exactly how much a
  `lambda`-per-window impression *can* wander under pure noise,

      Var(z_t) = sigma² · (lambda / (2 - lambda)) · (1 - (1 - lambda)^(2t))

  the allowed band is computed exactly for every time step. It starts
  narrow (an impression two windows old has had no time to wander, so
  even small deviations are meaningful) and widens to a fixed width.
  Early windows get exactly the scrutiny they deserve.

  A pleasant side effect: the smoothed impression is also the natural
  line to *draw on a dashboard* — it's the metric with the noise taken
  out.

  The recursion itself is inherently sequential, so it runs as a plain
  fold — the one place in this stack where tensors buy nothing. The
  band and violation mask are a `defn` kernel over the folded result.

  **Where it sits:** between `MobiusSmarts.Detect.Jump` (no memory
  — catches big sudden moves, deaf to small ones) and
  `MobiusSmarts.Detect.Drift` (full memory — catches tiny slow
  moves, slower to confirm). Shift catches the middle: moderate moves,
  quickly, without false-alarming on noise.
  """

  import Nx.Defn

  @type chart_result() :: %{
          smoothed: Nx.Tensor.t(),
          ucl: Nx.Tensor.t(),
          lcl: Nx.Tensor.t(),
          violations: Nx.Tensor.t(),
          first_violation: non_neg_integer() | nil
        }

  @type state() :: %{
          target: float(),
          sigma: float(),
          lambda: float(),
          l: float(),
          smoothed: float(),
          step: non_neg_integer()
        }

  @doc """
  Run the chart over a whole series of window averages.

  Options:

  - `:baseline` — a map with `:target` and `:sigma_avg`, as returned by
    `MobiusSmarts.Detect.Jump.baseline/3`; the chart reads those two
    fields and ignores the rest (notably `:sigma_reports`, which is the
    wrong scale here).
  - `:target` — in-control mean from a healthy baseline. Overrides the
    baseline's `target`; required when no `:baseline` is given.
  - `:sigma` — in-control standard deviation **of the window averages**
    (the baseline's `sigma_avg`, not the per-report `sigma_reports` —
    the two differ by `sqrt(reports_per_window)` and the wrong one
    makes the band uselessly wide). Overrides the baseline's
    `sigma_avg`; required when no `:baseline` is given.
  - `:lambda` — nudge weight in `(0, 1]`, default `0.2`. Smaller
    detects smaller shifts but reacts more slowly.
  - `:l` — band width in (asymptotic) sigma units, default `3.0`.

  Either `:baseline` or both `:target` and `:sigma` must be supplied;
  anything less raises an `ArgumentError`.

  Returns the smoothed series, both limit series, a u8 violation mask,
  and the first violating index (`nil` if none).

  ## Examples

      iex> alias MobiusSmarts.Detect.Shift
      iex> flat = List.duplicate(50.0, 30)
      iex> result = Shift.chart(flat, target: 50.0, sigma: 2.0)
      iex> result.first_violation
      nil

      iex> alias MobiusSmarts.Detect.Shift
      iex> shifted = List.duplicate(50.0, 15) ++ List.duplicate(53.0, 15)
      iex> result = Shift.chart(shifted, target: 50.0, sigma: 2.0)
      iex> is_integer(result.first_violation) and result.first_violation >= 15
      true

  The baseline map from `MobiusSmarts.Detect.Jump.baseline/3` plugs
  in directly — `target` and `sigma_avg` are picked for you:

      iex> alias MobiusSmarts.Detect.Shift
      iex> baseline = %{target: 50.0, sigma_reports: 10.0, sigma_avg: 2.0}
      iex> shifted = List.duplicate(50.0, 15) ++ List.duplicate(53.0, 15)
      iex> result = Shift.chart(shifted, baseline: baseline)
      iex> is_integer(result.first_violation) and result.first_violation >= 15
      true
  """
  @spec chart(Nx.Tensor.t() | [number()], keyword()) :: chart_result()
  def chart(values, opts)

  def chart([], _opts) do
    raise ArgumentError,
          "cannot chart an empty series — MobiusSmarts.Source returns :empty " <>
            "for windows with no data; handle that before detection"
  end

  def chart(values, opts) do
    {target, sigma} = resolve_target_sigma!(opts)
    target = target * 1.0
    sigma = sigma * 1.0
    lambda = Keyword.get(opts, :lambda, 0.2) * 1.0
    l = Keyword.get(opts, :l, 3.0) * 1.0

    xs = to_list(values)

    smoothed_list =
      Enum.scan(xs, target, fn x, z_prev -> lambda * x + (1.0 - lambda) * z_prev end)

    smoothed = Nx.tensor(smoothed_list, type: :f64)

    # Scalars enter the defn kernel as explicit f64 tensors — bare
    # floats would be wrapped as f32 and cost precision in the limits.
    {ucl, lcl, violations} = band(smoothed, f64(target), f64(sigma), f64(lambda), f64(l))

    first_violation =
      if Nx.to_number(Nx.any(violations)) == 1 do
        Nx.to_number(Nx.argmax(violations))
      else
        nil
      end

    %{
      smoothed: smoothed,
      ucl: ucl,
      lcl: lcl,
      violations: violations,
      first_violation: first_violation
    }
  end

  @doc """
  Initialize streaming state. Same options as `chart/2`. The impression
  starts at `target`.

  ## Examples

      iex> alias MobiusSmarts.Detect.Shift
      iex> baseline = %{target: 50.0, sigma_reports: 10.0, sigma_avg: 2.0}
      iex> state = Shift.new(baseline: baseline)
      iex> {state.target, state.sigma}
      {50.0, 2.0}
  """
  @spec new(keyword()) :: state()
  def new(opts) do
    {target, sigma} = resolve_target_sigma!(opts)
    target = target * 1.0

    %{
      target: target,
      sigma: sigma * 1.0,
      lambda: Keyword.get(opts, :lambda, 0.2) * 1.0,
      l: Keyword.get(opts, :l, 3.0) * 1.0,
      smoothed: target,
      step: 0
    }
  end

  @doc """
  Feed one window average into streaming state.

  Returns `{status, state}` with `status` one of `:ok`,
  `:upper_violation`, `:lower_violation`.

  ## Examples

      iex> alias MobiusSmarts.Detect.Shift
      iex> state = Shift.new(target: 50.0, sigma: 2.0)
      iex> {status, _state} = Shift.step(state, 50.4)
      iex> status
      :ok
  """
  @spec step(state(), number()) :: {:ok | :upper_violation | :lower_violation, state()}
  def step(state, x) do
    smoothed = state.lambda * x + (1.0 - state.lambda) * state.smoothed
    t = state.step + 1

    var_factor =
      state.lambda / (2.0 - state.lambda) * (1.0 - :math.pow(1.0 - state.lambda, 2 * t))

    half_width = state.l * state.sigma * :math.sqrt(var_factor)

    status =
      cond do
        smoothed > state.target + half_width -> :upper_violation
        smoothed < state.target - half_width -> :lower_violation
        true -> :ok
      end

    {status, %{state | smoothed: smoothed, step: t}}
  end

  # Exact time-varying limits and the violation mask, one traced graph:
  # Var(z_t) = sigma² · (lambda / (2 - lambda)) · (1 - (1 - lambda)^(2t))
  defnp band(smoothed, target, sigma, lambda, l) do
    t = Nx.iota(Nx.shape(smoothed), type: :f64) + 1.0
    decay = Nx.pow(1.0 - lambda, t * 2.0)

    half_width = l * sigma * Nx.sqrt(lambda / (2.0 - lambda) * (1.0 - decay))

    ucl = target + half_width
    lcl = target - half_width
    {ucl, lcl, smoothed > ucl or smoothed < lcl}
  end

  # Explicit :target/:sigma win over the :baseline map, field by field;
  # without a baseline both are required.
  defp resolve_target_sigma!(opts) do
    case Keyword.get(opts, :baseline) do
      %{target: target, sigma_avg: sigma_avg} ->
        {Keyword.get(opts, :target, target), Keyword.get(opts, :sigma, sigma_avg)}

      nil ->
        with {:ok, target} <- Keyword.fetch(opts, :target),
             {:ok, sigma} <- Keyword.fetch(opts, :sigma) do
          {target, sigma}
        else
          :error ->
            raise ArgumentError,
                  "no in-control level/noise scale: pass baseline: (the map from " <>
                    "MobiusSmarts.Detect.Jump.baseline/3 — its :target and :sigma_avg " <>
                    "are read) or both :target and :sigma explicitly. If setting :sigma " <>
                    "by hand, use the baseline's sigma_avg — sigma_reports is a " <>
                    "different scale (off by sqrt(reports_per_window)) and makes the " <>
                    "band uselessly wide."
        end

      other ->
        raise ArgumentError,
              ":baseline must be a map with :target and :sigma_avg, as returned by " <>
                "MobiusSmarts.Detect.Jump.baseline/3; got: #{inspect(other)}"
    end
  end

  defp to_list(values) when is_list(values), do: Enum.map(values, &(&1 * 1.0))
  defp to_list(values), do: values |> Nx.as_type(:f64) |> Nx.to_flat_list()

  defp f64(scalar), do: Nx.tensor(scalar * 1.0, type: :f64)
end
