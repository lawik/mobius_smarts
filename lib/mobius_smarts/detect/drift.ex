defmodule MobiusSmarts.Detect.Drift do
  @moduledoc """
  Detects small persistent drifts in a metric's level — and estimates
  when the drift began.

  Implements: two-sided CUSUM (Page, 1954).

  Picture a bucket that collects suspicion. Every window that comes in
  slightly above normal adds a few drops; every window at or below
  normal drains some out. The drain rate (`k`, the "allowance") is set
  so that honest random noise drains as fast as it fills — the bucket
  hovers near empty forever. But a real drift, even a tiny one, fills
  slightly faster than the drain empties, so the level ratchets up
  window after window and eventually reaches the alarm line (`h`) —
  the alarm fires the moment the level touches the line, not only
  once it spills past.
  No single window tipped it off; the *accumulated* evidence did. A
  second bucket catches downward drifts the same way.

  This is why it beats the per-window tripwire at its own game: a
  0.25-sigma drift is statistically invisible in any one window, but 40
  windows of it is overwhelming evidence, and the bucket has been
  quietly adding it all up. The bonus the bucket gives for free: rewind
  to the last moment it was *empty* — that's approximately when the
  drift actually began. "Memory has been leaking since Tuesday 03:00"
  is a far better diagnostic than "alarm fired Friday".

  Input is the natural Mobius shape: the per-window `average` series of
  a summary metric, calibrated against a healthy baseline period. Pass
  the map from `MobiusSmarts.Detect.Jump.baseline/3` as `:baseline`
  and the detector picks the right fields (`target` and **`sigma_avg`**)
  itself. Only when setting `:target`/`:sigma` by hand does the scale
  footgun apply: `sigma` here is the standard deviation *of the window
  averages*, not the per-report `sigma_reports`; the two differ by
  `sqrt(reports_per_window)` and using the per-report scale makes this
  detector nearly deaf.

  **Blind spots:** big sudden jumps land faster on
  `MobiusSmarts.Detect.Jump`; changes in *spread* rather than level
  are `Jump`'s wobble result; changes in distribution *shape* under a
  steady level are `MobiusSmarts.Detect.Shape`.

  ## Tuning — two honest tradeoffs, in sigma units

  - `:k` (drain rate, default `0.5`) — half the drift size you care
    about. `0.5` targets one-sigma drifts.
  - `:h` (alarm line, default `5.0`) — trades false alarms against
    detection delay. The in-control time-to-false-alarm grows
    *exponentially* in `h`; the defaults give an average run length of
    roughly 940 healthy windows per side (~470 for the two-sided
    scheme, Siegmund's approximation), so size `h` to your monitoring
    horizon (e.g. `6.0` buys roughly e times longer quiet stretches at
    the cost of slightly later detection). These rates assume
    independent in-control windows — see the calibration caveat in
    `MobiusSmarts.Detect` before trusting them on autocorrelated
    or seasonal metrics.

  Two interfaces:

  - `scan/2` — batch over a whole series, vectorized in Nx via the
    reflection identity `S_t = C_t - min(0, min_{s<=t} C_s)` where `C`
    is the cumulative sum of drift-adjusted deviations. One
    `cumulative_sum` and one `cumulative_min`, no per-element recursion.
  - `new/1` + `step/2` — O(1)-state streaming form for on-device use,
    one update per incoming window.
  """

  import Nx.Defn

  @type scan_result() :: %{
          upper: Nx.Tensor.t(),
          lower: Nx.Tensor.t(),
          upper_alarm: non_neg_integer() | nil,
          lower_alarm: non_neg_integer() | nil,
          upper_onset: non_neg_integer() | nil,
          lower_onset: non_neg_integer() | nil
        }

  @type state() :: %{
          target: float(),
          sigma: float(),
          k: float(),
          h: float(),
          upper: float(),
          lower: float(),
          step: non_neg_integer(),
          upper_onset: non_neg_integer() | nil,
          lower_onset: non_neg_integer() | nil
        }

  @doc """
  Scan a whole series of window averages for drifts.

  `values` is a 1D tensor (or list). Options:

  - `:baseline` — a map with `:target` and `:sigma_avg`, as returned by
    `MobiusSmarts.Detect.Jump.baseline/3`; the detector reads those
    two fields and ignores the rest (notably `:sigma_reports`, which is
    the wrong scale here).
  - `:target` — in-control mean, from a healthy baseline. Overrides the
    baseline's `target`; required when no `:baseline` is given.
  - `:sigma` — in-control standard deviation **of the window averages**
    over the same baseline (the baseline's `sigma_avg`, not
    `sigma_reports`). Overrides the baseline's `sigma_avg`; required
    when no `:baseline` is given.
  - `:k` — drain rate in sigma units, default `0.5`.
  - `:h` — alarm threshold in sigma units, default `5.0`.

  Either `:baseline` or both `:target` and `:sigma` must be supplied;
  anything less raises an `ArgumentError`.

  Returns the upper/lower bucket-level series, the first alarm index
  for each side (`nil` when no alarm), and the onset index for each
  alarmed side — the last window at which that bucket was empty before
  the alarm.

  ## Examples

      iex> alias MobiusSmarts.Detect.Drift
      iex> flat = List.duplicate(10.0, 20)
      iex> result = Drift.scan(flat, target: 10.0, sigma: 1.0)
      iex> {result.upper_alarm, result.lower_alarm}
      {nil, nil}

      iex> alias MobiusSmarts.Detect.Drift
      iex> shifted = List.duplicate(10.0, 10) ++ List.duplicate(12.0, 10)
      iex> result = Drift.scan(shifted, target: 10.0, sigma: 1.0)
      iex> result.upper_alarm
      13
      iex> result.upper_onset
      9

  The baseline map from `MobiusSmarts.Detect.Jump.baseline/3` plugs
  in directly — `target` and `sigma_avg` are picked for you:

      iex> alias MobiusSmarts.Detect.Drift
      iex> baseline = %{target: 10.0, sigma_reports: 5.0, sigma_avg: 1.0}
      iex> shifted = List.duplicate(10.0, 10) ++ List.duplicate(12.0, 10)
      iex> result = Drift.scan(shifted, baseline: baseline)
      iex> result.upper_alarm
      13
  """
  @spec scan(Nx.Tensor.t() | [number()], keyword()) :: scan_result()
  def scan(values, opts)

  def scan([], _opts) do
    raise ArgumentError,
          "cannot scan an empty series — MobiusSmarts.Source returns :empty " <>
            "for windows with no data; handle that before detection"
  end

  def scan(values, opts) do
    values = to_f64(values)
    {target, sigma} = resolve_target_sigma!(opts)
    k = Keyword.get(opts, :k, 0.5)
    h = Keyword.get(opts, :h, 5.0)

    # Scalars must be wrapped as f64 tensors — bare floats are wrapped
    # as f32 by Nx and silently cost precision.
    {upper, lower} = buckets(values, f64(target), f64(sigma), f64(k))

    upper_alarm = first_crossing(upper, h)
    lower_alarm = first_crossing(lower, h)

    %{
      upper: upper,
      lower: lower,
      upper_alarm: upper_alarm,
      lower_alarm: lower_alarm,
      upper_onset: onset(upper, upper_alarm),
      lower_onset: onset(lower, lower_alarm)
    }
  end

  @doc """
  Initialize streaming drift-detection state. Same options as `scan/2`.

  ## Examples

      iex> alias MobiusSmarts.Detect.Drift
      iex> baseline = %{target: 10.0, sigma_reports: 5.0, sigma_avg: 1.0}
      iex> state = Drift.new(baseline: baseline)
      iex> {state.target, state.sigma}
      {10.0, 1.0}
  """
  @spec new(keyword()) :: state()
  def new(opts) do
    {target, sigma} = resolve_target_sigma!(opts)

    %{
      target: target * 1.0,
      sigma: sigma * 1.0,
      k: Keyword.get(opts, :k, 0.5) * 1.0,
      h: Keyword.get(opts, :h, 5.0) * 1.0,
      upper: 0.0,
      lower: 0.0,
      step: 0,
      upper_onset: nil,
      lower_onset: nil
    }
  end

  @doc """
  Feed one window average into streaming state.

  Returns `{status, state}` where `status` is `:ok`, `:upper_alarm`, or
  `:lower_alarm`. After an alarm, the caller decides whether to reset
  (start watching for the next drift) or keep accumulating; `reset/1`
  empties the buckets while keeping the configuration.

  ## Examples

      iex> alias MobiusSmarts.Detect.Drift
      iex> state = Drift.new(target: 10.0, sigma: 1.0)
      iex> {status, _state} = Drift.step(state, 10.2)
      iex> status
      :ok
  """
  @spec step(state(), number()) :: {:ok | :upper_alarm | :lower_alarm, state()}
  def step(state, x) do
    y = (x - state.target) / state.sigma

    upper = max(0.0, state.upper + y - state.k)
    lower = max(0.0, state.lower - y - state.k)

    # Onset is the last window at which the bucket sat empty, matching
    # scan/2. `step` is the index of the incoming window, so the last
    # empty reading was the one before it.
    upper_onset = if upper == 0.0, do: nil, else: state.upper_onset || max(state.step - 1, 0)
    lower_onset = if lower == 0.0, do: nil, else: state.lower_onset || max(state.step - 1, 0)

    state = %{
      state
      | upper: upper,
        lower: lower,
        step: state.step + 1,
        upper_onset: upper_onset,
        lower_onset: lower_onset
    }

    status =
      cond do
        upper >= state.h -> :upper_alarm
        lower >= state.h -> :lower_alarm
        true -> :ok
      end

    {status, state}
  end

  @doc """
  Empty the buckets after a handled alarm, keeping configuration.
  """
  @spec reset(state()) :: state()
  def reset(state) do
    %{state | upper: 0.0, lower: 0.0, upper_onset: nil, lower_onset: nil}
  end

  # Standardization and both bucket series as one traced graph.
  defnp buckets(values, target, sigma, k) do
    y = (values - target) / sigma
    {accumulate(y, k), accumulate(-y, k)}
  end

  # S_t = max(0, S_{t-1} + y_t - k) via the reflection identity:
  # with C_t = cumsum(y - k) and C_0 = 0,
  # S_t = C_t - min(0, min_{s<=t} C_s).
  defnp accumulate(y, k) do
    c = Nx.cumulative_sum(y - k)
    c - Nx.min(Nx.cumulative_min(c), 0)
  end

  defp first_crossing(s, h) do
    mask = Nx.greater_equal(s, f64(h))

    if Nx.to_number(Nx.any(mask)) == 1 do
      Nx.to_number(Nx.argmax(mask))
    else
      nil
    end
  end

  # Last index at (or before) the alarm where the bucket was empty.
  # The clamp at zero makes zeros exact, so float equality is safe here.
  defp onset(_s, nil), do: nil

  defp onset(s, alarm_index) do
    s
    |> Nx.to_flat_list()
    |> Enum.take(alarm_index + 1)
    |> Enum.with_index()
    |> Enum.reverse()
    |> Enum.find_value(0, fn {v, i} -> if v == 0.0, do: i end)
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
                    "different scale (off by sqrt(reports_per_window)) and makes this " <>
                    "detector nearly deaf."
        end

      other ->
        raise ArgumentError,
              ":baseline must be a map with :target and :sigma_avg, as returned by " <>
                "MobiusSmarts.Detect.Jump.baseline/3; got: #{inspect(other)}"
    end
  end

  defp to_f64(values) when is_list(values), do: Nx.tensor(values, type: :f64)
  defp to_f64(values), do: Nx.as_type(values, :f64)

  defp f64(scalar), do: Nx.tensor(scalar * 1.0, type: :f64)
end
