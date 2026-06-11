defmodule MobiusSmarts.Calibrate do
  @moduledoc """
  Derives every detector threshold from one opinion: the false-alarm
  budget.

  Instead of tuning `h`, `l`, and `limit` per detector, the
  configuration states a tolerance — "this device may cry wolf about
  once a week" — and each detector's threshold is computed from its
  own average-run-length (ARL) mathematics so that the *instance as a
  whole* false-alarms at roughly the budgeted rate on a healthy
  device:

  - the budget is converted to windows (`false_alarm_every /
    resolution` — both stated in config, no inference),
  - split Bonferroni-style across every alarm stream being watched
    (four per metric: jump, wobble, shift, drift — plus novelty),
  - and inverted through each detector's ARL relation: the normal tail
    for Shewhart limits, Siegmund's approximation for CUSUM's `h`,
    and the chi-square quantile for the novelty threshold.

  The payoff is bigger than convenience: with all thresholds derived
  from one rate, every finding's `concern` (ratio to its threshold) is
  comparable across detectors — which is what makes the aggregate
  health level honest.

  ## What the budget actually guarantees

  The ARL math assumes independent, stationary, near-Gaussian
  windows. On data like that the budget is *accurate*: the synthetic
  replay harness measures 0 realized false alarms over 3 days of
  i.i.d. Gaussian minute-windows at a `{1, :day}` budget (slightly
  conservative, as the Bonferroni split predicts). Real device
  telemetry violates those assumptions — the same harness measures
  ~3x the budget on AR(1)-wandering data (phi 0.995) — so on real
  metrics treat the budget as *directional*: `{1, :day}` is reliably
  more sensitive than `{1, :week}`, but neither is a contract.
  Removing autocorrelation at the source (seasonal/residual modeling,
  issue #8) is what makes the absolute rate truthful, not tuning the
  budget.

  One coupling to know about: the Bonferroni split divides the budget
  across every alarm stream, so adding a metric to `:watch` slightly
  tightens every other detector.

  Two stated approximations:

  - The EWMA limit reuses the normal-tail inversion. The EWMA
    statistic is autocorrelated, so its true ARL at that limit is
    *longer* than the target — errors fall on the quiet side.
  - The chi-square quantile uses Wilson–Hilferty (accurate to ~1%
    in this range).

  Re-scanning overlapping history each tick does not multiply false
  alarms: an alarm is an event in the data (the statistic path
  crossing its limit), not in the scan, so examining the same windows
  again cannot create a second one.
  """

  alias MobiusSmarts.Config

  @type t() :: %{
          arl: float(),
          jump_limit: float(),
          ewma_l: float(),
          cusum_h: float()
        }

  @doc """
  The calibrated thresholds for a config.

  Returns `%{arl: ..., jump_limit: ..., ewma_l: ..., cusum_h: ...}`
  where `arl` is the per-alarm-stream target average run length in
  windows. The novelty threshold depends on how many metrics end up in
  the fitted model, so it is derived later via `chi2_threshold/2` with
  the same `arl`.
  """
  @spec for_config(Config.t()) :: t()
  def for_config(%Config{} = config) do
    budget_windows = Config.ms(config.false_alarm_every) / max(Config.ms(config.resolution), 1)

    streams =
      4 * length(config.watch) +
        if Config.novelty?(config), do: 1, else: 0

    arl = max(budget_windows * max(streams, 1), 2.0)

    %{
      arl: arl,
      jump_limit: shewhart_limit(arl),
      ewma_l: shewhart_limit(arl),
      cusum_h: cusum_h(2.0 * arl, config.cusum_k)
    }
  end

  @doc """
  The two-sided Shewhart limit (in sigma units) with in-control
  ARL `arl`: the `z` where `P(|Z| > z) = 1/arl`.

  The textbook 3-sigma chart false-alarms once in ~370 windows:

  ## Examples

      iex> MobiusSmarts.Calibrate.shewhart_limit(370.4) |> Float.round(2)
      3.0
  """
  @spec shewhart_limit(number()) :: float()
  def shewhart_limit(arl) when arl > 1 do
    probit(1.0 - 1.0 / (2.0 * arl))
  end

  @doc """
  The CUSUM alarm threshold `h` (in sigma units) whose one-sided
  in-control ARL is `arl`, at drain rate `k`.

  Inverts Siegmund's approximation
  `ARL ≈ (exp(2kb) − 2kb − 1) / (2k²)` with `b = h + 1.166` by
  bisection. The detector-stack default `h = 5, k = 0.5` corresponds
  to roughly 940 healthy windows per side:

  ## Examples

      iex> MobiusSmarts.Calibrate.cusum_h(938.0, 0.5) |> Float.round(1)
      5.0
  """
  @spec cusum_h(number(), number()) :: float()
  def cusum_h(arl, k) when arl > 1 and k > 0 do
    bisect(1.0e-6, 200.0, fn h -> siegmund_arl(h, k) - arl end)
  end

  @doc """
  The novelty alarm threshold for `df` metrics at one alarm per `arl`
  scores: the square root of the chi-square quantile at `1 - 1/arl`
  (Wilson–Hilferty approximation).

  ## Examples

      iex> MobiusSmarts.Calibrate.chi2_threshold(20.0, 2) |> Float.round(1)
      2.4
  """
  @spec chi2_threshold(number(), pos_integer()) :: float()
  def chi2_threshold(arl, df) when arl > 1 and df >= 1 do
    z = probit(1.0 - 1.0 / arl)
    a = 2.0 / (9.0 * df)
    quantile = df * :math.pow(1.0 - a + z * :math.sqrt(a), 3)
    :math.sqrt(max(quantile, 0.0))
  end

  @doc """
  The standard normal quantile function (inverse CDF), by bisection on
  `erf` — slow and obviously correct, and calibration runs once per
  startup.

  ## Examples

      iex> MobiusSmarts.Calibrate.probit(0.975) |> Float.round(3)
      1.96
  """
  @spec probit(float()) :: float()
  def probit(p) when p > 0.0 and p < 1.0 do
    bisect(-12.0, 12.0, fn z -> phi(z) - p end)
  end

  defp phi(z), do: 0.5 * (1.0 + :math.erf(z / :math.sqrt(2.0)))

  defp siegmund_arl(h, k) do
    b = h + 1.166
    (:math.exp(2.0 * k * b) - 2.0 * k * b - 1.0) / (2.0 * k * k)
  end

  # Bisection for a monotone-increasing f with a sign change in [lo, hi].
  defp bisect(lo, hi, f), do: bisect(lo, hi, f, 100)
  defp bisect(lo, hi, _f, 0), do: (lo + hi) / 2.0

  defp bisect(lo, hi, f, n) do
    mid = (lo + hi) / 2.0

    if f.(mid) < 0.0 do
      bisect(mid, hi, f, n - 1)
    else
      bisect(lo, mid, f, n - 1)
    end
  end
end
