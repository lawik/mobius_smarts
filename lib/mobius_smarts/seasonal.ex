defmodule MobiusSmarts.Seasonal do
  @moduledoc """
  An incremental per-slot seasonal model (issue #8): the in-memory
  expectation of "what does this metric look like at this point in
  its cycle", learned one window at a time.

  Plain bands on cyclic telemetry fail both ways — the cycle inflates
  the fitted sigma so in-envelope anomalies pass, and the cycle's
  peaks alarm as drift. The runtime's fix is a *residual transform*:
  the period (`Config` `:seasonality`, e.g. `{1, :day}`) is divided
  into one slot per `:resolution` window, each slot keeps an
  incrementally updated mean of the values it has seen, and once the
  model is warm the detectors run on `value − slot_mean` instead of
  the raw series. The transform is pure and the model is just data:
  you keep the coefficients, not the raw history, so warm-up does not
  depend on RRD retention.

  Warm-up follows field practice (~3 cycles): a slot is ready after
  `#{3}` visits, the model after 90% of slots are. Until then
  `residuals/2` is not meant to be used — the runtime detects on the
  raw series and shows the warming progress. Slots update as plain
  averages for their first visits and as an exponential moving
  average after, so old seasons fade rather than accumulate.
  """

  alias MobiusSmarts.Config

  @min_slot_visits 3
  @ready_fraction 0.9
  # After this many visits a slot stops averaging and starts fading:
  # effectively an EWMA with lambda 1/10, so a season's shape can
  # change over ~2 weeks of daily cycles without a refit.
  @fade_after 10

  @type t() :: %{
          slot_width: pos_integer(),
          slot_count: pos_integer(),
          updated_to: integer(),
          slots: %{non_neg_integer() => {float(), pos_integer()}}
        }

  @doc "A fresh model for a `:seasonality` period at a `:resolution` cadence."
  @spec new(Config.duration(), Config.duration()) :: t()
  def new(seasonality, resolution) do
    slot_width = Config.seconds(resolution)

    %{
      slot_width: slot_width,
      slot_count: div(Config.seconds(seasonality), slot_width),
      # Below any real timestamp, so the first window is not skipped.
      updated_to: -1,
      slots: %{}
    }
  end

  @doc """
  Fold windows newer than the model's high-water mark into the slot
  means. Re-presenting already-seen windows (the runtime re-scans
  trailing history every tick) is a no-op.
  """
  @spec update(t(), %{ts: [integer()], avg: [float()]}) :: t()
  def update(model, lists) do
    lists.ts
    |> Enum.zip(lists.avg)
    |> Enum.filter(fn {ts, _avg} -> ts > model.updated_to end)
    |> Enum.reduce(model, fn {ts, avg}, model ->
      slot = slot_of(model, ts)
      {mean, visits} = Map.get(model.slots, slot, {0.0, 0})
      weight = min(visits + 1, @fade_after)
      mean = mean + (avg - mean) / weight

      %{model | slots: Map.put(model.slots, slot, {mean, visits + 1}), updated_to: ts}
    end)
  end

  @doc """
  Whether enough of the cycle has been seen to trust the expectations:
  #{trunc(@ready_fraction * 100)}% of slots visited at least
  #{@min_slot_visits} times (about three full cycles of continuous
  data).
  """
  @spec ready?(t()) :: boolean()
  def ready?(model) do
    ready_slots(model) >= model.slot_count * @ready_fraction
  end

  @doc "Warm-up progress for the status surface: `{ready_slots, slot_count}`."
  @spec progress(t()) :: {non_neg_integer(), pos_integer()}
  def progress(model), do: {ready_slots(model), model.slot_count}

  @doc """
  The series with each window's seasonal expectation subtracted —
  centered around zero, which is what the detectors then baseline and
  scan. Slots the model has not seen enough of fall back to the mean
  of the ready slots, so a residual is always defined.
  """
  @spec residuals(t(), %{ts: [integer()], avg: [float()]}) :: map()
  def residuals(model, lists) do
    fallback = fallback_mean(model)

    avg =
      lists.ts
      |> Enum.zip(lists.avg)
      |> Enum.map(fn {ts, avg} -> avg - expectation(model, slot_of(model, ts), fallback) end)

    %{lists | avg: avg}
  end

  defp slot_of(model, ts), do: ts |> div(model.slot_width) |> rem(model.slot_count)

  defp ready_slots(model) do
    Enum.count(model.slots, fn {_slot, {_mean, visits}} -> visits >= @min_slot_visits end)
  end

  defp expectation(model, slot, fallback) do
    case Map.get(model.slots, slot) do
      {mean, visits} when visits >= @min_slot_visits -> mean
      _under_visited -> fallback
    end
  end

  defp fallback_mean(model) do
    ready =
      for {_slot, {mean, visits}} <- model.slots, visits >= @min_slot_visits, do: mean

    case ready do
      [] -> 0.0
      means -> Enum.sum(means) / length(means)
    end
  end
end
