# Hand-written `defn` kernels

Some patterns don't have a one-call form in Nx, Scholar or NxSignal,
but are short enough to write inline. A `defn` compiles through the
configured backend (`nx_eigen`, `EXLA`, `Torchx`) and runs at the same
speed as a built-in op.

Each recipe here is framed around a plausible question for the kinds
of metrics a Nerves device tends to emit. The data names are
illustrative — slot your own in.

## Events per hour from a sparse timestamp stream

You have a list of timestamps when a `door_opened` event fired. You
want a 24-bin histogram of events per hour-of-day, to see when the
device is used.

```elixir
defmodule DeviceUsage do
  import Nx.Defn

  defn events_per_hour(timestamps_unix_sec) do
    seconds_per_day = 86_400
    seconds_per_hour = 3_600

    bucket =
      timestamps_unix_sec
      |> Nx.remainder(seconds_per_day)
      |> Nx.quotient(seconds_per_hour)
      |> Nx.as_type(:s32)

    zeros = Nx.broadcast(0, {24})
    Nx.indexed_add(zeros, Nx.new_axis(bucket, 1), Nx.broadcast(1, Nx.shape(bucket)))
  end
end

door_opened_ts = pull_event_timestamps("door_opened")
hourly_counts  = DeviceUsage.events_per_hour(door_opened_ts)
# hourly_counts[h] is the number of opens that happened in hour h of the day.
```

The same shape works for any event stream — relay-on events, alarm
firings, GC pauses, packet errors. Change the bucket width to get
per-minute or per-day instead.

## How many transitions did this binary signal make today?

`presence_detected` is a 0/1 sensor sampled at 1Hz. The interesting
quantity is the number of transitions — that's the count of distinct
detection events.

```elixir
defmodule BinarySignal do
  import Nx.Defn

  defn transition_count(state_u8) do
    state_u8
    |> Nx.diff()
    |> Nx.abs()
    |> Nx.sum()
  end
end

presence_detected = pull_metric_tensor("presence_detected")
detections_today  = BinarySignal.transition_count(presence_detected) |> Nx.to_number()
```

A `0 → 1` transition is one detection start. Dividing by 2 gives the
number of complete detection episodes; without that, the count
includes both rising and falling edges.

## Time-in-state: how long was the heater on?

Same `heater_on` 0/1 signal, sampled every `dt` seconds. Total seconds
the heater was on is the sum of all the `1`s times `dt`.

```elixir
defmodule HeaterUsage do
  import Nx.Defn

  defn seconds_on(state_u8, dt_sec) do
    state_u8
    |> Nx.as_type(:f32)
    |> Nx.sum()
    |> Nx.multiply(dt_sec)
  end
end

heater_on = pull_metric_tensor("heater_on")
seconds_today = HeaterUsage.seconds_on(heater_on, 1.0) |> Nx.to_number()
duty_cycle_pct = seconds_today / 86_400.0 * 100.0
```

`Nx.cumulative_sum(state_u8) * dt` gives the same answer at every
prefix of the day — useful for plotting "cumulative on-time today" as
a step function. `Nx.mean(state_u8)` gives the duty cycle directly.

## When does the fan respond to CPU temperature?

Two metrics, `cpu_temp_c` and `fan_rpm`. You expect the fan to follow
the temperature with some lag. The lag is the argmax of their
cross-correlation.

```elixir
defmodule SignalLag do
  import Nx.Defn

  defn lag_samples(a, b, max_lag) do
    # Zero-mean both signals so the correlation isn't dominated
    # by their means.
    a = Nx.subtract(a, Nx.mean(a))
    b = Nx.subtract(b, Nx.mean(b))

    n = Nx.size(a)
    # Correlate b shifted against a for shifts in [-max_lag, +max_lag]
    shifts = Nx.iota({2 * max_lag + 1}) |> Nx.subtract(max_lag)
    # The argmax of the cross-correlation
    NxSignal.correlate(a, b)
    |> Nx.slice([n - 1 - max_lag], [2 * max_lag + 1])
    |> Nx.argmax()
    |> Nx.subtract(max_lag)
  end
end

cpu_temp_c = pull_metric_tensor("cpu_temperature")
fan_rpm    = pull_metric_tensor("fan_rpm")

lag_sec = SignalLag.lag_samples(cpu_temp_c, fan_rpm, 60) |> Nx.to_number()
# Positive lag means fan_rpm leads cpu_temp_c (the controller is
# predictive); negative means it lags (reactive).
```

The same recipe applies anywhere you have two related streams: a
sensor and the actuator that should respond to it, two correlated
metrics on different physical locations, a request rate and the
queue depth that follows.

## Run-length encoding of a state signal

How long does the heater stay on each time it turns on? That's
the run-length of `1`s in the `heater_on` series. The cumulative-segment
trick keeps it in tensor land.

```elixir
defmodule RunLengths do
  import Nx.Defn

  # Returns a per-sample tensor where each `1` carries the index of
  # the run it belongs to. Use Nx.bincount on the result to get the
  # length of every run.
  defn run_ids(state_u8) do
    rising_edge =
      state_u8
      |> Nx.diff()
      |> Nx.greater(0)
      |> Nx.as_type(:s32)

    # Pad the leading sample (no diff for index 0)
    edges = Nx.concatenate([Nx.tensor([0]), rising_edge])
    Nx.cumulative_sum(edges)
  end
end

heater_on = pull_metric_tensor("heater_on")
run_ids   = RunLengths.run_ids(heater_on)

# Only count samples where the heater is actually on
masked  = Nx.select(Nx.equal(heater_on, 1), run_ids, 0)
lengths = Nx.bincount(masked)
# lengths[0] is the count of "off" samples (ignore);
# lengths[1..] are the duration in seconds of each on-run.
```

The pattern — diff → edge-detect → cumulative_sum to get segment ids
→ bincount to summarize — is the general "vectorised segment
processing" toolkit. It's the closest tensor equivalent to a
`group_by` over consecutive runs.
